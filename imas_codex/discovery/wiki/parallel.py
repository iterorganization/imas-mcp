"""
Parallel wiki discovery engine with async workers.

Architecture:
- Two async workers: Scorer, Ingester (scan is bulk upfront)
- Graph + claimed_at timestamp for coordination (same pattern as paths discovery)
- Status transitions:
  - scanned → scored (content-aware scoring in single pass)
  - scored → ingested (embed for score >= 0.5)
- Workers claim pages by setting claimed_at, release by clearing it
- Orphan recovery: pages with claimed_at > 5 min old are reclaimed

Resilience:
- Supervised workers with automatic restart on crash (via base.supervision)
- Exponential backoff on infrastructure errors (Neo4j, network)
- Graceful degradation when services are temporarily unavailable

Workflow (after bulk discovery):
1. SCORE: Fetch content + LLM scoring in single pass → sets score, updates to 'scored'
2. INGEST: Chunk and embed high-value pages (score >= 0.5) → updates to 'ingested'
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.supervision import (
    SupervisedWorkerGroup,
    supervised_worker,
)
from imas_codex.graph import GraphClient
from imas_codex.graph.models import WikiArtifactStatus, WikiPageStatus

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Claim timeout - pages claimed longer than this are reclaimed
CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes


# =============================================================================
# Wiki Discovery State
# =============================================================================


@dataclass
class WikiDiscoveryState:
    """Shared state for parallel wiki discovery."""

    facility: str
    site_type: str  # mediawiki, confluence, twiki
    base_url: str
    portal_page: str
    ssh_host: str | None = None

    # Authentication for HTTP-based access (Tequila SSO)
    auth_type: str | None = None  # tequila, session, ssh_proxy, or None
    credential_service: str | None = None  # Keyring service for credentials

    # Shared async wiki client for native async HTTP (avoids re-auth per page)
    # Initialized lazily on first use, shared across all workers
    _async_wiki_client: Any = field(default=None, repr=False)
    _async_wiki_client_lock: Any = field(default=None, repr=False)

    # Limits
    cost_limit: float = 10.0
    page_limit: int | None = None
    max_depth: int | None = None
    focus: str | None = None

    # Worker stats (simplified: score + ingest only)
    scan_stats: WorkerStats = field(default_factory=WorkerStats)
    score_stats: WorkerStats = field(default_factory=WorkerStats)
    ingest_stats: WorkerStats = field(default_factory=WorkerStats)
    artifact_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False
    scan_idle_count: int = 0
    score_idle_count: int = 0
    ingest_idle_count: int = 0
    artifact_idle_count: int = 0

    # SSH retry tracking
    ssh_retry_count: int = 0
    max_ssh_retries: int = 5
    ssh_error_message: str | None = None

    @property
    def total_cost(self) -> float:
        return self.score_stats.cost + self.ingest_stats.cost

    @property
    def budget_exhausted(self) -> bool:
        return self.total_cost >= self.cost_limit

    @property
    def page_limit_reached(self) -> bool:
        if self.page_limit is None:
            return False
        return self.score_stats.processed >= self.page_limit

    def should_stop(self) -> bool:
        """Check if ALL workers should terminate.

        Used by the main loop to determine when discovery is complete.
        """
        if self.stop_requested:
            return True
        # Stop if all workers idle for 3+ iterations AND no pending work
        all_idle = (
            self.scan_idle_count >= 3
            and self.score_idle_count >= 3
            and self.ingest_idle_count >= 3
            and self.artifact_idle_count >= 3
        )
        if all_idle:
            if has_pending_work(self.facility) or has_pending_artifact_work(
                self.facility
            ):
                # Reset idle counts to force workers to re-poll
                self.scan_idle_count = 0
                self.score_idle_count = 0
                self.ingest_idle_count = 0
                self.artifact_idle_count = 0
                return False
            return True
        return False

    def should_stop_scanning(self) -> bool:
        """Check if scan workers should stop.

        Scan workers continue even when budget is exhausted. They only stop
        when explicitly requested or when no pending work remains.
        """
        if self.stop_requested:
            return True
        # Only stop scanning when idle with no work
        if self.scan_idle_count >= 3 and not has_pending_scan_work(self.facility):
            return True
        return False

    def should_stop_scoring(self) -> bool:
        """Check if score workers should stop.

        Score workers stop when budget exhausted or page limit reached.
        """
        if self.stop_requested:
            return True
        if self.budget_exhausted:
            return True
        if self.page_limit_reached:
            return True
        return False

    def should_stop_ingesting(self) -> bool:
        """Check if ingest workers should stop.

        Ingest workers continue AFTER budget is exhausted to drain the
        ingest queue. This ensures all scorable content gets ingested
        before termination. They only stop when:
        1. Explicitly requested
        2. Idle for 3+ iterations with no pending ingest work AND
           score workers are also idle (no more scoring happening)
        """
        if self.stop_requested:
            return True
        # Continue even when budget exhausted - drain the ingest queue
        # BUT don't exit early if scoring is still running - pages may arrive soon
        if self.ingest_idle_count >= 3:
            # Only stop if scoring is also idle AND no pending ingest work
            scoring_done = self.score_idle_count >= 3 or self.budget_exhausted
            if scoring_done and not has_pending_ingest_work(self.facility):
                return True
        return False

    def should_stop_artifact_worker(self) -> bool:
        """Check if artifact workers should stop.

        Artifact workers continue until no pending artifacts remain.
        They stop when explicitly requested or idle for 3+ iterations.
        """
        if self.stop_requested:
            return True
        if self.artifact_idle_count >= 3 and not has_pending_artifact_work(
            self.facility
        ):
            return True
        return False

    async def get_async_wiki_client(self):
        """Get shared AsyncMediaWikiClient for Tequila auth.

        Lazily initializes an async client that persists session cookies
        across all page fetches. Uses native async HTTP for better performance.
        """
        if self._async_wiki_client is None:
            from imas_codex.discovery.wiki.mediawiki import AsyncMediaWikiClient

            if self._async_wiki_client_lock is None:
                self._async_wiki_client_lock = asyncio.Lock()

            async with self._async_wiki_client_lock:
                if self._async_wiki_client is None:
                    self._async_wiki_client = AsyncMediaWikiClient(
                        base_url=self.base_url,
                        credential_service=self.credential_service
                        or f"{self.facility}-wiki",
                        verify_ssl=False,
                    )
                    # Pre-authenticate to warm up session
                    try:
                        await self._async_wiki_client.authenticate()
                        logger.info(
                            "Initialized shared AsyncMediaWikiClient with Tequila auth"
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to pre-authenticate AsyncMediaWikiClient: %s", e
                        )

        return self._async_wiki_client

    async def close_async_wiki_client(self):
        """Close the shared async wiki client."""
        if self._async_wiki_client is not None:
            try:
                await self._async_wiki_client.close()
            except Exception:
                pass
            self._async_wiki_client = None


# Artifact types we can extract text from
SUPPORTED_ARTIFACT_TYPES = {"pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "ipynb"}


def has_pending_artifact_work(facility: str) -> bool:
    """Check if there are pending artifacts for ingestion.

    Artifacts are ready for ingestion when:
    - status = 'discovered' AND artifact_type in SUPPORTED_ARTIFACT_TYPES
    - claimed_at is null or expired
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility})
            WHERE wa.status = $discovered
              AND wa.artifact_type IN $types
              AND (wa.claimed_at IS NULL OR wa.claimed_at < datetime() - duration($cutoff))
            RETURN count(wa) AS pending
            """,
            facility=facility,
            discovered=WikiArtifactStatus.discovered.value,
            types=list(SUPPORTED_ARTIFACT_TYPES),
            cutoff=cutoff,
        )
        return result[0]["pending"] > 0 if result else False


def has_pending_work(facility: str) -> bool:
    """Check if there's pending wiki work in the graph.

    Work exists if there are:
    - scanned pages awaiting scoring (claimed_at is null)
    - scored pages with score >= 0.5 awaiting ingest (claimed_at is null)
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE (wp.status = $scanned AND (wp.claimed_at IS NULL OR wp.claimed_at < datetime() - duration($cutoff)))
               OR (wp.status = $scored AND wp.score >= 0.5 AND (wp.claimed_at IS NULL OR wp.claimed_at < datetime() - duration($cutoff)))
            RETURN count(wp) AS pending
            """,
            facility=facility,
            scanned=WikiPageStatus.scanned.value,
            scored=WikiPageStatus.scored.value,
            cutoff=cutoff,
        )
        return result[0]["pending"] > 0 if result else False


def has_pending_scan_work(facility: str) -> bool:
    """Check if there's pending scoring work in the graph.

    With bulk discovery, there's no scan phase. This checks for:
    - scanned pages awaiting content-aware scoring
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $scanned
              AND (wp.claimed_at IS NULL OR wp.claimed_at < datetime() - duration($cutoff))
            RETURN count(wp) AS pending
            """,
            facility=facility,
            scanned=WikiPageStatus.scanned.value,
            cutoff=cutoff,
        )
        return result[0]["pending"] > 0 if result else False


def has_pending_ingest_work(facility: str) -> bool:
    """Check if there's pending ingest work in the graph.

    Returns True if there are scored pages with score >= 0.5 awaiting ingestion.
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $scored
              AND wp.score >= 0.5
              AND (wp.claimed_at IS NULL OR wp.claimed_at < datetime() - duration($cutoff))
            RETURN count(wp) AS pending
            """,
            facility=facility,
            scored=WikiPageStatus.scored.value,
            cutoff=cutoff,
        )
        return result[0]["pending"] > 0 if result else False


# =============================================================================
# Startup Reset
# =============================================================================


def reset_transient_pages(facility: str, *, silent: bool = False) -> dict[str, int]:
    """Reset claimed_at on all wiki pages on CLI startup.

    Since only one CLI process runs per facility at a time, any pages with
    claimed_at set are orphans from a previous crashed/killed process.
    Simply clear claimed_at to make them reclaimable.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.claimed_at IS NOT NULL
            SET wp.claimed_at = null
            RETURN count(wp) AS reset_count
            """,
            facility=facility,
        )
        reset_count = result[0]["reset_count"] if result else 0

    if not silent and reset_count > 0:
        logger.info(f"Reset {reset_count} orphaned wiki pages on startup")

    return {"orphan_reset": reset_count}


# =============================================================================
# Graph-based Work Claiming (uses claimed_at for coordination)
# =============================================================================


def claim_pages_for_scanning(facility: str, limit: int = 50) -> list[dict[str, Any]]:
    """Claim pages for link extraction (optional link-following workflow).

    With bulk MediaWiki discovery, all pages are created with status='scanned'
    so this function returns empty list. Link-following is not needed when
    the full page list is obtained from the API.

    For non-bulk (crawl) discovery, this would claim pages needing link extraction.
    Currently returns empty - link-following can be re-enabled by tracking
    'links_extracted' as a separate field from workflow status.
    """
    # Bulk discovery creates all pages as 'scanned' - no link following needed
    return []


def claim_pages_for_scoring(facility: str, limit: int = 50) -> list[dict[str, Any]]:
    """Claim scanned pages for content-aware scoring.

    Workflow: scanned + unclaimed → set claimed_at
    Score worker fetches content and scores in single pass.
    After scoring: update status to 'scored' and set score field.
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"  # ISO 8601 duration
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $scanned
              AND (wp.claimed_at IS NULL
                   OR wp.claimed_at < datetime() - duration($cutoff))
            WITH wp
            ORDER BY wp.discovered_at ASC
            LIMIT $limit
            SET wp.claimed_at = datetime()
            RETURN wp.id AS id, wp.title AS title, wp.url AS url
            """,
            facility=facility,
            scanned=WikiPageStatus.scanned.value,
            cutoff=cutoff,
            limit=limit,
        )
        return list(result)


def claim_pages_for_ingesting(
    facility: str, min_score: float = 0.5, limit: int = 10
) -> list[dict[str, Any]]:
    """Claim scored pages for ingestion (chunking and embedding).

    Workflow: scored + score >= min_score + unclaimed → set claimed_at
    After ingest: update status to 'ingested'.
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $scored
              AND wp.score >= $min_score
              AND (wp.claimed_at IS NULL
                   OR wp.claimed_at < datetime() - duration($cutoff))
            WITH wp
            ORDER BY wp.score DESC, wp.in_degree DESC
            LIMIT $limit
            SET wp.claimed_at = datetime()
            RETURN wp.id AS id, wp.title AS title, wp.url AS url,
                   wp.score AS score, wp.description AS description,
                   wp.physics_domain AS physics_domain,
                   wp.preview_text AS preview,
                   wp.in_degree AS in_degree, wp.out_degree AS out_degree
            """,
            facility=facility,
            scored=WikiPageStatus.scored.value,
            min_score=min_score,
            cutoff=cutoff,
            limit=limit,
        )
        return list(result)


# =============================================================================
# Mark Work Complete
# =============================================================================


def mark_pages_scanned(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark pages as scanned with extracted link data.

    Uses batched UNWIND for O(1) graph operations instead of O(n) individual queries.
    """
    if not results:
        return 0

    # Prepare batch data
    batch_data = [
        {"id": r.get("id"), "out_degree": r.get("out_degree", 0)}
        for r in results
        if r.get("id")
    ]

    if not batch_data:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS item
            MATCH (wp:WikiPage {id: item.id})
            SET wp.status = $scanned,
                wp.out_degree = item.out_degree,
                wp.scanned_at = datetime(),
                wp.claimed_at = null
            """,
            batch=batch_data,
            scanned=WikiPageStatus.scanned.value,
        )

    return len(batch_data)


def mark_pages_scored(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark pages as scored with all scoring data (content-aware single pass).

    All scored pages move to 'scored' status regardless of score.
    The ingester filters by score >= threshold, so low scores are
    effectively skipped without explicit status transition.

    Uses batched UNWIND for O(1) graph operations instead of O(n) individual queries.
    """
    if not results:
        return 0

    # Prepare batch data with all scoring fields
    batch_data = []
    for r in results:
        page_id = r.get("id")
        if not page_id:
            continue

        batch_data.append(
            {
                "id": page_id,
                "score": r.get("score", 0.0),
                "page_purpose": r.get("page_purpose", "other"),
                "description": (r.get("description", "") or "")[:150],
                "reasoning": r.get("reasoning", ""),
                "keywords": r.get("keywords", []),
                "physics_domain": r.get("physics_domain"),
                "preview_text": r.get("preview_text", ""),
                "score_data_documentation": r.get("score_data_documentation", 0.0),
                "score_physics_content": r.get("score_physics_content", 0.0),
                "score_code_documentation": r.get("score_code_documentation", 0.0),
                "score_data_access": r.get("score_data_access", 0.0),
                "score_calibration": r.get("score_calibration", 0.0),
                "score_imas_relevance": r.get("score_imas_relevance", 0.0),
                "is_physics": r.get("is_physics", False),
                "score_cost": r.get("score_cost", 0.0),
            }
        )

    if not batch_data:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS item
            MATCH (wp:WikiPage {id: item.id})
            SET wp.status = $status,
                wp.score = item.score,
                wp.page_purpose = item.page_purpose,
                wp.description = item.description,
                wp.score_reasoning = item.reasoning,
                wp.keywords = item.keywords,
                wp.physics_domain = item.physics_domain,
                wp.preview_text = item.preview_text,
                wp.score_data_documentation = item.score_data_documentation,
                wp.score_physics_content = item.score_physics_content,
                wp.score_code_documentation = item.score_code_documentation,
                wp.score_data_access = item.score_data_access,
                wp.score_calibration = item.score_calibration,
                wp.score_imas_relevance = item.score_imas_relevance,
                wp.is_physics_content = item.is_physics,
                wp.score_cost = item.score_cost,
                wp.scored_at = datetime(),
                wp.preview_fetched_at = datetime(),
                wp.claimed_at = null
            """,
            batch=batch_data,
            status=WikiPageStatus.scored.value,
        )

    return len(batch_data)


def mark_pages_ingested(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark pages as ingested with chunk data.

    Uses batched UNWIND for O(1) graph operations instead of O(n) individual queries.
    """
    if not results:
        return 0

    # Prepare batch data
    batch_data = [
        {"id": r.get("id"), "chunks": r.get("chunk_count", 0)}
        for r in results
        if r.get("id")
    ]

    if not batch_data:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS item
            MATCH (wp:WikiPage {id: item.id})
            SET wp.status = $ingested,
                wp.chunk_count = item.chunks,
                wp.ingested_at = datetime(),
                wp.claimed_at = null
            """,
            batch=batch_data,
            ingested=WikiPageStatus.ingested.value,
        )

    return len(batch_data)


def mark_page_failed(page_id: str, error: str, fallback_status: str) -> None:
    """Mark a page as failed with error message.

    If Neo4j is unavailable, logs a warning and silently fails.
    The page will be reclaimed after CLAIM_TIMEOUT_SECONDS.
    """
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (wp:WikiPage {id: $id})
                SET wp.status = $status,
                    wp.error = $error,
                    wp.failed_at = datetime(),
                    wp.claimed_at = null
                """,
                id=page_id,
                status=fallback_status,
                error=error,
            )
    except Exception as e:
        logger.warning(
            "Could not mark page %s as failed (Neo4j unavailable): %s", page_id, e
        )


def _release_claimed_pages(page_ids: list[str]) -> None:
    """Release claimed pages back for reprocessing.

    Clears claimed_at without changing status, allowing pages to be
    reclaimed by the next worker iteration.

    If Neo4j is unavailable, logs a warning and silently fails.
    Pages will be reclaimed after CLAIM_TIMEOUT_SECONDS anyway.
    """
    if not page_ids:
        return

    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $ids AS id
                MATCH (wp:WikiPage {id: id})
                SET wp.claimed_at = null
                """,
                ids=page_ids,
            )
    except Exception as e:
        logger.warning(
            "Could not release %d claimed pages (Neo4j unavailable): %s",
            len(page_ids),
            e,
        )


def release_orphaned_claims(facility: str) -> dict[str, int]:
    """Release all orphaned claims for a facility.

    Orphaned claims occur when a worker crashes without releasing its claimed
    pages. This function finds all pages with claimed_at older than the timeout
    and clears the claim.

    This is automatically called by the claim functions (they skip old claims),
    but can also be called explicitly to recover from stuck state.

    Args:
        facility: Facility ID

    Returns:
        Dict with counts: {"released": N, "pages": [...]}
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    try:
        with GraphClient() as gc:
            # Release stale page claims
            result = gc.query(
                """
                MATCH (wp:WikiPage {facility_id: $facility})
                WHERE wp.claimed_at IS NOT NULL
                  AND wp.claimed_at < datetime() - duration($cutoff)
                SET wp.claimed_at = null
                RETURN wp.id AS id, wp.status AS status
                """,
                facility=facility,
                cutoff=cutoff,
            )
            pages = list(result)

            # Release stale artifact claims
            result = gc.query(
                """
                MATCH (wa:WikiArtifact {facility_id: $facility})
                WHERE wa.claimed_at IS NOT NULL
                  AND wa.claimed_at < datetime() - duration($cutoff)
                SET wa.claimed_at = null
                RETURN wa.id AS id, wa.status AS status
                """,
                facility=facility,
                cutoff=cutoff,
            )
            artifacts = list(result)

            total = len(pages) + len(artifacts)
            if total > 0:
                logger.info(
                    "Released %d orphaned claims (%d pages, %d artifacts) for %s",
                    total,
                    len(pages),
                    len(artifacts),
                    facility,
                )

            return {
                "released_pages": len(pages),
                "released_artifacts": len(artifacts),
                "page_ids": [p["id"] for p in pages],
                "artifact_ids": [a["id"] for a in artifacts],
            }
    except Exception as e:
        logger.warning("Could not release orphaned claims: %s", e)
        return {"released_pages": 0, "released_artifacts": 0, "error": str(e)}


# =============================================================================
# Artifact Claim/Mark Functions
# =============================================================================


def claim_artifacts_for_ingesting(
    facility: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Claim discovered artifacts for ingestion.

    Claims artifacts with status='discovered' and supported artifact_type.
    Supported types: pdf, docx, pptx, xlsx, ipynb.

    Workflow: discovered + supported_type + unclaimed → set claimed_at
    After ingest: update status to 'ingested'.
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility})
            WHERE wa.status = $discovered
              AND wa.artifact_type IN $types
              AND (wa.claimed_at IS NULL
                   OR wa.claimed_at < datetime() - duration($cutoff))
            WITH wa
            ORDER BY wa.discovered_at ASC
            LIMIT $limit
            SET wa.claimed_at = datetime()
            RETURN wa.id AS id, wa.url AS url, wa.filename AS filename,
                   wa.artifact_type AS artifact_type
            """,
            facility=facility,
            discovered=WikiArtifactStatus.discovered.value,
            types=list(SUPPORTED_ARTIFACT_TYPES),
            cutoff=cutoff,
            limit=limit,
        )
        return list(result)


def mark_artifacts_ingested(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark artifacts as ingested with chunk data.

    Uses batched UNWIND for O(1) graph operations.
    """
    if not results:
        return 0

    batch_data = [
        {"id": r.get("id"), "chunks": r.get("chunk_count", 0)}
        for r in results
        if r.get("id")
    ]

    if not batch_data:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS item
            MATCH (wa:WikiArtifact {id: item.id})
            SET wa.status = $ingested,
                wa.chunk_count = item.chunks,
                wa.ingested_at = datetime(),
                wa.claimed_at = null
            """,
            batch=batch_data,
            ingested=WikiArtifactStatus.ingested.value,
        )

    return len(batch_data)


def mark_artifact_failed(artifact_id: str, error: str) -> None:
    """Mark an artifact as failed with error message."""
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (wa:WikiArtifact {id: $id})
                SET wa.status = $failed,
                    wa.error = $error,
                    wa.failed_at = datetime(),
                    wa.claimed_at = null
                """,
                id=artifact_id,
                failed=WikiArtifactStatus.failed.value,
                error=error,
            )
    except Exception as e:
        logger.warning(
            "Could not mark artifact %s as failed (Neo4j unavailable): %s",
            artifact_id,
            e,
        )


def mark_artifact_deferred(artifact_id: str, reason: str) -> None:
    """Mark an artifact as deferred (unsupported type or too large)."""
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (wa:WikiArtifact {id: $id})
                SET wa.status = $deferred,
                    wa.defer_reason = $reason,
                    wa.claimed_at = null
                """,
                id=artifact_id,
                deferred=WikiArtifactStatus.deferred.value,
                reason=reason,
            )
    except Exception as e:
        logger.warning(
            "Could not mark artifact %s as deferred (Neo4j unavailable): %s",
            artifact_id,
            e,
        )


# =============================================================================
# Bulk Page Discovery via Special:AllPages
# =============================================================================


def bulk_discover_all_pages_mediawiki(
    facility: str,
    base_url: str,
    ssh_host: str,
    on_progress: Callable | None = None,
) -> int:
    """Bulk discover all wiki pages via Special:AllPages.

    This is 100-300x faster than crawling links page-by-page.
    MediaWiki's Special:AllPages returns ~300-500 pages per request.

    Strategy:
    1. Fetch AllPages index to get alphabetical range links
    2. Fetch each range in parallel to extract page titles
    3. Create all pages as 'scanned' status (skip scanning phase)

    Args:
        facility: Facility ID
        base_url: Wiki base URL
        ssh_host: SSH host for proxied access
        on_progress: Progress callback

    Returns:
        Number of pages discovered
    """
    from urllib.parse import unquote

    logger.info("Starting bulk page discovery via Special:AllPages...")

    # Step 1: Get the alphabetical index to find range links
    index_url = f"{base_url}/index.php?title=Special:AllPages"
    cmd = f'curl -sk "{index_url}"'

    try:
        result = subprocess.run(
            ["ssh", ssh_host, cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("Failed to fetch AllPages index")
            return 0
    except subprocess.TimeoutExpired:
        logger.warning("Timeout fetching AllPages index")
        return 0

    # Parse range links from the allpageslist table
    # Format: from=Page_Name&to=Page_Name
    import re

    range_pattern = re.compile(
        r'href="[^"]*title=Special:AllPages[^"]*from=([^&"]+)[^"]*"'
    )
    ranges = list(set(range_pattern.findall(result.stdout)))

    if not ranges:
        logger.warning("No page ranges found in AllPages index")
        return 0

    logger.info(f"Found {len(ranges)} page ranges to process")
    if on_progress:
        on_progress(f"found {len(ranges)} ranges", None)

    # Step 2: Fetch each range to get page titles
    all_pages: set[str] = set()

    for i, from_page in enumerate(ranges):
        range_url = f"{base_url}/index.php?title=Special:AllPages&from={from_page}"
        cmd = f'curl -sk "{range_url}"'

        try:
            result = subprocess.run(
                ["ssh", ssh_host, cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                continue
        except subprocess.TimeoutExpired:
            continue

        # Extract page links from response
        # Format: href="/wiki/Page_Name" or href="/wiki/index.php?title=Page_Name"
        page_pattern = re.compile(r'href="/wiki/([^"?]+)"')
        for match in page_pattern.finditer(result.stdout):
            page_name = unquote(match.group(1))
            # Skip special pages and system namespaces
            if not any(
                page_name.startswith(prefix)
                for prefix in (
                    "Special:",
                    "File:",
                    "Talk:",
                    "User:",
                    "Template:",
                    "Category:",
                    "Help:",
                    "MediaWiki:",
                    "SPCwiki:",
                )
            ):
                all_pages.add(page_name)

        if on_progress:
            on_progress(f"range {i + 1}/{len(ranges)}: {len(all_pages)} pages", None)

        logger.debug(f"Range {i + 1}/{len(ranges)}: total {len(all_pages)} pages")

    logger.info(f"Discovered {len(all_pages)} unique pages")

    # Step 3: Create all pages in graph as 'scanned' status using batch insert
    from imas_codex.discovery.wiki.scraper import canonical_page_id

    # Prepare batch data for efficient insertion
    batch_data = []
    for page_name in all_pages:
        page_id = canonical_page_id(page_name, facility)
        url = f"{base_url}/{urllib.parse.quote(page_name, safe='/')}"
        batch_data.append(
            {
                "id": page_id,
                "title": page_name,
                "url": url,
            }
        )

    # Insert in batches of 500 for efficiency
    batch_size = 500
    created = 0
    with GraphClient() as gc:
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i : i + batch_size]
            result = gc.query(
                """
                UNWIND $pages AS page
                MERGE (wp:WikiPage {id: page.id})
                ON CREATE SET wp.title = page.title,
                              wp.url = page.url,
                              wp.facility_id = $facility,
                              wp.status = $scanned,
                              wp.link_depth = 1,
                              wp.discovered_at = datetime(),
                              wp.bulk_discovered = true
                ON MATCH SET wp.bulk_discovered = true
                RETURN count(wp) AS count
                """,
                pages=batch,
                facility=facility,
                scanned=WikiPageStatus.scanned.value,
            )
            if result:
                created += result[0]["count"]

            if on_progress:
                on_progress(
                    f"creating pages ({i + len(batch)}/{len(batch_data)})", None
                )

    logger.info(f"Created/updated {created} pages in graph (scanned status)")
    if on_progress:
        on_progress(f"created {created} pages", None)

    return created


def bulk_discover_all_pages_http(
    facility: str,
    base_url: str,
    credential_service: str,
    on_progress: Callable | None = None,
) -> int:
    """Bulk discover all wiki pages via Special:AllPages using HTTP with Tequila auth.

    This is 100-300x faster than crawling links page-by-page.
    MediaWiki's Special:AllPages returns ~300-500 pages per request.

    Strategy:
    1. Authenticate with Tequila (EPFL SSO)
    2. Fetch AllPages index to get alphabetical range links
    3. Fetch each range to extract page titles
    4. Create all pages as 'scanned' status (skip scanning phase)

    Args:
        facility: Facility ID
        base_url: Wiki base URL
        credential_service: Keyring service name for Tequila credentials
        on_progress: Progress callback

    Returns:
        Number of pages discovered
    """
    import re
    from urllib.parse import unquote

    from imas_codex.discovery.wiki.mediawiki import MediaWikiClient

    logger.info("Starting bulk page discovery via Special:AllPages (HTTP)...")

    # Create authenticated client
    client = MediaWikiClient(base_url=base_url, credential_service=credential_service)

    if not client.authenticate():
        logger.error("Failed to authenticate with Tequila")
        return 0

    try:
        session = client._get_session()

        # Step 1: Get the alphabetical index to find range links
        index_url = f"{base_url}/index.php?title=Special:AllPages"

        try:
            response = session.get(index_url, timeout=30, verify=client.verify_ssl)
            if response.status_code != 200:
                logger.warning(
                    f"Failed to fetch AllPages index: HTTP {response.status_code}"
                )
                return 0
            html_content = response.text
        except Exception as e:
            logger.warning(f"Error fetching AllPages index: {e}")
            return 0

        # Parse range links from the allpageslist table
        # Format: from=Page_Name&to=Page_Name
        range_pattern = re.compile(
            r'href="[^"]*title=Special:AllPages[^"]*from=([^&"]+)[^"]*"'
        )
        ranges = list(set(range_pattern.findall(html_content)))

        if not ranges:
            logger.warning("No page ranges found in AllPages index")
            return 0

        logger.info(f"Found {len(ranges)} page ranges to process")
        if on_progress:
            on_progress(f"found {len(ranges)} ranges", None)

        # Step 2: Fetch each range to get page titles
        all_pages: set[str] = set()

        for i, from_page in enumerate(ranges):
            range_url = f"{base_url}/index.php?title=Special:AllPages&from={from_page}"

            try:
                response = session.get(range_url, timeout=30, verify=client.verify_ssl)
                if response.status_code != 200:
                    continue
                range_html = response.text
            except Exception:
                continue

            # Extract page links from response
            # Format: href="/wiki/Page_Name" or href="/path/Page_Name"
            page_pattern = re.compile(r'href="/wiki/([^"?]+)"')
            for match in page_pattern.finditer(range_html):
                page_name = unquote(match.group(1))
                # Skip special pages and system namespaces
                if not any(
                    page_name.startswith(prefix)
                    for prefix in (
                        "Special:",
                        "File:",
                        "Talk:",
                        "User:",
                        "Template:",
                        "Category:",
                        "Help:",
                        "MediaWiki:",
                        "SPCwiki:",
                    )
                ):
                    all_pages.add(page_name)

            if on_progress:
                on_progress(
                    f"range {i + 1}/{len(ranges)}: {len(all_pages)} pages", None
                )

            logger.debug(f"Range {i + 1}/{len(ranges)}: total {len(all_pages)} pages")

        logger.info(f"Discovered {len(all_pages)} unique pages")

        # Step 3: Create all pages in graph as 'scanned' status using batch insert
        from imas_codex.discovery.wiki.scraper import canonical_page_id

        # Prepare batch data for efficient insertion
        batch_data = []
        for page_name in all_pages:
            page_id = canonical_page_id(page_name, facility)
            url = f"{base_url}/{urllib.parse.quote(page_name, safe='/')}"
            batch_data.append(
                {
                    "id": page_id,
                    "title": page_name,
                    "url": url,
                }
            )

        # Insert in batches of 500 for efficiency
        batch_size = 500
        created = 0
        with GraphClient() as gc:
            for i in range(0, len(batch_data), batch_size):
                batch = batch_data[i : i + batch_size]
                result = gc.query(
                    """
                    UNWIND $pages AS page
                    MERGE (wp:WikiPage {id: page.id})
                    ON CREATE SET wp.title = page.title,
                                  wp.url = page.url,
                                  wp.facility_id = $facility,
                                  wp.status = $scanned,
                                  wp.link_depth = 1,
                                  wp.discovered_at = datetime(),
                                  wp.bulk_discovered = true
                    ON MATCH SET wp.bulk_discovered = true
                    RETURN count(wp) AS count
                    """,
                    pages=batch,
                    facility=facility,
                    scanned=WikiPageStatus.scanned.value,
                )
                if result:
                    created += result[0]["count"]

                if on_progress:
                    on_progress(
                        f"creating pages ({i + len(batch)}/{len(batch_data)})", None
                    )

        logger.info(f"Created/updated {created} pages in graph (scanned status)")
        if on_progress:
            on_progress(f"created {created} pages", None)

        return created

    finally:
        client.close()


# =============================================================================
# Bulk Artifact Discovery via MediaWiki API
# =============================================================================


def bulk_discover_artifacts(
    facility: str,
    base_url: str,
    site_type: str = "mediawiki",
    ssh_host: str | None = None,
    wiki_client: Any = None,
    credential_service: str | None = None,
    on_progress: Callable | None = None,
) -> int:
    """Bulk discover all wiki artifacts via platform API.

    This is much faster than scanning pages - uses dedicated APIs:
    - MediaWiki: list=allimages API (returns all files in one call)
    - TWiki: /pub/ directory listing
    - Confluence: /rest/api/content/{id}/child/attachment

    Args:
        facility: Facility ID
        base_url: Wiki base URL
        site_type: Wiki platform type
        ssh_host: SSH host for proxied access
        wiki_client: Authenticated MediaWikiClient (for Tequila)
        credential_service: Keyring service name
        on_progress: Progress callback

    Returns:
        Number of artifacts discovered
    """
    from imas_codex.discovery.wiki.adapters import get_adapter
    from imas_codex.graph.models import WikiArtifactStatus

    logger.debug(f"Starting bulk artifact discovery for {site_type}...")

    # Get the appropriate adapter
    adapter = get_adapter(
        site_type=site_type,
        ssh_host=ssh_host,
        wiki_client=wiki_client,
        credential_service=credential_service,
    )

    # Discover artifacts
    artifacts = adapter.bulk_discover_artifacts(facility, base_url, on_progress)

    if not artifacts:
        logger.debug("No artifacts discovered")
        return 0

    logger.debug(f"Discovered {len(artifacts)} artifacts")

    # Create artifact nodes in graph
    created = 0
    with GraphClient() as gc:
        batch_size = 100
        for i in range(0, len(artifacts), batch_size):
            batch = artifacts[i : i + batch_size]
            batch_data = [
                {
                    "id": f"{facility}:{a.filename}",
                    "filename": a.filename,
                    "url": a.url,
                    "artifact_type": a.artifact_type,
                    "size_bytes": a.size_bytes,
                    "mime_type": a.mime_type,
                }
                for a in batch
            ]

            result = gc.query(
                """
                UNWIND $artifacts AS a
                MERGE (wa:WikiArtifact {id: a.id})
                ON CREATE SET wa.facility_id = $facility,
                              wa.filename = a.filename,
                              wa.url = a.url,
                              wa.artifact_type = a.artifact_type,
                              wa.size_bytes = a.size_bytes,
                              wa.mime_type = a.mime_type,
                              wa.status = $discovered,
                              wa.discovered_at = datetime(),
                              wa.bulk_discovered = true
                ON MATCH SET wa.bulk_discovered = true
                RETURN count(wa) AS count
                """,
                artifacts=batch_data,
                facility=facility,
                discovered=WikiArtifactStatus.discovered.value,
            )
            if result:
                created += result[0]["count"]

            if on_progress:
                on_progress(
                    f"created {i + len(batch)}/{len(artifacts)} artifacts", None
                )

    logger.info(f"Created/updated {created} artifact nodes in graph")
    if on_progress:
        on_progress(f"created {created} artifacts", None)

    return created


# =============================================================================
# Link Extraction (Scanner Worker Helpers)
# =============================================================================


def extract_links_mediawiki(
    page_url: str, ssh_host: str, base_url: str | None = None
) -> tuple[list[str], list[tuple[str, str]]]:
    """Extract links from a MediaWiki page via SSH.

    Args:
        page_url: Full URL of the page to scan
        ssh_host: SSH host for proxied access
        base_url: Base URL of the wiki (used to determine link prefix)

    The function handles multiple MediaWiki URL formats:
    1. /wiki/Page_Name (standard)
    2. /path/Page_Name (short URLs)
    3. /path/index.php?title=Page_Name (query string format)
    """
    from urllib.parse import parse_qs, urlparse

    # Determine the wiki path prefix from base_url
    wiki_path = ""
    if base_url:
        parsed = urlparse(base_url)
        if parsed.path and parsed.path != "/":
            wiki_path = parsed.path.rstrip("/")

    # Fetch the page and extract all href attributes
    cmd = f'''curl -sk "{page_url}" | grep -oP 'href="[^"]*"' | sed 's/href="//;s/"$//' | sort -u'''

    try:
        result = subprocess.run(
            ["ssh", ssh_host, cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return [], []

        page_links: list[str] = []
        artifact_links: list[tuple[str, str]] = []

        excluded_prefixes = (
            "Special:",
            "File:",
            "Talk:",
            "User_talk:",
            "Template:",
            "Category:",
            "Help:",
            "MediaWiki:",
            "User:",
        )

        excluded_actions = {"edit", "history", "delete", "protect", "watch"}

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            # Skip external links, javascript, mailto
            if line.startswith(("http://", "https://", "javascript:", "mailto:", "#")):
                # Check if it's a link to the same wiki (external but same host)
                if line.startswith(("http://", "https://")):
                    parsed = urlparse(line)
                    if base_url:
                        base_parsed = urlparse(base_url)
                        if parsed.netloc != base_parsed.netloc:
                            continue
                        # Same host - extract the path
                        line = parsed.path
                        if parsed.query:
                            line += "?" + parsed.query
                    else:
                        continue
                else:
                    continue

            # Skip non-wiki paths (images, js, css, etc)
            if any(
                x in line.lower()
                for x in [
                    "/images/",
                    "/skins/",
                    "/load.php",
                    ".css",
                    ".js",
                    ".png",
                    ".jpg",
                    ".gif",
                    ".ico",
                    "opensearch",
                    "api.php",
                ]
            ):
                continue

            page_name = None

            # Handle index.php?title=Page_Name format
            if "index.php" in line and "title=" in line:
                # Parse the query string
                if "?" in line:
                    query_part = line.split("?", 1)[1]
                    # Handle HTML entity encoded ampersands
                    query_part = query_part.replace("&amp;", "&")
                    params = parse_qs(query_part)
                    if "title" in params:
                        page_name = params["title"][0]
                        # Skip edit/history/etc actions
                        action = params.get("action", ["view"])[0]
                        if action in excluded_actions:
                            continue
                        # Skip redlinks (non-existent pages)
                        if "redlink" in params:
                            continue

            # Handle /wiki/Page_Name or /path/Page_Name format
            elif wiki_path and line.startswith(wiki_path + "/"):
                page_name = line[len(wiki_path) + 1 :]
            elif line.startswith("/wiki/"):
                page_name = line[6:]

            if not page_name:
                continue

            # Skip excluded namespaces
            if page_name.startswith(excluded_prefixes):
                continue

            # Skip query params in page name (shouldn't happen but be safe)
            if "?" in page_name:
                page_name = page_name.split("?")[0]

            decoded = urllib.parse.unquote(page_name)

            # Skip empty or just whitespace
            if not decoded.strip():
                continue

            # Classify as page or artifact
            if _is_artifact(decoded):
                artifact_type = _get_artifact_type(decoded)
                artifact_links.append((decoded, artifact_type))
            else:
                page_links.append(decoded)

        # Deduplicate while preserving order
        seen = set()
        unique_pages = []
        for p in page_links:
            if p not in seen:
                seen.add(p)
                unique_pages.append(p)

        return unique_pages, artifact_links

    except subprocess.TimeoutExpired:
        logger.warning("Timeout extracting links from %s", page_url)
        return [], []


def extract_links_twiki(
    page_name: str, base_url: str, ssh_host: str
) -> tuple[list[str], list[tuple[str, str]]]:
    """Extract links from a TWiki page via SSH."""
    if "/" not in page_name:
        page_name = f"Main/{page_name}"

    url = f"{base_url}/bin/view/{page_name}"
    cmd = f'''curl -s "{url}" | grep -oP 'href="[^"]*"' | sed 's/href="//;s/"$//' | sort -u'''

    try:
        result = subprocess.run(
            ["ssh", ssh_host, cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return [], []

        page_links: list[str] = []
        artifact_links: list[tuple[str, str]] = []

        excluded_patterns = (
            "/twiki/bin/edit/",
            "/twiki/bin/attach/",
            "/twiki/bin/rdiff/",
            "/twiki/bin/oops/",
            "/twiki/bin/search/",
            "?",
            "#",
            "mailto:",
            "javascript:",
        )

        import re

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            if any(pat in line for pat in excluded_patterns):
                continue

            if "/twiki/bin/view/" in line:
                match = re.search(r"/twiki/bin/view/(\w+/\w+)", line)
                if match:
                    topic = match.group(1)
                    if not topic.startswith(("TWiki/", "Sandbox/")):
                        page_links.append(topic)

            elif "/twiki/pub/" in line:
                if _is_artifact(line):
                    artifact_type = _get_artifact_type(line)
                    artifact_links.append((line, artifact_type))

        return page_links, artifact_links

    except subprocess.TimeoutExpired:
        logger.warning("Timeout extracting TWiki links from %s", page_name)
        return [], []


def _is_artifact(link: str) -> bool:
    """Check if a link points to an artifact (PDF, image, etc.)."""
    artifact_extensions = {
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".csv",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".zip",
        ".tar",
        ".gz",
        ".h5",
        ".hdf5",
        ".mat",
        ".ipynb",
    }
    link_lower = link.lower()
    return any(link_lower.endswith(ext) for ext in artifact_extensions)


def _get_artifact_type(link: str) -> str:
    """Get artifact type from link extension."""
    link_lower = link.lower()
    if link_lower.endswith(".pdf"):
        return "pdf"
    if link_lower.endswith((".doc", ".docx", ".odt", ".rtf")):
        return "document"
    if link_lower.endswith((".ppt", ".pptx", ".key")):
        return "presentation"
    if link_lower.endswith((".xls", ".xlsx", ".csv")):
        return "spreadsheet"
    if link_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
        return "image"
    if link_lower.endswith((".ipynb", ".nb")):
        return "notebook"
    if link_lower.endswith((".h5", ".hdf5", ".mat", ".nc")):
        return "data"
    if link_lower.endswith((".zip", ".tar", ".gz", ".tgz")):
        return "archive"
    return "document"


# =============================================================================
# Async Workers
# =============================================================================


async def scan_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Scanner worker: Extract links from pending pages.

    Transitions: pending → scanning → scanned

    Uses concurrent SSH calls with bounded parallelism (max 10 concurrent).
    """
    # Semaphore to limit concurrent SSH connections
    ssh_semaphore = asyncio.Semaphore(10)

    async def process_page(page: dict) -> dict | None:
        """Process a single page with semaphore-bounded concurrency."""
        async with ssh_semaphore:
            page_id = page["id"]
            title = page.get("title", "")
            url = page.get("url", "")

            try:
                # Run blocking SSH call in thread pool
                if state.site_type == "twiki" and state.ssh_host:
                    page_links, artifact_links = await asyncio.to_thread(
                        extract_links_twiki, title, state.base_url, state.ssh_host
                    )
                elif state.ssh_host:
                    page_links, artifact_links = await asyncio.to_thread(
                        extract_links_mediawiki, url, state.ssh_host, state.base_url
                    )
                else:
                    # No SSH host - can't scan remote wiki
                    page_links, artifact_links = [], []

                # Create new pending pages for discovered links
                await asyncio.to_thread(
                    _create_discovered_pages,
                    state.facility,
                    page_links,
                    page.get("depth", 0) + 1,
                    state.max_depth,
                    state.base_url,
                    state.site_type,
                )

                # Create artifact nodes and link to this page
                await asyncio.to_thread(
                    _create_discovered_artifacts,
                    state.facility,
                    artifact_links,
                    page_id,
                )

                return {
                    "id": page_id,
                    "out_degree": len(page_links) + len(artifact_links),
                    "page_links": len(page_links),
                    "artifact_links": len(artifact_links),
                }

            except Exception as e:
                logger.warning("Error scanning %s: %s", page_id, e)
                await asyncio.to_thread(
                    mark_page_failed, page_id, str(e), WikiPageStatus.scanned.value
                )
                return None

    while not state.should_stop_scanning():
        pages = claim_pages_for_scanning(state.facility, limit=50)

        if not pages:
            state.scan_idle_count += 1
            if on_progress:
                on_progress("idle", state.scan_stats)
            await asyncio.sleep(1.0)
            continue

        state.scan_idle_count = 0

        if on_progress:
            on_progress(f"scanning {len(pages)} pages", state.scan_stats)

        # Process pages concurrently with bounded parallelism
        tasks = [process_page(page) for page in pages]
        results_raw = await asyncio.gather(*tasks)
        results = [r for r in results_raw if r is not None]

        # Log progress after batch completes
        logger.debug("Scanned batch: %d/%d pages succeeded", len(results), len(pages))

        # Mark pages as scanned
        mark_pages_scanned(state.facility, results)
        state.scan_stats.processed += len(results)

        if on_progress:
            on_progress(
                f"scanned {len(results)} pages", state.scan_stats, results=results
            )


async def score_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Score worker: Content-aware LLM scoring in single pass.

    Transitions: scanned → scored

    Fetches page content preview, then scores with LLM.
    Uses centralized LLM access via get_model_for_task().
    Cost is tracked from actual OpenRouter response.
    """
    from imas_codex.agentic.agents import get_model_for_task

    logger.info("score_worker started")

    # Semaphore to limit concurrent HTTP fetches
    # Increased from 15 to 25 for better throughput with connection pooling
    fetch_semaphore = asyncio.Semaphore(25)

    # Get shared async wiki client for Tequila auth (native async HTTP)
    shared_async_wiki_client = (
        await state.get_async_wiki_client() if state.auth_type == "tequila" else None
    )

    async def fetch_content_for_page(page: dict) -> dict:
        """Fetch content preview for a single page."""
        async with fetch_semaphore:
            url = page.get("url", "")
            try:
                preview = await _fetch_and_summarize(
                    url,
                    state.ssh_host,
                    auth_type=state.auth_type,
                    credential_service=state.credential_service,
                    max_chars=1500,  # Reduced for scoring
                    async_wiki_client=shared_async_wiki_client,  # Native async HTTP
                )
                return {
                    "id": page["id"],
                    "title": page.get("title", ""),
                    "url": url,
                    "preview_text": preview,
                    "fetch_error": None,
                }
            except Exception as e:
                logger.debug("Failed to fetch %s: %s", url, e)
                return {
                    "id": page["id"],
                    "title": page.get("title", ""),
                    "url": url,
                    "preview_text": "",
                    "fetch_error": str(e),
                }

    while not state.should_stop_scoring():
        # Increased batch size from 25 to 50 for better LLM throughput
        pages = claim_pages_for_scoring(state.facility, limit=50)
        logger.debug(f"score_worker claimed {len(pages)} pages")

        if not pages:
            state.score_idle_count += 1
            if on_progress:
                on_progress("idle", state.score_stats)
            await asyncio.sleep(1.0)
            continue

        state.score_idle_count = 0

        if on_progress:
            on_progress(f"fetching {len(pages)} pages", state.score_stats)

        # Step 1: Fetch content for all pages in parallel
        fetch_tasks = [fetch_content_for_page(page) for page in pages]
        pages_with_content = await asyncio.gather(*fetch_tasks)

        if on_progress:
            on_progress(f"scoring {len(pages)} pages", state.score_stats)

        try:
            # Step 2: Score batch with LLM
            model = get_model_for_task("discovery")
            results, cost = await _score_pages_batch(
                pages_with_content, model, state.focus
            )

            # Add preview_text to results for persistence
            for r in results:
                matching_page = next(
                    (p for p in pages_with_content if p["id"] == r["id"]), {}
                )
                r["preview_text"] = matching_page.get("preview_text", "")
                r["score_cost"] = cost / len(results) if results else 0.0

            mark_pages_scored(state.facility, results)
            state.score_stats.processed += len(results)
            state.score_stats.cost += cost

            if on_progress:
                on_progress(
                    f"scored {len(results)} pages", state.score_stats, results=results
                )

        except ValueError as e:
            # LLM validation error (e.g., truncated JSON) - release pages and continue
            # Don't stop the whole process; pages will be reclaimed after timeout
            logger.warning(
                "LLM validation error for batch of %d pages: %s. "
                "Releasing pages for retry.",
                len(pages),
                e,
            )
            state.score_stats.errors = getattr(state.score_stats, "errors", 0) + 1
            # Release pages by clearing claimed_at (not marking as failed)
            _release_claimed_pages([p["id"] for p in pages])
            # Continue processing - don't stop the whole discovery
            continue
        except Exception as e:
            logger.error("Error in scoring batch: %s", e)
            for page in pages:
                mark_page_failed(page["id"], str(e), WikiPageStatus.scanned.value)


async def ingest_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
    min_score: float = 0.5,
) -> None:
    """Ingest worker: Chunk and embed high-value scored pages.

    Transitions: scored → ingested

    Claims scored pages with score >= min_score, fetches full content,
    chunks it, and creates embeddings.

    The ingest worker continues running even after cost limit is reached
    to drain the ingest queue. This ensures all scored content gets ingested.
    """
    # Get shared async wiki client for Tequila auth (native async HTTP)
    shared_async_wiki_client = (
        await state.get_async_wiki_client() if state.auth_type == "tequila" else None
    )

    while not state.should_stop_ingesting():
        # Increased batch size from 10 to 20 for better embedding throughput
        pages = claim_pages_for_ingesting(state.facility, min_score=min_score, limit=20)

        if not pages:
            state.ingest_idle_count += 1
            if on_progress:
                on_progress("idle", state.ingest_stats)
            await asyncio.sleep(1.0)
            continue

        state.ingest_idle_count = 0

        if on_progress:
            on_progress(f"ingesting {len(pages)} pages", state.ingest_stats)

        results = []
        for page in pages:
            page_id = page["id"]
            url = page.get("url", "")

            try:
                chunk_count = await _ingest_page(
                    url=url,
                    page_id=page_id,
                    facility=state.facility,
                    site_type=state.site_type,
                    ssh_host=state.ssh_host,
                    auth_type=state.auth_type,
                    credential_service=state.credential_service,
                    async_wiki_client=shared_async_wiki_client,
                )
                # Include score, description, physics_domain for display
                results.append(
                    {
                        "id": page_id,
                        "chunk_count": chunk_count,
                        "score": page.get("score"),
                        "description": page.get("description", ""),
                        "physics_domain": page.get("physics_domain"),
                    }
                )
            except Exception as e:
                logger.warning("Error ingesting %s: %s", page_id, e)
                mark_page_failed(page_id, str(e), WikiPageStatus.scored.value)

        mark_pages_ingested(state.facility, results)
        state.ingest_stats.processed += len(results)

        if on_progress:
            on_progress(
                f"ingested {len(results)} pages", state.ingest_stats, results=results
            )


async def artifact_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
    max_size_mb: float = 5.0,
) -> None:
    """Artifact worker: Download and ingest wiki artifacts.

    Transitions: discovered → ingesting → ingested

    Claims discovered artifacts with supported types (pdf, docx, pptx, xlsx, ipynb),
    downloads content, and extracts text.
    Unsupported artifact types are marked as deferred.

    Args:
        state: Shared discovery state
        on_progress: Progress callback (msg, stats, results=None)
        max_size_mb: Maximum artifact size in MB
    """
    from imas_codex.discovery.wiki.pipeline import (
        WikiArtifactPipeline,
        fetch_artifact_content,
        fetch_artifact_size,
    )

    max_size_bytes = int(max_size_mb * 1024 * 1024)
    pipeline = WikiArtifactPipeline(
        facility_id=state.facility,
        max_size_mb=max_size_mb,
        use_rich=False,
    )

    while not state.should_stop_artifact_worker():
        # Increased batch size from 3 to 8 for better throughput
        artifacts = claim_artifacts_for_ingesting(state.facility, limit=8)

        if not artifacts:
            state.artifact_idle_count += 1
            if on_progress:
                on_progress("idle", state.artifact_stats)
            await asyncio.sleep(1.0)
            continue

        state.artifact_idle_count = 0

        if on_progress:
            on_progress(f"ingesting {len(artifacts)} artifacts", state.artifact_stats)

        results = []
        for artifact in artifacts:
            artifact_id = artifact["id"]
            artifact_type = artifact.get("artifact_type", "unknown")
            url = artifact.get("url", "")
            filename = artifact.get("filename", "unknown")

            try:
                # Check size before downloading
                size_bytes = fetch_artifact_size(url, facility=state.facility)

                if size_bytes is not None and size_bytes > max_size_bytes:
                    size_mb = size_bytes / (1024 * 1024)
                    reason = (
                        f"File size {size_mb:.1f} MB exceeds limit {max_size_mb:.1f} MB"
                    )
                    logger.info("Deferring oversized artifact %s: %s", filename, reason)
                    mark_artifact_deferred(artifact_id, reason)
                    continue

                # Check if type is supported
                if artifact_type.lower() not in SUPPORTED_ARTIFACT_TYPES:
                    reason = f"Artifact type '{artifact_type}' not supported"
                    mark_artifact_deferred(artifact_id, reason)
                    continue

                # Download and ingest
                _, content = await fetch_artifact_content(url, facility=state.facility)
                stats = await pipeline.ingest_artifact(
                    artifact_id, content, artifact_type
                )
                results.append(
                    {
                        "id": artifact_id,
                        "chunk_count": stats["chunks"],
                        "filename": filename,
                        "artifact_type": artifact_type,
                    }
                )

            except Exception as e:
                # Unwrap common error wrappers to get the actual error message
                error_msg = str(e)
                if hasattr(e, "__cause__") and e.__cause__:
                    error_msg = str(e.__cause__)
                elif "RetryError" in type(e).__name__ and hasattr(e, "last_attempt"):
                    # tenacity RetryError - get the underlying exception
                    try:
                        error_msg = str(e.last_attempt.exception())
                    except Exception:
                        pass
                logger.warning(
                    "Error ingesting artifact %s: %s", artifact_id, error_msg
                )
                mark_artifact_failed(artifact_id, error_msg)

        mark_artifacts_ingested(state.facility, results)
        state.artifact_stats.processed += len(results)

        if on_progress:
            on_progress(
                f"ingested {len(results)} artifacts",
                state.artifact_stats,
                results=results,
            )


# =============================================================================
# Worker Helpers
# =============================================================================


def _create_discovered_pages(
    facility: str,
    page_names: list[str],
    depth: int,
    max_depth: int | None = None,
    base_url: str | None = None,
    site_type: str = "mediawiki",
) -> int:
    """Create pending page nodes for newly discovered links.

    Args:
        facility: Facility ID
        page_names: List of page names (not full URLs)
        depth: Link depth from portal
        max_depth: Maximum depth limit
        base_url: Base URL of the wiki (for constructing page URLs)
        site_type: Type of wiki site
    """
    if max_depth is not None and depth > max_depth:
        return 0

    if not page_names:
        return 0

    # Deduplicate and check for existing pages
    from imas_codex.discovery.wiki.scraper import canonical_page_id

    created = 0
    with GraphClient() as gc:
        for name in page_names:
            page_id = canonical_page_id(name, facility)

            # Construct URL based on site type
            url = None
            if base_url:
                if site_type == "twiki":
                    if "/" not in name:
                        name_with_web = f"Main/{name}"
                    else:
                        name_with_web = name
                    url = f"{base_url}/bin/view/{name_with_web}"
                elif site_type == "confluence":
                    url = f"{base_url}/pages/viewpage.action?pageId={name}"
                else:
                    # MediaWiki - use index.php format for consistency
                    url = f"{base_url}/index.php?title={urllib.parse.quote(name, safe='')}"

            # MERGE to avoid duplicates
            result = gc.query(
                """
                MERGE (wp:WikiPage {id: $id})
                ON CREATE SET wp.title = $title,
                              wp.url = $url,
                              wp.facility_id = $facility,
                              wp.status = $scanned,
                              wp.link_depth = $depth,
                              wp.discovered_at = datetime()
                ON MATCH SET wp.link_depth = CASE
                    WHEN wp.link_depth IS NULL OR wp.link_depth > $depth
                    THEN $depth ELSE wp.link_depth END
                RETURN wp.status AS status
                """,
                id=page_id,
                title=name,
                url=url,
                facility=facility,
                scanned=WikiPageStatus.scanned.value,
                depth=depth,
            )
            if result and result[0]["status"] == WikiPageStatus.scanned.value:
                created += 1

    return created


def extract_artifacts_from_html(html: str, base_url: str) -> list[tuple[str, str]]:
    """Extract artifact links from HTML content.

    Lightweight artifact extraction for use during scoring phase
    when bulk discovery is used (no scan workers).

    Args:
        html: HTML content of the page
        base_url: Base URL of the wiki (for resolving relative links)

    Returns:
        List of (url, artifact_type) tuples
    """
    import re
    from urllib.parse import urljoin

    artifacts: list[tuple[str, str]] = []

    # Find all href links
    href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)

    for match in href_pattern.finditer(html):
        link = match.group(1)

        # Skip anchors, javascript, mailto
        if link.startswith(("#", "javascript:", "mailto:")):
            continue

        # Skip MediaWiki File: description pages - these are NOT the actual files
        # The actual file URLs contain /images/ or /uploads/
        if "/File:" in link or "File:" in link.split("/")[-1]:
            continue

        # Check if it's an artifact
        if _is_artifact(link):
            # Resolve relative URLs
            if not link.startswith(("http://", "https://")):
                full_url = urljoin(base_url, link)
            else:
                full_url = link

            artifact_type = _get_artifact_type(link)
            artifacts.append((full_url, artifact_type))

    # Deduplicate
    return list(set(artifacts))


def _create_discovered_artifacts(
    facility: str,
    artifact_links: list[tuple[str, str]],
    source_page_id: str | None = None,
) -> int:
    """Create pending artifact nodes and link them to the source page.

    Args:
        facility: Facility ID
        artifact_links: List of (url, artifact_type) tuples
        source_page_id: WikiPage ID that links to these artifacts

    Returns:
        Number of artifacts created/linked
    """
    from imas_codex.graph.models import WikiArtifactStatus

    if not artifact_links:
        return 0

    created = 0
    with GraphClient() as gc:
        for path, artifact_type in artifact_links:
            filename = path.split("/")[-1]
            artifact_id = f"{facility}:{filename}"

            # Create artifact and link to source page in single query
            if source_page_id:
                gc.query(
                    """
                    MERGE (wa:WikiArtifact {id: $id})
                    ON CREATE SET wa.facility_id = $facility,
                                  wa.filename = $filename,
                                  wa.url = $path,
                                  wa.artifact_type = $artifact_type,
                                  wa.status = $discovered,
                                  wa.discovered_at = datetime()
                    WITH wa
                    MATCH (wp:WikiPage {id: $page_id})
                    MERGE (wp)-[:HAS_ARTIFACT]->(wa)
                    """,
                    id=artifact_id,
                    facility=facility,
                    filename=filename,
                    path=path,
                    artifact_type=artifact_type,
                    discovered=WikiArtifactStatus.discovered.value,
                    page_id=source_page_id,
                )
            else:
                # Fallback for bulk discovery (no source page)
                gc.query(
                    """
                    MERGE (wa:WikiArtifact {id: $id})
                    ON CREATE SET wa.facility_id = $facility,
                                  wa.filename = $filename,
                                  wa.url = $path,
                                  wa.artifact_type = $artifact_type,
                                  wa.status = $discovered,
                                  wa.discovered_at = datetime()
                    """,
                    id=artifact_id,
                    facility=facility,
                    filename=filename,
                    path=path,
                    artifact_type=artifact_type,
                    discovered=WikiArtifactStatus.discovered.value,
                )
            created += 1

    return created


async def _fetch_and_summarize(
    url: str,
    ssh_host: str | None,
    auth_type: str | None = None,
    credential_service: str | None = None,
    max_chars: int = 2000,
    async_wiki_client: Any = None,
) -> str:
    """Fetch page content and extract text preview.

    No LLM is used here - prefetch extracts text deterministically.
    The summary is just cleaned text for the scorer to evaluate.

    Args:
        url: Page URL to fetch
        ssh_host: Optional SSH host for proxied fetching
        auth_type: Authentication type (tequila, session, etc.)
        credential_service: Keyring service for credentials
        max_chars: Maximum characters to extract (default 2000, use 1500 for scoring)
        async_wiki_client: Shared AsyncMediaWikiClient for native async HTTP

    Returns:
        Extracted text preview or empty string on error
    """
    from imas_codex.discovery.wiki.prefetch import extract_text_from_html

    def _ssh_fetch() -> str:
        """Blocking SSH fetch - run in thread pool."""
        cmd = f'curl -sk "{url}" 2>/dev/null'
        try:
            result = subprocess.run(
                ["ssh", ssh_host, cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
            return ""
        except subprocess.TimeoutExpired:
            logger.warning("Timeout fetching %s via SSH", url)
            return ""
        except Exception as e:
            logger.warning("Error fetching %s via SSH: %s", url, e)
            return ""

    async def _async_tequila_fetch() -> str:
        """Fetch with Tequila authentication using async client."""
        import urllib.parse as urlparse

        # Extract page name from URL
        page_name = url.split("/wiki/")[-1] if "/wiki/" in url else url.split("/")[-1]
        if "?" in page_name:
            parsed = urlparse.parse_qs(urlparse.urlparse(url).query)
            page_name = parsed.get("title", [page_name])[0]
        page_name = urlparse.unquote(page_name)

        # Use provided async client
        if async_wiki_client is not None:
            try:
                page = await async_wiki_client.get_page(page_name)
                if page:
                    return page.content_html
                return ""
            except Exception as e:
                logger.debug("Async client fetch failed for %s: %s", url, e)
                return ""

        # No client provided - create a new async client
        from imas_codex.discovery.wiki.mediawiki import AsyncMediaWikiClient

        base_url = url.rsplit("/", 1)[0] if "/wiki/" in url else url.rsplit("/", 1)[0]
        if "/wiki" in base_url:
            base_url = base_url.rsplit("/wiki", 1)[0] + "/wiki"

        async with AsyncMediaWikiClient(
            base_url=base_url,
            credential_service=credential_service or "tcv-wiki",
            verify_ssl=False,
        ) as client:
            if not await client.authenticate():
                logger.warning("Tequila auth failed for %s", url)
                return ""
            page = await client.get_page(page_name)
            return page.content_html if page else ""

    # Determine fetch strategy
    if ssh_host:
        # Fetch via SSH proxy using curl in thread pool (subprocess is blocking)
        html = await asyncio.to_thread(_ssh_fetch)
    elif auth_type in ("tequila", "session"):
        # Fetch via Tequila SSO authentication using native async HTTP
        html = await _async_tequila_fetch()
    else:
        # Direct HTTP fetch (no auth) - already async
        from imas_codex.discovery.wiki.prefetch import fetch_page_content

        html, error = await fetch_page_content(url)
        if error:
            logger.debug("Failed to fetch %s: %s", url, error)
            html = ""

    if html:
        return extract_text_from_html(html, max_chars=max_chars)
    return ""


async def _score_pages_batch(
    pages: list[dict[str, Any]],
    model: str,
    focus: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Score a batch of pages using LLM with structured output.

    Uses litellm.acompletion with WikiScoreBatch Pydantic model for
    structured output. Content-based scoring with per-dimension scores.

    Args:
        pages: List of page dicts with id, title, summary, preview_text, etc.
        model: Model identifier from get_model_for_task()
        focus: Optional focus area for scoring

    Returns:
        (results, cost) tuple where cost is actual LLM cost from OpenRouter.
    """
    import os
    import re

    import litellm

    from imas_codex.agentic.prompt_loader import render_prompt
    from imas_codex.discovery.wiki.models import (
        WikiScoreBatch,
        grounded_wiki_score,
    )

    # Get API key - same pattern as discovery/scorer.py
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Set it in .env or export it."
        )

    # Ensure OpenRouter prefix
    model_id = model
    if not model_id.startswith("openrouter/"):
        model_id = f"openrouter/{model_id}"

    # Build system prompt using dynamic template with schema injection
    context: dict[str, Any] = {}
    if focus:
        context["focus"] = focus

    system_prompt = render_prompt("wiki/scorer", context)

    # Build user prompt with page content (not graph metrics)
    lines = [
        f"Score these {len(pages)} wiki pages based on their content.",
        "(Use the preview text to infer value - graph metrics like in_degree are NOT indicators.)\n",
    ]

    for i, p in enumerate(pages, 1):
        lines.append(f"\n## Page {i}")
        lines.append(f"ID: {p['id']}")
        lines.append(f"Title: {p.get('title', 'Unknown')}")

        # Use preview_text for content-based scoring (preferred over summary)
        preview = p.get("preview_text") or p.get("summary") or ""
        if preview:
            lines.append(f"Preview: {preview[:800]}")

        # Include URL for context (Confluence vs MediaWiki structure hints)
        url = p.get("url")
        if url:
            lines.append(f"URL: {url}")

    lines.append(
        "\n\nReturn results for each page in order. "
        "The response format is enforced by the schema."
    )

    user_prompt = "\n".join(lines)

    # Retry loop for rate limiting and JSON parsing errors (truncated responses)
    max_retries = 3
    retry_base_delay = 2.0
    last_error = None
    total_cost = 0.0

    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=model_id,
                api_key=api_key,
                response_format=WikiScoreBatch,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=32000,
                timeout=120,  # 2 minute timeout to prevent indefinite hangs
            )

            # Extract actual cost from OpenRouter response
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            if (
                hasattr(response, "_hidden_params")
                and "response_cost" in response._hidden_params
            ):
                cost = response._hidden_params["response_cost"]
            else:
                # Fallback: Claude Sonnet rates via OpenRouter ($3/$15 per 1M tokens)
                cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000

            total_cost += cost

            # Parse response using Pydantic structured output
            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM returned empty response, returning empty results")
                return [], total_cost

            # Sanitize: remove control characters (except newline/tab)
            content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
            content = content.encode("utf-8", errors="surrogateescape").decode(
                "utf-8", errors="replace"
            )

            batch = WikiScoreBatch.model_validate_json(content)
            llm_results = batch.results
            break  # Success - exit retry loop

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Retry on rate limiting, server errors, OR JSON parsing errors
            is_retryable = any(
                x in error_msg
                for x in [
                    "overloaded",
                    "rate",
                    "429",
                    "503",
                    "timeout",
                    "eof",  # EOF while parsing JSON
                    "json",  # JSON parsing errors
                    "truncated",
                    "validation",
                ]
            )

            if is_retryable and attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)
                logger.warning(
                    "LLM error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    str(e)[:100],
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                # Last attempt or non-retryable error
                logger.error(
                    "LLM validation error for batch of %d pages: %s. "
                    "Pages will be reverted to scanned status.",
                    len(pages),
                    e,
                )
                raise ValueError(f"LLM response validation failed: {e}") from e
    else:
        raise last_error  # type: ignore[misc]

    # Convert to result dicts, computing combined scores
    cost_per_page = total_cost / len(pages) if pages else 0.0
    results = []

    for r in llm_results[: len(pages)]:
        # Build per-dimension scores dict
        scores = {
            "score_data_documentation": r.score_data_documentation,
            "score_physics_content": r.score_physics_content,
            "score_code_documentation": r.score_code_documentation,
            "score_data_access": r.score_data_access,
            "score_calibration": r.score_calibration,
            "score_imas_relevance": r.score_imas_relevance,
        }

        # Compute combined score using grounded function
        combined_score = grounded_wiki_score(scores, r.page_purpose)

        results.append(
            {
                "id": r.id,
                "score": combined_score,
                "page_purpose": r.page_purpose.value,
                "description": r.description[:150],
                "reasoning": r.reasoning[:80],
                "keywords": r.keywords[:5],
                "physics_domain": r.physics_domain.value if r.physics_domain else None,
                "should_ingest": r.should_ingest,
                "skip_reason": r.skip_reason or None,
                # Per-dimension scores
                "score_data_documentation": r.score_data_documentation,
                "score_physics_content": r.score_physics_content,
                "score_code_documentation": r.score_code_documentation,
                "score_data_access": r.score_data_access,
                "score_calibration": r.score_calibration,
                "score_imas_relevance": r.score_imas_relevance,
                # Legacy fields for compatibility
                "page_type": r.page_purpose.value,
                "is_physics": r.physics_domain is not None
                and r.physics_domain.value != "general",
                "score_cost": cost_per_page,
            }
        )

    return results, cost


def _score_pages_heuristic(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Heuristic fallback scoring based on keywords. Zero cost."""
    results = []
    for page in pages:
        title = page.get("title", "").lower()
        summary = page.get("summary", "") or ""

        score = 0.5
        reasoning = "Default score"

        physics_keywords = [
            "thomson",
            "liuqe",
            "equilibrium",
            "mhd",
            "plasma",
            "diagnostic",
            "calibration",
            "signal",
            "node",
        ]
        low_value_keywords = [
            "meeting",
            "workshop",
            "todo",
            "draft",
            "notes",
            "personal",
            "test",
            "sandbox",
        ]

        for kw in physics_keywords:
            if kw in title or kw in summary.lower():
                score = min(score + 0.15, 1.0)
                reasoning = f"Contains physics keyword: {kw}"

        for kw in low_value_keywords:
            if kw in title:
                score = max(score - 0.2, 0.0)
                reasoning = f"Contains low-value keyword: {kw}"

        results.append(
            {
                "id": page["id"],
                "score": score,
                "reasoning": reasoning,
                "page_type": "documentation",
                "is_physics": score >= 0.6,
            }
        )

    return results


async def _ingest_page(
    url: str,
    page_id: str,
    facility: str,
    site_type: str,
    ssh_host: str | None,
    auth_type: str | None = None,
    credential_service: str | None = None,
    async_wiki_client: Any = None,
) -> int:
    """Ingest a page: fetch content, chunk, and embed.

    Uses the WikiIngestionPipeline for proper chunking and embedding.

    Args:
        url: Page URL to fetch
        page_id: Unique page identifier
        facility: Facility ID (e.g., 'tcv', 'jet')
        site_type: Site type ('mediawiki', 'confluence', 'twiki')
        ssh_host: Optional SSH host for proxied fetching
        auth_type: Authentication type (tequila, session, etc.)
        credential_service: Keyring service for credentials
        async_wiki_client: Optional shared AsyncMediaWikiClient for native async HTTP

    Returns:
        Number of chunks created
    """
    from imas_codex.discovery.wiki.pipeline import WikiIngestionPipeline
    from imas_codex.discovery.wiki.scraper import WikiPage

    # Extract page name from URL or page_id
    page_name = page_id.split(":", 1)[1] if ":" in page_id else page_id

    # Fetch HTML content with auth
    html = await _fetch_html(
        url,
        ssh_host,
        auth_type=auth_type,
        credential_service=credential_service,
        async_wiki_client=async_wiki_client,
    )
    if not html or len(html) < 100:
        logger.warning("Insufficient content for %s", page_id)
        return 0

    # Extract title from HTML
    import re

    title_match = re.search(r"<title>([^<]+)</title>", html)
    title = title_match.group(1) if title_match else page_name

    # Clean up title (remove wiki suffix)
    for suffix in [" - SPCwiki", " - Wikipedia", " - Confluence"]:
        if title.endswith(suffix):
            title = title[: -len(suffix)]

    # Create WikiPage object (fields from dataclass in scraper.py)
    page = WikiPage(
        url=url,
        title=title,
        content_html=html,
        content_text="",  # Will be extracted by pipeline
        sections={},
        mdsplus_paths=[],  # Will be extracted by pipeline
        imas_paths=[],
        units=[],
        conventions=[],
    )

    # Use the ingestion pipeline
    pipeline = WikiIngestionPipeline(
        facility_id=facility,
        use_rich=False,  # No progress display in worker
    )

    try:
        stats = await pipeline.ingest_page(page)
        return stats.get("chunks", 0)
    except Exception as e:
        logger.warning("Failed to ingest %s: %s", page_id, e)
        return 0


async def _fetch_html(
    url: str,
    ssh_host: str | None,
    auth_type: str | None = None,
    credential_service: str | None = None,
    async_wiki_client: Any = None,
) -> str:
    """Fetch HTML content from URL.

    Args:
        url: Page URL
        ssh_host: Optional SSH host for proxied fetching
        auth_type: Authentication type (tequila, session, etc.)
        credential_service: Keyring service for credentials
        async_wiki_client: Shared AsyncMediaWikiClient for native async HTTP

    Returns:
        HTML content string or empty string on error
    """

    def _ssh_fetch() -> str:
        """Fetch via SSH proxy."""
        cmd = f'curl -sk "{url}" 2>/dev/null'
        try:
            result = subprocess.run(
                ["ssh", ssh_host, cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
            return ""
        except Exception as e:
            logger.warning("SSH fetch failed for %s: %s", url, e)
            return ""

    async def _async_tequila_fetch() -> str:
        """Fetch with Tequila authentication using async client."""
        import urllib.parse as urlparse

        # Extract page name from URL
        page_name = url.split("/wiki/")[-1] if "/wiki/" in url else url.split("/")[-1]
        if "?" in page_name:
            parsed = urlparse.parse_qs(urlparse.urlparse(url).query)
            page_name = parsed.get("title", [page_name])[0]
        page_name = urlparse.unquote(page_name)

        # Use provided async client
        if async_wiki_client is not None:
            try:
                page = await async_wiki_client.get_page(page_name)
                if page:
                    return page.content_html
                return ""
            except Exception as e:
                logger.debug("Async client fetch failed for %s: %s", url, e)
                return ""

        # No client provided - create a new async client
        from imas_codex.discovery.wiki.mediawiki import AsyncMediaWikiClient

        base_url_local = (
            url.rsplit("/", 1)[0] if "/wiki/" in url else url.rsplit("/", 1)[0]
        )
        if "/wiki" in base_url_local:
            base_url_local = base_url_local.rsplit("/wiki", 1)[0] + "/wiki"

        async with AsyncMediaWikiClient(
            base_url=base_url_local,
            credential_service=credential_service or "tcv-wiki",
            verify_ssl=False,
        ) as client:
            if not await client.authenticate():
                logger.warning("Tequila auth failed for %s", url)
                return ""
            page = await client.get_page(page_name)
            return page.content_html if page else ""

    # Determine fetch strategy
    if ssh_host:
        return await asyncio.to_thread(_ssh_fetch)
    elif auth_type in ("tequila", "session"):
        return await _async_tequila_fetch()
    else:
        # Direct HTTP fetch (no auth) - already async
        from imas_codex.discovery.wiki.prefetch import fetch_page_content

        html, error = await fetch_page_content(url)
        if html:
            return html
        if error:
            logger.debug("HTTP fetch failed for %s: %s", url, error)
        return ""


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_parallel_wiki_discovery(
    facility: str,
    site_type: str,
    base_url: str,
    portal_page: str,
    ssh_host: str | None = None,
    auth_type: str | None = None,
    credential_service: str | None = None,
    cost_limit: float = 10.0,
    page_limit: int | None = None,
    max_depth: int | None = None,
    focus: str | None = None,
    num_scan_workers: int = 1,
    num_score_workers: int = 2,
    num_ingest_workers: int = 2,
    scan_only: bool = False,
    score_only: bool = False,
    bulk_discover: bool = True,
    ingest_artifacts: bool = True,
    on_scan_progress: Callable | None = None,
    on_score_progress: Callable | None = None,
    on_ingest_progress: Callable | None = None,
    on_artifact_progress: Callable | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
) -> dict[str, Any]:
    """Run parallel wiki discovery with async workers.

    Args:
        ssh_host: SSH host for proxied access (for non-public wikis)
        auth_type: Authentication type (tequila, session, ssh_proxy, or None)
        credential_service: Keyring service name for credentials
        bulk_discover: If True (default), use Special:AllPages for fast
            discovery of all pages upfront. This is 100-300x faster than
            crawling links page-by-page. Only works for MediaWiki sites.
        ingest_artifacts: If True (default), start artifact worker to
            download and ingest PDF artifacts discovered during scanning.
        on_artifact_progress: Progress callback for artifact worker.
        on_worker_status: Callback for worker status changes. Called with
            SupervisedWorkerGroup for live status display.

    Returns:
        Dict with discovery statistics
    """
    start_time = time.time()

    # Reset orphans from previous runs
    reset_transient_pages(facility)

    # Initialize state with auth info
    state = WikiDiscoveryState(
        facility=facility,
        site_type=site_type,
        base_url=base_url,
        portal_page=portal_page,
        ssh_host=ssh_host,
        auth_type=auth_type,
        credential_service=credential_service,
        cost_limit=cost_limit,
        page_limit=page_limit,
        max_depth=max_depth,
        focus=focus,
    )

    # Create worker group for status tracking
    worker_group = SupervisedWorkerGroup()

    # Bulk discovery: use Special:AllPages to find all pages instantly
    # This replaces the slow scan phase for MediaWiki sites
    bulk_discovered = 0
    if bulk_discover and site_type == "mediawiki" and not score_only:
        # Choose discovery method based on auth type
        if state.auth_type == "tequila" and state.credential_service:
            # HTTP-based discovery with Tequila auth
            logger.info("Using bulk discovery via Special:AllPages (HTTP/Tequila)...")

            def bulk_progress(msg, _stats):
                if on_scan_progress:
                    on_scan_progress(f"bulk: {msg}", state.scan_stats)

            # Run bulk discovery in thread pool (blocking HTTP calls)
            bulk_discovered = await asyncio.to_thread(
                bulk_discover_all_pages_http,
                facility,
                base_url,
                state.credential_service,
                bulk_progress,
            )
        elif ssh_host:
            # SSH-based discovery
            logger.info("Using bulk discovery via Special:AllPages (SSH)...")

            def bulk_progress(msg, _stats):
                if on_scan_progress:
                    on_scan_progress(f"bulk: {msg}", state.scan_stats)

            # Run bulk discovery in thread pool (blocking SSH calls)
            bulk_discovered = await asyncio.to_thread(
                bulk_discover_all_pages_mediawiki,
                facility,
                base_url,
                ssh_host,
                bulk_progress,
            )

        if bulk_discovered:
            logger.info(f"Bulk discovery found {bulk_discovered} pages")
            state.scan_stats.processed = bulk_discovered
        state.scan_stats.processed = bulk_discovered

    # Bulk artifact discovery: use platform API to find all artifacts
    # This is much faster than scanning each page for links
    bulk_artifacts_discovered = 0
    if bulk_discover and ingest_artifacts and not score_only:
        logger.info("Starting bulk artifact discovery via API...")

        def artifact_progress(msg, _stats):
            if on_artifact_progress:
                on_artifact_progress(f"bulk: {msg}", state.artifact_stats)

        # Bulk discovery uses SSH when available, no async wiki client support yet
        bulk_artifacts_discovered = await asyncio.to_thread(
            bulk_discover_artifacts,
            facility,
            base_url,
            site_type,
            ssh_host,
            None,  # wiki_client - removed, use SSH path
            state.credential_service,
            artifact_progress,
        )

        if bulk_artifacts_discovered:
            logger.info(f"Bulk discovery found {bulk_artifacts_discovered} artifacts")
            state.artifact_stats.processed = bulk_artifacts_discovered

    # Create portal page if not exists (may already exist from bulk discovery)
    _seed_portal_page(facility, portal_page, base_url, site_type)

    # Start supervised workers with status tracking
    if not score_only:
        # Skip scan workers if bulk discovery was used
        if not bulk_discover or bulk_discovered == 0:
            for i in range(num_scan_workers):
                worker_name = f"scan_worker_{i}"
                status = worker_group.create_status(worker_name)
                worker_group.add_task(
                    asyncio.create_task(
                        supervised_worker(
                            scan_worker,
                            worker_name,
                            state,
                            state.should_stop_scanning,
                            on_progress=on_scan_progress,
                            status_tracker=status,
                        )
                    )
                )

    if not scan_only:
        for i in range(num_score_workers):
            worker_name = f"score_worker_{i}"
            status = worker_group.create_status(worker_name)
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        score_worker,
                        worker_name,
                        state,
                        state.should_stop_scoring,
                        on_progress=on_score_progress,
                        status_tracker=status,
                    )
                )
            )

        for i in range(num_ingest_workers):
            worker_name = f"ingest_worker_{i}"
            ingest_status = worker_group.create_status(worker_name)
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        ingest_worker,
                        worker_name,
                        state,
                        state.should_stop_ingesting,
                        on_progress=on_ingest_progress,
                        status_tracker=ingest_status,
                    )
                )
            )

        # Artifact worker runs by default to ingest PDFs discovered during scanning
        if ingest_artifacts:
            artifact_status = worker_group.create_status("artifact_worker")
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        artifact_worker,
                        "artifact_worker",
                        state,
                        state.should_stop_artifact_worker,
                        on_progress=on_artifact_progress,
                        status_tracker=artifact_status,
                    )
                )
            )

    logger.info(
        f"Started {worker_group.get_active_count()} workers: "
        f"score_only={score_only}, scan_only={scan_only}, "
        f"ingest_artifacts={ingest_artifacts}"
    )

    # Send initial worker status update immediately so display shows workers
    if on_worker_status:
        try:
            on_worker_status(worker_group)
        except Exception as e:
            logger.warning("Initial worker status callback failed: %s", e)

    # Wait for termination condition with periodic orphan recovery
    orphan_check_interval = 60  # Check every 60 seconds
    last_orphan_check = time.time()
    status_update_interval = 0.5  # Update status every 0.5 seconds
    last_status_update = time.time()

    while not state.should_stop():
        await asyncio.sleep(0.25)

        # Update worker status for display
        if (
            on_worker_status
            and time.time() - last_status_update > status_update_interval
        ):
            try:
                on_worker_status(worker_group)
            except Exception as e:
                logger.warning("Worker status callback failed: %s", e)
            last_status_update = time.time()

        # Periodically release orphaned claims (from crashed workers)
        if time.time() - last_orphan_check > orphan_check_interval:
            try:
                released = release_orphaned_claims(facility)
                if released.get("released_pages", 0) or released.get(
                    "released_artifacts", 0
                ):
                    logger.info(
                        "Recovered %d orphaned page claims, %d artifact claims",
                        released.get("released_pages", 0),
                        released.get("released_artifacts", 0),
                    )
            except Exception as e:
                logger.debug("Orphan recovery check failed: %s", e)
            last_orphan_check = time.time()

    # Stop workers
    state.stop_requested = True
    await worker_group.cancel_all()

    # Clean up async wiki client
    await state.close_async_wiki_client()

    elapsed = time.time() - start_time

    return {
        "scanned": state.scan_stats.processed,
        "scored": state.score_stats.processed,
        "ingested": state.ingest_stats.processed,
        "artifacts": state.artifact_stats.processed,
        "cost": state.total_cost,
        "elapsed_seconds": elapsed,
        "scan_rate": state.scan_stats.rate,
        "score_rate": state.score_stats.rate,
    }


def _seed_portal_page(
    facility: str,
    portal_page: str,
    base_url: str,
    site_type: str,
) -> None:
    """Create the portal page as initial seed if it doesn't exist."""
    from imas_codex.discovery.wiki.scraper import canonical_page_id

    page_id = canonical_page_id(portal_page, facility)

    # Build URL based on site type
    if site_type == "twiki":
        if "/" not in portal_page:
            portal_page = f"Main/{portal_page}"
        url = f"{base_url}/bin/view/{portal_page}"
    elif site_type == "confluence":
        url = f"{base_url}/pages/viewpage.action?pageId={portal_page}"
    else:
        url = f"{base_url}/{urllib.parse.quote(portal_page, safe='/')}"

    with GraphClient() as gc:
        gc.query(
            """
            MERGE (wp:WikiPage {id: $id})
            ON CREATE SET wp.title = $title,
                          wp.url = $url,
                          wp.facility_id = $facility,
                          wp.status = $scanned,
                          wp.link_depth = 0,
                          wp.discovered_at = datetime()
            """,
            id=page_id,
            title=portal_page,
            url=url,
            facility=facility,
            scanned=WikiPageStatus.scanned.value,
        )


# =============================================================================
# Stats Query
# =============================================================================


def get_wiki_discovery_stats(facility: str) -> dict[str, int | float]:
    """Get wiki discovery statistics from graph.

    Returns stats needed for progress display:
    - total: Total wiki pages for this facility
    - scanned: Pages with status=scanned (awaiting scoring)
    - scored: Pages with status=scored (awaiting ingest or skipped)
    - ingested: Pages with status=ingested (final state)
    - pending_score: Same as scanned count (for progress display)
    - pending_ingest: Scored pages with score >= 0.5 awaiting ingestion
    - accumulated_cost: Total score_cost from all scored/ingested pages
    - skipped: Pages with status=skipped or score < 0.5
    """
    with GraphClient() as gc:
        # Get status counts
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WITH wp.status AS status
            RETURN status, count(*) AS count
            """,
            facility=facility,
        )

        stats: dict[str, int | float] = {
            "total": 0,
            "discovered": 0,
            "scanning": 0,
            "scanned": 0,
            "scoring": 0,
            "scored": 0,
            "ingesting": 0,
            "ingested": 0,
            "skipped": 0,
            "failed": 0,
        }

        for r in result:
            status = r["status"]
            count = r["count"]
            if status in stats:
                stats[status] = count
            stats["total"] += count

        # Get pending ingest count (scored pages with score >= 0.5)
        ingest_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $scored AND wp.score >= 0.5
            RETURN count(wp) AS pending_ingest
            """,
            facility=facility,
            scored=WikiPageStatus.scored.value,
        )
        stats["pending_score"] = stats["scanned"]
        stats["pending_ingest"] = (
            ingest_result[0]["pending_ingest"] if ingest_result else 0
        )

        # Get accumulated cost from all pages
        cost_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.score_cost IS NOT NULL
            RETURN sum(wp.score_cost) AS total_cost
            """,
            facility=facility,
        )
        stats["accumulated_cost"] = (
            cost_result[0]["total_cost"]
            if cost_result and cost_result[0]["total_cost"]
            else 0.0
        )

        # Add artifact stats
        artifact_result = gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility})
            RETURN count(*) AS total_artifacts
            """,
            facility=facility,
        )
        stats["total_artifacts"] = (
            artifact_result[0]["total_artifacts"] if artifact_result else 0
        )

        return stats
