"""Graph operations for wiki discovery.

Neo4j graph helpers for wiki page and artifact lifecycle management:
- Bulk node creation (WikiPage, WikiArtifact)
- Work claiming with claim_token pattern
- Status transitions (mark scored, ingested, failed)
- Orphan recovery and transient reset
- Pending work queries for worker coordination

All functions use the retry_on_deadlock decorator for Neo4j transient errors.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import TYPE_CHECKING, Any

from neo4j.exceptions import TransientError

from imas_codex.graph import GraphClient
from imas_codex.graph.models import WikiArtifactStatus, WikiPageStatus

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Claim timeout - pages claimed longer than this are reclaimed
CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes

# Retry configuration for Neo4j transient errors (deadlocks)
MAX_RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 0.1  # seconds
RETRY_MAX_DELAY = 2.0  # seconds

# Artifact type classification using semantic names (matching ArtifactType enum).
# Adapters produce these names; all downstream code uses them consistently.

# Types we can extract full text from for chunking and embedding
INGESTABLE_ARTIFACT_TYPES = {
    "pdf",
    "document",
    "presentation",
    "spreadsheet",
    "notebook",
    "json",
}

# Types that should become Image nodes (routed to image pipeline, not text extraction).
# Used by artifact_worker for routing; Cypher queries use score_exempt property instead.
IMAGE_ARTIFACT_TYPES = {"image"}

# All types worth scoring via LLM (metadata-only scoring for data/archive/other).
# Image artifacts are excluded — they are score_exempt and bypass LLM text scoring,
# going directly from discovered → ingested via the VLM captioning pipeline.
SCORABLE_ARTIFACT_TYPES = INGESTABLE_ARTIFACT_TYPES | {
    "data",
    "archive",
    "other",
}


def _bulk_create_wiki_pages(
    gc: GraphClient,
    facility: str,
    batch_data: list[dict],
    *,
    batch_size: int = 500,
    on_progress: Callable | None = None,
) -> int:
    """Create WikiPage nodes with AT_FACILITY relationship in batches.

    Every WikiPage gets a [:AT_FACILITY]->(:Facility) relationship at creation
    time, not deferred to ingestion.

    Args:
        gc: Open GraphClient
        facility: Facility ID
        batch_data: List of dicts with id, title, url keys
        batch_size: Nodes per batch
        on_progress: Optional progress callback

    Returns:
        Number of pages created/updated
    """
    created = 0
    total = len(batch_data)
    for i in range(0, total, batch_size):
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
            WITH wp
            MATCH (f:Facility {id: $facility})
            MERGE (wp)-[:AT_FACILITY]->(f)
            RETURN count(wp) AS count
            """,
            pages=batch,
            facility=facility,
            scanned=WikiPageStatus.scanned.value,
        )
        if result:
            created += result[0]["count"]

        if on_progress:
            on_progress(f"creating pages ({i + len(batch)}/{total})", None)

    return created


def _bulk_create_wiki_artifacts(
    gc: GraphClient,
    facility: str,
    batch_data: list[dict],
    *,
    batch_size: int = 500,
    on_progress: Callable | None = None,
) -> int:
    """Create WikiArtifact nodes with AT_FACILITY and HAS_ARTIFACT relationships.

    Args:
        gc: Open GraphClient
        facility: Facility ID
        batch_data: List of dicts with id, filename, url, artifact_type,
                    and optionally size_bytes, mime_type, linked_pages
        batch_size: Nodes per batch
        on_progress: Optional progress callback

    Returns:
        Number of artifacts created/updated
    """
    # Pre-compute score_exempt flag for each artifact.
    # Image artifacts bypass LLM text scoring and go directly to ingestion
    # (discovered → ingested) via the VLM captioning pipeline.
    for a in batch_data:
        a.setdefault(
            "score_exempt", a.get("artifact_type", "").lower() in IMAGE_ARTIFACT_TYPES
        )

    created = 0
    total = len(batch_data)
    for i in range(0, total, batch_size):
        batch = batch_data[i : i + batch_size]
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
                          wa.score_exempt = a.score_exempt,
                          wa.discovered_at = datetime(),
                          wa.bulk_discovered = true
            ON MATCH SET wa.bulk_discovered = true
            WITH wa
            MATCH (f:Facility {id: $facility})
            MERGE (wa)-[:AT_FACILITY]->(f)
            RETURN count(wa) AS count
            """,
            artifacts=batch,
            facility=facility,
            discovered=WikiArtifactStatus.discovered.value,
        )
        if result:
            created += result[0]["count"]

        if on_progress:
            on_progress(f"created {i + len(batch)}/{total} artifacts", None)

    # Create HAS_ARTIFACT relationships from linked pages
    # This is done in a separate pass to handle pages that may not exist yet
    page_links = []
    for a in batch_data:
        artifact_id = a["id"]
        for page_name in a.get("linked_pages", []):
            page_id = f"{facility}:{page_name}"
            page_links.append({"artifact_id": artifact_id, "page_id": page_id})

    if page_links:
        # Link artifacts to their parent pages
        # Use MERGE for WikiPage to handle cases where the page hasn't been
        # fully ingested yet — creates a stub node that will be enriched later
        # when the page is actually crawled and ingested.
        # Always set facility_id and AT_FACILITY so referential integrity holds.
        gc.query(
            """
            UNWIND $links AS link
            MATCH (wa:WikiArtifact {id: link.artifact_id})
            MERGE (wp:WikiPage {id: link.page_id})
            ON CREATE SET wp.facility_id = $facility,
                          wp.status = 'scanned',
                          wp.title = link.page_id,
                          wp.url = '',
                          wp.discovered_at = datetime()
            MERGE (wp)-[:HAS_ARTIFACT]->(wa)
            WITH wp
            MATCH (f:Facility {id: $facility})
            MERGE (wp)-[:AT_FACILITY]->(f)
            """,
            links=page_links,
            facility=facility,
        )

    return created


# Retry configuration for Neo4j transient errors (deadlocks)
MAX_RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 0.1  # seconds
RETRY_MAX_DELAY = 2.0  # seconds


def retry_on_deadlock(
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    base_delay: float = RETRY_BASE_DELAY,
    max_delay: float = RETRY_MAX_DELAY,
):
    """Decorator to retry functions on Neo4j transient errors (e.g., deadlocks).

    Uses exponential backoff with jitter to reduce contention.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except TransientError as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        # Exponential backoff with jitter
                        delay = min(base_delay * (2**attempt), max_delay)
                        jitter = random.uniform(0, delay * 0.5)
                        sleep_time = delay + jitter
                        logger.debug(
                            "%s: transient error (attempt %d/%d), "
                            "retrying in %.2fs: %s",
                            func.__name__,
                            attempt + 1,
                            max_attempts,
                            sleep_time,
                            e,
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.warning(
                            "%s: transient error after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            e,
                        )
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


def has_pending_artifact_work(facility: str) -> bool:
    """Check if there are pending artifacts (scoring or ingestion).

    Artifacts are pending when:
    - status = 'discovered' AND NOT score_exempt (needs LLM scoring)
    - status = 'discovered' AND score_exempt (needs direct ingestion)
    - status = 'scored' AND score >= 0.5 AND ingestable type (needs ingestion)
    - claimed_at is null or expired
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility})
            WHERE (
                (wa.status = $discovered)
                OR (wa.status = $scored AND wa.score >= 0.5
                    AND wa.artifact_type IN $ingestable)
              )
              AND (wa.claimed_at IS NULL OR wa.claimed_at < datetime() - duration($cutoff))
            RETURN count(wa) AS pending
            """,
            facility=facility,
            discovered=WikiArtifactStatus.discovered.value,
            scored=WikiArtifactStatus.scored.value,
            ingestable=list(INGESTABLE_ARTIFACT_TYPES),
            cutoff=cutoff,
        )
        return result[0]["pending"] > 0 if result else False


def has_pending_artifact_score_work(facility: str) -> bool:
    """Check if there are discovered artifacts awaiting scoring.

    Returns True if there are artifacts with:
    - status = 'discovered'
    - artifact_type in SCORABLE_ARTIFACT_TYPES
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
            types=list(SCORABLE_ARTIFACT_TYPES),
            cutoff=cutoff,
        )
        return result[0]["pending"] > 0 if result else False


def has_pending_artifact_ingest_work(facility: str) -> bool:
    """Check if there are artifacts awaiting ingestion.

    Returns True if there are artifacts with:
    - status = 'scored' AND score >= 0.5 AND ingestable type (text-extractable)
    - status = 'discovered' AND score_exempt = true (bypass LLM scoring)
    - claimed_at is null or expired
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility})
            WHERE (
                (wa.status = $scored AND wa.score >= 0.5
                 AND wa.artifact_type IN $types)
                OR (wa.status = $discovered AND wa.score_exempt = true)
              )
              AND (wa.claimed_at IS NULL OR wa.claimed_at < datetime() - duration($cutoff))
            RETURN count(wa) AS pending
            """,
            facility=facility,
            scored=WikiArtifactStatus.scored.value,
            discovered=WikiArtifactStatus.discovered.value,
            types=list(INGESTABLE_ARTIFACT_TYPES),
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
# Orphan Recovery (timeout-based, safe for parallel instances)
# =============================================================================


def reset_transient_pages(
    facility: str, *, silent: bool = False, force: bool = False
) -> dict[str, int]:
    """Release stale claims older than CLAIM_TIMEOUT_SECONDS.

    Uses timeout-based recovery so multiple CLI instances can run
    concurrently on the same facility without wiping each other's claims.
    Only claims older than the timeout are considered orphaned.

    Delegates to the common claims module for both WikiPage and
    WikiArtifact node types.

    Args:
        facility: Facility ID
        silent: Suppress logging
        force: Clear ALL claims regardless of age (use at startup)
    """
    from imas_codex.discovery.base.claims import reset_stale_claims

    timeout = 0 if force else CLAIM_TIMEOUT_SECONDS

    page_reset = reset_stale_claims(
        "WikiPage",
        facility,
        timeout_seconds=timeout,
        silent=True,  # We log our own combined message
    )

    artifact_reset = reset_stale_claims(
        "WikiArtifact",
        facility,
        timeout_seconds=timeout,
        silent=True,
    )

    total_reset = page_reset + artifact_reset
    if not silent and total_reset > 0:
        logger.info(
            "Released %d orphaned claims%s (%d pages, %d artifacts)",
            total_reset,
            "" if force else f" older than {CLAIM_TIMEOUT_SECONDS}s",
            page_reset,
            artifact_reset,
        )

    return {"orphan_reset": page_reset, "artifact_reset": artifact_reset}


# =============================================================================
# Graph-based Work Claiming (uses claimed_at for coordination)
# =============================================================================


@retry_on_deadlock()
def claim_pages_for_scoring(facility: str, limit: int = 50) -> list[dict[str, Any]]:
    """Claim scanned pages for content-aware scoring.

    Workflow: scanned + unclaimed → set claimed_at
    Score worker fetches content and scores in single pass.
    After scoring: update status to 'scored' and set score field.

    Uses claim token pattern to handle race conditions between workers:
    1. Generate unique claim token
    2. Atomically SET token on unclaimed pages
    3. Read back only pages with OUR token (pages we actually won)
    """
    import uuid

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"  # ISO 8601 duration
    claim_token = str(uuid.uuid4())

    with GraphClient() as gc:
        # Step 1: Attempt to claim pages with our unique token
        # Using random() in ORDER BY reduces collision probability further
        gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $scanned
              AND (wp.claimed_at IS NULL
                   OR wp.claimed_at < datetime() - duration($cutoff))
            WITH wp
            ORDER BY rand()
            LIMIT $limit
            SET wp.claimed_at = datetime(), wp.claim_token = $token
            """,
            facility=facility,
            scanned=WikiPageStatus.scanned.value,
            cutoff=cutoff,
            limit=limit,
            token=claim_token,
        )

        # Step 2: Read back only pages WE successfully claimed
        # If another worker raced us, they have a different token
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility, claim_token: $token})
            RETURN wp.id AS id, wp.title AS title, wp.url AS url
            """,
            facility=facility,
            token=claim_token,
        )
        claimed = list(result)

        logger.debug(
            "claim_pages_for_scoring: requested %d, won %d (token=%s)",
            limit,
            len(claimed),
            claim_token[:8],
        )
        return claimed


@retry_on_deadlock()
def claim_pages_for_ingesting(
    facility: str, min_score: float = 0.5, limit: int = 10
) -> list[dict[str, Any]]:
    """Claim scored pages for ingestion (chunking and embedding).

    Workflow: scored + score >= min_score + unclaimed → set claimed_at
    After ingest: update status to 'ingested'.

    Uses claim token pattern to handle race conditions between workers.
    """
    import uuid

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(uuid.uuid4())

    with GraphClient() as gc:
        # Step 1: Attempt to claim pages with our unique token
        gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $scored
              AND wp.score >= $min_score
              AND (wp.claimed_at IS NULL
                   OR wp.claimed_at < datetime() - duration($cutoff))
            WITH wp
            ORDER BY rand()
            LIMIT $limit
            SET wp.claimed_at = datetime(), wp.claim_token = $token
            """,
            facility=facility,
            scored=WikiPageStatus.scored.value,
            min_score=min_score,
            cutoff=cutoff,
            limit=limit,
            token=claim_token,
        )

        # Step 2: Read back only pages WE successfully claimed
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility, claim_token: $token})
            RETURN wp.id AS id, wp.title AS title, wp.url AS url,
                   wp.score AS score, wp.description AS description,
                   wp.physics_domain AS physics_domain,
                   wp.preview_text AS preview,
                   wp.in_degree AS in_degree, wp.out_degree AS out_degree
            """,
            facility=facility,
            token=claim_token,
        )
        claimed = list(result)

        logger.debug(
            "claim_pages_for_ingesting: requested %d, won %d (token=%s)",
            limit,
            len(claimed),
            claim_token[:8],
        )
        return claimed


# =============================================================================
# Mark Work Complete
# =============================================================================


def mark_pages_scored(
    facility: str,
    results: list[dict[str, Any]],
    skip_threshold: float = 0.5,
) -> int:
    """Mark pages as scored or skipped based on score threshold.

    Pages with score >= skip_threshold get status='scored' (proceed to ingest).
    Pages with score < skip_threshold get status='skipped' (filtered out).

    Uses batched UNWIND for O(1) graph operations instead of O(n) individual queries.
    """
    if not results:
        return 0

    # Prepare batch data with all scoring fields, split by threshold
    scored_batch: list[dict[str, Any]] = []
    skipped_batch: list[dict[str, Any]] = []
    # Safety: collect page IDs with empty preview_text to release for reprocessing
    released_ids: list[str] = []

    for r in results:
        page_id = r.get("id")
        if not page_id:
            continue

        # Pages without preview_text should not be marked as scored —
        # release them back to scanned so they can be retried
        if not r.get("preview_text"):
            released_ids.append(page_id)
            continue

        item = {
            "id": page_id,
            "score": r.get("score", 0.0),
            "purpose": r.get("purpose", r.get("page_purpose", "other")),
            "description": r.get("description", "") or "",
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
            "should_ingest": r.get("should_ingest", True),
            "skip_reason": r.get("skip_reason"),
            "is_physics": r.get("is_physics", False),
            "score_cost": r.get("score_cost", 0.0),
        }

        if item["score"] >= skip_threshold:
            scored_batch.append(item)
        else:
            skipped_batch.append(item)

    if not scored_batch and not skipped_batch and not released_ids:
        return 0

    with GraphClient() as gc:
        # Cypher template shared by scored and skipped batches
        _SET_SCORING_FIELDS = """
                UNWIND $batch AS item
                MATCH (wp:WikiPage {id: item.id})
                SET wp.status = $status,
                    wp.score = item.score,
                    wp.purpose = item.purpose,
                    wp.description = item.description,
                    wp.reasoning = item.reasoning,
                    wp.keywords = item.keywords,
                    wp.physics_domain = item.physics_domain,
                    wp.preview_text = item.preview_text,
                    wp.score_data_documentation = item.score_data_documentation,
                    wp.score_physics_content = item.score_physics_content,
                    wp.score_code_documentation = item.score_code_documentation,
                    wp.score_data_access = item.score_data_access,
                    wp.score_calibration = item.score_calibration,
                    wp.score_imas_relevance = item.score_imas_relevance,
                    wp.should_ingest = item.should_ingest,
                    wp.skip_reason = item.skip_reason,
                    wp.is_physics_content = item.is_physics,
                    wp.score_cost = item.score_cost,
                    wp.scored_at = datetime(),
                    wp.preview_fetched_at = datetime(),
                    wp.claimed_at = null
        """

        if scored_batch:
            gc.query(
                _SET_SCORING_FIELDS,
                batch=scored_batch,
                status=WikiPageStatus.scored.value,
            )

        if skipped_batch:
            gc.query(
                _SET_SCORING_FIELDS,
                batch=skipped_batch,
                status=WikiPageStatus.skipped.value,
            )
            logger.info(
                "mark_pages_scored: %d pages skipped (score < %.1f)",
                len(skipped_batch),
                skip_threshold,
            )

        # Release pages with empty preview_text back to scanned for reprocessing
        if released_ids:
            logger.info(
                "mark_pages_scored: releasing %d pages with empty preview_text "
                "back to scanned for retry",
                len(released_ids),
            )
            gc.query(
                """
                UNWIND $ids AS id
                MATCH (wp:WikiPage {id: id})
                SET wp.claimed_at = null
                """,
                ids=released_ids,
            )

    return len(scored_batch) + len(skipped_batch)


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


def _release_claimed_pages(page_ids: list[str], *, max_fetch_retries: int = 3) -> None:
    """Release claimed pages back for reprocessing.

    Increments fetch_retries counter on each page. Pages that exceed
    max_fetch_retries are marked as 'failed' instead of released,
    preventing infinite retry loops for unfetchable pages.

    If Neo4j is unavailable, logs a warning and silently fails.
    Pages will be reclaimed after CLAIM_TIMEOUT_SECONDS anyway.
    """
    if not page_ids:
        return

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                UNWIND $ids AS id
                MATCH (wp:WikiPage {id: id})
                SET wp.fetch_retries = coalesce(wp.fetch_retries, 0) + 1
                WITH wp
                CALL {
                    WITH wp
                    WITH wp WHERE wp.fetch_retries >= $max_retries
                    SET wp.status = $failed, wp.claimed_at = null,
                        wp.claim_token = null
                    RETURN wp.id AS failed_id
                }
                WITH wp WHERE wp.fetch_retries < $max_retries
                SET wp.claimed_at = null, wp.claim_token = null
                RETURN count(wp) AS released
                """,
                ids=page_ids,
                max_retries=max_fetch_retries,
                failed=WikiPageStatus.failed.value,
            )
            released = result[0]["released"] if result else 0
            failed_count = len(page_ids) - released
            if failed_count > 0:
                logger.warning(
                    "Marked %d pages as failed after %d fetch retries",
                    failed_count,
                    max_fetch_retries,
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


@retry_on_deadlock()
def claim_artifacts_for_scoring(
    facility: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Claim discovered artifacts for scoring.

    Claims artifacts with status='discovered' and any scorable artifact_type.
    Image artifacts are NOT scored here — they bypass LLM scoring and go
    directly to ingestion via claim_artifacts_for_ingesting.

    Workflow: discovered + scorable_type + unclaimed → set claimed_at
    Score worker extracts text preview and scores with LLM.
    After scoring: update status to 'scored' and set score/description/physics_domain.

    Uses claim token pattern to handle race conditions between workers.
    """
    import uuid

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(uuid.uuid4())

    with GraphClient() as gc:
        # Step 1: Attempt to claim artifacts with our unique token
        gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility})
            WHERE wa.status = $discovered
              AND wa.artifact_type IN $types
              AND (wa.claimed_at IS NULL
                   OR wa.claimed_at < datetime() - duration($cutoff))
            WITH wa
            ORDER BY rand()
            LIMIT $limit
            SET wa.claimed_at = datetime(), wa.claim_token = $token
            """,
            facility=facility,
            discovered=WikiArtifactStatus.discovered.value,
            types=list(SCORABLE_ARTIFACT_TYPES),
            cutoff=cutoff,
            limit=limit,
            token=claim_token,
        )

        # Step 2: Read back only artifacts WE successfully claimed
        result = gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility, claim_token: $token})
            RETURN wa.id AS id, wa.url AS url, wa.filename AS filename,
                   wa.artifact_type AS artifact_type, wa.size_bytes AS size_bytes
            """,
            facility=facility,
            token=claim_token,
        )
        claimed = list(result)

        logger.debug(
            "claim_artifacts_for_scoring: requested %d, won %d (token=%s)",
            limit,
            len(claimed),
            claim_token[:8],
        )
        return claimed


@retry_on_deadlock()
def claim_artifacts_for_ingesting(
    facility: str,
    min_score: float = 0.5,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Claim artifacts for ingestion.

    Claims two categories of artifacts:
    1. Scored text-extractable artifacts with score >= min_score
    2. Score-exempt artifacts (e.g. images) directly from discovered status

    Score-exempt artifacts bypass LLM text scoring and go directly to the
    ingestion pipeline, where type-specific handlers (e.g. VLM captioning
    for images) process them.

    Data/archive types are scored but never claimed for ingestion.

    Uses claim token pattern to handle race conditions between workers.
    """
    import uuid

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(uuid.uuid4())

    with GraphClient() as gc:
        # Step 1: Attempt to claim artifacts with our unique token.
        # Score-exempt artifacts: claim from 'discovered' (bypass LLM scoring).
        # Text artifacts: claim from 'scored' with score >= threshold.
        gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility})
            WHERE (
                (wa.status = $scored AND wa.score >= $min_score
                 AND wa.artifact_type IN $types)
                OR (wa.status = $discovered AND wa.score_exempt = true)
              )
              AND (wa.claimed_at IS NULL
                   OR wa.claimed_at < datetime() - duration($cutoff))
            WITH wa
            ORDER BY wa.score DESC, rand()
            LIMIT $limit
            SET wa.claimed_at = datetime(), wa.claim_token = $token
            """,
            facility=facility,
            scored=WikiArtifactStatus.scored.value,
            discovered=WikiArtifactStatus.discovered.value,
            min_score=min_score,
            types=list(INGESTABLE_ARTIFACT_TYPES),
            cutoff=cutoff,
            limit=limit,
            token=claim_token,
        )

        # Step 2: Read back only artifacts WE successfully claimed
        result = gc.query(
            """
            MATCH (wa:WikiArtifact {facility_id: $facility, claim_token: $token})
            RETURN wa.id AS id, wa.url AS url, wa.filename AS filename,
                   wa.artifact_type AS artifact_type, wa.score AS score,
                   wa.description AS description, wa.physics_domain AS physics_domain
            """,
            facility=facility,
            token=claim_token,
        )
        claimed = list(result)

        logger.debug(
            "claim_artifacts_for_ingesting: requested %d, won %d (token=%s)",
            limit,
            len(claimed),
            claim_token[:8],
        )
        return claimed


def mark_artifacts_scored(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark artifacts as scored with scoring data.

    Uses batched UNWIND for O(1) graph operations.
    All artifacts move to 'scored' status. The ingester filters by score >= threshold.
    """
    if not results:
        return 0

    batch_data = []
    for r in results:
        artifact_id = r.get("id")
        if not artifact_id:
            continue

        batch_data.append(
            {
                "id": artifact_id,
                "score": r.get("score", 0.0),
                "artifact_purpose": r.get("artifact_purpose", "other"),
                "description": r.get("description", "") or "",
                "reasoning": r.get("reasoning", ""),
                "keywords": r.get("keywords", []),
                "physics_domain": r.get("physics_domain"),
                "preview_text": r.get("preview_text", "")[:500],
                "score_data_documentation": r.get("score_data_documentation", 0.0),
                "score_physics_content": r.get("score_physics_content", 0.0),
                "score_code_documentation": r.get("score_code_documentation", 0.0),
                "score_data_access": r.get("score_data_access", 0.0),
                "score_calibration": r.get("score_calibration", 0.0),
                "score_imas_relevance": r.get("score_imas_relevance", 0.0),
                "should_ingest": r.get("should_ingest", False),
                "skip_reason": r.get("skip_reason"),
                "score_cost": r.get("score_cost", 0.0),
            }
        )

    if not batch_data:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS item
            MATCH (wa:WikiArtifact {id: item.id})
            SET wa.status = $status,
                wa.score = item.score,
                wa.artifact_purpose = item.artifact_purpose,
                wa.description = item.description,
                wa.reasoning = item.reasoning,
                wa.keywords = item.keywords,
                wa.physics_domain = item.physics_domain,
                wa.preview_text = item.preview_text,
                wa.score_data_documentation = item.score_data_documentation,
                wa.score_physics_content = item.score_physics_content,
                wa.score_code_documentation = item.score_code_documentation,
                wa.score_data_access = item.score_data_access,
                wa.score_calibration = item.score_calibration,
                wa.score_imas_relevance = item.score_imas_relevance,
                wa.should_ingest = item.should_ingest,
                wa.skip_reason = item.skip_reason,
                wa.score_cost = item.score_cost,
                wa.scored_at = datetime(),
                wa.claimed_at = null
            """,
            batch=batch_data,
            status=WikiArtifactStatus.scored.value,
        )

    return len(batch_data)


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
# Image Claim/Mark Functions
# =============================================================================


def has_pending_image_work(facility: str) -> bool:
    """Check if there are images pending scoring (status=ingested, not yet scored)."""
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (img:Image {facility_id: $facility})
                WHERE img.status = 'ingested'
                  AND img.description IS NULL
                RETURN count(img) > 0 AS has_work
                """,
                facility=facility,
            )
            rows = list(result)
            return rows[0]["has_work"] if rows else False
    except Exception:
        return False


@retry_on_deadlock()
def claim_images_for_scoring(
    facility: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Claim ingested images for VLM scoring.

    Images with status='ingested' and no description are ready for VLM processing.
    Uses same claim token pattern as page/artifact claiming.
    """
    import uuid

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(uuid.uuid4())

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (img:Image {facility_id: $facility})
            WHERE img.status = 'ingested'
              AND img.description IS NULL
              AND (img.claimed_at IS NULL
                   OR img.claimed_at < datetime() - duration($cutoff))
            WITH img
            ORDER BY rand()
            LIMIT $limit
            SET img.claimed_at = datetime(), img.claim_token = $token
            """,
            facility=facility,
            cutoff=cutoff,
            limit=limit,
            token=claim_token,
        )

        result = gc.query(
            """
            MATCH (img:Image {facility_id: $facility, claim_token: $token})
            RETURN img.id AS id,
                   img.source_url AS source_url,
                   img.source_type AS source_type,
                   img.image_format AS image_format,
                   img.page_title AS page_title,
                   img.section AS section,
                   img.surrounding_text AS surrounding_text,
                   img.alt_text AS alt_text,
                   img.image_data AS image_data
            """,
            facility=facility,
            token=claim_token,
        )
        claimed = list(result)

        logger.debug(
            "claim_images_for_scoring: requested %d, won %d (token=%s)",
            limit,
            len(claimed),
            claim_token[:8],
        )
        return claimed


@retry_on_deadlock()
def mark_images_scored(
    facility: str,
    results: list[dict[str, Any]],
    *,
    store_images: bool = False,
) -> int:
    """Mark images as scored with VLM results.

    Updates image status to 'captioned' and persists description + scoring fields.
    When store_images is False (default), clears image_data to free graph storage.
    Uses batched UNWIND for efficient graph updates.
    """
    if not results:
        return 0

    clear_data = "" if store_images else ", img.image_data = null"

    with GraphClient() as gc:
        gc.query(
            f"""
            UNWIND $batch AS item
            MATCH (img:Image {{id: item.id}})
            SET img.status = 'captioned',
                img.mermaid_diagram = item.mermaid_diagram,
                img.ocr_text = item.ocr_text,
                img.ocr_mdsplus_paths = item.ocr_mdsplus_paths,
                img.ocr_imas_paths = item.ocr_imas_paths,
                img.ocr_ppf_paths = item.ocr_ppf_paths,
                img.ocr_tool_mentions = item.ocr_tool_mentions,
                img.purpose = item.purpose,
                img.description = item.description,
                img.score = item.score,
                img.score_data_documentation = item.score_data_documentation,
                img.score_physics_content = item.score_physics_content,
                img.score_code_documentation = item.score_code_documentation,
                img.score_data_access = item.score_data_access,
                img.score_calibration = item.score_calibration,
                img.score_imas_relevance = item.score_imas_relevance,
                img.reasoning = item.reasoning,
                img.keywords = item.keywords,
                img.physics_domain = item.physics_domain,
                img.should_ingest = item.should_ingest,
                img.skip_reason = item.skip_reason,
                img.score_cost = item.score_cost,
                img.scored_at = datetime(),
                img.captioned_at = datetime(),
                img.claimed_at = null
                {clear_data}
            """,
            batch=results,
        )

    logger.info(
        "mark_images_scored: updated %d images to captioned for %s",
        len(results),
        facility,
    )
    return len(results)


def _release_claimed_images(image_ids: list[str]) -> None:
    """Release claimed images back to pool (e.g., on VLM failure)."""
    if not image_ids:
        return
    try:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $ids AS id
                MATCH (img:Image {id: id})
                SET img.claimed_at = null
                """,
                ids=image_ids,
            )
    except Exception as e:
        logger.warning("Could not release %d claimed images: %s", len(image_ids), e)
