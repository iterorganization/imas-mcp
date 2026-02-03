"""
Parallel wiki discovery engine with async workers.

Architecture:
- Four independent async workers: Scanner, Prefetcher, Scorer, Ingester
- Graph is the coordination mechanism (no locks needed)
- Atomic status transitions prevent race conditions:
  - pending → scanning → scanned (Scanner worker)
  - scanned → prefetching → prefetched (Prefetcher worker)
  - prefetched → scoring → scored (Scorer worker)
  - scored → ingesting → ingested (Ingester worker)
- Workers continuously poll graph for work
- Cost-based termination for Scorer/Ingester
- Orphan recovery: pages stuck in transient states >10 min are reset

Key insight: The graph acts as a thread-safe work queue. Each worker
claims work by atomically updating status, processes it, then marks complete.
No two workers can claim the same page.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.progress_common import WorkerStats
from imas_codex.graph import GraphClient
from imas_codex.graph.models import WikiPageStatus

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


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

    # Limits
    cost_limit: float = 10.0
    page_limit: int | None = None
    max_depth: int | None = None
    focus: str | None = None

    # Worker stats
    scan_stats: WorkerStats = field(default_factory=WorkerStats)
    prefetch_stats: WorkerStats = field(default_factory=WorkerStats)
    score_stats: WorkerStats = field(default_factory=WorkerStats)
    ingest_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False
    scan_idle_count: int = 0
    prefetch_idle_count: int = 0
    score_idle_count: int = 0
    ingest_idle_count: int = 0

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
            and self.prefetch_idle_count >= 3
            and self.score_idle_count >= 3
            and self.ingest_idle_count >= 3
        )
        if all_idle:
            if has_pending_work(self.facility):
                # Reset idle counts to force workers to re-poll
                self.scan_idle_count = 0
                self.prefetch_idle_count = 0
                self.score_idle_count = 0
                self.ingest_idle_count = 0
                return False
            return True
        return False

    def should_stop_scanning(self) -> bool:
        """Check if scan/prefetch workers should stop.

        Scan workers continue even when budget is exhausted. They only stop
        when explicitly requested or when no pending work remains.
        """
        if self.stop_requested:
            return True
        # Only stop scanning when both are idle with no work
        scan_idle = self.scan_idle_count >= 3 and self.prefetch_idle_count >= 3
        if scan_idle and not has_pending_scan_work(self.facility):
            return True
        return False

    def should_stop_scoring(self) -> bool:
        """Check if score/ingest workers should stop.

        Score workers stop when budget exhausted or page limit reached.
        """
        if self.stop_requested:
            return True
        if self.budget_exhausted:
            return True
        if self.page_limit_reached:
            return True
        return False


def has_pending_work(facility: str) -> bool:
    """Check if there's pending wiki work in the graph."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status IN $transient_states
               OR (wp.status = $scanned AND wp.preview_summary IS NULL)
               OR (wp.status = $prefetched AND wp.interest_score IS NULL)
               OR (wp.status = $scored AND wp.interest_score >= 0.5)
            RETURN count(wp) AS pending
            """,
            facility=facility,
            transient_states=[
                WikiPageStatus.scanning.value,
                WikiPageStatus.prefetching.value,
                WikiPageStatus.scoring.value,
                WikiPageStatus.ingesting.value,
            ],
            scanned=WikiPageStatus.scanned.value,
            prefetched=WikiPageStatus.prefetched.value,
            scored=WikiPageStatus.scored.value,
        )
        return result[0]["pending"] > 0 if result else False


def has_pending_scan_work(facility: str) -> bool:
    """Check if there's pending scan/prefetch work in the graph.

    Only checks for work that scan and prefetch workers handle.
    Does not consider scoring/ingesting work.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $discovered
               OR wp.status = $scanning
               OR wp.status = $scanned
               OR wp.status = $prefetching
            RETURN count(wp) AS pending
            """,
            facility=facility,
            discovered=WikiPageStatus.discovered.value,
            scanning=WikiPageStatus.scanning.value,
            scanned=WikiPageStatus.scanned.value,
            prefetching=WikiPageStatus.prefetching.value,
        )
        return result[0]["pending"] > 0 if result else False


# =============================================================================
# Startup Reset
# =============================================================================


def reset_transient_pages(facility: str, *, silent: bool = False) -> dict[str, int]:
    """Reset ALL wiki pages in transient states on CLI startup.

    Since only one CLI process runs per facility at a time, any pages in
    transient states are orphans from a previous crashed/killed process.

    Transient state fallbacks:
    - scanning → discovered
    - prefetching → scanned
    - scoring → prefetched
    - ingesting → scored
    """
    with GraphClient() as gc:
        # Reset scanning → discovered
        scanning_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $scanning
            SET wp.status = $discovered, wp.claimed_at = null
            RETURN count(wp) AS reset_count
            """,
            facility=facility,
            scanning=WikiPageStatus.scanning.value,
            discovered=WikiPageStatus.discovered.value,
        )
        scanning_reset = scanning_result[0]["reset_count"] if scanning_result else 0

        # Reset prefetching → scanned
        prefetching_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $prefetching
            SET wp.status = $scanned, wp.claimed_at = null
            RETURN count(wp) AS reset_count
            """,
            facility=facility,
            prefetching=WikiPageStatus.prefetching.value,
            scanned=WikiPageStatus.scanned.value,
        )
        prefetching_reset = (
            prefetching_result[0]["reset_count"] if prefetching_result else 0
        )

        # Reset scoring → prefetched
        scoring_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $scoring
            SET wp.status = $prefetched, wp.claimed_at = null
            RETURN count(wp) AS reset_count
            """,
            facility=facility,
            scoring=WikiPageStatus.scoring.value,
            prefetched=WikiPageStatus.prefetched.value,
        )
        scoring_reset = scoring_result[0]["reset_count"] if scoring_result else 0

        # Reset ingesting → scored
        ingesting_result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $ingesting
            SET wp.status = $scored, wp.claimed_at = null
            RETURN count(wp) AS reset_count
            """,
            facility=facility,
            ingesting=WikiPageStatus.ingesting.value,
            scored=WikiPageStatus.scored.value,
        )
        ingesting_reset = ingesting_result[0]["reset_count"] if ingesting_result else 0

        # No legacy migration needed - "discovered" is now the canonical first state
        discovered_reset = 0

    if not silent:
        total_reset = (
            scanning_reset
            + prefetching_reset
            + scoring_reset
            + ingesting_reset
            + discovered_reset
        )
        if total_reset > 0:
            logger.info(
                "Reset wiki pages on startup: "
                f"{scanning_reset} scanning, {prefetching_reset} prefetching, "
                f"{scoring_reset} scoring, {ingesting_reset} ingesting, "
                f"{discovered_reset} legacy discovered"
            )

    return {
        "scanning_reset": scanning_reset,
        "prefetching_reset": prefetching_reset,
        "scoring_reset": scoring_reset,
        "ingesting_reset": ingesting_reset,
        "discovered_reset": discovered_reset,
    }


# =============================================================================
# Graph-based Work Claiming
# =============================================================================


def claim_pages_for_scanning(facility: str, limit: int = 50) -> list[dict[str, Any]]:
    """Atomically claim discovered pages for scanning.

    Uses atomic status transition: discovered → scanning
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $discovered
            WITH wp
            ORDER BY wp.link_depth ASC, wp.discovered_at ASC
            LIMIT $limit
            SET wp.status = $scanning, wp.claimed_at = datetime()
            RETURN wp.id AS id, wp.title AS title, wp.url AS url,
                   wp.link_depth AS depth
            """,
            facility=facility,
            discovered=WikiPageStatus.discovered.value,
            scanning=WikiPageStatus.scanning.value,
            limit=limit,
        )
        return list(result)


def claim_pages_for_prefetching(facility: str, limit: int = 20) -> list[dict[str, Any]]:
    """Atomically claim scanned pages for prefetching.

    Uses atomic status transition: scanned → prefetching
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $scanned
            WITH wp
            ORDER BY wp.in_degree DESC, wp.link_depth ASC
            LIMIT $limit
            SET wp.status = $prefetching, wp.claimed_at = datetime()
            RETURN wp.id AS id, wp.title AS title, wp.url AS url
            """,
            facility=facility,
            scanned=WikiPageStatus.scanned.value,
            prefetching=WikiPageStatus.prefetching.value,
            limit=limit,
        )
        return list(result)


def claim_pages_for_scoring(facility: str, limit: int = 50) -> list[dict[str, Any]]:
    """Atomically claim prefetched pages for LLM scoring.

    Uses atomic status transition: prefetched → scoring
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $prefetched
            WITH wp
            ORDER BY wp.in_degree DESC, wp.link_depth ASC
            LIMIT $limit
            SET wp.status = $scoring, wp.claimed_at = datetime()
            RETURN wp.id AS id, wp.title AS title, wp.url AS url,
                   wp.preview_summary AS summary, wp.in_degree AS in_degree,
                   wp.out_degree AS out_degree, wp.link_depth AS depth
            """,
            facility=facility,
            prefetched=WikiPageStatus.prefetched.value,
            scoring=WikiPageStatus.scoring.value,
            limit=limit,
        )
        return list(result)


def claim_pages_for_ingesting(
    facility: str, min_score: float = 0.5, limit: int = 10
) -> list[dict[str, Any]]:
    """Atomically claim scored high-value pages for ingestion.

    Uses atomic status transition: scored → ingesting
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WHERE wp.status = $scored
              AND wp.interest_score >= $min_score
            WITH wp
            ORDER BY wp.interest_score DESC, wp.in_degree DESC
            LIMIT $limit
            SET wp.status = $ingesting, wp.claimed_at = datetime()
            RETURN wp.id AS id, wp.title AS title, wp.url AS url,
                   wp.interest_score AS score
            """,
            facility=facility,
            scored=WikiPageStatus.scored.value,
            ingesting=WikiPageStatus.ingesting.value,
            min_score=min_score,
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
    """Mark pages as scanned with extracted link data."""
    if not results:
        return 0

    with GraphClient() as gc:
        for r in results:
            page_id = r.get("id")
            if not page_id:
                continue

            gc.query(
                """
                MATCH (wp:WikiPage {id: $id})
                SET wp.status = $scanned,
                    wp.out_degree = $out_degree,
                    wp.scanned_at = datetime(),
                    wp.claimed_at = null
                """,
                id=page_id,
                scanned=WikiPageStatus.scanned.value,
                out_degree=r.get("out_degree", 0),
            )

    return len(results)


def mark_pages_prefetched(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark pages as prefetched with summary data."""
    if not results:
        return 0

    with GraphClient() as gc:
        for r in results:
            page_id = r.get("id")
            if not page_id:
                continue

            gc.query(
                """
                MATCH (wp:WikiPage {id: $id})
                SET wp.status = $prefetched,
                    wp.preview_summary = $summary,
                    wp.preview_fetch_error = $error,
                    wp.prefetched_at = datetime(),
                    wp.claimed_at = null
                """,
                id=page_id,
                prefetched=WikiPageStatus.prefetched.value,
                summary=r.get("summary"),
                error=r.get("error"),
            )

    return len(results)


def mark_pages_scored(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark pages as scored with interest scores."""
    if not results:
        return 0

    with GraphClient() as gc:
        for r in results:
            page_id = r.get("id")
            if not page_id:
                continue

            # Determine final status based on score
            score = r.get("score", 0.5)
            final_status = (
                WikiPageStatus.scored.value
                if score >= 0.3
                else WikiPageStatus.skipped.value
            )

            gc.query(
                """
                MATCH (wp:WikiPage {id: $id})
                SET wp.status = $status,
                    wp.interest_score = $score,
                    wp.score_reasoning = $reasoning,
                    wp.page_type = $page_type,
                    wp.is_physics_content = $is_physics,
                    wp.scored_at = datetime(),
                    wp.claimed_at = null
                """,
                id=page_id,
                status=final_status,
                score=score,
                reasoning=r.get("reasoning", ""),
                page_type=r.get("page_type", "other"),
                is_physics=r.get("is_physics", False),
            )

    return len(results)


def mark_pages_ingested(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark pages as ingested with chunk data."""
    if not results:
        return 0

    with GraphClient() as gc:
        for r in results:
            page_id = r.get("id")
            if not page_id:
                continue

            gc.query(
                """
                MATCH (wp:WikiPage {id: $id})
                SET wp.status = $ingested,
                    wp.chunk_count = $chunks,
                    wp.ingested_at = datetime(),
                    wp.claimed_at = null
                """,
                id=page_id,
                ingested=WikiPageStatus.ingested.value,
                chunks=r.get("chunk_count", 0),
            )

    return len(results)


def mark_page_failed(page_id: str, error: str, fallback_status: str) -> None:
    """Mark a page as failed with error message."""
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


# =============================================================================
# Link Extraction (Scanner Worker Helpers)
# =============================================================================


def extract_links_mediawiki(
    page_url: str, ssh_host: str
) -> tuple[list[str], list[tuple[str, str]]]:
    """Extract links from a MediaWiki page via SSH."""
    cmd = f'''curl -sk "{page_url}" | grep -oP 'href="/wiki/[^"#]+' | sed 's|href="/wiki/||' | sort -u'''

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
            "index.php",
            "skins/",
        )

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            if line.startswith(excluded_prefixes):
                continue
            if "?" in line or "&" in line:
                continue

            decoded = urllib.parse.unquote(line)

            # Classify as page or artifact
            if _is_artifact(decoded):
                artifact_type = _get_artifact_type(decoded)
                artifact_links.append((decoded, artifact_type))
            else:
                page_links.append(decoded)

        return page_links, artifact_links

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
    """
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

        results = []
        for page in pages:
            page_id = page["id"]
            title = page.get("title", "")
            url = page.get("url", "")

            try:
                # Extract links based on site type
                if state.site_type == "twiki":
                    page_links, artifact_links = extract_links_twiki(
                        title, state.base_url, state.ssh_host
                    )
                elif state.ssh_host:
                    page_links, artifact_links = extract_links_mediawiki(
                        url, state.ssh_host
                    )
                else:
                    page_links, artifact_links = [], []

                # Create new pending pages for discovered links
                _create_discovered_pages(
                    state.facility,
                    page_links,
                    page.get("depth", 0) + 1,
                    state.max_depth,
                )

                # Create artifact nodes
                _create_discovered_artifacts(state.facility, artifact_links)

                results.append(
                    {
                        "id": page_id,
                        "out_degree": len(page_links) + len(artifact_links),
                        "page_links": len(page_links),
                        "artifact_links": len(artifact_links),
                    }
                )

            except Exception as e:
                logger.warning("Error scanning %s: %s", page_id, e)
                mark_page_failed(page_id, str(e), WikiPageStatus.discovered.value)

        # Mark pages as scanned
        mark_pages_scanned(state.facility, results)
        state.scan_stats.processed += len(results)

        if on_progress:
            on_progress(
                f"scanned {len(results)} pages", state.scan_stats, results=results
            )


async def prefetch_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Prefetch worker: Fetch content and generate summaries.

    Transitions: scanned → prefetching → prefetched
    """
    while not state.should_stop_scanning():
        pages = claim_pages_for_prefetching(state.facility, limit=20)

        if not pages:
            state.prefetch_idle_count += 1
            if on_progress:
                on_progress("idle", state.prefetch_stats)
            await asyncio.sleep(1.0)
            continue

        state.prefetch_idle_count = 0

        if on_progress:
            on_progress(f"prefetching {len(pages)} pages", state.prefetch_stats)

        results = []
        for page in pages:
            page_id = page["id"]
            url = page.get("url", "")

            try:
                # Fetch content and generate summary
                summary = await _fetch_and_summarize(url, state.ssh_host)
                results.append(
                    {
                        "id": page_id,
                        "summary": summary,
                        "error": None,
                    }
                )
            except Exception as e:
                logger.warning("Error prefetching %s: %s", page_id, e)
                results.append(
                    {
                        "id": page_id,
                        "summary": None,
                        "error": str(e),
                    }
                )

        mark_pages_prefetched(state.facility, results)
        state.prefetch_stats.processed += len(results)

        if on_progress:
            on_progress(
                f"prefetched {len(results)} pages",
                state.prefetch_stats,
                results=results,
            )


async def score_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Scorer worker: LLM scoring of prefetched pages.

    Transitions: prefetched → scoring → scored/skipped
    """
    from imas_codex.agentic.agents import create_litellm_model, get_model_for_task

    while not state.should_stop_scoring():
        pages = claim_pages_for_scoring(state.facility, limit=50)

        if not pages:
            state.score_idle_count += 1
            if on_progress:
                on_progress("idle", state.score_stats)
            await asyncio.sleep(1.0)
            continue

        state.score_idle_count = 0

        if on_progress:
            on_progress(f"scoring {len(pages)} pages", state.score_stats)

        try:
            # Use LLM to score batch
            model = get_model_for_task("discovery")
            llm = create_litellm_model(model=model, temperature=0.3, max_tokens=4096)

            results, cost = await _score_pages_batch(pages, llm, state.focus)

            mark_pages_scored(state.facility, results)
            state.score_stats.processed += len(results)
            state.score_stats.cost += cost  # Actual cost from OpenRouter

            if on_progress:
                on_progress(
                    f"scored {len(results)} pages", state.score_stats, results=results
                )

        except Exception as e:
            logger.error("Error in scoring batch: %s", e)
            # Reset pages to prefetched state
            for page in pages:
                mark_page_failed(page["id"], str(e), WikiPageStatus.prefetched.value)


async def ingest_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
    min_score: float = 0.5,
) -> None:
    """Ingest worker: Chunk and embed high-value pages.

    Transitions: scored → ingesting → ingested
    """
    while not state.should_stop_scoring():
        pages = claim_pages_for_ingesting(state.facility, min_score=min_score, limit=10)

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
                chunk_count = await _ingest_page(url, page_id, state.ssh_host)
                results.append(
                    {
                        "id": page_id,
                        "chunk_count": chunk_count,
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


# =============================================================================
# Worker Helpers
# =============================================================================


def _create_discovered_pages(
    facility: str,
    page_names: list[str],
    depth: int,
    max_depth: int | None = None,
) -> int:
    """Create pending page nodes for newly discovered links."""
    if max_depth is not None and depth > max_depth:
        return 0

    if not page_names:
        return 0

    # Deduplicate and check for existing pages
    from imas_codex.wiki.scraper import canonical_page_id

    created = 0
    with GraphClient() as gc:
        for name in page_names:
            page_id = canonical_page_id(name, facility)
            # MERGE to avoid duplicates
            result = gc.query(
                """
                MERGE (wp:WikiPage {id: $id})
                ON CREATE SET wp.title = $title,
                              wp.facility_id = $facility,
                              wp.status = $discovered,
                              wp.link_depth = $depth,
                              wp.discovered_at = datetime()
                ON MATCH SET wp.link_depth = CASE
                    WHEN wp.link_depth IS NULL OR wp.link_depth > $depth
                    THEN $depth ELSE wp.link_depth END
                RETURN wp.status AS status
                """,
                id=page_id,
                title=name,
                facility=facility,
                discovered=WikiPageStatus.discovered.value,
                depth=depth,
            )
            if result and result[0]["status"] == WikiPageStatus.discovered.value:
                created += 1

    return created


def _create_discovered_artifacts(
    facility: str,
    artifact_links: list[tuple[str, str]],
) -> int:
    """Create pending artifact nodes for newly discovered links."""
    from imas_codex.graph.models import WikiArtifactStatus

    if not artifact_links:
        return 0

    created = 0
    with GraphClient() as gc:
        for path, artifact_type in artifact_links:
            filename = path.split("/")[-1]
            artifact_id = f"{facility}:{filename}"

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
                pending=WikiArtifactStatus.discovered.value,
            )
            created += 1

    return created


async def _fetch_and_summarize(url: str, ssh_host: str | None) -> str:
    """Fetch page content and generate a summary."""
    # Placeholder - actual implementation would fetch HTML and extract text
    # For now, return empty summary (page title serves as minimal context)
    return ""


async def _score_pages_batch(
    pages: list[dict[str, Any]],
    llm: Any,
    focus: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Score a batch of pages using LLM.

    Returns:
        (results, cost) tuple where cost is the actual LLM cost from OpenRouter.
    """
    import json

    import litellm

    # Build scoring prompt
    focus_instruction = f"\nFocus: {focus}" if focus else ""

    system_prompt = f"""You are a wiki page scorer for fusion research documentation.
Score each page from 0.0 to 1.0 based on technical value for plasma physics and IMAS data.
{focus_instruction}

Scoring guidelines:
- 0.8-1.0: Core technical content (data sources, diagnostics, physics codes, IMAS mappings)
- 0.6-0.8: Technical documentation (code docs, analysis methods, experimental procedures)
- 0.4-0.6: Supporting content (tutorials, overviews, general descriptions)
- 0.2-0.4: Administrative/process (meeting notes, project management)
- 0.0-0.2: Low value (personal pages, sandboxes, outdated drafts)

Respond with a JSON array of objects, each with:
- id: the page id
- score: float 0.0-1.0
- reasoning: brief explanation (max 50 chars)
- page_type: one of [data_source, diagnostic, code, documentation, tutorial, administrative, other]
- is_physics: boolean, true if related to plasma physics

Example response:
[{{"id": "jet:POG", "score": 0.85, "reasoning": "Core diagnostic data source", "page_type": "data_source", "is_physics": true}}]"""

    # Format pages for prompt
    page_data = []
    for p in pages:
        page_data.append(
            {
                "id": p["id"],
                "title": p.get("title", ""),
                "summary": p.get("summary", "")[:500] if p.get("summary") else None,
                "in_degree": p.get("in_degree", 0),
                "depth": p.get("depth", 0),
            }
        )

    user_prompt = (
        f"Score these {len(pages)} wiki pages:\n{json.dumps(page_data, indent=2)}"
    )

    # Call LLM
    try:
        response = await litellm.acompletion(
            model=llm.model
            if hasattr(llm, "model")
            else "openrouter/anthropic/claude-sonnet-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
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

        # Parse response
        content = response.choices[0].message.content
        if not content:
            logger.warning("LLM returned empty response, using heuristic fallback")
            return _score_pages_heuristic(pages), cost

        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("\n", 1)[0]
        content = content.strip()

        parsed = json.loads(content)
        results = []
        for r in parsed:
            results.append(
                {
                    "id": r["id"],
                    "score": max(0.0, min(1.0, float(r.get("score", 0.5)))),
                    "reasoning": r.get("reasoning", "")[:80],
                    "page_type": r.get("page_type", "other"),
                    "is_physics": bool(r.get("is_physics", False)),
                }
            )

        return results, cost

    except Exception as e:
        logger.warning("LLM scoring failed, using heuristic fallback: %s", e)
        # Fallback to heuristic scoring (zero cost)
        return _score_pages_heuristic(pages), 0.0


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


async def _ingest_page(url: str, page_id: str, ssh_host: str | None) -> int:
    """Ingest a page: fetch content, chunk, and embed."""
    # Placeholder - actual implementation would use WikiIngestionPipeline
    return 0


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_parallel_wiki_discovery(
    facility: str,
    site_type: str,
    base_url: str,
    portal_page: str,
    ssh_host: str | None = None,
    cost_limit: float = 10.0,
    page_limit: int | None = None,
    max_depth: int | None = None,
    focus: str | None = None,
    num_scan_workers: int = 1,
    num_score_workers: int = 1,
    scan_only: bool = False,
    score_only: bool = False,
    on_scan_progress: Callable | None = None,
    on_prefetch_progress: Callable | None = None,
    on_score_progress: Callable | None = None,
    on_ingest_progress: Callable | None = None,
) -> dict[str, Any]:
    """Run parallel wiki discovery with async workers.

    Returns:
        Dict with discovery statistics
    """
    start_time = time.time()

    # Reset orphans from previous runs
    reset_transient_pages(facility)

    # Initialize state
    state = WikiDiscoveryState(
        facility=facility,
        site_type=site_type,
        base_url=base_url,
        portal_page=portal_page,
        ssh_host=ssh_host,
        cost_limit=cost_limit,
        page_limit=page_limit,
        max_depth=max_depth,
        focus=focus,
    )

    # Create portal page if not exists
    _seed_portal_page(facility, portal_page, base_url, site_type)

    # Start workers
    workers = []

    if not score_only:
        for _ in range(num_scan_workers):
            workers.append(asyncio.create_task(scan_worker(state, on_scan_progress)))
        workers.append(
            asyncio.create_task(prefetch_worker(state, on_prefetch_progress))
        )

    if not scan_only:
        for _ in range(num_score_workers):
            workers.append(asyncio.create_task(score_worker(state, on_score_progress)))
        workers.append(asyncio.create_task(ingest_worker(state, on_ingest_progress)))

    # Wait for termination condition
    while not state.should_stop():
        await asyncio.sleep(0.5)

    # Stop workers
    state.stop_requested = True
    for worker in workers:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    elapsed = time.time() - start_time

    return {
        "scanned": state.scan_stats.processed,
        "prefetched": state.prefetch_stats.processed,
        "scored": state.score_stats.processed,
        "ingested": state.ingest_stats.processed,
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
    from imas_codex.wiki.scraper import canonical_page_id

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
                          wp.status = $discovered,
                          wp.link_depth = 0,
                          wp.discovered_at = datetime()
            """,
            id=page_id,
            title=portal_page,
            url=url,
            facility=facility,
            discovered=WikiPageStatus.discovered.value,
        )


# =============================================================================
# Stats Query
# =============================================================================


def get_wiki_discovery_stats(facility: str) -> dict[str, int]:
    """Get wiki discovery statistics from graph."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility})
            WITH wp.status AS status
            RETURN status, count(*) AS count
            """,
            facility=facility,
        )

        stats = {
            "total": 0,
            "discovered": 0,
            "scanning": 0,
            "scanned": 0,
            "prefetching": 0,
            "prefetched": 0,
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
