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

from imas_codex.discovery.base.progress import WorkerStats
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
    """Mark pages as scored with interest scores and per-dimension data."""
    if not results:
        return 0

    with GraphClient() as gc:
        for r in results:
            page_id = r.get("id")
            if not page_id:
                continue

            # Determine final status based on should_ingest flag
            # If should_ingest=True, mark as scored (to be ingested)
            # Otherwise mark as skipped
            should_ingest = r.get("should_ingest", False)
            score = r.get("score", 0.5)

            # Legacy fallback: also check score threshold
            final_status = (
                WikiPageStatus.scored.value
                if should_ingest or score >= 0.5
                else WikiPageStatus.skipped.value
            )

            gc.query(
                """
                MATCH (wp:WikiPage {id: $id})
                SET wp.status = $status,
                    wp.interest_score = $score,
                    wp.page_purpose = $page_purpose,
                    wp.description = $description,
                    wp.score_reasoning = $reasoning,
                    wp.keywords = $keywords,
                    wp.physics_domain = $physics_domain,
                    wp.should_ingest = $should_ingest,
                    wp.skip_reason = $skip_reason,
                    wp.score_data_documentation = $score_data_documentation,
                    wp.score_physics_content = $score_physics_content,
                    wp.score_code_documentation = $score_code_documentation,
                    wp.score_data_access = $score_data_access,
                    wp.score_calibration = $score_calibration,
                    wp.score_imas_relevance = $score_imas_relevance,
                    wp.page_type = $page_type,
                    wp.is_physics_content = $is_physics,
                    wp.score_cost = $score_cost,
                    wp.scored_at = datetime(),
                    wp.claimed_at = null
                """,
                id=page_id,
                status=final_status,
                score=score,
                page_purpose=r.get("page_purpose", "other"),
                description=r.get("description", "")[:150],
                reasoning=r.get("reasoning", ""),
                keywords=r.get("keywords", []),
                physics_domain=r.get("physics_domain"),
                should_ingest=should_ingest,
                skip_reason=r.get("skip_reason"),
                score_data_documentation=r.get("score_data_documentation", 0.0),
                score_physics_content=r.get("score_physics_content", 0.0),
                score_code_documentation=r.get("score_code_documentation", 0.0),
                score_data_access=r.get("score_data_access", 0.0),
                score_calibration=r.get("score_calibration", 0.0),
                score_imas_relevance=r.get("score_imas_relevance", 0.0),
                page_type=r.get("page_type", "other"),
                is_physics=r.get("is_physics", False),
                score_cost=r.get("score_cost", 0.0),
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

                # Create artifact nodes
                await asyncio.to_thread(
                    _create_discovered_artifacts, state.facility, artifact_links
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
                    mark_page_failed, page_id, str(e), WikiPageStatus.discovered.value
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


async def prefetch_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Prefetch worker: Fetch content and generate summaries.

    Transitions: scanned → prefetching → prefetched

    Uses concurrent SSH calls with bounded parallelism (max 10 concurrent).
    """
    # Semaphore to limit concurrent SSH connections
    ssh_semaphore = asyncio.Semaphore(10)

    async def process_page(page: dict) -> dict:
        """Process a single page with semaphore-bounded concurrency."""
        async with ssh_semaphore:
            page_id = page["id"]
            url = page.get("url", "")

            try:
                # Fetch content and generate summary
                summary = await _fetch_and_summarize(url, state.ssh_host)
                return {
                    "id": page_id,
                    "summary": summary,
                    "error": None,
                }
            except Exception as e:
                logger.warning("Error prefetching %s: %s", page_id, e)
                return {
                    "id": page_id,
                    "summary": None,
                    "error": str(e),
                }

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

        # Process pages concurrently with bounded parallelism
        tasks = [process_page(page) for page in pages]
        results = await asyncio.gather(*tasks)

        logger.debug("Prefetched batch: %d pages", len(results))

        mark_pages_prefetched(state.facility, list(results))
        state.prefetch_stats.processed += len(results)

        if on_progress:
            on_progress(
                f"prefetched {len(results)} pages",
                state.prefetch_stats,
                results=list(results),
            )


async def score_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Scorer worker: LLM scoring of prefetched pages.

    Transitions: prefetched → scoring → scored/skipped

    Uses centralized LLM access via get_model_for_task().
    Cost is tracked from actual OpenRouter response.
    """
    from imas_codex.agentic.agents import get_model_for_task

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
            # Get model from centralized config
            model = get_model_for_task("discovery")

            # Score batch with actual LLM call and cost tracking
            results, cost = await _score_pages_batch(pages, model, state.focus)

            mark_pages_scored(state.facility, results)
            state.score_stats.processed += len(results)
            state.score_stats.cost += cost  # Actual cost from OpenRouter

            if on_progress:
                on_progress(
                    f"scored {len(results)} pages", state.score_stats, results=results
                )

        except ValueError as e:
            # API key missing - log once and stop scoring
            logger.error("LLM configuration error: %s", e)
            state.stop_requested = True
            break
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

    Uses WikiIngestionPipeline for proper chunking and embedding.
    No LLM calls - all entity extraction uses regex patterns.
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
                chunk_count = await _ingest_page(
                    url=url,
                    page_id=page_id,
                    facility=state.facility,
                    site_type=state.site_type,
                    ssh_host=state.ssh_host,
                )
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
                url=url,
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
    """Fetch page content and extract text preview.

    No LLM is used here - prefetch extracts text deterministically.
    The summary is just cleaned text for the scorer to evaluate.

    Args:
        url: Page URL to fetch
        ssh_host: Optional SSH host for proxied fetching

    Returns:
        Extracted text preview (up to 2000 chars) or empty string on error
    """
    from imas_codex.discovery.wiki.prefetch import (
        extract_text_from_html,
        fetch_page_content,
    )

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

    if ssh_host:
        # Fetch via SSH proxy using curl in thread pool
        html = await asyncio.to_thread(_ssh_fetch)
        if html:
            return extract_text_from_html(html, max_chars=2000)
        return ""
    else:
        # Direct HTTP fetch
        html, error = await fetch_page_content(url)
        if html:
            return extract_text_from_html(html, max_chars=2000)
        if error:
            logger.debug("Failed to fetch %s: %s", url, error)
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

    # Retry loop for rate limiting
    max_retries = 3
    retry_base_delay = 2.0
    last_error = None

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
                max_tokens=8192,
            )
            break
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            if any(
                x in error_msg for x in ["overloaded", "rate", "429", "503", "timeout"]
            ):
                delay = retry_base_delay * (2**attempt)
                logger.debug(
                    "LLM rate limited (attempt %d/%d), waiting %.1fs...",
                    attempt + 1,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                raise
    else:
        raise last_error  # type: ignore[misc]

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

    # Parse response using Pydantic structured output
    content = response.choices[0].message.content
    if not content:
        logger.warning("LLM returned empty response, returning empty results")
        return [], cost

    # Sanitize: remove control characters (except newline/tab)
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
    content = content.encode("utf-8", errors="surrogateescape").decode(
        "utf-8", errors="replace"
    )

    try:
        batch = WikiScoreBatch.model_validate_json(content)
        llm_results = batch.results
    except Exception as e:
        logger.error(
            "LLM validation error for batch of %d pages: %s. "
            "Pages will be reverted to prefetched status.",
            len(pages),
            e,
        )
        raise ValueError(f"LLM response validation failed: {e}") from e

    # Convert to result dicts, computing combined scores
    cost_per_page = cost / len(pages) if pages else 0.0
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
) -> int:
    """Ingest a page: fetch content, chunk, and embed.

    Uses the WikiIngestionPipeline for proper chunking and embedding.

    Args:
        url: Page URL to fetch
        page_id: Unique page identifier
        facility: Facility ID (e.g., 'tcv', 'jet')
        site_type: Site type ('mediawiki', 'confluence', 'twiki')
        ssh_host: Optional SSH host for proxied fetching

    Returns:
        Number of chunks created
    """
    from imas_codex.discovery.wiki.pipeline import WikiIngestionPipeline
    from imas_codex.discovery.wiki.scraper import WikiPage

    # Extract page name from URL or page_id
    page_name = page_id.split(":", 1)[1] if ":" in page_id else page_id

    # Fetch HTML content
    html = await _fetch_html(url, ssh_host)
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


async def _fetch_html(url: str, ssh_host: str | None) -> str:
    """Fetch HTML content from URL.

    Args:
        url: Page URL
        ssh_host: Optional SSH host for proxied fetching

    Returns:
        HTML content string or empty string on error
    """
    from imas_codex.discovery.wiki.prefetch import fetch_page_content

    if ssh_host:
        # Fetch via SSH proxy
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
    else:
        # Direct HTTP fetch
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
