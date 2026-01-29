"""
Parallel discovery engine with true concurrent scan and score workers.

Architecture:
- Two independent async workers: Scanner and Scorer
- Graph is the coordination mechanism (no locks needed)
- Atomic status transitions prevent race conditions:
  - discovered → listing → listed (Scanner worker)
  - listed → scoring → scored (Scorer worker)
- Workers continuously poll graph for work
- Cost-based termination for Scorer
- Orphan recovery: paths stuck in transient states >10 min are reset

Key insight: The graph acts as a thread-safe work queue. Each worker
claims work by atomically updating status, processes it, then marks complete.
No two workers can claim the same path.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from imas_codex.graph.models import PathStatus, TerminalReason

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Orphan recovery timeout (10 minutes)
ORPHAN_TIMEOUT_MINUTES = 10


@dataclass
class WorkerStats:
    """Statistics for a single worker."""

    processed: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    last_batch_time: float = 0.0
    cost: float = 0.0  # For scorer only

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def rate(self) -> float | None:
        if self.processed == 0 or self.elapsed <= 0:
            return None
        return self.processed / self.elapsed


@dataclass
class DiscoveryState:
    """Shared state for parallel discovery."""

    facility: str
    cost_limit: float
    path_limit: int | None = None
    focus: str | None = None
    threshold: float = 0.7

    # Worker stats
    scan_stats: WorkerStats = field(default_factory=WorkerStats)
    expand_stats: WorkerStats = field(default_factory=WorkerStats)
    score_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    rescore_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False
    scan_idle_count: int = 0
    expand_idle_count: int = 0
    score_idle_count: int = 0
    enrich_idle_count: int = 0
    rescore_idle_count: int = 0

    # SSH retry tracking for exponential backoff
    ssh_retry_count: int = 0
    max_ssh_retries: int = 5
    ssh_error_message: str | None = None

    # Session tracking for --limit
    initial_terminal_count: int | None = None

    @property
    def total_cost(self) -> float:
        return self.score_stats.cost + self.rescore_stats.cost

    @property
    def total_processed(self) -> int:
        return self.scan_stats.processed + self.score_stats.processed

    @property
    def terminal_count(self) -> int:
        """Count of paths in terminal states (scored, not pending expand/enrich).

        For --limit purposes, we only count paths that have completed their
        pipeline: scored and not awaiting expansion or enrichment.
        """
        from imas_codex.graph import GraphClient

        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
                WHERE p.status = $scored
                  AND (p.should_expand = false OR p.expanded_at IS NOT NULL)
                  AND (p.should_enrich = false OR p.is_enriched = true)
                RETURN count(p) AS terminal_count
                """,
                facility=self.facility,
                scored=PathStatus.scored.value,
            )
            return result[0]["terminal_count"] if result else 0

    @property
    def session_terminal_count(self) -> int:
        """Count of terminal paths created in this session."""
        if self.initial_terminal_count is None:
            return 0
        return max(0, self.terminal_count - self.initial_terminal_count)

    @property
    def budget_exhausted(self) -> bool:
        return self.total_cost >= self.cost_limit

    @property
    def path_limit_reached(self) -> bool:
        """Check if path limit reached using session terminal count.

        Uses paths completed in THIS SESSION, not cumulative graph total.
        E.g., with 28 existing paths and --limit 30, we process 30 more.
        """
        if self.path_limit is None:
            return False
        return self.session_terminal_count >= self.path_limit

    def should_stop(self) -> bool:
        """Check if discovery should terminate."""
        if self.stop_requested:
            return True
        if self.budget_exhausted:
            return True
        if self.path_limit_reached:
            return True
        # Stop if all workers idle for 3+ iterations AND no pending work
        all_idle = (
            self.scan_idle_count >= 3
            and self.expand_idle_count >= 3
            and self.score_idle_count >= 3
            and self.enrich_idle_count >= 3
            and self.rescore_idle_count >= 3
        )
        if all_idle:
            # Check for pending expansion work before terminating
            if has_pending_work(self.facility):
                # Reset idle counts to force workers to re-poll for new work
                self.scan_idle_count = 0
                self.expand_idle_count = 0
                self.score_idle_count = 0
                self.enrich_idle_count = 0
                self.rescore_idle_count = 0
                return False
            return True
        return False


def has_pending_work(facility: str) -> bool:
    """Check if there's pending work in the graph.

    Returns True if any of:
    - Discovered paths awaiting first scan
    - Paths currently being scanned (listing) or scored (scoring)
    - Listed paths awaiting scoring
    - Scored paths with should_expand=true that haven't been expanded yet
    - Scored paths with should_enrich=true that haven't been enriched yet
    - Enriched paths that haven't been rescored yet
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.status = $discovered
               OR p.status = $listing
               OR p.status = $scoring
               OR (p.status = $listed AND p.score IS NULL)
               OR (p.status = $scored AND p.should_expand = true
                   AND p.expanded_at IS NULL)
               OR (p.status = $scored AND p.should_enrich = true
                   AND (p.is_enriched IS NULL OR p.is_enriched = false))
               OR (p.is_enriched = true AND p.rescored_at IS NULL)
            RETURN count(p) AS pending
            """,
            facility=facility,
            discovered=PathStatus.discovered.value,
            listing=PathStatus.listing.value,
            scoring=PathStatus.scoring.value,
            listed=PathStatus.listed.value,
            scored=PathStatus.scored.value,
        )
        return result[0]["pending"] > 0 if result else False


# ============================================================================
# Startup Reset (single CLI process per facility)
# ============================================================================


def reset_transient_paths(facility: str, *, silent: bool = False) -> dict[str, int]:
    """Reset ALL paths in transient states (listing, scoring) on CLI startup.

    Since only one CLI process runs per facility at a time, any paths in
    transient states are orphans from a previous crashed/killed process.
    Reset them immediately without waiting for timeout.

    For listing paths:
    - If score IS NULL (first scan): reset to 'discovered'
    - If score IS NOT NULL (expansion): reset to 'scored'

    For scoring paths:
    - Reset to 'listed'

    Args:
        facility: Facility identifier
        silent: If True, suppress logging (caller will log)

    Returns:
        Dict with counts: listing_reset, scoring_reset
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        # Reset listing paths: first-scan → discovered, expansion → scored
        listing_result = gc.query(
            """
            MATCH (p:FacilityPath {facility_id: $facility})
            WHERE p.status = $listing
            WITH p, CASE WHEN p.score IS NULL THEN $discovered ELSE $scored END AS new_status
            SET p.status = new_status, p.claimed_at = null
            RETURN count(p) AS reset_count
            """,
            facility=facility,
            listing=PathStatus.listing.value,
            discovered=PathStatus.discovered.value,
            scored=PathStatus.scored.value,
        )
        listing_reset = listing_result[0]["reset_count"] if listing_result else 0

        # Reset scoring paths → listed
        scoring_result = gc.query(
            """
            MATCH (p:FacilityPath {facility_id: $facility})
            WHERE p.status = $scoring
            SET p.status = $listed, p.claimed_at = null
            RETURN count(p) AS reset_count
            """,
            facility=facility,
            scoring=PathStatus.scoring.value,
            listed=PathStatus.listed.value,
        )
        scoring_reset = scoring_result[0]["reset_count"] if scoring_result else 0

    if (listing_reset or scoring_reset) and not silent:
        logger.info(
            f"Reset transient paths on startup: {listing_reset} listing, "
            f"{scoring_reset} scoring"
        )

    return {
        "listing_reset": listing_reset,
        "scoring_reset": scoring_reset,
    }


# ============================================================================
# Graph-based work claiming (atomic status transitions)
# ============================================================================


def claim_paths_for_scanning(facility: str, limit: int = 50) -> list[dict[str, Any]]:
    """Atomically claim discovered paths for initial scanning.

    Claims only unscored discovered paths (first scan, enumerate only).
    Expansion is handled by separate expand_worker.

    Uses atomic status transition: discovered → listing
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        # Claim unscored discovered paths (breadth-first by depth)
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.status = $discovered AND p.score IS NULL
            WITH p ORDER BY p.depth ASC, p.path ASC LIMIT $limit
            SET p.status = $listing, p.claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.depth AS depth, false AS is_expanding
            """,
            facility=facility,
            limit=limit,
            discovered=PathStatus.discovered.value,
            listing=PathStatus.listing.value,
        )
        return list(result)


def claim_paths_for_expanding(facility: str, limit: int = 50) -> list[dict[str, Any]]:
    """Atomically claim scored paths for expansion scanning.

    Claims paths with should_expand=true that haven't been expanded yet.
    These are scored high-value directories that need child enumeration.

    Uses atomic status transition: scored → listing
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        # Claim expansion paths (score-descending for valuable first)
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.status = $scored
              AND p.should_expand = true
              AND p.expanded_at IS NULL
            WITH p ORDER BY p.score DESC, p.depth ASC LIMIT $limit
            SET p.status = $listing, p.claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.depth AS depth, true AS is_expanding
            """,
            facility=facility,
            limit=limit,
            scored=PathStatus.scored.value,
            listing=PathStatus.listing.value,
        )
        return list(result)


def claim_paths_for_scoring(facility: str, limit: int = 25) -> list[dict[str, Any]]:
    """Atomically claim listed paths for scoring.

    Uses atomic status transition: listed → scoring
    Returns paths that this worker now owns.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.status = $listed AND p.score IS NULL
            WITH p ORDER BY p.depth ASC, p.path ASC LIMIT $limit
            SET p.status = $scoring, p.claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.depth AS depth,
                   p.total_files AS total_files, p.total_dirs AS total_dirs,
                   p.file_type_counts AS file_type_counts,
                   p.has_readme AS has_readme, p.has_makefile AS has_makefile,
                   p.has_git AS has_git, p.patterns_detected AS patterns_detected,
                   p.child_names AS child_names
            """,
            facility=facility,
            limit=limit,
            listed=PathStatus.listed.value,
            scoring=PathStatus.scoring.value,
        )
        return list(result)


async def mark_scan_complete(
    facility: str,
    scan_results: list[tuple[str, dict, list[dict], str | None, bool]],
    excluded: list[tuple[str, str, str]] | None = None,
) -> dict[str, int]:
    """Mark scanned paths complete and conditionally create children.

    Transition: listing → listed (first scan) or scored (expansion scan)

    Args:
        facility: Facility ID
        scan_results: List of (path, stats_dict, child_dirs, error, is_expanding) tuples.
                      child_dirs is list of {path, is_symlink, realpath, device_inode} dicts.
        excluded: Optional list of (path, parent_path, reason) for excluded dirs
    """
    from imas_codex.discovery.frontier import persist_scan_results

    return await persist_scan_results(facility, scan_results, excluded=excluded)


def mark_score_complete(
    facility: str,
    score_data: list[dict[str, Any]],
) -> int:
    """Mark scored paths complete.

    Transition: scoring → scored
    """
    from imas_codex.discovery.frontier import mark_paths_scored

    return mark_paths_scored(facility, score_data)


def claim_paths_for_enriching(facility: str, limit: int = 25) -> list[dict[str, Any]]:
    """Atomically claim scored paths for enrichment.

    Claims paths where:
    - status = 'scored'
    - should_enrich = true
    - is_enriched is null or false

    Uses claimed_at timestamp for orphan recovery.
    Returns paths that this worker now owns.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.status = $scored
              AND p.should_enrich = true
              AND (p.is_enriched IS NULL OR p.is_enriched = false)
            WITH p ORDER BY p.score DESC, p.depth ASC LIMIT $limit
            SET p.claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.depth AS depth, p.score AS score
            """,
            facility=facility,
            limit=limit,
            scored=PathStatus.scored.value,
        )
        return list(result)


def claim_paths_for_rescoring(facility: str, limit: int = 10) -> list[dict[str, Any]]:
    """Atomically claim enriched paths for rescoring.

    Claims paths where:
    - is_enriched = true
    - rescored_at is null

    Rescoring uses enrichment data (total_bytes, total_lines, language_breakdown)
    to refine the score.

    Returns paths that this worker now owns.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.is_enriched = true
              AND p.rescored_at IS NULL
            WITH p ORDER BY p.score DESC LIMIT $limit
            SET p.claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.depth AS depth,
                   p.score AS score, p.score_code AS score_code,
                   p.score_data AS score_data, p.score_imas AS score_imas,
                   p.total_bytes AS total_bytes, p.total_lines AS total_lines,
                   p.language_breakdown AS language_breakdown,
                   p.is_multiformat AS is_multiformat,
                   p.path_purpose AS path_purpose
            """,
            facility=facility,
            limit=limit,
        )
        return list(result)


def mark_enrichment_complete(
    facility: str,
    enrichment_results: list[dict[str, Any]],
) -> int:
    """Mark paths as enriched with deep scan data.

    Args:
        facility: Facility ID
        enrichment_results: List of dicts with path and enrichment data

    Returns:
        Number of paths updated
    """
    from datetime import UTC, datetime

    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    updated = 0

    with GraphClient() as gc:
        for result in enrichment_results:
            if result.get("error"):
                continue

            path_id = f"{facility}:{result['path']}"
            gc.query(
                """
                MATCH (p:FacilityPath {id: $id})
                SET p.is_enriched = true,
                    p.enriched_at = $now,
                    p.total_bytes = $total_bytes,
                    p.total_lines = $total_lines,
                    p.language_breakdown = $language_breakdown,
                    p.is_multiformat = $is_multiformat,
                    p.claimed_at = null
                """,
                id=path_id,
                now=now,
                total_bytes=result.get("total_bytes"),
                total_lines=result.get("total_lines"),
                language_breakdown=result.get("language_breakdown"),
                is_multiformat=result.get("is_multiformat", False),
            )
            updated += 1

    return updated


def mark_rescore_complete(
    facility: str,
    rescore_results: list[dict[str, Any]],
) -> int:
    """Mark paths with refined scores after rescoring.

    Args:
        facility: Facility ID
        rescore_results: List of dicts with path and new score data

    Returns:
        Number of paths updated
    """
    from datetime import UTC, datetime

    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    updated = 0

    with GraphClient() as gc:
        for result in rescore_results:
            path_id = f"{facility}:{result['path']}"
            gc.query(
                """
                MATCH (p:FacilityPath {id: $id})
                SET p.rescored_at = $now,
                    p.score = $score,
                    p.score_cost = coalesce(p.score_cost, 0) + $score_cost,
                    p.claimed_at = null
                """,
                id=path_id,
                now=now,
                score=result.get("score"),
                score_cost=result.get("score_cost", 0.0),
            )
            updated += 1

    return updated


# ============================================================================
# Async Workers
# ============================================================================


async def scan_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[str] | None, list[dict] | None], None]
    | None = None,
    batch_size: int = 50,
) -> None:
    """Async scanner worker.

    Continuously claims pending paths, scans via SSH, marks complete.
    Runs until stop_requested or no more pending paths.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per SSH call (default 50)
    """
    from imas_codex.discovery.scanner import scan_paths

    while not state.should_stop():
        # Claim work from graph
        paths = claim_paths_for_scanning(state.facility, limit=batch_size)

        if not paths:
            state.scan_idle_count += 1
            if on_progress:
                on_progress("idle", state.scan_stats, None, None)
            # Wait before polling again
            await asyncio.sleep(1.0)
            continue

        state.scan_idle_count = 0
        path_strs = [p["path"] for p in paths]

        if on_progress:
            on_progress(f"scanning {len(paths)} paths", state.scan_stats, None, None)

        # Run scan in thread pool (blocking SSH call)
        # Capture variables for lambda to avoid late binding issue
        loop = asyncio.get_running_loop()
        start = time.time()
        facility, paths_to_scan = state.facility, path_strs
        # Use enable_rg=False and enable_size=False for speed
        # Pattern detection is expensive and not needed for basic discovery
        try:
            results = await loop.run_in_executor(
                None,
                lambda fac=facility, pts=paths_to_scan: scan_paths(
                    fac, pts, enable_rg=False, enable_size=False
                ),
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            # Transient failure (timeout or SSH connection error)
            # Revert paths to 'discovered' for retry
            state.ssh_retry_count += 1
            state.ssh_error_message = str(e)[:100]

            # Exponential backoff: 2s, 4s, 8s, 16s, 32s
            backoff_seconds = min(2**state.ssh_retry_count, 32)

            logger.warning(
                f"SSH failure {state.ssh_retry_count}/{state.max_ssh_retries}, "
                f"retry in {backoff_seconds}s: {e}"
            )
            _revert_listing_claim(state.facility, path_strs)
            state.scan_stats.errors += len(paths)

            if state.ssh_retry_count >= state.max_ssh_retries:
                logger.error(
                    f"SSH connection to {state.facility} failed after "
                    f"{state.max_ssh_retries} attempts. Check VPN and SSH config."
                )
                state.stop_requested = True
                if on_progress:
                    on_progress(
                        f"SSH failed: {state.ssh_error_message}",
                        state.scan_stats,
                        None,
                        None,
                    )
                break

            if on_progress:
                on_progress(
                    f"SSH retry {state.ssh_retry_count} in {backoff_seconds}s",
                    state.scan_stats,
                    None,
                    None,
                )
            await asyncio.sleep(backoff_seconds)
            continue

        # SSH succeeded - reset retry counter
        state.ssh_retry_count = 0
        state.ssh_error_message = None
        state.scan_stats.last_batch_time = time.time() - start

        # Persist results (marks scanning → scanned)
        # Run in executor to avoid blocking event loop
        # is_expanding=False since scan_worker only handles initial scans
        # Convert ChildDirInfo objects to dicts for serialization
        batch_data = [
            (
                r.path,
                r.stats.to_dict(),
                [
                    {
                        "path": c.path,
                        "is_symlink": c.is_symlink,
                        "realpath": c.realpath,
                        "device_inode": c.device_inode,
                    }
                    for c in r.child_dirs
                ],
                r.error,
                False,  # Not expanding - that's handled by expand_worker
            )
            for r in results
        ]

        # Collect excluded directories with parent paths and reasons
        excluded_data = []
        for r in results:
            for excluded_path, reason in r.excluded_dirs:
                excluded_data.append((excluded_path, r.path, reason))

        stats = await mark_scan_complete(
            state.facility,
            batch_data,
            excluded=excluded_data if excluded_data else None,
        )

        state.scan_stats.processed += stats["scanned"]
        state.scan_stats.errors += stats["errors"]

        # Build detailed scan results for progress display
        scan_results = [
            {
                "path": r.path,
                "total_files": r.stats.total_files,
                "total_dirs": r.stats.total_dirs,
                "has_readme": r.stats.has_readme,
                "has_makefile": r.stats.has_makefile,
                "has_git": r.stats.has_git,
                "file_types": {},  # Could be extracted if available
                "error": r.error,
            }
            for r in results
            if not r.error
        ]

        # Pass detailed scan results to progress callback
        scanned_paths = [r.path for r in results if not r.error]
        if on_progress:
            on_progress(
                f"scanned {stats['scanned']}",
                state.scan_stats,
                scanned_paths,
                scan_results,
            )

        # Brief yield to allow score worker to run
        await asyncio.sleep(0.1)


async def expand_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[str] | None, list[dict] | None], None]
    | None = None,
    batch_size: int = 50,
) -> None:
    """Async expansion worker.

    Expands scored high-value paths by enumerating their children.
    Runs independently of scan_worker, claiming paths with should_expand=true.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per SSH call (default 50)
    """
    from imas_codex.discovery.scanner import scan_paths

    while not state.should_stop():
        # Claim expansion work from graph - paths with should_expand=true
        paths = claim_paths_for_expanding(state.facility, limit=batch_size)

        if not paths:
            state.expand_idle_count += 1
            if on_progress:
                on_progress("idle", state.expand_stats, None, None)
            # Wait before polling again
            await asyncio.sleep(1.0)
            continue

        state.expand_idle_count = 0
        path_strs = [p["path"] for p in paths]

        if on_progress:
            on_progress(f"expanding {len(paths)} paths", state.expand_stats, None, None)

        # Run scan in thread pool (blocking SSH call)
        loop = asyncio.get_running_loop()
        start = time.time()
        facility, paths_to_scan = state.facility, path_strs

        try:
            results = await loop.run_in_executor(
                None,
                lambda fac=facility, pts=paths_to_scan: scan_paths(
                    fac, pts, enable_rg=False, enable_size=False
                ),
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            # Transient failure - revert paths for retry
            logger.warning(f"Expand SSH failure: {e}")
            _revert_listing_claim(state.facility, path_strs)
            state.expand_stats.errors += len(paths)
            await asyncio.sleep(5.0)
            continue

        state.expand_stats.last_batch_time = time.time() - start

        # Persist results with is_expanding=True to create child paths
        # Convert ChildDirInfo objects to dicts for serialization
        batch_data = [
            (
                r.path,
                r.stats.to_dict(),
                [
                    {
                        "path": c.path,
                        "is_symlink": c.is_symlink,
                        "realpath": c.realpath,
                        "device_inode": c.device_inode,
                    }
                    for c in r.child_dirs
                ],
                r.error,
                True,  # is_expanding - creates child paths
            )
            for r in results
        ]

        # Collect excluded directories
        excluded_data = []
        for r in results:
            for excluded_path, reason in r.excluded_dirs:
                excluded_data.append((excluded_path, r.path, reason))

        stats = await mark_scan_complete(
            state.facility,
            batch_data,
            excluded=excluded_data if excluded_data else None,
        )

        state.expand_stats.processed += stats["scanned"]
        state.expand_stats.errors += stats["errors"]

        # Build detailed scan results for progress display
        scan_results = [
            {
                "path": r.path,
                "total_files": r.stats.total_files,
                "total_dirs": r.stats.total_dirs,
                "has_readme": r.stats.has_readme,
                "has_makefile": r.stats.has_makefile,
                "has_git": r.stats.has_git,
                "error": r.error,
            }
            for r in results
            if not r.error
        ]

        expanded_paths = [r.path for r in results if not r.error]
        if on_progress:
            on_progress(
                f"expanded {stats['scanned']}",
                state.expand_stats,
                expanded_paths,
                scan_results,
            )

        await asyncio.sleep(0.1)


async def score_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[dict] | None], None] | None = None,
    batch_size: int = 25,
) -> None:
    """Async scorer worker.

    Continuously claims scanned paths, scores via LLM, marks complete.
    Runs until stop_requested, budget exhausted, or no more scanned paths.

    Optimization: Empty directories (total_files=0 AND total_dirs=0) are
    auto-skipped without LLM call since they have no content to evaluate.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per LLM call (default 25)
    """
    from imas_codex.discovery.scorer import DirectoryScorer

    scorer = DirectoryScorer(facility=state.facility)
    loop = asyncio.get_running_loop()

    while not state.should_stop():
        # Check budget before claiming work
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.score_stats, None)
            break

        # Claim work from graph
        paths = claim_paths_for_scoring(state.facility, limit=batch_size)

        if not paths:
            state.score_idle_count += 1
            if on_progress:
                on_progress("waiting for scanned paths", state.score_stats, None)
            # Wait before polling again
            await asyncio.sleep(2.0)
            continue

        state.score_idle_count = 0

        # Split paths into empty (auto-skip) and non-empty (need LLM)
        empty_paths = []
        paths_to_score = []
        for p in paths:
            total_files = p.get("total_files", 0) or 0
            total_dirs = p.get("total_dirs", 0) or 0
            if total_files == 0 and total_dirs == 0:
                empty_paths.append(p)
            else:
                paths_to_score.append(p)

        # Auto-skip empty directories with 0.0 score
        skipped_results = []
        if empty_paths:
            skip_data = [
                {
                    "path": p["path"],
                    "score": 0.0,
                    "path_purpose": "empty_directory",
                    "description": "Empty directory - no files or subdirectories",
                    "score_code": 0.0,
                    "score_data": 0.0,
                    "score_imas": 0.0,
                    "should_expand": False,
                    "skip_reason": "empty",
                    "terminal_reason": TerminalReason.empty.value,
                }
                for p in empty_paths
            ]
            await loop.run_in_executor(
                None, lambda sd=skip_data: mark_score_complete(state.facility, sd)
            )
            state.score_stats.processed += len(empty_paths)

            skipped_results = [
                {
                    "path": p["path"],
                    "score": 0.0,
                    "label": "empty_directory",
                    "path_purpose": "empty_directory",
                    "description": "Empty directory - no files or subdirectories",
                    "score_code": 0.0,
                    "score_data": 0.0,
                    "score_imas": 0.0,
                    "skip_reason": "empty",
                    "should_expand": False,
                    "total_files": 0,
                }
                for p in empty_paths
            ]

            if on_progress and not paths_to_score:
                # Only skipped paths, show progress
                on_progress(
                    f"skipped {len(empty_paths)} empty",
                    state.score_stats,
                    skipped_results,
                )
                continue

        if not paths_to_score:
            continue

        if on_progress:
            on_progress(f"scoring {len(paths_to_score)} paths", state.score_stats, None)

        # Run scoring in thread pool (blocking LLM call)
        loop = asyncio.get_running_loop()
        start = time.time()
        dirs_to_score, focus_val, thresh_val = (
            paths_to_score,
            state.focus,
            state.threshold,
        )
        try:
            result = await loop.run_in_executor(
                None,
                lambda d=dirs_to_score, f=focus_val, t=thresh_val: scorer.score_batch(
                    directories=d,
                    focus=f,
                    threshold=t,
                ),
            )
            state.score_stats.last_batch_time = time.time() - start
            state.score_stats.cost += result.total_cost

            # Persist results (marks scoring → scored)
            # Run in executor to avoid blocking event loop
            score_data = [d.to_graph_dict() for d in result.scored_dirs]
            await loop.run_in_executor(
                None, lambda sd=score_data: mark_score_complete(state.facility, sd)
            )

            state.score_stats.processed += len(result.scored_dirs)

            # Build detailed score results for progress callback
            detailed_results = [
                {
                    "path": d.path,
                    "score": d.score,
                    "label": d.path_purpose.value if d.path_purpose else "",
                    "path_purpose": d.path_purpose.value if d.path_purpose else "",
                    "description": d.description,
                    "score_code": d.score_code,
                    "score_data": d.score_data,
                    "score_imas": d.score_imas,
                    "skip_reason": d.skip_reason or "",
                    "should_expand": d.should_expand,
                    "total_files": 0,  # Not available at score time
                }
                for d in result.scored_dirs
            ]
            # Combine with any skipped results
            all_results = skipped_results + detailed_results
            if on_progress:
                skipped_msg = (
                    f" (+{len(skipped_results)} skipped)" if skipped_results else ""
                )
                on_progress(
                    f"scored {len(result.scored_dirs)} (${result.total_cost:.3f}){skipped_msg}",
                    state.score_stats,
                    all_results,
                )

        except ValueError:
            # LLM validation error - revert paths to 'listed' status for retry
            # DO NOT increment error count - this will be retried automatically
            logger.warning(
                f"LLM validation error for batch of {len(paths_to_score)} paths. "
                "Reverting to listed status for retry."
            )
            _revert_scoring_claim(state.facility, [p["path"] for p in paths_to_score])
            # Don't show validation errors in progress display
        except Exception as e:
            # Other errors - increment error count and revert
            logger.exception(f"Score error: {e}")
            state.score_stats.errors += len(paths_to_score)
            _revert_scoring_claim(state.facility, [p["path"] for p in paths_to_score])

        # Brief yield
        await asyncio.sleep(0.1)


def _revert_scoring_claim(facility: str, paths: list[str]) -> None:
    """Revert paths from scoring back to listed on error."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.path IN $paths AND p.status = $scoring
            SET p.status = $listed, p.claimed_at = null
            """,
            facility=facility,
            paths=paths,
            scoring=PathStatus.scoring.value,
            listed=PathStatus.listed.value,
        )


def _revert_listing_claim(facility: str, paths: list[str]) -> None:
    """Revert paths from listing back to their pre-claim state on transient error.

    For first-scan paths (score IS NULL): revert to 'discovered'
    For expansion paths (score IS NOT NULL): revert to 'scored'
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        # Revert first-scan paths to discovered
        gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.path IN $paths AND p.status = $listing AND p.score IS NULL
            SET p.status = $discovered, p.claimed_at = null
            """,
            facility=facility,
            paths=paths,
            listing=PathStatus.listing.value,
            discovered=PathStatus.discovered.value,
        )
        # Revert expansion paths to scored
        gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.path IN $paths AND p.status = $listing AND p.score IS NOT NULL
            SET p.status = $scored, p.claimed_at = null
            """,
            facility=facility,
            paths=paths,
            listing=PathStatus.listing.value,
            scored=PathStatus.scored.value,
        )


def _revert_enrich_claim(facility: str, paths: list[str]) -> None:
    """Revert paths from enriching back to unenriched state on error."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.path IN $paths
            SET p.claimed_at = null
            """,
            facility=facility,
            paths=paths,
        )


async def enrich_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[dict] | None], None] | None = None,
    batch_size: int = 25,
) -> None:
    """Async enrichment worker.

    Continuously claims scored paths with should_enrich=true, runs deep
    analysis (dust, tokei, patterns) via SSH, marks complete.
    Runs in PARALLEL with scan/score/expand workers.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per SSH call (default 25)
    """
    from imas_codex.discovery.frontier import (
        claim_paths_for_enriching,
        mark_enrichment_complete,
    )
    from imas_codex.discovery.path_enrichment import enrich_paths

    loop = asyncio.get_running_loop()

    while not state.should_stop():
        # Claim work from graph
        paths = claim_paths_for_enriching(state.facility, limit=batch_size)

        if not paths:
            state.enrich_idle_count += 1
            if on_progress:
                on_progress("waiting for enrichable paths", state.enrich_stats, None)
            # Wait before polling again
            await asyncio.sleep(2.0)
            continue

        state.enrich_idle_count = 0
        path_strs = [p["path"] for p in paths]

        if on_progress:
            on_progress(f"enriching {len(paths)} paths", state.enrich_stats, None)

        # Run enrichment in thread pool (blocking SSH call)
        start = time.time()
        try:
            results = await loop.run_in_executor(
                None,
                lambda fac=state.facility, pts=path_strs: enrich_paths(fac, pts),
            )
            state.enrich_stats.last_batch_time = time.time() - start

            # Convert EnrichmentResult to dict for persistence
            result_dicts = [
                {
                    "path": r.path,
                    "total_bytes": r.total_bytes,
                    "total_lines": r.total_lines,
                    "language_breakdown": r.language_breakdown,
                    "is_multiformat": r.is_multiformat,
                    "error": r.error,
                }
                for r in results
            ]

            # Persist results
            enriched = await loop.run_in_executor(
                None,
                lambda rd=result_dicts: mark_enrichment_complete(state.facility, rd),
            )

            state.enrich_stats.processed += enriched
            state.enrich_stats.errors += len([r for r in results if r.error])

            if on_progress:
                on_progress(f"enriched {enriched}", state.enrich_stats, result_dicts)

        except Exception as e:
            logger.exception(f"Enrich error: {e}")
            state.enrich_stats.errors += len(paths)
            _revert_enrich_claim(state.facility, path_strs)

        # Brief yield
        await asyncio.sleep(0.1)


async def rescore_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[dict] | None], None] | None = None,
    batch_size: int = 10,
) -> None:
    """Async rescore worker.

    Continuously claims enriched paths, rescores using enrichment data
    (total_bytes, total_lines, language_breakdown) for refined scoring.
    Cheaper LLM call since we use a simpler prompt with concrete metrics.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per rescore operation (default 10)
    """
    from imas_codex.discovery.frontier import (
        claim_paths_for_rescoring,
        mark_rescore_complete,
    )

    loop = asyncio.get_running_loop()

    while not state.should_stop():
        # Check budget before claiming work
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.rescore_stats, None)
            break

        # Claim work from graph
        paths = claim_paths_for_rescoring(state.facility, limit=batch_size)

        if not paths:
            state.rescore_idle_count += 1
            if on_progress:
                on_progress("waiting for enriched paths", state.rescore_stats, None)
            # Wait before polling again
            await asyncio.sleep(2.0)
            continue

        state.rescore_idle_count = 0

        if on_progress:
            on_progress(f"rescoring {len(paths)} paths", state.rescore_stats, None)

        # Run rescoring in thread pool
        start = time.time()
        try:
            # Simple rescoring: adjust score based on enrichment metrics
            # This is a lightweight heuristic, not a full LLM call
            rescore_results = []
            for p in paths:
                new_score = _compute_rescore(p)
                rescore_results.append(
                    {
                        "path": p["path"],
                        "score": new_score,
                        "score_cost": 0.0,  # Heuristic rescore has no LLM cost
                        "should_expand": p.get("should_expand", True),
                    }
                )

            state.rescore_stats.last_batch_time = time.time() - start

            # Persist results
            rescored = await loop.run_in_executor(
                None,
                lambda rr=rescore_results: mark_rescore_complete(state.facility, rr),
            )

            state.rescore_stats.processed += rescored

            if on_progress:
                on_progress(
                    f"rescored {rescored}", state.rescore_stats, rescore_results
                )

        except Exception as e:
            logger.exception(f"Rescore error: {e}")
            state.rescore_stats.errors += len(paths)

        # Brief yield
        await asyncio.sleep(0.1)


def _compute_rescore(path_data: dict) -> float:
    """Compute refined score using enrichment data.

    Applies heuristic adjustments based on:
    - total_lines: More lines of code = higher score
    - is_multiformat: Format conversion code = higher IMAS value
    - language_breakdown: Python/Fortran = higher value
    """
    import json

    base_score = path_data.get("score", 0.5)

    # Boost for lines of code (max +0.15)
    total_lines = path_data.get("total_lines", 0) or 0
    if total_lines > 10000:
        base_score += 0.15
    elif total_lines > 1000:
        base_score += 0.10
    elif total_lines > 100:
        base_score += 0.05

    # Boost for multiformat (indicates data conversion/mapping)
    if path_data.get("is_multiformat"):
        base_score += 0.10

    # Boost for Python/Fortran presence
    lang_breakdown = path_data.get("language_breakdown")
    if lang_breakdown:
        if isinstance(lang_breakdown, str):
            try:
                lang_breakdown = json.loads(lang_breakdown)
            except json.JSONDecodeError:
                lang_breakdown = {}
        if lang_breakdown:
            if "Python" in lang_breakdown or "Fortran" in lang_breakdown:
                base_score += 0.05

    return min(1.5, max(0.0, base_score))


# ============================================================================
# SSH Preflight Check
# ============================================================================


def check_ssh_connectivity(facility: str, timeout: int = 10) -> tuple[bool, str]:
    """Check if SSH connection to facility is working.

    Args:
        facility: Facility ID
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, message)
    """
    from imas_codex.discovery.facility import get_facility
    from imas_codex.remote.tools import run

    try:
        config = get_facility(facility)
        ssh_host = config.get("ssh_host", facility)
    except ValueError:
        ssh_host = facility

    try:
        result = run("echo ok", facility=ssh_host, timeout=timeout)
        if "ok" in result:
            return True, f"SSH to {ssh_host} working"
        return False, f"SSH to {ssh_host} returned unexpected output"
    except subprocess.TimeoutExpired:
        return False, f"SSH to {ssh_host} timed out after {timeout}s"
    except subprocess.CalledProcessError as e:
        return False, f"SSH to {ssh_host} failed: {e}"
    except Exception as e:
        return False, f"SSH check failed: {e}"


# ============================================================================
# Main Discovery Loop
# ============================================================================


async def run_parallel_discovery(
    facility: str,
    cost_limit: float = 10.0,
    path_limit: int | None = None,
    focus: str | None = None,
    threshold: float = 0.7,
    num_scan_workers: int = 1,
    num_expand_workers: int = 1,
    num_score_workers: int = 4,
    num_enrich_workers: int = 1,
    num_rescore_workers: int = 0,
    scan_batch_size: int = 50,
    expand_batch_size: int = 50,
    score_batch_size: int = 50,  # Increased: more work per API call
    enrich_batch_size: int = 10,  # Smaller: heavy SSH operations (dust/tokei)
    rescore_batch_size: int = 50,  # Increased: lightweight heuristic rescore
    on_scan_progress: Callable[
        [str, WorkerStats, list[str] | None, list[dict] | None], None
    ]
    | None = None,
    on_expand_progress: Callable[
        [str, WorkerStats, list[str] | None, list[dict] | None], None
    ]
    | None = None,
    on_score_progress: Callable[[str, WorkerStats, list[dict] | None], None]
    | None = None,
    on_enrich_progress: Callable[[str, WorkerStats, list[dict] | None], None]
    | None = None,
    on_rescore_progress: Callable[[str, WorkerStats, list[dict] | None], None]
    | None = None,
    graceful_shutdown_timeout: float = 5.0,
) -> dict[str, Any]:
    """Run parallel discovery with all worker types.

    Workers:
    - scan: Lists directories, discovers child paths (SSH-bound)
    - expand: Expands scored high-value paths (should_expand=true)
    - score: LLM-based scoring with should_expand/should_enrich decisions
    - enrich: Deep analysis (dust/tokei/patterns) for should_enrich=true paths
    - rescore: Refines scores using enrichment data (optional)

    All workers run in PARALLEL - expand/enrich do not wait for each other.

    Args:
        facility: Facility ID to discover
        cost_limit: Maximum LLM cost in dollars
        path_limit: Maximum paths to process (optional)
        focus: Focus string for scoring (optional)
        threshold: Score threshold for expansion
        num_scan_workers: Number of concurrent scan workers (default: 1)
        num_expand_workers: Number of concurrent expand workers (default: 1)
        num_score_workers: Number of concurrent score workers (default: 4)
        num_enrich_workers: Number of concurrent enrich workers (default: 1)
        num_rescore_workers: Number of concurrent rescore workers (default: 0)
        scan_batch_size: Paths per SSH call (default: 50)
        expand_batch_size: Paths per expand SSH call (default: 50)
        score_batch_size: Paths per LLM call (default: 50)
        enrich_batch_size: Paths per SSH call (default: 10)
        rescore_batch_size: Paths per rescore (default: 50)
        on_scan_progress: Callback for scan progress
        on_expand_progress: Callback for expand progress
        on_score_progress: Callback for score progress
        on_enrich_progress: Callback for enrich progress
        on_rescore_progress: Callback for rescore progress
        graceful_shutdown_timeout: Seconds to wait for workers to finish after
            limit reached before cancelling (default: 5.0)

    Terminates when:
    - Cost limit reached
    - Path limit reached (if set)
    - All workers idle (no more work)

    Returns:
        Summary dict with scanned, expanded, scored, enriched, rescored, cost, elapsed, rates
    """
    from imas_codex.discovery import get_discovery_stats, seed_facility_roots

    # SSH preflight check - fail fast if facility is unreachable
    ssh_ok, ssh_message = check_ssh_connectivity(facility, timeout=15)
    if not ssh_ok:
        logger.error(f"SSH preflight failed: {ssh_message}")
        raise ConnectionError(f"Cannot connect to facility {facility}: {ssh_message}")

    # Reset any transient paths from previous runs (single CLI per facility)
    reset_transient_paths(facility)

    # Ensure we have paths to discover
    stats = get_discovery_stats(facility)
    if stats["total"] == 0:
        seed_facility_roots(facility)

    # Create shared state
    state = DiscoveryState(
        facility=facility,
        cost_limit=cost_limit,
        path_limit=path_limit,
        focus=focus,
        threshold=threshold,
    )

    # Capture initial terminal count for session-based --limit tracking
    state.initial_terminal_count = state.terminal_count

    # Create worker tasks
    scan_tasks = [
        scan_worker(state, on_progress=on_scan_progress, batch_size=scan_batch_size)
        for _ in range(num_scan_workers)
    ]
    expand_tasks = [
        expand_worker(
            state, on_progress=on_expand_progress, batch_size=expand_batch_size
        )
        for _ in range(num_expand_workers)
    ]
    score_tasks = [
        score_worker(state, on_progress=on_score_progress, batch_size=score_batch_size)
        for _ in range(num_score_workers)
    ]
    enrich_tasks = [
        enrich_worker(
            state, on_progress=on_enrich_progress, batch_size=enrich_batch_size
        )
        for _ in range(num_enrich_workers)
    ]
    rescore_tasks = [
        rescore_worker(
            state, on_progress=on_rescore_progress, batch_size=rescore_batch_size
        )
        for _ in range(num_rescore_workers)
    ]

    all_tasks = [
        asyncio.create_task(t)
        for t in scan_tasks + expand_tasks + score_tasks + enrich_tasks + rescore_tasks
    ]

    # Monitor for limit reached and cancel workers gracefully
    async def limit_monitor():
        """Monitor for budget/path limits and cancel workers when reached."""
        while not state.should_stop():
            await asyncio.sleep(0.25)
        # Limit reached - give workers a moment to finish current batch
        logger.info("Limit reached, initiating graceful shutdown...")
        await asyncio.sleep(graceful_shutdown_timeout)
        # Cancel any still-running tasks
        for task in all_tasks:
            if not task.done():
                task.cancel()

    monitor_task = asyncio.create_task(limit_monitor())

    # Run all workers concurrently with cancellation support
    try:
        await asyncio.gather(*all_tasks, return_exceptions=True)
    except asyncio.CancelledError:
        state.stop_requested = True
    finally:
        # Ensure monitor is stopped
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        # Graceful shutdown: reset any in-progress paths for next run
        reset_counts = reset_transient_paths(facility, silent=True)
        if reset_counts["listing_reset"] or reset_counts["scoring_reset"]:
            logger.info(
                f"Shutdown cleanup: {reset_counts['listing_reset']} listing, "
                f"{reset_counts['scoring_reset']} scoring paths reset"
            )

    # Auto-normalize scores after scoring completes
    if state.score_stats.processed > 0:
        from imas_codex.discovery.frontier import normalize_scores

        normalize_scores(facility)

    elapsed = max(
        state.scan_stats.elapsed,
        state.expand_stats.elapsed,
        state.score_stats.elapsed,
        state.enrich_stats.elapsed,
        state.rescore_stats.elapsed,
    )

    return {
        "scanned": state.scan_stats.processed,
        "expanded": state.expand_stats.processed,
        "scored": state.score_stats.processed,
        "enriched": state.enrich_stats.processed,
        "rescored": state.rescore_stats.processed,
        "cost": state.total_cost,
        "elapsed_seconds": elapsed,
        "scan_rate": state.scan_stats.rate,
        "expand_rate": state.expand_stats.rate,
        "score_rate": state.score_stats.rate,
        "enrich_rate": state.enrich_stats.rate,
        "rescore_rate": state.rescore_stats.rate,
        "scan_errors": state.scan_stats.errors,
        "expand_errors": state.expand_stats.errors,
        "score_errors": state.score_stats.errors,
        "enrich_errors": state.enrich_stats.errors,
        "rescore_errors": state.rescore_stats.errors,
    }
