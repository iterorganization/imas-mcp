"""
Parallel discovery engine with true concurrent scan and score workers.

Architecture:
- Two independent async workers: Scanner and Scorer
- Graph is the coordination mechanism (no locks needed)
- Atomic status transitions prevent race conditions:
  - pending → scanning → scanned (Scanner worker)
  - scanned → scoring → scored (Scorer worker)
- Workers continuously poll graph for work
- Cost-based termination for Scorer

Key insight: The graph acts as a thread-safe work queue. Each worker
claims work by atomically updating status, processes it, then marks complete.
No two workers can claim the same path.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


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
    score_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False
    scan_idle_count: int = 0
    score_idle_count: int = 0

    @property
    def total_cost(self) -> float:
        return self.score_stats.cost

    @property
    def total_processed(self) -> int:
        return self.scan_stats.processed + self.score_stats.processed

    @property
    def budget_exhausted(self) -> bool:
        return self.total_cost >= self.cost_limit

    @property
    def path_limit_reached(self) -> bool:
        if self.path_limit is None:
            return False
        return self.total_processed >= self.path_limit

    def should_stop(self) -> bool:
        """Check if discovery should terminate."""
        if self.stop_requested:
            return True
        if self.budget_exhausted:
            return True
        if self.path_limit_reached:
            return True
        # Stop if both workers idle for 3+ iterations
        if self.scan_idle_count >= 3 and self.score_idle_count >= 3:
            return True
        return False


# ============================================================================
# Graph-based work claiming (atomic status transitions)
# ============================================================================


def claim_paths_for_scanning(facility: str, limit: int = 50) -> list[dict[str, Any]]:
    """Atomically claim pending paths for scanning.

    Uses atomic status transition: pending → scanning
    Returns paths that this worker now owns.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        # Atomic claim: find pending, set to scanning, return claimed
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.status = 'pending'
            WITH p ORDER BY p.depth ASC, p.path ASC LIMIT $limit
            SET p.status = 'scanning', p.claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.depth AS depth
            """,
            facility=facility,
            limit=limit,
        )
        return list(result)


def claim_paths_for_scoring(facility: str, limit: int = 25) -> list[dict[str, Any]]:
    """Atomically claim scanned paths for scoring.

    Uses atomic status transition: scanned → scoring
    Returns paths that this worker now owns.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.status = 'scanned' AND p.score IS NULL
            WITH p ORDER BY p.depth ASC, p.path ASC LIMIT $limit
            SET p.status = 'scoring', p.claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.depth AS depth,
                   p.total_files AS total_files, p.total_dirs AS total_dirs,
                   p.file_type_counts AS file_type_counts,
                   p.has_readme AS has_readme, p.has_makefile AS has_makefile,
                   p.has_git AS has_git, p.patterns_detected AS patterns_detected,
                   p.child_names AS child_names
            """,
            facility=facility,
            limit=limit,
        )
        return list(result)


def mark_scan_complete(
    facility: str,
    scan_results: list[tuple[str, dict, list[str], str | None]],
    excluded: list[tuple[str, str, str]] | None = None,
) -> dict[str, int]:
    """Mark scanned paths complete and create children.

    Transition: scanning → scanned (or skipped on error)

    Args:
        facility: Facility ID
        scan_results: List of (path, stats_dict, child_dirs, error) tuples
        excluded: Optional list of (path, parent_path, reason) for excluded dirs
    """
    from imas_codex.discovery.frontier import persist_scan_results

    return persist_scan_results(facility, scan_results, excluded=excluded)


def mark_score_complete(
    facility: str,
    score_data: list[dict[str, Any]],
) -> int:
    """Mark scored paths complete.

    Transition: scoring → scored
    """
    from imas_codex.discovery.frontier import mark_paths_scored

    return mark_paths_scored(facility, score_data)


# ============================================================================
# Async Workers
# ============================================================================


async def scan_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[str] | None, list[dict] | None], None]
    | None = None,
) -> None:
    """Async scanner worker.

    Continuously claims pending paths, scans via SSH, marks complete.
    Runs until stop_requested or no more pending paths.
    """
    from imas_codex.discovery.scanner import scan_paths

    batch_size = 50  # Paths per SSH call

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
        loop = asyncio.get_event_loop()
        start = time.time()
        facility, paths_to_scan = state.facility, path_strs
        # Use enable_rg=False and enable_size=False for speed
        # Pattern detection is expensive and not needed for basic discovery
        results = await loop.run_in_executor(
            None,
            lambda fac=facility, pts=paths_to_scan: scan_paths(
                fac, pts, enable_rg=False, enable_size=False
            ),
        )
        state.scan_stats.last_batch_time = time.time() - start

        # Persist results (marks scanning → scanned)
        batch_data = [
            (r.path, r.stats.to_dict(), r.child_dirs, r.error) for r in results
        ]

        # Collect excluded directories with parent paths and reasons
        excluded_data = []
        for r in results:
            for excluded_path, reason in r.excluded_dirs:
                excluded_data.append((excluded_path, r.path, reason))

        stats = mark_scan_complete(
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


async def score_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[dict] | None], None] | None = None,
) -> None:
    """Async scorer worker.

    Continuously claims scanned paths, scores via LLM, marks complete.
    Runs until stop_requested, budget exhausted, or no more scanned paths.

    Optimization: Empty directories (total_files=0 AND total_dirs=0) are
    auto-skipped without LLM call since they have no content to evaluate.
    """
    from imas_codex.discovery.scorer import DirectoryScorer

    scorer = DirectoryScorer()
    batch_size = 25  # Paths per LLM call

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
                }
                for p in empty_paths
            ]
            mark_score_complete(state.facility, skip_data)
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
        # Capture variables for lambda to avoid late binding issue
        loop = asyncio.get_event_loop()
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
            score_data = [d.to_graph_dict() for d in result.scored_dirs]
            mark_score_complete(state.facility, score_data)

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

        except Exception as e:
            logger.exception(f"Score error: {e}")
            state.score_stats.errors += len(paths_to_score)
            # Revert claimed paths to scanned status
            _revert_scoring_claim(state.facility, [p["path"] for p in paths_to_score])

        # Brief yield
        await asyncio.sleep(0.1)


def _revert_scoring_claim(facility: str, paths: list[str]) -> None:
    """Revert paths from scoring back to scanned on error."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.path IN $paths AND p.status = 'scoring'
            SET p.status = 'scanned', p.claimed_at = null
            """,
            facility=facility,
            paths=paths,
        )


# ============================================================================
# Main Discovery Loop
# ============================================================================


async def run_parallel_discovery(
    facility: str,
    cost_limit: float = 10.0,
    path_limit: int | None = None,
    focus: str | None = None,
    threshold: float = 0.7,
    num_scan_workers: int = 2,
    num_score_workers: int = 4,
    on_scan_progress: Callable[
        [str, WorkerStats, list[str] | None, list[dict] | None], None
    ]
    | None = None,
    on_score_progress: Callable[[str, WorkerStats, list[dict] | None], None]
    | None = None,
) -> dict[str, Any]:
    """Run parallel scan and score workers.

    Args:
        facility: Facility ID to discover
        cost_limit: Maximum LLM cost in dollars
        path_limit: Maximum paths to process (optional)
        focus: Focus string for scoring (optional)
        threshold: Score threshold for expansion
        num_scan_workers: Number of concurrent scan workers (default: 2)
        num_score_workers: Number of concurrent score workers (default: 4)
        on_scan_progress: Callback for scan progress
        on_score_progress: Callback for score progress

    Terminates when:
    - Cost limit reached
    - Path limit reached (if set)
    - All workers idle (no more work)

    Returns:
        Summary dict with scanned, scored, cost, elapsed, rates
    """
    from imas_codex.discovery import get_discovery_stats, seed_facility_roots

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

    # Create worker tasks
    scan_tasks = [
        scan_worker(state, on_progress=on_scan_progress)
        for _ in range(num_scan_workers)
    ]
    score_tasks = [
        score_worker(state, on_progress=on_score_progress)
        for _ in range(num_score_workers)
    ]

    # Run all workers concurrently
    try:
        await asyncio.gather(*scan_tasks, *score_tasks)
    except asyncio.CancelledError:
        state.stop_requested = True

    elapsed = max(state.scan_stats.elapsed, state.score_stats.elapsed)

    return {
        "scanned": state.scan_stats.processed,
        "scored": state.score_stats.processed,
        "expanded": 0,  # TODO: track expansions
        "cost": state.total_cost,
        "elapsed_seconds": elapsed,
        "scan_rate": state.scan_stats.rate,
        "score_rate": state.score_stats.rate,
        "scan_errors": state.scan_stats.errors,
        "score_errors": state.score_stats.errors,
    }
