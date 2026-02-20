"""
Worker supervision infrastructure for parallel discovery engines.

Provides resilient async worker execution with:
- Automatic restart on crash with exponential backoff
- Infrastructure error detection (Neo4j, network, SSH)
- Live worker status tracking for progress displays
- Orphan claim recovery patterns

This module is used by all parallel discovery engines (paths, wiki, data)
to ensure workers recover gracefully from transient failures.

Usage:
    from imas_codex.discovery.base.supervision import (
        SupervisedWorkerGroup,
        WorkerStatus,
        is_infrastructure_error,
    )

    # Create supervised worker group
    group = SupervisedWorkerGroup()

    # Add workers with supervision
    group.add_worker(
        "score_worker_0",
        supervised_worker(
            score_worker,
            "score_worker_0",
            state,
            state.should_stop_scoring,
            on_progress=callback,
        )
    )

    # Start all workers
    await group.start()

    # Check status for display
    statuses = group.get_status_summary()

    # Clean shutdown
    await group.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

# Re-export for backwards compatibility
__all__ = [
    "DEFAULT_INITIAL_BACKOFF",
    "DEFAULT_MAX_BACKOFF",
    "DEFAULT_MAX_RESTARTS",
    "OrphanRecoveryResult",
    "OrphanRecoverySpec",
    "PipelinePhase",
    "SupervisedWorkerGroup",
    "WorkerState",
    "WorkerStatus",
    "is_infrastructure_error",
    "make_orphan_recovery_tick",
    "run_supervised_loop",
    "supervised_worker",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Worker Resilience Constants
# =============================================================================

# Default settings - can be overridden per worker
DEFAULT_MAX_RESTARTS = 10  # Maximum restarts before giving up
DEFAULT_INITIAL_BACKOFF = 2.0  # Initial backoff on infrastructure error (seconds)
DEFAULT_MAX_BACKOFF = 60.0  # Maximum backoff time (seconds)


# =============================================================================
# Infrastructure Error Detection
# =============================================================================


def is_infrastructure_error(exc: Exception) -> bool:
    """Check if an exception is a transient infrastructure error.

    Infrastructure errors warrant retry with backoff rather than immediate
    failure. They include:
    - Neo4j connection issues (ServiceUnavailable, SessionExpired, TransientError)
    - Neo4j critical database errors (OOM, txlog corruption on GPFS)
    - Network errors (timeouts, connection refused, reset)
    - SSH failures

    Args:
        exc: The exception to check

    Returns:
        True if this is a transient infrastructure error
    """
    from neo4j.exceptions import DatabaseError

    # Neo4j transient errors
    if isinstance(exc, ServiceUnavailable | SessionExpired | TransientError):
        return True

    # Neo4j critical database errors — the database needs a restart
    # but the error is transient from the worker's perspective
    if isinstance(exc, DatabaseError):
        msg = str(exc).lower()
        if "critical error" in msg or "needs to be restarted" in msg:
            return True

    # Connection errors (often wrapped in other exception types)
    error_msg = str(exc).lower()
    infrastructure_patterns = [
        "connection refused",
        "timed out",
        "connection reset",
        "broken pipe",
        "network is unreachable",
        "couldn't connect",
        "failed to establish",
        "handshake",
        "ssl",
        "ssh",
        "eof",
        "socket",
    ]
    return any(pattern in error_msg for pattern in infrastructure_patterns)


# =============================================================================
# Pipeline Phase Tracking
# =============================================================================


class PipelinePhase:
    """Track completion state for a pipeline phase (e.g., scan, score, enrich).

    Replaces ad-hoc idle counters with explicit completion semantics.
    A phase is considered idle when workers find no work for ``idle_threshold``
    consecutive polls.  A phase is done when it is idle AND a graph-level
    ``has_work_fn`` confirms nothing remains (including claimed items).

    Workers call :meth:`record_activity` after processing items and
    :meth:`record_idle` when a claim attempt returns nothing.

    The ``done`` property combines idle detection with an authoritative
    graph query, eliminating the race where a worker goes idle before
    upstream has flushed its output.

    Downstream phases can wait on :meth:`wait_until_done` to avoid
    exiting before upstream finishes producing work.

    Args:
        name: Human-readable phase name for logging.
        has_work_fn: Callable returning True if the graph has pending work
            for this phase (both unclaimed AND claimed items).  Passed as
            a zero-argument callable so the caller can bind facility etc.
            If None, only idle detection is used (no graph check).
        idle_threshold: Consecutive idle polls before considering idle.
            Default 3 (~3 seconds at 1s poll interval).
    """

    def __init__(
        self,
        name: str,
        has_work_fn: Callable[[], bool] | None = None,
        idle_threshold: int = 3,
    ) -> None:
        self.name = name
        self._has_work_fn = has_work_fn
        self._idle_threshold = idle_threshold
        self._idle_count = 0
        self._done_event = asyncio.Event()
        self._force_done = False
        self._total_processed = 0

    def record_activity(self, count: int = 1) -> None:
        """Record that the worker processed items — resets idle state."""
        self._idle_count = 0
        self._total_processed += count

    def record_idle(self) -> None:
        """Record an idle poll (no work found)."""
        self._idle_count += 1

    def mark_done(self) -> None:
        """Explicitly mark this phase as complete.

        Used when a phase knows it has finished deterministically
        (e.g., scan worker completed all scanner types).
        """
        self._force_done = True
        self._done_event.set()

    @property
    def idle(self) -> bool:
        """True if idle for at least ``idle_threshold`` consecutive polls."""
        return self._force_done or self._idle_count >= self._idle_threshold

    # Alias for readability in should_stop() methods
    is_idle = idle

    @property
    def done(self) -> bool:
        """True if the phase is complete: idle AND no pending graph work.

        When ``has_work_fn`` is provided, idle alone is not sufficient —
        we also verify that no unclaimed or claimed work remains in the
        graph.  This prevents premature termination when the upstream
        phase hasn't yet flushed to the graph.
        """
        if self._force_done:
            return True
        if self._idle_count < self._idle_threshold:
            return False
        # Idle threshold met — check graph for remaining work
        if self._has_work_fn is not None:
            try:
                if self._has_work_fn():
                    # Graph still has work — reset idle count so workers re-poll
                    self._idle_count = 0
                    return False
            except Exception as e:
                logger.debug("PipelinePhase %s: has_work_fn error: %s", self.name, e)
                return False
        # Genuinely done
        self._done_event.set()
        return True

    # Aliases for readability
    is_done = done

    @property
    def is_idle_or_done(self) -> bool:
        """True if idle or explicitly done — used in should_stop() checks."""
        return self.idle or self.done

    async def wait_until_done(self, timeout: float | None = None) -> bool:
        """Block until this phase is done (or timeout expires).

        Returns True if done, False on timeout.
        """
        try:
            await asyncio.wait_for(self._done_event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    def reset(self) -> None:
        """Reset idle counter (e.g., when graph check finds new work)."""
        self._idle_count = 0

    @property
    def idle_count(self) -> int:
        """Current consecutive idle count (for backwards compat / logging)."""
        return self._idle_count

    @property
    def total_processed(self) -> int:
        """Total items processed through this phase."""
        return self._total_processed

    def __repr__(self) -> str:
        return (
            f"PipelinePhase({self.name!r}, idle={self.idle}, done={self.done}, "
            f"processed={self._total_processed})"
        )


# =============================================================================
# Worker Status
# =============================================================================


class WorkerState(str, Enum):
    """Current state of a supervised worker."""

    starting = "starting"  # Worker is starting up
    running = "running"  # Worker is actively processing
    idle = "idle"  # Worker is waiting for work
    backoff = "backoff"  # Worker is in backoff after error
    stopped = "stopped"  # Worker has stopped (normally or max restarts)
    crashed = "crashed"  # Worker crashed and won't restart


@dataclass
class WorkerStatus:
    """Status of a single supervised worker."""

    name: str
    group: str = ""  # Functional group: "score" (LLM/VLM) or "ingest" (I/O)
    state: WorkerState = WorkerState.starting
    restart_count: int = 0
    max_restarts: int = DEFAULT_MAX_RESTARTS
    last_error: str | None = None
    last_error_time: float | None = None
    backoff_until: float | None = None
    items_processed: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def is_active(self) -> bool:
        """Worker is actively running or will restart."""
        return self.state in (
            WorkerState.starting,
            WorkerState.running,
            WorkerState.idle,
            WorkerState.backoff,
        )

    @property
    def uptime(self) -> float:
        """Seconds since worker started."""
        return time.time() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "state": self.state.value,
            "restart_count": self.restart_count,
            "max_restarts": self.max_restarts,
            "is_active": self.is_active,
            "last_error": self.last_error,
            "items_processed": self.items_processed,
        }


# =============================================================================
# Supervised Worker Wrapper
# =============================================================================


async def supervised_worker(
    worker_fn: Callable[..., Coroutine],
    worker_name: str,
    state: Any,  # Domain-specific state object
    should_stop_fn: Callable[[], bool],
    *args: Any,
    status_tracker: WorkerStatus | None = None,
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
    **kwargs: Any,
) -> None:
    """Run a worker with supervision - automatic restart on crash.

    This wrapper catches any exception from the worker, logs it, and restarts
    the worker with exponential backoff. This ensures transient infrastructure
    errors (Neo4j timeouts, network issues) don't permanently kill workers.

    Args:
        worker_fn: The async worker function to supervise
        worker_name: Name for logging (e.g., "score_worker_0")
        state: Shared discovery state (for stop checks)
        should_stop_fn: Function to check if worker should stop
        *args, **kwargs: Arguments passed to worker_fn
        status_tracker: Optional WorkerStatus for live monitoring
        max_restarts: Maximum restarts before giving up
        initial_backoff: Initial backoff delay in seconds
        max_backoff: Maximum backoff delay in seconds
    """
    restart_count = 0
    backoff = initial_backoff

    # Initialize status tracker if provided
    if status_tracker:
        status_tracker.state = WorkerState.starting
        status_tracker.max_restarts = max_restarts

    while not should_stop_fn() and restart_count < max_restarts:
        try:
            if status_tracker:
                status_tracker.state = WorkerState.running

            await worker_fn(state, *args, **kwargs)

            # Worker exited normally (should_stop returned True)
            logger.debug("%s exited normally", worker_name)
            if status_tracker:
                status_tracker.state = WorkerState.stopped
            return

        except asyncio.CancelledError:
            # Explicit cancellation - don't restart
            logger.debug("%s cancelled", worker_name)
            if status_tracker:
                status_tracker.state = WorkerState.stopped
            raise

        except Exception as e:
            restart_count += 1
            is_infra = is_infrastructure_error(e)

            # Update status tracker
            if status_tracker:
                status_tracker.restart_count = restart_count
                status_tracker.last_error = str(e)[:200]
                status_tracker.last_error_time = time.time()

            if is_infra:
                # Infrastructure error - backoff and retry
                logger.warning(
                    "%s infrastructure error (restart %d/%d): %s. Backing off %.1fs...",
                    worker_name,
                    restart_count,
                    max_restarts,
                    e,
                    backoff,
                )
                if status_tracker:
                    status_tracker.state = WorkerState.backoff
                    status_tracker.backoff_until = time.time() + backoff

                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            else:
                # Application error - log full traceback, shorter backoff
                logger.error(
                    "%s crashed (restart %d/%d): %s\n%s",
                    worker_name,
                    restart_count,
                    max_restarts,
                    e,
                    traceback.format_exc(),
                )
                if status_tracker:
                    status_tracker.state = WorkerState.backoff
                    status_tracker.backoff_until = time.time() + initial_backoff

                await asyncio.sleep(initial_backoff)
                # Reset backoff for non-infrastructure errors
                backoff = initial_backoff

    # Exceeded max restarts or stop requested
    if restart_count >= max_restarts:
        logger.error(
            "%s exceeded max restarts (%d), giving up",
            worker_name,
            max_restarts,
        )
        if status_tracker:
            status_tracker.state = WorkerState.crashed
    else:
        if status_tracker:
            status_tracker.state = WorkerState.stopped


# =============================================================================
# Supervised Worker Group
# =============================================================================


@dataclass
class SupervisedWorkerGroup:
    """Manages a group of supervised workers with status tracking.

    Provides:
    - Centralized worker status tracking
    - Clean startup and shutdown
    - Status summaries for live display

    Usage:
        group = SupervisedWorkerGroup()

        # Create status trackers
        score_status = group.create_status("score_worker_0")

        # Add supervised worker task
        group.add_task(
            asyncio.create_task(
                supervised_worker(
                    score_worker, "score_worker_0", state,
                    state.should_stop_scoring,
                    status_tracker=score_status,
                )
            )
        )

        # Get status for display
        summary = group.get_status_summary()
    """

    _workers: dict[str, WorkerStatus] = field(default_factory=dict)
    _tasks: list[asyncio.Task] = field(default_factory=list)

    @property
    def workers(self) -> dict[str, WorkerStatus]:
        """Access to worker status dict for display."""
        return self._workers

    def create_status(self, name: str, group: str = "") -> WorkerStatus:
        """Create and register a worker status tracker.

        Args:
            name: Unique worker name (e.g., "score_worker_0")
            group: Functional group for display grouping.
                   Convention: "score" for LLM/VLM workers,
                   "ingest" for I/O/embed workers.
        """
        status = WorkerStatus(name=name, group=group)
        self._workers[name] = status
        return status

    def add_task(self, task: asyncio.Task) -> None:
        """Add a worker task to the group."""
        self._tasks.append(task)

    def get_status(self, name: str) -> WorkerStatus | None:
        """Get status for a specific worker."""
        return self._workers.get(name)

    def get_all_status(self) -> list[WorkerStatus]:
        """Get status for all workers."""
        return list(self._workers.values())

    def get_status_summary(self) -> dict[str, dict[str, int]]:
        """Get status summary grouped by worker type.

        Returns dict like:
        {
            "score": {"running": 2, "idle": 0, "backoff": 1},
            "ingest": {"running": 1, "idle": 0, "backoff": 0},
        }
        """
        summary: dict[str, dict[str, int]] = {}

        for status in self._workers.values():
            # Extract worker type from name (e.g., "score_worker_0" -> "score")
            worker_type = status.name.split("_worker")[0]

            if worker_type not in summary:
                summary[worker_type] = {
                    "running": 0,
                    "idle": 0,
                    "backoff": 0,
                    "stopped": 0,
                    "crashed": 0,
                }

            state_key = status.state.value
            if state_key in summary[worker_type]:
                summary[worker_type][state_key] += 1
            elif state_key == "starting":
                summary[worker_type]["running"] += 1

        return summary

    def get_active_count(self) -> int:
        """Count of workers that are active (not stopped/crashed)."""
        return sum(1 for s in self._workers.values() if s.is_active)

    def get_error_count(self) -> int:
        """Count of workers in error state (backoff or crashed)."""
        return sum(
            1
            for s in self._workers.values()
            if s.state in (WorkerState.backoff, WorkerState.crashed)
        )

    async def cancel_all(self, timeout: float = 10.0) -> None:
        """Cancel all worker tasks.

        Args:
            timeout: Maximum seconds to wait for each task after cancellation.
        """
        for task in self._tasks:
            task.cancel()
        # Wait for all to finish with a hard timeout
        pending = [t for t in self._tasks if not t.done()]
        if pending:
            _, still_pending = await asyncio.wait(pending, timeout=timeout)
            if still_pending:
                logger.warning(
                    "%d task(s) did not finish within %ss, abandoning",
                    len(still_pending),
                    timeout,
                )


# =============================================================================
# Orphan Recovery
# =============================================================================


@dataclass
class OrphanRecoveryResult:
    """Result of orphan claim recovery."""

    released_count: int = 0
    released_ids: list[str] = field(default_factory=list)
    error: str | None = None


def make_orphan_recovery_query(
    label: str,
    facility_field: str = "facility_id",
    claimed_field: str = "claimed_at",
    timeout_seconds: int = 300,
) -> str:
    """Generate Cypher query for releasing orphaned claims.

    Args:
        label: Node label (e.g., "WikiPage", "FacilityPath")
        facility_field: Field containing facility ID
        claimed_field: Field containing claim timestamp
        timeout_seconds: How old a claim must be to be orphaned

    Returns:
        Cypher query string with $facility parameter
    """
    cutoff = f"PT{timeout_seconds}S"
    return f"""
        MATCH (n:{label} {{{facility_field}: $facility}})
        WHERE n.{claimed_field} IS NOT NULL
          AND n.{claimed_field} < datetime() - duration("{cutoff}")
        SET n.{claimed_field} = null
        RETURN n.id AS id, n.status AS status
    """


async def release_orphaned_claims_generic(
    facility: str,
    queries: list[tuple[str, str]],  # [(label, query), ...]
) -> dict[str, OrphanRecoveryResult]:
    """Release orphaned claims using provided queries.

    Args:
        facility: Facility ID
        queries: List of (label, cypher_query) tuples

    Returns:
        Dict mapping label to OrphanRecoveryResult
    """
    from imas_codex.graph import GraphClient

    results = {}

    for label, query in queries:
        try:
            with GraphClient() as gc:
                result = gc.query(query, facility=facility)
                released = list(result)
                results[label] = OrphanRecoveryResult(
                    released_count=len(released),
                    released_ids=[r["id"] for r in released],
                )
                if released:
                    logger.info(
                        "Released %d orphaned %s claims for %s",
                        len(released),
                        label,
                        facility,
                    )
        except Exception as e:
            logger.warning("Could not release orphaned %s claims: %s", label, e)
            results[label] = OrphanRecoveryResult(error=str(e))

    return results


# =============================================================================
# Periodic Orphan Recovery
# =============================================================================


@dataclass
class OrphanRecoverySpec:
    """Specification for a node type's orphan recovery.

    Args:
        label: Neo4j node label (e.g., "FacilityPath", "WikiPage")
        facility_field: Property containing the facility ID (default: "facility_id")
        timeout_seconds: How old a claim must be to be orphaned (default: 300)
    """

    label: str
    facility_field: str = "facility_id"
    timeout_seconds: int = 300


def make_orphan_recovery_tick(
    facility: str,
    specs: list[OrphanRecoverySpec],
    *,
    interval: float = 60.0,
) -> Callable[[], None]:
    """Create a periodic orphan recovery tick for ``run_supervised_loop``.

    Returns a callable suitable for the ``on_tick`` parameter of
    :func:`run_supervised_loop`.  Each invocation checks whether enough
    time has elapsed since the last recovery pass.  If so, it releases
    stale claims for every node type listed in *specs*.

    Args:
        facility: Facility ID to recover claims for
        specs: List of :class:`OrphanRecoverySpec` describing which node
            labels to check
        interval: Minimum seconds between recovery passes (default: 60)

    Returns:
        A zero-argument callable that performs time-gated orphan recovery
    """
    from imas_codex.discovery.base.claims import reset_stale_claims

    last_check = time.time()

    def _tick() -> None:
        nonlocal last_check
        if time.time() - last_check < interval:
            return
        for spec in specs:
            try:
                released = reset_stale_claims(
                    spec.label,
                    facility,
                    timeout_seconds=spec.timeout_seconds,
                    facility_field=spec.facility_field,
                )
                if released:
                    logger.info(
                        "Orphan recovery: released %d %s claims for %s",
                        released,
                        spec.label,
                        facility,
                    )
            except Exception as e:
                logger.debug("Orphan recovery check failed for %s: %s", spec.label, e)
        last_check = time.time()

    return _tick


# =============================================================================
# Common Discovery Loop
# =============================================================================


async def run_supervised_loop(
    worker_group: SupervisedWorkerGroup,
    should_stop: Callable[[], bool],
    *,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
    on_tick: Callable[[], Coroutine | None] | None = None,
    status_interval: float = 0.5,
    poll_interval: float = 0.25,
    shutdown_timeout: float = 10.0,
) -> None:
    """Run the common supervision polling loop used by all discovery engines.

    Polls ``should_stop`` periodically, sends worker status updates to the
    display callback, and performs clean shutdown when discovery completes
    or is cancelled.

    Args:
        worker_group: Group of supervised workers to monitor
        should_stop: Function returning True when discovery should stop
        on_worker_status: Optional callback for worker status updates
        on_tick: Optional async callback called each tick (e.g., orphan recovery)
        status_interval: Seconds between worker status updates (default: 0.5)
        poll_interval: Seconds between stop-condition checks (default: 0.25)
        shutdown_timeout: Seconds to wait for workers after cancellation
    """
    # Send initial worker status update
    if on_worker_status:
        try:
            on_worker_status(worker_group)
        except Exception as e:
            logger.warning("Initial worker status callback failed: %s", e)

    last_status_update = time.time()

    try:
        while not should_stop():
            await asyncio.sleep(poll_interval)

            # Update worker status for display
            if on_worker_status and time.time() - last_status_update > status_interval:
                try:
                    on_worker_status(worker_group)
                except Exception as e:
                    logger.warning("Worker status callback failed: %s", e)
                last_status_update = time.time()

            # Run optional per-tick callback (e.g., orphan recovery)
            if on_tick:
                result = on_tick()
                if asyncio.iscoroutine(result):
                    await result
    except asyncio.CancelledError:
        logger.debug("Supervised loop cancelled")
    finally:
        await worker_group.cancel_all(timeout=shutdown_timeout)
