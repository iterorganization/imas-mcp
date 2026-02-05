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
    - Network errors (timeouts, connection refused, reset)
    - SSH failures

    Args:
        exc: The exception to check

    Returns:
        True if this is a transient infrastructure error
    """
    # Neo4j transient errors
    if isinstance(exc, ServiceUnavailable | SessionExpired | TransientError):
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

    def create_status(self, name: str) -> WorkerStatus:
        """Create and register a worker status tracker."""
        status = WorkerStatus(name=name)
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

    async def cancel_all(self) -> None:
        """Cancel all worker tasks."""
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


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
