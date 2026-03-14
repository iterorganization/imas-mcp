"""Generic discovery engine skeleton.

Provides ``run_discovery_engine()``, the single entry point for running
a parallel discovery pipeline with supervised workers, orphan recovery,
and stop-event watching.  Domain engines call this instead of
reimplementing the 7-step boilerplate (preflight → state → workers →
supervision → cleanup).

Usage::

    from imas_codex.discovery.base.engine import (
        WorkerSpec,
        run_discovery_engine,
    )

    result = await run_discovery_engine(
        state=my_state,
        workers=[
            WorkerSpec("extract", "extract_phase", extract_worker,
                       count=2, on_progress=on_extract),
            WorkerSpec("enrich", "enrich_phase", enrich_worker,
                       on_progress=on_enrich, enabled=do_enrich),
        ],
        orphan_labels=["SignalEpoch", "SignalNode"],
        on_worker_status=on_worker_status,
        stop_event=stop_event,
    )
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.supervision import (
    OrphanRecoverySpec,
    PipelinePhase,
    SupervisedWorkerGroup,
    make_orphan_recovery_tick,
    run_supervised_loop,
    supervised_worker,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerSpec:
    """Declarative specification for a single worker type.

    Args:
        name: Human-readable worker name (used for logging and status,
            e.g. "extract", "score").  When ``count > 1`` the engine
            appends ``_0``, ``_1``, etc.
        phase_attr: Name of the ``PipelinePhase`` attribute on the state
            object (e.g. ``"extract_phase"``).
        worker_fn: Async worker function with signature
            ``async def fn(state, **kwargs)``.
        count: Number of parallel instances (default: 1).
        enabled: Set False to mark the phase done and skip registration.
        should_stop_fn: Optional per-worker stop function.  When None,
            ``state.should_stop`` is used.
        on_progress: Optional progress callback forwarded to the worker
            via ``on_progress=`` kwarg.
        group: Worker group name for display grouping (defaults to name).
        kwargs: Extra keyword arguments forwarded to ``worker_fn``.
    """

    name: str
    phase_attr: str
    worker_fn: Callable[..., Coroutine]
    count: int = 1
    enabled: bool = True
    should_stop_fn: Callable[[], bool] | None = None
    on_progress: Callable | None = None
    group: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


async def run_discovery_engine(
    state: Any,
    workers: list[WorkerSpec],
    *,
    stop_event: asyncio.Event | None = None,
    orphan_specs: list[OrphanRecoverySpec] | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
    cleanup: Callable[[], Coroutine] | None = None,
    stop_fn: Callable[[], bool] | None = None,
    shutdown_timeout: float | None = None,
) -> None:
    """Run a parallel discovery pipeline using the standard skeleton.

    Steps executed:
    1. Disable phases for workers that are not enabled
    2. Create stop-event watcher (if stop_event provided)
    3. Register all workers into a SupervisedWorkerGroup
    4. Set up periodic orphan recovery
    5. Run the supervised polling loop
    6. Mark state as stopped and run cleanup

    The caller is responsible for:
    - Preflight (SSH, graph seeding, state construction)
    - Phase wiring (``set_has_work_fn`` on each PipelinePhase)
    - Building the results dict from state after this returns

    Args:
        state: Domain-specific state object (should extend
            ``DiscoveryStateBase`` or at least have ``stop_requested``
            and ``should_stop()``).
        workers: Declarative list of worker specifications.
        stop_event: Optional asyncio.Event from the CLI shutdown handler.
        orphan_specs: Optional list of OrphanRecoverySpec for periodic
            claim cleanup.
        on_worker_status: Optional callback for display updates.
        cleanup: Optional async cleanup function called in the finally
            block (e.g. close SSH pools, HTTP clients).
        stop_fn: Optional override for the top-level stop condition.
            When None, ``state.should_stop`` is used.
        shutdown_timeout: Optional timeout for graceful worker shutdown.
    """
    # --- Step 1: Disable phases for workers that are not enabled or have count=0 ---
    for spec in workers:
        if not spec.enabled or spec.count == 0:
            phase = getattr(state, spec.phase_attr, None)
            if phase is not None:
                phase.mark_done()

    # --- Step 2: Stop-event watcher ---
    stop_watcher: asyncio.Task | None = None
    if stop_event is not None:
        from imas_codex.cli.shutdown import watch_stop_event

        stop_watcher = asyncio.create_task(watch_stop_event(stop_event, state))

    # --- Step 3: Register workers ---
    worker_group = SupervisedWorkerGroup()

    for spec in workers:
        if not spec.enabled or spec.count == 0:
            continue

        phase = getattr(state, spec.phase_attr)
        worker_stop_fn = spec.should_stop_fn or state.should_stop
        group_name = spec.group or spec.name

        extra_kwargs: dict[str, Any] = {}
        if spec.on_progress is not None:
            extra_kwargs["on_progress"] = spec.on_progress
        extra_kwargs.update(spec.kwargs)

        for i in range(spec.count):
            suffix = f"_{i}" if spec.count > 1 else "_0"
            worker_name = f"{spec.name}_worker{suffix}"
            status = worker_group.create_status(worker_name, group=group_name)

            task = asyncio.create_task(
                supervised_worker(
                    spec.worker_fn,
                    worker_name,
                    state,
                    worker_stop_fn,
                    status_tracker=status,
                    **extra_kwargs,
                )
            )
            worker_group.add_task(task)

    # --- Step 4: Orphan recovery tick ---
    orphan_tick = None
    if orphan_specs:
        orphan_tick = make_orphan_recovery_tick(state.facility, orphan_specs)

    # Collect all PipelinePhase objects for cache refresh
    phases = list(getattr(state, "all_phases", {}).values())

    # Resolve the effective stop function
    effective_stop_fn = stop_fn or state.should_stop

    # --- Step 5: Supervision loop ---
    try:
        loop_kwargs: dict[str, Any] = {}
        if shutdown_timeout is not None:
            loop_kwargs["shutdown_timeout"] = shutdown_timeout

        await run_supervised_loop(
            worker_group,
            effective_stop_fn,
            on_worker_status=on_worker_status,
            on_tick=orphan_tick,
            phases=phases or None,
            **loop_kwargs,
        )
    finally:
        # --- Step 6: Cleanup ---
        state.stop_requested = True
        if stop_watcher and not stop_watcher.done():
            stop_watcher.cancel()
        if cleanup:
            try:
                await cleanup()
            except Exception as e:
                logger.warning("Engine cleanup failed: %s", e)
