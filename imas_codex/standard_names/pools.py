"""Concurrent worker-pool orchestrator for ``sn run`` (Phase 8).

Replaces the per-domain serial ``run_sn_loop`` with N persistent async
tasks that pull work from the graph independently and share a single
:class:`BudgetManager` for cost coordination.

Six pools run under a single ``asyncio.gather``:

* ``generate_name``  — composes new ``StandardName`` rows from
                       ``StandardNameSource(status='extracted')``.
* ``review_name``    — scores name+grammar (name_stage='drafted').
* ``refine_name``    — recomposes names below the review threshold.
* ``generate_docs``  — adds description+documentation to accepted names.
* ``review_docs``    — scores description+documentation.
* ``refine_docs``    — refines docs below the review threshold.

Each pool follows a cooperative shutdown contract:

    while not stop_event.is_set():
        wait for admission via mgr.pool_admit(...)
        claim a coherent batch (seed-and-expand)
        if no work: sleep with exponential backoff
        process_batch(...)  # checks stop_event between LLM and persist
        release_claim     (always, in finally)

After ``stop_event``, the outer harness gives pools a 60s grace window
to complete in-flight batches before issuing ``task.cancel()``.  The
budget writer queue is drained before the SNRun is finalized so cost
accounting remains exact.

This module deliberately contains only the orchestration scaffolding.
The six claim queries live in ``graph_ops.py`` and the per-pool
batch-processor functions live in their respective worker modules
(``workers.py`` for generate_name+refine_name, ``review/pipeline.py``
for legacy review, ``workers.py`` for all Phase 8.1 workers).  This
separation keeps the orchestrator agnostic of prompt/persist details —
its only job is to schedule.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetManager

logger = logging.getLogger(__name__)


# Default per-pool weights for soft-fairness admission control.
# Sum to 1.0.  Six pools per Phase 8.1 refine pipeline.
POOL_WEIGHTS: dict[str, float] = {
    "generate_name": 0.25,
    "review_name": 0.15,
    "refine_name": 0.15,
    "generate_docs": 0.20,
    "review_docs": 0.15,
    "refine_docs": 0.10,
}

POOL_NAMES: tuple[str, ...] = tuple(POOL_WEIGHTS.keys())

# Number of consecutive admitted-but-empty claim attempts before a pool
# is temporarily excluded from the active-pools admission denominator.
# This prevents fairness deadlocks when a pool has non-zero pending_count
# (as reported by the display query) but its claim function consistently
# returns nothing (different eligibility criteria between display and claim).
# Resets to zero on any successful claim.
_EMPTY_CLAIM_EXCLUDE_THRESHOLD: int = 3


@dataclass
class _PoolBackoff:
    """Exponential backoff for empty-claim spins.

    Each pool maintains its own backoff state.  On an empty claim
    result, ``next_sleep`` returns the next interval; on a successful
    claim, ``reset()`` returns the pool to the base interval.

    Schedule: 3s → 6s → 12s → 24s → 30s (capped).
    """

    base: float = 3.0
    cap: float = 30.0
    _current: float = 3.0

    def reset(self) -> None:
        self._current = self.base

    def next_sleep(self) -> float:
        # Add ±10% jitter to avoid lock-step polling across pools.
        jitter = random.uniform(0.9, 1.1)
        sleep_for = self._current * jitter
        self._current = min(self._current * 2.0, self.cap)
        return sleep_for


@dataclass
class PoolHealth:
    """Liveness telemetry for a single pool.

    Mirrors the per-subpool wedge detection rules from plan.md Phase 8
    finding M7.  ``last_progress_at`` is updated on each successful
    batch persist.  ``pending_count`` is refreshed by the orchestrator
    via the pending-fn callback.  ``in_flight`` tracks claimed but
    not-yet-persisted batches for restart-safety reasoning.

    ``consecutive_empty_claims`` counts successive admitted-but-empty
    claim attempts.  The admission gate uses this to temporarily exclude
    a pool from the active-pools denominator when it has pending work
    recorded in the graph but consistently returns nothing from claim —
    which happens when the display's pending query and the claim query
    have different eligibility criteria.
    The counter resets on any successful claim.

    Self-healing re-admission: when a pool has been excluded due to
    consecutive_empty_claims, ``active_pools_fn`` checks whether the
    pool's ``pending_count`` has grown since the last check.  A pending
    count increase means new eligible nodes have appeared (e.g. the
    generate pool produced names that are now enrich-eligible).  When
    that happens ``consecutive_empty_claims`` is reset to 0 so the pool
    re-enters contention.  ``_last_pending_count`` tracks the value from
    the most recent active_pools_fn evaluation.

    ``total_processed`` accumulates the sum of all successful
    ``spec.process(batch)`` return values.  Used by ``run_sn_pools`` to
    populate ``SNRun.names_composed/enriched/reviewed/regenerated``.
    """

    pool: str
    last_progress_at: float = field(default_factory=time.time)
    pending_count: int = 0
    in_flight: int = 0
    error_count: int = 0
    last_error: str | None = None
    consecutive_empty_claims: int = 0
    total_processed: int = 0
    _last_pending_count: int = 0

    def mark_progress(self) -> None:
        self.last_progress_at = time.time()

    def is_wedged(self, *, poll_interval: float, now: float | None = None) -> bool:
        """A pool is wedged when it has pending work but hasn't progressed
        for at least 2× the poll interval.
        """
        if self.pending_count <= 0:
            return False
        ts = now if now is not None else time.time()
        return (ts - self.last_progress_at) > 2.0 * poll_interval


# Public type aliases for pool callables.  ``ClaimFn`` returns either a
# claimed batch (opaque dict; pool-specific shape) or ``None`` when no
# eligible work is available.  ``ProcessFn`` consumes the batch and
# performs LLM + persist work; it is responsible for releasing the
# claim (typically in a try/finally) and returning the count of items
# successfully processed.  ``ReleaseFn`` is an optional error-recovery
# hook called when ``ProcessFn`` raises — it should unlock the claimed
# items so other workers can pick them up.
ClaimFn = Callable[[], Awaitable[dict[str, Any] | None]]
ProcessFn = Callable[[dict[str, Any]], Awaitable[int]]
ReleaseFn = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class PoolSpec:
    """Configuration for a single pool's worker loop.

    Each pool wraps a graph-side claim query and a batch processor.
    The orchestrator only needs these two callables plus the pool name
    to drive the loop — all pipeline-specific logic stays inside the
    callables themselves.
    """

    name: str
    claim: ClaimFn
    process: ProcessFn
    weight: float = 0.0
    release: ReleaseFn | None = None
    health: PoolHealth = field(init=False)
    backoff: _PoolBackoff = field(default_factory=_PoolBackoff)

    def __post_init__(self) -> None:
        if self.weight == 0.0:
            self.weight = POOL_WEIGHTS.get(self.name, 0.0)
        self.health = PoolHealth(pool=self.name)


async def pool_loop(
    spec: PoolSpec,
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    active_pools_fn: Callable[[], set[str]],
    weights: dict[str, float] = POOL_WEIGHTS,
    admission_poll: float = 0.5,
) -> None:
    """Cooperative pool worker.

    Loop semantics (per plan.md Phase 8 finding H6):

    * On each iteration, check ``stop_event``.  If set, exit cleanly
      without claiming new work.
    * Wait for admission via ``mgr.pool_admit(spec.name, weights,
      active_pools_fn())`` — this enforces weighted fairness across
      live pools.  Sleep ``admission_poll`` seconds and retry while
      not admitted.
    * Claim a batch.  If ``None``, increment backoff and sleep.
    * Process the batch; mark progress on success; reset backoff.
    * Errors are logged and counted but do not crash the pool — the
      next iteration will retry (the claim infrastructure handles
      orphan recovery via ``claimed_at`` timeout).
    """
    logger.info("pool[%s] starting", spec.name)
    while not stop_event.is_set():
        # ── Admission gate ────────────────────────────────────────
        if not mgr.pool_admit(spec.name, weights, active_pools_fn()):
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=admission_poll)
                break  # stop_event triggered during wait
            except TimeoutError:
                continue

        # ── Claim ────────────────────────────────────────────────
        try:
            batch = await spec.claim()
        except Exception as exc:  # noqa: BLE001 — log and continue
            spec.health.error_count += 1
            spec.health.last_error = f"claim: {exc!r}"
            logger.exception("pool[%s] claim failed: %s", spec.name, exc)
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=spec.backoff.next_sleep()
                )
                break
            except TimeoutError:
                continue

        if batch is None:
            spec.health.consecutive_empty_claims += 1
            sleep_for = spec.backoff.next_sleep()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_for)
                break
            except TimeoutError:
                continue

        # Reset backoff and empty-claim counter after a successful claim.
        spec.backoff.reset()
        spec.health.consecutive_empty_claims = 0
        spec.health.in_flight += 1

        # ── Process ───────────────────────────────────────────────
        try:
            count = await spec.process(batch)
            spec.health.total_processed += count
            spec.health.mark_progress()
            logger.debug("pool[%s] processed %d items", spec.name, count)
        except asyncio.CancelledError:
            logger.info("pool[%s] cancelled mid-batch", spec.name)
            raise
        except Exception as exc:  # noqa: BLE001
            spec.health.error_count += 1
            spec.health.last_error = f"process: {exc!r}"
            logger.exception("pool[%s] process failed: %s", spec.name, exc)
            if spec.release is not None:
                try:
                    await spec.release(batch)
                except Exception as rel_exc:  # noqa: BLE001
                    logger.exception(
                        "pool[%s] release failed (continuing): %s",
                        spec.name,
                        rel_exc,
                    )
        finally:
            spec.health.in_flight = max(0, spec.health.in_flight - 1)
    logger.info("pool[%s] exiting cleanly", spec.name)


async def _budget_watchdog(
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    poll: float = 5.0,
) -> None:
    """Poll ``mgr.exhausted()`` and set ``stop_event`` when budget runs out.

    This ensures the pools receive a clean shutdown signal even in headless
    mode where the Rich display ticker never updates ``PoolHealth.pending_count``
    (which was the root cause of the live smoke run's 10.5× overshoot).

    The function terminates as soon as ``stop_event`` is set — whether by
    this watchdog itself or by an external signal — so it never outlives the
    pool tasks.
    """
    while not stop_event.is_set():
        if mgr.exhausted():
            logger.info("run_pools: budget exhausted — signalling graceful shutdown")
            stop_event.set()
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll)
        except TimeoutError:
            continue


async def _pending_count_watchdog(
    pools: list[PoolSpec],
    stop_event: asyncio.Event,
    pending_fn: Callable[[], dict[str, int]],
    poll: float = 5.0,
) -> None:
    """Poll ``pending_fn`` and push per-pool pending counts to ``PoolHealth``.

    Runs alongside the pool tasks and the budget watchdog.  Calling
    ``pending_fn()`` every *poll* seconds makes ``active_pools_fn`` accurate
    in headless / ``--quiet`` mode where the Rich display ticker is absent.
    Without this watchdog the fairness mechanism degrades to unconditional
    admission for all pools, allowing cheap pools (generate, regen) to starve
    expensive-but-important pools (enrich, review_docs).

    The watchdog performs an initial poll immediately so that pending counts are
    populated before the first admission decision.  It then sleeps *poll*
    seconds between subsequent refreshes.

    Pool names in the ``pending_fn`` result that do not match a live pool are
    silently ignored.  Pools absent from the result default to 0.
    """
    pool_by_name = {p.name: p for p in pools}

    def _refresh() -> None:
        try:
            counts = pending_fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("pending_count_watchdog: pending_fn raised: %s", exc)
            return
        for name, pool in pool_by_name.items():
            pool.health.pending_count = counts.get(name, 0)
        logger.debug(
            "pending_count_watchdog: updated pending counts — %s",
            {n: pool_by_name[n].health.pending_count for n in pool_by_name},
        )

    # First poll before any admission decisions.
    _refresh()

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll)
            break  # stop_event triggered
        except TimeoutError:
            _refresh()


async def run_pools(
    pools: list[PoolSpec],
    mgr: BudgetManager,
    stop_event: asyncio.Event,
    *,
    pending_fn: Callable[[], dict[str, int]] | None = None,
    pending_poll_interval: float = 5.0,
    grace_period: float = 60.0,
    weights: dict[str, float] = POOL_WEIGHTS,
) -> dict[str, PoolHealth]:
    """Run all pool loops concurrently and orchestrate cooperative shutdown.

    Returns a mapping of pool name → final ``PoolHealth`` snapshot.

    Shutdown sequence (plan.md Phase 8 finding H6):

    1. ``stop_event.set()`` (set by caller, e.g. via ``run_discovery``
       3-press shutdown, or by the budget watchdog when exhausted).
    2. Each pool exits its claim loop and returns; pools currently
       processing a batch finish that batch normally.
    3. The harness waits up to ``grace_period`` seconds for all pool
       tasks to complete naturally.
    4. Any pool still running after grace is hard-cancelled.
    5. ``mgr.drain_pending()`` flushes the LLMCost write queue.

    A ``_budget_watchdog`` task runs alongside the pool tasks and sets
    ``stop_event`` as soon as ``mgr.exhausted()`` flips True, propagating
    clean shutdown through the existing grace path.

    When *pending_fn* is provided, a ``_pending_count_watchdog`` task polls
    it every *pending_poll_interval* seconds and updates each pool's
    ``PoolHealth.pending_count``.  This keeps ``active_pools_fn`` accurate in
    headless / ``--quiet`` mode where the Rich display ticker is absent.
    Without it, the ``not active_pools`` bypass in ``BudgetManager.pool_admit``
    admits all pools unconditionally, allowing cheap pools to exhaust the
    budget before slower-but-higher-priority pools (e.g. enrich) can claim
    any work.

    The caller is responsible for finalizing the SNRun row after this
    function returns.
    """

    def active_pools_fn() -> set[str]:
        # A pool is "active" if its claim queue has pending work AND it has
        # not been repeatedly admitted but returned nothing from claim.
        # ``consecutive_empty_claims`` tracks how many times in a row a pool
        # was admitted but claim returned None — this happens when the display
        # pending query and the claim eligibility query have different criteria.
        # Excluding such pools from the denominator prevents a fairness deadlock
        # where productive pools can never be admitted because pools with phantom
        # pending counts occupy weight share.
        # The counter resets on any successful claim, so pools re-enter when
        # conditions change (e.g. after a stale-claim timeout expires).
        #
        # Self-healing: if pending_count has grown since the last evaluation,
        # new eligible nodes appeared in the graph (e.g. the generate pool just
        # produced names that are now enrich-eligible).  Reset
        # consecutive_empty_claims so the pool can re-enter contention
        # immediately rather than waiting for a stale-claim timeout.
        result: set[str] = set()
        for p in pools:
            current = p.health.pending_count
            if (
                current > p.health._last_pending_count
                and p.health.consecutive_empty_claims > 0
            ):
                logger.debug(
                    "pool[%s] pending grew %d→%d — resetting consecutive_empty_claims",
                    p.name,
                    p.health._last_pending_count,
                    current,
                )
                p.health.consecutive_empty_claims = 0
            p.health._last_pending_count = current
            if (
                current > 0
                and p.health.consecutive_empty_claims < _EMPTY_CLAIM_EXCLUDE_THRESHOLD
            ):
                result.add(p.name)
        return result

    tasks = [
        asyncio.create_task(
            pool_loop(
                p,
                mgr,
                stop_event,
                active_pools_fn=active_pools_fn,
                weights=weights,
            ),
            name=f"pool[{p.name}]",
        )
        for p in pools
    ]

    watchdog_task = asyncio.create_task(
        _budget_watchdog(mgr, stop_event),
        name="budget_watchdog",
    )

    # Optional: pending-count watchdog for headless / --quiet mode.
    pending_watchdog_task: asyncio.Task[None] | None = None
    if pending_fn is not None:
        pending_watchdog_task = asyncio.create_task(
            _pending_count_watchdog(
                pools, stop_event, pending_fn, pending_poll_interval
            ),
            name="pending_count_watchdog",
        )

    # Wait until stop_event is set OR all pools complete naturally.
    stop_waiter = asyncio.create_task(stop_event.wait(), name="stop_waiter")
    done, pending = await asyncio.wait(
        tasks + [stop_waiter],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # If only stop_waiter completed, give pools a grace period.
    if stop_waiter in done:
        logger.info(
            "stop signal received — granting %.0fs grace for in-flight batches",
            grace_period,
        )
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=grace_period)
        except TimeoutError:
            logger.warning("grace period expired — cancelling stragglers")
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    else:
        stop_waiter.cancel()
        await asyncio.gather(stop_waiter, return_exceptions=True)
        # If a pool exited with an exception, surface it after others finish.
        await asyncio.gather(*tasks, return_exceptions=True)

    # Cancel watchdog (it may already be done if it triggered the shutdown).
    if not watchdog_task.done():
        watchdog_task.cancel()
    await asyncio.gather(watchdog_task, return_exceptions=True)

    if pending_watchdog_task is not None:
        if not pending_watchdog_task.done():
            pending_watchdog_task.cancel()
        await asyncio.gather(pending_watchdog_task, return_exceptions=True)

    # Drain LLMCost queue before final accounting.
    await mgr.drain_pending()

    return {p.name: p.health for p in pools}


__all__ = [
    "POOL_NAMES",
    "POOL_WEIGHTS",
    "PoolHealth",
    "PoolSpec",
    "_pending_count_watchdog",
    "pool_loop",
    "run_pools",
]
