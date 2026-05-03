"""Lease-style budget manager for LLM batch pipelines.

Provides a shared ``BudgetManager`` that tracks reserved, spent, and
available budget.  Callers acquire a ``BudgetLease`` via ``reserve()``,
charge actual costs against it via ``charge_event()``, and release
unused headroom on completion.

Invariant: ``pool + sum(active_reserved) + spent == total``

Thread-safe: ``threading.Lock`` protects all mutations.  The lock
critical sections are pure arithmetic (no I/O), so blocking is
negligible even in async contexts.

**Graph-backed cost tracking:**

``BudgetManager`` delegates spend recording to an async write queue
that persists ``LLMCost`` rows in Neo4j via ``record_llm_cost``.
In-memory ``_spent`` / ``_phase_spent`` are maintained as a local
cache for low-latency lease decisions.  The graph is the source of
truth; the in-memory counters are a write-ahead shadow.

The ``LLMCostEvent`` dataclass carries per-call metadata (model,
tokens, sn_ids, phase, etc.).  ``BudgetLease.charge_event()`` is the
single typed entry point that enqueues a graph write and updates the
local cache.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

EPSILON = 1e-9

# DEPRECATED (Phase C): MIN_VIABLE_TURN is no longer used in runtime
# shutdown logic.  Retained only for backward-compatible test fixtures
# that exercise :meth:`BudgetManager.near_exhausted` as a unit method.
# The pool orchestrator now uses a signal-driven ``budget_saturated``
# watchdog instead of a magic dollar threshold.
MIN_VIABLE_TURN: float = 0.75

# Writer retry parameters (matches retry_on_deadlock defaults)
_WRITER_MAX_RETRIES = 5
_WRITER_BASE_DELAY = 0.1
_WRITER_MAX_DELAY = 2.0

# Per-attempt timeout for record_llm_cost (run in a thread).
# Chosen to sit just above the worst-case retry_on_deadlock window
# (5 attempts × ~3s = 15s) so we don't cut into natural retries.
_WRITER_CALL_TIMEOUT = 20.0

# Heartbeat interval: how long the writer waits on an empty queue
# before emitting a health log line.
_WRITER_HEARTBEAT_SEC = 60.0


# =====================================================================
# LLMCostEvent — typed metadata for a single LLM call
# =====================================================================


@dataclass(frozen=True, slots=True)
class LLMCostEvent:
    """Per-call metadata for an LLM invocation.

    Carried alongside the dollar ``cost`` through ``charge_event`` to
    the async writer queue, which persists it as an ``LLMCost`` node.
    """

    model: str
    tokens_in: int
    tokens_out: int
    tokens_cached_read: int = 0
    tokens_cached_write: int = 0
    sn_ids: tuple[str, ...] = ()
    batch_id: str | None = None
    """Generic correlation id linking related ``LLMCost`` rows.

    Two callers stamp this field today:

    - **B2 grammar-retry** (``workers.py``): writes
      ``f"{group_key}-grammar-retry"`` so the original-vs-retry pair
      can be joined in analytics.
    - **Plan 39 structured fan-out** (``fanout/dispatcher.py``):
      writes the ``fanout_run_id`` (uuid4) onto the proposer charge,
      and call-sites stamp the same id onto their synthesizer charge,
      enabling the ``Fanout`` ↔ ``LLMCost`` join (plan 39 §8.3).

    The field is intentionally unstructured — callers pick the
    encoding that suits their analytics query.
    """
    cycle: str | None = None  # e.g. "c0", "c1", "c2"
    phase: str = ""  # generate|regen|enrich|review_names|review_docs
    service: str = "standard-names"
    llm_at: datetime | None = None  # defaults to now(UTC) at write time


# =====================================================================
# ChargeResult — returned by charge_event
# =====================================================================


@dataclass(frozen=True, slots=True)
class ChargeResult:
    """Result of a ``charge_event`` call."""

    overspend: float = 0.0
    """Amount charged beyond reserved+pool (0.0 if within budget)."""
    hard_stop: bool = False
    """True if the charge was rejected because budget is exhausted."""


# =====================================================================
# _PendingWrite — internal queue item
# =====================================================================


@dataclass(frozen=True, slots=True)
class _PendingWrite:
    """Enqueued graph write for the async writer."""

    cost: float
    event: LLMCostEvent
    overspend: float
    run_id: str
    llm_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class BudgetExceeded(RuntimeError):
    """Raised when a lease charge exceeds the reserved amount."""


class BudgetLease:
    """A bounded spending grant from a :class:`BudgetManager`.

    Tracks charges against a reserved amount.  Use as a context manager
    for automatic release of unused budget::

        with mgr.reserve(0.5) as lease:
            result = lease.charge_event(0.3, LLMCostEvent(...))
        # Remaining 0.2 released to pool
    """

    __slots__ = ("_mgr", "_reserved", "_lease_id", "_charged", "_released", "_phase")

    def __init__(
        self,
        manager: BudgetManager,
        reserved: float,
        lease_id: str,
        phase: str = "",
    ) -> None:
        self._mgr = manager
        self._reserved = reserved
        self._lease_id = lease_id
        self._charged = 0.0
        self._released = False
        self._phase = phase

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def reserved(self) -> float:
        """Original reservation amount."""
        return self._reserved

    @property
    def charged(self) -> float:
        """Cumulative spend charged so far."""
        return self._charged

    @property
    def remaining(self) -> float:
        """Unspent portion of the reservation."""
        return self._reserved - self._charged

    @property
    def lease_id(self) -> str:
        return self._lease_id

    @property
    def phase(self) -> str:
        """Phase tag this lease is attributed to (empty string if untagged)."""
        return self._phase

    # ------------------------------------------------------------------
    # New typed charge API (Phase 3)
    # ------------------------------------------------------------------

    def charge_event(self, cost: float, event: LLMCostEvent) -> ChargeResult:
        """Atomic charge: record spend + enqueue an ``LLMCost`` graph write.

        Extends the reservation from the pool when needed (soft-charge
        semantics — never raises ``BudgetExceeded``).  Returns a
        :class:`ChargeResult` with overspend information.

        This is the preferred entry point for instrumented call-sites.
        Legacy ``charge_soft`` / ``charge_or_extend`` are thin wrappers
        around this method without metadata (kept for Phase 4 migration).
        """
        if cost < 0:
            raise ValueError("charge must be non-negative")
        shortfall = (self._charged + cost) - self._reserved
        if shortfall > EPSILON:
            extended = self._mgr._extend_reservation(self._lease_id, shortfall)
            self._reserved += extended
        # Record spend unconditionally — the LLM has already been paid.
        self._charged += cost
        self._mgr._record_spend(self._lease_id, cost)
        overspend = max(self._charged - self._reserved, 0.0)
        # Enqueue async graph write
        self._mgr._enqueue_write(cost, event, overspend)
        return ChargeResult(overspend=overspend)

    def release_unused(self) -> float:
        """Return unspent portion to manager pool.  Idempotent."""
        if self._released:
            return 0.0
        unused = max(self._reserved - self._charged, 0.0)
        self._mgr._release(self._lease_id, unused)
        self._released = True
        return unused

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> BudgetLease:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release_unused()

    def __repr__(self) -> str:
        return (
            f"BudgetLease(id={self._lease_id!r}, phase={self._phase!r}, "
            f"reserved={self._reserved:.4f}, "
            f"charged={self._charged:.4f}, released={self._released})"
        )


class BudgetManager:
    """Concurrent-safe budget manager with lease-based tracking.

    Usage::

        mgr = BudgetManager(total_budget=5.0, run_id="run-001")

        lease = mgr.reserve(0.50)
        if lease is None:
            return  # budget exhausted

        with lease:
            cost = await call_llm(...)
            result = lease.charge_event(cost, LLMCostEvent(...))
        # Unused portion auto-released

    Invariant: ``pool + sum(active_reserved) + spent == total``

    When ``run_id`` is set, ``charge_event`` enqueues async graph writes
    via :func:`record_llm_cost`.  Call :meth:`start` before first use
    and :meth:`drain_pending` at shutdown.
    """

    def __init__(
        self,
        total_budget: float,
        phase_caps: dict[str, float] | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        self._total = total_budget
        self._pool = total_budget
        self._reserved: dict[str, float] = {}  # lease_id → remaining reservation
        self._spent = 0.0
        self._batch_count = 0
        self._lock = threading.Lock()
        # Per-phase hard caps.  Keys are phase names; values are the cap in
        # dollars.  Reservations that would push a phase's total committed
        # budget beyond cap × 1.5 are rejected.
        self._phase_caps: dict[str, float] = phase_caps or {}
        # Running total of amounts reserved for each tagged phase (cumulative;
        # not decremented on release so over-reservation is prevented even
        # after partial refunds).
        self._phase_committed: dict[str, float] = {}
        # Per-lease phase tag (for spend attribution).
        self._lease_phases: dict[str, str] = {}
        # Actual spend per phase tag (in-memory shadow — graph is SoT).
        self._phase_spent: dict[str, float] = {}

        # ── Graph-backed cost tracking (Phase 3) ──────────────────────
        self.run_id: str | None = run_id
        self._write_queue: asyncio.Queue[_PendingWrite | None] = asyncio.Queue()
        self._pending_cost: float = 0.0  # in-flight cost not yet flushed
        self._pending_lock = threading.Lock()
        self._writer_task: asyncio.Task[None] | None = None
        self._write_failed: bool = False
        self._write_dropped: int = 0
        self._started: bool = False
        # Cached graph total (refreshed at most once per second)
        self._graph_total_cache: float = 0.0
        self._graph_total_ts: float = 0.0  # monotonic timestamp of last fetch
        self._graph_cache_ttl: float = 1.0  # seconds

        # ── Budget-saturation tracking (Phase C) ─────────────────────
        # Per-pool consecutive reserve-failure counter.  Incremented each
        # time ``reserve()`` returns ``None`` for a given phase, reset to 0
        # on success.  When ALL tracked pools exceed
        # ``SATURATION_THRESHOLD`` simultaneously, the budget is too small
        # to fund any batch.
        self._consecutive_reserve_failures: dict[str, int] = {}
        self.SATURATION_THRESHOLD: int = 10

    # ------------------------------------------------------------------
    # Async lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background writer task.

        Called from harness setup.  Safe to call multiple times (idempotent).
        """
        if self._started:
            return
        self._started = True
        self._writer_task = asyncio.create_task(self._writer_loop())

    async def drain_pending(self, *, raise_on_failure: bool = False) -> bool:
        """Wait for the write queue to drain and the writer to finish.

        Returns ``True`` if everything was written, ``False`` if any write
        failed terminally.  Called from harness shutdown.

        When ``raise_on_failure`` is True, raises ``RuntimeError`` on
        terminal write failures instead of returning False.
        """
        # Send sentinel to tell writer loop to exit
        await self._write_queue.put(None)
        if self._writer_task is not None:
            try:
                await self._writer_task
            except Exception:  # noqa: BLE001
                logger.exception("Writer task raised during drain")
                self._write_failed = True

        # Surface dropped-write count so operators know telemetry is incomplete.
        if self._write_dropped > 0:
            logger.error(
                "drain_pending: %d LLMCost write(s) dropped after retry exhaustion",
                self._write_dropped,
            )

        if self._write_failed:
            logger.error(
                "drain_pending: _write_failed=True — one or more LLMCost "
                "writes failed terminally; cost_is_exact should be False"
            )

        success = not self._write_failed
        if not success and raise_on_failure:
            raise RuntimeError(
                "One or more LLMCost writes failed terminally; "
                "cost_is_exact should be set to False"
            )
        return success

    async def _writer_loop(self) -> None:
        """Pull events from the queue and call ``record_llm_cost`` for each.

        On transient errors, retries with exponential backoff.  On terminal
        failure (after retries exhausted), marks ``_write_failed = True`` and
        continues processing the queue (best-effort for remaining writes).

        A heartbeat fires every ``_WRITER_HEARTBEAT_SEC`` of idle time so
        operators can confirm the writer is alive.  INFO when there is
        pending work; DEBUG when idle and drained.
        """
        while True:
            try:
                item = await asyncio.wait_for(
                    self._write_queue.get(),
                    timeout=_WRITER_HEARTBEAT_SEC,
                )
            except TimeoutError:
                # Heartbeat — no items arrived within the window
                with self._pending_lock:
                    pc = self._pending_cost
                qs = self._write_queue.qsize()
                if pc > 0 or qs > 0:
                    logger.info(
                        "writer_loop heartbeat: pending=$%.4f qsize=%d",
                        pc,
                        qs,
                    )
                else:
                    logger.debug(
                        "writer_loop heartbeat: pending=$%.4f qsize=%d",
                        pc,
                        qs,
                    )
                continue

            if item is None:
                # Sentinel — drain complete
                self._write_queue.task_done()
                break
            try:
                await self._write_single(item)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Terminal write failure for LLMCost (run=%s, phase=%s, cost=%.6f)",
                    item.run_id,
                    item.event.phase,
                    item.cost,
                )
                self._write_failed = True
                self._write_dropped += 1
            finally:
                with self._pending_lock:
                    self._pending_cost -= item.cost
                self._write_queue.task_done()

    async def _write_single(self, item: _PendingWrite) -> None:
        """Write a single ``LLMCost`` to the graph with retry.

        Each attempt runs ``record_llm_cost`` in a thread via
        ``asyncio.to_thread`` with a per-attempt timeout so a wedged
        Neo4j connection cannot block the writer loop indefinitely.
        """
        from imas_codex.standard_names.graph_ops import record_llm_cost

        last_exc: Exception | None = None
        for attempt in range(_WRITER_MAX_RETRIES):
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        record_llm_cost,
                        run_id=item.run_id,
                        phase=item.event.phase,
                        cycle=item.event.cycle,
                        sn_ids=list(item.event.sn_ids) if item.event.sn_ids else None,
                        model=item.event.model,
                        cost=item.cost,
                        tokens_in=item.event.tokens_in,
                        tokens_out=item.event.tokens_out,
                        tokens_cached_read=item.event.tokens_cached_read,
                        tokens_cached_write=item.event.tokens_cached_write,
                        service=item.event.service,
                        batch_id=item.event.batch_id,
                        overspend=item.overspend,
                        llm_at=item.llm_at,
                    ),
                    timeout=_WRITER_CALL_TIMEOUT,
                )
                return  # success
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _WRITER_MAX_RETRIES - 1:
                    delay = min(_WRITER_BASE_DELAY * (2**attempt), _WRITER_MAX_DELAY)
                    logger.warning(
                        "Writer retry %d/%d for run=%s: %s",
                        attempt + 1,
                        _WRITER_MAX_RETRIES,
                        item.run_id,
                        exc,
                    )
                    await asyncio.sleep(delay)
        # All retries exhausted — propagate to caller
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Internal: enqueue a graph write
    # ------------------------------------------------------------------

    def _enqueue_write(
        self, cost: float, event: LLMCostEvent, overspend: float
    ) -> None:
        """Enqueue an ``LLMCost`` write (called from ``charge_event``).

        If no ``run_id`` is configured, the write is silently skipped
        (useful for tests without a graph).

        Detects a crashed/cancelled writer task and recreates it under
        ``_pending_lock`` to prevent TOCTOU double-recreation.
        """
        if self.run_id is None:
            return

        # Check writer health under lock (B3: TOCTOU-safe).
        with self._pending_lock:
            self._pending_cost += cost
            if self._writer_task is not None and self._writer_task.done():
                try:
                    exc = self._writer_task.exception()
                except asyncio.CancelledError:
                    exc = "cancelled"
                logger.error("writer_task died unexpectedly (%s) — recreating", exc)
                self._writer_task = asyncio.create_task(self._writer_loop())

        pw = _PendingWrite(
            cost=cost,
            event=event,
            overspend=overspend,
            run_id=self.run_id,
            llm_at=event.llm_at or datetime.now(UTC),
        )
        try:
            qsize = self._write_queue.qsize()
            if qsize > 256:
                logger.warning(
                    "Write queue backpressure: qsize=%d — persist workers may be "
                    "falling behind LLM throughput",
                    qsize,
                )
            self._write_queue.put_nowait(pw)
        except asyncio.QueueFull:  # pragma: no cover — unbounded queue
            logger.error("Write queue full — dropping LLMCost event")
            with self._pending_lock:
                self._pending_cost -= cost
            self._write_failed = True

    # ------------------------------------------------------------------
    # Reserve
    # ------------------------------------------------------------------

    def reserve(self, amount: float, phase: str = "") -> BudgetLease | None:
        """Atomically reserve *amount* from the pool.

        Returns a :class:`BudgetLease` on success, ``None`` if the pool
        has insufficient funds or the named *phase* would exceed its hard
        cap (``phase_caps[phase] × 1.5``).

        Also tracks consecutive reserve failures per *phase* for the
        ``budget_saturated`` shutdown signal.

        Args:
            amount: Amount to reserve.
            phase: Optional phase tag (e.g. ``"compose"``, ``"review_names"``).
                When a cap is configured for this phase, the reservation is
                rejected if it would push the phase's cumulative committed
                spend beyond ``cap × 1.5``.
        """
        with self._lock:
            # ── Per-phase cap check ────────────────────────────────────────
            if phase and phase in self._phase_caps:
                cap = self._phase_caps[phase]
                committed = self._phase_committed.get(phase, 0.0)
                if committed + amount > cap * 1.5 + EPSILON:
                    logger.debug(
                        "Phase %r cap exceeded: committed=%.4f + amount=%.4f"
                        " > cap*1.5=%.4f — reservation rejected",
                        phase,
                        committed,
                        amount,
                        cap * 1.5,
                    )
                    if phase:
                        self._consecutive_reserve_failures[phase] = (
                            self._consecutive_reserve_failures.get(phase, 0) + 1
                        )
                    return None
            # ── Global pool check ──────────────────────────────────────────
            if self._pool < amount - EPSILON:
                if phase:
                    self._consecutive_reserve_failures[phase] = (
                        self._consecutive_reserve_failures.get(phase, 0) + 1
                    )
                return None
            lease_id = str(uuid.uuid4())
            self._pool -= amount
            self._reserved[lease_id] = amount
            self._batch_count += 1
            if phase:
                self._phase_committed[phase] = (
                    self._phase_committed.get(phase, 0.0) + amount
                )
                # Reset failure counter on successful reservation.
                self._consecutive_reserve_failures[phase] = 0
            self._lease_phases[lease_id] = phase
            return BudgetLease(self, amount, lease_id, phase=phase)

    # ------------------------------------------------------------------
    # Internal helpers (called by BudgetLease)
    # ------------------------------------------------------------------

    def _record_spend(self, lease_id: str, amount: float) -> None:
        """Record actual spend from a lease.

        Decrements the reservation's remaining balance and increments
        the manager-wide spent counter.  Also tracks per-phase spend
        for diagnostic attribution.
        """
        with self._lock:
            self._spent += amount
            phase = self._lease_phases.get(lease_id, "")
            if phase:
                self._phase_spent[phase] = self._phase_spent.get(phase, 0.0) + amount
            if lease_id in self._reserved:
                self._reserved[lease_id] -= amount

    def _extend_reservation(self, lease_id: str, amount: float) -> float:
        """Atomically extend an active reservation by drawing from the pool.

        Returns the amount actually extended, which may be less than
        *amount* when the pool is insufficient or the lease's phase would
        exceed its hard cap (``phase_caps[phase] × 1.5``).  The caller is
        responsible for checking whether the extension was sufficient.

        Phase-cap enforcement on extension prevents a compose batch from
        draining the global pool past its allocated share via in-flight
        overshoot, which would starve downstream phases (review, regen).
        """
        with self._lock:
            extended = min(amount, self._pool)
            # ── Per-phase cap check on extension ───────────────────────────
            phase = self._lease_phases.get(lease_id, "")
            if extended > 0 and phase and phase in self._phase_caps:
                cap = self._phase_caps[phase]
                committed = self._phase_committed.get(phase, 0.0)
                room = cap * 1.5 - committed
                if room < extended:
                    if room <= EPSILON:
                        logger.debug(
                            "Phase %r cap exhausted on extension: "
                            "committed=%.4f cap*1.5=%.4f — extension refused",
                            phase,
                            committed,
                            cap * 1.5,
                        )
                        return 0.0
                    extended = room
            if extended > 0:
                self._pool -= extended
                if lease_id in self._reserved:
                    self._reserved[lease_id] += extended
                if phase:
                    self._phase_committed[phase] = (
                        self._phase_committed.get(phase, 0.0) + extended
                    )
                logger.info(
                    "budget: extended lease %s by $%.4f "
                    "(requested $%.4f, reservation now $%.4f, pool $%.4f)",
                    lease_id,
                    extended,
                    amount,
                    self._reserved.get(lease_id, 0.0),
                    self._pool,
                )
            return extended

    def _release(self, lease_id: str, unused: float) -> None:
        """Return unused reservation back to the pool."""
        with self._lock:
            self._pool += unused
            self._reserved.pop(lease_id, None)
            self._lease_phases.pop(lease_id, None)

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    @property
    def remaining(self) -> float:
        """Available pool (excludes active reservations)."""
        with self._lock:
            return self._pool

    @property
    def spent(self) -> float:
        """Total spend recorded across all leases (in-memory shadow)."""
        with self._lock:
            return self._spent

    @property
    def phase_spent(self) -> dict[str, float]:
        """Per-phase spend snapshot.  Keys are phase tags; values are USD."""
        with self._lock:
            return dict(self._phase_spent)

    @property
    def total_budget(self) -> float:
        """Original total budget."""
        return self._total

    def pool_spent_total(self, pool: str) -> float:
        """Return cumulative spend attributed to *pool* (via phase tag).

        Reads directly from ``_phase_spent`` — the same dict that
        :meth:`pool_admit` consults for fairness decisions.  Returns 0.0
        when the pool has never been charged.
        """
        with self._lock:
            return self._phase_spent.get(pool, 0.0)

    # ------------------------------------------------------------------
    # Pool admission control (Phase 8)
    # ------------------------------------------------------------------

    def pool_admit(
        self,
        pool: str,
        weights: dict[str, float],
        active_pools: set[str],
    ) -> bool:
        """Soft-fairness admission gate for a pool requesting a new batch.

        Implements the weighted-share rule described in plan.md Phase 8:

            share = pool_spent[p] / sum(pool_spent.values() or epsilon)
            effective_weight = weights[p] / sum(weights[q] for q in active_pools)
            admit iff share < effective_weight  OR  no other pool is active

        Idle pools (queue empty → not in ``active_pools``) immediately
        forfeit their weight share so active pools can borrow it.

        ``pool`` here is a logical pool name matching the keys in
        ``weights`` (e.g. ``"generate_name"``, ``"review_name"``,
        ``"refine_name"``, ``"generate_docs"``, ``"review_docs"``,
        ``"refine_docs"``).  These map 1:1 to ``_phase_spent`` keys.

        Returns True if the pool is permitted to claim its next batch.
        """
        if pool not in weights:
            return False
        # Hard gate: never admit when budget is exhausted, regardless of
        # active_pools state.  This prevents headless mode from bypassing the
        # cost cap (the bug that caused the 10.5× live smoke overshoot).
        if self.exhausted():
            return False
        # When active_pools is empty (e.g. headless/non-TTY mode where the
        # Rich display never updates pending_count, or at startup before the
        # first display refresh), admit all known pools unconditionally so
        # they can discover their own work via claim() and self-regulate via
        # backoff when claim() returns None.
        #
        # With ``pending_fn`` wired in ``run_pools`` (Phase 8 fix), this path
        # is only hit transiently: the ``_pending_count_watchdog`` updates
        # pending counts immediately on its first poll, so ``active_pools_fn``
        # returns a non-empty set before the pools have issued any claims.
        # Without ``pending_fn`` the bypass persists indefinitely, letting all
        # pools compete without fairness weighting.
        if not active_pools:
            return True
        if pool not in active_pools:
            # This pool has no pending work — forfeit its share.
            return False
        # Sole active pool always admitted — the "no other pool is
        # active" branch from plan.md Phase 8.
        if len(active_pools) == 1:
            return True
        with self._lock:
            spent = dict(self._phase_spent)
        total_spent = sum(spent.values())
        if total_spent < EPSILON:
            return True  # nothing spent yet; everyone gets a turn
        share = spent.get(pool, 0.0) / total_spent
        active_weight_sum = sum(
            weights.get(q, 0.0) for q in active_pools if q in weights
        )
        if active_weight_sum < EPSILON:
            return True
        effective = weights[pool] / active_weight_sum
        return share < effective

    @property
    def write_failed(self) -> bool:
        """True if any graph write failed terminally (cost_is_exact → False)."""
        return self._write_failed

    @property
    def pending_cost(self) -> float:
        """Cost enqueued but not yet flushed to the graph."""
        with self._pending_lock:
            return self._pending_cost

    def exhausted(self) -> bool:
        """Return ``True`` when the pool is non-positive."""
        with self._lock:
            return self._pool <= EPSILON

    def hard_exhausted(self) -> bool:
        """Return ``True`` when committed spend has reached the cost limit.

        Unlike :meth:`exhausted`, which triggers when the pool is drained
        (including by active reservations that may later be partially
        refunded), this predicate fires only when *actual spend* has
        consumed the budget.  Use for global-shutdown decisions (watchdog,
        final stop-reason) where a transient reservation spike should NOT
        terminate the run.
        """
        with self._lock:
            return self._spent >= self._total - EPSILON

    def near_exhausted(self, min_remaining: float = MIN_VIABLE_TURN) -> bool:
        """Return ``True`` when the remaining budget is below *min_remaining*.

        Implements the "remaining budget < MIN_VIABLE_TURN" stop check
        described in ``AGENTS.md``.  A run at, say, $2.97 of $3.00 has
        only $0.03 left — well below the cost of one LLM call — so it
        should finalize rather than spin admitting cheap-but-fruitless
        turns.

        The check is based on committed spend (``_spent``), matching
        :meth:`hard_exhausted`, so transient reservation spikes do not
        cause premature shutdown.
        """
        with self._lock:
            return (self._total - self._spent) < (min_remaining - EPSILON)

    def all_pools_budget_saturated(
        self,
        pool_names: tuple[str, ...] = (
            "generate_name",
            "review_name",
            "refine_name",
            "generate_docs",
            "review_docs",
            "refine_docs",
        ),
    ) -> bool:
        """Return ``True`` when all *pool_names* have exceeded the
        consecutive reserve-failure threshold.

        This is the signal-driven replacement for the old
        ``near_exhausted()`` shutdown gate.  Instead of comparing
        remaining budget against a magic dollar floor, we observe that
        every pool has failed to reserve budget ``SATURATION_THRESHOLD``
        times in a row — meaning the remaining budget cannot fund any
        batch in any pool.
        """
        with self._lock:
            return all(
                self._consecutive_reserve_failures.get(p, 0)
                >= self.SATURATION_THRESHOLD
                for p in pool_names
            )

    # ------------------------------------------------------------------
    # Graph-aware reads
    # ------------------------------------------------------------------

    async def get_total_spent(self, *, force_refresh: bool = False) -> float:
        """Return total spend for this run from graph + in-flight pending.

        Cached for ``_graph_cache_ttl`` seconds to avoid hammering Neo4j.
        Falls back to in-memory ``_spent`` when no ``run_id`` is set.
        """
        return self._get_total_spent_sync(force_refresh=force_refresh)

    def _get_total_spent_sync(self, *, force_refresh: bool = False) -> float:
        """Synchronous implementation of :meth:`get_total_spent`.

        Separated so callers in async shutdown paths can wrap this in
        ``asyncio.to_thread`` + ``wait_for`` without nesting coroutines.
        """
        if self.run_id is None:
            with self._lock:
                return self._spent

        now = time.monotonic()
        if force_refresh or (now - self._graph_total_ts) > self._graph_cache_ttl:
            try:
                from imas_codex.standard_names.graph_ops import (
                    aggregate_spend_for_run,
                )

                self._graph_total_cache = aggregate_spend_for_run(self.run_id)
                self._graph_total_ts = now
            except Exception:  # noqa: BLE001
                logger.debug("graph aggregate failed, using in-memory fallback")
                with self._lock:
                    return self._spent

        with self._pending_lock:
            return self._graph_total_cache + self._pending_cost

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def summary(self) -> dict[str, Any]:
        """Snapshot of budget state for logging / display."""
        with self._lock:
            return {
                "total_budget": self._total,
                "remaining": self._pool,
                "total_spent": self._spent,
                "active_reservations": len(self._reserved),
                "total_reserved": sum(self._reserved.values()),
                "batch_count": self._batch_count,
                "phase_committed": dict(self._phase_committed),
                "phase_spent": dict(self._phase_spent),
                "run_id": self.run_id,
                "write_failed": self._write_failed,
                "pending_writes": self._write_queue.qsize(),
            }

    def check_invariant(self) -> bool:
        """Verify pool + sum(active_reserved) + spent == total."""
        with self._lock:
            expected = self._total
            actual = self._pool + sum(self._reserved.values()) + self._spent
            return abs(expected - actual) < EPSILON * 100
