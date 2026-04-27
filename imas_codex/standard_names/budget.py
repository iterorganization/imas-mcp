"""Lease-style budget manager for LLM batch pipelines.

Provides a shared ``BudgetManager`` that tracks reserved, spent, and
available budget.  Callers acquire a ``BudgetLease`` via ``reserve()``,
``charge()`` actual costs against it, and release unused headroom on
completion.

Invariant: ``pool + sum(active_reserved) + spent == total``

Thread-safe: ``threading.Lock`` protects all mutations.  The lock
critical sections are pure arithmetic (no I/O), so blocking is
negligible even in async contexts.

**Graph-backed cost tracking (Phase 3):**

``BudgetManager`` delegates spend recording to an async write queue
that persists ``LLMCost`` rows in Neo4j via ``record_llm_cost``.
In-memory ``_spent`` / ``_phase_spent`` are maintained as a local
cache for low-latency lease decisions and backwards-compat property
access.  The graph is the source of truth; the in-memory counters
are a write-ahead shadow.

The ``LLMCostEvent`` dataclass carries per-call metadata (model,
tokens, sn_ids, phase, etc.).  ``BudgetLease.charge_event()`` is the
single typed entry point that enqueues a graph write and updates the
local cache.  Legacy ``charge_soft`` / ``charge_or_extend`` remain as
thin backwards-compat wrappers (Phase 4 will migrate callers).
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

# Writer retry parameters (matches retry_on_deadlock defaults)
_WRITER_MAX_RETRIES = 5
_WRITER_BASE_DELAY = 0.1
_WRITER_MAX_DELAY = 2.0


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
            lease.charge(0.2)
            lease.charge(0.1)
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

    # ------------------------------------------------------------------
    # Legacy charge API (backwards-compat — Phase 4 will remove)
    # ------------------------------------------------------------------

    def charge(self, amount: float) -> None:
        """Deduct *amount* from lease and record spend in manager.

        Raises ``BudgetExceeded`` if cumulative charges exceed the
        reserved amount (with floating-point epsilon tolerance).
        """
        if amount < 0:
            raise ValueError("charge must be non-negative")
        if self._charged + amount > self._reserved + EPSILON:
            raise BudgetExceeded(
                f"Lease {self._lease_id}: charge {amount:.4f} would exceed "
                f"reserved {self._reserved:.4f} "
                f"(already charged {self._charged:.4f})"
            )
        self._charged += amount
        self._mgr._record_spend(self._lease_id, amount)

    def charge_or_extend(self, amount: float) -> None:
        """Charge *amount*, extending the reservation from the pool if needed.

        Unlike :meth:`charge`, this method never raises ``BudgetExceeded``
        due to an estimation mismatch: when the actual cost exceeds the
        original reservation it atomically borrows the shortfall from the
        manager's pool before charging.  ``BudgetExceeded`` is only raised
        if the pool itself is also exhausted (i.e. truly no budget left).

        Use this in callers where the per-batch cost estimate may be
        imprecise (e.g. LLM compose) so that a slight under-estimate does
        not discard a completed LLM result.

        Raises:
            ValueError: if *amount* is negative.
            BudgetExceeded: if neither reservation nor pool can cover
                the charge.
        """
        if amount < 0:
            raise ValueError("charge must be non-negative")
        shortfall = (self._charged + amount) - self._reserved
        if shortfall > EPSILON:
            # Borrow shortfall from pool to extend this reservation
            extended = self._mgr._extend_reservation(self._lease_id, shortfall)
            self._reserved += extended
            if self._charged + amount > self._reserved + EPSILON:
                raise BudgetExceeded(
                    f"Lease {self._lease_id}: charge {amount:.4f} would exceed "
                    f"reservation+pool (reserved={self._reserved:.4f}, "
                    f"already charged={self._charged:.4f})"
                )
        self._charged += amount
        self._mgr._record_spend(self._lease_id, amount)

    def charge_soft(self, amount: float) -> float:
        """Record *amount* as spend, extending reservation+pool as needed.

        Unlike :meth:`charge` and :meth:`charge_or_extend`, this method
        NEVER raises ``BudgetExceeded``.  The LLM cost has already been
        incurred by the caller, so spend is always recorded.

        Behaviour:
          1. If the charge fits within the current reservation, record it.
          2. Otherwise, try to extend the reservation from the pool.
          3. If the pool is insufficient, record the overspend anyway —
             ``manager.spent`` will exceed ``manager.total_budget``.
             The caller is expected to log a warning.

        Returns:
            overspend: amount charged beyond reserved+pool (0.0 if fit).
        """
        if amount < 0:
            raise ValueError("charge must be non-negative")
        shortfall = (self._charged + amount) - self._reserved
        if shortfall > EPSILON:
            extended = self._mgr._extend_reservation(self._lease_id, shortfall)
            self._reserved += extended
        # Record spend unconditionally — the LLM has already been paid.
        self._charged += amount
        self._mgr._record_spend(self._lease_id, amount)
        # Report overspend (reservation may now be < charged if pool empty)
        overspend = max(self._charged - self._reserved, 0.0)
        return overspend

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
        self._started: bool = False
        # Cached graph total (refreshed at most once per second)
        self._graph_total_cache: float = 0.0
        self._graph_total_ts: float = 0.0  # monotonic timestamp of last fetch
        self._graph_cache_ttl: float = 1.0  # seconds

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
        """
        while True:
            item = await self._write_queue.get()
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
            finally:
                with self._pending_lock:
                    self._pending_cost -= item.cost
                self._write_queue.task_done()

    async def _write_single(self, item: _PendingWrite) -> None:
        """Write a single ``LLMCost`` to the graph with retry."""
        from imas_codex.standard_names.graph_ops import record_llm_cost

        last_exc: Exception | None = None
        for attempt in range(_WRITER_MAX_RETRIES):
            try:
                record_llm_cost(
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
                )
                return  # success
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _WRITER_MAX_RETRIES - 1:
                    delay = min(_WRITER_BASE_DELAY * (2**attempt), _WRITER_MAX_DELAY)
                    logger.debug(
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
        """
        if self.run_id is None:
            return
        with self._pending_lock:
            self._pending_cost += cost
        pw = _PendingWrite(
            cost=cost,
            event=event,
            overspend=overspend,
            run_id=self.run_id,
            llm_at=event.llm_at or datetime.now(UTC),
        )
        try:
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
                    return None
            # ── Global pool check ──────────────────────────────────────────
            if self._pool < amount - EPSILON:
                return None
            lease_id = str(uuid.uuid4())
            self._pool -= amount
            self._reserved[lease_id] = amount
            self._batch_count += 1
            if phase:
                self._phase_committed[phase] = (
                    self._phase_committed.get(phase, 0.0) + amount
                )
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

    # ------------------------------------------------------------------
    # Graph-aware reads
    # ------------------------------------------------------------------

    async def get_total_spent(self, *, force_refresh: bool = False) -> float:
        """Return total spend for this run from graph + in-flight pending.

        Cached for ``_graph_cache_ttl`` seconds to avoid hammering Neo4j.
        Falls back to in-memory ``_spent`` when no ``run_id`` is set.
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
            }

    def check_invariant(self) -> bool:
        """Verify pool + sum(active_reserved) + spent == total."""
        with self._lock:
            expected = self._total
            actual = self._pool + sum(self._reserved.values()) + self._spent
            return abs(expected - actual) < EPSILON * 100
