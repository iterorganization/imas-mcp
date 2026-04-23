"""Lease-style budget manager for LLM batch pipelines.

Provides a shared ``BudgetManager`` that tracks reserved, spent, and
available budget.  Callers acquire a ``BudgetLease`` via ``reserve()``,
``charge()`` actual costs against it, and release unused headroom on
completion.

Invariant: ``pool + sum(active_reserved) + spent == total``

Thread-safe: ``threading.Lock`` protects all mutations.  The lock
critical sections are pure arithmetic (no I/O), so blocking is
negligible even in async contexts.
"""

from __future__ import annotations

import logging
import threading
import uuid

logger = logging.getLogger(__name__)

EPSILON = 1e-9


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

    __slots__ = ("_mgr", "_reserved", "_lease_id", "_charged", "_released")

    def __init__(self, manager: BudgetManager, reserved: float, lease_id: str) -> None:
        self._mgr = manager
        self._reserved = reserved
        self._lease_id = lease_id
        self._charged = 0.0
        self._released = False

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

    # ------------------------------------------------------------------
    # Charge / release
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
            f"BudgetLease(id={self._lease_id!r}, reserved={self._reserved:.4f}, "
            f"charged={self._charged:.4f}, released={self._released})"
        )


class BudgetManager:
    """Concurrent-safe budget manager with lease-based tracking.

    Usage::

        mgr = BudgetManager(total_budget=5.0)

        lease = mgr.reserve(0.50)
        if lease is None:
            return  # budget exhausted

        with lease:
            cost = await call_llm(...)
            lease.charge(cost)
        # Unused portion auto-released

    Invariant: ``pool + sum(active_reserved) + spent == total``
    """

    def __init__(self, total_budget: float) -> None:
        self._total = total_budget
        self._pool = total_budget
        self._reserved: dict[str, float] = {}  # lease_id → remaining reservation
        self._spent = 0.0
        self._batch_count = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Reserve
    # ------------------------------------------------------------------

    def reserve(self, amount: float) -> BudgetLease | None:
        """Atomically reserve *amount* from the pool.

        Returns a :class:`BudgetLease` on success, ``None`` if the pool
        has insufficient funds.
        """
        with self._lock:
            if self._pool < amount - EPSILON:
                return None
            lease_id = str(uuid.uuid4())
            self._pool -= amount
            self._reserved[lease_id] = amount
            self._batch_count += 1
            return BudgetLease(self, amount, lease_id)

    # ------------------------------------------------------------------
    # Internal helpers (called by BudgetLease)
    # ------------------------------------------------------------------

    def _record_spend(self, lease_id: str, amount: float) -> None:
        """Record actual spend from a lease.

        Decrements the reservation's remaining balance and increments
        the manager-wide spent counter.
        """
        with self._lock:
            self._spent += amount
            if lease_id in self._reserved:
                self._reserved[lease_id] -= amount

    def _release(self, lease_id: str, unused: float) -> None:
        """Return unused reservation back to the pool."""
        with self._lock:
            self._pool += unused
            self._reserved.pop(lease_id, None)

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
        """Total spend recorded across all leases."""
        with self._lock:
            return self._spent

    @property
    def total_budget(self) -> float:
        """Original total budget."""
        return self._total

    def exhausted(self) -> bool:
        """Return ``True`` when the pool is non-positive."""
        with self._lock:
            return self._pool <= EPSILON

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def summary(self) -> dict:
        """Snapshot of budget state for logging / display."""
        with self._lock:
            return {
                "total_budget": self._total,
                "remaining": self._pool,
                "total_spent": self._spent,
                "active_reservations": len(self._reserved),
                "total_reserved": sum(self._reserved.values()),
                "batch_count": self._batch_count,
            }

    def check_invariant(self) -> bool:
        """Verify pool + sum(active_reserved) + spent == total."""
        with self._lock:
            expected = self._total
            actual = self._pool + sum(self._reserved.values()) + self._spent
            return abs(expected - actual) < EPSILON * 100
