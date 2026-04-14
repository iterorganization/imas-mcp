"""Concurrent-safe budget manager for LLM review batches.

Tracks reserved vs actual spend so callers can reserve headroom before
issuing an LLM call and reconcile the difference in a ``finally`` block.
Thread-safe: multiple workers can reserve/reconcile concurrently.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class ReviewBudgetManager:
    """Concurrent-safe budget manager for LLM review batches.

    Usage::

        mgr = ReviewBudgetManager(total_budget=5.0)

        estimated = 0.10
        if not mgr.reserve(estimated):
            break  # budget exhausted

        reserved = estimated * 1.3
        try:
            actual = await call_llm(...)
        finally:
            mgr.reconcile(reserved, actual)
    """

    def __init__(self, total_budget: float) -> None:
        self.total_budget = total_budget
        self._remaining = total_budget
        self._lock = threading.Lock()
        self._total_reserved = 0.0
        self._total_actual = 0.0
        self._batch_count = 0

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    @property
    def remaining(self) -> float:
        with self._lock:
            return self._remaining

    # ------------------------------------------------------------------
    # Reserve / reconcile
    # ------------------------------------------------------------------

    def reserve(self, estimated_cost: float) -> bool:
        """Atomically reserve budget.  Returns ``True`` if reservation succeeded.

        Reserves 1.3× the *estimated_cost* to provide retry headroom.
        """
        with self._lock:
            # Reserve 1.3x estimated for retry headroom
            reservation = estimated_cost * 1.3
            if self._remaining < reservation:
                return False
            self._remaining -= reservation
            self._total_reserved += reservation
            self._batch_count += 1
            return True

    def reconcile(self, reserved: float, actual: float) -> None:
        """Return unused reservation.  **MUST** be called in ``finally`` blocks."""
        with self._lock:
            unused = reserved - actual
            if unused > 0:
                self._remaining += unused
            self._total_actual += actual

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def exhausted(self) -> bool:
        """Return ``True`` when remaining budget is non-positive."""
        with self._lock:
            return self._remaining <= 0

    @property
    def summary(self) -> dict:
        """Snapshot of budget state for logging / display."""
        with self._lock:
            return {
                "total_budget": self.total_budget,
                "remaining": self._remaining,
                "total_reserved": self._total_reserved,
                "total_actual": self._total_actual,
                "batch_count": self._batch_count,
            }
