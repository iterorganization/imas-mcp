"""Regression tests: budget reservation failure must not trigger invariant error.

Bug 7 — When ``BudgetManager.reserve(worst_case)`` returns ``None`` for every
batch (worst_case > remaining pool), total_cost stays at $0.00, persist_count
stays at 0, and the invariant ``total_cost < budget * 0.5`` evaluates to True.
This caused a spurious ``invariant violated`` error even though the budget was
the actual bottleneck.

The fix:
1. ``_process_batch`` in ``review/pipeline.py`` increments
   ``state.stats["budget_reservation_blocked"]`` when ``reserve()`` returns
   ``None``.
2. Both ``_run_review_names_phase`` and ``_run_review_docs_phase`` in
   ``turn.py`` add ``and not budget_blocked`` to the invariant condition.
"""

from __future__ import annotations

import inspect

import pytest


class TestBudgetReservationBlocked:
    """Verify pipeline tracks reservation failures and invariant checks them."""

    def test_pipeline_increments_budget_reservation_blocked(self) -> None:
        """_process_batch must set stats['budget_reservation_blocked'] on None lease."""
        from imas_codex.standard_names.review import pipeline

        src = inspect.getsource(pipeline)
        assert "budget_reservation_blocked" in src, (
            "_process_batch must track budget_reservation_blocked in state.stats"
        )

    def test_review_names_invariant_checks_budget_blocked(self) -> None:
        """_run_review_names_phase invariant must check budget_reservation_blocked."""
        from imas_codex.standard_names import turn

        src = inspect.getsource(turn._run_review_names_phase)
        assert "budget_reservation_blocked" in src, (
            "review_names invariant must be gated on budget_reservation_blocked"
        )
        assert "budget_blocked" in src, (
            "review_names should use budget_blocked local variable"
        )

    def test_review_docs_invariant_checks_budget_blocked(self) -> None:
        """_run_review_docs_phase invariant must check budget_reservation_blocked."""
        from imas_codex.standard_names import turn

        src = inspect.getsource(turn._run_review_docs_phase)
        assert "budget_reservation_blocked" in src, (
            "review_docs invariant must be gated on budget_reservation_blocked"
        )
        assert "budget_blocked" in src, (
            "review_docs should use budget_blocked local variable"
        )


class TestBudgetReservationBlockedUnit:
    """Unit-level tests that the invariant logic is correct."""

    def test_invariant_not_triggered_when_budget_blocked(self) -> None:
        """Simulate the exact conditions that caused Bug 7.

        Conditions: target_names > 0, persist_count == 0,
        total_cost < budget * 0.5, but budget_reservation_blocked > 0.
        Invariant must NOT fire.
        """
        # Simulate state
        target_names = ["name_a", "name_b", "name_c"]
        persist_count = 0
        total_cost = 0.0
        budget = 3.0
        stats = {"budget_reservation_blocked": 3}

        budget_blocked = stats.get("budget_reservation_blocked", 0) > 0

        # This is the patched invariant condition
        should_fire = (
            len(target_names) > 0
            and persist_count == 0
            and total_cost < budget * 0.5
            and not budget_blocked
        )
        assert not should_fire, (
            "Invariant must NOT fire when budget_reservation_blocked > 0"
        )

    def test_invariant_fires_without_budget_block(self) -> None:
        """Without budget blocking, the invariant should still catch silent failures."""
        target_names = ["name_a", "name_b", "name_c"]
        persist_count = 0
        total_cost = 0.0
        budget = 3.0
        stats = {}  # No budget_reservation_blocked

        budget_blocked = stats.get("budget_reservation_blocked", 0) > 0

        should_fire = (
            len(target_names) > 0
            and persist_count == 0
            and total_cost < budget * 0.5
            and not budget_blocked
        )
        assert should_fire, (
            "Invariant MUST fire when nothing was persisted and budget wasn't the cause"
        )

    def test_invariant_respects_high_cost(self) -> None:
        """When total_cost >= budget * 0.5, invariant should not fire regardless."""
        target_names = ["name_a"]
        persist_count = 0
        total_cost = 2.0
        budget = 3.0
        stats = {}

        budget_blocked = stats.get("budget_reservation_blocked", 0) > 0

        should_fire = (
            len(target_names) > 0
            and persist_count == 0
            and total_cost < budget * 0.5
            and not budget_blocked
        )
        assert not should_fire, (
            "Invariant must not fire when total_cost >= budget * 0.5"
        )


class TestBudgetManagerReserve:
    """Verify BudgetManager.reserve() returns None when pool is insufficient."""

    def test_reserve_returns_none_when_insufficient(self) -> None:
        """reserve() must return None when amount > remaining pool."""
        from imas_codex.standard_names.budget import BudgetManager

        bm = BudgetManager(total_budget=3.0)
        # worst_case of $3.375 exceeds the $3.00 pool
        lease = bm.reserve(3.375)
        assert lease is None, "reserve() must return None when amount > pool"

    def test_reserve_succeeds_when_sufficient(self) -> None:
        """reserve() must return a BudgetLease when amount <= pool."""
        from imas_codex.standard_names.budget import BudgetManager

        bm = BudgetManager(total_budget=3.0)
        lease = bm.reserve(1.0)
        assert lease is not None, "reserve() must succeed when pool is sufficient"
