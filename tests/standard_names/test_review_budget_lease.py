"""Tests for lease-style budget in the review pipeline.

Verifies that both primary and secondary reviewer costs are tracked
through the lease, and that leases are released on completion.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.standard_names.budget import BudgetManager, LLMCostEvent

# ── Test helper ───────────────────────────────────────────────────────


def _ce(lease, amount, phase=None):
    """Simulate LLM spend via charge_event (soft semantics, never raises)."""
    evt_phase = phase or lease.phase or "test"
    return lease.charge_event(
        amount,
        LLMCostEvent(model="test-model", tokens_in=0, tokens_out=0, phase=evt_phase),
    )


class TestReviewPipelineBudget:
    """Simulate the review pipeline's budget usage pattern."""

    def test_secondary_reviewer_cost_charged(self):
        """Bug A fix: 2-model setup charges both cycles to the same lease.

        Before the fix, the secondary reviewer cost was added to
        ``total_cost`` and ``state.review_stats.cost`` but NOT reconciled
        against the reservation, causing un-budgeted spend.
        """
        mgr = BudgetManager(total_budget=5.0)

        # Simulate _process_batch with 2 models
        num_names = 10
        num_models = 2
        per_name_cost = 0.002
        estimated = num_names * per_name_cost * num_models
        worst_case = estimated * 1.3

        lease = mgr.reserve(worst_case)
        assert lease is not None

        with lease:
            # Primary review: costs $0.018
            primary_cost = 0.018
            _ce(lease, primary_cost)

            # Secondary review: costs $0.015
            secondary_cost = 0.015
            _ce(lease, secondary_cost)

        # Both costs tracked
        total_charged = primary_cost + secondary_cost
        assert abs(mgr.spent - total_charged) < 1e-9

        # Unused headroom returned to pool
        expected_pool = 5.0 - total_charged
        assert abs(mgr.remaining - expected_pool) < 1e-9
        assert mgr.check_invariant()

    def test_lease_released_on_successful_completion(self):
        """Successful batch releases unused budget back to the pool."""
        mgr = BudgetManager(total_budget=1.0)

        lease = mgr.reserve(0.5)
        assert lease is not None

        with lease:
            _ce(lease, 0.02)  # Primary
            _ce(lease, 0.02)  # Secondary

        # 0.5 - 0.04 = 0.46 returned
        assert abs(mgr.remaining - 0.96) < 1e-9
        assert mgr.check_invariant()

    def test_lease_released_on_exception(self):
        """Failed batch still releases unused budget."""
        mgr = BudgetManager(total_budget=1.0)

        lease = mgr.reserve(0.5)
        assert lease is not None

        with pytest.raises(RuntimeError):
            with lease:
                _ce(lease, 0.02)  # Primary completed
                raise RuntimeError("Secondary reviewer crashed")

        # 0.5 - 0.02 = 0.48 returned
        assert abs(mgr.remaining - 0.98) < 1e-9
        assert abs(mgr.spent - 0.02) < 1e-9
        assert mgr.check_invariant()

    def test_multi_batch_budget_progression(self):
        """Multiple batches progressively consume and release budget."""
        mgr = BudgetManager(total_budget=1.0)

        batch_costs = [
            (0.03, 0.02),  # (primary, secondary)
            (0.04, 0.03),
            (0.02, 0.01),
        ]

        total_spent = 0.0
        for primary, secondary in batch_costs:
            worst_case = (primary + secondary) * 1.5  # generous reservation
            lease = mgr.reserve(worst_case)
            assert lease is not None, f"Should reserve for batch (pool={mgr.remaining})"

            with lease:
                _ce(lease, primary)
                _ce(lease, secondary)
                total_spent += primary + secondary

        assert abs(mgr.spent - total_spent) < 1e-9
        assert abs(mgr.remaining - (1.0 - total_spent)) < 1e-9
        assert mgr.check_invariant()

    def test_budget_stops_new_batches_after_exhaustion(self):
        """Once budget is exhausted, new batches get None from reserve."""
        mgr = BudgetManager(total_budget=0.10)

        # First batch succeeds
        lease1 = mgr.reserve(0.08)
        assert lease1 is not None
        with lease1:
            _ce(lease1, 0.06)

        # Second batch: only 0.04 remaining, needs 0.08
        lease2 = mgr.reserve(0.08)
        assert lease2 is None

        assert mgr.check_invariant()


class TestReviewBudgetAsync:
    """Async-specific review budget tests."""

    def test_concurrent_review_batches(self):
        """Multiple concurrent review batches with correct budget tracking."""

        async def _run():
            mgr = BudgetManager(total_budget=1.0)
            completed = 0

            async def _review_batch(batch_id: int):
                nonlocal completed
                lease = mgr.reserve(0.15)  # worst_case per batch
                if lease is None:
                    return

                try:
                    # Primary
                    await asyncio.sleep(0.001)
                    _ce(lease, 0.04)

                    # Secondary
                    await asyncio.sleep(0.001)
                    _ce(lease, 0.03)

                    completed += 1
                finally:
                    lease.release_unused()

            tasks = [_review_batch(i) for i in range(20)]
            await asyncio.gather(*tasks)

            # 1.0 / 0.15 = 6 batches can reserve
            assert completed == 6
            assert abs(mgr.spent - 6 * 0.07) < 1e-9
            assert mgr.check_invariant()

        asyncio.run(_run())
