"""Tests for lease-style budget in the review pipeline.

Verifies that both primary and secondary reviewer costs are tracked
through the lease, and that leases are released on completion.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.standard_names.budget import BudgetManager


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
            lease.charge(primary_cost)

            # Secondary review: costs $0.015
            secondary_cost = 0.015
            lease.charge(secondary_cost)

        # Both costs tracked
        total_charged = primary_cost + secondary_cost
        assert abs(mgr.spent - total_charged) < 1e-9

        # Unused headroom returned to pool
        expected_pool = 5.0 - total_charged
        assert abs(mgr.remaining - expected_pool) < 1e-9
        assert mgr.check_invariant()

    def test_secondary_reviewer_cost_stays_within_reservation(self):
        """When secondary reviewer costs push over reservation, BudgetExceeded fires."""
        from imas_codex.standard_names.budget import BudgetExceeded

        mgr = BudgetManager(total_budget=5.0)

        # Reserve just enough for 1 model
        lease = mgr.reserve(0.03)
        assert lease is not None

        with lease:
            # Primary review at near-limit
            lease.charge(0.025)

            # Secondary reviewer would overshoot
            with pytest.raises(BudgetExceeded):
                lease.charge(0.010)  # 0.025 + 0.010 = 0.035 > 0.030

        assert mgr.check_invariant()

    def test_lease_released_on_successful_completion(self):
        """Successful batch releases unused budget back to the pool."""
        mgr = BudgetManager(total_budget=1.0)

        lease = mgr.reserve(0.5)
        assert lease is not None

        with lease:
            lease.charge(0.02)  # Primary
            lease.charge(0.02)  # Secondary

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
                lease.charge(0.02)  # Primary completed
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
                lease.charge(primary)
                lease.charge(secondary)
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
            lease1.charge(0.06)

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
                    lease.charge(0.04)

                    # Secondary
                    await asyncio.sleep(0.001)
                    lease.charge(0.03)

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


class TestReviewPipelineChargeOrExtendFix:
    """Regression tests for Bug 1: pipeline.py lease.charge → charge_or_extend.

    Before the fix, charge() raised BudgetExceeded when actual LLM cost
    exceeded the per-batch reservation estimate, silently discarding the
    completed LLM result while the cost was already billed.

    After the fix, charge_or_extend() extends the reservation from the
    manager pool if available, so overruns are handled gracefully.
    """

    def test_overrun_extends_not_raises(self):
        """When actual cost exceeds reservation, charge_or_extend borrows from pool.

        Simulates the Opus-4.6 scenario: reservation formula underestimates
        by ~30%, but the pool has headroom.
        """
        from imas_codex.standard_names.budget import BudgetExceeded

        mgr = BudgetManager(total_budget=5.0)

        # 15 names × $0.05 per name × 1.5× = $1.125 reservation
        estimated = 15 * 0.05
        worst_case = estimated * 1.5  # = 1.125
        lease = mgr.reserve(worst_case)
        assert lease is not None

        # Opus-4.6 actually cost 30% more than reserved
        actual_c0_cost = worst_case * 1.3  # ~$1.46

        # Old code: lease.charge(actual_c0_cost) → BudgetExceeded
        # New code: charge_or_extend extends from pool
        lease.charge_or_extend(actual_c0_cost)  # must NOT raise
        assert abs(lease.charged - actual_c0_cost) < 1e-9
        assert mgr.check_invariant()

    def test_overrun_raises_when_pool_also_exhausted(self):
        """charge_or_extend still raises if both reservation and pool are empty."""
        from imas_codex.standard_names.budget import BudgetExceeded

        mgr = BudgetManager(total_budget=0.5)
        lease = mgr.reserve(0.5)  # Pool now empty
        assert lease is not None

        # Both lease and pool are exhausted — must raise
        with pytest.raises(BudgetExceeded):
            lease.charge_or_extend(0.6)

        assert mgr.check_invariant()

    def test_all_three_review_cycles_extend(self):
        """All three review cycles (c0, c1, c2) can overrun reservation.

        Models the 3-cycle RD-quorum pattern in pipeline.py where
        c0_cost, c1_cost, and c2_cost each call charge_or_extend.
        """
        mgr = BudgetManager(total_budget=5.0)

        # Reserve for 3 models × 15 names × $0.05 × 1.5
        worst_case = 3 * 15 * 0.05 * 1.5  # $3.375
        lease = mgr.reserve(worst_case)
        assert lease is not None

        # Each cycle costs slightly more than 1/3 of reservation
        c0_cost = worst_case / 3 * 1.2  # 20% over per-cycle share
        c1_cost = worst_case / 3 * 1.2
        c2_cost = worst_case / 3 * 0.8  # escalator only handles disputed

        # All three should succeed without raising
        with lease:
            lease.charge_or_extend(c0_cost)
            lease.charge_or_extend(c1_cost)
            lease.charge_or_extend(c2_cost)

        total = c0_cost + c1_cost + c2_cost
        assert abs(mgr.spent - total) < 1e-9
        assert mgr.check_invariant()
