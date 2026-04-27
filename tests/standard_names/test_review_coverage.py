"""Tests for review coverage — ensures the review pipeline reviews all eligible
names when the budget allows, not just a tiny fraction.

Regression test for the EMW pilot bug where 220 valid StandardNames were
produced but only 3 got reviewed (1.4% coverage).  Root cause: two
interacting issues:

1. **Over-conservative reservation**: ``worst_case = estimated_cost *
   len(models) * 3.0`` produced a 9.0× multiplier for 3-model RD-quorum,
   making the per-batch reservation ($6.75 for 15 names) larger than the
   entire phase budget ($3.00 = 15% of $20).

2. **Fixed 15% budget split**: Review phases got a fixed 15% of the turn
   budget regardless of how much prior phases (generate/enrich) actually
   spent.  When generate spent $0 (all names already composed), the unspent
   budget was wasted.

Fix: (a) reduce the per-model reservation multiplier from 3.0 to 1.5,
(b) compute review budgets adaptively from remaining budget after prior phases.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.budget import BudgetManager, LLMCostEvent


def _ce(lease, amount, phase=None):
    """Simulate LLM spend via charge_event (soft semantics, never raises)."""
    evt_phase = phase or lease.phase or "test"
    return lease.charge_event(
        amount,
        LLMCostEvent(model="test-model", tokens_in=0, tokens_out=0, phase=evt_phase),
    )


# ═══════════════════════════════════════════════════════════════════════
# Unit test: budget reservation feasibility
# ═══════════════════════════════════════════════════════════════════════


class TestReviewBudgetReservationFeasibility:
    """Verify that standard review batches can obtain budget reservations
    under realistic phase budgets.

    The EMW pilot bug: with 3 models and a 3.0× per-model multiplier,
    the worst-case reservation for a 15-name batch was $6.75, but the
    review phase budget was only $3.00 (15% of $20).  This caused
    ``BudgetManager.reserve()`` to return ``None`` for every batch.
    """

    @pytest.mark.parametrize(
        "num_models,batch_size,phase_budget",
        [
            (1, 15, 3.00),  # Single model, standard budget
            (2, 15, 3.00),  # Dual model, standard budget
            (3, 15, 7.00),  # 3-model quorum, adaptive budget
            (3, 10, 3.00),  # 3-model quorum, smaller batch/budget
            (3, 15, 8.00),  # 3-model quorum, typical adaptive (40% of $20)
        ],
    )
    def test_first_batch_can_always_reserve(
        self, num_models: int, batch_size: int, phase_budget: float
    ):
        """At least ONE review batch must be able to reserve budget
        under any realistic (models × budget) configuration."""
        mgr = BudgetManager(total_budget=phase_budget)

        # Reproduce the reservation logic from review_review_worker
        estimated_cost = batch_size * 0.05
        # This is the formula we're fixing — 1.5 per model instead of 3.0.
        worst_case = estimated_cost * num_models * 1.5

        lease = mgr.reserve(worst_case)
        assert lease is not None, (
            f"First batch reservation FAILED: worst_case=${worst_case:.2f} "
            f"> budget=${phase_budget:.2f} "
            f"(models={num_models}, batch={batch_size})"
        )
        lease.release_unused()

    def test_old_multiplier_blocks_3model_batch(self):
        """Demonstrate the bug: the old 3.0× per-model multiplier makes
        a 15-name batch unreservable with even a generous $5 budget."""
        mgr = BudgetManager(total_budget=5.00)

        estimated_cost = 15 * 0.05  # $0.75
        old_worst_case = estimated_cost * 3 * 3.0  # $6.75 — the bug

        lease = mgr.reserve(old_worst_case)
        assert lease is None, "Old multiplier should fail to reserve — this IS the bug"

    def test_fixed_multiplier_with_adaptive_budget(self):
        """With adaptive budget ($8.00 = 40% of $20 remaining after
        generate/enrich) and the fixed multiplier, multiple batches
        can be processed."""
        adaptive_budget = 8.00  # 40% of $20 when prior phases cost ~$0
        mgr = BudgetManager(total_budget=adaptive_budget)

        estimated_cost = 15 * 0.05  # $0.75
        worst_case = estimated_cost * 3 * 1.5  # $3.375

        batches_reserved = 0
        for _ in range(15):  # 15 batches = 225 names
            lease = mgr.reserve(worst_case)
            if lease is None:
                break
            # Simulate actual review cost (~$0.10/name)
            actual_cost = 15 * 0.10
            _ce(lease, actual_cost)
            lease.release_unused()
            batches_reserved += 1

        # With $8 budget, $3.375 reservation, $1.50 actual cost:
        # Sequential processing allows budget release to fund later batches.
        assert batches_reserved >= 3, f"Expected ≥3 batches but got {batches_reserved}"
        assert mgr.check_invariant()


# ═══════════════════════════════════════════════════════════════════════
# Turn-level test: adaptive budget computation
# ═══════════════════════════════════════════════════════════════════════


class TestAdaptiveTurnBudget:
    """Verify that the turn runner computes adaptive review budgets
    based on actual prior-phase spend, not a fixed 15% split."""

    def test_review_gets_more_when_generate_is_cheap(self):
        """When generate spends $0 (all names already composed), the
        review phase should get ~40% of the FULL turn budget, not 15%."""
        from imas_codex.standard_names.turn import (
            PhaseResult,
            _adaptive_review_budget,
        )

        # Simulate prior phases costing nothing
        prior_results = [
            PhaseResult(name="reconcile", cost=0.0),
            PhaseResult(name="generate", cost=0.0),
            PhaseResult(name="enrich", cost=0.0),
            PhaseResult(name="link", cost=0.0),
        ]

        budget = _adaptive_review_budget(20.0, prior_results, "review_names")
        # Should be ~40% of $20 = $8.00, NOT 15% = $3.00
        assert budget >= 7.0, (
            f"review_names budget ${budget:.2f} is too small when prior phases cost $0"
        )

    def test_review_respects_prior_spend(self):
        """When generate spends $6, review gets 40% of the remaining $14."""
        from imas_codex.standard_names.turn import (
            PhaseResult,
            _adaptive_review_budget,
        )

        prior_results = [
            PhaseResult(name="reconcile", cost=0.0),
            PhaseResult(name="generate", cost=6.0),
            PhaseResult(name="enrich", cost=2.0),
            PhaseResult(name="link", cost=0.0),
        ]

        budget = _adaptive_review_budget(20.0, prior_results, "review_names")
        # $12 remaining; review_names should get ~40-50% of that
        assert 4.0 <= budget <= 8.0, (
            f"review_names budget ${budget:.2f} out of expected range"
        )

    def test_skipped_phases_dont_consume_budget(self):
        """Skipped phases (cost=0, skipped=True) don't reduce the budget
        available for review."""
        from imas_codex.standard_names.turn import (
            PhaseResult,
            _adaptive_review_budget,
        )

        prior_results = [
            PhaseResult(name="reconcile", cost=0.0, skipped=True),
            PhaseResult(name="generate", cost=0.0, skipped=True),
            PhaseResult(name="enrich", cost=0.0, skipped=True),
            PhaseResult(name="link", cost=0.0, skipped=True),
        ]

        budget = _adaptive_review_budget(20.0, prior_results, "review_names")
        assert budget >= 7.0


# ═══════════════════════════════════════════════════════════════════════
# Integration-level test: review phase coverage
# ═══════════════════════════════════════════════════════════════════════


class TestReviewPhaseCoverage:
    """End-to-end test that the review pipeline's budget reservation
    allows processing all eligible batches when budget permits.

    Uses direct budget simulation to avoid deep mocking of graph/LLM.
    """

    def test_review_covers_all_batches_with_adaptive_budget(self):
        """With adaptive budget and the fixed multiplier, the review
        pipeline can reserve budget for ALL batches in a turn."""
        # Scenario: 45 valid names (3 batches of 15), $20 total budget,
        # prior phases spent $0, review_names gets 45% of remaining = $9.
        total_names = 45
        batch_size = 15
        cost_limit = 20.0

        # Simulate adaptive budget: prior phases cost $0
        from imas_codex.standard_names.turn import PhaseResult, _adaptive_review_budget

        prior = [
            PhaseResult(name="reconcile", cost=0.0),
            PhaseResult(name="generate", cost=0.0),
            PhaseResult(name="enrich", cost=0.0),
            PhaseResult(name="link", cost=0.0),
        ]
        adaptive_budget = _adaptive_review_budget(cost_limit, prior, "review_names")

        # Budget simulation: sequential batch processing with lease release
        mgr = BudgetManager(total_budget=adaptive_budget)
        num_models = 3
        batches_processed = 0
        names_reviewed = 0

        num_batches = (total_names + batch_size - 1) // batch_size
        for i in range(num_batches):
            current_batch_size = min(batch_size, total_names - i * batch_size)
            estimated_cost = current_batch_size * 0.05
            worst_case = estimated_cost * num_models * 1.5  # FIXED multiplier

            lease = mgr.reserve(worst_case)
            if lease is None:
                break

            # Simulate actual cost: ~$0.10/name (observed in EMW pilot)
            actual_cost = current_batch_size * 0.10
            _ce(lease, actual_cost)
            lease.release_unused()

            batches_processed += 1
            names_reviewed += current_batch_size

        coverage = names_reviewed / total_names
        assert coverage >= 0.95, (
            f"Review coverage {coverage:.1%} ({names_reviewed}/{total_names}) "
            f"is below 95% threshold with adaptive budget=${adaptive_budget:.2f}"
        )
        assert mgr.check_invariant()

    def test_old_config_reviews_almost_nothing(self):
        """Demonstrate the EMW pilot failure: with old multiplier (3.0×)
        and fixed 15% budget split, only tail batches slip through."""
        total_names = 220
        batch_size = 15
        fixed_budget = 20.0 * 0.15  # $3.00

        mgr = BudgetManager(total_budget=fixed_budget)
        num_models = 3
        names_reviewed = 0

        num_batches = (total_names + batch_size - 1) // batch_size
        for i in range(num_batches):
            current_batch_size = min(batch_size, total_names - i * batch_size)
            estimated_cost = current_batch_size * 0.05
            worst_case = estimated_cost * num_models * 3.0  # OLD multiplier

            lease = mgr.reserve(worst_case)
            if lease is None:
                continue  # Skip this batch (the bug!)

            actual_cost = current_batch_size * 0.10
            _ce(lease, actual_cost)
            lease.release_unused()
            names_reviewed += current_batch_size

        coverage = names_reviewed / total_names
        # The bug: only tail batches (≤6 names) fit, so coverage is tiny
        assert coverage < 0.10, (
            f"Old config unexpectedly reviewed {coverage:.1%} — "
            f"the bug reproduction may be wrong"
        )
