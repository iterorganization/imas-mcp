"""Tests for soft-limit budget gate semantics (W34A).

Verifies that the SN loop uses a small per-unit-cost estimate
(``EST_UNIT_COST``) as the budget gate rather than the old $0.75
hard floor (``MIN_VIABLE_TURN``).

Key property: a rotation with budget comfortably above EST_UNIT_COST
must schedule work — even if budget < $0.75 (old floor).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.standard_names.loop import EST_UNIT_COST, run_sn_loop
from imas_codex.standard_names.turn import PhaseResult


@pytest.mark.asyncio
class TestBudgetSoftLimit:
    """Soft-limit gate: continue while remaining > EST_UNIT_COST."""

    async def test_budget_below_old_floor_still_runs(self):
        """Budget=$0.60 (below old $0.75 floor) must still schedule work.

        This is the regression test for the W34A fix: the old
        MIN_VIABLE_TURN=$0.75 floor prevented any work from starting
        with budgets in the $0.50–$0.74 range.
        """

        async def cheap_turn(cfg):
            return [
                PhaseResult(name="generate", count=2, cost=0.15),
                PhaseResult(name="review_names", count=2, cost=0.10),
            ]

        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                side_effect=[
                    {"domain": "equilibrium", "remaining": 5},
                    {"domain": "magnetics", "remaining": 3},
                    None,
                ],
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                side_effect=cheap_turn,
            ),
            patch("imas_codex.standard_names.loop._write_sn_run"),
        ):
            summary = await run_sn_loop(cost_limit=0.60, dry_run=False)

        # Work was done — NOT blocked by old floor.
        assert summary.names_composed == 4
        assert summary.names_reviewed == 4
        assert summary.cost_spent == pytest.approx(0.50)
        assert summary.stop_reason == "completed"

    async def test_budget_barely_above_est_unit_cost_runs_once(self):
        """Budget just above EST_UNIT_COST should run at least one turn."""
        budget = EST_UNIT_COST + 0.01  # e.g. $0.06

        async def small_turn(cfg):
            return [
                PhaseResult(name="generate", count=1, cost=budget),
            ]

        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                return_value={"domain": "equilibrium", "remaining": 3},
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                side_effect=small_turn,
            ),
            patch("imas_codex.standard_names.loop._write_sn_run"),
        ):
            summary = await run_sn_loop(cost_limit=budget, dry_run=False)

        assert summary.names_composed == 1
        assert summary.stop_reason == "budget_exhausted"

    async def test_budget_below_est_unit_cost_stops_immediately(self):
        """Budget below EST_UNIT_COST → immediate stop, no work scheduled."""
        mock_turn = AsyncMock()
        with (
            patch("imas_codex.standard_names.loop._write_sn_run"),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                mock_turn,
            ),
        ):
            summary = await run_sn_loop(cost_limit=0.03, dry_run=False)

        assert summary.stop_reason == "budget_exhausted"
        assert summary.cost_spent == 0.0
        mock_turn.assert_not_awaited()

    async def test_spends_to_budget_extent(self):
        """With $1.00 budget, loop should spend close to $1.00 — not $0.25.

        This is the live-evidence regression from W34A: a $1.00 run on
        current_drive exited after $0.253 due to the old $0.75 floor.
        """
        call_count = 0

        async def moderate_turn(cfg):
            nonlocal call_count
            call_count += 1
            return [
                PhaseResult(name="generate", count=2, cost=0.25),
                PhaseResult(name="review_names", count=2, cost=0.05),
            ]

        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                return_value={"domain": "current_drive", "remaining": 20},
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                side_effect=moderate_turn,
            ),
            patch("imas_codex.standard_names.loop._write_sn_run"),
        ):
            summary = await run_sn_loop(cost_limit=1.00, dry_run=False)

        # With $0.30/turn, we expect ~3 turns ($0.90) then remaining=$0.10
        # → 4th turn ($1.20 total, but remaining = -$0.20 by local tracker)
        # The key assertion: spent significantly more than $0.25 (old behavior).
        assert summary.cost_spent > 0.50
        assert call_count >= 3
        assert summary.stop_reason == "budget_exhausted"

    async def test_est_unit_cost_is_soft_not_hard(self):
        """EST_UNIT_COST should be much smaller than the old $0.75 floor."""
        assert EST_UNIT_COST < 0.75
        assert EST_UNIT_COST <= 0.10
        assert EST_UNIT_COST > 0.0
