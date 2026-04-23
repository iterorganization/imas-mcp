"""Tests for one-domain-per-turn rotation and turn-entry floor.

Plan 39, phase 5b: verifies that the SN loop uses stale-first rotation
to pick ONE domain per turn, gives it the full remaining budget, and
stops when budget drops below MIN_VIABLE_TURN.

Covers:
  - ``select_next_domain`` stale-first ordering
  - ``_pick_stalest_domain`` with null / dated SNRun.ended_at
  - Turn-entry floor enforcement
  - Loop iteration with domain rotation
  - Explicit ``--physics-domain`` bypass
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.loop import (
    MIN_VIABLE_TURN,
    _pick_stalest_domain,
    run_sn_loop,
    select_next_domain,
)
from imas_codex.standard_names.turn import PhaseResult

# ═══════════════════════════════════════════════════════════════════════
# _pick_stalest_domain (unit tests — mock GraphClient)
# ═══════════════════════════════════════════════════════════════════════


class TestPickStalestDomain:
    """Unit tests for the stalest-domain picker."""

    def test_single_candidate_returns_immediately(self):
        """With one candidate, skip the graph query entirely."""
        entry = {"domain": "equilibrium", "remaining": 10}
        assert _pick_stalest_domain([entry]) is entry

    def test_picks_oldest_last_run(self):
        """Domain with the oldest SNRun.ended_at is selected."""
        candidates = [
            {"domain": "equilibrium", "remaining": 5},
            {"domain": "magnetics", "remaining": 5},
            {"domain": "transport", "remaining": 5},
        ]
        # Mock: magnetics ran yesterday, equilibrium ran last week, transport ran today
        staleness_rows = [
            {"domain": "equilibrium", "last_run": datetime(2025, 1, 1, tzinfo=UTC)},
            {"domain": "magnetics", "last_run": datetime(2025, 1, 6, tzinfo=UTC)},
            {"domain": "transport", "last_run": datetime(2025, 1, 7, tzinfo=UTC)},
        ]
        mock_gc = MagicMock()
        mock_gc.query.return_value = staleness_rows
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.graph.client.GraphClient",
            return_value=mock_gc,
        ):
            winner = _pick_stalest_domain(candidates)

        assert winner["domain"] == "equilibrium"

    def test_prefers_unrun_domain(self):
        """A domain with no prior SNRun (null last_run) is selected first."""
        candidates = [
            {"domain": "equilibrium", "remaining": 5},
            {"domain": "magnetics", "remaining": 5},
            {"domain": "mhd", "remaining": 3},
        ]
        staleness_rows = [
            {"domain": "equilibrium", "last_run": datetime(2025, 1, 1, tzinfo=UTC)},
            {"domain": "magnetics", "last_run": None},  # never run
            {"domain": "mhd", "last_run": datetime(2025, 1, 5, tzinfo=UTC)},
        ]
        mock_gc = MagicMock()
        mock_gc.query.return_value = staleness_rows
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.graph.client.GraphClient",
            return_value=mock_gc,
        ):
            winner = _pick_stalest_domain(candidates)

        assert winner["domain"] == "magnetics"

    def test_tiebreak_by_eligible_count(self):
        """When two domains have the same last_run, higher remaining wins."""
        candidates = [
            {"domain": "equilibrium", "remaining": 5},
            {"domain": "magnetics", "remaining": 20},
        ]
        # Both ran at the same time
        staleness_rows = [
            {"domain": "equilibrium", "last_run": datetime(2025, 1, 1, tzinfo=UTC)},
            {"domain": "magnetics", "last_run": datetime(2025, 1, 1, tzinfo=UTC)},
        ]
        mock_gc = MagicMock()
        mock_gc.query.return_value = staleness_rows
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.graph.client.GraphClient",
            return_value=mock_gc,
        ):
            winner = _pick_stalest_domain(candidates)

        assert winner["domain"] == "magnetics"

    def test_all_unrun_tiebreak_by_count(self):
        """When no domain has a prior SNRun, highest remaining wins."""
        candidates = [
            {"domain": "equilibrium", "remaining": 5},
            {"domain": "magnetics", "remaining": 20},
        ]
        staleness_rows = [
            {"domain": "equilibrium", "last_run": None},
            {"domain": "magnetics", "last_run": None},
        ]
        mock_gc = MagicMock()
        mock_gc.query.return_value = staleness_rows
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.graph.client.GraphClient",
            return_value=mock_gc,
        ):
            winner = _pick_stalest_domain(candidates)

        assert winner["domain"] == "magnetics"

    def test_empty_query_result_returns_first_candidate(self):
        """If the graph returns no rows, fall back to first candidate."""
        candidates = [
            {"domain": "equilibrium", "remaining": 10},
            {"domain": "magnetics", "remaining": 5},
        ]
        mock_gc = MagicMock()
        mock_gc.query.return_value = []
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.graph.client.GraphClient",
            return_value=mock_gc,
        ):
            winner = _pick_stalest_domain(candidates)

        assert winner["domain"] == "equilibrium"


# ═══════════════════════════════════════════════════════════════════════
# select_next_domain (unit tests)
# ═══════════════════════════════════════════════════════════════════════


class TestSelectNextDomain:
    """Verify select_next_domain wiring."""

    def test_returns_none_when_no_eligible(self):
        """No eligible domains → None."""
        with (
            patch(
                "imas_codex.standard_names.loop._count_eligible_domains",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.loop._existing_domain_targets",
                return_value=[],
            ),
        ):
            assert select_next_domain() is None

    def test_falls_back_to_existing_targets(self):
        """When no extract-eligible, falls back to maintenance targets."""
        with (
            patch(
                "imas_codex.standard_names.loop._count_eligible_domains",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.loop._existing_domain_targets",
                return_value=[{"domain": "magnetics", "remaining": 3}],
            ),
        ):
            result = select_next_domain()
        assert result is not None
        assert result["domain"] == "magnetics"

    def test_explicit_domain_bypasses_rotation(self):
        """When only_domain is set, always return that domain."""
        with patch(
            "imas_codex.standard_names.loop._count_eligible_domains",
            return_value=[{"domain": "equilibrium", "remaining": 7}],
        ):
            result = select_next_domain(only_domain="equilibrium")
        assert result is not None
        assert result["domain"] == "equilibrium"

    def test_explicit_domain_returns_none_if_no_work(self):
        """Explicit domain with no work → None (don't fall through to other domains)."""
        with (
            patch(
                "imas_codex.standard_names.loop._count_eligible_domains",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.loop._existing_domain_targets",
                return_value=[],
            ),
        ):
            assert select_next_domain(only_domain="empty_domain") is None

    def test_skip_generate_uses_maintenance_targets(self):
        """With skip_generate, uses _existing_domain_targets directly."""
        with patch(
            "imas_codex.standard_names.loop._existing_domain_targets",
            return_value=[{"domain": "transport", "remaining": 5}],
        ):
            result = select_next_domain(skip_generate=True)
        assert result is not None
        assert result["domain"] == "transport"


# ═══════════════════════════════════════════════════════════════════════
# run_sn_loop — turn-entry floor and rotation (integration-level)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestLoopFloor:
    """Turn-entry floor enforcement."""

    async def test_stops_below_floor(self, caplog):
        """Budget below MIN_VIABLE_TURN → immediate stop with INFO log."""
        with (
            patch("imas_codex.standard_names.loop._write_sn_run"),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                new_callable=AsyncMock,
            ) as mock_turn,
        ):
            with caplog.at_level(logging.INFO, logger="imas_codex.standard_names.loop"):
                summary = await run_sn_loop(cost_limit=0.50)

        assert summary.stop_reason == "budget_exhausted"
        assert summary.cost_spent == 0.0
        mock_turn.assert_not_awaited()
        assert any("Budget exhausted" in m for m in caplog.messages)

    async def test_floor_constant_is_meaningful(self):
        """MIN_VIABLE_TURN is positive and reasonable."""
        assert MIN_VIABLE_TURN > 0.0
        assert MIN_VIABLE_TURN <= 2.0  # sanity: not absurdly large


@pytest.mark.asyncio
class TestLoopRotation:
    """Loop-level rotation and budget propagation."""

    async def test_full_budget_goes_to_turn(self):
        """The selected domain gets the full remaining budget, not a fair share."""
        captured_configs = []

        async def capture_turn(cfg):
            captured_configs.append(cfg)
            return [
                PhaseResult(name="generate", count=1, cost=2.0),
            ]

        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                return_value={"domain": "equilibrium", "remaining": 10},
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                side_effect=capture_turn,
            ),
            patch("imas_codex.standard_names.loop._write_sn_run"),
        ):
            summary = await run_sn_loop(cost_limit=2.0)  # noqa: F841

        assert len(captured_configs) == 1
        # The turn should get the FULL budget, not budget/N_domains
        assert captured_configs[0].cost_limit == 2.0
        assert captured_configs[0].domain == "equilibrium"

    async def test_rotates_domains_across_iterations(self):
        """Multiple iterations pick different domains from select_next_domain."""
        call_count = 0
        domains_in_order = ["transport", "magnetics", "equilibrium"]

        def rotating_selector(**kwargs):
            nonlocal call_count
            if call_count >= len(domains_in_order):
                return None  # no more work
            dom = domains_in_order[call_count]
            call_count += 1
            return {"domain": dom, "remaining": 5}

        async def cheap_turn(cfg):
            return [
                PhaseResult(name="generate", count=1, cost=0.30),
            ]

        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                side_effect=rotating_selector,
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                side_effect=cheap_turn,
            ),
            patch("imas_codex.standard_names.loop._write_sn_run"),
        ):
            summary = await run_sn_loop(cost_limit=5.0)

        assert summary.domains_touched == {"transport", "magnetics", "equilibrium"}
        assert summary.cost_spent == pytest.approx(0.90)
        assert summary.stop_reason == "completed"
        assert len(summary.pass_records) == 3

    async def test_explicit_domain_bypasses_rotation(self):
        """--physics-domain flag is forwarded to select_next_domain."""
        captured_kwargs = []

        def spy_selector(**kwargs):
            captured_kwargs.append(kwargs)
            if len(captured_kwargs) > 1:
                return None
            return {"domain": "equilibrium", "remaining": 5}

        async def cheap_turn(cfg):
            return [
                PhaseResult(name="generate", count=1, cost=1.0),
            ]

        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                side_effect=spy_selector,
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                side_effect=cheap_turn,
            ),
            patch("imas_codex.standard_names.loop._write_sn_run"),
        ):
            summary = await run_sn_loop(
                cost_limit=5.0,
                only_domain="equilibrium",
            )

        # Verify only_domain was passed through
        assert captured_kwargs[0]["only_domain"] == "equilibrium"
        assert "equilibrium" in summary.domains_touched

    async def test_zero_cost_turn_stops_loop(self):
        """If a turn produces zero cost, the loop stops to prevent spinning."""
        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                return_value={"domain": "magnetics", "remaining": 10},
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                new=AsyncMock(
                    return_value=[
                        PhaseResult(name="generate", count=0, cost=0.0),
                    ]
                ),
            ),
            patch("imas_codex.standard_names.loop._write_sn_run"),
        ):
            summary = await run_sn_loop(cost_limit=5.0)

        assert summary.stop_reason == "completed"
        # Only one iteration — didn't spin
        assert len(summary.pass_records) == 1

    async def test_remaining_budget_decreases_each_iteration(self):
        """Each successive turn gets the remaining budget after prior spending."""
        captured_budgets = []

        async def track_budget_turn(cfg):
            captured_budgets.append(cfg.cost_limit)
            return [
                PhaseResult(name="generate", count=1, cost=1.0),
            ]

        call_count = 0

        def domain_factory(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                return None
            return {"domain": f"domain_{call_count}", "remaining": 5}

        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                side_effect=domain_factory,
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                side_effect=track_budget_turn,
            ),
            patch("imas_codex.standard_names.loop._write_sn_run"),
        ):
            summary = await run_sn_loop(cost_limit=5.0)  # noqa: F841

        assert len(captured_budgets) == 3
        assert captured_budgets[0] == pytest.approx(5.0)
        assert captured_budgets[1] == pytest.approx(4.0)
        assert captured_budgets[2] == pytest.approx(3.0)

    async def test_no_fair_share_division(self):
        """Verify the old per_domain_budget = cost_limit/N logic is gone.

        With 10 eligible domains and $2 budget, the old code would give
        $0.20/domain.  The new code gives the full $2.00 to the first
        domain.
        """
        captured_configs = []

        async def capture_turn(cfg):
            captured_configs.append(cfg)
            return [
                PhaseResult(name="generate", count=1, cost=2.0),
            ]

        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                return_value={"domain": "equilibrium", "remaining": 100},
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                side_effect=capture_turn,
            ),
            patch("imas_codex.standard_names.loop._write_sn_run"),
        ):
            summary = await run_sn_loop(cost_limit=2.0)  # noqa: F841

        # The turn MUST receive the full budget, not budget/N
        assert captured_configs[0].cost_limit == 2.0
