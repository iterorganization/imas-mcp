"""Tests for the DD completion rotator and skip-by-design filter.

Covers plan 32 Phase 4 deliverables:

- ``run_sn_loop`` stop conditions (dry_run, budget_exhausted, plateau,
  interrupted).
- ``RunSummary`` / ``summary_table`` shape.
- ``_apply_skip_by_design`` filters ``/process/`` paths and writes
  ``configurable_meaning`` skip sources.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.loop import (
    RunSummary,
    run_sn_loop,
    summary_table,
)
from imas_codex.standard_names.sources.dd import _apply_skip_by_design

# ═══════════════════════════════════════════════════════════════════════
# Skip-by-design filter
# ═══════════════════════════════════════════════════════════════════════


class TestApplySkipByDesign:
    """``_apply_skip_by_design`` drops /process/ paths and records them."""

    def test_keeps_normal_paths(self):
        rows = [
            {"path": "equilibrium/time_slice/profiles_1d/psi"},
            {"path": "core_profiles/profiles_1d/electrons/temperature"},
        ]
        kept = _apply_skip_by_design(rows, write_skipped=False)
        assert len(kept) == 2

    def test_drops_process_paths(self):
        rows = [
            {"path": "equilibrium/time_slice/profiles_1d/psi"},
            {"path": "edge_sources/source/process/reactions/rate"},
        ]
        kept = _apply_skip_by_design(rows, write_skipped=False)
        assert len(kept) == 1
        assert kept[0]["path"] == "equilibrium/time_slice/profiles_1d/psi"

    def test_writes_skip_records(self):
        """Verify skip records are passed to write_skipped_sources with the
        expected ``configurable_meaning`` reason."""
        captured: list[list[dict]] = []

        def fake_write(records):
            captured.append(records)
            return len(records)

        with patch(
            "imas_codex.standard_names.graph_ops.write_skipped_sources",
            side_effect=fake_write,
        ):
            rows = [
                {
                    "path": "edge_sources/source/process/reactions/rate",
                    "description": "Reaction rate coefficient",
                }
            ]
            kept = _apply_skip_by_design(rows, write_skipped=True)

        assert kept == []
        assert len(captured) == 1
        records = captured[0]
        assert len(records) == 1
        assert records[0]["source_type"] == "dd"
        assert records[0]["skip_reason"] == "configurable_meaning"
        assert "identifier.index" in records[0]["skip_reason_detail"]
        assert records[0]["description"] == "Reaction rate coefficient"

    def test_graph_write_failure_is_non_fatal(self):
        """A Neo4j failure during skip-write must not abort extraction."""
        rows = [{"path": "edge_sources/source/process/x"}]
        with patch(
            "imas_codex.standard_names.graph_ops.write_skipped_sources",
            side_effect=RuntimeError("neo4j down"),
        ):
            # Should not raise; returns empty kept list regardless.
            kept = _apply_skip_by_design(rows, write_skipped=True)
        assert kept == []


# ═══════════════════════════════════════════════════════════════════════
# RunSummary / summary_table
# ═══════════════════════════════════════════════════════════════════════


class TestRunSummary:
    def test_summary_table_shape(self):
        s = RunSummary(
            run_id="abc123",
            turn_number=3,
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
            ended_at=datetime(2025, 1, 1, 0, 5, tzinfo=UTC),
            cost_spent=1.2345,
            cost_limit=5.0,
            names_composed=10,
            names_enriched=8,
            names_reviewed=8,
            names_regenerated=2,
            domains_touched={"equilibrium", "magnetics"},
            stop_reason="completed",
        )
        row = summary_table(s)
        assert row["run_id"] == "abc123"
        assert row["turn_number"] == 3
        assert row["cost_spent"] == 1.2345
        assert row["stop_reason"] == "completed"
        assert row["domains_touched"] == ["equilibrium", "magnetics"]
        assert row["elapsed_s"] == 300.0


# ═══════════════════════════════════════════════════════════════════════
# run_sn_loop stop conditions
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestRunDdCompletion:
    """Verify the DD completion loop obeys plan-32 stop conditions."""

    async def test_dry_run_short_circuits(self):
        """Dry run returns immediately without invoking run_turn."""
        with (
            patch(
                "imas_codex.standard_names.loop._count_eligible_domains",
                return_value=[{"domain": "equilibrium", "remaining": 42}],
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                new_callable=AsyncMock,
            ) as mock_run,
            patch("imas_codex.standard_names.loop._write_sn_run") as mock_write,
        ):
            summary = await run_sn_loop(cost_limit=5.0, dry_run=True)

        assert summary.stop_reason == "dry_run"
        assert mock_run.await_count == 0
        # Dry-run must NOT persist a SNRun row.
        mock_write.assert_not_called()
        assert summary.pass_records[0]["eligible_domains"] == [
            {"domain": "equilibrium", "remaining": 42}
        ]

    async def test_no_work_exits_completed(self):
        """When no eligible domains and no maintenance work, the loop exits."""
        with (
            patch(
                "imas_codex.standard_names.loop._count_eligible_domains",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.loop._existing_domain_targets",
                return_value=[],
            ),
            patch("imas_codex.standard_names.loop._write_sn_run"),
        ):
            summary = await run_sn_loop(cost_limit=5.0, dry_run=False)
        assert summary.stop_reason in ("completed", "no_work")
        assert summary.cost_spent == 0.0

    async def test_budget_exhausted_stops_mid_pass(self):
        """Once cost_spent >= cost_limit, the loop exits with budget_exhausted."""
        from imas_codex.standard_names.turn import PhaseResult

        # run_turn burns the full budget on the selected domain.
        # With one-domain-per-turn rotation, remaining drops below
        # MIN_VIABLE_TURN and the loop stops.

        def make_results(*_args, **_kwargs):
            return [
                PhaseResult(
                    name="generate", count=3, cost=5.0, skipped=False, error=None
                ),
                PhaseResult(
                    name="enrich", count=3, cost=0.0, skipped=False, error=None
                ),
                PhaseResult(
                    name="review_names", count=3, cost=0.0, skipped=False, error=None
                ),
                PhaseResult(name="regen", count=0, cost=0.0, skipped=True, error=None),
            ]

        mock_run = AsyncMock(side_effect=lambda cfg: make_results())
        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                return_value={"domain": "equilibrium", "remaining": 10},
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                mock_run,
            ),
            patch("imas_codex.standard_names.loop._write_sn_run"),
        ):
            summary = await run_sn_loop(cost_limit=5.0, dry_run=False)

        assert summary.stop_reason == "budget_exhausted"
        assert summary.cost_spent >= 5.0
        assert summary.names_composed == 3
        # Only one domain should have been processed before budget cut.
        assert len(summary.domains_touched) == 1

    async def test_run_id_propagates_to_config(self):
        """The run_id is threaded through every per-domain
        TurnConfig so provenance stamping is coherent."""
        from imas_codex.standard_names.turn import PhaseResult

        seen_ids: list[str] = []

        async def fake_run(cfg):
            seen_ids.append(cfg.run_id)
            return [
                PhaseResult(
                    name="generate",
                    count=0,
                    cost=10.0,  # exhaust budget immediately
                    skipped=False,
                    error=None,
                ),
                PhaseResult(name="enrich", count=0, cost=0.0, skipped=True, error=None),
                PhaseResult(name="review", count=0, cost=0.0, skipped=True, error=None),
                PhaseResult(name="regen", count=0, cost=0.0, skipped=True, error=None),
            ]

        with (
            patch(
                "imas_codex.standard_names.loop._count_eligible_domains",
                return_value=[{"domain": "equilibrium", "remaining": 5}],
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                side_effect=fake_run,
            ),
            patch("imas_codex.standard_names.loop._write_sn_run"),
        ):
            summary = await run_sn_loop(cost_limit=5.0, dry_run=False)

        assert seen_ids == [summary.run_id]
