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
            stopped_at=datetime(2025, 1, 1, 0, 5, tzinfo=UTC),
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
            patch(
                "imas_codex.standard_names.graph_ops.finalize_sn_run"
            ) as mock_finalize,
            patch(
                "imas_codex.standard_names.graph_ops.create_sn_run_open"
            ) as mock_create,
        ):
            summary = await run_sn_loop(cost_limit=5.0, dry_run=True)

        assert summary.stop_reason == "dry_run"
        assert mock_run.await_count == 0
        # Dry-run must NOT persist a SNRun row.
        mock_create.assert_not_called()
        mock_finalize.assert_not_called()
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
            patch("imas_codex.standard_names.graph_ops.finalize_sn_run"),
            patch("imas_codex.standard_names.graph_ops.create_sn_run_open"),
        ):
            summary = await run_sn_loop(cost_limit=5.0, dry_run=False)
        assert summary.stop_reason in ("completed", "no_work")
        assert summary.cost_spent == 0.0

    async def test_budget_exhausted_stops_mid_pass(self):
        """Once cost_spent >= cost_limit, the loop exits with budget_exhausted."""
        from imas_codex.standard_names.turn import PhaseResult

        # run_turn burns the full budget on the selected domain.
        # With one-domain-per-turn rotation, remaining drops below
        # EST_UNIT_COST and the loop stops.

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
            patch("imas_codex.standard_names.graph_ops.finalize_sn_run"),
            patch("imas_codex.standard_names.graph_ops.create_sn_run_open"),
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
            patch("imas_codex.standard_names.graph_ops.finalize_sn_run"),
            patch("imas_codex.standard_names.graph_ops.create_sn_run_open"),
        ):
            summary = await run_sn_loop(cost_limit=5.0, dry_run=False)

        assert seen_ids == [summary.run_id]


# ═══════════════════════════════════════════════════════════════════════
# Per-domain stall detection (B1 fix — infinite-turn-loop guard)
# ═══════════════════════════════════════════════════════════════════════


class TestStallDetection:
    """A domain that re-selects with unchanged `remaining` AND makes zero
    forward progress (no compose/review/regen) for 2 consecutive turns
    must stop the loop with stop_reason="stalled".

    This prevents the pre-fix failure mode where a domain with
    unrecoverable items (vocab gaps, invariant violations) caused the
    DD loop to spin forever at ~$0.05/turn in extract/enrich overhead.
    """

    @pytest.mark.asyncio
    async def test_two_consecutive_stalls_stops_loop(self):
        """Same domain, same remaining, zero progress → stop after 2 turns."""
        from imas_codex.standard_names.turn import PhaseResult

        # Each turn burns a small non-zero cost (extract/enrich overhead)
        # but composes/reviews/regenerates nothing — exactly the stuck pattern.
        def make_stuck_results(*_a, **_kw):
            return [
                PhaseResult(
                    name="generate", count=0, cost=0.04, skipped=False, error=None
                ),
                PhaseResult(
                    name="enrich", count=0, cost=0.01, skipped=False, error=None
                ),
                PhaseResult(
                    name="review_names",
                    count=0,
                    cost=0.0,
                    skipped=True,
                    error=None,
                ),
            ]

        mock_run = AsyncMock(side_effect=lambda cfg: make_stuck_results())
        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                return_value={"domain": "equilibrium", "remaining": 5},
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                mock_run,
            ),
            patch("imas_codex.standard_names.graph_ops.finalize_sn_run"),
            patch("imas_codex.standard_names.graph_ops.create_sn_run_open"),
        ):
            summary = await run_sn_loop(cost_limit=50.0, dry_run=False)

        assert summary.stop_reason == "stalled"
        # Turn 1 establishes baseline (prev_remaining=-1), turn 2 is stall 1/2,
        # turn 3 is stall 2/2 → abort.
        assert mock_run.await_count == 3
        # Budget was preserved: well under the $50 cap.
        assert summary.cost_spent < 1.0
        assert summary.names_composed == 0

    @pytest.mark.asyncio
    async def test_progress_resets_stall_counter(self):
        """A productive turn between two stalls resets the counter so the
        loop does not stop prematurely when a domain recovers."""
        from imas_codex.standard_names.turn import PhaseResult

        # Sequence:
        #   turn 1: stuck (cost>0, no compose)  → stall=1
        #   turn 2: productive (3 composed)     → stall reset to 0
        #   turn 3: stuck again                 → stall=1
        #   turn 4: productive (exhausts budget via cost)
        sequence = iter(
            [
                # turn 1 — stuck
                [
                    PhaseResult(
                        name="generate", count=0, cost=0.05, skipped=False, error=None
                    ),
                ],
                # turn 2 — productive
                [
                    PhaseResult(
                        name="generate", count=3, cost=0.20, skipped=False, error=None
                    ),
                ],
                # turn 3 — stuck
                [
                    PhaseResult(
                        name="generate", count=0, cost=0.05, skipped=False, error=None
                    ),
                ],
                # turn 4 — exhaust remaining budget
                [
                    PhaseResult(
                        name="generate", count=2, cost=10.0, skipped=False, error=None
                    ),
                ],
            ]
        )

        def make_results(*_a, **_kw):
            return next(sequence)

        mock_run = AsyncMock(side_effect=lambda cfg: make_results())
        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                return_value={"domain": "equilibrium", "remaining": 5},
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                mock_run,
            ),
            patch("imas_codex.standard_names.graph_ops.finalize_sn_run"),
            patch("imas_codex.standard_names.graph_ops.create_sn_run_open"),
        ):
            summary = await run_sn_loop(cost_limit=10.34, dry_run=False)

        # All 4 turns ran; stop is budget_exhausted, not stalled.
        # (Total spend = 0.05+0.20+0.05+10.0 = 10.30; remaining = $0.04 < EST_UNIT_COST)
        assert mock_run.await_count == 4
        assert summary.stop_reason == "budget_exhausted"
        assert summary.names_composed == 5

    @pytest.mark.asyncio
    async def test_remaining_changed_does_not_stall(self):
        """If `remaining` decreases (real work happened in a prior turn
        that flushed some items), no stall is recorded even when the
        current turn itself has zero progress counters."""
        from imas_codex.standard_names.turn import PhaseResult

        # Select returns decreasing remaining each call (10→8→6→...).
        remaining_seq = iter([10, 8, 6, 4, 2, 0])

        def fake_select(**_kw):
            try:
                r = next(remaining_seq)
            except StopIteration:
                return None
            return {"domain": "equilibrium", "remaining": r}

        def make_results(*_a, **_kw):
            # Non-zero cost, zero compose/review counters — but remaining
            # shrinks between turns so no stall is recorded.
            return [
                PhaseResult(
                    name="generate", count=0, cost=0.05, skipped=False, error=None
                ),
            ]

        mock_run = AsyncMock(side_effect=lambda cfg: make_results())
        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                side_effect=lambda **kw: fake_select(**kw),
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                mock_run,
            ),
            patch("imas_codex.standard_names.graph_ops.finalize_sn_run"),
            patch("imas_codex.standard_names.graph_ops.create_sn_run_open"),
        ):
            summary = await run_sn_loop(cost_limit=5.0, dry_run=False)

        # Loop ran all 6 iterations then selector returned None → completed.
        assert summary.stop_reason == "completed"
        assert mock_run.await_count == 6
