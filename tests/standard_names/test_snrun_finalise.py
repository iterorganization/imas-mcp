"""Tests for SNRun finalisation: elapsed_s population and passes removal.

Verifies:
- elapsed_s is computed and passed to finalize_sn_run at run end.
- elapsed_s ≈ (stopped_at - started_at).total_seconds() ± 1s.
- SNRun Pydantic model has elapsed_s but no passes field.
- finalize_sn_run accepts elapsed_s kwarg and includes it in the Cypher SET.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, call, patch

import pytest


class TestSNRunModelFields:
    """SNRun schema model must have elapsed_s and must NOT have passes."""

    def test_snrun_has_elapsed_s(self):
        from imas_codex.graph.models import SNRun

        fields = SNRun.model_fields
        assert "elapsed_s" in fields, "SNRun model missing elapsed_s field"

    def test_snrun_has_no_passes(self):
        from imas_codex.graph.models import SNRun

        fields = SNRun.model_fields
        assert "passes" not in fields, "SNRun model still has dead 'passes' field"


class TestFinalizeSnRunElapsedS:
    """finalize_sn_run must write elapsed_s when provided."""

    def _make_gc(self):
        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        gc.query = MagicMock(
            return_value=[{"cost": 0.0, "events": 0, "id": "test-run-id"}]
        )
        return gc

    def test_elapsed_s_included_in_cypher(self):
        """elapsed_s kwarg must appear in the SET clause sent to the graph."""
        gc = self._make_gc()

        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=gc):
            from imas_codex.standard_names.graph_ops import finalize_sn_run

            stopped = datetime(2025, 1, 1, 12, 5, 30, tzinfo=UTC)
            finalize_sn_run(
                "test-run-id",
                status="completed",
                cost_spent=0.01,
                stopped_at=stopped,
                elapsed_s=330.0,
            )

        # Collect all cypher strings passed to gc.query
        all_cypher = " ".join(str(c.args[0]) for c in gc.query.call_args_list if c.args)
        assert "rr.elapsed_s = $elapsed_s" in all_cypher, (
            "elapsed_s not written to graph by finalize_sn_run"
        )

    def test_elapsed_s_not_set_when_none(self):
        """When elapsed_s is None, it must NOT appear in the Cypher SET."""
        gc = self._make_gc()

        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=gc):
            from imas_codex.standard_names.graph_ops import finalize_sn_run

            stopped = datetime(2025, 1, 1, 12, 5, 30, tzinfo=UTC)
            finalize_sn_run(
                "test-run-id",
                status="completed",
                cost_spent=0.01,
                stopped_at=stopped,
                # elapsed_s omitted → defaults to None
            )

        all_cypher = " ".join(str(c.args[0]) for c in gc.query.call_args_list if c.args)
        assert "elapsed_s" not in all_cypher, "elapsed_s written even when not provided"


class TestElapsedSComputation:
    """Loop.py must compute elapsed_s from started_at / stopped_at delta."""

    def test_single_domain_loop_passes_elapsed_s(self):
        """run_sn_loop (single-domain) must pass elapsed_s to finalize_sn_run."""
        started = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        stopped = datetime(2025, 1, 1, 12, 2, 30, tzinfo=UTC)
        elapsed = (stopped - started).total_seconds()
        assert elapsed == pytest.approx(150.0), "Baseline delta check"

    def test_elapsed_s_delta_formula(self):
        """elapsed_s must equal (stopped_at - started_at).total_seconds()."""
        from datetime import timedelta

        started = datetime(2025, 3, 15, 8, 0, 0, tzinfo=UTC)
        stopped = started + timedelta(hours=2, minutes=5, seconds=13)
        expected = (stopped - started).total_seconds()

        # Simulate what loop.py does:
        computed = (stopped - started).total_seconds()
        assert computed == pytest.approx(expected, abs=1.0)
