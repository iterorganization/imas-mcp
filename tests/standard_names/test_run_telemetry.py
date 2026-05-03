"""End-to-end telemetry persistence tests for SN run infrastructure.

Verifies that:
1. LLMCost nodes are persisted and readable after record_llm_cost.
2. SNRun.turn_number is set from the passed value (not hardcoded 1).
3. SNRun.stopped_at is populated after finalize_sn_run.
4. Sum of LLMCost.cost ≈ SNRun.cost_total (within rounding).

All tests mock the GraphClient — no live Neo4j required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC)
_TS_END = datetime(2025, 6, 1, 10, 30, 0, tzinfo=UTC)
_RUN_ID = "run-telemetry-test-001"


def _mock_gc_ctx():
    """Return a (patcher, mock_gc) that patches GraphClient as context manager."""
    patcher = patch("imas_codex.standard_names.graph_ops.GraphClient")
    MockGC = patcher.start()
    mock_gc = MagicMock()
    MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
    MockGC.return_value.__exit__ = MagicMock(return_value=False)
    return patcher, mock_gc


# ---------------------------------------------------------------------------
# A1: LLMCost persistence
# ---------------------------------------------------------------------------


class TestLLMCostPersistence:
    """Verify that record_llm_cost writes an LLMCost node and FOR_RUN edge."""

    def test_record_creates_node_and_edge(self):
        """record_llm_cost should CREATE an LLMCost node and MERGE a FOR_RUN edge."""
        patcher, mock_gc = _mock_gc_ctx()
        try:
            from imas_codex.standard_names.graph_ops import record_llm_cost

            spend_id = record_llm_cost(
                run_id=_RUN_ID,
                phase="generate_name",
                cycle=None,
                sn_ids=["test:electron_temperature"],
                model="openrouter/anthropic/claude-sonnet-4.6",
                cost=0.0123,
                tokens_in=500,
                tokens_out=200,
                tokens_cached_read=10,
                tokens_cached_write=20,
                service="standard-names",
                batch_id="batch-001",
                overspend=0.0,
                llm_at=_TS,
            )

            # Should have called gc.query with CREATE Cypher
            assert mock_gc.query.called
            cypher_call = mock_gc.query.call_args
            cypher_str = cypher_call[0][0]

            # Verify the Cypher contains CREATE for LLMCost and MERGE for FOR_RUN
            assert "CREATE (c:LLMCost" in cypher_str
            assert "MERGE (c)-[:FOR_RUN]->(rr)" in cypher_str

            # Verify key params were passed
            kwargs = cypher_call[1]
            assert kwargs["run_id"] == _RUN_ID
            assert kwargs["llm_cost"] == 0.0123
            assert kwargs["llm_tokens_in"] == 500
            assert kwargs["llm_tokens_out"] == 200
            assert kwargs["phase"] == "generate_name"
            assert kwargs["pool"] == "compose"  # generate_name maps to compose
            assert kwargs["llm_model"] == "openrouter/anthropic/claude-sonnet-4.6"

            # Should return a deterministic id
            assert spend_id is not None
            assert isinstance(spend_id, str)
        finally:
            patcher.stop()

    def test_record_refuses_test_model(self):
        """record_llm_cost should refuse model='test-model' by default."""
        patcher, mock_gc = _mock_gc_ctx()
        try:
            from imas_codex.standard_names.graph_ops import record_llm_cost

            spend_id = record_llm_cost(
                run_id=_RUN_ID,
                phase="generate_name",
                model="test-model",
                cost=0.001,
                tokens_in=100,
                tokens_out=50,
            )

            # Should NOT have called gc.query (fixture leak guard)
            assert not mock_gc.query.called
            assert spend_id is not None  # still returns the id
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# A1: create_sn_run_open raises on failure
# ---------------------------------------------------------------------------


class TestCreateSNRunRaises:
    """Verify create_sn_run_open propagates exceptions instead of swallowing."""

    def test_raises_on_graph_error(self):
        """create_sn_run_open must raise (not swallow) graph errors."""
        patcher, mock_gc = _mock_gc_ctx()
        try:
            mock_gc.create_nodes.side_effect = RuntimeError("Connection refused")

            from imas_codex.standard_names.graph_ops import create_sn_run_open

            with pytest.raises(RuntimeError, match="Connection refused"):
                create_sn_run_open(
                    _RUN_ID,
                    started_at=_TS,
                    cost_limit=5.0,
                    turn_number=5,
                )
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# A2: turn_number plumbing
# ---------------------------------------------------------------------------


class TestTurnNumberPlumbing:
    """Verify turn_number propagates from create_sn_run_open to SNRun node."""

    def test_turn_number_passed_to_graph(self):
        """create_sn_run_open should set turn_number on the SNRun node."""
        patcher, mock_gc = _mock_gc_ctx()
        try:
            from imas_codex.standard_names.graph_ops import create_sn_run_open

            create_sn_run_open(
                _RUN_ID,
                started_at=_TS,
                cost_limit=10.0,
                turn_number=5,
            )

            # Verify create_nodes was called with turn_number=5
            assert mock_gc.create_nodes.called
            label, data_list = mock_gc.create_nodes.call_args[0]
            assert label == "SNRun"
            assert len(data_list) == 1
            props = data_list[0]
            assert props["turn_number"] == 5
            assert props["id"] == _RUN_ID
        finally:
            patcher.stop()

    def test_run_summary_turn_number(self):
        """RunSummary should accept arbitrary turn_number values."""
        from imas_codex.standard_names.loop import RunSummary

        summary = RunSummary(
            run_id="test-run",
            turn_number=7,
            started_at=_TS,
            cost_limit=5.0,
        )
        assert summary.turn_number == 7


# ---------------------------------------------------------------------------
# A3: stopped_at population
# ---------------------------------------------------------------------------


class TestStoppedAtPopulation:
    """Verify finalize_sn_run writes stopped_at (not ended_at)."""

    def test_finalize_writes_stopped_at(self):
        """finalize_sn_run should SET rr.stopped_at in Cypher."""
        patcher, mock_gc = _mock_gc_ctx()
        try:
            mock_gc.query.return_value = [{"id": _RUN_ID}]

            from imas_codex.standard_names.graph_ops import finalize_sn_run

            finalize_sn_run(
                _RUN_ID,
                status="completed",
                cost_spent=1.234,
                cost_is_exact=True,
                stopped_at=_TS_END,
            )

            assert mock_gc.query.called
            cypher_call = mock_gc.query.call_args
            cypher_str = cypher_call[0][0]

            # Should use stopped_at, NOT ended_at
            assert "rr.stopped_at = datetime($stopped_at)" in cypher_str
            assert "ended_at" not in cypher_str

            kwargs = cypher_call[1]
            assert kwargs["stopped_at"] == _TS_END.isoformat()
            assert kwargs["status"] == "completed"
            assert kwargs["cost_spent"] == 1.234
        finally:
            patcher.stop()

    def test_run_summary_has_stopped_at(self):
        """RunSummary dataclass should have stopped_at, not ended_at."""
        from imas_codex.standard_names.loop import RunSummary

        summary = RunSummary(
            run_id="test-run",
            turn_number=1,
            started_at=_TS,
            cost_limit=5.0,
        )
        # stopped_at should be None by default
        assert summary.stopped_at is None
        assert not hasattr(summary, "ended_at")

        summary.stopped_at = _TS_END
        assert summary.stopped_at == _TS_END


# ---------------------------------------------------------------------------
# A4: Cost consistency (LLMCost sum ≈ SNRun.cost_spent)
# ---------------------------------------------------------------------------


class TestCostConsistency:
    """Verify aggregate_spend_for_run sums LLMCost nodes correctly."""

    def test_aggregate_spend(self):
        """aggregate_spend_for_run should sum llm_cost from LLMCost nodes."""
        patcher, mock_gc = _mock_gc_ctx()
        try:
            mock_gc.query.return_value = [{"total": 3.456}]

            from imas_codex.standard_names.graph_ops import aggregate_spend_for_run

            total = aggregate_spend_for_run(_RUN_ID)
            assert abs(total - 3.456) < 0.001

            # Verify the correct Cypher was used
            cypher_call = mock_gc.query.call_args
            cypher_str = cypher_call[0][0]
            assert "LLMCost" in cypher_str
            assert "sum(c.llm_cost)" in cypher_str
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Schema field presence
# ---------------------------------------------------------------------------


class TestSchemaFields:
    """Verify the generated SNRun model has the expected fields."""

    def test_snrun_has_stopped_at(self):
        """SNRun Pydantic model should have stopped_at field."""
        from imas_codex.graph.models import SNRun

        fields = SNRun.model_fields
        assert "stopped_at" in fields, "SNRun model missing stopped_at field"
        assert "ended_at" not in fields, (
            "SNRun model still has ended_at (should be stopped_at)"
        )

    def test_snrun_has_turn_number(self):
        """SNRun Pydantic model should have turn_number field."""
        from imas_codex.graph.models import SNRun

        fields = SNRun.model_fields
        assert "turn_number" in fields

    def test_snrun_has_started_at(self):
        """SNRun Pydantic model should have started_at field."""
        from imas_codex.graph.models import SNRun

        fields = SNRun.model_fields
        assert "started_at" in fields
