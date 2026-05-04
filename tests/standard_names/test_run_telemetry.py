"""End-to-end telemetry persistence tests for SN run infrastructure.

Verifies that:
1. LLMCost nodes are persisted and readable after record_llm_cost.
2. SNRun creation sets created_at and initial telemetry fields.
3. SNRun.stopped_at and ended_at are populated after finalize_sn_run.
4. finalize_sn_run aggregates cost_total/events_total from LLMCost children.
5. MIN_VIABLE_TURN is fully removed.

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
                )
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# A2: create sets initial telemetry fields
# ---------------------------------------------------------------------------


class TestCreateTelemetryFields:
    """Verify create_sn_run_open sets cost_total, events_total, created_at."""

    def test_initial_telemetry_fields_set(self):
        """create_sn_run_open should set cost_total=0, events_total=0."""
        patcher, mock_gc = _mock_gc_ctx()
        try:
            from imas_codex.standard_names.graph_ops import create_sn_run_open

            create_sn_run_open(
                _RUN_ID,
                started_at=_TS,
                cost_limit=10.0,
            )

            # Verify create_nodes was called with telemetry fields
            assert mock_gc.create_nodes.called
            label, data_list = mock_gc.create_nodes.call_args[0]
            assert label == "SNRun"
            assert len(data_list) == 1
            props = data_list[0]
            assert props["cost_total"] == 0.0
            assert props["events_total"] == 0
            assert props["id"] == _RUN_ID

            # Verify created_at Cypher was issued
            assert mock_gc.query.called
            cypher_str = mock_gc.query.call_args[0][0]
            assert "created_at = datetime()" in cypher_str
        finally:
            patcher.stop()

    def test_turn_number_not_in_snrun(self):
        """SNRun schema should no longer have turn_number."""
        from imas_codex.graph.models import SNRun

        assert "turn_number" not in SNRun.model_fields


# ---------------------------------------------------------------------------
# A3: stopped_at population
# ---------------------------------------------------------------------------


class TestStoppedAtPopulation:
    """Verify finalize_sn_run writes stopped_at AND ended_at."""

    def test_finalize_writes_stopped_at_and_ended_at(self):
        """finalize_sn_run should SET both rr.stopped_at and rr.ended_at."""
        patcher, mock_gc = _mock_gc_ctx()
        try:
            # First call is the LLMCost aggregation query
            # Second call is the SET query
            mock_gc.query.side_effect = [
                [{"cost": 1.234, "events": 5}],  # aggregation
                [{"id": _RUN_ID}],  # SET
            ]

            from imas_codex.standard_names.graph_ops import finalize_sn_run

            finalize_sn_run(
                _RUN_ID,
                status="completed",
                cost_spent=1.234,
                cost_is_exact=True,
                stopped_at=_TS_END,
            )

            assert mock_gc.query.call_count == 2

            # Second call should be the SET query
            cypher_call = mock_gc.query.call_args_list[1]
            cypher_str = cypher_call[0][0]

            assert "rr.stopped_at = datetime($stopped_at)" in cypher_str
            assert "rr.ended_at = datetime()" in cypher_str
            assert "rr.cost_total = $cost_total" in cypher_str
            assert "rr.events_total = $events_total" in cypher_str

            kwargs = cypher_call[1]
            assert kwargs["stopped_at"] == _TS_END.isoformat()
            assert kwargs["status"] == "completed"
            assert kwargs["cost_spent"] == 1.234
            assert kwargs["cost_total"] == 1.234
            assert kwargs["events_total"] == 5
        finally:
            patcher.stop()

    def test_run_summary_has_stopped_at(self):
        """RunSummary dataclass should have stopped_at."""
        from imas_codex.standard_names.loop import RunSummary

        summary = RunSummary(
            run_id="test-run",
            turn_number=1,
            started_at=_TS,
            cost_limit=5.0,
        )
        assert summary.stopped_at is None
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

    def test_snrun_has_ended_at(self):
        """SNRun Pydantic model should have ended_at field."""
        from imas_codex.graph.models import SNRun

        fields = SNRun.model_fields
        assert "ended_at" in fields, "SNRun model missing ended_at field"

    def test_snrun_has_cost_total(self):
        """SNRun Pydantic model should have cost_total field."""
        from imas_codex.graph.models import SNRun

        fields = SNRun.model_fields
        assert "cost_total" in fields

    def test_snrun_has_events_total(self):
        """SNRun Pydantic model should have events_total field."""
        from imas_codex.graph.models import SNRun

        fields = SNRun.model_fields
        assert "events_total" in fields

    def test_snrun_has_created_at(self):
        """SNRun Pydantic model should have created_at field."""
        from imas_codex.graph.models import SNRun

        fields = SNRun.model_fields
        assert "created_at" in fields

    def test_snrun_has_last_heartbeat(self):
        """SNRun Pydantic model should have last_heartbeat field."""
        from imas_codex.graph.models import SNRun

        fields = SNRun.model_fields
        assert "last_heartbeat" in fields

    def test_snrun_no_turn_number(self):
        """SNRun Pydantic model should NOT have turn_number."""
        from imas_codex.graph.models import SNRun

        fields = SNRun.model_fields
        assert "turn_number" not in fields

    def test_snrun_has_started_at(self):
        """SNRun Pydantic model should have started_at field."""
        from imas_codex.graph.models import SNRun

        fields = SNRun.model_fields
        assert "started_at" in fields

    def test_min_viable_turn_removed(self):
        """MIN_VIABLE_TURN must no longer exist in the budget module."""
        import imas_codex.standard_names.budget as budget_mod

        assert not hasattr(budget_mod, "MIN_VIABLE_TURN")

    def test_near_exhausted_removed(self):
        """near_exhausted must no longer exist on BudgetManager."""
        from imas_codex.standard_names.budget import BudgetManager

        assert not hasattr(BudgetManager, "near_exhausted")
