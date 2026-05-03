"""Tests for graph-backed LLM cost tracking API.

All tests mock the GraphClient — no live Neo4j required.
The ``@pytest.mark.graph`` marker is reserved for integration tests
that hit a real database; these are pure unit tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_gc_ctx():
    """Return a (patcher, mock_gc) that patches GraphClient as context manager."""
    patcher = patch("imas_codex.standard_names.graph_ops.GraphClient")
    MockGC = patcher.start()
    mock_gc = MagicMock()
    MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
    MockGC.return_value.__exit__ = MagicMock(return_value=False)
    return patcher, mock_gc


_TS = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
_RUN_ID = "run-cost-test-001"


# ---------------------------------------------------------------------------
# create_sn_run_open
# ---------------------------------------------------------------------------


class TestCreateSNRunOpen:
    def test_creates_with_started_status(self):
        patcher, mock_gc = _mock_gc_ctx()
        try:
            from imas_codex.standard_names.graph_ops import create_sn_run_open

            create_sn_run_open(
                _RUN_ID,
                started_at=_TS,
                cost_limit=5.0,
                turn_number=2,
                min_score=0.6,
            )

            # create_nodes should have been called with SNRun label
            mock_gc.create_nodes.assert_called_once()
            args = mock_gc.create_nodes.call_args
            assert args[0][0] == "SNRun"
            props = args[0][1][0]
            assert props["id"] == _RUN_ID
            assert props["status"] == "started"
            assert props["cost_is_exact"] is True
            assert props["cost_spent"] == 0.0
            assert props["cost_limit"] == 5.0
            assert props["turn_number"] == 2
            assert props["min_score"] == 0.6
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# finalize_sn_run
# ---------------------------------------------------------------------------


class TestFinalizeSNRun:
    def test_updates_status_and_cost(self):
        patcher, mock_gc = _mock_gc_ctx()
        mock_gc.query.return_value = [{"id": _RUN_ID}]
        try:
            from imas_codex.standard_names.graph_ops import finalize_sn_run

            finalize_sn_run(
                _RUN_ID,
                status="completed",
                cost_spent=3.14,
                cost_is_exact=True,
                stopped_at=_TS,
                stop_reason="budget_exhausted",
                names_composed=42,
            )

            mock_gc.query.assert_called_once()
            cypher = mock_gc.query.call_args[0][0]
            kwargs = mock_gc.query.call_args[1]

            assert "MATCH (rr:SNRun {id: $run_id})" in cypher
            assert "rr.status = $status" in cypher
            assert "rr.cost_spent = $cost_spent" in cypher
            assert "rr.cost_is_exact = $cost_is_exact" in cypher
            assert kwargs["status"] == "completed"
            assert kwargs["cost_spent"] == 3.14
            assert kwargs["stop_reason"] == "budget_exhausted"
            assert kwargs["names_composed"] == 42
        finally:
            patcher.stop()

    def test_warns_when_no_snrun_found(self, caplog):
        patcher, mock_gc = _mock_gc_ctx()
        mock_gc.query.return_value = []
        try:
            import logging

            from imas_codex.standard_names.graph_ops import finalize_sn_run

            with caplog.at_level(logging.WARNING):
                finalize_sn_run(
                    _RUN_ID,
                    status="failed",
                    cost_spent=0.0,
                    stopped_at=_TS,
                )
            assert "no SNRun found" in caplog.text
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# record_llm_cost
# ---------------------------------------------------------------------------


class TestRecordLLMCost:
    def test_returns_deterministic_id(self):
        patcher, mock_gc = _mock_gc_ctx()
        mock_gc.query.return_value = []
        try:
            from imas_codex.standard_names.graph_ops import record_llm_cost

            id1 = record_llm_cost(
                run_id=_RUN_ID,
                phase="generate",
                sn_ids=["electron_temperature"],
                model="gpt-4",
                cost=0.05,
                tokens_in=100,
                tokens_out=200,
                llm_at=_TS,
            )
            id2 = record_llm_cost(
                run_id=_RUN_ID,
                phase="generate",
                sn_ids=["electron_temperature"],
                model="gpt-4",
                cost=0.05,
                tokens_in=100,
                tokens_out=200,
                llm_at=_TS,
            )
            assert id1 == id2, "Same args must produce the same deterministic id"
        finally:
            patcher.stop()

    def test_idempotent_duplicate_swallowed(self):
        """Calling twice with same args → ConstraintError is swallowed."""
        from neo4j.exceptions import ConstraintError

        patcher, mock_gc = _mock_gc_ctx()
        # First call succeeds, second raises ConstraintError
        mock_gc.query.side_effect = [[], ConstraintError("duplicate")]
        try:
            from imas_codex.standard_names.graph_ops import record_llm_cost

            # First call
            record_llm_cost(
                run_id=_RUN_ID,
                phase="generate",
                sn_ids=["a"],
                model="gpt-4",
                cost=0.01,
                tokens_in=10,
                tokens_out=20,
                llm_at=_TS,
            )
            # Second call — should NOT raise
            record_llm_cost(
                run_id=_RUN_ID,
                phase="generate",
                sn_ids=["a"],
                model="gpt-4",
                cost=0.01,
                tokens_in=10,
                tokens_out=20,
                llm_at=_TS,
            )
        finally:
            patcher.stop()

    def test_non_constraint_error_propagates(self):
        patcher, mock_gc = _mock_gc_ctx()
        mock_gc.query.side_effect = RuntimeError("boom")
        try:
            from imas_codex.standard_names.graph_ops import record_llm_cost

            with pytest.raises(RuntimeError, match="boom"):
                record_llm_cost(
                    run_id=_RUN_ID,
                    phase="generate",
                    sn_ids=[],
                    model="gpt-4",
                    cost=0.01,
                    tokens_in=10,
                    tokens_out=20,
                    llm_at=_TS,
                )
        finally:
            patcher.stop()

    def test_cypher_uses_create_not_merge(self):
        patcher, mock_gc = _mock_gc_ctx()
        mock_gc.query.return_value = []
        try:
            from imas_codex.standard_names.graph_ops import record_llm_cost

            record_llm_cost(
                run_id=_RUN_ID,
                phase="enrich",
                sn_ids=["a", "b"],
                model="gpt-4",
                cost=0.10,
                tokens_in=50,
                tokens_out=100,
                llm_at=_TS,
            )
            cypher = mock_gc.query.call_args[0][0]
            assert "CREATE (c:LLMCost" in cypher
            assert "MERGE (c:LLMCost" not in cypher
            assert "MERGE (c)-[:FOR_RUN]->(rr)" in cypher
        finally:
            patcher.stop()

    def test_empty_sn_ids_allowed(self):
        """L7 audit calls may have no sn_ids."""
        patcher, mock_gc = _mock_gc_ctx()
        mock_gc.query.return_value = []
        try:
            from imas_codex.standard_names.graph_ops import record_llm_cost

            spend_id = record_llm_cost(
                run_id=_RUN_ID,
                phase="review_names",
                sn_ids=[],
                model="gpt-4",
                cost=0.02,
                tokens_in=30,
                tokens_out=40,
                llm_at=_TS,
            )
            assert spend_id  # should still return a valid id
            kwargs = mock_gc.query.call_args[1]
            assert kwargs["sn_ids"] == []
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# aggregate_spend_for_run
# ---------------------------------------------------------------------------


class TestAggregateSpendForRun:
    def test_sums_costs(self):
        patcher, mock_gc = _mock_gc_ctx()
        mock_gc.query.return_value = [{"total": 1.5}]
        try:
            from imas_codex.standard_names.graph_ops import aggregate_spend_for_run

            total = aggregate_spend_for_run(_RUN_ID)
            assert total == 1.5

            cypher = mock_gc.query.call_args[0][0]
            assert "sum(c.llm_cost)" in cypher
            assert mock_gc.query.call_args[1]["run_id"] == _RUN_ID
        finally:
            patcher.stop()

    def test_empty_returns_zero(self):
        patcher, mock_gc = _mock_gc_ctx()
        mock_gc.query.return_value = [{"total": 0.0}]
        try:
            from imas_codex.standard_names.graph_ops import aggregate_spend_for_run

            assert aggregate_spend_for_run("nonexistent") == 0.0
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# aggregate_spend_per_phase
# ---------------------------------------------------------------------------


class TestAggregateSpendPerPhase:
    def test_groups_by_phase(self):
        patcher, mock_gc = _mock_gc_ctx()
        mock_gc.query.return_value = [
            {"phase": "enrich", "total": 0.45},
            {"phase": "generate", "total": 1.23},
        ]
        try:
            from imas_codex.standard_names.graph_ops import aggregate_spend_per_phase

            result = aggregate_spend_per_phase(_RUN_ID)
            assert result == {"enrich": 0.45, "generate": 1.23}
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# aggregate_spend_per_name
# ---------------------------------------------------------------------------


class TestAggregateSpendPerName:
    def test_apportions_cost(self):
        """A $1.00 cost with sn_ids=[a, b] → each gets $0.50."""
        patcher, mock_gc = _mock_gc_ctx()
        mock_gc.query.return_value = [
            {"sn_id": "a", "apportioned": 0.50},
            {"sn_id": "b", "apportioned": 0.50},
        ]
        try:
            from imas_codex.standard_names.graph_ops import aggregate_spend_per_name

            result = aggregate_spend_per_name(_RUN_ID)
            assert result == {"a": 0.50, "b": 0.50}

            # Verify the Cypher uses UNWIND + division
            cypher = mock_gc.query.call_args[0][0]
            assert "UNWIND c.sn_ids AS sn_id" in cypher
            assert "size(c.sn_ids)" in cypher
        finally:
            patcher.stop()

    def test_skips_empty_sn_ids(self):
        """Rows with empty sn_ids should be excluded from apportionment."""
        patcher, mock_gc = _mock_gc_ctx()
        mock_gc.query.return_value = []
        try:
            from imas_codex.standard_names.graph_ops import aggregate_spend_per_name

            result = aggregate_spend_per_name(_RUN_ID)
            assert result == {}

            cypher = mock_gc.query.call_args[0][0]
            assert "WHERE size(c.sn_ids) > 0" in cypher
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# update_sn_per_phase_costs
# ---------------------------------------------------------------------------


class TestUpdateSNPerPhaseCosts:
    def test_writes_per_phase_fields(self):
        patcher, mock_gc = _mock_gc_ctx()
        # Aggregation query returns per-name per-phase apportionment
        mock_gc.query.side_effect = [
            # First call: aggregation query
            [
                {
                    "sn_id": "electron_temperature",
                    "phase": "generate_name",
                    "apportioned": 0.30,
                },
                {
                    "sn_id": "electron_temperature",
                    "phase": "review_name",
                    "apportioned": 0.10,
                },
                {
                    "sn_id": "plasma_current",
                    "phase": "generate_name",
                    "apportioned": 0.20,
                },
            ],
            # Subsequent calls: per-name SET queries
            [{"id": "electron_temperature"}],
            [{"id": "plasma_current"}],
        ]
        try:
            from imas_codex.standard_names.graph_ops import update_sn_per_phase_costs

            count = update_sn_per_phase_costs(_RUN_ID)
            assert count == 2

            # Check that SET queries targeted the right fields
            set_calls = mock_gc.query.call_args_list[1:]
            assert len(set_calls) == 2

            # Find the electron_temperature call
            for c in set_calls:
                kwargs = c[1]
                if kwargs.get("sn_id") == "electron_temperature":
                    cypher = c[0][0]
                    assert "sn.llm_cost_generate_name" in cypher
                    assert "sn.llm_cost_review_name" in cypher
                    assert "sn.llm_cost" in cypher  # total
                    assert kwargs["total"] == pytest.approx(0.40, abs=1e-6)
                    assert kwargs["llm_cost_generate_name"] == pytest.approx(
                        0.30, abs=1e-6
                    )
                    assert kwargs["llm_cost_review_name"] == pytest.approx(
                        0.10, abs=1e-6
                    )
                    break
            else:
                pytest.fail("electron_temperature SET call not found")
        finally:
            patcher.stop()

    def test_returns_zero_when_no_costs(self):
        patcher, mock_gc = _mock_gc_ctx()
        mock_gc.query.return_value = []
        try:
            from imas_codex.standard_names.graph_ops import update_sn_per_phase_costs

            assert update_sn_per_phase_costs(_RUN_ID) == 0
        finally:
            patcher.stop()
