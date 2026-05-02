"""End-to-end integration tests for graph-backed LLM cost tracking.

All tests hit the live Neo4j instance and are marked ``@pytest.mark.graph``.
Each test creates a unique ``run_id`` and cleans up with
``MATCH (n {run_id: $rid}) DETACH DELETE n`` at teardown.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.budget import BudgetManager, LLMCostEvent

pytestmark = pytest.mark.graph


@pytest.fixture(autouse=True)
def _allow_test_model_writes(monkeypatch):
    """Opt this module into the ``test-model`` write path.

    ``record_llm_cost`` rejects ``model='test-model'`` by default to
    prevent fixture leaks into the production graph. These tests
    legitimately exercise the graph-backed write path and clean up by
    unique ``run_id`` via :func:`_cleanup_run`, so they explicitly
    bypass the guard.

    NB: requires the active graph profile to be a test-scoped one,
    NEVER ``codex``. Cleanup is by ``run_id`` so a stray run only
    leaks its own ``test-...`` rows, not anyone else's.
    """
    monkeypatch.setenv("IMAS_CODEX_ALLOW_TEST_MODEL", "1")


# ── Helpers ──────────────────────────────────────────────────────────────


def _unique_run_id() -> str:
    """Return a short unique run id for test isolation."""
    return f"test-{uuid.uuid4().hex[:8]}"


def _cleanup_run(run_id: str) -> None:
    """Delete all nodes tied to *run_id* (SNRun + LLMCost)."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        gc.query("MATCH (n {run_id: $rid}) DETACH DELETE n", rid=run_id)
        # Also delete by id (SNRun uses id = run_id)
        gc.query("MATCH (n:SNRun {id: $rid}) DETACH DELETE n", rid=run_id)


def _make_event(**overrides) -> LLMCostEvent:
    defaults = {
        "model": "test-model",
        "tokens_in": 100,
        "tokens_out": 50,
        "phase": "generate",
        "service": "standard-names",
    }
    defaults.update(overrides)
    return LLMCostEvent(**defaults)


def _count_llm_cost_rows(run_id: str) -> int:
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (c:LLMCost {run_id: $rid}) RETURN count(c) AS cnt",
            rid=run_id,
        )
        return int(rows[0]["cnt"]) if rows else 0


# ═════════════════════════════════════════════════════════════════════════
# Test 1: Idempotent double charge creates one row
# ═════════════════════════════════════════════════════════════════════════


class TestIdempotentDoubleCharge:
    def test_idempotent_double_charge_creates_one_row(self):
        """Calling record_llm_cost twice with identical args → 1 row."""
        from imas_codex.standard_names.graph_ops import (
            create_sn_run_open,
            record_llm_cost,
        )

        run_id = _unique_run_id()
        ts = datetime(2025, 4, 27, 12, 0, 0, tzinfo=UTC)
        try:
            create_sn_run_open(run_id, started_at=ts, cost_limit=5.0, turn_number=1)

            kwargs = {
                "run_id": run_id,
                "phase": "generate",
                "model": "test-model",
                "cost": 0.10,
                "tokens_in": 100,
                "tokens_out": 50,
                "batch_id": "batch-idem",
                "llm_at": ts,
            }

            id1 = record_llm_cost(**kwargs)
            id2 = record_llm_cost(**kwargs)

            assert id1 == id2, "Deterministic UUID should match"
            assert _count_llm_cost_rows(run_id) == 1
        finally:
            _cleanup_run(run_id)


# ═════════════════════════════════════════════════════════════════════════
# Test 2: Per-name apportionment correct
# ═════════════════════════════════════════════════════════════════════════


class TestPerNameApportionment:
    def test_per_name_apportionment_correct(self):
        """1 LLMCost with sn_ids=[a,b], cost=1.0 → each gets 0.5."""
        from imas_codex.standard_names.graph_ops import (
            aggregate_spend_per_name,
            create_sn_run_open,
            record_llm_cost,
        )

        run_id = _unique_run_id()
        ts = datetime(2025, 4, 27, 12, 0, 0, tzinfo=UTC)
        try:
            create_sn_run_open(run_id, started_at=ts, cost_limit=5.0, turn_number=1)
            record_llm_cost(
                run_id=run_id,
                phase="generate",
                model="test-model",
                cost=1.0,
                tokens_in=200,
                tokens_out=100,
                sn_ids=["a", "b"],
                batch_id="batch-apportion",
                llm_at=ts,
            )

            result = aggregate_spend_per_name(run_id)
            assert abs(result["a"] - 0.5) < 1e-6
            assert abs(result["b"] - 0.5) < 1e-6
        finally:
            _cleanup_run(run_id)


# ═════════════════════════════════════════════════════════════════════════
# Test 3: Per-phase aggregation
# ═════════════════════════════════════════════════════════════════════════


class TestPerPhaseAggregation:
    def test_per_phase_aggregation(self):
        """LLMCost rows with different phases → grouped correctly."""
        from imas_codex.standard_names.graph_ops import (
            aggregate_spend_per_phase,
            create_sn_run_open,
            record_llm_cost,
        )

        run_id = _unique_run_id()
        ts = datetime(2025, 4, 27, 12, 0, 0, tzinfo=UTC)
        try:
            create_sn_run_open(run_id, started_at=ts, cost_limit=10.0, turn_number=1)
            for i, (phase, cost) in enumerate(
                [
                    ("generate", 0.30),
                    ("generate", 0.20),
                    ("enrich", 0.15),
                    ("review_names", 0.10),
                ]
            ):
                record_llm_cost(
                    run_id=run_id,
                    phase=phase,
                    model="test-model",
                    cost=cost,
                    tokens_in=100 + i,
                    tokens_out=50 + i,
                    batch_id=f"batch-{phase}-{i}",
                    llm_at=datetime(2025, 4, 27, 12, i, 0, tzinfo=UTC),
                )

            result = aggregate_spend_per_phase(run_id)
            assert abs(result.get("generate", 0) - 0.50) < 1e-6
            assert abs(result.get("enrich", 0) - 0.15) < 1e-6
            assert abs(result.get("review_names", 0) - 0.10) < 1e-6
        finally:
            _cleanup_run(run_id)


# ═════════════════════════════════════════════════════════════════════════
# Test 4: Aggregate spend matches budget manager after drain
# ═════════════════════════════════════════════════════════════════════════


class TestAggregateMatchesBudgetManager:
    @pytest.mark.asyncio
    async def test_aggregate_spend_matches_budget_manager_pending(self):
        """After drain, graph aggregate ≈ BudgetManager.get_total_spent."""
        from imas_codex.standard_names.graph_ops import (
            aggregate_spend_for_run,
            create_sn_run_open,
        )

        run_id = _unique_run_id()
        ts = datetime(2025, 4, 27, 12, 0, 0, tzinfo=UTC)
        try:
            create_sn_run_open(run_id, started_at=ts, cost_limit=10.0, turn_number=1)

            mgr = BudgetManager(total_budget=10.0, run_id=run_id)
            await mgr.start()

            lease = mgr.reserve(5.0, phase="generate")
            assert lease is not None
            for i in range(3):
                evt = _make_event(
                    phase="generate",
                    sn_ids=(f"sn_{i}",),
                    batch_id=f"batch-{i}",
                )
                lease.charge_event(0.10, evt)
            lease.release_unused()

            success = await mgr.drain_pending()
            assert success

            # After drain, all pending should be flushed
            graph_total = aggregate_spend_for_run(run_id)
            mgr_total = await mgr.get_total_spent(force_refresh=True)

            assert abs(graph_total - 0.30) < 1e-4
            assert abs(mgr_total - 0.30) < 1e-4
        finally:
            _cleanup_run(run_id)


# ═════════════════════════════════════════════════════════════════════════
# Test 5: Drain barrier writes all pending
# ═════════════════════════════════════════════════════════════════════════


class TestDrainBarrier:
    @pytest.mark.asyncio
    async def test_drain_barrier_on_shutdown_writes_all_pending(self):
        """Enqueue 5 events, drain → all 5 LLMCost rows in graph."""
        from imas_codex.standard_names.graph_ops import create_sn_run_open

        run_id = _unique_run_id()
        ts = datetime(2025, 4, 27, 12, 0, 0, tzinfo=UTC)
        try:
            create_sn_run_open(run_id, started_at=ts, cost_limit=10.0, turn_number=1)

            mgr = BudgetManager(total_budget=10.0, run_id=run_id)
            await mgr.start()

            lease = mgr.reserve(5.0, phase="generate")
            assert lease is not None
            for i in range(5):
                evt = _make_event(
                    phase="generate",
                    sn_ids=(f"drain_sn_{i}",),
                    batch_id=f"drain-batch-{i}",
                )
                lease.charge_event(0.05, evt)
            lease.release_unused()

            success = await mgr.drain_pending()
            assert success

            assert _count_llm_cost_rows(run_id) == 5
        finally:
            _cleanup_run(run_id)


# ═════════════════════════════════════════════════════════════════════════
# Test 6: SNRun pre-created at start with status='started'
# ═════════════════════════════════════════════════════════════════════════


class TestSNRunPreCreated:
    def test_loop_pre_creates_sn_run_at_start(self):
        """create_sn_run_open → node exists with status='started'."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.graph_ops import create_sn_run_open

        run_id = _unique_run_id()
        ts = datetime(2025, 4, 27, 12, 0, 0, tzinfo=UTC)
        try:
            create_sn_run_open(run_id, started_at=ts, cost_limit=5.0, turn_number=1)

            with GraphClient() as gc:
                rows = gc.query(
                    "MATCH (rr:SNRun {id: $rid}) "
                    "RETURN rr.status AS status, rr.cost_is_exact AS exact",
                    rid=run_id,
                )
            assert len(rows) == 1
            assert rows[0]["status"] == "started"
            assert rows[0]["exact"] is True
        finally:
            _cleanup_run(run_id)


# ═════════════════════════════════════════════════════════════════════════
# Test 7: Finalize marks cost_is_exact False on drain failure
# ═════════════════════════════════════════════════════════════════════════


class TestFinalizeDrainFailure:
    @pytest.mark.asyncio
    async def test_finalize_sn_run_marks_cost_is_exact_false_on_drain_failure(self):
        """When writer fails, drain returns False → cost_is_exact=False."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.graph_ops import (
            create_sn_run_open,
            finalize_sn_run,
        )

        run_id = _unique_run_id()
        ts = datetime(2025, 4, 27, 12, 0, 0, tzinfo=UTC)
        try:
            create_sn_run_open(run_id, started_at=ts, cost_limit=5.0, turn_number=1)

            mgr = BudgetManager(total_budget=5.0, run_id=run_id)
            await mgr.start()

            # Make record_llm_cost raise on every call
            lease = mgr.reserve(1.0, phase="generate")
            assert lease is not None
            with patch(
                "imas_codex.standard_names.graph_ops.record_llm_cost",
                side_effect=RuntimeError("boom"),
            ):
                evt = _make_event(phase="generate", batch_id="fail-batch")
                lease.charge_event(0.10, evt)
                lease.release_unused()

                success = await mgr.drain_pending()

            assert not success, "drain should report failure"

            # Finalize with degraded status
            finalize_sn_run(
                run_id,
                status="degraded",
                cost_spent=0.10,
                cost_is_exact=False,
                ended_at=datetime.now(UTC),
            )

            with GraphClient() as gc:
                rows = gc.query(
                    "MATCH (rr:SNRun {id: $rid}) "
                    "RETURN rr.status AS status, rr.cost_is_exact AS exact",
                    rid=run_id,
                )
            assert len(rows) == 1
            assert rows[0]["status"] == "degraded"
            assert rows[0]["exact"] is False
        finally:
            _cleanup_run(run_id)


# ═════════════════════════════════════════════════════════════════════════
# Test 8: FOR_RUN edge links LLMCost → SNRun
# ═════════════════════════════════════════════════════════════════════════


class TestForRunEdge:
    def test_llm_cost_has_for_run_edge(self):
        """LLMCost row → FOR_RUN → SNRun edge exists."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.graph_ops import (
            create_sn_run_open,
            record_llm_cost,
        )

        run_id = _unique_run_id()
        ts = datetime(2025, 4, 27, 12, 0, 0, tzinfo=UTC)
        try:
            create_sn_run_open(run_id, started_at=ts, cost_limit=5.0, turn_number=1)
            record_llm_cost(
                run_id=run_id,
                phase="generate",
                model="test-model",
                cost=0.25,
                tokens_in=200,
                tokens_out=100,
                batch_id="edge-test",
                llm_at=ts,
            )

            with GraphClient() as gc:
                rows = gc.query(
                    "MATCH (c:LLMCost {run_id: $rid})-[:FOR_RUN]->(rr:SNRun {id: $rid}) "
                    "RETURN c.llm_cost AS cost, rr.status AS status",
                    rid=run_id,
                )
            assert len(rows) == 1
            assert abs(rows[0]["cost"] - 0.25) < 1e-6
            assert rows[0]["status"] == "started"
        finally:
            _cleanup_run(run_id)
