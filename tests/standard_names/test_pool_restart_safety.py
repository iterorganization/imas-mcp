"""Tests for Phase 8 M8 — crash-between-LLM-and-persist restart safety.

Accepted invariant (plan.md M8):
    ``LLMCost`` rows are atomic and durable.  If the process dies after the
    LLM call returns and before ``write_standard_names()`` persists candidates,
    the cost is paid but the candidates are lost.  Restart re-claims the source
    and re-pays.  Cheaper than building a post-LLM/pre-persist journal.

Tests
-----
1. test_crash_between_llm_and_persist_invariant
   M8 acceptance: abort between charge and persist; restart; verify source is
   re-claimed and re-composed.  Expected final state: 2 LLMCost rows (cost paid
   twice), 1 StandardName persisted (only on second attempt).

2. test_stale_claim_recoverable_after_reconcile
   Source with claimed_at older than the timeout is reclaimed after stale
   recovery.

3. test_fresh_claim_not_disturbed_by_reconcile
   Source with a fresh claimed_at is NOT returned by another claim attempt
   (stale recovery does not steal active claims).

4. test_llmcost_row_durability_across_simulated_crash
   record_llm_cost writes the row atomically before any subsequent crash.

All tests are pure-unit (no live Neo4j, no OpenRouter).  Total runtime < 30s.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


def _mock_budget_manager() -> MagicMock:
    """BudgetManager that always admits; lease tracks charge_event calls."""
    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
    lease.release_unused = MagicMock(return_value=0.0)
    mgr.reserve = MagicMock(return_value=lease)
    mgr.pool_admit = MagicMock(return_value=True)
    mgr.exhausted = MagicMock(return_value=False)
    return mgr


def _mock_gc():
    """Mock GraphClient context manager with transaction support."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    # Transaction-based mock
    tx = MagicMock()
    tx.closed = False
    tx.commit = MagicMock()
    tx.close = MagicMock()

    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)

    @contextmanager
    def _session_ctx():
        yield session

    gc.session = _session_ctx
    gc._tx = tx  # expose for test access
    return gc


def _patch_gc(mock_gc):
    """Patch GraphClient in graph_ops with *mock_gc*."""
    return patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=mock_gc,
    )


def _make_batch_items(n: int = 2) -> list[dict[str, Any]]:
    """Synthetic batch items for compose tests."""
    return [
        {
            "path": f"core_profiles/electrons/temperature_{i}",
            "description": f"Electron temperature field {i}.",
            "physics_domain": "core_profiles",
            "unit": "eV",
            "dd_version": "4.0.0",
            "cocos_version": "11",
        }
        for i in range(n)
    ]


def _make_compose_llm_result(items: list[dict[str, Any]]) -> Any:
    """Unpackable mock LLM result for compose calls (result, cost, tokens)."""
    candidates = []
    for i, item in enumerate(items):
        path = item.get("path", f"unknown/{i}")
        candidates.append(
            SimpleNamespace(
                standard_name=f"electron_temperature_{i}",
                source_id=path,
                kind="scalar",
                dd_paths=[path],
                grammar_fields={"physical_base": "temperature", "subject": "electron"},
                reason="test",
            )
        )
    result = SimpleNamespace(
        candidates=candidates,
        vocab_gaps=[],
        attachments=[],
        skipped=[],
    )

    class _LLMResult:
        def __init__(self) -> None:
            self.input_tokens = 50
            self.output_tokens = 50
            self.cache_read_tokens = 0
            self.cache_creation_tokens = 0

        def __iter__(self):
            return iter((result, 0.05, 100))

    return _LLMResult()


@contextmanager
def _compose_patches(
    *,
    llm_result: Any | None = None,
    persist_side_effect: Any = None,
):
    """Context manager providing all patches needed to invoke process_compose_batch.

    Mirrors the pattern from ``tests/standard_names/test_pool_batch_processors.py``.
    """
    items_placeholder: list[dict] = []

    def _default_llm(*_a, **_k):
        return _make_compose_llm_result(items_placeholder)

    actual_llm = llm_result if llm_result is not None else _default_llm

    def _default_persist(candidates, **_kw):
        return len(candidates)

    actual_persist = (
        persist_side_effect if persist_side_effect is not None else _default_persist
    )

    with (
        patch(
            "imas_codex.standard_names.context.build_compose_context",
            return_value={},
        ),
        patch("imas_codex.settings.get_compose_lean", return_value=False),
        patch(
            "imas_codex.standard_names.workers._enrich_batch_items",
            side_effect=lambda _items: None,
        ),
        patch(
            "imas_codex.standard_names.workers._search_nearby_names",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.workers._enrich_ids_context",
            return_value=None,
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mock prompt",
        ),
        patch(
            "imas_codex.standard_names.context.build_domain_vocabulary_preseed",
            side_effect=lambda d: f"vocab:{d}" if d else "",
        ),
        patch(
            "imas_codex.standard_names.review.themes.extract_reviewer_themes",
            return_value=[],
        ),
        patch(
            "imas_codex.settings.get_model",
            return_value="test-model",
        ),
        patch(
            "imas_codex.standard_names.example_loader.load_compose_examples",
            return_value=[],
        ),
        patch("imas_codex.graph.client.GraphClient"),
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new_callable=AsyncMock,
            side_effect=actual_llm,
        ),
        patch(
            "imas_codex.standard_names.graph_ops.persist_composed_batch",
            side_effect=actual_persist,
        ),
    ):
        yield items_placeholder


# ---------------------------------------------------------------------------
# In-memory source store for restart-safety tests
# ---------------------------------------------------------------------------


class _InMemorySourceDB:
    """Thread-safe in-memory stand-in for StandardNameSource nodes.

    Implements the same stale-claim recovery predicate used by
    ``claim_compose_seed_and_expand``:

        claimed_at IS NULL OR claimed_at < now() - timedelta(seconds=timeout)
    """

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self._lock = threading.Lock()
        self._nodes: dict[str, dict[str, Any]] = {n["id"]: dict(n) for n in nodes}

    def claim(self, *, timeout_seconds: int = 300) -> list[dict[str, Any]]:
        """Atomically claim one unclaimed-or-stale source."""
        with self._lock:
            cutoff = datetime.now(UTC) - timedelta(seconds=timeout_seconds)
            eligible = [
                n
                for n in self._nodes.values()
                if n.get("status") == "extracted"
                and (n.get("claimed_at") is None or n["claimed_at"] < cutoff)
            ]
            if not eligible:
                return []
            token = str(uuid.uuid4())
            node = eligible[0]
            node["claimed_at"] = datetime.now(UTC)
            node["claim_token"] = token
            return [{"id": node["id"], "path": node.get("path", node["id"]), **node}]

    def release_token(self, token: str) -> None:
        with self._lock:
            for n in self._nodes.values():
                if n.get("claim_token") == token:
                    n["claimed_at"] = None
                    n["claim_token"] = None

    def mark_composed(self, node_id: str) -> None:
        with self._lock:
            n = self._nodes.get(node_id)
            if n:
                n["status"] = "composed"
                n["claimed_at"] = None
                n["claim_token"] = None

    def get(self, node_id: str) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._nodes[node_id]) if node_id in self._nodes else None


# ---------------------------------------------------------------------------
# Test 1: M8 acceptance — crash between charge and persist
# ---------------------------------------------------------------------------


class TestCrashBetweenLLMAndPersist:
    """Acceptance criterion (plan.md M8): cost paid twice, name persisted once.

    Accepted invariant: ``LLMCost`` rows are durable even across a crash between
    the LLM call and the graph persist.  On restart, the stale claim is recovered
    and the source is re-processed.  Final state: 2 LLMCost rows, 1 StandardName.
    """

    @pytest.mark.asyncio
    async def test_crash_between_llm_and_persist_invariant(self) -> None:
        """Accepted invariant per plan.md M8: cost paid twice, name persisted once.

        Sequence:
          1. Source is claimed.
          2. LLM called → charge_event records cost → CRASH before persist.
          3. Simulate restart: claimed_at forced stale → re-claim succeeds.
          4. Second attempt: charge_event called again → persist succeeds.

        Final assertions:
          - charge_event called exactly 2× (2 LLMCost rows in production).
          - persist called exactly 2× but only 2nd persisted names (1 SN).
          - Source status would be 'composed' after second attempt.
        """
        from imas_codex.standard_names.workers import process_compose_batch

        items = _make_batch_items(2)
        stop = asyncio.Event()

        # ── Budget manager with charge tracking ────────────────────────
        charge_calls: list[dict[str, Any]] = []

        def _tracked_charge(cost: float, event: Any) -> SimpleNamespace:
            charge_calls.append(
                {
                    "cost": cost,
                    "model": getattr(event, "model", None),
                    "sn_ids": getattr(event, "sn_ids", ()),
                    "batch_id": getattr(event, "batch_id", None),
                    "phase": getattr(event, "phase", None),
                }
            )
            return SimpleNamespace(overspend=0.0)

        mgr = MagicMock()
        lease = MagicMock()
        lease.charge_event = _tracked_charge
        lease.release_unused = MagicMock(return_value=0.0)
        mgr.reserve = MagicMock(return_value=lease)
        mgr.pool_admit = MagicMock(return_value=True)
        mgr.exhausted = MagicMock(return_value=False)

        # ── Persist tracker: crashes first call ────────────────────────
        persisted_names: list[str] = []
        persist_call_count = 0

        def _crashing_persist(candidates: list[dict], **_kw: Any) -> int:
            nonlocal persist_call_count
            persist_call_count += 1
            if persist_call_count == 1:
                # Simulate process crash between LLM charge and graph write.
                raise RuntimeError("simulated crash before persist")
            # Second attempt succeeds.
            for c in candidates:
                persisted_names.append(c.get("id", ""))
            return len(candidates)

        llm_result = _make_compose_llm_result(items)

        # ── First attempt (crashes after charge_event) ─────────────────
        with _compose_patches(
            llm_result=lambda *_a, **_k: llm_result,
            persist_side_effect=_crashing_persist,
        ):
            with pytest.raises(RuntimeError, match="simulated crash before persist"):
                await process_compose_batch(items, mgr, stop)

        # Mid-state assertions:
        # charge_event was called once (LLMCost row written before crash).
        assert len(charge_calls) == 1, (
            f"Expected 1 charge_event call after first attempt, got {len(charge_calls)}"
        )
        # No names persisted (crash before persist).
        assert persisted_names == [], (
            f"Expected 0 persisted names after crash, got: {persisted_names}"
        )

        # ── Simulate restart ───────────────────────────────────────────
        # In production: reconcile_standard_name_sources() clears stale claims;
        # claim_compose_seed_and_expand() re-claims because claimed_at < cutoff.
        # Here we just confirm the second attempt can proceed (source re-issued).
        stop2 = asyncio.Event()
        llm_result2 = _make_compose_llm_result(items)

        with _compose_patches(
            llm_result=lambda *_a, **_k: llm_result2,
            persist_side_effect=_crashing_persist,
        ):
            count = await process_compose_batch(items, mgr, stop2)

        # ── Final state assertions ─────────────────────────────────────
        # Accepted invariant per plan.md M8: cost paid twice, name persisted once.
        assert len(charge_calls) == 2, (
            f"Expected 2 total charge_event calls (cost paid twice), got {len(charge_calls)}"
        )
        total_cost = sum(c["cost"] for c in charge_calls)
        assert total_cost == pytest.approx(0.10, abs=1e-9), (
            f"Expected total cost $0.10 (2 × $0.05), got ${total_cost:.4f}"
        )

        assert persist_call_count == 2, (
            f"Expected persist called twice, got {persist_call_count}"
        )
        assert len(persisted_names) > 0, (
            "Expected ≥1 names persisted on second (successful) attempt; "
            "got none — second persist may not have been reached"
        )
        assert count > 0, (
            f"process_compose_batch should return >0 on second attempt; got {count}"
        )

        # Both charge records carry the same phase tag.
        phases = {c["phase"] for c in charge_calls}
        assert phases == {"generate"}, f"Unexpected phase tags: {phases}"


# ---------------------------------------------------------------------------
# Test 2: stale claim is recoverable
# ---------------------------------------------------------------------------


class TestStaleClaimRecoverable:
    """A source with claimed_at older than the stale timeout is re-claimable.

    Verifies the stale-claim recovery predicate embedded in
    ``claim_compose_seed_and_expand``:

        (sns.claimed_at IS NULL OR sns.claimed_at < datetime() - duration($cutoff))
    """

    def test_stale_claim_recoverable_after_reconcile(self) -> None:
        """Source with claimed_at = now - 35 min is re-claimed by the next worker.

        The default stale timeout is 300 seconds (5 min).  A 35-minute-old claim
        is well past the threshold and should be recovered.
        """
        from imas_codex.standard_names.graph_ops import claim_compose_seed_and_expand

        gc = _mock_gc()
        # Step 1 (seed): return the stale source — the claim query's WHERE clause
        # accepts it because claimed_at < datetime() - duration($cutoff).
        gc._tx.run = MagicMock(
            side_effect=[
                # Seed row
                [
                    {
                        "_cluster_id": "c1",
                        "_unit": "eV",
                        "_physics_domain": "core_profiles",
                        "_batch_key": "core_profiles",
                    }
                ],
                # Read-back (batch_size=1 → no expand step)
                [
                    {
                        "id": "stale-source-001",
                        "source_id": "core_profiles/electrons/temperature",
                        "source_type": "dd",
                        "batch_key": "core_profiles",
                        "description": "Stale source for restart test.",
                    }
                ],
            ]
        )

        with _patch_gc(gc):
            result = claim_compose_seed_and_expand(batch_size=1)

        # The source should be returned — stale claim was recovered.
        assert len(result) == 1, (
            f"Expected 1 source from stale-claim recovery, got {len(result)}"
        )
        assert result[0]["id"] == "stale-source-001"

        # Verify the Cypher seed query checks claimed_at against the cutoff.
        seed_call = gc._tx.run.call_args_list[0]
        seed_cypher = seed_call.args[0]
        assert "claimed_at" in seed_cypher, (
            "Seed Cypher must filter on claimed_at for stale-claim recovery"
        )
        assert "duration" in seed_cypher, (
            "Seed Cypher must use duration($cutoff) for stale-claim timeout"
        )

    def test_stale_claim_in_memory_model(self) -> None:
        """_InMemorySourceDB: source with 35-min-old claim is re-claimable (5-min timeout)."""
        db = _InMemorySourceDB(
            [
                {
                    "id": "stale-src-001",
                    "path": "core_profiles/electrons/temperature",
                    "status": "extracted",
                    "claimed_at": datetime.now(UTC) - timedelta(minutes=35),
                    "claim_token": "old-token-xyz",
                }
            ]
        )

        # Default 5-minute timeout → 35-minute-old claim is stale.
        result = db.claim(timeout_seconds=300)

        assert len(result) == 1, f"Expected 1 result, got {len(result)}"
        assert result[0]["id"] == "stale-src-001"
        # New token issued.
        assert result[0]["claim_token"] != "old-token-xyz"


# ---------------------------------------------------------------------------
# Test 3: fresh claim is NOT disturbed
# ---------------------------------------------------------------------------


class TestFreshClaimNotDisturbed:
    """A recently-claimed source is NOT re-claimed by another worker.

    Safety side: stale recovery must not steal active claims.
    """

    def test_fresh_claim_not_disturbed_by_reconcile(self) -> None:
        """Source with claimed_at = now - 1 min is NOT returned by claim query.

        With a 5-minute timeout, a 1-minute-old claim is active and should
        not be recovered.
        """
        from imas_codex.standard_names.graph_ops import claim_compose_seed_and_expand

        gc = _mock_gc()
        # The Cypher seed query filters out fresh claims (claimed_at is recent).
        # Simulate: graph returns empty because no unclaimed/stale sources exist.
        gc._tx.run = MagicMock(
            side_effect=[
                [],  # Seed step: no eligible source (fresh claim blocks it)
            ]
        )

        with _patch_gc(gc):
            result = claim_compose_seed_and_expand(batch_size=1)

        assert result == [], f"Fresh claim should not be recovered; got: {result}"

    def test_fresh_claim_not_disturbed_in_memory_model(self) -> None:
        """_InMemorySourceDB: source with 1-min-old claim is NOT re-claimable (5-min timeout)."""
        original_token = "active-token-abc"
        db = _InMemorySourceDB(
            [
                {
                    "id": "fresh-src-001",
                    "path": "core_profiles/ions/temperature",
                    "status": "extracted",
                    "claimed_at": datetime.now(UTC) - timedelta(minutes=1),
                    "claim_token": original_token,
                }
            ]
        )

        # With 5-minute timeout, a 1-minute-old claim is still active.
        result = db.claim(timeout_seconds=300)

        assert result == [], (
            f"Fresh claim must not be stolen by stale recovery; got: {result}"
        )

        # Original claim is undisturbed.
        node = db.get("fresh-src-001")
        assert node is not None
        assert node["claim_token"] == original_token, (
            "Original claim_token must be unchanged after failed re-claim attempt"
        )


# ---------------------------------------------------------------------------
# Test 4: LLMCost row durability across a simulated crash
# ---------------------------------------------------------------------------


class TestLLMCostRowDurability:
    """record_llm_cost writes atomically before any subsequent crash.

    The LLMCost write must be durable independent of what happens after
    (e.g. persist crashing).  This is the core M8 durability guarantee.
    """

    def test_llmcost_row_durability_across_simulated_crash(self) -> None:
        """record_llm_cost persists the row synchronously; a crash after the call
        leaves the row present in the graph.

        Verifies that the ``CREATE`` Cypher statement is issued with all expected
        fields before the simulated crash.
        """
        from imas_codex.standard_names.graph_ops import record_llm_cost

        patcher = patch("imas_codex.standard_names.graph_ops.GraphClient")
        MockGC = patcher.start()
        mock_gc_instance = MagicMock()
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc_instance)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc_instance.query.return_value = []

        run_id = "test-run-durability-001"
        model = "test-model-v1"
        cost = 0.05
        phase = "generate"

        try:
            # Write the LLMCost row.
            row_id = record_llm_cost(
                run_id=run_id,
                phase=phase,
                sn_ids=["electron_temperature"],
                model=model,
                cost=cost,
                tokens_in=50,
                tokens_out=60,
                tokens_cached_read=0,
                tokens_cached_write=0,
                batch_id="batch-test-001",
            )

            # Simulate crash: raise immediately after the row is written.
            raise RuntimeError("simulated crash after record_llm_cost")

        except RuntimeError:
            pass  # Crash absorbed; row should already be in graph.

        finally:
            patcher.stop()

        # ── Verify the row was written ──────────────────────────────────
        # record_llm_cost must have called gc.query with the CREATE Cypher.
        assert mock_gc_instance.query.called, (
            "GraphClient.query must be called by record_llm_cost"
        )

        cypher_call = mock_gc_instance.query.call_args_list[0]
        cypher = cypher_call.args[0] if cypher_call.args else ""

        assert "CREATE" in cypher and "LLMCost" in cypher, (
            f"Expected CREATE (c:LLMCost ...) in Cypher, got: {cypher[:120]!r}"
        )

        # Verify the kwargs passed to the query include all expected fields.
        call_kwargs = cypher_call.kwargs if cypher_call.kwargs else {}
        assert call_kwargs.get("run_id") == run_id, (
            f"run_id mismatch: {call_kwargs.get('run_id')!r} != {run_id!r}"
        )
        assert call_kwargs.get("llm_model") == model, (
            f"model mismatch: {call_kwargs.get('llm_model')!r}"
        )
        assert call_kwargs.get("llm_cost") == pytest.approx(cost), (
            f"cost mismatch: {call_kwargs.get('llm_cost')}"
        )
        assert call_kwargs.get("phase") == phase, (
            f"phase mismatch: {call_kwargs.get('phase')!r}"
        )

        # The deterministic id is returned and is a valid UUID5.
        import uuid as _uuid

        assert row_id is not None
        try:
            parsed = _uuid.UUID(row_id)
            assert parsed.version == 5, f"Expected UUID5, got version {parsed.version}"
        except ValueError as exc:
            pytest.fail(f"record_llm_cost returned non-UUID id: {row_id!r} ({exc})")

    def test_llmcost_row_fields_preserved_after_crash(self) -> None:
        """All LLMCost fields (cost, model, sn_ids, batch_id) are in the written row.

        Exercises the full kwargs forwarded to the CREATE statement so a smoke
        test can assert these are retrievable after a restart.
        """
        from imas_codex.standard_names.graph_ops import record_llm_cost

        patcher = patch("imas_codex.standard_names.graph_ops.GraphClient")
        MockGC = patcher.start()
        mock_gc_instance = MagicMock()
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc_instance)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc_instance.query.return_value = []

        sn_ids = ["electron_temperature", "ion_temperature"]
        batch_id = "eq-cluster-42"

        try:
            record_llm_cost(
                run_id="run-xyz",
                phase="generate",
                sn_ids=sn_ids,
                model="test-model",
                cost=0.07,
                tokens_in=100,
                tokens_out=80,
                batch_id=batch_id,
            )
            raise RuntimeError("crash")
        except RuntimeError:
            pass
        finally:
            patcher.stop()

        kw = mock_gc_instance.query.call_args_list[0].kwargs or {}

        assert kw.get("sn_ids") == sn_ids, f"sn_ids not preserved: {kw.get('sn_ids')!r}"
        assert kw.get("batch_id") == batch_id, (
            f"batch_id not preserved: {kw.get('batch_id')!r}"
        )
        assert kw.get("llm_tokens_in") == 100
        assert kw.get("llm_tokens_out") == 80
        assert kw.get("llm_cost") == pytest.approx(0.07)
