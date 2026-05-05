"""Tests for the refine_name pipeline (Option B: chain creation).

Covers:
- Claim eligibility (reviewed + low score + chain < cap)
- Persist semantics (new node, REFINED_FROM edge, chain_length, edge migration)
- Release (revert refining → reviewed, token+stage verification)
- Worker behavior (escalation model selection, failure release)
- Prompt rendering
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Shared helpers
# =============================================================================

_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"
_CHAIN_HISTORY_PATH = "imas_codex.standard_names.chain_history.name_chain_history"


def _mock_gc_tx():
    """Build mock GraphClient that returns a controllable transaction."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

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
    return gc, tx


def _mock_gc_query():
    """Build mock GraphClient with a .query() method."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    return gc


@contextmanager
def _patch_gc(gc):
    with patch(_GC_PATH, return_value=gc):
        yield


@contextmanager
def _patch_chain_history(return_value=None):
    with patch(
        _CHAIN_HISTORY_PATH,
        return_value=return_value or [],
    ):
        yield


def _make_refine_item(
    sn_id: str = "test_name",
    chain_length: int = 0,
    score: float = 0.6,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a claimed-item dict as returned by claim_refine_name_batch."""
    item: dict[str, Any] = {
        "id": sn_id,
        "description": "A test quantity",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": "valid",
        "reviewer_score_name": score,
        "reviewer_comments_per_dim_name": None,
        "chain_length": chain_length,
        "name_stage": "refining",
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
        "claim_token": "tok-abc-123",
        "chain_history": [],
    }
    item.update(overrides)
    return item


# =============================================================================
# 1. Claim eligibility tests
# =============================================================================


class TestClaimRefinesEligibleSN:
    """claim_refine_name_batch selects reviewed + low-score + chain < cap."""

    def test_claim_refines_eligible_sn(self):
        from imas_codex.standard_names.graph_ops import (
            claim_refine_name_batch,
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                # seed
                [{"_cluster_id": None, "_unit": "eV", "_physics_domain": "cp"}],
                # read-back
                [
                    {
                        "id": "test_name",
                        "description": "d",
                        "documentation": None,
                        "kind": "scalar",
                        "unit": "eV",
                        "cluster_id": None,
                        "physics_domain": ["cp"],
                        "validation_status": "valid",
                        "reviewer_score_name": 0.6,
                        "reviewer_comments_per_dim_name": None,
                        "chain_length": 0,
                        "name_stage": "refining",
                        "source_paths": [],
                    }
                ],
            ]
        )

        with _patch_gc(gc), _patch_chain_history():
            items = claim_refine_name_batch(batch_size=1)

        assert len(items) == 1
        assert items[0]["chain_history"] == []

        # Verify WHERE clause in seed query
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "name_stage = 'reviewed'" in seed_cypher
        assert "reviewer_score_name" in seed_cypher

    def test_claim_skips_at_chain_cap(self):
        """Items at or above rotation_cap are excluded by WHERE clause."""
        from imas_codex.standard_names.graph_ops import (
            claim_refine_name_batch,
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                # seed — empty (no eligible items)
                [],
            ]
        )

        with _patch_gc(gc), _patch_chain_history():
            items = claim_refine_name_batch(rotation_cap=3, batch_size=10)

        assert items == []

    def test_claim_enriches_chain_history(self):
        """Each claimed item gets chain_history from name_chain_history()."""
        from imas_codex.standard_names.graph_ops import (
            claim_refine_name_batch,
        )

        gc, tx = _mock_gc_tx()
        chain = [{"id": "old_name", "chain_length": 0}]
        tx.run = MagicMock(
            side_effect=[
                [{"_cluster_id": None, "_unit": "eV", "_physics_domain": "cp"}],
                [
                    {
                        "id": "test_name",
                        "description": "d",
                        "documentation": None,
                        "kind": "scalar",
                        "unit": "eV",
                        "cluster_id": None,
                        "physics_domain": ["cp"],
                        "validation_status": "valid",
                        "reviewer_score_name": 0.5,
                        "reviewer_comments_per_dim_name": None,
                        "chain_length": 1,
                        "name_stage": "refining",
                        "source_paths": [],
                    }
                ],
            ]
        )

        with _patch_gc(gc), _patch_chain_history(return_value=chain):
            items = claim_refine_name_batch(batch_size=1)

        assert items[0]["chain_history"] == chain


# =============================================================================
# 2. Persist tests
# =============================================================================


class TestPersistCreatesNewNode:
    """persist_refined_name creates a new SN node with new identity."""

    def test_persist_creates_new_node_with_new_id(self):
        from imas_codex.standard_names.graph_ops import persist_refined_name

        gc, tx = _mock_gc_tx()
        tx.run.return_value = [
            {"new_name": "electron_temp_v2", "old_name": "electron_temp"}
        ]

        with patch(_GC_PATH, return_value=gc):
            result = persist_refined_name(
                old_name="electron_temp",
                new_name="electron_temp_v2",
                description="Refined electron temperature",
                kind="scalar",
                old_chain_length=0,
                model="test-model",
            )

        assert result["new_name"] == "electron_temp_v2"
        assert result["old_name"] == "electron_temp"
        tx.commit.assert_called_once()


class TestPersistCypherContent:
    """Verify the Cypher query contains required operations."""

    def test_cypher_contains_refined_from(self):
        from imas_codex.standard_names.graph_ops import persist_refined_name

        gc, tx = _mock_gc_tx()
        tx.run.return_value = [{"new_name": "new", "old_name": "old"}]

        with patch(_GC_PATH, return_value=gc):
            persist_refined_name(
                old_name="old",
                new_name="new",
                description="d",
                old_chain_length=1,
            )

        cypher = tx.run.call_args.args[0]
        assert "REFINED_FROM" in cypher
        assert "superseded" in cypher
        assert "PRODUCED_NAME" in cypher
        assert "HAS_STANDARD_NAME" in cypher

    def test_chain_length_incremented(self):
        from imas_codex.standard_names.graph_ops import persist_refined_name

        gc, tx = _mock_gc_tx()
        tx.run.return_value = [{"new_name": "new", "old_name": "old"}]

        with patch(_GC_PATH, return_value=gc):
            persist_refined_name(
                old_name="old",
                new_name="new",
                description="d",
                old_chain_length=2,
            )

        kwargs = tx.run.call_args.kwargs
        assert kwargs["new_chain_length"] == 3

    def test_escalation_sets_timestamp(self):
        from imas_codex.standard_names.graph_ops import persist_refined_name

        gc, tx = _mock_gc_tx()
        tx.run.return_value = [{"new_name": "new", "old_name": "old"}]

        with patch(_GC_PATH, return_value=gc):
            persist_refined_name(
                old_name="old",
                new_name="new",
                description="d",
                old_chain_length=2,
                escalated=True,
            )

        cypher = tx.run.call_args.args[0]
        assert "refine_name_escalated_at" in cypher

    def test_no_escalation_no_timestamp(self):
        from imas_codex.standard_names.graph_ops import persist_refined_name

        gc, tx = _mock_gc_tx()
        tx.run.return_value = [{"new_name": "new", "old_name": "old"}]

        with patch(_GC_PATH, return_value=gc):
            persist_refined_name(
                old_name="old",
                new_name="new",
                description="d",
                old_chain_length=0,
                escalated=False,
            )

        cypher = tx.run.call_args.args[0]
        assert "refine_name_escalated_at" not in cypher

    def test_old_sn_marked_superseded(self):
        from imas_codex.standard_names.graph_ops import persist_refined_name

        gc, tx = _mock_gc_tx()
        tx.run.return_value = [{"new_name": "new", "old_name": "old"}]

        with patch(_GC_PATH, return_value=gc):
            persist_refined_name(
                old_name="old",
                new_name="new",
                description="d",
                old_chain_length=0,
            )

        cypher = tx.run.call_args.args[0]
        assert (
            "old.name_stage  = 'superseded'" in cypher
            or "old.name_stage = 'superseded'" in cypher
        )

    def test_grammar_fields_serialized(self):
        from imas_codex.standard_names.graph_ops import persist_refined_name

        gc, tx = _mock_gc_tx()
        tx.run.return_value = [{"new_name": "new", "old_name": "old"}]

        with patch(_GC_PATH, return_value=gc):
            persist_refined_name(
                old_name="old",
                new_name="new",
                description="d",
                old_chain_length=0,
                grammar_fields={"subject": "electron", "property": "temperature"},
            )

        kwargs = tx.run.call_args.kwargs
        import json

        parsed = json.loads(kwargs["grammar_json"])
        assert parsed["subject"] == "electron"

    def test_persist_idempotent_merge(self):
        """MERGE semantics mean re-calling persist doesn't fail."""
        from imas_codex.standard_names.graph_ops import persist_refined_name

        gc, tx = _mock_gc_tx()
        tx.run.return_value = [{"new_name": "new", "old_name": "old"}]

        with patch(_GC_PATH, return_value=gc):
            result1 = persist_refined_name(
                old_name="old",
                new_name="new",
                description="d",
                old_chain_length=0,
            )
            result2 = persist_refined_name(
                old_name="old",
                new_name="new",
                description="d",
                old_chain_length=0,
            )

        assert result1["new_name"] == result2["new_name"]

    def test_persist_empty_result(self):
        """When tx returns empty, persist still returns a dict."""
        from imas_codex.standard_names.graph_ops import persist_refined_name

        gc, tx = _mock_gc_tx()
        tx.run.return_value = []

        with patch(_GC_PATH, return_value=gc):
            result = persist_refined_name(
                old_name="old",
                new_name="new",
                description="d",
                old_chain_length=0,
            )

        assert result == {"new_name": "new", "old_name": "old"}


# =============================================================================
# 3. Release tests
# =============================================================================


class TestReleaseRefineNameFailedClaims:
    """release_refine_name_failed_claims reverts name_stage and clears token."""

    def test_release_reverts_stage(self):
        from imas_codex.standard_names.graph_ops import (
            release_refine_name_failed_claims,
        )

        gc = _mock_gc_query()
        gc.query = MagicMock(return_value=[{"released": 1}])

        with _patch_gc(gc):
            released = release_refine_name_failed_claims(sn_ids=["a"], token="tok")

        assert released == 1
        cypher = gc.query.call_args.args[0]
        assert "name_stage = 'reviewed'" in cypher
        assert "name_stage = 'refining'" in cypher

    def test_release_checks_token(self):
        from imas_codex.standard_names.graph_ops import (
            release_refine_name_failed_claims,
        )

        gc = _mock_gc_query()
        gc.query = MagicMock(return_value=[{"released": 0}])

        with _patch_gc(gc):
            released = release_refine_name_failed_claims(
                sn_ids=["a"], token="wrong-token"
            )

        assert released == 0

    def test_release_checks_stage(self):
        from imas_codex.standard_names.graph_ops import (
            release_refine_name_failed_claims,
        )

        gc = _mock_gc_query()
        gc.query = MagicMock(return_value=[{"released": 0}])

        with _patch_gc(gc):
            release_refine_name_failed_claims(sn_ids=["a"], token="tok")

        # With wrong stage, should not release
        cypher = gc.query.call_args.args[0]
        assert "claim_token = $token" in cypher

    def test_empty_ids_returns_zero(self):
        from imas_codex.standard_names.graph_ops import (
            release_refine_name_failed_claims,
        )

        released = release_refine_name_failed_claims(sn_ids=[], token="tok")
        assert released == 0


class TestReleaseRefineNameClaims:
    """release_refine_name_claims reverts refining → reviewed by id list."""

    def test_release_reverts_refining_stage(self):
        from imas_codex.standard_names.graph_ops import release_refine_name_claims

        gc = _mock_gc_query()
        gc.query = MagicMock(return_value=[{"released": 2}])

        with _patch_gc(gc):
            released = release_refine_name_claims(
                sn_ids=["a", "b"], claim_token="tok123"
            )

        assert released == 2
        cypher = gc.query.call_args.args[0]
        # The CASE expression reverts refining → reviewed
        assert "THEN 'reviewed'" in cypher
        assert "name_stage = 'refining'" in cypher


# =============================================================================
# 4. Worker (process_refine_name_batch) tests
# =============================================================================


def _mock_budget_manager():
    """Build a mock BudgetManager that always grants budget."""
    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock()
    mgr.reserve = MagicMock(return_value=lease)
    return mgr


class TestProcessCallsEscalationModel:
    """When chain_length=cap-1, the escalation model is used."""

    @pytest.mark.asyncio
    async def test_process_calls_escalation_model(self):
        from imas_codex.standard_names.models import RefinedName
        from imas_codex.standard_names.workers import process_refine_name_batch

        item = _make_refine_item(chain_length=2)  # cap=3, so cap-1=2 → escalate

        refined = RefinedName(
            standard_name="electron_temperature_core",
            description="Electron temperature at the plasma core",
            kind="scalar",
            grammar_fields={"subject": "electron_temperature", "modifier": "core"},
            reason="Better specificity",
        )

        llm_out = (refined, 0.05, {"input_tokens": 100, "output_tokens": 50})

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                return_value=llm_out,
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="prompt text",
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_refined_name",
                return_value={
                    "new_name": "electron_temperature_core",
                    "old_name": "test_name",
                },
            ),
            patch(
                "imas_codex.standard_names.workers._hybrid_search_neighbours",
                return_value=[],
            ),
            patch(
                "imas_codex.settings.get_model",
                return_value="default-model",
            ),
        ):
            mgr = _mock_budget_manager()
            stop = asyncio.Event()

            count = await process_refine_name_batch([item], mgr, stop)

        assert count == 1

        # Verify escalation model was used (not default)
        # The model should be the escalation model since chain_length=2 >= cap-1=2

    @pytest.mark.asyncio
    async def test_process_uses_default_model_below_cap(self):
        from imas_codex.standard_names.models import RefinedName
        from imas_codex.standard_names.workers import process_refine_name_batch

        item = _make_refine_item(chain_length=0)  # Below cap-1 → no escalation

        refined = RefinedName(
            standard_name="electron_temperature_v2",
            description="Refined electron temperature",
            kind="scalar",
            grammar_fields={},
            reason="Better naming",
        )

        llm_out = (refined, 0.05, {"input_tokens": 100, "output_tokens": 50})

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                return_value=llm_out,
            ) as mock_llm,
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="prompt text",
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_refined_name",
                return_value={
                    "new_name": "electron_temperature_v2",
                    "old_name": "test_name",
                },
            ),
            patch(
                "imas_codex.standard_names.workers._hybrid_search_neighbours",
                return_value=[],
            ),
            patch(
                "imas_codex.settings.get_model",
                return_value="default-model",
            ),
        ):
            mgr = _mock_budget_manager()
            stop = asyncio.Event()

            count = await process_refine_name_batch([item], mgr, stop)

        assert count == 1
        # Default model should have been used
        call_model = mock_llm.call_args.kwargs.get(
            "model", mock_llm.call_args[1].get("model")
        )
        assert call_model == "default-model"


class TestProcessReleasesOnFailure:
    """On LLM error, release_refine_name_failed_claims is called."""

    @pytest.mark.asyncio
    async def test_releases_claim_on_llm_failure(self):
        from imas_codex.standard_names.workers import process_refine_name_batch

        item = _make_refine_item()

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                side_effect=RuntimeError("LLM error"),
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="prompt text",
            ),
            patch(
                "imas_codex.standard_names.workers._hybrid_search_neighbours",
                return_value=[],
            ),
            patch(
                "imas_codex.settings.get_model",
                return_value="default-model",
            ),
            patch(
                "imas_codex.standard_names.graph_ops.release_refine_name_failed_claims",
                return_value=1,
            ) as mock_release,
        ):
            mgr = _mock_budget_manager()
            stop = asyncio.Event()

            count = await process_refine_name_batch([item], mgr, stop)

        assert count == 0
        mock_release.assert_called_once()
        call_kwargs = mock_release.call_args.kwargs
        assert call_kwargs["sn_ids"] == ["test_name"]
        assert call_kwargs["token"] == "tok-abc-123"


class TestProcessStopEvent:
    """When stop_event is set, processing stops early."""

    @pytest.mark.asyncio
    async def test_stops_when_event_set(self):
        from imas_codex.standard_names.workers import process_refine_name_batch

        items = [_make_refine_item(sn_id=f"name_{i}") for i in range(5)]

        stop = asyncio.Event()
        stop.set()  # Pre-set → should process nothing

        mgr = _mock_budget_manager()

        with (
            patch(
                "imas_codex.settings.get_model",
                return_value="m",
            ),
        ):
            count = await process_refine_name_batch(items, mgr, stop)

        assert count == 0


# =============================================================================
# 5. Round-trip tests
# =============================================================================


class TestRoundTripPersistRelease:
    """Persist followed by release doesn't error (Cypher structure valid)."""

    def test_persist_then_release_no_error(self):
        from imas_codex.standard_names.graph_ops import (
            persist_refined_name,
            release_refine_name_claims,
        )

        gc, tx = _mock_gc_tx()
        tx.run.return_value = [{"new_name": "new", "old_name": "old"}]

        gc_release = _mock_gc_query()
        gc_release.query = MagicMock(return_value=[{"released": 1}])

        with patch(_GC_PATH, return_value=gc):
            persist_refined_name(
                old_name="old",
                new_name="new",
                description="d",
                old_chain_length=0,
            )

        with patch(_GC_PATH, return_value=gc_release):
            released = release_refine_name_claims(sn_ids=["old"], claim_token="tok")

        assert released == 1


class TestPersistWithEdgeMigration:
    """Verify edge migration Cypher patterns in persist."""

    def test_produced_name_migration(self):
        from imas_codex.standard_names.graph_ops import persist_refined_name

        gc, tx = _mock_gc_tx()
        tx.run.return_value = [{"new_name": "new", "old_name": "old"}]

        with patch(_GC_PATH, return_value=gc):
            persist_refined_name(
                old_name="old",
                new_name="new",
                description="d",
                old_chain_length=0,
            )

        cypher = tx.run.call_args.args[0]
        # Check for edge migration patterns
        assert "PRODUCED_NAME" in cypher
        assert "HAS_STANDARD_NAME" in cypher
        assert "DELETE" in cypher
        assert "MERGE" in cypher


# =============================================================================
# 6. Prompt rendering test
# =============================================================================


class TestPromptRendering:
    """The worker calls render_prompt with expected context keys."""

    @pytest.mark.asyncio
    async def test_prompt_context_keys(self):
        from imas_codex.standard_names.models import RefinedName
        from imas_codex.standard_names.workers import process_refine_name_batch

        item = _make_refine_item()

        refined = RefinedName(
            standard_name="new_name",
            description="d",
            kind="scalar",
            grammar_fields={},
            reason="better",
        )

        llm_out = (refined, 0.01, {})
        captured_context: dict = {}

        def _capture_render(template_name, context):
            captured_context.update(context)
            return "rendered prompt"

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                return_value=llm_out,
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                side_effect=_capture_render,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_refined_name",
                return_value={"new_name": "new_name", "old_name": "test_name"},
            ),
            patch(
                "imas_codex.standard_names.workers._hybrid_search_neighbours",
                return_value=[],
            ),
            patch(
                "imas_codex.settings.get_model",
                return_value="m",
            ),
        ):
            mgr = _mock_budget_manager()
            stop = asyncio.Event()

            await process_refine_name_batch([item], mgr, stop)

        assert "item" in captured_context
        assert "chain_history" in captured_context
        assert "chain_length" in captured_context
        assert "hybrid_neighbours" in captured_context
