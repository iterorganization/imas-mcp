"""Tests for the refine_docs pipeline (P4.3: DocsRevision snapshot architecture).

Covers:
- Claim eligibility (reviewed + low docs score + chain < cap + verdict != accept)
- Persist semantics (DocsRevision snapshot, DOCS_REVISION_OF edge, in-place update)
- Release (revert refining → reviewed, token+stage verification)
- Worker behavior (escalation model selection, failure release)
- Chain history walking
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Shared helpers
# =============================================================================

_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"
_CHAIN_HISTORY_PATH = "imas_codex.standard_names.chain_history.docs_chain_history"


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
def _patch_docs_chain_history(return_value=None):
    with patch(
        _CHAIN_HISTORY_PATH,
        return_value=return_value or [],
    ):
        yield


def _make_refine_docs_item(
    sn_id: str = "electron_temperature",
    docs_chain_length: int = 0,
    score: float = 0.5,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a claimed-item dict as returned by claim_refine_docs_batch."""
    item: dict[str, Any] = {
        "id": sn_id,
        "description": "Electron temperature in the plasma core.",
        "documentation": "## Electron temperature\n\nBasic docs.",
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": "valid",
        "docs_stage": "refining",
        "docs_chain_length": docs_chain_length,
        "docs_model": "google/gemini-3-flash",
        "docs_generated_at": "2025-01-15T10:00:00Z",
        "reviewer_score_docs": score,
        "reviewer_comments_per_dim_docs": json.dumps(
            {"completeness": "Missing edge cases"}
        ),
        "reviewer_comments_docs": "Needs more detail on measurement methods.",
        "tags": ["electron", "temperature"],
        "claim_token": "tok-docs-abc-123",
        "docs_chain_history": [],
    }
    item.update(overrides)
    return item


# =============================================================================
# 1. Claim eligibility tests
# =============================================================================


class TestClaimRefineDocsEligible:
    """claim_refine_docs_batch selects reviewed + low docs score + chain < cap."""

    def test_claim_eligibility(self):
        """docs_stage='reviewed' + score < min_score + chain < cap → claimable."""
        from imas_codex.standard_names.graph_ops import (
            claim_refine_docs_batch,
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                # seed
                [{"_cluster_id": None, "_unit": "eV", "_physics_domain": "cp"}],
                # read-back
                [
                    {
                        "id": "electron_temperature",
                        "description": "Electron temp",
                        "documentation": "Basic docs",
                        "kind": "scalar",
                        "unit": "eV",
                        "cluster_id": None,
                        "physics_domain": ["cp"],
                        "validation_status": "valid",
                        "docs_stage": "refining",
                        "docs_chain_length": 0,
                        "docs_model": "model-a",
                        "docs_generated_at": None,
                        "reviewer_score_docs": 0.5,
                        "reviewer_comments_per_dim_docs": None,
                        "reviewer_comments_docs": None,
                    }
                ],
            ]
        )

        with _patch_gc(gc), _patch_docs_chain_history():
            items = claim_refine_docs_batch(batch_size=1)

        assert len(items) == 1
        assert items[0]["docs_chain_history"] == []

        # Verify WHERE clause in seed query
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "docs_stage = 'reviewed'" in seed_cypher
        assert "reviewer_score_docs" in seed_cypher

    def test_claim_excludes_chain_at_cap(self):
        """docs_chain_length >= rotation_cap → not claimed."""
        from imas_codex.standard_names.graph_ops import (
            claim_refine_docs_batch,
        )

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(side_effect=[[], []])

        with _patch_gc(gc), _patch_docs_chain_history():
            items = claim_refine_docs_batch(rotation_cap=3, batch_size=10)

        assert items == []

        # Verify the WHERE clause includes chain length cap
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "docs_chain_length" in seed_cypher
        assert "$rotation_cap" in seed_cypher

    def test_claim_enriches_chain_history(self):
        """Each claimed item gets docs_chain_history from docs_chain_history()."""
        from imas_codex.standard_names.graph_ops import (
            claim_refine_docs_batch,
        )

        gc, tx = _mock_gc_tx()
        chain = [
            {
                "documentation": "old docs v0",
                "model": "model-a",
                "reviewer_score": 0.4,
                "reviewer_comments_per_dim": {},
                "created_at": "2025-01-10T10:00:00Z",
            }
        ]
        tx.run = MagicMock(
            side_effect=[
                [{"_cluster_id": None, "_unit": "eV", "_physics_domain": "cp"}],
                [
                    {
                        "id": "electron_temperature",
                        "description": "d",
                        "documentation": "doc",
                        "kind": "scalar",
                        "unit": "eV",
                        "cluster_id": None,
                        "physics_domain": ["cp"],
                        "validation_status": "valid",
                        "docs_stage": "refining",
                        "docs_chain_length": 1,
                        "docs_model": "model-a",
                        "docs_generated_at": None,
                        "reviewer_score_docs": 0.5,
                        "reviewer_comments_per_dim_docs": None,
                        "reviewer_comments_docs": None,
                    }
                ],
            ]
        )

        with _patch_gc(gc), _patch_docs_chain_history(return_value=chain):
            items = claim_refine_docs_batch(batch_size=1)

        assert len(items) == 1
        assert items[0]["docs_chain_history"] == chain
        assert len(items[0]["docs_chain_history"]) == 1


# =============================================================================
# 2. Persist tests
# =============================================================================


class TestPersistRefinedDocs:
    """persist_refined_docs creates DocsRevision + updates SN in-place."""

    def test_persist_creates_revision_node(self):
        """After persist, DocsRevision node is created with snapshot of OLD state."""
        from imas_codex.standard_names.graph_ops import persist_refined_docs

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            return_value=[
                {
                    "docs_chain_length": 1,
                    "revision_id": "electron_temperature#rev-0",
                }
            ]
        )

        with _patch_gc(gc):
            result = persist_refined_docs(
                sn_id="electron_temperature",
                claim_token="tok-abc",
                description="Improved description",
                documentation="Improved docs",
                model="model-b",
                current_description="Old description",
                current_documentation="Old docs",
                current_model="model-a",
                current_generated_at="2025-01-15T10:00:00Z",
                reviewer_score_to_snapshot=0.5,
                reviewer_comments_to_snapshot="Needs work",
                reviewer_comments_per_dim_to_snapshot='{"completeness": "Missing"}',
            )

        assert result["docs_chain_length"] == 1
        assert result["revision_id"] == "electron_temperature#rev-0"

        # Verify Cypher was run with correct params
        cypher_call = tx.run.call_args
        params = cypher_call.kwargs
        assert params["sn_id"] == "electron_temperature"
        assert params["cur_desc"] == "Old description"
        assert params["cur_doc"] == "Old docs"
        assert params["snap_score"] == 0.5
        assert params["new_desc"] == "Improved description"
        assert params["new_doc"] == "Improved docs"

    def test_persist_links_revision(self):
        """Cypher creates (sn)-[:DOCS_REVISION_OF]->(rev) edge."""
        from imas_codex.standard_names.graph_ops import persist_refined_docs

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            return_value=[{"docs_chain_length": 1, "revision_id": "test#rev-0"}]
        )

        with _patch_gc(gc):
            persist_refined_docs(
                sn_id="test",
                claim_token="tok",
                description="new desc",
                documentation="new doc",
                model="m",
                current_description="old desc",
                current_documentation="old doc",
            )

        cypher = tx.run.call_args.args[0]
        assert "DOCS_REVISION_OF" in cypher
        assert "MERGE (sn)-[:DOCS_REVISION_OF]->(rev)" in cypher

    def test_persist_advances_stage(self):
        """docs_stage='refining' → 'drafted', docs_chain_length increments."""
        from imas_codex.standard_names.graph_ops import persist_refined_docs

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            return_value=[{"docs_chain_length": 2, "revision_id": "test#rev-1"}]
        )

        with _patch_gc(gc):
            result = persist_refined_docs(
                sn_id="test",
                claim_token="tok",
                description="new desc",
                documentation="new doc",
                model="m",
                current_description="old desc",
                current_documentation="old doc",
            )

        assert result["docs_chain_length"] == 2

        # Verify stage transition in Cypher
        cypher = tx.run.call_args.args[0]
        assert "docs_stage" in cypher
        assert "'drafted'" in cypher

    def test_persist_writes_new_docs(self):
        """SN gets new description and documentation values."""
        from imas_codex.standard_names.graph_ops import persist_refined_docs

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            return_value=[{"docs_chain_length": 1, "revision_id": "test#rev-0"}]
        )

        with _patch_gc(gc):
            persist_refined_docs(
                sn_id="test",
                claim_token="tok",
                description="BRAND NEW description",
                documentation="BRAND NEW documentation",
                model="model-x",
                current_description="old desc",
                current_documentation="old doc",
            )

        params = tx.run.call_args.kwargs
        assert params["new_desc"] == "BRAND NEW description"
        assert params["new_doc"] == "BRAND NEW documentation"
        assert params["model"] == "model-x"

    def test_persist_clears_reviewer_docs_on_sn(self):
        """reviewer_*_docs fields on the SN are cleared after persist."""
        from imas_codex.standard_names.graph_ops import persist_refined_docs

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            return_value=[{"docs_chain_length": 1, "revision_id": "test#rev-0"}]
        )

        with _patch_gc(gc):
            persist_refined_docs(
                sn_id="test",
                claim_token="tok",
                description="new",
                documentation="new doc",
                model="m",
                current_description="old",
                current_documentation="old doc",
                reviewer_score_to_snapshot=0.5,
            )

        cypher = tx.run.call_args.args[0]
        # The Cypher must SET reviewer fields to null on the SN
        assert "sn.reviewer_score_docs" in cypher
        assert "sn.reviewer_comments_docs" in cypher
        assert "null" in cypher

    def test_persist_token_mismatch_no_op(self):
        """Wrong token → no DocsRevision created, SN unchanged."""
        from imas_codex.standard_names.graph_ops import persist_refined_docs

        gc, tx = _mock_gc_tx()
        # Empty result = token didn't match
        tx.run = MagicMock(return_value=[])

        with _patch_gc(gc):
            result = persist_refined_docs(
                sn_id="test",
                claim_token="wrong-token",
                description="new",
                documentation="new doc",
                model="m",
                current_description="old",
                current_documentation="old doc",
            )

        assert result["docs_chain_length"] == -1
        assert result["revision_id"] == ""

    def test_persist_stage_mismatch_no_op(self):
        """docs_stage != 'refining' → no-op (verified in WHERE clause)."""
        from imas_codex.standard_names.graph_ops import persist_refined_docs

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with _patch_gc(gc):
            result = persist_refined_docs(
                sn_id="test",
                claim_token="tok",
                description="new",
                documentation="new doc",
                model="m",
                current_description="old",
                current_documentation="old doc",
            )

        # Verify WHERE clause checks docs_stage = 'refining'
        cypher = tx.run.call_args.args[0]
        assert "docs_stage = 'refining'" in cypher
        assert result["docs_chain_length"] == -1

    def test_persist_idempotent_revision_id(self):
        """Deterministic revision id via MERGE; same persist twice → no duplicate."""
        from imas_codex.standard_names.graph_ops import persist_refined_docs

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            return_value=[{"docs_chain_length": 1, "revision_id": "test#rev-0"}]
        )

        with _patch_gc(gc):
            persist_refined_docs(
                sn_id="test",
                claim_token="tok",
                description="new",
                documentation="new doc",
                model="m",
                current_description="old",
                current_documentation="old doc",
            )

        # Verify MERGE (not CREATE) is used for DocsRevision
        cypher = tx.run.call_args.args[0]
        assert "MERGE (rev:DocsRevision" in cypher
        assert "ON CREATE SET" in cypher


# =============================================================================
# 3. Release tests
# =============================================================================


class TestReleaseRefineDocs:
    """Release functions revert docs_stage and clear claims."""

    def test_release_reverts_refining(self):
        """release_refine_docs_claims reverts docs_stage='refining' → 'reviewed'."""
        from imas_codex.standard_names.graph_ops import (
            release_refine_docs_claims,
        )

        gc = _mock_gc_query()
        gc.query = MagicMock(return_value=[{"released": 1}])

        with _patch_gc(gc):
            released = release_refine_docs_claims(
                sn_ids=["test"],
                claim_token="tok-abc",
            )

        assert released == 1
        cypher = gc.query.call_args.args[0]
        assert "docs_stage = 'refining'" in cypher
        assert "'reviewed'" in cypher

    def test_failed_release_reverts_refining(self):
        """LLM error → docs_stage reverts to 'reviewed', claim_token cleared."""
        from imas_codex.standard_names.graph_ops import (
            release_refine_docs_failed_claims,
        )

        gc = _mock_gc_query()
        gc.query = MagicMock(return_value=[{"released": 2}])

        with _patch_gc(gc):
            released = release_refine_docs_failed_claims(
                sn_ids=["sn1", "sn2"],
                claim_token="tok-xyz",
            )

        assert released == 2
        cypher = gc.query.call_args.args[0]
        assert "docs_stage = 'refining'" in cypher
        assert "'reviewed'" in cypher
        assert "claim_token = null" in cypher

    def test_release_empty_ids_no_op(self):
        """Empty sn_ids → returns 0, no query issued."""
        from imas_codex.standard_names.graph_ops import (
            release_refine_docs_claims,
        )

        gc = _mock_gc_query()
        gc.query = MagicMock(return_value=[{"released": 0}])

        with _patch_gc(gc):
            released = release_refine_docs_claims(
                sn_ids=[],
                claim_token="tok",
            )

        assert released == 0
        gc.query.assert_not_called()


# =============================================================================
# 4. Worker / escalation tests
# =============================================================================


class TestRefineDocsEscalation:
    """Escalation model selection at final chain attempt."""

    def test_escalation_at_final_attempt(self):
        """docs_chain_length=2, rotation_cap=3 → uses DEFAULT_ESCALATION_MODEL."""
        from imas_codex.standard_names.defaults import (
            DEFAULT_ESCALATION_MODEL,
        )

        item = _make_refine_docs_item(docs_chain_length=2)

        # Simulate escalation logic from worker
        rotation_cap = 3
        escalate = item["docs_chain_length"] >= rotation_cap - 1
        assert escalate is True

        if escalate:
            model = DEFAULT_ESCALATION_MODEL
        else:
            model = "google/gemini-3-flash"

        assert model == DEFAULT_ESCALATION_MODEL

    def test_no_escalation_below_cap(self):
        """docs_chain_length=0, rotation_cap=3 → uses standard model."""
        item = _make_refine_docs_item(docs_chain_length=0)

        rotation_cap = 3
        escalate = item["docs_chain_length"] >= rotation_cap - 1
        assert escalate is False


# =============================================================================
# 5. Chain history tests
# =============================================================================


class TestDocsChainHistory:
    """Chain history walking via docs_chain_history()."""

    def test_chain_history_walks_revisions(self):
        """Build chain of 3 revisions; docs_chain_history returns entries."""
        from imas_codex.standard_names.chain_history import docs_chain_history

        mock_rows = [
            {
                "documentation": "docs v0",
                "model": "model-a",
                "reviewer_score": 0.3,
                "reviewer_comments_per_dim": '{"completeness": "bad"}',
                "created_at": "2025-01-01T10:00:00Z",
            },
            {
                "documentation": "docs v1",
                "model": "model-b",
                "reviewer_score": 0.5,
                "reviewer_comments_per_dim": '{"completeness": "ok"}',
                "created_at": "2025-01-02T10:00:00Z",
            },
            {
                "documentation": "docs v2",
                "model": "model-c",
                "reviewer_score": 0.6,
                "reviewer_comments_per_dim": None,
                "created_at": "2025-01-03T10:00:00Z",
            },
        ]

        gc = _mock_gc_query()
        gc.query = MagicMock(return_value=mock_rows)

        with patch(
            "imas_codex.standard_names.chain_history.GraphClient",
            return_value=gc,
        ):
            history = docs_chain_history("test_sn", limit=5)

        assert len(history) == 3
        assert history[0]["documentation"] == "docs v0"
        assert history[1]["documentation"] == "docs v1"
        assert history[2]["documentation"] == "docs v2"
        assert history[0]["reviewer_score"] == 0.3
        assert history[2]["model"] == "model-c"


# =============================================================================
# 6. Round-trip test
# =============================================================================


class TestRefineDocsRoundTrip:
    """Full round-trip: claim → persist → verify state coherent."""

    def test_round_trip(self):
        """persist → assert revision created → assert state coherent."""
        from imas_codex.standard_names.graph_ops import persist_refined_docs

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            return_value=[
                {
                    "docs_chain_length": 1,
                    "revision_id": "electron_temperature#rev-0",
                }
            ]
        )

        with _patch_gc(gc):
            result = persist_refined_docs(
                sn_id="electron_temperature",
                claim_token="tok-rt",
                description="Refined electron temp description",
                documentation="Refined electron temp documentation with LaTeX",
                model="model-b",
                current_description="Old electron temp description",
                current_documentation="Old electron temp docs",
                current_model="model-a",
                current_generated_at="2025-01-15T10:00:00Z",
                reviewer_score_to_snapshot=0.5,
                reviewer_comments_to_snapshot="Needs improvement",
                reviewer_comments_per_dim_to_snapshot='{"phys": "weak"}',
            )

        # Verify result
        assert result["docs_chain_length"] == 1
        assert result["revision_id"] == "electron_temperature#rev-0"

        # Verify transaction was committed
        tx.commit.assert_called_once()

        # Verify Cypher contained all necessary operations
        cypher = tx.run.call_args.args[0]
        assert "DocsRevision" in cypher
        assert "DOCS_REVISION_OF" in cypher
        assert "docs_stage" in cypher
        assert "'drafted'" in cypher
        assert "claim_token" in cypher

        # Verify snapshot params passed correctly
        params = tx.run.call_args.kwargs
        assert params["sn_id"] == "electron_temperature"
        assert params["cur_desc"] == "Old electron temp description"
        assert params["snap_score"] == 0.5
        assert params["new_desc"] == "Refined electron temp description"


# =============================================================================
# 7. Worker batch integration test
# =============================================================================


class TestProcessRefineDocsBatch:
    """process_refine_docs_batch worker tests."""

    @pytest.mark.asyncio
    async def test_worker_calls_persist(self):
        """Worker calls persist_refined_docs for each item."""
        from imas_codex.standard_names.models import RefinedDocs
        from imas_codex.standard_names.workers import (
            process_refine_docs_batch,
        )

        item = _make_refine_docs_item()
        stop = asyncio.Event()

        refined = RefinedDocs(
            description="New description for testing",
            documentation="New docs for testing",
            links=[],
            tags=[],
        )
        llm_out = (refined, 0.05, {"input_tokens": 100, "output_tokens": 50})

        mock_mgr = MagicMock()
        mock_mgr.reserve = MagicMock(return_value=MagicMock())

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                return_value=llm_out,
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="test prompt",
            ),
            patch(
                "imas_codex.settings.get_model",
                return_value="model-x",
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_refined_docs",
                return_value={
                    "docs_chain_length": 1,
                    "revision_id": "electron_temperature#rev-0",
                },
            ) as mock_persist,
        ):
            count = await process_refine_docs_batch([item], mock_mgr, stop)

        assert count == 1
        mock_persist.assert_called_once()

    @pytest.mark.asyncio
    async def test_worker_releases_on_failure(self):
        """Worker calls release_refine_docs_failed_claims on error."""
        from imas_codex.standard_names.workers import (
            process_refine_docs_batch,
        )

        item = _make_refine_docs_item()
        stop = asyncio.Event()

        mock_mgr = MagicMock()
        mock_mgr.reserve = MagicMock(return_value=MagicMock())

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                side_effect=RuntimeError("LLM down"),
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="test prompt",
            ),
            patch(
                "imas_codex.settings.get_model",
                return_value="model-x",
            ),
            patch(
                "imas_codex.standard_names.graph_ops.release_refine_docs_failed_claims",
                return_value=1,
            ) as mock_release,
        ):
            count = await process_refine_docs_batch([item], mock_mgr, stop)

        assert count == 0
        mock_release.assert_called_once()
        call_kwargs = mock_release.call_args.kwargs
        assert call_kwargs["sn_ids"] == ["electron_temperature"]
        assert call_kwargs["claim_token"] == "tok-docs-abc-123"
