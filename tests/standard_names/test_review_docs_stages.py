"""Tests for the review_docs pipeline (Phase 8.1 stage transitions).

Covers:
- Claim eligibility: only docs_stage='drafted' nodes are claimed
- persist_reviewed_docs three-way stage decision
  (accepted / reviewed / exhausted)
- Token-mismatch no-op
- Reviewer docs fields are written to graph
- Name reviewer fields are unchanged after docs review
- Failed release reverts claim on LLM error
- Worker streams per-item progress
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Shared helpers / paths
# =============================================================================

_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"


def _mock_gc_query(return_values: list[list[dict]] | None = None):
    """Return a mock GraphClient whose .query() returns successive values."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    if return_values is not None:
        gc.query = MagicMock(side_effect=return_values)
    else:
        gc.query = MagicMock(return_value=[])
    return gc


@contextmanager
def _patch_gc(gc):
    with patch(_GC_PATH, return_value=gc):
        yield


def _make_docs_item(
    sn_id: str = "electron_temperature",
    docs_stage: str = "drafted",
    docs_chain_length: int = 0,
    claim_token: str = "tok-review-docs-123",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a claimed-item dict as returned by claim_review_docs_batch."""
    item: dict[str, Any] = {
        "id": sn_id,
        "name": sn_id,
        "description": "Electron temperature profile",
        "documentation": "The electron temperature $T_e$.",
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": "valid",
        "docs_stage": docs_stage,
        "docs_chain_length": docs_chain_length,
        "claim_token": claim_token,
    }
    item.update(overrides)
    return item


def _mock_budget_manager() -> MagicMock:
    from types import SimpleNamespace

    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
    lease.release_unused = MagicMock(return_value=0.0)
    mgr.reserve = MagicMock(return_value=lease)
    return mgr


# =============================================================================
# 1. Claim eligibility
# =============================================================================


class TestClaimOnlyDraftedDocs:
    """claim_review_docs_batch uses docs_stage='drafted' as gate."""

    def test_claim_only_drafted_docs(self):
        """The claim WHERE clause gates on docs_stage='drafted'."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        captured: list[str] = []

        def _fake_claim_sn_atomic(
            *,
            eligibility_where: str,
            **kwargs: Any,
        ) -> list[dict]:
            captured.append(eligibility_where)
            return []

        with patch(
            "imas_codex.standard_names.graph_ops._claim_sn_atomic",
            side_effect=_fake_claim_sn_atomic,
        ):
            from imas_codex.standard_names.graph_ops import (
                claim_review_docs_batch,
            )

            claim_review_docs_batch(batch_size=5)

        assert len(captured) == 1
        where = captured[0]
        assert "docs_stage" in where
        assert "'drafted'" in where

    def test_claim_does_not_transition_stage(self):
        """Stage is NOT transitioned at claim time (stage_field is None/not set)."""
        kwargs_captured: list[dict] = []

        def _fake_claim_sn_atomic(**kwargs: Any) -> list[dict]:
            kwargs_captured.append(kwargs)
            return []

        with patch(
            "imas_codex.standard_names.graph_ops._claim_sn_atomic",
            side_effect=_fake_claim_sn_atomic,
        ):
            from imas_codex.standard_names.graph_ops import (
                claim_review_docs_batch,
            )

            claim_review_docs_batch(batch_size=5)

        assert kwargs_captured
        kw = kwargs_captured[0]
        assert kw.get("stage_field") is None or kw.get("to_stage") is None


class TestClaimSkipsPendingDocs:
    """SN with docs_stage='pending' must NOT be claimed."""

    def test_claim_skips_pending_docs(self):
        """docs_stage='pending' is excluded from WHERE clause."""
        captured: list[str] = []

        def _fake_claim_sn_atomic(
            *,
            eligibility_where: str,
            **kwargs: Any,
        ) -> list[dict]:
            captured.append(eligibility_where)
            return []

        with patch(
            "imas_codex.standard_names.graph_ops._claim_sn_atomic",
            side_effect=_fake_claim_sn_atomic,
        ):
            from imas_codex.standard_names.graph_ops import (
                claim_review_docs_batch,
            )

            claim_review_docs_batch(batch_size=5)

        assert captured
        where = captured[0]
        # The gate is strictly on 'drafted' — 'pending' is not included
        assert "'pending'" not in where
        assert "'drafted'" in where


# =============================================================================
# 2. persist_reviewed_docs — three-way stage decision
# =============================================================================


class TestPersistToAccepted:
    def test_persist_to_accepted(self):
        """verdict='accept' + score >= min_score → docs_stage='accepted'."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 0}],  # readback
                [],  # SET write
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="electron_temperature",
                claim_token="tok-123",
                score=0.9,
                scores={"description_quality": 18},
                comments="Excellent docs.",
                comments_per_dim={"description_quality": "Clear"},
                model="test/model",
                min_score=0.75,
                rotation_cap=3,
            )

        assert result == "accepted"


class TestPersistToReviewedLowScore:
    def test_persist_to_reviewed_low_score(self):
        """score=0.5, docs_chain_length=0, rotation_cap=3 → 'reviewed'."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 0}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="test_name",
                claim_token="tok",
                score=0.5,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        assert result == "reviewed"


class TestPersistToExhaustedAtCap:
    def test_persist_to_reviewed_at_escalator_attempt(self):
        """score=0.5, docs_chain_length=2, rotation_cap=3 → 'reviewed'.

        Pre-fix this returned 'exhausted', pre-empting the Opus
        escalator in process_refine_docs_batch (which fires at
        docs_chain_length == rotation_cap-1).  Post-fix the SN stays
        'reviewed' so the final escalated refine attempt can fire.
        """
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 2}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="test_name",
                claim_token="tok",
                score=0.5,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        assert result == "reviewed"

    def test_persist_to_exhausted_post_escalator(self):
        """score=0.5, docs_chain_length=3, rotation_cap=3 → 'exhausted'.

        After the escalated final refine has produced a chain=3 SN,
        the next review step must mark it exhausted.
        """
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 3}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="test_name",
                claim_token="tok",
                score=0.5,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        assert result == "exhausted"


class TestPersistToReviewedBelowCap:
    def test_persist_to_reviewed_below_cap(self):
        """score=0.5, docs_chain_length=1, rotation_cap=3 → 'reviewed' (not yet at cap)."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 1}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="test_name",
                claim_token="tok",
                score=0.5,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        assert result == "reviewed"


class TestAcceptOverridesChainLengthAtCap:
    def test_accept_overrides_chain_length_at_cap(self):
        """docs_chain_length=3, rotation_cap=3 → 'accepted' (acceptance wins)."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 3}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="test_name",
                claim_token="tok",
                score=0.9,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        assert result == "accepted"


class TestPersistTokenMismatchNoOp:
    def test_persist_token_mismatch_no_op(self):
        """Wrong claim_token → returns '' and no SET is executed."""
        # Token mismatch: first query returns empty rows
        gc = _mock_gc_query(return_values=[[]])

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="test_name",
                claim_token="wrong-token",
                score=0.9,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        assert result == ""
        # Only one query (the readback) should have been called
        assert gc.query.call_count == 1


class TestPersistWritesReviewerDocsFields:
    def test_persist_writes_reviewer_docs_fields(self):
        """All reviewer_*_docs fields are populated in the SET call."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 0}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            persist_reviewed_docs(
                sn_id="electron_temperature",
                claim_token="tok",
                score=0.8,
                scores={"description_quality": 16, "documentation_quality": 18},
                comments="Good docs.",
                comments_per_dim={
                    "description_quality": "OK",
                    "documentation_quality": "Great",
                },
                model="openrouter/test/model",
                min_score=0.75,
                rotation_cap=3,
            )

        # Check the SET query kwargs
        set_call = gc.query.call_args_list[1]
        call_kwargs = set_call[1]  # keyword args dict

        assert call_kwargs.get("score") == 0.8
        assert call_kwargs.get("model") == "openrouter/test/model"
        assert call_kwargs.get("comments") == "Good docs."

        # Scores and comments_per_dim should be JSON strings
        scores_json = call_kwargs.get("scores_json")
        assert scores_json is not None
        scores_parsed = json.loads(scores_json)
        assert scores_parsed.get("description_quality") == 16

        cpd_json = call_kwargs.get("comments_per_dim_json")
        assert cpd_json is not None
        cpd_parsed = json.loads(cpd_json)
        assert cpd_parsed.get("description_quality") == "OK"

        # Cypher should include reviewed_docs_at, docs_stage, reviewer_score_docs
        cypher = set_call[0][0]  # first positional arg is the cypher string
        assert "reviewed_docs_at" in cypher
        assert "docs_stage" in cypher
        assert "reviewer_score_docs" in cypher


class TestPersistDoesNotChangeNameFields:
    def test_persist_does_not_change_name_fields(self):
        """Reviewer name fields are NOT written during docs review."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 0}],
                [],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            persist_reviewed_docs(
                sn_id="electron_temperature",
                claim_token="tok",
                score=0.8,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        set_call = gc.query.call_args_list[1]
        cypher = set_call[0][0]

        # Docs review must NOT set name-axis fields.
        # name_stage may appear in the WHERE clause as a filter — only check the SET block.
        set_part = cypher.split("SET", 1)[1] if "SET" in cypher else cypher
        assert "name_stage" not in set_part
        assert "reviewer_score_name" not in cypher
        assert "reviewer_comments_name" not in cypher
        assert "reviewed_name_at" not in cypher


# =============================================================================
# 3. Worker tests
# =============================================================================


class TestFailedReleaseKeepsDrafted:
    def test_failed_release_keeps_drafted(self, mock_llm):
        """LLM error path calls release_review_docs_failed_claims with from/to_stage='drafted'."""
        _ = mock_llm  # consume fixture — will raise RuntimeError for missing response

        release_calls: list[dict] = []

        def _fake_release(**kwargs):
            release_calls.append(kwargs)
            return 1

        with (
            patch(
                "imas_codex.settings.get_sn_review_docs_models",
                return_value=["openrouter/test/model"],
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="Review.",
            ),
            patch(
                "imas_codex.standard_names.graph_ops.release_review_docs_failed_claims",
                side_effect=_fake_release,
            ),
        ):
            from imas_codex.standard_names.workers import process_review_docs_batch

            items = [_make_docs_item(claim_token="tok-failed")]
            mgr = _mock_budget_manager()

            result = asyncio.run(process_review_docs_batch(items, mgr, asyncio.Event()))

        assert result == 0  # nothing processed on error
        assert len(release_calls) == 1
        rc = release_calls[0]
        assert rc.get("from_stage") == "drafted"
        assert rc.get("to_stage") == "drafted"
        assert "tok-failed" in str(rc.get("claim_token", ""))


class TestWorkerStreamsPerItemDocs:
    def test_worker_streams_per_item(self, mock_llm):
        """process_review_docs_batch returns one processed item per SN."""
        from imas_codex.standard_names.models import (
            StandardNameQualityCommentsDocs,
            StandardNameQualityReviewDocs,
            StandardNameQualityScoreDocs,
        )

        for i in range(3):
            mock_llm.add_response(
                "review_docs",
                response=StandardNameQualityReviewDocs(
                    source_id=f"sn_{i}",
                    standard_name=f"sn_{i}",
                    scores=StandardNameQualityScoreDocs(
                        description_quality=16,
                        documentation_quality=18,
                        completeness=17,
                        physics_accuracy=16,
                    ),
                    reasoning=f"Good docs {i}.",
                ),
            )

        items = [
            _make_docs_item(sn_id=f"sn_{i}", claim_token=f"tok-{i}") for i in range(3)
        ]

        with (
            patch(
                "imas_codex.settings.get_sn_review_docs_models",
                return_value=["openrouter/test/model"],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_reviewed_docs",
                return_value="accepted",
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="documentation quality review.",
            ),
        ):
            from imas_codex.standard_names.workers import process_review_docs_batch

            result = asyncio.run(
                process_review_docs_batch(
                    items, _mock_budget_manager(), asyncio.Event()
                )
            )

        assert result == 3
