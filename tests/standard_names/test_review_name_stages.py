"""Tests for the review_name pipeline (Phase 8.1 stage transitions).

Covers:
- Claim eligibility: only name_stage='drafted' nodes are claimed
- persist_reviewed_name three-way stage decision
  (accepted / reviewed / exhausted)
- Token-mismatch no-op
- Reviewer fields are written to graph
- Failed release reverts claim on LLM error
- Worker uses canonical primary model from settings
- Worker streams per-item progress
- Min-score boundary cases
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, call, patch

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


def _make_reviewed_item(
    sn_id: str = "electron_temperature",
    name_stage: str = "drafted",
    chain_length: int = 0,
    claim_token: str = "tok-review-123",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a claimed-item dict as returned by claim_review_name_batch."""
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
        "name_stage": name_stage,
        "chain_length": chain_length,
        "tags": ["electron", "temperature"],
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


class TestClaimOnlyDrafted:
    """claim_review_name_batch uses name_stage='drafted' as gate."""

    def test_claim_only_drafted(self):
        """The claim WHERE clause gates on name_stage='drafted'."""
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
                claim_review_name_batch,
            )

            claim_review_name_batch(batch_size=5)

        assert len(captured) == 1
        where = captured[0]
        assert "name_stage" in where
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
                claim_review_name_batch,
            )

            claim_review_name_batch(batch_size=5)

        assert kwargs_captured
        kw = kwargs_captured[0]
        # No stage transition should be set (stage_field absent or None)
        assert kw.get("stage_field") is None or kw.get("to_stage") is None


# =============================================================================
# 2. persist_reviewed_name — three-way stage decision
# =============================================================================


class TestPersistToAccepted:
    def test_persist_to_accepted(self):
        """verdict='accept' + score >= min_score → name_stage='accepted'."""
        # First call returns chain_length row, second is the SET call
        gc = _mock_gc_query(
            return_values=[
                [{"chain_length": 0}],  # readback
                [],  # SET write
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            result = persist_reviewed_name(
                sn_id="electron_temperature",
                claim_token="tok-123",
                score=0.9,
                scores={"grammar": 18},
                comments="Excellent.",
                comments_per_dim={"grammar": "Great"},
                model="test/model",
                min_score=0.75,
                rotation_cap=3,
            )

        assert result == "accepted"

    def test_persist_to_accepted_writes_stage(self):
        """The Cypher SET includes name_stage='accepted'."""
        gc = _mock_gc_query(
            return_values=[
                [{"chain_length": 0}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            persist_reviewed_name(
                sn_id="electron_temperature",
                claim_token="tok-123",
                score=0.9,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        # Second call is the SET query
        set_call_kwargs = gc.query.call_args_list[1]
        assert "accepted" in str(set_call_kwargs)


class TestPersistToReviewed:
    def test_persist_to_reviewed_low_score(self):
        """score=0.5, chain_length=0, rotation_cap=3 → 'reviewed'."""
        gc = _mock_gc_query(
            return_values=[
                [{"chain_length": 0}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            result = persist_reviewed_name(
                sn_id="test_name",
                claim_token="tok",
                score=0.5,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        assert result == "reviewed"

    def test_persist_to_reviewed_below_cap(self):
        """score=0.5, chain_length=1, rotation_cap=3 → 'reviewed' (not yet at cap)."""
        gc = _mock_gc_query(
            return_values=[
                [{"chain_length": 1}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            result = persist_reviewed_name(
                sn_id="test_name",
                claim_token="tok",
                score=0.5,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        assert result == "reviewed"


class TestPersistToExhausted:
    def test_persist_to_exhausted_at_cap(self):
        """score=0.5, chain_length=2, rotation_cap=3 → 'exhausted'."""
        gc = _mock_gc_query(
            return_values=[
                [{"chain_length": 2}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            result = persist_reviewed_name(
                sn_id="test_name",
                claim_token="tok",
                score=0.5,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        assert result == "exhausted"

    def test_accept_overrides_chain_length(self):
        """verdict='accept' with chain_length=2, rotation_cap=3 → 'accepted' (acceptance wins)."""
        gc = _mock_gc_query(
            return_values=[
                [{"chain_length": 2}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            result = persist_reviewed_name(
                sn_id="test_name",
                claim_token="tok",
                score=0.9,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        assert result == "accepted"


class TestScoreCanonicalPolicy:
    """Score is authoritative; verdict is informational only.

    Regression: 14 SNs were stuck in 'reviewed' with rsn>=0.75 because the
    reviewer LLM returned verdict='revise' even at high scores. The old
    AND-gate (verdict=='accept' AND score>=min_score) stranded them above
    refine threshold but below promotion gate.
    """

    def test_revise_with_high_score_promotes_to_accepted(self):
        """score=0.9 → 'accepted' (score wins)."""
        gc = _mock_gc_query(return_values=[[{"chain_length": 0}], []])
        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            result = persist_reviewed_name(
                sn_id="time",
                claim_token="tok",
                score=0.9,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )
        assert result == "accepted"

    def test_reject_with_high_score_promotes_to_accepted(self):
        """verdict='reject' with score>=min_score still promotes (rubric is canonical)."""
        gc = _mock_gc_query(return_values=[[{"chain_length": 0}], []])
        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            result = persist_reviewed_name(
                sn_id="x",
                claim_token="tok",
                score=0.8,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )
        assert result == "accepted"

    def test_high_score_at_cap_still_accepts(self):
        """score>=min_score at chain_length>=cap-1 → accepted (not exhausted)."""
        gc = _mock_gc_query(return_values=[[{"chain_length": 2}], []])
        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            result = persist_reviewed_name(
                sn_id="x",
                claim_token="tok",
                score=0.85,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )
        assert result == "accepted"


class TestPersistTokenMismatch:
    def test_persist_token_mismatch_no_op(self):
        """Wrong claim_token → returns '' and no SET is executed."""
        # Token mismatch: first query returns empty rows
        gc = _mock_gc_query(return_values=[[]])

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            result = persist_reviewed_name(
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


class TestPersistWritesReviewerFields:
    def test_persist_writes_reviewer_fields(self):
        """score, scores (JSON), comments, comments_per_dim (JSON), verdict, model, reviewed_at populated."""
        gc = _mock_gc_query(
            return_values=[
                [{"chain_length": 0}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            persist_reviewed_name(
                sn_id="electron_temperature",
                claim_token="tok",
                score=0.8,
                scores={"grammar": 16, "semantic": 18},
                comments="Good name.",
                comments_per_dim={"grammar": "OK", "semantic": "Great"},
                model="openrouter/test/model",
                min_score=0.75,
                rotation_cap=3,
            )

        # Check the SET query kwargs
        set_call = gc.query.call_args_list[1]
        # The second positional/keyword argument set should contain reviewer fields
        call_kwargs = set_call[1]  # keyword args dict

        assert call_kwargs.get("score") == 0.8
        assert call_kwargs.get("model") == "openrouter/test/model"
        assert call_kwargs.get("comments") == "Good name."

        # Scores and comments_per_dim should be JSON strings
        scores_json = call_kwargs.get("scores_json")
        assert scores_json is not None
        scores_parsed = json.loads(scores_json)
        assert scores_parsed.get("grammar") == 16

        cpd_json = call_kwargs.get("comments_per_dim_json")
        assert cpd_json is not None
        cpd_parsed = json.loads(cpd_json)
        assert cpd_parsed.get("grammar") == "OK"

        # Cypher should include reviewed_name_at and name_stage
        cypher = set_call[0][0]  # first positional arg is the cypher string
        assert "reviewed_name_at" in cypher
        assert "name_stage" in cypher
        assert "reviewer_score_name" in cypher


# =============================================================================
# 3. Min-score boundary cases
# =============================================================================


class TestMinScoreThreshold:
    def test_score_equals_min_score_with_accept_verdict(self):
        """score == min_score with verdict='accept' → 'accepted'."""
        gc = _mock_gc_query(return_values=[[{"chain_length": 0}], []])
        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            result = persist_reviewed_name(
                sn_id="x",
                claim_token="t",
                score=0.75,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )
        assert result == "accepted"

    def test_score_just_below_min_score(self):
        """score < min_score with verdict='accept' → stage depends on chain_length."""
        gc = _mock_gc_query(return_values=[[{"chain_length": 0}], []])
        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            result = persist_reviewed_name(
                sn_id="x",
                claim_token="t",
                score=0.74,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )
        # score < min_score so accept condition fails → reviewed (chain_length=0 < cap-1=2)
        assert result == "reviewed"

    def test_score_below_min_at_rotation_cap(self):
        """score < min_score at chain_length=rotation_cap-1 → 'exhausted'."""
        gc = _mock_gc_query(return_values=[[{"chain_length": 2}], []])
        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            result = persist_reviewed_name(
                sn_id="x",
                claim_token="t",
                score=0.5,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )
        assert result == "exhausted"


# =============================================================================
# 4. Worker tests
# =============================================================================


class TestWorkerUsesCanonicalModel:
    def test_worker_uses_canonical_model(self, mock_llm):
        """process_review_name_batch uses primary-model from settings."""
        from imas_codex.standard_names.models import (
            StandardNameQualityReviewNameOnly,
            StandardNameQualityScoreNameOnly,
        )

        review_response = StandardNameQualityReviewNameOnly(
            source_id="electron_temperature",
            standard_name="electron_temperature",
            scores=StandardNameQualityScoreNameOnly(
                grammar=16, semantic=18, convention=17, completeness=16
            ),
            reasoning="Good name.",
        )
        mock_llm.add_response("review_name", response=review_response, model=None)

        # Patch persist to avoid graph calls
        with (
            patch(
                "imas_codex.settings.get_sn_review_names_models",
                return_value=["openrouter/test/review-model"],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_reviewed_name",
                return_value="accepted",
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="Review this name.",
            ),
        ):
            from imas_codex.standard_names.workers import process_review_name_batch

            items = [_make_reviewed_item()]
            mgr = _mock_budget_manager()

            result = asyncio.run(process_review_name_batch(items, mgr, asyncio.Event()))

        assert result == 1
        # Check that the LLM call used the canonical model
        assert mock_llm.calls_for("review_name") == 1
        lm_call = mock_llm.calls[0]
        assert lm_call["model"] == "openrouter/test/review-model"

    def test_worker_streams_per_item(self, mock_llm):
        """process_review_name_batch returns one processed item per SN."""
        from imas_codex.standard_names.models import (
            StandardNameQualityReviewNameOnly,
            StandardNameQualityScoreNameOnly,
        )

        for i in range(3):
            mock_llm.add_response(
                "review_name",
                response=StandardNameQualityReviewNameOnly(
                    source_id=f"sn_{i}",
                    standard_name=f"sn_{i}",
                    scores=StandardNameQualityScoreNameOnly(
                        grammar=16, semantic=18, convention=17, completeness=16
                    ),
                    reasoning=f"Good {i}.",
                ),
            )

        items = [
            _make_reviewed_item(sn_id=f"sn_{i}", claim_token=f"tok-{i}")
            for i in range(3)
        ]

        with (
            patch(
                "imas_codex.settings.get_sn_review_names_models",
                return_value=["openrouter/test/model"],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_reviewed_name",
                return_value="accepted",
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="Review.",
            ),
        ):
            from imas_codex.standard_names.workers import process_review_name_batch

            result = asyncio.run(
                process_review_name_batch(
                    items, _mock_budget_manager(), asyncio.Event()
                )
            )

        assert result == 3


class TestFailedReleaseRevertsTo:
    def test_failed_release_reverts_drafted(self, mock_llm):
        """LLM error path calls release_review_names_failed_claims with from_stage/to_stage='drafted'."""
        _ = mock_llm  # consume fixture — will raise RuntimeError for missing response

        release_calls: list[dict] = []

        def _fake_release(**kwargs):
            release_calls.append(kwargs)
            return 1

        with (
            patch(
                "imas_codex.settings.get_sn_review_names_models",
                return_value=["openrouter/test/model"],
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="Review.",
            ),
            patch(
                "imas_codex.standard_names.graph_ops.release_review_names_failed_claims",
                side_effect=_fake_release,
            ),
        ):
            from imas_codex.standard_names.workers import process_review_name_batch

            items = [_make_reviewed_item(claim_token="tok-failed")]
            mgr = _mock_budget_manager()

            # MockLLM will raise RuntimeError (no scripted response)
            result = asyncio.run(process_review_name_batch(items, mgr, asyncio.Event()))

        assert result == 0  # nothing processed on error
        assert len(release_calls) == 1
        rc = release_calls[0]
        assert rc.get("from_stage") == "drafted"
        assert rc.get("to_stage") == "drafted"
        assert "tok-failed" in str(rc.get("claim_token", ""))


# =============================================================================
# 5. Quarantined-skip optimization
# =============================================================================


class TestQuarantinedSkipsLLM:
    def test_quarantined_skips_llm_and_persists_zero_score(self, mock_llm):
        """Quarantined names skip the reviewer LLM and persist score=0 directly."""
        persist_calls: list[dict[str, Any]] = []

        def _fake_persist(**kwargs):
            persist_calls.append(kwargs)
            return "exhausted"

        with (
            patch(
                "imas_codex.settings.get_sn_review_names_models",
                return_value=["openrouter/test/review-model"],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_reviewed_name",
                side_effect=_fake_persist,
            ),
        ):
            from imas_codex.standard_names.workers import process_review_name_batch

            items = [
                _make_reviewed_item(
                    sn_id="bad_name",
                    validation_status="quarantined",
                    claim_token="tok-q",
                )
            ]
            mgr = _mock_budget_manager()

            result = asyncio.run(process_review_name_batch(items, mgr, asyncio.Event()))

        # Persisted once, with score 0 and skip-marker model name
        assert result == 1
        assert len(persist_calls) == 1
        kw = persist_calls[0]
        assert kw["sn_id"] == "bad_name"
        assert kw["claim_token"] == "tok-q"
        assert kw["score"] == 0.0
        assert kw["model"].startswith("(skipped")
        # No LLM call was issued for the quarantined item
        assert mock_llm.calls_for("review_name") == 0

    def test_valid_status_still_calls_llm(self, mock_llm):
        """Names with validation_status='valid' must NOT be skipped."""
        from imas_codex.standard_names.models import (
            StandardNameQualityReviewNameOnly,
            StandardNameQualityScoreNameOnly,
        )

        review_response = StandardNameQualityReviewNameOnly(
            source_id="electron_temperature",
            standard_name="electron_temperature",
            scores=StandardNameQualityScoreNameOnly(
                grammar=16, semantic=18, convention=17, completeness=16
            ),
            reasoning="ok",
        )
        mock_llm.add_response("review_name", response=review_response, model=None)

        with (
            patch(
                "imas_codex.settings.get_sn_review_names_models",
                return_value=["openrouter/test/review-model"],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_reviewed_name",
                return_value="accepted",
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="Review this name.",
            ),
        ):
            from imas_codex.standard_names.workers import process_review_name_batch

            items = [_make_reviewed_item(validation_status="valid")]
            mgr = _mock_budget_manager()

            result = asyncio.run(process_review_name_batch(items, mgr, asyncio.Event()))

        assert result == 1
        assert mock_llm.calls_for("review_name") == 1


# =============================================================================
# StandardNameReview node persistence (Finding 1 — was missing entirely)
# =============================================================================


class TestPersistWritesReviewNode:
    def test_persist_reviewed_name_calls_write_reviews(self):
        """persist_reviewed_name must MERGE a :StandardNameReview node via write_reviews."""
        gc = _mock_gc_query(return_values=[[{"chain_length": 0}], []])

        with (
            _patch_gc(gc),
            patch(
                "imas_codex.standard_names.graph_ops.write_reviews"
            ) as mock_write_reviews,
        ):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            persist_reviewed_name(
                sn_id="electron_temperature",
                claim_token="tok",
                score=0.82,
                scores={"grammar": 16, "semantic": 17},
                comments="Solid name.",
                comments_per_dim={"grammar": "ok"},
                model="openrouter/anthropic/claude-opus-4.6",
                min_score=0.75,
                rotation_cap=3,
                llm_cost=0.025,
                llm_tokens_in=1234,
                llm_tokens_out=200,
                llm_tokens_cached_read=900,
                llm_tokens_cached_write=300,
                llm_service="standard-names",
            )

        assert mock_write_reviews.called, (
            "persist_reviewed_name must call write_reviews to create StandardNameReview"
        )
        records = mock_write_reviews.call_args[0][0]
        assert len(records) == 1
        rec = records[0]
        assert rec["standard_name_id"] == "electron_temperature"
        assert rec["review_axis"] == "names"
        assert rec["cycle_index"] == 0
        assert rec["resolution_role"] == "primary"
        assert rec["is_canonical"] is True
        assert rec["score"] == 0.82
        assert rec["tier"] == "good"  # 0.82 ∈ [0.6, 0.85)
        assert rec["llm_cost"] == 0.025
        assert rec["llm_tokens_in"] == 1234
        assert rec["llm_tokens_out"] == 200
        assert rec["llm_tokens_cached_read"] == 900
        assert rec["llm_tokens_cached_write"] == 300
        # Composite id format
        assert rec["id"].startswith("electron_temperature:names:")
        assert rec["id"].endswith(":0")

    def test_persist_reviewed_docs_calls_write_reviews(self):
        """persist_reviewed_docs must MERGE a :StandardNameReview node with axis='docs'."""
        gc = _mock_gc_query(return_values=[[{"docs_chain_length": 0}], []])

        with (
            _patch_gc(gc),
            patch(
                "imas_codex.standard_names.graph_ops.write_reviews"
            ) as mock_write_reviews,
        ):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            persist_reviewed_docs(
                sn_id="electron_temperature",
                claim_token="tok",
                score=0.91,
                scores={"description_quality": 19},
                comments="Outstanding docs.",
                comments_per_dim=None,
                model="openrouter/anthropic/claude-opus-4.6",
                llm_cost=0.04,
                llm_tokens_in=2000,
                llm_tokens_out=300,
            )

        assert mock_write_reviews.called
        rec = mock_write_reviews.call_args[0][0][0]
        assert rec["review_axis"] == "docs"
        assert rec["tier"] == "outstanding"  # 0.91 >= 0.85
        assert rec["id"].startswith("electron_temperature:docs:")
        assert rec["llm_cost"] == 0.04

    def test_persist_does_not_fail_when_write_reviews_raises(self):
        """A failure inside write_reviews must NOT block the stage transition."""
        gc = _mock_gc_query(return_values=[[{"chain_length": 0}], []])

        with (
            _patch_gc(gc),
            patch(
                "imas_codex.standard_names.graph_ops.write_reviews",
                side_effect=RuntimeError("graph down"),
            ),
        ):
            from imas_codex.standard_names.graph_ops import persist_reviewed_name

            new_stage = persist_reviewed_name(
                sn_id="x",
                claim_token="tok",
                score=0.9,
                model="m",
                min_score=0.75,
                rotation_cap=3,
            )

        # Stage transition must still succeed even if review-node write fails
        assert new_stage == "accepted"
