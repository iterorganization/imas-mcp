"""Tests for per-dimension reviewer scores, comments, and verdict persistence."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _call_write(names: list[dict], mock_gc: MagicMock) -> int:
    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_standard_names

        return write_standard_names(names)


def _call_write_reviews(reviews: list[dict], mock_gc: MagicMock) -> int:
    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_reviews

        return write_reviews(reviews)


def _find_merge_cypher(mock_gc: MagicMock, needle: str) -> str | None:
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0]
        if needle in cypher:
            return cypher
    return None


def _find_merge_batch(mock_gc: MagicMock, needle: str) -> list[dict] | None:
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0]
        if needle in cypher:
            return c[1].get("batch") or c.kwargs.get("batch")
    return None


class TestQualityCommentModels:
    def test_full_comments_round_trip(self):
        from imas_codex.standard_names.models import StandardNameQualityComments

        c = StandardNameQualityComments(
            grammar="ok",
            semantic="needs improvement",
            documentation=None,
            convention=None,
            completeness=None,
            compliance="follows CF well",
        )
        d = c.model_dump()
        assert d["grammar"] == "ok"
        assert d["documentation"] is None
        assert d["compliance"] == "follows CF well"
        assert StandardNameQualityComments.model_validate(d) == c

    def test_name_only_comments_round_trip(self):
        from imas_codex.standard_names.models import StandardNameQualityCommentsNameOnly

        c = StandardNameQualityCommentsNameOnly(
            grammar="good", semantic=None, convention="minor issue", completeness=None
        )
        d = c.model_dump()
        assert set(d.keys()) == {"grammar", "semantic", "convention", "completeness"}
        assert StandardNameQualityCommentsNameOnly.model_validate(d) == c

    def test_docs_comments_independent_dim_names(self):
        from imas_codex.standard_names.models import StandardNameQualityCommentsDocs

        c = StandardNameQualityCommentsDocs(
            description_quality="clear",
            documentation_quality=None,
            completeness="missing units",
            physics_accuracy=None,
        )
        d = c.model_dump()
        assert "description_quality" in d
        assert "grammar" not in d


class TestQualityReviewWithComments:
    def test_full_review_with_comments(self):
        from imas_codex.standard_names.models import (
            StandardNameQualityComments,
            StandardNameQualityReview,
        )

        review = StandardNameQualityReview(
            source_id="eq/psi",
            standard_name="electron_temperature",
            scores={
                "grammar": 18,
                "semantic": 19,
                "documentation": 17,
                "convention": 20,
                "completeness": 16,
                "compliance": 15,
            },
            comments=StandardNameQualityComments(grammar="good", semantic="accurate"),
            verdict="accept",
            reasoning="Fine overall",
        )
        assert review.comments is not None
        assert review.comments.grammar == "good"
        assert review.comments.documentation is None

    def test_full_review_without_comments(self):
        from imas_codex.standard_names.models import StandardNameQualityReview

        review = StandardNameQualityReview(
            source_id="eq/psi",
            standard_name="electron_temperature",
            scores={
                "grammar": 18,
                "semantic": 19,
                "documentation": 17,
                "convention": 20,
                "completeness": 16,
                "compliance": 15,
            },
            verdict="accept",
            reasoning="Fine overall",
        )
        assert review.comments is None

    def test_docs_review_with_comments(self):
        from imas_codex.standard_names.models import (
            StandardNameQualityCommentsDocs,
            StandardNameQualityReviewDocs,
        )

        review = StandardNameQualityReviewDocs(
            source_id="eq/psi",
            standard_name="electron_temperature",
            scores={
                "description_quality": 18,
                "documentation_quality": 19,
                "completeness": 17,
                "physics_accuracy": 20,
            },
            comments=StandardNameQualityCommentsDocs(
                description_quality="clear", physics_accuracy="spot on"
            ),
            verdict="accept",
            reasoning="Docs are good",
        )
        assert review.comments is not None
        assert review.comments.description_quality == "clear"


class TestWriteStandardNamesPerDim:
    def test_shared_reviewer_slots_not_in_cypher(self):
        """write_standard_names must NOT write any shared reviewer slots.
        Review data is written exclusively via write_name_review_results /
        write_docs_review_results."""
        mock_gc = MagicMock()
        mock_gc.query.return_value = []
        per_dim = {"grammar": "fine", "semantic": "needs work"}
        _call_write(
            [
                {
                    "id": "electron_temperature",
                    "reviewer_comments_per_dim": per_dim,
                    "reviewer_verdict": "accept",
                }
            ],
            mock_gc,
        )
        cypher = _find_merge_cypher(mock_gc, "MERGE (sn:StandardName")
        assert cypher is not None, "MERGE StandardName query not found"
        # Shared slots must not appear — review is axis-specific only
        assert "sn.reviewer_comments_per_dim =" not in cypher
        assert "sn.reviewer_verdict =" not in cypher


class TestWriteReviewsPerDim:
    def test_comments_per_dim_json_in_cypher_and_batch(self):
        mock_gc = MagicMock()
        mock_gc.query.return_value = []
        per_dim = {"grammar": "ok"}
        _call_write_reviews(
            [
                {
                    "id": "electron_temperature:test:2024-01-01",
                    "standard_name_id": "electron_temperature",
                    "model": "test/model",
                    "model_family": "test",
                    "is_canonical": True,
                    "score": 0.85,
                    "scores_json": "{}",
                    "tier": "good",
                    "comments": "all good",
                    "comments_per_dim_json": per_dim,
                    "reviewed_at": "2024-01-01T00:00:00",
                }
            ],
            mock_gc,
        )
        cypher = _find_merge_cypher(mock_gc, "MERGE (r:StandardNameReview")
        assert cypher is not None, "MERGE StandardNameReview query not found"
        assert "r.comments_per_dim_json" in cypher
        batch = _find_merge_batch(mock_gc, "MERGE (r:StandardNameReview")
        assert batch is not None and len(batch) == 1
        assert json.loads(batch[0]["comments_per_dim_json"]) == per_dim


class TestBuildReviewRecordPerDim:
    def test_explicit_comments_per_dim(self):
        from imas_codex.standard_names.review.pipeline import _build_review_record

        reviewed_at = datetime.now(UTC).isoformat()
        per_dim = {"grammar": "fine", "semantic": "accurate"}
        record = _build_review_record(
            {"id": "electron_temperature"},
            model="test/model",
            is_canonical=True,
            reviewed_at=reviewed_at,
            comments_per_dim=per_dim,
        )
        assert record["comments_per_dim_json"] == json.dumps(per_dim)

    def test_comments_per_dim_from_item(self):
        from imas_codex.standard_names.review.pipeline import _build_review_record

        reviewed_at = datetime.now(UTC).isoformat()
        per_dim_json = json.dumps({"grammar": "fine", "semantic": "accurate"})
        record = _build_review_record(
            {"id": "electron_temperature", "reviewer_comments_per_dim": per_dim_json},
            model="test/model",
            is_canonical=True,
            reviewed_at=reviewed_at,
        )
        assert record["comments_per_dim_json"] == per_dim_json

    def test_no_comments_per_dim(self):
        from imas_codex.standard_names.review.pipeline import _build_review_record

        reviewed_at = datetime.now(UTC).isoformat()
        record = _build_review_record(
            {"id": "electron_temperature"},
            model="test/model",
            is_canonical=True,
            reviewed_at=reviewed_at,
        )
        assert record["comments_per_dim_json"] is None


class TestMatchReviewsPerDim:
    def test_reviewer_verdict_and_comments_per_dim_set(self):
        from imas_codex.standard_names.models import (
            StandardNameQualityComments,
            StandardNameQualityReview,
        )
        from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

        entries = [
            {"id": "electron_temperature", "standard_name": "electron_temperature"}
        ]
        review = StandardNameQualityReview(
            source_id="electron_temperature",
            standard_name="electron_temperature",
            scores={
                "grammar": 20,
                "semantic": 20,
                "documentation": 20,
                "convention": 20,
                "completeness": 20,
                "compliance": 20,
            },
            comments=StandardNameQualityComments(grammar="perfect"),
            verdict="accept",
            reasoning="All dimensions outstanding",
        )
        wlog = logging.LoggerAdapter(logging.getLogger("test"))
        scored, unmatched, revised = _match_reviews_to_entries(
            [review], entries, wlog, target="full"
        )
        entry = scored[0]
        assert entry["reviewer_verdict"] == "accept"
        per_dim = json.loads(entry["reviewer_comments_per_dim"])
        assert per_dim["grammar"] == "perfect"
        assert per_dim["semantic"] is None
        assert entry["review_tier"] == "outstanding"

    def test_docs_review_sets_verdict_and_comments(self):
        from imas_codex.standard_names.models import (
            StandardNameQualityCommentsDocs,
            StandardNameQualityReviewDocs,
        )
        from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

        entries = [
            {"id": "electron_temperature", "standard_name": "electron_temperature"}
        ]
        review = StandardNameQualityReviewDocs(
            source_id="electron_temperature",
            standard_name="electron_temperature",
            scores={
                "description_quality": 15,
                "documentation_quality": 14,
                "completeness": 13,
                "physics_accuracy": 16,
            },
            comments=StandardNameQualityCommentsDocs(completeness="missing edge cases"),
            verdict="revise",
            reasoning="Needs more detail",
        )
        wlog = logging.LoggerAdapter(logging.getLogger("test"))
        scored, unmatched, revised = _match_reviews_to_entries(
            [review], entries, wlog, target="docs"
        )
        entry = scored[0]
        assert entry["reviewer_verdict"] == "revise"
        per_dim = json.loads(entry["reviewer_comments_per_dim"])
        assert per_dim["completeness"] == "missing edge cases"
        assert "description_quality" in per_dim
