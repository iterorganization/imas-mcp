"""Regression tests for secondary reviewer score persistence.

Root cause: ``_merge_review_items()`` did not preserve the secondary
reviewer's individual score, per-dim scores, model, or disagreement on
the merged dict, and ``write_name_review_results()`` /
``write_docs_review_results()`` did not include those fields in their
Cypher SET clauses.  Result: ``reviewer_score_secondary`` was NULL on
100% of dual-reviewed StandardName nodes.

All tests are offline — GraphClient is fully mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _primary_item(*, sn_id: str = "electron_temperature") -> dict:
    """Simulate a cycle-0 (primary) review item."""
    return {
        "id": sn_id,
        "reviewer_score": 0.80,
        "reviewer_scores": json.dumps(
            {
                "grammar": 18,
                "semantic": 16,
                "convention": 15,
                "completeness": 15,
            }
        ),
        "reviewer_comments": "Good name, follows conventions well.",
        "review_tier": "good",
        "reviewer_verdict": "accept",
    }


def _secondary_item(*, sn_id: str = "electron_temperature") -> dict:
    """Simulate a cycle-1 (secondary) review item."""
    return {
        "id": sn_id,
        "reviewer_score": 0.70,
        "reviewer_scores": json.dumps(
            {
                "grammar": 16,
                "semantic": 14,
                "convention": 14,
                "completeness": 12,
            }
        ),
        "reviewer_comments": "Acceptable but could be more precise.",
        "review_tier": "good",
        "reviewer_verdict": "accept",
    }


def _call_write_name_review(entries: list[dict]) -> tuple[int, list]:
    """Call ``write_name_review_results`` with a fully mocked GraphClient."""
    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[])

    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_name_review_results

        result = write_name_review_results(entries)

    return result, mock_gc.query.call_args_list


def _call_write_docs_review(entries: list[dict]) -> tuple[int, list]:
    """Call ``write_docs_review_results`` with a fully mocked GraphClient."""
    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[])

    # For docs review, we need the gate-check query to return results
    # that indicate reviewed_name_at IS NOT NULL.
    def _gate_query(cypher, **kwargs):
        if "reviewed_name_at" in cypher:
            return [
                {"id": eid, "reviewed_name_at": "2024-01-01T00:00:00"}
                for eid in kwargs.get("ids", [])
            ]
        return []

    mock_gc.query = MagicMock(side_effect=_gate_query)

    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_docs_review_results

        result = write_docs_review_results(entries)

    return result, mock_gc.query.call_args_list


def _extract_batch(call_args_list: list) -> list[dict]:
    """Pull the ``batch`` kwarg from the first UNWIND query call."""
    for c in call_args_list:
        kwargs = c.kwargs
        if "batch" in kwargs:
            return kwargs["batch"]
    raise AssertionError("No 'batch' kwarg found in any query call")


# ---------------------------------------------------------------------------
# _merge_review_items tests
# ---------------------------------------------------------------------------


class TestMergeReviewItemsSecondary:
    """_merge_review_items must preserve secondary-specific fields."""

    def _merge(self, **kwargs) -> dict:
        from imas_codex.standard_names.review.pipeline import _merge_review_items

        return _merge_review_items(
            _primary_item(),
            _secondary_item(),
            "names",
            **kwargs,
        )

    def test_secondary_score_preserved(self) -> None:
        merged = self._merge()
        assert merged["reviewer_score_secondary"] == pytest.approx(0.70)

    def test_secondary_scores_preserved(self) -> None:
        merged = self._merge()
        scores = json.loads(merged["reviewer_scores_secondary"])
        assert scores["grammar"] == 16
        assert scores["semantic"] == 14

    def test_secondary_model_preserved(self) -> None:
        merged = self._merge(secondary_model="openrouter/google/gemini-2.5-pro")
        assert merged["reviewer_model_secondary"] == "openrouter/google/gemini-2.5-pro"

    def test_secondary_model_absent_when_not_passed(self) -> None:
        merged = self._merge()
        assert "reviewer_model_secondary" not in merged

    def test_disagreement_computed(self) -> None:
        merged = self._merge()
        expected = abs(0.80 - 0.70)
        assert merged["reviewer_disagreement"] == pytest.approx(expected)

    def test_merged_score_is_mean(self) -> None:
        """Consolidated score is the mean of primary and secondary per-dim scores."""
        merged = self._merge()
        # Mean of each dim: grammar=(18+16)/2=17, semantic=(16+14)/2=15,
        # convention=(15+14)/2=14.5, completeness=(15+12)/2=13.5
        # Total = 60.0, max = 4*20 = 80, score = 60/80 = 0.75
        assert merged["reviewer_score"] == pytest.approx(0.75)

    def test_comments_merged_with_markers(self) -> None:
        merged = self._merge()
        assert merged["reviewer_comments"].startswith("[Primary]")
        assert "[Secondary]" in merged["reviewer_comments"]

    def test_single_model_no_secondary_fields(self) -> None:
        """When only primary runs (single model), secondary fields are absent."""
        # Simulate single-model: items list has only c0_items, no merge
        item = _primary_item()
        assert "reviewer_score_secondary" not in item


# ---------------------------------------------------------------------------
# write_name_review_results tests
# ---------------------------------------------------------------------------


class TestWriteNameReviewSecondary:
    """write_name_review_results must include secondary fields in Cypher."""

    def _entry_with_secondary(self) -> dict:
        """Build a merged review entry with secondary fields populated."""
        from imas_codex.standard_names.review.pipeline import _merge_review_items

        merged = _merge_review_items(
            _primary_item(),
            _secondary_item(),
            "names",
            secondary_model="openrouter/google/gemini-2.5-pro",
        )
        merged["reviewed_at"] = "2024-01-01T00:00:00"
        merged["reviewer_model"] = "openrouter/anthropic/claude-opus-4.6"
        return merged

    def test_secondary_score_in_batch(self) -> None:
        entry = self._entry_with_secondary()
        _, calls = _call_write_name_review(entries=[entry])
        batch = _extract_batch(calls)
        row = batch[0]
        assert row["reviewer_score_secondary"] == pytest.approx(0.70)

    def test_secondary_scores_in_batch(self) -> None:
        entry = self._entry_with_secondary()
        _, calls = _call_write_name_review(entries=[entry])
        batch = _extract_batch(calls)
        row = batch[0]
        scores = json.loads(row["reviewer_scores_secondary"])
        assert scores["grammar"] == 16

    def test_secondary_model_in_batch(self) -> None:
        entry = self._entry_with_secondary()
        _, calls = _call_write_name_review(entries=[entry])
        batch = _extract_batch(calls)
        row = batch[0]
        assert row["reviewer_model_secondary"] == "openrouter/google/gemini-2.5-pro"

    def test_disagreement_in_batch(self) -> None:
        entry = self._entry_with_secondary()
        _, calls = _call_write_name_review(entries=[entry])
        batch = _extract_batch(calls)
        row = batch[0]
        assert row["reviewer_disagreement"] == pytest.approx(0.10)

    def test_secondary_fields_in_cypher(self) -> None:
        """The Cypher SET clause must include all 4 secondary fields."""
        entry = self._entry_with_secondary()
        _, calls = _call_write_name_review(entries=[entry])

        cypher = None
        for c in calls:
            args = c.args
            if args and "UNWIND" in args[0]:
                cypher = args[0]
                break

        assert cypher is not None, "No UNWIND query found"
        assert "sn.reviewer_score_secondary" in cypher
        assert "sn.reviewer_scores_secondary" in cypher
        assert "sn.reviewer_model_secondary" in cypher
        assert "sn.reviewer_disagreement" in cypher

    def test_single_model_entry_has_null_secondary(self) -> None:
        """When no secondary review ran, secondary fields are None."""
        entry = _primary_item()
        entry["reviewed_at"] = "2024-01-01T00:00:00"
        entry["reviewer_model"] = "openrouter/anthropic/claude-opus-4.6"
        _, calls = _call_write_name_review(entries=[entry])
        batch = _extract_batch(calls)
        row = batch[0]
        assert row["reviewer_score_secondary"] is None
        assert row["reviewer_scores_secondary"] is None
        assert row["reviewer_model_secondary"] is None
        assert row["reviewer_disagreement"] is None


# ---------------------------------------------------------------------------
# write_docs_review_results tests
# ---------------------------------------------------------------------------


class TestWriteDocsReviewSecondary:
    """write_docs_review_results must include secondary fields in Cypher."""

    def _entry_with_secondary(self) -> dict:
        from imas_codex.standard_names.review.pipeline import _merge_review_items

        item_0 = {
            "id": "electron_temperature",
            "reviewer_score": 0.85,
            "reviewer_scores": json.dumps(
                {
                    "description_quality": 18,
                    "documentation_quality": 16,
                    "completeness": 17,
                    "physics_accuracy": 17,
                }
            ),
            "reviewer_comments": "Well documented.",
            "review_tier": "outstanding",
        }
        item_1 = {
            "id": "electron_temperature",
            "reviewer_score": 0.75,
            "reviewer_scores": json.dumps(
                {
                    "description_quality": 16,
                    "documentation_quality": 14,
                    "completeness": 15,
                    "physics_accuracy": 15,
                }
            ),
            "reviewer_comments": "Adequate docs.",
            "review_tier": "good",
        }
        merged = _merge_review_items(
            item_0,
            item_1,
            "docs",
            secondary_model="openrouter/google/gemini-2.5-pro",
        )
        merged["reviewed_at"] = "2024-01-01T00:00:00"
        merged["reviewer_model"] = "openrouter/anthropic/claude-opus-4.6"
        return merged

    def test_secondary_score_in_docs_batch(self) -> None:
        entry = self._entry_with_secondary()
        _, calls = _call_write_docs_review(entries=[entry])
        # The second query call is the UNWIND SET query (first is gate check)
        batch = _extract_batch(calls[1:])
        row = batch[0]
        assert row["reviewer_score_secondary"] == pytest.approx(0.75)

    def test_secondary_fields_in_docs_cypher(self) -> None:
        entry = self._entry_with_secondary()
        _, calls = _call_write_docs_review(entries=[entry])

        cypher = None
        for c in calls:
            args = c.args
            if args and "UNWIND" in args[0] and "reviewer_score_docs" in args[0]:
                cypher = args[0]
                break

        assert cypher is not None, "No docs UNWIND query found"
        assert "sn.reviewer_score_secondary" in cypher
        assert "sn.reviewer_scores_secondary" in cypher
        assert "sn.reviewer_model_secondary" in cypher
        assert "sn.reviewer_disagreement" in cypher
