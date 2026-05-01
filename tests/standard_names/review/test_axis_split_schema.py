"""Axis-split review storage tests.

Validates that name-axis writes never touch docs columns (and vice-versa),
and that shared aggregate slots have been removed from all write paths.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.graph_ops import (
    write_docs_review_results,
    write_name_review_results,
)


def _review_entry(**overrides):
    base = {
        "id": "sn:x",
        "reviewer_score": 0.8,
        "reviewed_at": "2026-04-22T00:00:00Z",
        "reviewer_scores": {"grammar": 16},
        "reviewer_comments": "ok",
        "reviewer_comments_per_dim": {"grammar": "good"},
        "reviewer_verdict": "accept",
        "reviewer_model": "openrouter/anthropic/claude-opus-4.6",
        "review_tier": "good",
        "review_input_hash": "abc",
    }
    base.update(overrides)
    return base


def _capture_cypher_params():
    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[])

    @MagicMock
    def ctx():
        return None

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_gc)
    cm.__exit__ = MagicMock(return_value=False)
    return mock_gc, cm


class TestAxisIsolationNameMode:
    def test_name_mode_sets_name_axis_only(self) -> None:
        mock_gc, cm = _capture_cypher_params()
        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=cm):
            write_name_review_results([_review_entry()])

        assert mock_gc.query.called
        cypher = mock_gc.query.call_args[0][0]
        # Must touch all five name-axis columns
        for col in (
            "reviewer_score_name",
            "reviewer_scores_name",
            "reviewer_comments_name",
            "reviewer_comments_per_dim_name",
            "reviewer_model_name",
        ):
            assert col in cypher, f"expected {col} in SET clause"
        # Must not touch docs-axis columns
        for col in (
            "reviewer_score_docs",
            "reviewer_scores_docs",
            "reviewer_comments_docs",
            "reviewer_comments_per_dim_docs",
            "reviewer_model_docs",
        ):
            assert col not in cypher, f"docs-axis {col} must NOT be touched"

    def test_name_mode_does_not_write_shared_slots(self) -> None:
        """Shared aggregate slots have been removed — name writer must not touch them."""
        mock_gc, cm = _capture_cypher_params()
        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=cm):
            write_name_review_results([_review_entry()])

        cypher = mock_gc.query.call_args[0][0]
        # SET clause must not assign shared slots
        # (note: the word "reviewer_score" appears as prefix of axis columns;
        # assert no standalone `sn.reviewer_score =` or `sn.reviewer_scores =`)
        for bad in (
            "sn.reviewer_score =",
            "sn.reviewer_scores =",
            "sn.reviewer_comments =",
            "sn.reviewer_comments_per_dim =",
            "sn.reviewer_verdict =",
            "sn.reviewer_model =",
        ):
            assert bad not in cypher, f"name mode wrote shared slot: {bad!r}"


class TestAxisIsolationDocsMode:
    def test_docs_mode_sets_docs_axis_only(self) -> None:
        mock_gc, cm = _capture_cypher_params()
        # Docs mode preflights reviewed_name_at; return non-null so entry passes
        mock_gc.query = MagicMock(
            side_effect=[
                [{"id": "sn:x", "reviewed_name_at": "2026-04-22T00:00:00Z"}],
                [],
            ]
        )
        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=cm):
            write_docs_review_results([_review_entry()])

        # Second call is the SET statement we care about
        assert mock_gc.query.call_count >= 2
        cypher = mock_gc.query.call_args_list[1][0][0]
        for col in (
            "reviewer_score_docs",
            "reviewer_scores_docs",
            "reviewer_comments_docs",
            "reviewer_comments_per_dim_docs",
            "reviewer_model_docs",
        ):
            assert col in cypher, f"expected {col} in SET clause"
        for col in (
            "reviewer_score_name",
            "reviewer_scores_name",
            "reviewer_comments_name",
            "reviewer_comments_per_dim_name",
            "reviewer_model_name",
        ):
            assert col not in cypher, f"name-axis {col} must NOT be touched"

    def test_docs_mode_does_not_write_shared_slots(self) -> None:
        mock_gc, cm = _capture_cypher_params()
        mock_gc.query = MagicMock(
            side_effect=[
                [{"id": "sn:x", "reviewed_name_at": "2026-04-22T00:00:00Z"}],
                [],
            ]
        )
        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=cm):
            write_docs_review_results([_review_entry()])

        cypher = mock_gc.query.call_args_list[1][0][0]
        for bad in (
            "sn.reviewer_score =",
            "sn.reviewer_scores =",
            "sn.reviewer_comments =",
            "sn.reviewer_comments_per_dim =",
            "sn.reviewer_verdict =",
            "sn.reviewer_model =",
        ):
            assert bad not in cypher, f"docs mode wrote shared slot: {bad!r}"


class TestPresenceGuard:
    """Pipeline guard blocks same-axis overwrite unless --force."""

    def test_name_axis_blocked_by_existing_name_axis(self) -> None:
        from imas_codex.standard_names.review.pipeline import (
            _axis_overwrite_blocked,
        )

        row = {
            "reviewer_scores_name": {"grammar": 16},
            "reviewer_scores_docs": None,
        }
        assert _axis_overwrite_blocked(row, "name") is True

    def test_docs_axis_blocked_by_existing_docs_axis(self) -> None:
        from imas_codex.standard_names.review.pipeline import (
            _axis_overwrite_blocked,
        )

        row = {
            "reviewer_scores_name": None,
            "reviewer_scores_docs": {"description_quality": 17},
        }
        assert _axis_overwrite_blocked(row, "docs") is True

    def test_docs_axis_not_blocked_by_name_axis(self) -> None:
        """Name-axis presence must not block docs-axis write."""
        from imas_codex.standard_names.review.pipeline import (
            _axis_overwrite_blocked,
        )

        row = {
            "reviewer_scores_name": {"grammar": 16},
            "reviewer_scores_docs": None,
        }
        assert _axis_overwrite_blocked(row, "docs") is False

    def test_name_axis_not_blocked_by_docs_axis(self) -> None:
        """Docs-axis presence must not block name-axis write."""
        from imas_codex.standard_names.review.pipeline import (
            _axis_overwrite_blocked,
        )

        row = {
            "reviewer_scores_name": None,
            "reviewer_scores_docs": {"description_quality": 17},
        }
        assert _axis_overwrite_blocked(row, "name") is False

    def test_empty_row_never_blocks(self) -> None:
        from imas_codex.standard_names.review.pipeline import (
            _axis_overwrite_blocked,
        )

        row = {
            "reviewer_scores_name": None,
            "reviewer_scores_docs": None,
        }
        assert _axis_overwrite_blocked(row, "name") is False
        assert _axis_overwrite_blocked(row, "docs") is False
