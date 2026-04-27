"""Regression tests for Wave-12B review pipeline bugs.

Bug 1: ``reviewer_score`` is an in-memory key, not a graph property.
Bug 2: ``sn review`` CLI was missing ``setup_logging()``.
Bug 3: ``_extend_reservation()`` was silent when extending a lease.
Bug 4: Per-name cost propagation and double-counting prevention.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

# =====================================================================
# Bug 1: reviewer_score is in-memory only — graph writes axis-specific
# =====================================================================


def _review_entry(**overrides: object) -> dict:
    """Minimal review-result dict for axis writer tests."""
    base = {
        "id": "sn:test",
        "reviewer_score": 0.75,
        "reviewed_at": "2026-04-25T00:00:00Z",
        "reviewer_scores": '{"grammar": 15}',
        "reviewer_comments": "ok",
        "reviewer_comments_per_dim": '{"grammar": "fine"}',
        "reviewer_verdict": "accept",
        "reviewer_model": "test-model",
        "review_tier": "good",
        "review_input_hash": "hash123",
    }
    base.update(overrides)
    return base


def _mock_graph_client():
    """Build a mock GraphClient context manager."""
    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[])
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_gc)
    cm.__exit__ = MagicMock(return_value=False)
    return mock_gc, cm


class TestBug1ReviewerScoreNotGraphProperty:
    """reviewer_score must NOT appear as a standalone graph property."""

    def test_name_writer_does_not_set_sn_reviewer_score(self) -> None:
        """write_name_review_results maps reviewer_score → reviewer_score_name,
        never writes a generic ``sn.reviewer_score`` property."""
        from imas_codex.standard_names.graph_ops import write_name_review_results

        mock_gc, cm = _mock_graph_client()
        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=cm):
            write_name_review_results([_review_entry()])

        assert mock_gc.query.called
        cypher = mock_gc.query.call_args[0][0]
        # Must contain axis-specific property
        assert "reviewer_score_name" in cypher
        # Must NOT contain generic property (check for the exact SET pattern)
        assert "sn.reviewer_score =" not in cypher
        assert "sn.reviewer_score=" not in cypher

    def test_docs_writer_does_not_set_sn_reviewer_score(self) -> None:
        """write_docs_review_results maps reviewer_score → reviewer_score_docs,
        never writes a generic ``sn.reviewer_score`` property."""
        from imas_codex.standard_names.graph_ops import write_docs_review_results

        mock_gc, cm = _mock_graph_client()
        # First call: gate check returns reviewed_name_at as non-null
        # Second call: the actual SET
        mock_gc.query = MagicMock(
            side_effect=[
                [{"id": "sn:test", "reviewed_name_at": "2026-04-25T00:00:00Z"}],
                [],
            ]
        )
        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=cm):
            write_docs_review_results([_review_entry()])

        assert mock_gc.query.call_count >= 2
        cypher = mock_gc.query.call_args_list[1][0][0]
        assert "reviewer_score_docs" in cypher
        assert "sn.reviewer_score =" not in cypher
        assert "sn.reviewer_score=" not in cypher

    def test_batch_dict_carries_reviewer_score_name_key(self) -> None:
        """The batch dict sent to Cypher must use reviewer_score_name, not reviewer_score."""
        from imas_codex.standard_names.graph_ops import write_name_review_results

        mock_gc, cm = _mock_graph_client()
        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=cm):
            write_name_review_results([_review_entry(reviewer_score=0.9)])

        batch = mock_gc.query.call_args[1]["batch"]
        assert len(batch) == 1
        assert batch[0]["reviewer_score_name"] == 0.9
        assert "reviewer_score" not in batch[0]


# =====================================================================
# Bug 2: sn review CLI must call setup_logging
# =====================================================================


class TestBug2SetupLogging:
    """sn review must wire up logging so per-phase progress is visible."""

    def test_sn_review_calls_setup_logging(self) -> None:
        """The ``sn review`` Click command must call ``setup_logging``."""
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn_review

        runner = CliRunner()

        with (
            patch("imas_codex.cli.discover.common.setup_logging") as mock_setup,
            patch("imas_codex.standard_names.budget.BudgetManager"),
            patch("imas_codex.standard_names.review.state.StandardNameReviewState"),
        ):
            # --dry-run to avoid the full pipeline running
            runner.invoke(sn_review, ["--dry-run"], catch_exceptions=False)

        # setup_logging must have been called
        mock_setup.assert_called_once()
        call_args = mock_setup.call_args
        assert call_args[0][0] == "sn"
        assert call_args[0][1] == "sn-review"


# =====================================================================
# Bug 3: _extend_reservation must log
# =====================================================================


# =====================================================================
# Bug 4: Per-name cost propagation + skip_cost double-count fix
# =====================================================================


class TestBug4PerNameCost:
    """write_name_review_results must propagate per-name llm_cost_review."""

    def test_name_writer_includes_cost_in_set_clause(self) -> None:
        """Cypher SET must include llm_cost_review accumulation."""
        from imas_codex.standard_names.graph_ops import write_name_review_results

        mock_gc, cm = _mock_graph_client()
        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=cm):
            write_name_review_results([_review_entry(llm_cost=0.005)])

        cypher = mock_gc.query.call_args[0][0]
        assert "llm_cost_review" in cypher
        assert "llm_cost" in cypher

        # Verify batch dict carries cost
        batch = mock_gc.query.call_args[1]["batch"]
        assert batch[0]["llm_cost_review"] == 0.005

    def test_name_writer_cost_defaults_to_zero(self) -> None:
        """When no llm_cost key, batch dict defaults to 0.0."""
        from imas_codex.standard_names.graph_ops import write_name_review_results

        mock_gc, cm = _mock_graph_client()
        entry = _review_entry()
        # No llm_cost key
        assert "llm_cost" not in entry

        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=cm):
            write_name_review_results([entry])

        batch = mock_gc.query.call_args[1]["batch"]
        assert batch[0]["llm_cost_review"] == 0.0

    def test_docs_writer_includes_cost_in_set_clause(self) -> None:
        """Docs-axis writer also propagates per-name cost."""
        from imas_codex.standard_names.graph_ops import write_docs_review_results

        mock_gc, cm = _mock_graph_client()
        mock_gc.query = MagicMock(
            side_effect=[
                [{"id": "sn:test", "reviewed_name_at": "2026-04-25T00:00:00Z"}],
                [],
            ]
        )

        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=cm):
            write_docs_review_results([_review_entry(llm_cost=0.01)])

        # Second call is the SET
        cypher = mock_gc.query.call_args_list[1][0][0]
        assert "llm_cost_review" in cypher

        batch = mock_gc.query.call_args_list[1][1]["batch"]
        assert batch[0]["llm_cost_review"] == 0.01

    def test_write_reviews_skip_cost_prevents_accumulation(self) -> None:
        """write_reviews(skip_cost=True) must NOT run the SN cost query."""
        from imas_codex.standard_names.graph_ops import write_reviews

        mock_gc, cm = _mock_graph_client()
        records = [
            {
                "id": "sn:test:names:g1:0",
                "standard_name_id": "sn:test",
                "model": "test-model",
                "model_family": "test",
                "is_canonical": True,
                "score": 0.8,
                "scores_json": "{}",
                "tier": "good",
                "reviewed_at": "2026-04-25T00:00:00Z",
                "llm_cost": 0.005,
            },
        ]

        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=cm):
            write_reviews(records, skip_cost=True)

        # Only 1 query call (MERGE Review nodes), not 2 (no cost accumulation)
        assert mock_gc.query.call_count == 1
        cypher = mock_gc.query.call_args[0][0]
        assert "MERGE (r:Review" in cypher
        # Should NOT contain the cost accumulation query
        assert "llm_cost_review" not in cypher

    def test_write_reviews_default_does_accumulate_cost(self) -> None:
        """write_reviews() without skip_cost runs the cost accumulation."""
        from imas_codex.standard_names.graph_ops import write_reviews

        mock_gc, cm = _mock_graph_client()
        records = [
            {
                "id": "sn:test:names:g1:0",
                "standard_name_id": "sn:test",
                "model": "test-model",
                "model_family": "test",
                "is_canonical": True,
                "score": 0.8,
                "scores_json": "{}",
                "tier": "good",
                "reviewed_at": "2026-04-25T00:00:00Z",
                "llm_cost": 0.005,
            },
        ]

        with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=cm):
            write_reviews(records, skip_cost=False)

        # Should be 2 calls: MERGE Review + cost accumulation
        assert mock_gc.query.call_count == 2
        cost_cypher = mock_gc.query.call_args_list[1][0][0]
        assert "llm_cost_review" in cost_cypher
