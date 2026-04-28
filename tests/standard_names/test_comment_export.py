"""Tests for pre-clear Review comment export (Phase F).

Validates:
- export_review_comments() writes a JSONL file with the expected schema
- export_review_comments() returns 0 and writes nothing when no Reviews exist
- sn clear respects --no-comment-export to skip the dump
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REVIEW_ROWS = [
    {
        "name": "electron_temperature",
        "domain": "kinetics",
        "reviewer_model": "claude-haiku-4",
        "score": 0.72,
        "verdict": "revise",
        "comments_per_dim_json": '{"grammar": "Name too long", "semantic": "OK"}',
        "comments": "Name too long. Semantic is fine.",
        "review_axis": "names",
        "generated_at": "2025-01-01T00:00:00Z",
        "reviewed_at": "2025-01-02T00:00:00Z",
    },
    {
        "name": "plasma_current",
        "domain": "magnetics",
        "reviewer_model": "claude-haiku-4",
        "score": 0.88,
        "verdict": "accept",
        "comments_per_dim_json": '{"grammar": "Good", "semantic": "Good"}',
        "comments": "Looks great.",
        "review_axis": "names",
        "generated_at": "2025-01-01T00:00:00Z",
        "reviewed_at": "2025-01-02T00:00:00Z",
    },
    {
        "name": "ion_density",
        "domain": "kinetics",
        "reviewer_model": "claude-opus-3",
        "score": 0.55,
        "verdict": "reject",
        "comments_per_dim_json": '{"grammar": "Wrong base term", "semantic": "Misleading"}',
        "comments": "Wrong base term and misleading.",
        "review_axis": "docs",
        "generated_at": "2025-01-01T00:00:00Z",
        "reviewed_at": "2025-01-02T01:00:00Z",
    },
]


# ---------------------------------------------------------------------------
# Unit tests for export_review_comments
# ---------------------------------------------------------------------------


class TestExportReviewComments:
    """Unit tests for graph_ops.export_review_comments."""

    def test_export_dumps_jsonl(self, tmp_path: Path) -> None:
        """Three Review rows should produce three JSONL lines with expected keys."""
        from imas_codex.standard_names.graph_ops import export_review_comments

        output = tmp_path / "comments.jsonl"

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            instance = MagicMock()
            instance.__enter__ = MagicMock(return_value=instance)
            instance.__exit__ = MagicMock(return_value=False)
            instance.query = MagicMock(return_value=_REVIEW_ROWS)
            MockGC.return_value = instance

            count = export_review_comments(output)

        assert count == 3
        assert output.exists()

        lines = output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

        first = json.loads(lines[0])
        assert "name" in first
        assert "domain" in first
        assert "reviewer_model" in first
        assert "score" in first
        assert "verdict" in first
        assert "comments_per_dim" in first
        assert isinstance(first["comments_per_dim"], dict)
        assert "reviewed_at" in first

    def test_export_empty_returns_zero(self, tmp_path: Path) -> None:
        """No Review rows → returns 0, no file written."""
        from imas_codex.standard_names.graph_ops import export_review_comments

        output = tmp_path / "empty.jsonl"

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            instance = MagicMock()
            instance.__enter__ = MagicMock(return_value=instance)
            instance.__exit__ = MagicMock(return_value=False)
            instance.query = MagicMock(return_value=[])
            MockGC.return_value = instance

            count = export_review_comments(output)

        assert count == 0
        assert not output.exists()

    def test_export_domain_filter_passes_to_query(self, tmp_path: Path) -> None:
        """Domain parameter should appear in the Cypher query call."""
        from imas_codex.standard_names.graph_ops import export_review_comments

        output = tmp_path / "filtered.jsonl"
        captured_kwargs: dict = {}

        def _fake_query(cypher, **kwargs):
            captured_kwargs.update(kwargs)
            return _REVIEW_ROWS[:1]

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            instance = MagicMock()
            instance.__enter__ = MagicMock(return_value=instance)
            instance.__exit__ = MagicMock(return_value=False)
            instance.query = _fake_query
            MockGC.return_value = instance

            export_review_comments(output, domain="kinetics")

        assert captured_kwargs.get("domain") == "kinetics"

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directory is created if it doesn't exist."""
        from imas_codex.standard_names.graph_ops import export_review_comments

        nested = tmp_path / "deep" / "nested" / "out.jsonl"

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            instance = MagicMock()
            instance.__enter__ = MagicMock(return_value=instance)
            instance.__exit__ = MagicMock(return_value=False)
            instance.query = MagicMock(return_value=_REVIEW_ROWS[:1])
            MockGC.return_value = instance

            export_review_comments(nested)

        assert nested.exists()

    def test_export_parses_json_string_comments(self, tmp_path: Path) -> None:
        """comments_per_dim_json string is parsed into a dict in the output."""
        from imas_codex.standard_names.graph_ops import export_review_comments

        output = tmp_path / "parsed.jsonl"

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            instance = MagicMock()
            instance.__enter__ = MagicMock(return_value=instance)
            instance.__exit__ = MagicMock(return_value=False)
            instance.query = MagicMock(return_value=_REVIEW_ROWS)
            MockGC.return_value = instance

            export_review_comments(output)

        lines = output.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            rec = json.loads(line)
            assert isinstance(rec["comments_per_dim"], dict)


# ---------------------------------------------------------------------------
# CLI integration tests for sn clear
# ---------------------------------------------------------------------------

_PREVIEW = {
    "StandardName": 2,
    "StandardNameReview": 3,
    "StandardNameSource": 0,
    "VocabGap": 0,
    "SNRun": 0,
}
_DELETED = {
    "StandardName": 2,
    "StandardNameReview": 3,
    "StandardNameSource": 0,
    "VocabGap": 0,
    "SNRun": 0,
}


class TestSnClearCommentExport:
    """Tests for the pre-clear export hook in sn clear."""

    @pytest.fixture()
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_export_triggered_on_clear(self, runner: CliRunner, tmp_path: Path) -> None:
        """sn clear with StandardNameReview nodes exports JSONL before deleting."""
        with (
            patch(
                "imas_codex.standard_names.graph_ops.clear_sn_subsystem",
                side_effect=[_PREVIEW, _DELETED],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.export_review_comments",
                return_value=3,
            ) as mock_export,
            runner.isolated_filesystem(temp_dir=tmp_path),
        ):
            result = runner.invoke(sn, ["clear", "--force"])

        assert result.exit_code == 0, result.output
        mock_export.assert_called_once()
        # Path arg contains 'comments-'
        export_path = str(mock_export.call_args[0][0])
        assert "comments-" in export_path
        assert export_path.endswith(".jsonl")

    def test_no_comment_export_flag_skips_export(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--no-comment-export prevents the JSONL dump."""
        with (
            patch(
                "imas_codex.standard_names.graph_ops.clear_sn_subsystem",
                side_effect=[_PREVIEW, _DELETED],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.export_review_comments",
            ) as mock_export,
            runner.isolated_filesystem(temp_dir=tmp_path),
        ):
            result = runner.invoke(sn, ["clear", "--force", "--no-comment-export"])

        assert result.exit_code == 0
        mock_export.assert_not_called()

    def test_export_skipped_when_no_reviews(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """When StandardNameReview count is 0 the export function is not called."""
        preview_no_reviews = dict(_PREVIEW)
        preview_no_reviews["StandardNameReview"] = 0
        deleted_no_reviews = dict(_DELETED)
        deleted_no_reviews["StandardNameReview"] = 0

        with (
            patch(
                "imas_codex.standard_names.graph_ops.clear_sn_subsystem",
                side_effect=[preview_no_reviews, deleted_no_reviews],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.export_review_comments",
            ) as mock_export,
            runner.isolated_filesystem(temp_dir=tmp_path),
        ):
            result = runner.invoke(sn, ["clear", "--force"])

        assert result.exit_code == 0
        mock_export.assert_not_called()

    def test_export_failure_does_not_abort_clear(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """An export error is warned but clear still proceeds."""
        with (
            patch(
                "imas_codex.standard_names.graph_ops.clear_sn_subsystem",
                side_effect=[_PREVIEW, _DELETED],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.export_review_comments",
                side_effect=OSError("disk full"),
            ),
            runner.isolated_filesystem(temp_dir=tmp_path),
        ):
            result = runner.invoke(sn, ["clear", "--force"])

        assert result.exit_code == 0
        assert "Comment export skipped" in result.output
        assert "Deleted" in result.output
