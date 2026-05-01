"""Tests for ``sn analyse-comments`` CLI command (Phase F).

Validates:
- Command reads JSONL glob and emits a markdown report with expected sections.
- Handles empty input gracefully.
- Respects --domain filter.
- Respects --output-file to write report to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RECORDS = [
    {
        "name": "electron_temperature",
        "domain": "kinetics",
        "reviewer_model": "claude-haiku-4",
        "score": 0.72,
        "comments_per_dim": {
            "grammar": "Name too long, redundant qualifiers present",
            "semantic": "Missing sign convention for COCOS-dependent field",
        },
        "comments": "Name too long. Sign convention missing.",
        "review_axis": "names",
        "generated_at": "2025-01-01T00:00:00Z",
        "reviewed_at": "2025-01-02T00:00:00Z",
    },
    {
        "name": "ion_density",
        "domain": "kinetics",
        "reviewer_model": "claude-haiku-4",
        "score": 0.55,
        "comments_per_dim": {
            "grammar": "Wrong base term, redundant qualifiers detected",
            "semantic": "Misleading coordinate convention",
        },
        "comments": "Wrong base term. Redundant qualifiers.",
        "review_axis": "names",
        "generated_at": "2025-01-01T00:00:00Z",
        "reviewed_at": "2025-01-02T01:00:00Z",
    },
    {
        "name": "plasma_current",
        "domain": "magnetics",
        "reviewer_model": "claude-haiku-4",
        "score": 0.88,
        "comments_per_dim": {
            "grammar": "Good",
            "semantic": "Good",
        },
        "comments": "Looks great.",
        "review_axis": "names",
        "generated_at": "2025-01-01T00:00:00Z",
        "reviewed_at": "2025-01-02T02:00:00Z",
    },
    # Second review of electron_temperature (repeat-reviewed name)
    {
        "name": "electron_temperature",
        "domain": "kinetics",
        "reviewer_model": "claude-opus-3",
        "score": 0.81,
        "comments_per_dim": {
            "grammar": "Better but still slightly long",
            "semantic": "Sign convention now mentioned, OK",
        },
        "comments": "Better now but still slightly long.",
        "review_axis": "names",
        "generated_at": "2025-01-03T00:00:00Z",
        "reviewed_at": "2025-01-04T00:00:00Z",
    },
]


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def jsonl_dir(tmp_path: Path) -> Path:
    """Write _RECORDS to two JSONL files and return the dir."""
    # Split records across two files to test multi-file glob
    f1 = tmp_path / "comments-20250101T000000Z.jsonl"
    f2 = tmp_path / "comments-20250103T000000Z.jsonl"

    with f1.open("w") as fh:
        for rec in _RECORDS[:2]:
            fh.write(json.dumps(rec) + "\n")
    with f2.open("w") as fh:
        for rec in _RECORDS[2:]:
            fh.write(json.dumps(rec) + "\n")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnalyseCommentsOutput:
    """Validate markdown report content."""

    def test_emits_top_criticisms_header(
        self, runner: CliRunner, jsonl_dir: Path
    ) -> None:
        """Report contains the 'Top Criticisms' section."""
        glob = str(jsonl_dir / "comments-*.jsonl")
        result = runner.invoke(sn, ["analyse-comments", "--input-glob", glob])
        assert result.exit_code == 0, result.output
        assert "Top Criticisms" in result.output

    def test_emits_per_dimension_section(
        self, runner: CliRunner, jsonl_dir: Path
    ) -> None:
        """Report contains per-dimension breakdown."""
        glob = str(jsonl_dir / "comments-*.jsonl")
        result = runner.invoke(sn, ["analyse-comments", "--input-glob", glob])
        assert result.exit_code == 0, result.output
        # Both 'grammar' and 'semantic' dimensions should appear
        assert "grammar" in result.output
        assert "semantic" in result.output

    def test_emits_repeat_reviewed_names_section(
        self, runner: CliRunner, jsonl_dir: Path
    ) -> None:
        """Repeat-reviewed names section shows electron_temperature."""
        glob = str(jsonl_dir / "comments-*.jsonl")
        result = runner.invoke(sn, ["analyse-comments", "--input-glob", glob])
        assert result.exit_code == 0, result.output
        assert "Repeat-Reviewed Names" in result.output
        assert "electron_temperature" in result.output

    def test_emits_score_distribution_section(
        self, runner: CliRunner, jsonl_dir: Path
    ) -> None:
        """Per-dimension score distribution section is present."""
        glob = str(jsonl_dir / "comments-*.jsonl")
        result = runner.invoke(sn, ["analyse-comments", "--input-glob", glob])
        assert result.exit_code == 0, result.output
        assert "Score Distribution" in result.output

    def test_common_phrase_appears_in_output(
        self, runner: CliRunner, jsonl_dir: Path
    ) -> None:
        """'redundant' appears in top criticisms (appears in 2 records)."""
        glob = str(jsonl_dir / "comments-*.jsonl")
        result = runner.invoke(sn, ["analyse-comments", "--input-glob", glob])
        assert result.exit_code == 0, result.output
        # 'redundant' or 'redundant qualifiers' should appear
        assert "redundant" in result.output.lower()


class TestAnalyseCommentsDomainFilter:
    """Validate --domain filtering."""

    def test_domain_filter_excludes_other_domains(
        self, runner: CliRunner, jsonl_dir: Path
    ) -> None:
        """With --domain kinetics, plasma_current (magnetics) should not appear."""
        glob = str(jsonl_dir / "comments-*.jsonl")
        result = runner.invoke(
            sn, ["analyse-comments", "--input-glob", glob, "--domain", "kinetics"]
        )
        assert result.exit_code == 0, result.output
        assert "plasma_current" not in result.output


class TestAnalyseCommentsEmptyInput:
    """Graceful handling of missing or empty input."""

    def test_no_matching_files_exits_gracefully(self, runner: CliRunner) -> None:
        """No files found → informative message, exit 0."""
        result = runner.invoke(
            sn,
            ["analyse-comments", "--input-glob", "/nonexistent/path/comments-*.jsonl"],
        )
        assert result.exit_code == 0
        assert "No files matched" in result.output

    def test_empty_jsonl_files_exits_gracefully(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Files exist but contain no records → graceful message."""
        empty = tmp_path / "comments-empty.jsonl"
        empty.write_text("", encoding="utf-8")
        glob = str(tmp_path / "comments-*.jsonl")
        result = runner.invoke(sn, ["analyse-comments", "--input-glob", glob])
        assert result.exit_code == 0
        assert "No review records found" in result.output

    def test_domain_filter_with_no_matches(
        self, runner: CliRunner, jsonl_dir: Path
    ) -> None:
        """--domain with no matching records → graceful message."""
        glob = str(jsonl_dir / "comments-*.jsonl")
        result = runner.invoke(
            sn,
            [
                "analyse-comments",
                "--input-glob",
                glob,
                "--domain",
                "nonexistent_domain",
            ],
        )
        assert result.exit_code == 0
        assert "No review records found" in result.output


class TestAnalyseCommentsOutputFile:
    """--output-file writes report to disk."""

    def test_output_file_written(
        self, runner: CliRunner, jsonl_dir: Path, tmp_path: Path
    ) -> None:
        """With --output-file the markdown is written to disk."""
        glob = str(jsonl_dir / "comments-*.jsonl")
        out = tmp_path / "report.md"
        result = runner.invoke(
            sn,
            ["analyse-comments", "--input-glob", glob, "--output-file", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Top Criticisms" in content
        assert "Report written to" in result.output

    def test_output_file_creates_parent_dirs(
        self, runner: CliRunner, jsonl_dir: Path, tmp_path: Path
    ) -> None:
        """Parent dirs are created for --output-file."""
        glob = str(jsonl_dir / "comments-*.jsonl")
        out = tmp_path / "deep" / "nested" / "report.md"
        result = runner.invoke(
            sn,
            ["analyse-comments", "--input-glob", glob, "--output-file", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()


class TestAnalyseCommentsHelpText:
    """Verify --help shows expected flags."""

    def test_help_shows_all_options(self, runner: CliRunner) -> None:
        result = runner.invoke(sn, ["analyse-comments", "--help"])
        assert result.exit_code == 0
        for flag in ["--input-glob", "--domain", "--top", "--output-file"]:
            assert flag in result.output, f"Missing {flag} in help"
