"""Tests for `imas-codex graph sync-isn-grammar` CLI (plan 29 E.6)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from imas_codex.cli.graph.sync import sync_isn_grammar


def _fake_report():
    """Build a fake SyncReport-like object matching ISN's dataclass shape."""
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class FakeReport:
        version: str = "0.7.0rc10"
        applied: bool = False
        created_version: bool = True
        segments_written: int = 11
        tokens_written: int = 7
        templates_written: int = 6
        next_edges_written: int = 10
        defines_edges_written: int = 4
        has_token_edges_written: int = 3
        elapsed_seconds: float = 0.01
        planned_statements: list[tuple[str, dict[str, Any]]] = field(
            default_factory=list
        )

    return FakeReport()


def test_sync_isn_grammar_skip_flag():
    """--skip-grammar-sync exits cleanly without touching Neo4j."""
    runner = CliRunner()
    result = runner.invoke(sync_isn_grammar, ["--skip-grammar-sync"])
    assert result.exit_code == 0
    assert "skipped" in result.output.lower()


def test_sync_isn_grammar_dry_run():
    """--dry-run calls sync_grammar with dry_run=True and prints report."""
    runner = CliRunner()
    fake_gc = MagicMock()
    fake_gc.__enter__.return_value = fake_gc
    fake_gc.__exit__.return_value = None

    fake_report = _fake_report()

    with (
        patch("imas_codex.cli.graph.sync.GraphClient", return_value=fake_gc),
        patch(
            "imas_standard_names.graph.sync.sync_grammar", return_value=fake_report
        ) as mock_sync,
    ):
        result = runner.invoke(sync_isn_grammar, ["--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Grammar sync complete" in result.output
    assert "dry_run=True" in result.output
    assert "segments_written: 11" in result.output
    mock_sync.assert_called_once()
    _, kwargs = mock_sync.call_args
    assert kwargs["dry_run"] is True
    assert kwargs["active_version"]  # ISN version propagated


def test_sync_isn_grammar_unreachable_raises():
    """Neo4j connection failure raises ClickException (hard-fail per M3)."""
    runner = CliRunner()

    def _boom(*_a, **_kw):
        raise ConnectionError("Neo4j unreachable")

    with patch("imas_codex.cli.graph.sync.GraphClient", side_effect=_boom):
        result = runner.invoke(sync_isn_grammar, [])

    assert result.exit_code != 0
    assert "Failed to sync grammar" in result.output
    assert "Neo4j unreachable" in result.output


def test_sync_isn_grammar_skip_bypasses_unreachable():
    """--skip-grammar-sync does not attempt connection even when Neo4j is down."""
    runner = CliRunner()

    with patch(
        "imas_codex.cli.graph.sync.GraphClient",
        side_effect=ConnectionError("Neo4j unreachable"),
    ) as mock_gc:
        result = runner.invoke(sync_isn_grammar, ["--skip-grammar-sync"])

    assert result.exit_code == 0
    mock_gc.assert_not_called()
