"""Tests for `imas-codex graph sync-isn-grammar` CLI (plan 29 E.6 + E.8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from imas_codex.cli.graph.sync import sync_isn_grammar


@dataclass
class FakeReport:
    """Fake SyncReport matching ISN's dataclass shape."""

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
    planned_statements: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def _fake_report():
    """Build a fake SyncReport-like object."""
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


# ---------------------------------------------------------------------------
# Sync idempotency tests (plan 29 E.8)
# ---------------------------------------------------------------------------


def _make_mock_spec():
    """Return a fixed grammar graph spec for idempotency testing."""
    return {
        "version": "0.7.0rc10",
        "segment_order": ["subject", "physical_base", "object"],
        "segments": [
            {
                "name": "subject",
                "position": 0,
                "required": False,
                "tokens": [
                    {"value": "electron", "aliases": []},
                    {"value": "ion", "aliases": []},
                ],
            },
            {
                "name": "physical_base",
                "position": 1,
                "required": True,
                "tokens": [],  # open vocab
            },
            {
                "name": "object",
                "position": 2,
                "required": False,
                "tokens": [
                    {"value": "plasma", "aliases": []},
                ],
            },
        ],
        "templates": [
            {"name": "of_object", "segment": "object", "pattern": "of_{token}"},
        ],
    }


def _run_sync_with_mock(mock_gc: MagicMock, spec: dict) -> FakeReport:
    """Invoke sync_grammar with mocked spec and GraphClient, return report."""
    from imas_standard_names.graph.sync import sync_grammar

    with patch(
        "imas_standard_names.graph.sync.get_grammar_graph_spec", return_value=spec
    ):
        return sync_grammar(mock_gc, active_version="0.7.0rc10")


class TestSyncIdempotency:
    """Verify that running sync twice produces no duplicates (E.8)."""

    def test_second_sync_no_new_statements(self) -> None:
        """Running sync twice should produce the same Cypher MERGE statements.

        Since sync uses MERGE throughout, the second run should be a no-op
        at the database level — same statements, same parameters.
        """
        spec = _make_mock_spec()
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        _run_sync_with_mock(mock_gc, spec)
        calls_after_first = len(mock_gc.query.call_args_list)

        _run_sync_with_mock(mock_gc, spec)
        calls_after_second = len(mock_gc.query.call_args_list)

        # Both runs issue the same statements
        first_count = calls_after_first
        second_count = calls_after_second - calls_after_first
        assert first_count == second_count, (
            f"First run issued {first_count} statements, "
            f"second run issued {second_count}. "
            f"Expected identical MERGE-based statements."
        )

    def test_second_sync_same_version(self) -> None:
        """Both sync runs should report the same version."""
        spec = _make_mock_spec()
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        report1 = _run_sync_with_mock(mock_gc, spec)
        report2 = _run_sync_with_mock(mock_gc, spec)

        assert report1.version == report2.version == "0.7.0rc10"

    def test_merge_cypher_used_not_create(self) -> None:
        """All node/edge writes must use MERGE, never CREATE."""
        spec = _make_mock_spec()
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        _run_sync_with_mock(mock_gc, spec)

        for call_obj in mock_gc.query.call_args_list:
            cypher = call_obj[0][0] if call_obj[0] else ""
            # Skip constraint DDL statements
            if "CONSTRAINT" in cypher or "INDEX" in cypher:
                continue
            # All data writes should use MERGE
            if "GrammarSegment" in cypher or "GrammarToken" in cypher:
                assert "MERGE" in cypher, (
                    f"Expected MERGE in Cypher, got CREATE-style: {cypher[:120]}"
                )

    def test_version_node_count_stable(self) -> None:
        """After two syncs the ISNGrammarVersion MERGE should target the same id."""
        spec = _make_mock_spec()
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        _run_sync_with_mock(mock_gc, spec)
        _run_sync_with_mock(mock_gc, spec)

        # Find calls that MERGE the ISNGrammarVersion node itself
        # (not just MATCH it as a parent for segment writes)
        version_merges = [
            c
            for c in mock_gc.query.call_args_list
            if "MERGE" in str(c[0][0])
            and "ISNGrammarVersion" in str(c[0][0])
            and "MERGE (v:ISNGrammarVersion" in str(c[0][0])
        ]
        # Should have exactly 2 (one per sync) with identical version
        assert len(version_merges) == 2, (
            f"Expected 2 ISNGrammarVersion MERGEs (one per sync), "
            f"got {len(version_merges)}"
        )
        # Both should use the same version parameter
        versions = {c[1].get("version", "") for c in version_merges}
        assert len(versions) == 1, (
            f"Expected same version for both syncs, got {versions}"
        )
