"""Tests for ``imas-codex sn status`` CLI command.

Validates:
- Name Stage table is rendered with all stages
- Acceptance rate lines (excl. / incl. superseded) are present and correct
- Docs Stage table is rendered
- Zero-division is handled gracefully when no names exist
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn


@pytest.fixture
def runner():
    return CliRunner()


def _make_gc_instance(
    total_row=None,
    vstatus_rows=None,
    name_stage_rows=None,
    docs_stage_rows=None,
):
    """Return a context-manager mock instance for GraphClient.query().

    ``sn_status`` uses two separate ``with GraphClient() as gc:`` blocks.
    The first block makes 4 queries (total, vstatus, name_stage, docs_stage).
    The second block queries for SNRun rows.  We return [] for all calls
    beyond the first four so the SNRun section silently renders nothing.
    """
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    default_total = [
        {"total": 497, "from_dd": 300, "from_signals": 100, "from_manual": 97}
    ]
    default_vstatus = [{"status": "valid", "cnt": 497}]
    default_ns = [
        {"stage": "superseded", "n": 202},
        {"stage": "accepted", "n": 179},
        {"stage": "drafted", "n": 88},
        {"stage": "exhausted", "n": 27},
        {"stage": "reviewed", "n": 1},
    ]
    default_ds = [
        {"stage": "pending", "n": 320},
        {"stage": "accepted", "n": 161},
        {"stage": "drafted", "n": 16},
    ]

    fixed = [
        total_row if total_row is not None else default_total,
        vstatus_rows if vstatus_rows is not None else default_vstatus,
        name_stage_rows if name_stage_rows is not None else default_ns,
        docs_stage_rows if docs_stage_rows is not None else default_ds,
    ]

    def _side_effect(*_args, **_kwargs):
        return fixed.pop(0) if fixed else []

    gc.query.side_effect = _side_effect
    return gc


def _patch_sn_status(name_stage_rows=None, docs_stage_rows=None, **kw):
    """Context manager that patches everything sn_status touches."""
    gc_instance = _make_gc_instance(
        name_stage_rows=name_stage_rows,
        docs_stage_rows=docs_stage_rows,
        **kw,
    )
    gc_cls = MagicMock(return_value=gc_instance)
    return (
        patch("imas_codex.graph.client.GraphClient", gc_cls),
        patch(
            "imas_codex.standard_names.graph_ops.get_standard_name_source_stats",
            return_value={},
        ),
        patch(
            "imas_codex.standard_names.graph_ops.get_skipped_source_counts",
            return_value={},
        ),
    )


class TestSnStatusNameStage:
    """Confirm Name Stage table and acceptance rates are displayed."""

    def test_name_stage_table_rendered(self, runner: CliRunner) -> None:
        patches = _patch_sn_status()
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(sn, ["status"])
        assert result.exit_code == 0, result.output
        assert "Name Stage" in result.output
        assert "accepted" in result.output
        assert "superseded" in result.output
        assert "drafted" in result.output

    def test_acceptance_rate_excl_superseded(self, runner: CliRunner) -> None:
        patches = _patch_sn_status()
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(sn, ["status"])
        assert result.exit_code == 0, result.output
        # excl. superseded: 179 / (497 - 202) = 179/295 ≈ 60.7%
        assert "excl. superseded" in result.output
        assert "179" in result.output
        assert "295" in result.output

    def test_acceptance_rate_incl_superseded(self, runner: CliRunner) -> None:
        patches = _patch_sn_status()
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(sn, ["status"])
        assert result.exit_code == 0, result.output
        # incl. superseded: 179/497 ≈ 36.0%
        assert "incl. superseded" in result.output
        assert "497" in result.output

    def test_docs_stage_table_rendered(self, runner: CliRunner) -> None:
        patches = _patch_sn_status()
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(sn, ["status"])
        assert result.exit_code == 0, result.output
        assert "Docs Stage" in result.output
        assert "pending" in result.output

    def test_empty_name_stage_no_crash(self, runner: CliRunner) -> None:
        """If no names exist, acceptance rate section is simply omitted."""
        patches = _patch_sn_status(
            total_row=[{"total": 0, "from_dd": 0, "from_signals": 0, "from_manual": 0}],
            vstatus_rows=[],
            name_stage_rows=[],
            docs_stage_rows=[],
        )
        with patches[0], patches[1], patches[2]:
            result = runner.invoke(sn, ["status"])
        assert result.exit_code == 0, result.output
        # Sections should be absent when there's no data
        assert "Name Stage" not in result.output
        assert "Docs Stage" not in result.output
