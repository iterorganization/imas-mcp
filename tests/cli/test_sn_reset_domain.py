"""Tests for ``imas-codex sn reset-domain`` CLI subcommand (Wave 8A).

Validates:
- --help renders correctly and shows required --domain flag
- --dry-run reports count, makes no graph writes
- soft reset calls correct Cypher (claimed_at/claim_token cleared)
- --hard issues DETACH DELETE query
- missing --domain raises UsageError
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn


@pytest.fixture
def runner():
    return CliRunner()


class TestResetDomainHelp:
    """Verify help text is registered."""

    def test_help_renders(self, runner: CliRunner) -> None:
        result = runner.invoke(sn, ["reset-domain", "--help"])
        assert result.exit_code == 0
        assert "--domain" in result.output
        assert "--hard" in result.output
        assert "--dry-run" in result.output

    def test_missing_domain_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(sn, ["reset-domain"])
        assert result.exit_code != 0
        assert "domain" in result.output.lower() or "Missing" in result.output


class TestResetDomainDryRun:
    """--dry-run reports count but does not write."""

    def _make_gc(self, count: int):
        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        gc.query = MagicMock(return_value=[{"n": count}])
        return gc

    def test_dry_run_prints_count(self, runner: CliRunner) -> None:
        gc = self._make_gc(42)
        with patch("imas_codex.graph.client.GraphClient", return_value=gc):
            result = runner.invoke(
                sn, ["reset-domain", "--domain", "equilibrium", "--dry-run"]
            )
        assert result.exit_code == 0
        assert "42" in result.output

    def test_dry_run_does_not_write(self, runner: CliRunner) -> None:
        gc = self._make_gc(5)
        with patch("imas_codex.graph.client.GraphClient", return_value=gc):
            runner.invoke(sn, ["reset-domain", "--domain", "equilibrium", "--dry-run"])
        # Only the count query should have been called (once)
        assert gc.query.call_count == 1

    def test_zero_nodes_exits_cleanly(self, runner: CliRunner) -> None:
        gc = self._make_gc(0)
        with patch("imas_codex.graph.client.GraphClient", return_value=gc):
            result = runner.invoke(
                sn, ["reset-domain", "--domain", "equilibrium", "--dry-run"]
            )
        assert result.exit_code == 0
        assert "No" in result.output or "0" in result.output


class TestResetDomainSoftReset:
    """Default (soft) reset clears claims and resets status."""

    def _make_gc(self, count: int):
        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        # First call = count, second call = update
        gc.query = MagicMock(side_effect=[[{"n": count}], []])
        return gc

    def test_soft_reset_calls_set_query(self, runner: CliRunner) -> None:
        gc = self._make_gc(7)
        with patch("imas_codex.graph.client.GraphClient", return_value=gc):
            result = runner.invoke(sn, ["reset-domain", "--domain", "equilibrium"])
        assert result.exit_code == 0
        assert gc.query.call_count == 2
        # Second call should contain SET (soft reset)
        second_cypher = gc.query.call_args_list[1][0][0]
        assert "SET" in second_cypher
        assert "claimed_at" in second_cypher
        assert "extracted" in second_cypher
        assert "DETACH DELETE" not in second_cypher


class TestResetDomainHard:
    """--hard issues DETACH DELETE."""

    def _make_gc(self, count: int):
        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        gc.query = MagicMock(side_effect=[[{"n": count}], []])
        return gc

    def test_hard_uses_detach_delete(self, runner: CliRunner) -> None:
        gc = self._make_gc(3)
        with patch("imas_codex.graph.client.GraphClient", return_value=gc):
            result = runner.invoke(
                sn, ["reset-domain", "--domain", "equilibrium", "--hard"]
            )
        assert result.exit_code == 0
        assert gc.query.call_count == 2
        second_cypher = gc.query.call_args_list[1][0][0]
        assert "DETACH DELETE" in second_cypher
