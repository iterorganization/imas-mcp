"""Test ``sn run --help`` output includes the Phase 5 flags.

Verifies that the ``sn run`` subcommand advertises:
  - ``--only``
  - ``--override-edits``
"""

from __future__ import annotations

from click.testing import CliRunner

from imas_codex.cli.sn import sn


class TestSnRunHelpShowsNewFlags:
    """sn run --help must list the Phase 5 flags."""

    def _run_help(self) -> str:
        runner = CliRunner()
        result = runner.invoke(sn, ["run", "--help"])
        assert result.exit_code == 0
        return result.output

    def test_only_flag(self):
        output = self._run_help()
        assert "--only" in output
        # Check at least a couple of valid choices are shown
        assert "link" in output
        assert "reconcile" in output

    def test_override_edits_flag(self):
        output = self._run_help()
        assert "--override-edits" in output

    def test_no_skip_reconcile_flag(self):
        """--skip-reconcile should no longer appear in help."""
        output = self._run_help()
        assert "--skip-reconcile" not in output

    def test_no_skip_resolve_links_flag(self):
        """--skip-link should no longer appear in help."""
        output = self._run_help()
        assert "--skip-link" not in output
