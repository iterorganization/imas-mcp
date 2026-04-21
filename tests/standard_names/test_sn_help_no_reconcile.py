"""Test ``sn --help`` output contains no reconcile, link, seed tokens.

Verifies Phase 5 acceptance criterion: the standalone CLI verbs are removed
and their functionality is reachable only via ``sn run``.
"""

from __future__ import annotations

from click.testing import CliRunner

from imas_codex.cli.sn import sn


class TestSnHelpNoLegacyVerbs:
    """sn --help must not show removed verbs."""

    def test_no_reconcile_in_help(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["--help"])
        assert result.exit_code == 0
        # The word "reconcile" should not appear as a listed command
        for line in result.output.splitlines():
            # Skip docstring lines that mention the concept
            stripped = line.strip()
            if stripped.startswith("sn "):
                # Example usage lines in the group docstring
                continue
            if stripped.startswith("reconcile") and not stripped.startswith(
                "reconcile →"
            ):
                # This would be a subcommand listing
                assert False, f"Found 'reconcile' as a subcommand: {line}"

    def test_no_resolve_links_as_command(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["--help"])
        assert result.exit_code == 0
        # Check there's no "resolve-links" or "link" as a top-level subcommand
        lines = result.output.splitlines()
        command_lines = [ln.strip() for ln in lines if ln.strip().startswith("resolve")]
        # Should be empty — resolve-links is no longer a verb
        assert not command_lines, f"Found resolve-links as a command: {command_lines}"

    def test_no_seed_command(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["--help"])
        assert result.exit_code == 0
        lines = result.output.splitlines()
        command_lines = [ln.strip() for ln in lines if ln.strip().startswith("seed")]
        assert not command_lines, f"Found seed as a command: {command_lines}"

    def test_reconcile_verb_is_error(self):
        """Invoking ``sn reconcile`` should fail (unknown command)."""
        runner = CliRunner()
        result = runner.invoke(sn, ["reconcile"])
        assert result.exit_code != 0

    def test_resolve_links_verb_is_error(self):
        """Invoking ``sn resolve-links`` should fail (unknown command)."""
        runner = CliRunner()
        result = runner.invoke(sn, ["resolve-links"])
        assert result.exit_code != 0

    def test_seed_verb_is_error(self):
        """Invoking ``sn seed`` should fail (unknown command)."""
        runner = CliRunner()
        result = runner.invoke(sn, ["seed"])
        assert result.exit_code != 0
