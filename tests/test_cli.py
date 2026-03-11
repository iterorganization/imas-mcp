"""Smoke tests for the modular CLI.

These tests verify that all CLI modules import correctly and that
the main command groups are properly registered.
"""

import pytest
from click.testing import CliRunner


class TestCLIImports:
    """Test that all CLI modules can be imported."""

    def test_import_main(self):
        """Main CLI entry point can be imported."""
        from imas_codex.cli import main

        assert main is not None
        assert hasattr(main, "commands")

    def test_import_serve(self):
        """Serve module imports."""
        from imas_codex.cli.serve import serve

        assert serve is not None

    def test_import_facilities(self):
        """Facilities module imports."""
        from imas_codex.cli.facilities import facilities

        assert facilities is not None

    def test_import_tools(self):
        """Tools module imports."""
        from imas_codex.cli.tools import tools

        assert tools is not None

    def test_import_hosts(self):
        """Hosts module imports."""
        from imas_codex.cli.hosts import hosts

        assert hosts is not None

    def test_import_graph_cli(self):
        """Graph CLI module imports."""
        from imas_codex.cli.graph import graph

        assert graph is not None

    def test_import_config_cli(self):
        """Config CLI module imports."""
        from imas_codex.cli.config_cli import config

        assert config is not None

    def test_import_release(self):
        """Release module imports."""
        from imas_codex.cli.release import release

        assert release is not None

    def test_import_imas_dd(self):
        """IMAS DD module imports."""
        from imas_codex.cli.imas_dd import imas

        assert imas is not None

    def test_import_discover(self):
        """Discover module imports."""
        from imas_codex.cli.discover import discover

        assert discover is not None

    def test_import_utils(self):
        """Utils module imports with expected symbols."""
        from imas_codex.cli.utils import (
            ProgressReporter,
            console,
            format_duration,
            format_size,
            run_async,
        )

        assert console is not None
        assert callable(format_size)
        assert callable(format_duration)
        assert callable(run_async)
        assert ProgressReporter is not None


class TestCLICommands:
    """Test that CLI commands are properly registered and have help text."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_main_help(self, runner):
        """Main CLI shows help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "IMAS Codex" in result.output
        assert "Commands:" in result.output

    def test_main_version(self, runner):
        """--version flag works."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        # Version string is output
        assert result.output.strip()  # Some version text present

    def test_serve_help(self, runner):
        """serve group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "MCP servers" in result.output

    def test_facilities_help(self, runner):
        """facilities group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["facilities", "--help"])
        assert result.exit_code == 0
        assert "Manage facility" in result.output

    def test_discover_help(self, runner):
        """discover group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["discover", "--help"])
        assert result.exit_code == 0
        assert "Discover facility" in result.output

    def test_imas_help(self, runner):
        """imas group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "--help"])
        assert result.exit_code == 0
        assert "IMAS" in result.output

    def test_tools_help(self, runner):
        """tools group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["tools", "--help"])
        assert result.exit_code == 0
        assert "tools" in result.output.lower()

    def test_hosts_help(self, runner):
        """hosts group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["hosts", "--help"])
        assert result.exit_code == 0
        assert "SSH" in result.output or "host" in result.output.lower()

    def test_release_help(self, runner):
        """release command has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["release", "--help"])
        assert result.exit_code == 0
        assert "release" in result.output.lower()


class TestCLISubcommands:
    """Test specific subcommands are registered."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_discover_paths_help(self, runner):
        """discover paths subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["discover", "paths", "--help"])
        assert result.exit_code == 0
        assert "directory structure" in result.output.lower()

    def test_imas_dd_build_help(self, runner):
        """imas dd build subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "dd", "build", "--help"])
        assert result.exit_code == 0

    def test_imas_map_help(self, runner):
        """imas map subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "map", "--help"])
        assert result.exit_code == 0
        assert "IMAS mapping pipeline" in result.output

    def test_imas_list_help(self, runner):
        """imas list subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "list", "--help"])
        assert result.exit_code == 0
        assert "List available IDS recipes" in result.output

    def test_imas_export_help(self, runner):
        """imas export subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "export", "--help"])
        assert result.exit_code == 0

    def test_imas_epochs_help(self, runner):
        """imas epochs subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "epochs", "--help"])
        assert result.exit_code == 0

    def test_facilities_list_help(self, runner):
        """facilities list subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["facilities", "list", "--help"])
        assert result.exit_code == 0

    def test_no_top_level_ids(self, runner):
        """ids is no longer a top-level command."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["ids", "--help"])
        assert result.exit_code != 0

    def test_no_top_level_map(self, runner):
        """map is no longer a top-level command."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["map", "--help"])
        assert result.exit_code != 0
