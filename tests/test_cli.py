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

    def test_import_data(self):
        """Data module imports."""
        from imas_codex.cli.data import data

        assert data is not None

    def test_import_ingest(self):
        """Ingest module imports."""
        from imas_codex.cli.ingest import ingest

        assert ingest is not None

    def test_import_release(self):
        """Release module imports."""
        from imas_codex.cli.release import release

        assert release is not None

    def test_import_imas_dd(self):
        """IMAS DD module imports."""
        from imas_codex.cli.imas_dd import imas

        assert imas is not None

    def test_import_clusters(self):
        """Clusters module imports."""
        from imas_codex.cli.clusters import clusters

        assert clusters is not None

    def test_import_enrich(self):
        """Enrich module imports."""
        from imas_codex.cli.enrich import enrich

        assert enrich is not None

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
            setup_age,
        )

        assert console is not None
        assert callable(format_size)
        assert callable(format_duration)
        assert callable(run_async)
        assert setup_age is not None
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
        assert "Start MCP servers" in result.output

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
        assert "IMAS Data Dictionary" in result.output

    def test_clusters_help(self, runner):
        """clusters group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["clusters", "--help"])
        assert result.exit_code == 0
        assert "semantic clusters" in result.output.lower()

    def test_enrich_help(self, runner):
        """enrich group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["enrich", "--help"])
        assert result.exit_code == 0
        assert "Enrich graph" in result.output

    def test_ingest_help(self, runner):
        """ingest group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "Ingest" in result.output

    def test_data_help(self, runner):
        """data group has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["data", "--help"])
        assert result.exit_code == 0
        assert "database" in result.output.lower() or "data" in result.output.lower()

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

    def test_setup_age_help(self, runner):
        """setup-age command has help."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["setup-age", "--help"])
        assert result.exit_code == 0
        assert (
            "age encryption" in result.output.lower()
            or "age key" in result.output.lower()
        )

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

    def test_imas_build_help(self, runner):
        """imas build subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["imas", "build", "--help"])
        assert result.exit_code == 0

    def test_clusters_build_help(self, runner):
        """clusters build subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["clusters", "build", "--help"])
        assert result.exit_code == 0

    def test_facilities_list_help(self, runner):
        """facilities list subcommand exists."""
        from imas_codex.cli import main

        result = runner.invoke(main, ["facilities", "list", "--help"])
        assert result.exit_code == 0
