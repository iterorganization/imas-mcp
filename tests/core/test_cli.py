"""
Tests for CLI interface.

Uses Click's CliRunner for testing command-line parsing without starting the server.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli import main


class TestCLI:
    """Tests for CLI command parsing and options."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_version_flag(self, runner):
        """Test --version flag prints version and exits."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        # Version should be a semver-like string
        assert result.output.strip()
        # Should not include "version:" prefix - just the raw version
        assert "." in result.output

    def test_help_flag(self, runner):
        """Test --help flag shows usage information."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "serve" in result.output
        assert "facilities" in result.output

    def test_serve_help(self, runner):
        """Test serve --help shows subcommands."""
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "imas" in result.output
        assert "agents" in result.output

    def test_serve_imas_help(self, runner):
        """Test serve imas --help shows options."""
        result = runner.invoke(main, ["serve", "imas", "--help"])
        assert result.exit_code == 0
        assert "--transport" in result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--log-level" in result.output

    @patch("imas_codex.server.Server")
    def test_serve_imas_default_options(self, mock_server_cls, runner):
        """Test default options are applied correctly."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        runner.invoke(main, ["serve", "imas"], catch_exceptions=False)

        # Server should be created with defaults
        mock_server_cls.assert_called_once()
        call_kwargs = mock_server_cls.call_args[1]
        assert call_kwargs["use_rich"] is True
        assert call_kwargs["ids_set"] is None

    @patch("imas_codex.server.Server")
    def test_transport_options(self, mock_server_cls, runner):
        """Test transport option validation."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        # Test valid transports
        for transport in ["stdio", "sse", "streamable-http"]:
            result = runner.invoke(main, ["serve", "imas", "--transport", transport])
            # Just check it doesn't error on invalid choice
            assert "Invalid value for '--transport'" not in result.output

    def test_invalid_transport(self, runner):
        """Test invalid transport option is rejected."""
        result = runner.invoke(main, ["serve", "imas", "--transport", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value for '--transport'" in result.output

    @patch("imas_codex.server.Server")
    def test_host_port_options(self, mock_server_cls, runner):
        """Test host and port options."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        runner.invoke(
            main,
            ["serve", "imas", "--host", "0.0.0.0", "--port", "9000"],
            catch_exceptions=False,
        )

        # Server.run should be called with custom host/port
        mock_server.run.assert_called_once()
        call_kwargs = mock_server.run.call_args[1]
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 9000

    @patch("imas_codex.server.Server")
    def test_log_level_options(self, mock_server_cls, runner):
        """Test log level options."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            result = runner.invoke(main, ["serve", "imas", "--log-level", level])
            assert "Invalid value for '--log-level'" not in result.output

    def test_invalid_log_level(self, runner):
        """Test invalid log level is rejected."""
        result = runner.invoke(main, ["serve", "imas", "--log-level", "INVALID"])
        assert result.exit_code != 0
        assert "Invalid value for '--log-level'" in result.output

    @patch("imas_codex.server.Server")
    def test_no_rich_flag(self, mock_server_cls, runner):
        """Test --no-rich flag disables rich output."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        runner.invoke(main, ["serve", "imas", "--no-rich"], catch_exceptions=False)

        mock_server_cls.assert_called_once()
        call_kwargs = mock_server_cls.call_args[1]
        assert call_kwargs["use_rich"] is False

    @patch("imas_codex.server.Server")
    def test_stdio_disables_rich(self, mock_server_cls, runner):
        """Test stdio transport automatically disables rich output."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        runner.invoke(
            main, ["serve", "imas", "--transport", "stdio"], catch_exceptions=False
        )

        mock_server_cls.assert_called_once()
        call_kwargs = mock_server_cls.call_args[1]
        # Rich should be disabled for stdio
        assert call_kwargs["use_rich"] is False

    @patch("imas_codex.server.Server")
    def test_ids_filter_option(self, mock_server_cls, runner):
        """Test --ids-filter option parses IDS names."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        runner.invoke(
            main,
            ["serve", "imas", "--ids-filter", "core_profiles equilibrium"],
            catch_exceptions=False,
        )

        mock_server_cls.assert_called_once()
        call_kwargs = mock_server_cls.call_args[1]
        assert call_kwargs["ids_set"] == {"core_profiles", "equilibrium"}


class TestFacilitiesCLI:
    """Tests for facilities command group."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_facilities_list(self, runner):
        """Test facilities list command."""
        result = runner.invoke(main, ["facilities", "list"])
        assert result.exit_code == 0
        assert "epfl" in result.output.lower() or "Available" in result.output

    def test_facilities_show(self, runner):
        """Test facilities show command."""
        result = runner.invoke(main, ["facilities", "show", "epfl"])
        assert result.exit_code == 0
        assert "epfl" in result.output
        assert "ssh_host" in result.output

    def test_facilities_show_unknown(self, runner):
        """Test facilities show with unknown facility."""
        result = runner.invoke(main, ["facilities", "show", "unknown_facility"])
        assert result.exit_code != 0


class TestAgentsCLI:
    """Tests for agents serve command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_serve_agents_help(self, runner):
        """Test serve agents --help shows options."""
        result = runner.invoke(main, ["serve", "agents", "--help"])
        assert result.exit_code == 0
        assert "--transport" in result.output
        assert "facility exploration" in result.output.lower()
