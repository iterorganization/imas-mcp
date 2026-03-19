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


class TestServeCLI:
    """Tests for unified serve command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_serve_help(self, runner):
        """Test serve --help shows options."""
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--transport" in result.output
        assert "--read-only" in result.output

    @patch("imas_codex.llm.server.AgentsServer")
    def test_serve_default_options(self, mock_server_cls, runner):
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server
        runner.invoke(main, ["serve"], catch_exceptions=False)
        mock_server_cls.assert_called_once_with(read_only=False)
        mock_server.run.assert_called_once()

    @patch("imas_codex.llm.server.AgentsServer")
    def test_serve_read_only(self, mock_server_cls, runner):
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server
        runner.invoke(main, ["serve", "--read-only"], catch_exceptions=False)
        mock_server_cls.assert_called_once_with(read_only=True)

    @patch("imas_codex.llm.server.AgentsServer")
    def test_serve_transport_options(self, mock_server_cls, runner):
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server
        for transport in ["stdio", "sse", "streamable-http"]:
            result = runner.invoke(main, ["serve", "--transport", transport])
            assert result.exit_code == 0
        result = runner.invoke(main, ["serve", "--transport", "invalid"])
        assert result.exit_code != 0

    @patch("imas_codex.llm.server.AgentsServer")
    def test_serve_host_port(self, mock_server_cls, runner):
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server
        runner.invoke(
            main,
            ["serve", "--host", "0.0.0.0", "--port", "9000"],
            catch_exceptions=False,
        )
        mock_server.run.assert_called_once()
        call_kwargs = mock_server.run.call_args[1]
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 9000


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
        assert "tcv" in result.output.lower() or "Available" in result.output

    def test_facilities_show(self, runner):
        """Test facilities show command."""
        result = runner.invoke(main, ["facilities", "show", "tcv"])
        assert result.exit_code == 0
        assert "tcv" in result.output
        assert "ssh_host" in result.output

    def test_facilities_show_unknown(self, runner):
        """Test facilities show with unknown facility."""
        result = runner.invoke(main, ["facilities", "show", "unknown_facility"])
        assert result.exit_code != 0
