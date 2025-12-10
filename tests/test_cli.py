"""
Tests for CLI interface.

Uses Click's CliRunner for testing command-line parsing without starting the server.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_mcp.cli import main


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
        assert "--transport" in result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--log-level" in result.output

    @patch("imas_mcp.cli.Server")
    def test_default_options(self, mock_server_cls, runner):
        """Test default options are applied correctly."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        runner.invoke(main, [], catch_exceptions=False)

        # Server should be created with defaults
        mock_server_cls.assert_called_once()
        call_kwargs = mock_server_cls.call_args[1]
        assert call_kwargs["use_rich"] is True
        assert call_kwargs["ids_set"] is None

    @patch("imas_mcp.cli.Server")
    def test_transport_options(self, mock_server_cls, runner):
        """Test transport option validation."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        # Test valid transports
        for transport in ["stdio", "sse", "streamable-http"]:
            result = runner.invoke(main, ["--transport", transport])
            # Just check it doesn't error on invalid choice
            assert "Invalid value for '--transport'" not in result.output

    def test_invalid_transport(self, runner):
        """Test invalid transport option is rejected."""
        result = runner.invoke(main, ["--transport", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value for '--transport'" in result.output

    @patch("imas_mcp.cli.Server")
    def test_host_port_options(self, mock_server_cls, runner):
        """Test host and port options."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        runner.invoke(
            main, ["--host", "0.0.0.0", "--port", "9000"], catch_exceptions=False
        )

        # Server.run should be called with custom host/port
        mock_server.run.assert_called_once()
        call_kwargs = mock_server.run.call_args[1]
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 9000

    @patch("imas_mcp.cli.Server")
    def test_log_level_options(self, mock_server_cls, runner):
        """Test log level options."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            result = runner.invoke(main, ["--log-level", level])
            assert "Invalid value for '--log-level'" not in result.output

    def test_invalid_log_level(self, runner):
        """Test invalid log level is rejected."""
        result = runner.invoke(main, ["--log-level", "INVALID"])
        assert result.exit_code != 0
        assert "Invalid value for '--log-level'" in result.output

    @patch("imas_mcp.cli.Server")
    def test_no_rich_flag(self, mock_server_cls, runner):
        """Test --no-rich flag disables rich output."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        runner.invoke(main, ["--no-rich"], catch_exceptions=False)

        mock_server_cls.assert_called_once()
        call_kwargs = mock_server_cls.call_args[1]
        assert call_kwargs["use_rich"] is False

    @patch("imas_mcp.cli.Server")
    def test_stdio_disables_rich(self, mock_server_cls, runner):
        """Test stdio transport automatically disables rich output."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        runner.invoke(main, ["--transport", "stdio"], catch_exceptions=False)

        mock_server_cls.assert_called_once()
        call_kwargs = mock_server_cls.call_args[1]
        # Rich should be disabled for stdio
        assert call_kwargs["use_rich"] is False

    @patch("imas_mcp.cli.Server")
    def test_ids_filter_option(self, mock_server_cls, runner):
        """Test --ids-filter option parses IDS names."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        runner.invoke(
            main,
            ["--ids-filter", "core_profiles equilibrium"],
            catch_exceptions=False,
        )

        mock_server_cls.assert_called_once()
        call_kwargs = mock_server_cls.call_args[1]
        assert call_kwargs["ids_set"] == {"core_profiles", "equilibrium"}

    @patch("imas_mcp.cli.Server")
    def test_docs_server_port_option(self, mock_server_cls, runner):
        """Test --docs-server-port option."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        runner.invoke(main, ["--docs-server-port", "7000"], catch_exceptions=False)

        mock_server_cls.assert_called_once()
        # Docs manager port should be set
        assert mock_server.docs_manager.default_port == 7000

    @patch("imas_mcp.cli.Server")
    def test_disable_docs_server_flag(self, mock_server_cls, runner):
        """Test --disable-docs-server flag."""
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        runner.invoke(main, ["--disable-docs-server"], catch_exceptions=False)

        # Docs manager default_port should not be set when disabled
        # The flag just prevents auto-start
        mock_server_cls.assert_called_once()
