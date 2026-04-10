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
        mock_server_cls.assert_called_once_with(
            read_only=False, dd_only=None, no_embed=False
        )
        mock_server.run.assert_called_once()

    @patch("imas_codex.llm.server.AgentsServer")
    def test_serve_read_only(self, mock_server_cls, runner):
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server
        runner.invoke(main, ["serve", "--read-only"], catch_exceptions=False)
        mock_server_cls.assert_called_once_with(
            read_only=True, dd_only=None, no_embed=False
        )

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


class TestReadOnlyServer:
    """Tests that read-only mode suppresses write tools at the server level."""

    def test_read_only_suppresses_write_tools(self):
        """Read-only mode suppresses write tools and REPL."""
        from imas_codex.llm.server import AgentsServer

        server = AgentsServer(read_only=True)
        comps = server.mcp._local_provider._components
        tool_names = [v.name for k, v in comps.items() if k.startswith("tool:")]

        # These should NOT be present in read-only mode
        assert "repl" not in tool_names
        assert "add_to_graph" not in tool_names
        assert "update_facility_config" not in tool_names
        assert "list_logs" not in tool_names
        assert "get_logs" not in tool_names
        assert "tail_logs" not in tool_names

        # These SHOULD be present in read-only mode
        assert "get_graph_schema" in tool_names
        assert "get_facility_coverage" in tool_names
        assert "search_signals" in tool_names
        assert "search_docs" in tool_names
        assert "search_code" in tool_names
        assert "search_dd_paths" in tool_names
        assert "fetch_content" in tool_names
        assert "check_dd_paths" in tool_names

    def test_default_mode_has_all_tools(self):
        """Default mode includes all tools."""
        from imas_codex.llm.server import AgentsServer

        server = AgentsServer()
        comps = server.mcp._local_provider._components
        tool_names = [v.name for k, v in comps.items() if k.startswith("tool:")]

        # Write tools present
        assert "repl" in tool_names
        assert "add_to_graph" in tool_names
        assert "update_facility_config" in tool_names
        assert "list_logs" in tool_names
        assert "get_logs" in tool_names
        assert "tail_logs" in tool_names

        # Read tools also present
        assert "get_graph_schema" in tool_names
        assert "search_signals" in tool_names
        assert "fetch_content" in tool_names

    def test_read_only_server_name(self):
        """Read-only mode uses distinct server name."""
        from imas_codex.llm.server import AgentsServer

        ro_server = AgentsServer(read_only=True)
        rw_server = AgentsServer(read_only=False)

        assert ro_server.mcp.name == "imas-codex-readonly"
        assert rw_server.mcp.name == "imas-codex"

    def test_read_only_fewer_tools(self):
        """Read-only mode has strictly fewer tools than default."""
        from imas_codex.llm.server import AgentsServer

        ro_server = AgentsServer(read_only=True)
        rw_server = AgentsServer(read_only=False)

        ro_comps = ro_server.mcp._local_provider._components
        rw_comps = rw_server.mcp._local_provider._components
        ro_tools = {v.name for k, v in ro_comps.items() if k.startswith("tool:")}
        rw_tools = {v.name for k, v in rw_comps.items() if k.startswith("tool:")}

        assert ro_tools < rw_tools  # strict subset
        assert len(rw_tools) - len(ro_tools) == 6  # exactly 6 write tools suppressed

    def test_dd_only_excludes_facility_tools(self):
        """DD-only mode does not register any facility tools."""
        from imas_codex.llm.server import AgentsServer

        server = AgentsServer(dd_only=True)
        comps = server.mcp._local_provider._components
        tool_names = [v.name for k, v in comps.items() if k.startswith("tool:")]

        # Facility tools must NOT be registered
        for tool_name in AgentsServer.FACILITY_TOOLS:
            assert tool_name not in tool_names, (
                f"{tool_name} should not be in DD-only mode"
            )

        # Write tools must NOT be registered (dd-only implies read-only)
        assert "repl" not in tool_names
        assert "add_to_graph" not in tool_names
        assert "update_facility_config" not in tool_names

        # DD tools SHOULD still be present
        assert "search_dd_paths" in tool_names
        assert "check_dd_paths" in tool_names
        assert "fetch_dd_paths" in tool_names
        assert "find_related_dd_paths" in tool_names
        assert "get_graph_schema" in tool_names
        assert "get_dd_catalog" in tool_names

    def test_dd_only_implies_read_only(self):
        """DD-only mode automatically sets read_only=True."""
        from imas_codex.llm.server import AgentsServer

        server = AgentsServer(dd_only=True)
        assert server.read_only is True
        assert server.mcp.name == "imas-codex-readonly"


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
