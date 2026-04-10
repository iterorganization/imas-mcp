"""Tests for graph-native tool registration in the Tools class."""

import pytest

pytestmark = pytest.mark.graph_mcp


class TestToolRegistration:
    """Verify graph-native tools are registered when graph_client is provided."""

    def test_graph_tools_registered(self, graph_client):
        """Graph tools appear in registered tool names when client provided."""
        from imas_codex.tools import Tools

        tools = Tools(graph_client=graph_client)
        names = tools.get_registered_tool_names()
        # query_imas_graph and get_dd_graph_schema were removed in the unified server cleanup
        assert "get_dd_versions" in names
        assert "search_dd_paths" in names

    def test_graph_client_required(self):
        """Tools raises ValueError when no graph_client provided."""
        from imas_codex.tools import Tools

        with pytest.raises(ValueError, match="GraphClient is required"):
            Tools()

    def test_existing_tools_still_registered(self, graph_client):
        """Original tools still present alongside graph tools."""
        from imas_codex.tools import Tools

        tools = Tools(graph_client=graph_client)
        names = tools.get_registered_tool_names()
        assert "search_dd_paths" in names
        assert "fetch_dd_paths" in names
        assert "list_dd_paths" in names
        assert "get_dd_catalog" in names

    def test_total_tool_count(self, graph_client):
        """Total tool count matches expected number of graph-backed tools."""
        from imas_codex.tools import Tools

        tools = Tools(graph_client=graph_client)
        names = tools.get_registered_tool_names()
        # All tools are graph-backed: 11 tool classes each with their MCP tools
        assert len(names) >= 11
