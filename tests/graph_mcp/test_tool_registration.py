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
        assert "query_imas_graph" in names
        assert "get_dd_graph_schema" in names
        assert "get_dd_versions" in names

    def test_graph_tools_absent_without_client(self):
        """Graph tools are not registered when no graph_client provided."""
        from imas_codex.tools import Tools

        tools = Tools()
        names = tools.get_registered_tool_names()
        assert "query_imas_graph" not in names
        assert "get_dd_graph_schema" not in names
        assert "get_dd_versions" not in names

    def test_existing_tools_still_registered(self, graph_client):
        """Original tools still present alongside graph tools."""
        from imas_codex.tools import Tools

        tools = Tools(graph_client=graph_client)
        names = tools.get_registered_tool_names()
        assert "search_imas_paths" in names
        assert "fetch_imas_paths" in names
        assert "list_imas_paths" in names
        assert "get_imas_overview" in names

    def test_total_tool_count(self, graph_client):
        """Total tool count increases with graph tools."""
        from imas_codex.tools import Tools

        without = Tools()
        with_graph = Tools(graph_client=graph_client)
        assert len(with_graph.get_registered_tool_names()) == (
            len(without.get_registered_tool_names()) + 3
        )
