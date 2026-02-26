"""
Test MCP server composition and protocol integration.

Tests the graph-backed server with mock GraphClient, focusing on
MCP protocol compliance and component integration.
"""

from unittest.mock import MagicMock

import pytest
from fastmcp import Client, FastMCP

from imas_codex.resource_provider import Resources
from imas_codex.server import Server
from imas_codex.tools import Tools
from tests.conftest import STANDARD_TEST_IDS_SET, _create_mock_graph_client


class TestMCPServer:
    """Test MCP server composition and protocol integration."""

    def test_server_initialization(self, server):
        """Test server initializes correctly with all components."""
        assert server is not None
        assert hasattr(server, "tools")
        assert hasattr(server, "resources")
        assert hasattr(server, "mcp")

        assert isinstance(server.tools, Tools)
        assert isinstance(server.resources, Resources)
        assert isinstance(server.mcp, FastMCP)

    def test_server_mcp_integration(self, server):
        """Test MCP protocol integration."""
        assert server.mcp is not None
        assert hasattr(server.mcp, "name")
        # Server name derived from DDVersion in graph
        assert server.mcp.name == "imas-data-dictionary-4.0.0"

    def test_server_ids_set_configuration(self, server):
        """Test server is configured with test IDS set."""
        assert server.tools.ids_set == STANDARD_TEST_IDS_SET

    @pytest.mark.asyncio
    async def test_server_tool_access(self, server):
        """Test tools are accessible through server composition."""
        expected_tools = [
            "check_imas_paths",
            "fetch_imas_paths",
            "get_imas_overview",
            "get_imas_identifiers",
            "list_imas_paths",
            "search_imas_clusters",
            "search_imas_paths",
        ]

        for tool_name in expected_tools:
            assert hasattr(server.tools, tool_name)
            assert callable(getattr(server.tools, tool_name))

    def test_server_resources_access(self, server):
        """Test resources are accessible through server composition."""
        assert server.resources is not None
        assert hasattr(server.resources, "register")
        assert server.resources.name == "resources"

    def test_no_legacy_delegation_methods(self, server):
        """Test server doesn't have old delegation methods."""
        delegation_methods = [
            "search_imas",
            "explain_concept",
            "get_overview",
            "analyze_ids_structure",
            "document_store",
            "search_tool",
            "embeddings",
        ]

        for method_name in delegation_methods:
            assert not hasattr(server, method_name), (
                f"Server should not have legacy method: {method_name}"
            )

    def test_server_run_method(self, server):
        """Test server has run method for MCP execution."""
        assert hasattr(server, "run")
        assert callable(server.run)

    def test_multiple_server_instances_isolation(self, server):
        """Test multiple server instances don't interfere."""
        mock_gc = _create_mock_graph_client()
        server2 = Server(ids_set={"equilibrium"}, graph_client=mock_gc)

        assert server is not server2
        assert server.tools is not server2.tools
        assert server.resources is not server2.resources
        assert server.mcp is not server2.mcp

        assert server.tools.ids_set == STANDARD_TEST_IDS_SET
        assert server2.tools.ids_set == {"equilibrium"}

    def test_server_requires_graph_client(self):
        """Server raises error when Neo4j is not reachable."""
        import neo4j.exceptions

        with pytest.raises(
            (
                RuntimeError,
                OSError,
                neo4j.exceptions.ServiceUnavailable,
                neo4j.exceptions.DatabaseUnavailable,
            )
        ):
            Server(ids_set=STANDARD_TEST_IDS_SET)

    def test_server_name_from_graph(self, server):
        """Server name is derived from DDVersion nodes in graph."""
        assert "4.0.0" in server.mcp.name

    def test_server_name_fallback(self):
        """Fallback name when DDVersion query fails."""
        from imas_codex.server import _server_name

        mock_gc = MagicMock()
        mock_gc.query.side_effect = Exception("connection refused")
        name = _server_name(mock_gc)
        assert name == "imas-data-dictionary"


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance using FastMCP Client."""

    @pytest.mark.asyncio
    async def test_mcp_basic_connectivity(self, server):
        """Test basic MCP connectivity through client."""
        async with Client(server.mcp) as client:
            await client.ping()

    @pytest.mark.asyncio
    async def test_tool_discovery_via_mcp(self, server):
        """Test tools can be discovered through MCP protocol."""
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
            tool_names = [tool.name for tool in tools]

            expected_tools = [
                "check_imas_paths",
                "fetch_imas_paths",
                "get_imas_overview",
                "get_imas_identifiers",
                "list_imas_paths",
                "search_imas_clusters",
                "search_imas_paths",
                "query_imas_graph",
                "get_dd_graph_schema",
                "get_dd_versions",
            ]

            for expected_tool in expected_tools:
                assert expected_tool in tool_names, f"Tool {expected_tool} not found"

    @pytest.mark.asyncio
    async def test_resource_discovery_via_mcp(self, server):
        """Test resources can be discovered through MCP protocol."""
        async with Client(server.mcp) as client:
            resources = await client.list_resources()
            resource_uris = [str(resource.uri) for resource in resources]

            # Only examples resource in graph-native mode
            assert "examples://resource-usage" in resource_uris

            # No schema-file resources
            assert "ids://catalog" not in resource_uris
            assert "ids://identifiers" not in resource_uris
            assert "ids://clusters" not in resource_uris


class TestServerComponentIntegration:
    """Test integration between server components."""

    def test_tools_component_initialization(self, server):
        """Test tools component is properly initialized."""
        tools = server.tools

        # Graph-backed tools — no document_store
        assert not hasattr(tools, "document_store")
        assert hasattr(tools, "search_tool")
        assert tools.search_tool is not None

    def test_resources_component_initialization(self, server):
        """Test resources component is properly initialized."""
        resources = server.resources
        assert resources.name == "resources"
        # No schema_dir in graph-native mode
        assert not hasattr(resources, "schema_dir")

    def test_component_independence(self, server):
        """Test components are properly decoupled."""
        assert server.tools is not server.resources
        assert server.tools.name == "tools"
        assert server.resources.name == "resources"
        assert not hasattr(server.tools, "schema_dir")
        assert not hasattr(server.resources, "document_store")

    @pytest.mark.asyncio
    async def test_component_registration_with_mcp(self, server):
        """Test that components are properly registered with MCP."""
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()

            assert len(tools) > 0
            assert len(resources) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
