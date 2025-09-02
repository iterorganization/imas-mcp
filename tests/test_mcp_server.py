"""
Test MCP server composition and protocol integration.

This module tests the core server functionality with the new composition pattern,
focusing on MCP protocol compliance and component integration using FastMCP
in-memory testing patterns.
"""

import asyncio

import pytest
from fastmcp import Client, FastMCP

from imas_mcp.models.result_models import OverviewResult
from imas_mcp.resource_provider import Resources
from imas_mcp.server import Server
from imas_mcp.tools import Tools
from tests.conftest import STANDARD_TEST_IDS_SET


class TestMCPServer:
    """Test MCP server composition and protocol integration."""

    def test_server_initialization(self, server):
        """Test server initializes correctly with all components."""
        assert server is not None
        assert hasattr(server, "tools")
        assert hasattr(server, "resources")
        assert hasattr(server, "mcp")

        # Check component types
        assert isinstance(server.tools, Tools)
        assert isinstance(server.resources, Resources)
        assert isinstance(server.mcp, FastMCP)

    def test_server_mcp_integration(self, server):
        """Test MCP protocol integration."""
        # MCP should be properly initialized
        assert server.mcp is not None
        assert server.mcp.name == "imas"

    def test_server_ids_set_configuration(self, server):
        """Test server is configured with test IDS set."""
        assert server.tools.ids_set == STANDARD_TEST_IDS_SET

    @pytest.mark.asyncio
    async def test_server_tool_access(self, server):
        """Test tools are accessible through server composition."""
        # Test that all expected tools are available
        expected_tools = [
            "search_imas",
            "explain_concept",
            "get_overview",
            "analyze_ids_structure",
            "explore_relationships",
            "explore_identifiers",
            "export_ids",
            "export_physics_domain",
        ]

        for tool_name in expected_tools:
            assert hasattr(server.tools, tool_name)
            assert callable(getattr(server.tools, tool_name))

    def test_server_resources_access(self, server):
        """Test resources are accessible through server composition."""
        assert server.resources is not None
        assert hasattr(server.resources, "register")
        assert hasattr(server.resources, "schema_dir")
        assert server.resources.name == "resources"

    def test_no_legacy_delegation_methods(self, server):
        """Test server doesn't have old delegation methods."""
        # These methods should NOT exist on server directly
        delegation_methods = [
            "search_imas",
            "explain_concept",
            "get_overview",
            "analyze_ids_structure",
            "document_store",
            "search_tool",
        ]

        for method_name in delegation_methods:
            assert not hasattr(server, method_name), (
                f"Server should not have legacy method: {method_name}"
            )

    def test_server_run_method(self, server):
        """Test server has run method for MCP execution."""
        assert hasattr(server, "run")
        assert callable(server.run)

    def test_multiple_server_instances_isolation(self):
        """Test multiple server instances don't interfere."""
        server1 = Server(ids_set=STANDARD_TEST_IDS_SET)
        server2 = Server(ids_set={"equilibrium"})

        # Should be separate instances
        assert server1 is not server2
        assert server1.tools is not server2.tools
        assert server1.resources is not server2.resources
        assert server1.mcp is not server2.mcp

        # Should have different configurations
        assert server1.tools.ids_set == STANDARD_TEST_IDS_SET
        assert server2.tools.ids_set == {"equilibrium"}


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance and interaction patterns using FastMCP Client."""

    @pytest.mark.asyncio
    async def test_mcp_basic_connectivity(self, server):
        """Test basic MCP connectivity through client."""
        async with Client(server.mcp) as client:
            # Test ping connectivity
            await client.ping()

    @pytest.mark.asyncio
    async def test_tool_discovery_via_mcp(self, server):
        """Test tools can be discovered through MCP protocol."""
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
            tool_names = [tool.name for tool in tools]

            # Check that expected tools are registered
            expected_tools = [
                "search_imas",
                "explain_concept",
                "get_overview",
                "analyze_ids_structure",
                "explore_relationships",
                "explore_identifiers",
                "export_ids",
                "export_physics_domain",
            ]

            for expected_tool in expected_tools:
                assert expected_tool in tool_names, f"Tool {expected_tool} not found"

    @pytest.mark.asyncio
    async def test_resource_discovery_via_mcp(self, server):
        """Test resources can be discovered through MCP protocol."""
        async with Client(server.mcp) as client:
            resources = await client.list_resources()
            resource_uris = [str(resource.uri) for resource in resources]

            # Check that expected resources are registered
            expected_resources = [
                "ids://catalog",
                "ids://identifiers",
                "ids://relationships",
                "examples://resource-usage",
            ]

            for expected_resource in expected_resources:
                assert expected_resource in resource_uris, (
                    f"Resource {expected_resource} not found"
                )

    @pytest.mark.asyncio
    async def test_tool_execution_via_mcp(self, server):
        """Test tools can be executed through MCP client."""
        async with Client(server.mcp) as client:
            # Test simple tool execution
            result = await client.call_tool("search_imas", {"query": "plasma"})
            assert result is not None
            assert hasattr(result, "content") or hasattr(result, "data")

            # Test tool with different parameters
            result = await client.call_tool("get_overview")
            assert result is not None
            assert hasattr(result, "content") or hasattr(result, "data")

    @pytest.mark.asyncio
    async def test_resource_access_via_mcp(self, server):
        """Test resources can be accessed through MCP client."""
        async with Client(server.mcp) as client:
            # Test resource reading
            resource_data = await client.read_resource("ids://catalog")
            assert resource_data is not None
            assert len(resource_data) > 0

            # Resource data should have text content
            first_content = resource_data[0]
            assert hasattr(first_content, "text")

    @pytest.mark.asyncio
    async def test_multiple_concurrent_operations(self, server):
        """Test concurrent MCP operations work correctly."""
        async with Client(server.mcp) as client:
            # Execute multiple operations concurrently
            results = await asyncio.gather(
                client.call_tool("search_imas", {"query": "equilibrium"}),
                client.call_tool("get_overview"),
                client.read_resource("ids://identifiers"),
                return_exceptions=True,
            )

            # All operations should succeed
            for result in results:
                assert not isinstance(result, Exception)
                assert result is not None


class TestServerComponentIntegration:
    """Test integration between server components."""

    def test_tools_component_initialization(self, server):
        """Test tools component is properly initialized."""
        tools = server.tools

        # Check core properties exist
        assert hasattr(tools, "document_store")
        assert hasattr(tools, "search_tool")
        assert hasattr(tools, "analysis_tool")

        # Check that document store is shared properly
        assert tools.document_store is not None

        # Check that individual tools are initialized
        assert tools.search_tool is not None
        assert tools.analysis_tool is not None

    def test_resources_component_initialization(self, server):
        """Test resources component is properly initialized."""
        resources = server.resources

        # Check resources configuration
        assert resources.name == "resources"
        assert hasattr(resources, "schema_dir")
        assert resources.schema_dir.exists()

    @pytest.mark.asyncio
    async def test_cross_component_functionality(self, server):
        """Test functionality that spans multiple components."""
        # Test that tools and resources work together
        # For example, tools might access schema resources

        # Get an overview (uses tools)
        overview_result = await server.tools.get_overview()
        assert isinstance(overview_result, OverviewResult)

        # Resources should provide schema information
        assert server.resources.schema_dir.exists()

    def test_component_independence(self, server):
        """Test components are properly decoupled."""
        # Tools and resources should be independent
        assert server.tools is not server.resources
        assert server.tools.name == "tools"
        assert server.resources.name == "resources"

        # They should not share state beyond what's necessary
        assert not hasattr(server.tools, "schema_dir")
        assert not hasattr(server.resources, "document_store")

    @pytest.mark.asyncio
    async def test_component_registration_with_mcp(self, server):
        """Test that components are properly registered with MCP."""
        async with Client(server.mcp) as client:
            # Verify both tools and resources are available through MCP
            tools = await client.list_tools()
            resources = await client.list_resources()

            # Should have tools from tools component
            assert len(tools) > 0

            # Should have resources from resources component
            assert len(resources) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
