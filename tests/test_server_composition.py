"""Test the server composition pattern implementation."""

import pytest
from fastmcp import FastMCP

from imas_mcp.server import Server
from imas_mcp.tools import Tools
from imas_mcp.resources import Resources
from tests.conftest import STANDARD_TEST_IDS_SET


class TestServerComposition:
    """Test server composition pattern with tools and resources."""

    def test_server_initialization(self):
        """Test that server initializes correctly with composition pattern."""
        server = Server(ids_set=STANDARD_TEST_IDS_SET)

        # Check server has the expected components
        assert hasattr(server, "tools")
        assert hasattr(server, "resources")
        assert hasattr(server, "mcp")

        # Check components are of correct type
        assert isinstance(server.tools, Tools)
        assert isinstance(server.resources, Resources)
        assert isinstance(server.mcp, FastMCP)

    def test_server_initialization_with_ids_set(self):
        """Test server initialization with specific IDS set."""
        server = Server(ids_set=STANDARD_TEST_IDS_SET)

        # Check that tools component received the IDS set
        assert server.tools.ids_set == STANDARD_TEST_IDS_SET

        # Resources doesn't use ids_set parameter
        assert not hasattr(server.resources, "ids_set")

    def test_tools_component_access(self):
        """Test direct access to tools component."""
        server = Server(ids_set=STANDARD_TEST_IDS_SET)

        # Check tools component has expected methods
        assert hasattr(server.tools, "search_imas")
        assert hasattr(server.tools, "explain_concept")
        assert hasattr(server.tools, "get_overview")
        assert hasattr(server.tools, "analyze_ids_structure")
        assert hasattr(server.tools, "explore_relationships")
        assert hasattr(server.tools, "explore_identifiers")
        assert hasattr(server.tools, "export_ids")
        assert hasattr(server.tools, "export_physics_domain")

        # Check tools component has expected properties
        assert hasattr(server.tools, "document_store")
        assert hasattr(server.tools, "search_composer")
        assert hasattr(server.tools, "graph_analyzer")

    def test_resources_component_access(self):
        """Test direct access to resources component."""
        server = Server(ids_set=STANDARD_TEST_IDS_SET)

        # Check resources component has expected properties and methods
        assert hasattr(server.resources, "register")
        assert hasattr(server.resources, "schema_dir")

        # Check resources component has expected properties
        assert server.resources.name == "resources"

        # Check schema directory exists
        assert server.resources.schema_dir.exists()

    def test_mcp_registration(self):
        """Test that tools and resources are properly registered with MCP."""
        server = Server(ids_set=STANDARD_TEST_IDS_SET)

        # Check that MCP server has tools registered
        # FastMCP should have registered the tools
        assert server.mcp is not None

        # Check that both components were registered (they should have called register)
        # This is tested indirectly by checking the components exist and are initialized

    def test_no_delegation_methods(self):
        """Test that server doesn't have delegation methods anymore."""
        server = Server(ids_set=STANDARD_TEST_IDS_SET)

        # These methods should NOT exist on the server directly
        delegation_methods = [
            "search_imas",
            "explain_concept",
            "get_overview",
            "analyze_ids_structure",
            "explore_relationships",
            "explore_identifiers",
            "export_ids",
            "export_physics_domain",
            "document_store",
            "search_composer",
            "graph_analyzer",
        ]

        for method_name in delegation_methods:
            assert not hasattr(server, method_name), (
                f"Server should not have delegation method: {method_name}"
            )

    def test_tools_cached_properties(self):
        """Test that tools component has cached properties working."""
        server = Server(ids_set=STANDARD_TEST_IDS_SET)

        # Access cached properties - they should initialize lazily
        document_store = server.tools.document_store
        search_composer = server.tools.search_composer
        graph_analyzer = server.tools.graph_analyzer

        # Check they return the same instance on subsequent calls (cached)
        assert server.tools.document_store is document_store
        assert server.tools.search_composer is search_composer
        assert server.tools.graph_analyzer is graph_analyzer

    def test_server_run_method_exists(self):
        """Test that server still has run method for MCP."""
        server = Server(ids_set=STANDARD_TEST_IDS_SET)

        # Should have run method
        assert hasattr(server, "run")
        assert callable(server.run)

    def test_component_independence(self):
        """Test that tools and resources components are independent."""
        server = Server(ids_set=STANDARD_TEST_IDS_SET)

        # Tools and resources should be separate instances
        assert server.tools is not server.resources

        # They should have different names
        assert server.tools.name != server.resources.name
        assert server.tools.name == "tools"
        assert server.resources.name == "resources"

    def test_multiple_server_instances(self):
        """Test that multiple server instances don't interfere."""
        server1 = Server(ids_set=STANDARD_TEST_IDS_SET)
        server2 = Server(ids_set={"equilibrium"})  # Different subset

        # Should be separate instances
        assert server1 is not server2
        assert server1.tools is not server2.tools
        assert server1.resources is not server2.resources
        assert server1.mcp is not server2.mcp

        # Should have different IDS sets
        assert server1.tools.ids_set == STANDARD_TEST_IDS_SET
        assert server2.tools.ids_set == {"equilibrium"}


class TestServerComponentIntegration:
    """Test integration between server components."""

    @pytest.mark.asyncio
    async def test_tools_methods_callable(self):
        """Test that tools methods can be called through composition."""
        server = Server(ids_set=STANDARD_TEST_IDS_SET)

        # Test async methods exist and are callable
        assert callable(server.tools.search_imas)
        assert callable(server.tools.explain_concept)
        assert callable(server.tools.get_overview)

        # Note: We don't actually call them here since they might require
        # specific setup/data, but we verify they're accessible

    def test_resources_methods_callable(self):
        """Test that resources component is properly configured."""
        server = Server(ids_set=STANDARD_TEST_IDS_SET)

        # Test that resources component has required attributes
        assert hasattr(server.resources, "register")
        assert hasattr(server.resources, "schema_dir")
        assert hasattr(server.resources, "name")

        # Resources uses MCP decorators, so methods are registered with MCP, not directly accessible
        # The important thing is that register() can be called successfully
        assert callable(server.resources.register)

    def test_server_mcp_integration(self):
        """Test that server MCP integration works."""
        server = Server(ids_set=STANDARD_TEST_IDS_SET)

        # MCP should be initialized
        assert server.mcp is not None
        assert isinstance(server.mcp, FastMCP)

        # MCP should have name
        assert server.mcp.name == "imas"


if __name__ == "__main__":
    pytest.main([__file__])
