"""
Test resources component functionality and MCP integration.
"""

import pytest


class TestResourcesComponent:
    """Test resources component functionality."""

    def test_resources_initialization(self, resources):
        """Test resources component initializes correctly."""
        assert resources is not None
        assert resources.name == "resources"

    def test_resources_register_method(self, resources):
        """Test resources component has register method."""
        assert hasattr(resources, "register")
        assert callable(resources.register)


class TestResourcesMCPIntegration:
    """Test resources MCP integration and registration."""

    def test_resources_mcp_registration(self, server):
        """Test resources are properly registered with MCP."""
        assert server.resources is not None
        assert server.mcp is not None


class TestResourcesIndependence:
    """Test resources component independence from tools."""

    def test_resources_tool_independence(self, server):
        """Test resources component is independent from tools."""
        assert server.resources is not server.tools
        assert not hasattr(server.resources, "document_store")
        assert not hasattr(server.tools, "schema_dir")

    def test_resources_configuration_independence(self, resources):
        """Test resources component has its own configuration."""
        assert resources.name == "resources"
        if hasattr(resources, "ids_set"):
            value = resources.ids_set
            assert value is None or isinstance(value, set)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
