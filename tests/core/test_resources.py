"""
Test resources component functionality and MCP integration.

This module tests the resources component, focusing on schema access,
MCP registration, and resource management.
"""

from pathlib import Path

import pytest


class TestResourcesComponent:
    """Test resources component functionality."""

    def test_resources_initialization(self, resources):
        """Test resources component initializes correctly."""
        assert resources is not None
        assert resources.name == "resources"

    def test_resources_schema_directory(self, resources):
        """Test resources component has valid schema directory."""
        assert hasattr(resources, "schema_dir")
        assert isinstance(resources.schema_dir, Path)
        assert resources.schema_dir.exists()

    def test_resources_register_method(self, resources):
        """Test resources component has register method."""
        assert hasattr(resources, "register")
        assert callable(resources.register)

    def test_schema_directory_contents(self, resources):
        """Test schema directory contains expected files."""
        schema_dir = resources.schema_dir

        # Should contain schema files
        schema_files = list(schema_dir.glob("*.json"))
        assert len(schema_files) > 0, "Schema directory should contain JSON files"

        # Check for expected schema types
        expected_patterns = ["*.json"]
        for pattern in expected_patterns:
            files = list(schema_dir.glob(pattern))
            assert len(files) > 0, f"Should contain files matching {pattern}"


class TestResourcesMCPIntegration:
    """Test resources MCP integration and registration."""

    def test_resources_mcp_registration(self, server):
        """Test resources are properly registered with MCP."""
        # Resources should be registered with the MCP server
        assert server.resources is not None

        # MCP server should be initialized with resources
        assert server.mcp is not None


class TestResourcesIndependence:
    """Test resources component independence from tools."""

    def test_resources_tool_independence(self, server):
        """Test resources component is independent from tools."""
        # Resources and tools should be separate
        assert server.resources is not server.tools

        # Resources should not depend on tools
        assert not hasattr(server.resources, "document_store")
        assert not hasattr(server.resources, "search_composer")

        # Tools should not depend on resources internals
        assert not hasattr(server.tools, "schema_dir")

    def test_resources_configuration_independence(self, resources):
        """Test resources component has its own configuration."""
        # Resources should have its own name and configuration
        assert resources.name == "resources"

        # ids_set is an optional relationships optimization and not a tools config.
        # If present, it should be either None (default) or a set of IDS names.
        if hasattr(resources, "ids_set"):
            value = resources.ids_set
            assert value is None or isinstance(value, set)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
