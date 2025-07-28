"""
Tests for RelationshipsTool implementation.

This module tests the explore_relationships tool with all decorators applied.
"""

import pytest
from unittest.mock import Mock
from imas_mcp.tools.relationships_tool import RelationshipsTool
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.search_strategy import SearchComposer


class TestRelationshipsTool:
    """Test cases for RelationshipsTool."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store."""
        mock_store = Mock(spec=DocumentStore)
        mock_store.get_available_ids.return_value = [
            "core_profiles",
            "equilibrium",
            "transport",
        ]
        return mock_store

    @pytest.fixture
    def mock_search_composer(self):
        """Create a mock search composer."""
        mock_composer = Mock(spec=SearchComposer)
        mock_composer.search_with_params.return_value = {
            "results": [
                {
                    "path": "equilibrium/magnetic_field",
                    "documentation": "Magnetic field equilibrium data related to core temperature",
                    "physics_domain": "equilibrium",
                    "data_type": "float",
                    "units": "T",
                },
                {
                    "path": "transport/heat_flux",
                    "documentation": "Heat transport flux related to temperature gradients",
                    "physics_domain": "transport",
                    "data_type": "float",
                    "units": "MW/m^2",
                },
            ]
        }
        return mock_composer

    @pytest.fixture
    def relationships_tool(self, mock_document_store, mock_search_composer):
        """Create RelationshipsTool instance with mocked dependencies."""
        return RelationshipsTool(
            document_store=mock_document_store, search_composer=mock_search_composer
        )

    def test_get_tool_name(self, relationships_tool):
        """Test that tool returns correct name."""
        assert relationships_tool.get_tool_name() == "explore_relationships"

    @pytest.mark.asyncio
    async def test_explore_relationships_with_specific_path(self, relationships_tool):
        """Test relationship exploration with specific path."""
        result = await relationships_tool.explore_relationships(
            path="core_profiles/temperature"
        )

        assert isinstance(result, dict)
        assert result["path"] == "core_profiles/temperature"
        assert "relationship_type" in result
        assert "max_depth" in result
        assert "connections" in result
        assert "paths" in result
        assert "count" in result
        assert "physics_domains" in result

        # Check connections structure
        connections = result["connections"]
        assert "total_relationships" in connections
        assert "physics_connections" in connections
        assert "cross_ids_connections" in connections

    @pytest.mark.asyncio
    async def test_explore_relationships_with_ids_only(self, relationships_tool):
        """Test relationship exploration with IDS name only."""
        result = await relationships_tool.explore_relationships(path="core_profiles")

        assert isinstance(result, dict)
        assert result["path"] == "core_profiles"
        assert "paths" in result
        assert result["count"] > 0

    @pytest.mark.asyncio
    async def test_explore_relationships_invalid_ids(self, relationships_tool):
        """Test relationship exploration with invalid IDS name."""
        result = await relationships_tool.explore_relationships(path="invalid_ids")

        assert isinstance(result, dict)
        assert "error" in result
        assert "invalid_ids" in result["error"]
        assert "available_ids" in result
        assert "suggestions" in result

    @pytest.mark.asyncio
    async def test_explore_relationships_search_error(self, relationships_tool):
        """Test relationship exploration when search fails."""
        # Make search raise an exception
        relationships_tool.search_composer.search_with_params.side_effect = Exception(
            "Search failed"
        )

        result = await relationships_tool.explore_relationships(path="core_profiles")

        assert isinstance(result, dict)
        assert "error" in result
        assert "Failed to search relationships" in result["error"]

    @pytest.mark.asyncio
    async def test_explore_relationships_different_types(self, relationships_tool):
        """Test different relationship types."""
        for rel_type in ["all", "parent", "child", "sibling"]:
            result = await relationships_tool.explore_relationships(
                path="core_profiles", relationship_type=rel_type
            )

            assert isinstance(result, dict)
            assert result["relationship_type"].value == rel_type.upper()

    @pytest.mark.asyncio
    async def test_explore_relationships_max_depth_validation(self, relationships_tool):
        """Test max_depth validation and limits."""
        # Test depth > 3 gets limited to 3
        result = await relationships_tool.explore_relationships(
            path="core_profiles", max_depth=5
        )
        assert result["max_depth"] == 3

        # Test depth < 1 gets set to 1
        result = await relationships_tool.explore_relationships(
            path="core_profiles", max_depth=0
        )
        assert result["max_depth"] == 1

    @pytest.mark.asyncio
    async def test_explore_relationships_error_handling(self, relationships_tool):
        """Test general error handling."""
        # Make document store raise an exception
        relationships_tool.document_store.get_available_ids.side_effect = Exception(
            "Database error"
        )

        result = await relationships_tool.explore_relationships(path="core_profiles")

        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result

    def test_build_relationships_sample_prompt(self, relationships_tool):
        """Test sample prompt building."""
        prompt = relationships_tool._build_relationships_sample_prompt(
            "core_profiles/temperature", "parent"
        )

        assert "core_profiles/temperature" in prompt
        assert "parent" in prompt
        assert "IMAS Relationship Exploration Request" in prompt
        assert "Connected Data Paths" in prompt

    @pytest.mark.asyncio
    async def test_decorator_integration(self, relationships_tool):
        """Test that decorators are properly applied."""
        # The tool should have the _mcp_tool attribute from the decorator
        assert hasattr(relationships_tool.explore_relationships, "_mcp_tool")
        assert relationships_tool.explore_relationships._mcp_tool is True
        assert hasattr(relationships_tool.explore_relationships, "_mcp_description")

        # Test that the method can be called (decorators don't break it)
        result = await relationships_tool.explore_relationships(path="core_profiles")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_physics_context_processing(self, relationships_tool):
        """Test physics context processing from search results."""
        result = await relationships_tool.explore_relationships(
            path="core_profiles/temperature"
        )

        # Should have found physics domains from the mock results
        physics_domains = result["physics_domains"]
        assert "equilibrium" in physics_domains
        assert "transport" in physics_domains

    @pytest.mark.asyncio
    async def test_cross_ids_connections(self, relationships_tool):
        """Test cross-IDS connection detection."""
        result = await relationships_tool.explore_relationships(
            path="core_profiles/temperature"
        )

        connections = result["connections"]
        cross_ids = connections["cross_ids_connections"]

        # Should detect connections to equilibrium and transport IDS
        assert "equilibrium" in cross_ids
        assert "transport" in cross_ids

    @pytest.mark.asyncio
    async def test_path_filtering(self, relationships_tool):
        """Test that the original path is filtered out from results."""
        result = await relationships_tool.explore_relationships(
            path="core_profiles/temperature"
        )

        # Original path should not appear in the related paths
        for path_obj in result["paths"]:
            assert path_obj["path"] != "core_profiles/temperature"
