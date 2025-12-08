"""
Test tools component composition and interface consistency.

This module tests the tools component functionality, focusing on interface
consistency, error handling, and composition patterns.
"""

import inspect

import pytest

from imas_mcp.models.error_models import ToolError
from imas_mcp.models.result_models import (
    IdentifierResult,
    OverviewResult,
    RelationshipResult,
    SearchResult,
)
from tests.conftest import STANDARD_TEST_IDS_SET


class TestToolsComposition:
    """Test tools component functionality and interface consistency."""

    @pytest.mark.asyncio
    async def test_all_tools_have_consistent_interfaces(self, tools, mcp_test_context):
        """Test all tools have consistent interfaces and can be called."""
        expected_tools = mcp_test_context["expected_tools"]

        for tool_name in expected_tools:
            tool_method = getattr(tools, tool_name)
            assert callable(tool_method)

            # All tools should be async methods
            assert inspect.iscoroutinefunction(tool_method)

    @pytest.mark.asyncio
    async def test_search_tool_interface(self, tools):
        """Test search tool interface and basic functionality."""
        result = await tools.search_imas_paths(
            query="plasma temperature", max_results=5
        )

        assert isinstance(result, SearchResult)
        assert hasattr(result, "hits")
        assert isinstance(result.hits, list)
        assert len(result.hits) <= 5
        assert hasattr(result, "hit_count")
        assert result.hit_count == len(result.hits)

    @pytest.mark.asyncio
    async def test_overview_tool_interface(self, tools):
        """Test overview tool interface and basic functionality."""
        result = await tools.get_imas_overview()

        # Test interface contract
        assert isinstance(result, OverviewResult)
        assert hasattr(result, "available_ids")
        assert isinstance(result.available_ids, list)
        assert hasattr(result, "content")
        assert hasattr(result, "physics_domains")

    @pytest.mark.asyncio
    async def test_relationships_tool_interface(self, tools, mcp_test_context):
        """Test relationships tool interface and basic functionality."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.search_imas_clusters(path=f"{ids_name}/profiles_1d/time")

        # Test interface contract - accept either RelationshipResult or ToolError
        assert isinstance(result, RelationshipResult | ToolError)

        if isinstance(result, RelationshipResult):
            assert hasattr(result, "path")
            assert hasattr(result, "connections")

    @pytest.mark.asyncio
    async def test_identifiers_tool_interface(self, tools, mcp_test_context):
        """Test identifiers tool interface and basic functionality."""
        result = await tools.list_imas_identifiers()

        # Test interface contract
        assert isinstance(result, IdentifierResult)
        assert hasattr(result, "schemas")
        assert hasattr(result, "analytics")


class TestToolsErrorHandling:
    """Test error handling patterns across all tools."""

    @pytest.mark.asyncio
    async def test_search_tool_invalid_parameters(self, tools):
        """Test search tool handles invalid parameters gracefully."""
        # Test with invalid max_results
        result = await tools.search_imas_paths(query="test", max_results=-1)

        # Should handle gracefully - either clamp to valid range or return error
        assert isinstance(result, ToolError)
        assert "max_results" in result.error.lower()


class TestToolsParameterValidation:
    """Test parameter validation patterns across tools."""

    @pytest.mark.asyncio
    async def test_search_tool_parameter_validation(self, tools):
        """Test search tool parameter validation."""
        # Test required parameter
        result = await tools.search_imas_paths(query="test")
        assert isinstance(result, SearchResult)

        # Test optional parameters
        result = await tools.search_imas_paths(
            query="test", max_results=10, ids_filter=["core_profiles"]
        )
        assert isinstance(result, SearchResult)


class TestToolsCompositionPattern:
    """Test tools composition pattern implementation."""

    def test_tools_component_properties(self, tools):
        """Test tools component has expected properties."""
        # Check main properties exist
        assert hasattr(tools, "document_store")
        assert hasattr(tools, "search_tool")

        # Document store should be consistent
        doc_store1 = tools.document_store
        doc_store2 = tools.document_store
        assert doc_store1 is doc_store2

    def test_tools_component_name(self, tools):
        """Test tools component has correct name."""
        assert tools.name == "tools"

    def test_tools_ids_set_configuration(self, tools):
        """Test tools component is configured with correct IDS set."""
        assert tools.ids_set == STANDARD_TEST_IDS_SET

    def test_tools_no_direct_instantiation_artifacts(self, tools):
        """Test tools component doesn't have direct instantiation artifacts."""
        # Should not have old-style individual tool instances
        old_style_attributes = [
            "search_composer",
            "graph_analyzer",
            "_search_cache",
            "_semantic_search",
        ]

        for attr in old_style_attributes:
            assert not hasattr(tools, attr), (
                f"Tools should not have old-style attribute: {attr}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
