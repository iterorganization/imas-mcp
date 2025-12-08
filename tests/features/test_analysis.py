"""
Test analysis features through user interface.

This module tests analysis functionality as user-facing features,
focusing on relationship exploration and identifier discovery.
"""

import time

import pytest

from imas_mcp.models.error_models import ToolError
from imas_mcp.models.result_models import (
    GetIdentifiersResult,
    SearchClustersResult,
)


class TestAnalysisFeatures:
    """Test analysis functionality."""

    @pytest.mark.asyncio
    async def test_relationship_exploration_basic(self, tools, mcp_test_context):
        """Test basic relationship exploration."""
        ids_name = mcp_test_context["test_ids"]
        # Use a proper path with hierarchical separators, not just IDS name
        test_path = f"{ids_name}/profiles_1d/electrons/temperature"
        result = await tools.search_imas_clusters(path=test_path)

        # Accept either SearchClustersResult or ToolError (when clusters.json is missing)
        assert isinstance(result, SearchClustersResult | ToolError)

        if isinstance(result, SearchClustersResult):
            assert hasattr(result, "path")
            assert result.path == test_path

    @pytest.mark.asyncio
    async def test_relationship_types(self, tools, mcp_test_context):
        """Test relationship exploration identifies different relationship types."""
        ids_name = mcp_test_context["test_ids"]
        # Use a proper path with hierarchical separators
        test_path = f"{ids_name}/profiles_1d/electrons/temperature"
        result = await tools.search_imas_clusters(path=test_path)

        # Accept either SearchClustersResult or ToolError (when clusters.json is missing)
        assert isinstance(result, SearchClustersResult | ToolError)

        if isinstance(result, SearchClustersResult) and hasattr(result, "connections"):
            connections = result.connections
            # Should provide structured relationship information
            assert isinstance(connections, dict)

    @pytest.mark.asyncio
    async def test_identifier_exploration_basic(self, tools, mcp_test_context):
        """Test basic identifier exploration."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.get_imas_identifiers(query=ids_name)

        assert isinstance(result, GetIdentifiersResult)
        # Check for the actual fields that are returned
        assert hasattr(result, "analytics")
        assert hasattr(result, "schemas")

    @pytest.mark.asyncio
    async def test_identifier_structure_information(self, tools, mcp_test_context):
        """Test identifier exploration provides structure information."""
        ids_name = mcp_test_context["test_ids"]
        result = await tools.get_imas_identifiers(query=ids_name)

        assert isinstance(result, GetIdentifiersResult)
        if hasattr(result, "schemas"):
            schemas = result.schemas
            # Should provide identifier structure information
            assert isinstance(schemas, dict | list)


class TestAnalysisErrorHandling:
    """Test analysis tools error handling."""

    @pytest.mark.asyncio
    async def test_relationships_invalid_ids_name(self, tools):
        """Test relationship exploration handles invalid IDS names."""
        invalid_path = "nonexistent_ids_name/invalid/path"

        result = await tools.search_imas_clusters(path=invalid_path)
        assert isinstance(result, ToolError)
        # Should provide helpful error information
        assert isinstance(result.error, str)

    @pytest.mark.asyncio
    async def test_identifiers_invalid_ids_name(self, tools):
        """Test identifier exploration handles invalid IDS names."""
        invalid_ids = "nonexistent_ids_name"

        result = await tools.get_imas_identifiers(query=invalid_ids)
        # May return either success with empty results or error - both valid
        assert isinstance(result, GetIdentifiersResult | ToolError)


class TestAnalysisPerformance:
    """Test analysis performance characteristics."""

    @pytest.mark.asyncio
    async def test_identifiers_response_time(self, tools):
        """Test identifiers tool responds in reasonable time."""
        start_time = time.time()
        result = await tools.get_imas_identifiers()
        end_time = time.time()

        execution_time = end_time - start_time
        assert execution_time < 10.0, (
            f"Identifiers took {execution_time:.2f}s, too slow"
        )
        assert isinstance(result, GetIdentifiersResult | ToolError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
