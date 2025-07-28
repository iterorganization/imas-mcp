"""
Tests for OverviewTool implementation.

This module tests the get_overview tool with all decorators applied.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from imas_mcp.tools.overview_tool import OverviewTool
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.services.search_service import SearchService


class TestOverviewTool:
    """Test cases for OverviewTool."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store."""
        mock_store = Mock(spec=DocumentStore)
        mock_store.get_available_ids.return_value = [
            "core_profiles",
            "equilibrium",
            "transport",
            "heating",
            "wall",
        ]
        return mock_store

    @pytest.fixture
    def mock_search_service(self):
        """Create a mock search service."""
        mock_service = Mock(spec=SearchService)
        mock_service.search = AsyncMock(return_value=[])
        return mock_service

    @pytest.fixture
    def overview_tool(self, mock_document_store):
        """Create OverviewTool instance with mocked dependencies."""
        tool = OverviewTool()
        # Replace the ids_set and search service with our mocks
        tool.ids_set = {"core_profiles", "equilibrium", "transport", "heating", "wall"}
        tool._search_service = Mock(spec=SearchService)
        tool._search_service.search = AsyncMock(return_value=[])
        return tool

    def test_get_tool_name(self, overview_tool):
        """Test that tool returns correct name."""
        assert overview_tool.get_tool_name() == "get_overview"

    @pytest.mark.asyncio
    async def test_get_overview_without_query(self, overview_tool):
        """Test get_overview without specific query."""
        result = await overview_tool.get_overview()

        assert isinstance(result, dict)
        assert "content" in result
        assert "available_ids" in result
        assert len(result["available_ids"]) > 0
        assert "physics_domains" in result
        assert "ids_statistics" in result
        assert "usage_guidance" in result

    @pytest.mark.asyncio
    async def test_get_overview_with_query(self, overview_tool):
        """Test get_overview with specific query."""
        # Mock search results
        mock_result = Mock()
        mock_result.to_dict.return_value = {
            "path": "core_profiles/temperature",
            "documentation": "Test temperature data",
            "relevance_score": 0.9,
        }
        overview_tool._search_service.search.return_value = [mock_result]

        result = await overview_tool.get_overview(query="temperature")

        assert isinstance(result, dict)
        assert result["query"] == "temperature"
        assert "query_results" in result
        assert "query_results_count" in result
        assert result["query_results_count"] > 0

    @pytest.mark.asyncio
    async def test_get_overview_search_error(self, overview_tool):
        """Test get_overview when search fails."""
        # Make search raise an exception
        overview_tool._search_service.search.side_effect = Exception("Search failed")

        result = await overview_tool.get_overview(query="test")

        # Should not fail, just log warning and continue without search results
        assert isinstance(result, dict)
        assert "content" in result
        assert result["query"] == "test"

    @pytest.mark.asyncio
    async def test_get_overview_error_handling(self, overview_tool):
        """Test error handling in get_overview."""
        # Make the tool raise an exception
        overview_tool._search_service = None  # This will cause an error

        result = await overview_tool.get_overview()

        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result

    def test_build_overview_sample_prompt(self, overview_tool):
        """Test sample prompt building."""
        # Test with query
        prompt = overview_tool._build_overview_sample_prompt("temperature")
        assert "temperature" in prompt
        assert "IMAS Data Dictionary Overview Request" in prompt

        # Test without query
        prompt = overview_tool._build_overview_sample_prompt()
        assert "IMAS Data Dictionary General Overview" in prompt

    @pytest.mark.asyncio
    async def test_decorator_integration(self, overview_tool):
        """Test that decorators are properly applied."""
        # The tool should have the _mcp_tool attribute from the decorator
        assert hasattr(overview_tool.get_overview, "_mcp_tool")
        assert overview_tool.get_overview._mcp_tool is True
        assert hasattr(overview_tool.get_overview, "_mcp_description")

        # Test that the method can be called (decorators don't break it)
        result = await overview_tool.get_overview()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_ids_statistics_generation(self, overview_tool):
        """Test that IDS statistics are properly generated."""
        result = await overview_tool.get_overview()

        assert "ids_statistics" in result
        stats = result["ids_statistics"]

        # Should have stats for each available IDS
        assert len(stats) > 0
        for ids_name, ids_stats in stats.items():
            assert "path_count" in ids_stats
            assert "identifier_count" in ids_stats
            assert "description" in ids_stats
            assert isinstance(ids_stats["path_count"], int)
            assert isinstance(ids_stats["identifier_count"], int)

    @pytest.mark.asyncio
    async def test_usage_guidance_content(self, overview_tool):
        """Test that usage guidance contains expected content."""
        result = await overview_tool.get_overview()

        assert "usage_guidance" in result
        guidance = result["usage_guidance"]

        assert "tools_available" in guidance
        assert "getting_started" in guidance

        # Check for key tools in the list
        tools = guidance["tools_available"]
        tool_names = [tool.split(" - ")[0] for tool in tools]
        expected_tools = ["search_imas", "explain_concept", "analyze_ids_structure"]
        for expected in expected_tools:
            assert any(expected in tool for tool in tool_names)
