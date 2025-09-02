"""Integration tests for service composition in tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_mcp.tools.search_tool import SearchTool


class TestServiceComposition:
    """Test service integration across tools."""

    @pytest.fixture
    def search_tool(self):
        with patch("imas_mcp.tools.base.DocumentStore"):
            return SearchTool()

    @pytest.mark.asyncio
    async def test_search_tool_with_services(self, search_tool):
        """Test SearchTool uses core services for search functionality."""
        # Import SearchResponse for proper mocking
        from imas_mcp.search.search_strategy import SearchResponse

        # Mock search execution
        with patch.object(search_tool, "_search_service") as mock_search:
            # Mock SearchResponse return from search service
            mock_search.search = AsyncMock(
                return_value=SearchResponse(hits=[], all_paths=[])
            )

            # Mock response service
            mock_response = MagicMock()
            mock_response.hits = []
            search_tool.response.build_search_response = MagicMock(
                return_value=mock_response
            )

            # Execute search
            result = await search_tool.search_imas(
                query="test query", search_mode="semantic"
            )

            # Verify search service was used
            assert mock_search.search.called
            # Result should contain service-generated content
            assert result is not None

    def test_template_method_customization(self, search_tool):
        """Test template method pattern allows tool-specific customization."""
        # Verify SearchTool has core services available
        assert hasattr(search_tool, "physics")
        assert hasattr(search_tool, "response")
        assert hasattr(search_tool, "documents")
        assert hasattr(search_tool, "search_config")

    @pytest.mark.asyncio
    async def test_apply_services_method(self, search_tool):
        """Test the service composition works correctly through decorators."""
        from imas_mcp.models.result_models import SearchResult

        # Mock the search execution to return controlled results
        mock_result = SearchResult(hits=[], query="test")
        search_tool.execute_search = AsyncMock(return_value=mock_result)

        # Mock physics enhancement
        search_tool.physics.enhance_query = AsyncMock(return_value=None)

        # Execute the search which should trigger service composition through decorators
        result = await search_tool.search_imas(query="test")

        # Verify search was executed
        search_tool.execute_search.assert_called_once()

        # Verify physics enhancement was attempted
        search_tool.physics.enhance_query.assert_called_once()

        # Result should be the mock result
        assert result == mock_result
