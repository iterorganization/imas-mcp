"""Integration tests for service composition in tools."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from imas_mcp.tools.search_tool import SearchTool


class TestServiceComposition:
    """Test service integration across tools."""

    @pytest.fixture
    def search_tool(self):
        with patch("imas_mcp.tools.base.DocumentStore"):
            return SearchTool()

    @pytest.mark.asyncio
    async def test_search_tool_with_services(self, search_tool):
        """Test SearchTool uses both sampling and recommendation services."""
        # Mock search execution
        with patch.object(search_tool, "_search_service") as mock_search:
            mock_result = MagicMock()
            mock_result.hits = []
            mock_result.total_hits = 0
            mock_search.search = AsyncMock(return_value=[])

            # Mock response service
            mock_response = MagicMock()
            mock_response.hits = []
            mock_response.tool_recommendations = []
            search_tool.response.build_search_response = MagicMock(
                return_value=mock_response
            )
            search_tool.response.add_standard_metadata = MagicMock(
                return_value=mock_response
            )

            # Mock services
            search_tool.sampling.apply_sampling = AsyncMock(return_value=mock_response)
            search_tool.recommendations.apply_recommendations = MagicMock(
                return_value=mock_response
            )

            # Execute search
            result = await search_tool.search_imas(
                query="test query", search_mode="semantic"
            )

            # Verify services were integrated
            assert mock_search.search.called
            # Result should contain service-generated content
            assert result is not None

    def test_service_dependency_injection(self, search_tool):
        """Test all services are properly injected."""
        # Verify core services are present (Phase 2.5 focus)
        assert hasattr(search_tool, "sampling")
        assert hasattr(search_tool, "recommendations")

        # Verify service types for Phase 2.5 services
        from imas_mcp.services.sampling import SamplingService
        from imas_mcp.services.tool_recommendations import ToolRecommendationService

        assert isinstance(search_tool.sampling, SamplingService)
        assert isinstance(search_tool.recommendations, ToolRecommendationService)

    def test_template_method_customization(self, search_tool):
        """Test template method pattern allows tool-specific customization."""
        # Verify SearchTool has appropriate settings
        assert search_tool.enable_sampling
        assert search_tool.enable_recommendations
        assert search_tool.max_recommended_tools == 5

    @pytest.mark.asyncio
    async def test_apply_services_method(self, search_tool):
        """Test the apply_services method works correctly."""
        from imas_mcp.models.response_models import SearchResponse

        # Create a proper AIResponse instance instead of MagicMock
        mock_result = SearchResponse(hits=[], query="test", ai_insights={})

        # Mock the services
        search_tool.sampling.apply_sampling = AsyncMock(return_value=mock_result)
        search_tool.recommendations.apply_recommendations = MagicMock(
            return_value=mock_result
        )

        result = await search_tool.apply_services(
            result=mock_result, query="test", ctx=MagicMock()
        )

        # Verify sampling was called
        search_tool.sampling.apply_sampling.assert_called_once()

        # Verify recommendations were called
        search_tool.recommendations.apply_recommendations.assert_called_once()

        # Result should be returned (services modify in place)
        assert result is mock_result
