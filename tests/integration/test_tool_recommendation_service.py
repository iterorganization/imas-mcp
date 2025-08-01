"""Integration tests for tool recommendation service."""

import pytest
from imas_mcp.services.tool_recommendations import (
    ToolRecommendationService,
    RecommendationStrategy,
)


class TestToolRecommendationServiceIntegration:
    """Test tool recommendation service integration."""

    @pytest.fixture
    def recommendation_service(self):
        return ToolRecommendationService()

    @pytest.fixture
    def search_result_with_hits(self):
        """Search result with hits for testing."""
        from unittest.mock import MagicMock

        hit1 = MagicMock()
        hit1.path = "core_profiles/profiles_1d/temperature"
        hit1.physics_domain = "core_transport"

        hit2 = MagicMock()
        hit2.path = "equilibrium/time_slice/boundary"
        hit2.physics_domain = "equilibrium"

        result = MagicMock()
        result.hits = [hit1, hit2]
        return result

    def test_search_based_recommendations(
        self, recommendation_service, search_result_with_hits
    ):
        """Test search-based recommendation generation."""
        recommendations = recommendation_service.generate_recommendations(
            result=search_result_with_hits,
            strategy=RecommendationStrategy.SEARCH_BASED,
            max_tools=4,
            query="temperature profile",
        )

        assert len(recommendations) > 0
        assert len(recommendations) <= 4

        # Check recommendation structure
        for rec in recommendations:
            assert "tool" in rec
            assert "reason" in rec
            assert "description" in rec

    def test_concept_based_recommendations(self, recommendation_service):
        """Test concept-based recommendations."""
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.concept = "temperature"

        recommendations = recommendation_service.generate_recommendations(
            result=mock_result,
            strategy=RecommendationStrategy.CONCEPT_BASED,
            max_tools=3,
        )

        assert len(recommendations) <= 3
        # Concept-based recommendations should include overview as a common suggestion
        tool_names = [rec["tool"] for rec in recommendations]
        assert "get_overview" in tool_names

    def test_error_result_recommendations(self, recommendation_service):
        """Test recommendations for error results."""
        error_result = {"error": "Search failed"}

        recommendations = recommendation_service.generate_recommendations(
            result=error_result, strategy=RecommendationStrategy.SEARCH_BASED
        )

        # Should provide helpful fallback recommendations
        assert len(recommendations) > 0
        tool_names = [rec["tool"] for rec in recommendations]
        assert "get_overview" in tool_names

    def test_empty_result_recommendations(self, recommendation_service):
        """Test recommendations for empty search results."""
        from unittest.mock import MagicMock

        empty_result = MagicMock()
        empty_result.hits = []

        recommendations = recommendation_service.generate_recommendations(
            result=empty_result,
            strategy=RecommendationStrategy.SEARCH_BASED,
            query="nonexistent",
        )

        assert len(recommendations) > 0
        # Should suggest overview for empty results
        tool_names = [rec["tool"] for rec in recommendations]
        assert "get_overview" in tool_names
