"""Tests for SearchTool with service composition."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from imas_mcp.tools.search_tool import SearchTool
from imas_mcp.models.response_models import SearchResponse
from imas_mcp.models.constants import SearchMode


class TestSearchToolServices:
    """Test SearchTool service composition functionality."""

    @pytest.fixture
    def search_tool(self):
        with patch("imas_mcp.tools.base.DocumentStore"):
            return SearchTool()

    @pytest.mark.asyncio
    async def test_search_with_physics_enhancement(self, search_tool):
        """Test search with physics enhancement through service."""

        # Mock search service
        search_tool._search_service.search = AsyncMock(return_value=[])

        # Mock physics service
        mock_physics_result = MagicMock()
        search_tool.physics.enhance_query = AsyncMock(return_value=mock_physics_result)

        # Mock response service
        mock_response = SearchResponse(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="test query",
            ai_response={},
            ai_prompt={},
        )
        search_tool.response.build_search_response = MagicMock(
            return_value=mock_response
        )
        search_tool.response.add_standard_metadata = MagicMock(
            return_value=mock_response
        )

        # Mock context for physics enhancement
        mock_ctx = MagicMock()

        result = await search_tool.search_imas(query="plasma temperature", ctx=mock_ctx)

        # Verify services were called
        search_tool.physics.enhance_query.assert_called_once_with("plasma temperature")
        search_tool.response.build_search_response.assert_called_once()
        search_tool.response.add_standard_metadata.assert_called_once()

        assert isinstance(result, SearchResponse)

    @pytest.mark.asyncio
    async def test_search_configuration_optimization(self, search_tool):
        """Test search configuration optimization based on query."""

        # Mock services
        search_tool._search_service.search = AsyncMock(return_value=[])
        search_tool.physics.enhance_query = AsyncMock(return_value=None)

        mock_response = SearchResponse(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="complex query",
            ai_response={},
            ai_prompt={},
        )
        search_tool.response.build_search_response = MagicMock(
            return_value=mock_response
        )
        search_tool.response.add_standard_metadata = MagicMock(
            return_value=mock_response
        )

        # Test with complex query that should trigger semantic search
        await search_tool.search_imas(
            query="plasma temperature profile equilibrium magnetic field"
        )

        # Verify search configuration service was used
        # (Implementation detail: service should optimize to semantic mode for complex queries)
        search_tool._search_service.search.assert_called_once()
        call_args = search_tool._search_service.search.call_args
        config = call_args[0][1]  # Second argument is config

        # Complex query should use semantic search
        assert config.search_mode == SearchMode.SEMANTIC

    @pytest.mark.asyncio
    async def test_response_building_with_results(self, search_tool):
        """Test response building when search returns results."""

        # Mock search results
        mock_result = MagicMock()
        mock_result.document.metadata.path_name = "core_profiles/temperature"
        mock_result.document.documentation = "Plasma temperature measurement"
        mock_result.score = 0.95

        search_tool._search_service.search = AsyncMock(return_value=[mock_result])
        search_tool.physics.enhance_query = AsyncMock(return_value=None)

        # Mock response service to capture arguments
        mock_response = SearchResponse(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="temperature",
            ai_response={},
            ai_prompt={},
        )
        search_tool.response.build_search_response = MagicMock(
            return_value=mock_response
        )
        search_tool.response.add_standard_metadata = MagicMock(
            return_value=mock_response
        )

        await search_tool.search_imas(query="temperature")

        # Verify response service received correct arguments
        build_call = search_tool.response.build_search_response.call_args
        assert build_call[1]["query"] == "temperature"
        assert len(build_call[1]["results"]) == 1
        assert "ai_prompt" in build_call[1]
        assert "ai_response" in build_call[1]
        # Physics is now always enabled, so no explicit enable_physics parameter

    @pytest.mark.asyncio
    async def test_no_results_guidance(self, search_tool):
        """Test guidance generation when no results found."""

        search_tool._search_service.search = AsyncMock(return_value=[])
        search_tool.physics.enhance_query = AsyncMock(return_value=None)

        mock_response = SearchResponse(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="nonexistent",
            ai_response={},
            ai_prompt={},
        )
        search_tool.response.build_search_response = MagicMock(
            return_value=mock_response
        )
        search_tool.response.add_standard_metadata = MagicMock(
            return_value=mock_response
        )

        await search_tool.search_imas(query="nonexistent")

        # Verify guidance was built for empty results
        build_call = search_tool.response.build_search_response.call_args
        assert "guidance" in build_call[1]["ai_prompt"]
        guidance = build_call[1]["ai_prompt"]["guidance"]
        assert "No results found" in guidance
        assert "alternatives" in guidance

    @pytest.mark.asyncio
    async def test_service_initialization(self, search_tool):
        """Test that all services are properly initialized."""
        # Verify all services are initialized
        assert hasattr(search_tool, "physics")
        assert hasattr(search_tool, "response")
        assert hasattr(search_tool, "documents")
        assert hasattr(search_tool, "search_config")

        # Verify service types
        from imas_mcp.services import (
            PhysicsService,
            ResponseService,
            DocumentService,
            SearchConfigurationService,
        )

        assert isinstance(search_tool.physics, PhysicsService)
        assert isinstance(search_tool.response, ResponseService)
        assert isinstance(search_tool.documents, DocumentService)
        assert isinstance(search_tool.search_config, SearchConfigurationService)

    @pytest.mark.asyncio
    async def test_search_config_service_integration(self, search_tool):
        """Test search configuration service creates proper config."""

        # Mock other services
        search_tool._search_service.search = AsyncMock(return_value=[])
        search_tool.physics.enhance_query = AsyncMock(return_value=None)
        mock_response = SearchResponse(
            hits=[],
            search_mode=SearchMode.LEXICAL,
            query="test",
            ai_response={},
            ai_prompt={},
        )
        search_tool.response.build_search_response = MagicMock(
            return_value=mock_response
        )
        search_tool.response.add_standard_metadata = MagicMock(
            return_value=mock_response
        )

        # Test boolean query that should use lexical search
        await search_tool.search_imas(
            query="temperature AND pressure", max_results=15, search_mode="auto"
        )

        # Verify config was created and optimized
        search_call = search_tool._search_service.search.call_args
        config = search_call[0][1]

        assert config.max_results == 15
        # Physics is now always enabled at the core level, no longer a config parameter
        # Boolean query should be optimized to lexical
        assert config.search_mode == SearchMode.LEXICAL
