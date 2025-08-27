"""Tests for SearchTool with service composition."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.result_models import SearchResult
from imas_mcp.search.search_strategy import SearchResponse
from imas_mcp.tools.search_tool import SearchTool


class TestSearchToolServices:
    """Test SearchTool service composition functionality."""

    @pytest.fixture
    def search_tool(self):
        with patch("imas_mcp.tools.base.DocumentStore"):
            return SearchTool()

    @pytest.mark.asyncio
    async def test_search_with_physics_enhancement(self, search_tool):
        """Test search with physics enhancement through service."""

        # Mock execute_search to return controlled response
        mock_response = SearchResult(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="plasma temperature",
            ai_response={},
            ai_prompt={},
        )
        search_tool.execute_search = AsyncMock(return_value=mock_response)

        # Mock context for physics enhancement
        mock_ctx = MagicMock()

        result = await search_tool.search_imas(query="plasma temperature", ctx=mock_ctx)

        # Verify execute_search was called (which handles physics enhancement internally)
        search_tool.execute_search.assert_called_once()

        # Verify the result structure
        assert isinstance(result, SearchResult)
        assert result.query == "plasma temperature"

        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_configuration_optimization(self, search_tool):
        """Test search configuration optimization based on query."""

        # Mock services
        search_tool._search_service.search = AsyncMock(
            return_value=SearchResponse(hits=[], all_paths=[])
        )
        search_tool.physics.enhance_query = AsyncMock(return_value=None)

        mock_response = SearchResult(
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

        # Import needed classes for proper mock objects
        from imas_mcp.models.constants import SearchMode
        from imas_mcp.search.document_store import Document, DocumentMetadata
        from imas_mcp.search.search_strategy import SearchMatch

        # Create proper SearchMatch object
        mock_metadata = DocumentMetadata(
            path_name="core_profiles/temperature",
            path_id="test_path_1",
            ids_name="core_profiles",
            data_type="temperature",
            physics_domain="plasma_core",
        )
        mock_document = Document(
            metadata=mock_metadata,
            documentation="Plasma temperature measurement",
            units=None,
        )
        mock_result = SearchMatch(
            document=mock_document,
            score=0.95,
            rank=0,
            search_mode=SearchMode.SEMANTIC,
            highlights="temperature",
        )

        search_tool._search_service.search = AsyncMock(
            return_value=SearchResponse(
                hits=[mock_result], all_paths=["core_profiles/temperature"]
            )
        )
        search_tool.physics.enhance_query = AsyncMock(return_value=None)

        # Mock response service to capture arguments
        mock_response = SearchResult(
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

        # Verify response service received correct arguments from SearchResponse.hits
        build_call = search_tool.response.build_search_response.call_args
        assert build_call[1]["query"] == "temperature"
        assert len(build_call[1]["results"]) == 1  # Should get hits from SearchResponse
        # Check for expected fields in the response builder call
        assert "search_mode" in build_call[1]
        assert "ids_filter" in build_call[1]
        assert "max_results" in build_call[1]

    @pytest.mark.asyncio
    async def test_no_results_guidance(self, search_tool):
        """Test guidance generation when no results found."""

        # Mock execute_search to return empty results
        mock_response = SearchResult(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="nonexistent",
            ai_response={},
            ai_prompt={},
        )
        search_tool.execute_search = AsyncMock(return_value=mock_response)

        result = await search_tool.search_imas(query="nonexistent")

        # Verify empty results were handled gracefully
        search_tool.execute_search.assert_called_once()
        assert isinstance(result, SearchResult)
        assert len(result.hits) == 0
        assert result.query == "nonexistent"

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
            DocumentService,
            PhysicsService,
            ResponseService,
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
        search_tool._search_service.search = AsyncMock(
            return_value=SearchResponse(hits=[], all_paths=[])
        )
        search_tool.physics.enhance_query = AsyncMock(return_value=None)
        mock_response = SearchResult(
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
