"""Integration tests for multi-tool service usage."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from imas_mcp.tools.explain_tool import ExplainTool
from imas_mcp.tools.search_tool import SearchTool
from imas_mcp.models.constants import DetailLevel, SearchMode
from imas_mcp.models.response_models import ConceptResult, SearchResponse


class TestServiceIntegration:
    """Test physics service consistency across tools."""

    @pytest.fixture
    def explain_tool(self):
        with patch("imas_mcp.tools.base.DocumentStore"):
            return ExplainTool()

    @pytest.fixture
    def search_tool(self):
        with patch("imas_mcp.tools.base.DocumentStore"):
            return SearchTool()

    @pytest.mark.asyncio
    async def test_physics_service_consistency(self, explain_tool, search_tool):
        """Test physics service consistency across tools."""

        # Mock physics service to return consistent results
        from imas_mcp.models.physics_models import PhysicsSearchResult

        mock_physics_result = PhysicsSearchResult(
            query="plasma temperature",
            physics_matches=[],
            concept_suggestions=[],
            unit_suggestions=[],
            symbol_suggestions=[],
            imas_path_suggestions=[],
        )

        explain_tool.physics.enhance_query = AsyncMock(return_value=mock_physics_result)
        search_tool.physics.enhance_query = AsyncMock(return_value=mock_physics_result)

        # Mock search services
        mock_search_result = MagicMock()
        mock_search_result.document.metadata.path_name = "core_profiles/temperature"
        mock_search_result.document.documentation = "Plasma temperature"
        mock_search_result.document.metadata.physics_domain = "core_plasma"
        mock_search_result.document.metadata.ids_name = "core_profiles"
        mock_search_result.document.metadata.data_type = "FLT_1D"
        mock_search_result.document.units = None
        mock_search_result.score = 0.95
        mock_search_result.to_hit = MagicMock(return_value=MagicMock())

        explain_tool._search_service.search = AsyncMock(
            return_value=[mock_search_result]
        )
        search_tool._search_service.search = AsyncMock(
            return_value=[mock_search_result]
        )

        # Mock response services
        explain_tool.response.add_standard_metadata = MagicMock(
            side_effect=lambda x, _: x
        )
        search_tool.response.build_search_response = MagicMock(return_value=MagicMock())
        search_tool.response.add_standard_metadata = MagicMock(
            side_effect=lambda x, _: x
        )

        # Mock service application
        explain_tool.apply_services = AsyncMock(
            side_effect=lambda result, **kwargs: result
        )
        search_tool.apply_services = AsyncMock(
            side_effect=lambda result, **kwargs: result
        )

        # Execute both tools with same concept
        concept = "plasma temperature"

        explain_result = await explain_tool.explain_concept(concept=concept)

        # Verify both tools used physics service consistently
        explain_tool.physics.enhance_query.assert_called_once_with(concept)

        # Verify results are proper types
        assert isinstance(explain_result, ConceptResult)
        assert explain_result.concept == concept

    @pytest.mark.asyncio
    async def test_response_service_metadata_consistency(
        self, explain_tool, search_tool
    ):
        """Test response service metadata consistency."""

        # Mock services to capture metadata addition
        mock_metadata_calls = []

        def capture_metadata(response, tool_name):
            mock_metadata_calls.append(tool_name)
            return response

        explain_tool.response.add_standard_metadata = MagicMock(
            side_effect=capture_metadata
        )
        search_tool.response.add_standard_metadata = MagicMock(
            side_effect=capture_metadata
        )

        # Mock search result for explain tool to get proper path
        mock_search_result = MagicMock()
        mock_search_result.document.metadata.path_name = "test/path"
        mock_search_result.document.metadata.physics_domain = "test_domain"
        mock_search_result.document.metadata.ids_name = "test_ids"
        mock_search_result.document.metadata.data_type = "FLT_1D"
        mock_search_result.document.documentation = "Test documentation"
        mock_search_result.document.units = None
        mock_search_result.score = 0.8

        # Mock other dependencies - provide search results for explain tool
        explain_tool._search_service.search = AsyncMock(
            return_value=[mock_search_result]
        )
        explain_tool.physics.enhance_query = AsyncMock(return_value=None)
        explain_tool.apply_services = AsyncMock(
            side_effect=lambda result, **kwargs: result
        )

        search_tool._search_service.search = AsyncMock(return_value=[])
        search_tool.physics.enhance_query = AsyncMock(return_value=None)
        search_tool.response.build_search_response = MagicMock(return_value=MagicMock())
        search_tool.apply_services = AsyncMock(
            side_effect=lambda result, **kwargs: result
        )

        # Execute tools
        await explain_tool.explain_concept(concept="test")
        await search_tool.search_imas(query="test")

        # Verify metadata service was called with correct tool names
        assert "explain_concept" in mock_metadata_calls
        assert "search_imas" in mock_metadata_calls

    @pytest.mark.asyncio
    async def test_service_error_handling_consistency(self, explain_tool):
        """Test consistent error handling across services."""

        # Mock service to raise exception
        explain_tool._search_service.search = AsyncMock(
            side_effect=Exception("Search failed")
        )

        # Execute tool and verify error response
        result = await explain_tool.explain_concept(concept="test")

        # Should return ErrorResponse, not raise exception
        from imas_mcp.models.response_models import ErrorResponse

        assert isinstance(result, ErrorResponse)
        assert "Search failed" in result.error

    @pytest.mark.asyncio
    async def test_service_composition_workflow(self, explain_tool):
        """Test complete service composition workflow."""

        # Mock all services in the workflow
        mock_search_result = MagicMock()
        mock_search_result.document.metadata.path_name = "test/path"
        mock_search_result.document.metadata.physics_domain = "test_domain"
        mock_search_result.document.metadata.ids_name = "test_ids"
        mock_search_result.document.metadata.data_type = "FLT_1D"
        mock_search_result.document.documentation = "Test documentation"
        mock_search_result.document.units = None
        mock_search_result.score = 0.8

        # Mock service chain - return proper types
        from imas_mcp.models.physics_models import PhysicsSearchResult

        mock_physics_result = PhysicsSearchResult(
            query="test concept",
            physics_matches=[],
            concept_suggestions=[],
            unit_suggestions=[],
            symbol_suggestions=[],
            imas_path_suggestions=[],
        )

        explain_tool.search_config.create_config = MagicMock(return_value=MagicMock())
        explain_tool.search_config.optimize_for_query = MagicMock(
            side_effect=lambda query, config: config
        )
        explain_tool._search_service.search = AsyncMock(
            return_value=[mock_search_result]
        )
        explain_tool.physics.enhance_query = AsyncMock(return_value=mock_physics_result)
        explain_tool.response.add_standard_metadata = MagicMock(
            side_effect=lambda x, _: x
        )
        explain_tool.apply_services = AsyncMock(
            side_effect=lambda result, **kwargs: result
        )

        # Execute tool
        result = await explain_tool.explain_concept(
            concept="test concept", detail_level=DetailLevel.INTERMEDIATE
        )

        # Verify service chain was executed
        explain_tool.search_config.create_config.assert_called_once()
        explain_tool.search_config.optimize_for_query.assert_called_once()
        explain_tool._search_service.search.assert_called_once()
        explain_tool.physics.enhance_query.assert_called_once()
        explain_tool.response.add_standard_metadata.assert_called_once()
        explain_tool.apply_services.assert_called_once()

        # Verify result structure
        assert isinstance(result, ConceptResult)
        assert result.concept == "test concept"
        assert result.detail_level == DetailLevel.INTERMEDIATE

    @pytest.mark.asyncio
    async def test_search_tool_service_integration(self, search_tool):
        """Test SearchTool service integration with proper SearchResponse."""

        # Mock search service to return results
        from imas_mcp.search.search_strategy import SearchHit

        # Create a proper SearchHit for the response
        mock_search_hit = SearchHit(
            path="test/path",
            documentation="Test documentation",
            score=0.9,
            rank=1,
            search_mode=SearchMode.SEMANTIC,
            highlights="",
            units=None,
            data_type="FLT_1D",
            physics_domain="test_domain",
            ids_name="test_ids",
        )

        mock_search_result = MagicMock()
        mock_search_result.to_hit = MagicMock(return_value=mock_search_hit)

        search_tool._search_service.search = AsyncMock(
            return_value=[mock_search_result]
        )
        search_tool.physics.enhance_query = AsyncMock(return_value=None)

        # Mock response service to return proper SearchResponse
        mock_search_response = SearchResponse(
            hits=[mock_search_hit],
            search_mode=SearchMode.SEMANTIC,
            query="test query",
            ai_insights={},
        )
        search_tool.response.build_search_response = MagicMock(
            return_value=mock_search_response
        )
        search_tool.response.add_standard_metadata = MagicMock(
            side_effect=lambda x, _: x
        )
        search_tool.apply_services = AsyncMock(
            side_effect=lambda result, **kwargs: result
        )

        # Execute search tool
        result = await search_tool.search_imas(query="test query")

        # Verify services were called in proper order
        search_tool._search_service.search.assert_called_once()
        search_tool.response.build_search_response.assert_called_once()
        search_tool.response.add_standard_metadata.assert_called_once()
        search_tool.apply_services.assert_called_once()

        # Verify result is proper SearchResponse type
        assert isinstance(result, SearchResponse)
        assert result.query == "test query"
        assert len(result.hits) == 1
        assert result.hits[0].path == "test/path"
