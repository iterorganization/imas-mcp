"""Integration tests for multi-tool service usage."""

from unittest.mock import AsyncMock, patch

import pytest

from imas_mcp.models.constants import DetailLevel, SearchMode
from imas_mcp.models.result_models import ConceptResult, SearchResult
from imas_mcp.tools.explain_tool import ExplainTool
from imas_mcp.tools.search_tool import SearchTool


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

        # Mock execute_search for both tools to control the flow
        mock_search_result = SearchResult(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="plasma temperature",
            ai_response={},
        )

        explain_tool.execute_search = AsyncMock(return_value=mock_search_result)
        search_tool.execute_search = AsyncMock(return_value=mock_search_result)

        # Execute both tools with same concept
        concept = "plasma temperature"

        explain_result = await explain_tool.explain_concept(concept=concept)
        search_result = await search_tool.search_imas(query=concept)

        # Verify both tools used execute_search (which handles physics enhancement internally)
        explain_tool.execute_search.assert_called_once()
        search_tool.execute_search.assert_called_once()

        # Verify results are proper types
        assert isinstance(explain_result, ConceptResult)
        assert explain_result.concept == concept
        assert isinstance(search_result, SearchResult)
        assert search_result.query == concept

    @pytest.mark.asyncio
    async def test_response_service_metadata_consistency(
        self, explain_tool, search_tool
    ):
        """Test response service metadata consistency."""

        # Create a SearchHit for the explain tool
        from imas_mcp.search.search_strategy import SearchHit

        mock_search_hit = SearchHit(
            path="test/path",
            documentation="Test documentation",
            score=0.8,
            rank=1,
            search_mode=SearchMode.SEMANTIC,
            highlights="",
            units=None,
            data_type="FLT_1D",
            physics_domain="test_domain",
            ids_name="test_ids",
        )

        # Mock execute_search for controlled responses with results for explain tool
        mock_search_result_explain = SearchResult(
            hits=[mock_search_hit],  # Provide hits so explain tool doesn't return early
            search_mode=SearchMode.SEMANTIC,
            query="test",
            ai_response={},
        )

        mock_search_result_search = SearchResult(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="test",
            ai_response={},
        )

        explain_tool.execute_search = AsyncMock(return_value=mock_search_result_explain)
        search_tool.execute_search = AsyncMock(return_value=mock_search_result_search)

        # Mock physics enhancement to track service usage
        explain_tool.physics.enhance_query = AsyncMock(return_value=None)
        search_tool.physics.enhance_query = AsyncMock(return_value=None)

        # Execute tools
        await explain_tool.explain_concept(concept="test")
        await search_tool.search_imas(query="test")

        # Verify services were used (via execute_search which calls response service)
        explain_tool.execute_search.assert_called_once()
        search_tool.execute_search.assert_called_once()

        # Note: Physics enhancement is handled through decorators, not direct method calls

    @pytest.mark.asyncio
    async def test_service_error_handling_consistency(self, explain_tool):
        """Test consistent error handling across services."""

        # Mock service to raise exception
        explain_tool._search_service.search = AsyncMock(
            side_effect=Exception("Search failed")
        )

        # Execute tool and verify error response
        result = await explain_tool.explain_concept(concept="test")

        # Should return ToolError, not raise exception
        from imas_mcp.models.error_models import ToolError

        assert isinstance(result, ToolError)
        assert "Search failed" in result.error

    @pytest.mark.asyncio
    async def test_service_composition_workflow(self, explain_tool):
        """Test complete service composition workflow."""

        # Create a SearchHit so the explain tool processes normally
        from imas_mcp.search.search_strategy import SearchHit

        mock_search_hit = SearchHit(
            path="test/path",
            documentation="Test documentation",
            score=0.8,
            rank=1,
            search_mode=SearchMode.SEMANTIC,
            highlights="",
            units=None,
            data_type="FLT_1D",
            physics_domain="test_domain",
            ids_name="test_ids",
        )

        # Mock execute_search to return controlled results with hits
        mock_search_result = SearchResult(
            hits=[mock_search_hit],  # Provide hits so tool doesn't return early
            search_mode=SearchMode.SEMANTIC,
            query="test concept",
            ai_response={},
        )

        explain_tool.execute_search = AsyncMock(return_value=mock_search_result)

        # Mock physics enhancement to track service usage
        explain_tool.physics.enhance_query = AsyncMock(return_value=None)

        # Execute tool
        result = await explain_tool.explain_concept(
            concept="test concept", detail_level=DetailLevel.INTERMEDIATE
        )

        # Verify service chain was executed
        explain_tool.execute_search.assert_called_once()

        # Note: Physics enhancement is handled through decorators, not direct method calls

        # Verify result structure
        assert isinstance(result, ConceptResult)
        assert result.concept == "test concept"
        assert result.detail_level == DetailLevel.INTERMEDIATE

    @pytest.mark.asyncio
    async def test_search_tool_service_integration(self, search_tool):
        """Test SearchTool service integration with proper SearchResult."""

        # Create a proper SearchResult for mocking
        from imas_mcp.search.search_strategy import SearchHit

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

        mock_search_response = SearchResult(
            hits=[mock_search_hit],
            search_mode=SearchMode.SEMANTIC,
            query="test query",
            ai_response={},
        )

        # Mock execute_search since that's what the tool actually calls
        search_tool.execute_search = AsyncMock(return_value=mock_search_response)

        # Execute search tool
        result = await search_tool.search_imas(query="test query")

        # Verify execute_search was called (which handles all the service orchestration)
        search_tool.execute_search.assert_called_once()

        # Verify the result is as expected
        assert isinstance(result, SearchResult)
        assert len(result.hits) == 1
        assert result.hits[0].path == "test/path"

        # Verify result is proper SearchResult type
        assert isinstance(result, SearchResult)
        assert result.query == "test query"
        assert len(result.hits) == 1
        assert result.hits[0].path == "test/path"
