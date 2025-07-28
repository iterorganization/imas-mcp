"""
Tests for SearchService orchestration and functionality.

Tests the service layer that coordinates search engines and handles requests.
"""

import pytest
from unittest.mock import AsyncMock, Mock

from imas_mcp.search.services.search_service import (
    SearchService,
    SearchServiceError,
    SearchRequest,
    SearchResponse,
)
from imas_mcp.search.engines.base_engine import MockSearchEngine
from imas_mcp.search.search_modes import SearchConfig, SearchResult
from imas_mcp.models.enums import SearchMode


class TestSearchService:
    """Test SearchService orchestration functionality."""

    def test_search_service_default_instantiation(self):
        """Test SearchService can be instantiated with default engines."""
        service = SearchService()
        assert service is not None
        assert len(service.engines) > 0

    def test_search_service_custom_engines(self):
        """Test SearchService with custom engine configuration."""
        mock_engine = MockSearchEngine()
        engines = {SearchMode.SEMANTIC: mock_engine}

        service = SearchService(engines)  # type: ignore
        assert service.engines == engines

    def test_get_available_modes(self):
        """Test getting available search modes."""
        service = SearchService()
        modes = service.get_available_modes()

        assert isinstance(modes, list)
        assert len(modes) > 0
        assert SearchMode.AUTO not in modes  # AUTO should be filtered out

    def test_register_engine(self):
        """Test registering a new engine."""
        service = SearchService()
        mock_engine = MockSearchEngine()

        service.register_engine(SearchMode.LEXICAL, mock_engine)
        assert SearchMode.LEXICAL in service.engines
        assert service.engines[SearchMode.LEXICAL] == mock_engine

    def test_health_check(self):
        """Test health check functionality."""
        service = SearchService()
        health_status = service.health_check()

        assert isinstance(health_status, dict)
        assert len(health_status) > 0

        # All engines should be healthy by default
        for status in health_status.values():
            assert status is True

    def test_health_check_with_broken_engine(self):
        """Test health check with a broken engine."""
        broken_engine = Mock()
        broken_engine.get_engine_type.side_effect = Exception("Engine broken")

        engines = {SearchMode.SEMANTIC: broken_engine}
        service = SearchService(engines)  # type: ignore

        health_status = service.health_check()
        assert health_status["semantic"] is False

    @pytest.mark.asyncio
    async def test_search_with_semantic_mode(self):
        """Test search execution with semantic mode."""
        service = SearchService()
        config = SearchConfig(mode=SearchMode.SEMANTIC, max_results=1)

        results = await service.search("temperature", config)

        assert isinstance(results, list)
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_search_mode_resolution_auto(self):
        """Test AUTO mode resolution to specific modes."""
        service = SearchService()

        # Test path-like query (should resolve to LEXICAL)
        config = SearchConfig(mode=SearchMode.AUTO)
        results = await service.search("core_profiles/temperature", config)
        assert isinstance(results, list)

        # Test conceptual query (should resolve to SEMANTIC)
        results = await service.search("what is plasma temperature", config)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_unavailable_engine(self):
        """Test search fails gracefully when engine is unavailable."""
        # Create service with limited engines
        mock_engine = MockSearchEngine()
        engines = {SearchMode.SEMANTIC: mock_engine}
        service = SearchService(engines)  # type: ignore

        config = SearchConfig(mode=SearchMode.LEXICAL)  # Not available

        with pytest.raises(SearchServiceError):
            await service.search("test", config)

    @pytest.mark.asyncio
    async def test_search_post_processing(self):
        """Test search result post-processing."""
        service = SearchService()
        config = SearchConfig(
            mode=SearchMode.SEMANTIC, max_results=1, similarity_threshold=0.5
        )

        results = await service.search("temperature", config)

        # Results should be limited and ranked
        assert len(results) <= 1
        for i, result in enumerate(results):
            assert result.rank == i

    @pytest.mark.asyncio
    async def test_search_engine_failure_handling(self):
        """Test handling when search engine fails."""
        broken_engine = AsyncMock()
        broken_engine.search.side_effect = Exception("Engine error")

        engines = {SearchMode.SEMANTIC: broken_engine}
        service = SearchService(engines)  # type: ignore

        config = SearchConfig(mode=SearchMode.SEMANTIC)

        with pytest.raises(SearchServiceError):
            await service.search("test", config)


class TestSearchRequest:
    """Test SearchRequest structured request object."""

    def test_search_request_creation(self):
        """Test basic SearchRequest creation."""
        request = SearchRequest("test query")

        assert request.query == "test query"
        assert request.config.mode == SearchMode.AUTO
        assert request.config.max_results == 10

    def test_search_request_with_parameters(self):
        """Test SearchRequest with custom parameters."""
        request = SearchRequest(
            query=["multi", "term"],
            mode=SearchMode.SEMANTIC,
            max_results=5,
            ids_filter=["core_profiles"],
            similarity_threshold=0.7,
        )

        assert request.query == ["multi", "term"]
        assert request.config.mode == SearchMode.SEMANTIC
        assert request.config.max_results == 5
        assert request.config.filter_ids == ["core_profiles"]
        assert request.config.similarity_threshold == 0.7


class TestSearchResponse:
    """Test SearchResponse structured response object."""

    def test_search_response_creation(self):
        """Test SearchResponse creation with results."""
        # Create mock results
        mock_results = [Mock(spec=SearchResult)]
        request = SearchRequest("test")

        response = SearchResponse(mock_results, request)  # type: ignore

        assert response.results == mock_results
        assert response.results_count == 1
        assert response.request == request

    def test_search_response_to_dict(self):
        """Test SearchResponse conversion to dictionary."""
        # Create mock result with to_dict method
        mock_result = Mock()
        mock_result.to_dict.return_value = {"path": "test/path", "score": 0.9}

        request = SearchRequest("test", max_results=5)
        response = SearchResponse([mock_result], request)

        result_dict = response.to_dict()

        assert "results" in result_dict
        assert "results_count" in result_dict
        assert "search_mode" in result_dict
        assert "max_results" in result_dict
        assert result_dict["results_count"] == 1
        assert result_dict["max_results"] == 5


class TestSearchServiceError:
    """Test SearchServiceError exception."""

    def test_search_service_error_creation(self):
        """Test SearchServiceError creation."""
        error = SearchServiceError("Test error message")

        assert error.query == ""
        assert "Test error message" in str(error)

    def test_search_service_error_with_query(self):
        """Test SearchServiceError with query."""
        error = SearchServiceError("Test error", "test query")

        assert error.query == "test query"

    def test_search_service_error_inheritance(self):
        """Test SearchServiceError inherits from Exception."""
        error = SearchServiceError("test")
        assert isinstance(error, Exception)
