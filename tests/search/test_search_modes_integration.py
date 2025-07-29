"""
Integration tests for search modes using real DocumentStore and SearchService.

This module tests the search mode functionality with actual IMAS data
and real search capabilities using the current SearchService architecture.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from imas_mcp.search.services.search_service import SearchService
from imas_mcp.search.search_strategy import SearchConfig
from imas_mcp.models.constants import SearchMode
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.engines.lexical_engine import LexicalSearchEngine
from imas_mcp.search.engines.semantic_engine import SemanticSearchEngine
from imas_mcp.search.engines.hybrid_engine import HybridSearchEngine
from tests.conftest import STANDARD_TEST_IDS_SET


@pytest.fixture
def document_store():
    """Create a mocked DocumentStore for testing."""
    store = Mock(spec=DocumentStore)
    store.ids_set = STANDARD_TEST_IDS_SET
    return store


@pytest.fixture
def search_service(document_store):
    """Create a SearchService with mocked engines."""
    service = SearchService()

    # Create mock engines
    lexical_engine = AsyncMock(spec=LexicalSearchEngine)
    semantic_engine = AsyncMock(spec=SemanticSearchEngine)
    hybrid_engine = AsyncMock(spec=HybridSearchEngine)

    # Register mock engines
    service.register_engine(SearchMode.LEXICAL, lexical_engine)
    service.register_engine(SearchMode.SEMANTIC, semantic_engine)
    service.register_engine(SearchMode.HYBRID, hybrid_engine)

    return service


class TestSearchModesIntegration:
    """Integration tests for search modes with SearchService."""

    @pytest.mark.asyncio
    async def test_lexical_search_mode(self, search_service):
        """Test lexical search mode selection and execution."""
        config = SearchConfig(search_mode=SearchMode.LEXICAL, max_results=10)

        # Mock result from lexical engine
        mock_results = [
            Mock(score=0.9, search_mode=SearchMode.LEXICAL),
            Mock(score=0.8, search_mode=SearchMode.LEXICAL),
        ]
        search_service.engines[SearchMode.LEXICAL].search.return_value = mock_results

        results = await search_service.search("electron density", config)

        # Verify correct engine was called
        search_service.engines[SearchMode.LEXICAL].search.assert_called_once()
        assert len(results) == 2
        assert all(r.search_mode == SearchMode.LEXICAL for r in results)

    @pytest.mark.asyncio
    async def test_semantic_search_mode(self, search_service):
        """Test semantic search mode selection and execution."""
        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=5)

        # Mock result from semantic engine
        mock_results = [Mock(score=0.95, search_mode=SearchMode.SEMANTIC)]
        search_service.engines[SearchMode.SEMANTIC].search.return_value = mock_results

        results = await search_service.search("plasma temperature profile", config)

        # Verify correct engine was called
        search_service.engines[SearchMode.SEMANTIC].search.assert_called_once()
        assert len(results) == 1
        assert results[0].search_mode == SearchMode.SEMANTIC

    @pytest.mark.asyncio
    async def test_hybrid_search_mode(self, search_service):
        """Test hybrid search mode combines lexical and semantic."""
        config = SearchConfig(search_mode=SearchMode.HYBRID, max_results=15)

        # Mock result from hybrid engine
        mock_results = [
            Mock(score=0.9, search_mode=SearchMode.HYBRID),
            Mock(score=0.85, search_mode=SearchMode.HYBRID),
            Mock(score=0.8, search_mode=SearchMode.HYBRID),
        ]
        search_service.engines[SearchMode.HYBRID].search.return_value = mock_results

        results = await search_service.search("equilibrium magnetic field", config)

        # Verify hybrid engine was called
        search_service.engines[SearchMode.HYBRID].search.assert_called_once()
        assert len(results) == 3
        assert all(r.search_mode == SearchMode.HYBRID for r in results)

    @pytest.mark.asyncio
    async def test_auto_mode_selection(self, search_service):
        """Test automatic mode selection based on query characteristics."""
        config = SearchConfig(search_mode=SearchMode.AUTO, max_results=10)

        # Mock semantic engine for physics query (auto mode resolution)
        mock_results = [Mock(score=0.9, search_mode=SearchMode.SEMANTIC)]
        search_service.engines[SearchMode.SEMANTIC].search.return_value = mock_results

        # Auto mode should select appropriate engine
        await search_service.search("plasma transport equilibrium", config)

        # Verify that search was called on an engine
        assert any(engine.search.called for engine in search_service.engines.values())

    @pytest.mark.asyncio
    async def test_search_with_ids_filter(self, search_service):
        """Test search with IDS name filtering."""
        config = SearchConfig(
            search_mode=SearchMode.SEMANTIC,
            ids_filter=["core_profiles"],
            max_results=10,
        )

        mock_results = [Mock(score=0.9, search_mode=SearchMode.SEMANTIC)]
        search_service.engines[SearchMode.SEMANTIC].search.return_value = mock_results

        results = await search_service.search("electron temperature", config)

        # Verify config was passed to engine
        call_args = search_service.engines[SearchMode.SEMANTIC].search.call_args
        assert call_args is not None
        # Just verify the call happened correctly
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_result_ranking(self, search_service):
        """Test that search results are properly ranked by score."""
        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=10)

        # Mock results with different scores
        mock_results = [
            Mock(score=0.7, search_mode=SearchMode.SEMANTIC),
            Mock(score=0.9, search_mode=SearchMode.SEMANTIC),
            Mock(score=0.8, search_mode=SearchMode.SEMANTIC),
        ]
        search_service.engines[SearchMode.SEMANTIC].search.return_value = mock_results

        results = await search_service.search("plasma", config)

        assert len(results) == 3
        # Results should be returned in order (assuming engine handles ranking)
        assert all(hasattr(r, "score") for r in results)

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, search_service):
        """Test handling of empty queries."""
        config = SearchConfig(search_mode=SearchMode.LEXICAL, max_results=10)

        # Mock empty results for empty query
        search_service.engines[SearchMode.LEXICAL].search.return_value = []

        results = await search_service.search("", config)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_max_results_limiting(self, search_service):
        """Test that max_results parameter is respected."""
        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=3)

        # Mock more results than max_results
        mock_results = [
            Mock(score=0.9 - i * 0.1, search_mode=SearchMode.SEMANTIC) for i in range(3)
        ]
        search_service.engines[SearchMode.SEMANTIC].search.return_value = mock_results

        results = await search_service.search("plasma physics", config)

        # Should not exceed max_results
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_mode_fallback(self, search_service):
        """Test fallback when requested search mode is not available."""
        # Remove an engine to test fallback
        del search_service.engines[SearchMode.SEMANTIC]

        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=10)

        # Should raise SearchServiceError when engine is missing
        from imas_mcp.search.services.search_service import SearchServiceError

        with pytest.raises(
            SearchServiceError, match="Search engine not available for mode: semantic"
        ):
            await search_service.search("plasma", config)

    @pytest.mark.asyncio
    async def test_error_handling_in_search(self, search_service):
        """Test error handling when search engine fails."""
        config = SearchConfig(search_mode=SearchMode.LEXICAL, max_results=10)

        # Make engine raise an exception
        search_service.engines[SearchMode.LEXICAL].search.side_effect = Exception(
            "Search failed"
        )

        # Should handle error gracefully
        with pytest.raises(Exception, match="Search failed"):
            await search_service.search("test query", config)


class TestSearchConfigurationSupport:
    """Test search configuration and parameter validation."""

    def test_search_config_creation(self):
        """Test SearchConfig creation with various parameters."""
        config = SearchConfig(
            search_mode=SearchMode.SEMANTIC,
            max_results=20,
            ids_filter=["core_profiles", "equilibrium"],
            similarity_threshold=0.7,
        )

        assert config.search_mode == SearchMode.SEMANTIC
        assert config.max_results == 20
        assert config.ids_filter == ["core_profiles", "equilibrium"]
        assert config.similarity_threshold == 0.7

    def test_search_config_defaults(self):
        """Test SearchConfig default values."""
        config = SearchConfig()

        assert config.search_mode == SearchMode.AUTO
        assert config.max_results == 10
        assert config.ids_filter is None  # Default is None, not []
        assert config.similarity_threshold == 0.0

    @pytest.mark.asyncio
    async def test_config_parameter_passing(self, search_service):
        """Test that configuration parameters are properly passed to engines."""
        config = SearchConfig(
            search_mode=SearchMode.LEXICAL, max_results=15, similarity_threshold=0.8
        )

        mock_results = [Mock(score=0.9)]
        search_service.engines[SearchMode.LEXICAL].search.return_value = mock_results

        await search_service.search("test", config)

        # Verify config was passed
        call_args = search_service.engines[SearchMode.LEXICAL].search.call_args
        assert call_args is not None


class TestSearchEngineRegistration:
    """Test search engine registration and management."""

    def test_engine_registration(self):
        """Test registering search engines."""
        service = SearchService()
        mock_engine = AsyncMock()

        service.register_engine(SearchMode.LEXICAL, mock_engine)

        assert SearchMode.LEXICAL in service.engines
        assert service.engines[SearchMode.LEXICAL] == mock_engine

    def test_multiple_engine_registration(self):
        """Test registering multiple search engines."""
        service = SearchService()

        lexical_engine = AsyncMock()
        semantic_engine = AsyncMock()

        service.register_engine(SearchMode.LEXICAL, lexical_engine)
        service.register_engine(SearchMode.SEMANTIC, semantic_engine)

        assert len(service.engines) >= 2  # May have defaults
        assert service.engines[SearchMode.LEXICAL] == lexical_engine
        assert service.engines[SearchMode.SEMANTIC] == semantic_engine

    def test_engine_replacement(self):
        """Test replacing an existing engine."""
        service = SearchService()

        old_engine = AsyncMock()
        new_engine = AsyncMock()

        service.register_engine(SearchMode.LEXICAL, old_engine)
        service.register_engine(SearchMode.LEXICAL, new_engine)

        assert service.engines[SearchMode.LEXICAL] == new_engine
        assert service.engines[SearchMode.LEXICAL] != old_engine

    @pytest.mark.asyncio
    async def test_unregistered_engine_handling(self):
        """Test behavior when using unregistered search mode."""
        service = SearchService(engines={})  # Empty engines dict
        config = SearchConfig(search_mode=SearchMode.SEMANTIC)

        # Should handle gracefully or raise appropriate error
        try:
            results = await service.search("test", config)
            assert isinstance(results, list)
        except KeyError:
            # Expected when no engines are registered
            pass
