"""
Test suite for search modes and SearchService functionality.

This module tests the different search modes using SearchService.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from imas_mcp.search.services.search_service import SearchService
from imas_mcp.search.search_strategy import SearchConfig
from imas_mcp.models.constants import SearchMode


class TestSearchModes:
    """Test search mode enumeration and validation."""

    def test_search_mode_values(self):
        """Test SearchMode enum values."""
        assert SearchMode.SEMANTIC.value == "semantic"
        assert SearchMode.LEXICAL.value == "lexical"
        assert SearchMode.HYBRID.value == "hybrid"
        assert SearchMode.AUTO.value == "auto"

    def test_search_mode_from_string(self):
        """Test SearchMode creation from string."""
        assert SearchMode("semantic") == SearchMode.SEMANTIC
        assert SearchMode("lexical") == SearchMode.LEXICAL
        assert SearchMode("hybrid") == SearchMode.HYBRID
        assert SearchMode("auto") == SearchMode.AUTO


class TestSearchConfig:
    """Test SearchConfig model and validation."""

    def test_search_config_defaults(self):
        """Test SearchConfig default values."""
        config = SearchConfig()
        assert config.search_mode == SearchMode.AUTO
        assert config.max_results == 10
        assert config.ids_filter is None
        assert config.similarity_threshold == 0.0
        assert config.boost_exact_matches is True
        assert config.enable_physics_enhancement is True

    def test_search_config_custom_values(self):
        """Test SearchConfig with custom values."""
        config = SearchConfig(
            search_mode=SearchMode.SEMANTIC,
            max_results=20,
            ids_filter=["core_profiles", "equilibrium"],
            similarity_threshold=0.5,
            boost_exact_matches=False,
            enable_physics_enhancement=False,
        )
        assert config.search_mode == SearchMode.SEMANTIC
        assert config.max_results == 20
        assert config.ids_filter == ["core_profiles", "equilibrium"]
        assert config.similarity_threshold == 0.5
        assert config.boost_exact_matches is False
        assert config.enable_physics_enhancement is False

    def test_search_config_enum_validation(self):
        """Test SearchConfig field validation with proper enum types."""
        # Test search_mode validation with enum
        config = SearchConfig(search_mode=SearchMode.SEMANTIC)
        assert config.search_mode == SearchMode.SEMANTIC

        # Test ids_filter validation with list
        config = SearchConfig(ids_filter=["core_profiles"])
        assert config.ids_filter == ["core_profiles"]

        config = SearchConfig(ids_filter=["core_profiles", "equilibrium"])
        assert config.ids_filter == ["core_profiles", "equilibrium"]


class TestSearchServiceModeSelection:
    """Test SearchService mode selection functionality."""

    @pytest.fixture
    def search_service(self):
        """Create SearchService with mock engines."""
        service = SearchService()

        # Mock all engines to return empty results by default
        for mode in [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]:
            mock_engine = Mock()
            mock_engine.search = AsyncMock(return_value=[])
            service.engines[mode] = mock_engine

        return service

    @pytest.mark.asyncio
    async def test_search_service_semantic_mode(self, search_service):
        """Test SearchService with semantic mode."""
        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=5)

        results = await search_service.search("temperature", config)
        assert isinstance(results, list)

        # Verify the semantic engine was called
        search_service.engines[SearchMode.SEMANTIC].search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_service_lexical_mode(self, search_service):
        """Test SearchService with lexical mode."""
        config = SearchConfig(search_mode=SearchMode.LEXICAL, max_results=5)

        results = await search_service.search("temperature", config)
        assert isinstance(results, list)

        # Verify the lexical engine was called
        search_service.engines[SearchMode.LEXICAL].search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_service_hybrid_mode(self, search_service):
        """Test SearchService with hybrid mode."""
        config = SearchConfig(search_mode=SearchMode.HYBRID, max_results=5)

        results = await search_service.search("temperature", config)
        assert isinstance(results, list)

        # Verify the hybrid engine was called
        search_service.engines[SearchMode.HYBRID].search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_service_auto_mode(self, search_service):
        """Test SearchService with auto mode."""
        config = SearchConfig(search_mode=SearchMode.AUTO, max_results=5)

        results = await search_service.search("temperature", config)
        assert isinstance(results, list)

        # In AUTO mode, one of the engines should be called
        engines_called = sum(
            1
            for mode in [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]
            if search_service.engines[mode].search.called
        )
        assert engines_called >= 1


class TestSearchServiceConfiguration:
    """Test SearchService configuration handling."""

    @pytest.fixture
    def search_service(self):
        """Create SearchService for configuration testing."""
        service = SearchService()

        # Mock engines with specific return values
        for mode in [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]:
            mock_engine = Mock()
            mock_engine.search = AsyncMock(return_value=[])
            service.engines[mode] = mock_engine

        return service

    @pytest.mark.asyncio
    async def test_max_results_parameter(self, search_service):
        """Test that max_results parameter is passed to engines."""
        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=50)

        await search_service.search("test query", config)

        # Verify the config was passed to the engine
        search_service.engines[SearchMode.SEMANTIC].search.assert_called_once_with(
            "test query", config
        )

    @pytest.mark.asyncio
    async def test_ids_filter_parameter(self, search_service):
        """Test that ids_filter parameter is passed to engines."""
        config = SearchConfig(
            search_mode=SearchMode.LEXICAL, ids_filter=["core_profiles", "equilibrium"]
        )

        await search_service.search("test query", config)

        # Verify the config with filter was passed to the engine
        search_service.engines[SearchMode.LEXICAL].search.assert_called_once_with(
            "test query", config
        )

    @pytest.mark.asyncio
    async def test_similarity_threshold_parameter(self, search_service):
        """Test that similarity_threshold parameter is passed to engines."""
        config = SearchConfig(
            search_mode=SearchMode.SEMANTIC, similarity_threshold=0.75
        )

        await search_service.search("test query", config)

        # Verify the config was passed to the engine
        search_service.engines[SearchMode.SEMANTIC].search.assert_called_once_with(
            "test query", config
        )


class TestSearchServiceErrorHandling:
    """Test SearchService error handling."""

    @pytest.fixture
    def search_service(self):
        """Create SearchService for error testing."""
        return SearchService()

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, search_service):
        """Test handling of empty queries."""
        config = SearchConfig(search_mode=SearchMode.LEXICAL, max_results=5)

        # Mock engine to return empty results for empty query
        mock_engine = Mock()
        mock_engine.search = AsyncMock(return_value=[])
        search_service.engines[SearchMode.LEXICAL] = mock_engine

        results = await search_service.search("", config)
        assert isinstance(results, list)
        assert len(results) == 0

        results = await search_service.search("   ", config)
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_whitespace_query_handling(self, search_service):
        """Test handling of whitespace-only queries."""
        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=5)

        # Mock engine to return empty results
        mock_engine = Mock()
        mock_engine.search = AsyncMock(return_value=[])
        search_service.engines[SearchMode.SEMANTIC] = mock_engine

        whitespace_queries = ["", "   ", "\n", "\t", "  \n\t  "]

        for query in whitespace_queries:
            results = await search_service.search(query, config)
            assert isinstance(results, list)
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, search_service):
        """Test handling of special characters in queries."""
        config = SearchConfig(search_mode=SearchMode.LEXICAL, max_results=5)

        # Mock engine to handle special characters
        mock_engine = Mock()
        mock_engine.search = AsyncMock(return_value=[])
        search_service.engines[SearchMode.LEXICAL] = mock_engine

        special_queries = [
            "test@query.com",
            "test/path*",
            "query-with-dashes",
            "query_with_underscores",
            "query.with.dots",
            "query[with]brackets",
            "query{with}braces",
            "query(with)parentheses",
        ]

        for query in special_queries:
            results = await search_service.search(query, config)
            assert isinstance(results, list)
            # Verify the engine was called with the special query
            # Note: We can't easily verify this with our current mock setup


class TestSearchServiceEdgeCases:
    """Test SearchService edge cases and boundary conditions."""

    @pytest.fixture
    def search_service(self):
        """Create SearchService for edge case testing."""
        return SearchService()

    @pytest.mark.asyncio
    async def test_very_long_query(self, search_service):
        """Test handling of very long queries."""
        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=5)

        # Mock engine
        mock_engine = Mock()
        mock_engine.search = AsyncMock(return_value=[])
        search_service.engines[SearchMode.SEMANTIC] = mock_engine

        # Create a very long query
        long_query = "temperature " * 1000

        results = await search_service.search(long_query, config)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_unicode_query(self, search_service):
        """Test handling of unicode characters in queries."""
        config = SearchConfig(search_mode=SearchMode.LEXICAL, max_results=5)

        # Mock engine
        mock_engine = Mock()
        mock_engine.search = AsyncMock(return_value=[])
        search_service.engines[SearchMode.LEXICAL] = mock_engine

        unicode_queries = [
            "température",  # French
            "温度",  # Chinese
            "температура",  # Russian
            "θερμοκρασία",  # Greek
            "🔥 temperature 🌡️",  # Emojis
        ]

        for query in unicode_queries:
            results = await search_service.search(query, config)
            assert isinstance(results, list)
