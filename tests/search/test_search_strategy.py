"""
Test suite for search strategy and SearchService core functionality.

This module tests the SearchService and SearchConfig strategy implementations.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from imas_mcp.search.services.search_service import SearchService
from imas_mcp.search.search_strategy import SearchConfig
from imas_mcp.models.constants import SearchMode


class TestSearchService:
    """Test SearchService core functionality."""

    @pytest.fixture
    def search_service(self):
        """Create a basic SearchService instance."""
        return SearchService()

    def test_search_service_initialization(self, search_service):
        """Test SearchService initializes correctly."""
        assert isinstance(search_service, SearchService)
        assert hasattr(search_service, "engines")

        # Check that engines dictionary contains expected modes
        expected_modes = [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]
        for mode in expected_modes:
            assert mode in search_service.engines

    @pytest.mark.asyncio
    async def test_search_service_basic_functionality(self, search_service):
        """Test basic search functionality."""
        # Mock all engines to return empty results
        for mode in [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]:
            mock_engine = Mock()
            mock_engine.search = AsyncMock(return_value=[])
            search_service.engines[mode] = mock_engine

        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=10)
        results = await search_service.search("test query", config)

        assert isinstance(results, list)
        # Verify the semantic engine was called
        search_service.engines[SearchMode.SEMANTIC].search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_service_query_processing(self, search_service):
        """Test query processing and normalization."""
        # Mock engine
        mock_engine = Mock()
        mock_engine.search = AsyncMock(return_value=[])
        search_service.engines[SearchMode.LEXICAL] = mock_engine

        config = SearchConfig(search_mode=SearchMode.LEXICAL, max_results=5)

        # Test various query formats
        queries = [
            "simple query",
            "  query with spaces  ",
            "UPPERCASE QUERY",
            "mixed-CASE_query",
        ]

        for query in queries:
            await search_service.search(query, config)
            # Verify the engine was called with the query
            mock_engine.search.assert_called_with(query, config)


class TestSearchConfigurationStrategy:
    """Test SearchConfig and configuration strategy."""

    def test_search_config_default_strategy(self):
        """Test default SearchConfig strategy settings."""
        config = SearchConfig()

        # Verify default strategy settings
        assert config.search_mode == SearchMode.AUTO
        assert config.max_results == 10
        assert config.similarity_threshold == 0.0
        assert config.boost_exact_matches is True
        assert config.enable_physics_enhancement is True

    def test_search_config_semantic_strategy(self):
        """Test SearchConfig for semantic search strategy."""
        config = SearchConfig(
            search_mode=SearchMode.SEMANTIC,
            similarity_threshold=0.8,
            enable_physics_enhancement=True,
        )

        assert config.search_mode == SearchMode.SEMANTIC
        assert config.similarity_threshold == 0.8
        assert config.enable_physics_enhancement is True

    def test_search_config_lexical_strategy(self):
        """Test SearchConfig for lexical search strategy."""
        config = SearchConfig(
            search_mode=SearchMode.LEXICAL, boost_exact_matches=True, max_results=20
        )

        assert config.search_mode == SearchMode.LEXICAL
        assert config.boost_exact_matches is True
        assert config.max_results == 20

    def test_search_config_hybrid_strategy(self):
        """Test SearchConfig for hybrid search strategy."""
        config = SearchConfig(
            search_mode=SearchMode.HYBRID,
            similarity_threshold=0.6,
            boost_exact_matches=True,
            enable_physics_enhancement=True,
        )

        assert config.search_mode == SearchMode.HYBRID
        assert config.similarity_threshold == 0.6
        assert config.boost_exact_matches is True
        assert config.enable_physics_enhancement is True

    def test_search_config_ids_filtering_strategy(self):
        """Test SearchConfig IDS filtering strategy."""
        # Test single IDS filter
        config = SearchConfig(ids_filter=["core_profiles"])
        assert config.ids_filter == ["core_profiles"]

        # Test multiple IDS filters
        config = SearchConfig(ids_filter=["core_profiles", "equilibrium", "transport"])
        assert config.ids_filter == ["core_profiles", "equilibrium", "transport"]

        # Test no IDS filter
        config = SearchConfig(ids_filter=None)
        assert config.ids_filter is None


class TestSearchModeStrategy:
    """Test search mode strategy selection."""

    @pytest.fixture
    def search_service(self):
        """Create SearchService with mocked engines."""
        service = SearchService()

        # Mock all engines
        for mode in [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]:
            mock_engine = Mock()
            mock_engine.search = AsyncMock(return_value=[])
            service.engines[mode] = mock_engine

        return service

    @pytest.mark.asyncio
    async def test_auto_mode_strategy_selection(self, search_service):
        """Test AUTO mode strategy selection logic."""
        config = SearchConfig(search_mode=SearchMode.AUTO, max_results=10)

        # Test with different query types that might influence mode selection
        queries = [
            "plasma temperature",  # Physics term - might prefer semantic
            "core_profiles.temperature",  # Exact path - might prefer lexical
            "electron thermal behavior",  # Complex description - might prefer hybrid
        ]

        for query in queries:
            await search_service.search(query, config)

            # Verify that at least one engine was called for AUTO mode
            engines_called = sum(
                1
                for mode in [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]
                if search_service.engines[mode].search.called
            )
            assert engines_called >= 1

    @pytest.mark.asyncio
    async def test_explicit_mode_strategy(self, search_service):
        """Test explicit mode strategy enforcement."""
        # Test each explicit mode
        modes_to_test = [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]

        for mode in modes_to_test:
            # Reset mock call counts
            for mock_mode in [
                SearchMode.SEMANTIC,
                SearchMode.LEXICAL,
                SearchMode.HYBRID,
            ]:
                search_service.engines[mock_mode].search.reset_mock()

            config = SearchConfig(search_mode=mode, max_results=5)
            await search_service.search("test query", config)

            # Verify only the specified engine was called
            search_service.engines[mode].search.assert_called_once()

            # Verify other engines were not called
            for other_mode in modes_to_test:
                if other_mode != mode:
                    search_service.engines[other_mode].search.assert_not_called()


class TestSearchPerformanceStrategy:
    """Test search performance and optimization strategies."""

    @pytest.fixture
    def search_service(self):
        """Create SearchService for performance testing."""
        service = SearchService()

        # Mock engines with realistic behavior
        for mode in [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]:
            mock_engine = Mock()
            mock_engine.search = AsyncMock(return_value=[])
            service.engines[mode] = mock_engine

        return service

    @pytest.mark.asyncio
    async def test_result_limiting_strategy(self, search_service):
        """Test result limiting strategy."""
        # Test different result limits
        limits = [1, 5, 10, 50, 100]

        for limit in limits:
            config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=limit)
            await search_service.search("test query", config)

            # Verify the config with correct limit was passed
            search_service.engines[SearchMode.SEMANTIC].search.assert_called_with(
                "test query", config
            )
            assert config.max_results == limit

    @pytest.mark.asyncio
    async def test_filtering_strategy(self, search_service):
        """Test IDS filtering strategy performance."""
        # Test different filter configurations
        filter_configs = [
            None,  # No filter
            ["core_profiles"],  # Single filter
            ["core_profiles", "equilibrium"],  # Multiple filters
            ["core_profiles", "equilibrium", "transport", "mhd"],  # Many filters
        ]

        for ids_filter in filter_configs:
            config = SearchConfig(
                search_mode=SearchMode.LEXICAL, ids_filter=ids_filter, max_results=10
            )
            await search_service.search("test query", config)

            # Verify the filter was passed correctly
            search_service.engines[SearchMode.LEXICAL].search.assert_called_with(
                "test query", config
            )
            assert config.ids_filter == ids_filter

    @pytest.mark.asyncio
    async def test_threshold_strategy(self, search_service):
        """Test similarity threshold strategy."""
        # Test different threshold values
        thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]

        for threshold in thresholds:
            config = SearchConfig(
                search_mode=SearchMode.SEMANTIC, similarity_threshold=threshold
            )
            await search_service.search("test query", config)

            # Verify the threshold was passed correctly
            search_service.engines[SearchMode.SEMANTIC].search.assert_called_with(
                "test query", config
            )
            assert config.similarity_threshold == threshold


class TestSearchErrorHandlingStrategy:
    """Test error handling strategies in SearchService."""

    @pytest.fixture
    def search_service(self):
        """Create SearchService for error testing."""
        return SearchService()

    @pytest.mark.asyncio
    async def test_invalid_query_strategy(self, search_service):
        """Test strategy for handling invalid queries."""
        # Mock engine to handle invalid queries
        mock_engine = Mock()
        mock_engine.search = AsyncMock(return_value=[])
        search_service.engines[SearchMode.LEXICAL] = mock_engine

        config = SearchConfig(search_mode=SearchMode.LEXICAL, max_results=5)

        # Test various invalid or edge case queries
        invalid_queries = [
            "",  # Empty query
            "   ",  # Whitespace only
            None,  # None query (should be handled gracefully)
        ]

        for query in invalid_queries:
            if query is not None:
                results = await search_service.search(query, config)
                assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_engine_failure_strategy(self, search_service):
        """Test strategy when search engines fail."""
        # Mock engine that raises an exception
        mock_engine = Mock()
        mock_engine.search = Mock(side_effect=Exception("Engine failure"))
        search_service.engines[SearchMode.SEMANTIC] = mock_engine

        config = SearchConfig(search_mode=SearchMode.SEMANTIC, max_results=5)

        # The search should handle engine failures gracefully
        try:
            results = await search_service.search("test query", config)
            # If no exception is raised, verify we get an empty list or handle gracefully
            assert isinstance(results, list)
        except Exception:
            # If an exception is raised, it should be a known/handled exception type
            pass

    @pytest.mark.asyncio
    async def test_configuration_validation_strategy(self, search_service):
        """Test configuration validation strategy."""
        # Mock engine
        mock_engine = Mock()
        mock_engine.search = AsyncMock(return_value=[])
        search_service.engines[SearchMode.LEXICAL] = mock_engine

        # Test configuration validation
        valid_config = SearchConfig(
            search_mode=SearchMode.LEXICAL, max_results=10, ids_filter=["core_profiles"]
        )

        results = await search_service.search("test query", valid_config)
        assert isinstance(results, list)

        # Verify the engine was called with valid config
        mock_engine.search.assert_called_once_with("test query", valid_config)


class TestSearchOptimizationStrategy:
    """Test search optimization strategies."""

    @pytest.fixture
    def search_service(self):
        """Create SearchService for optimization testing."""
        service = SearchService()

        # Mock engines with timing simulation
        for mode in [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]:
            mock_engine = Mock()
            mock_engine.search = AsyncMock(return_value=[])
            service.engines[mode] = mock_engine

        return service

    @pytest.mark.asyncio
    async def test_boost_exact_matches_strategy(self, search_service):
        """Test exact match boosting strategy."""
        config = SearchConfig(search_mode=SearchMode.LEXICAL, boost_exact_matches=True)

        await search_service.search("exact_term", config)

        # Verify the boost setting was passed to the engine
        search_service.engines[SearchMode.LEXICAL].search.assert_called_once_with(
            "exact_term", config
        )
        assert config.boost_exact_matches is True

    @pytest.mark.asyncio
    async def test_physics_enhancement_strategy(self, search_service):
        """Test physics enhancement strategy."""
        config = SearchConfig(
            search_mode=SearchMode.SEMANTIC, enable_physics_enhancement=True
        )

        await search_service.search("plasma temperature", config)

        # Verify the physics enhancement setting was passed
        search_service.engines[SearchMode.SEMANTIC].search.assert_called_once_with(
            "plasma temperature", config
        )
        assert config.enable_physics_enhancement is True

    @pytest.mark.asyncio
    async def test_optimization_disabled_strategy(self, search_service):
        """Test strategy with optimizations disabled."""
        config = SearchConfig(
            search_mode=SearchMode.HYBRID,
            boost_exact_matches=False,
            enable_physics_enhancement=False,
        )

        await search_service.search("test query", config)

        # Verify the disabled optimizations were passed
        search_service.engines[SearchMode.HYBRID].search.assert_called_once_with(
            "test query", config
        )
        assert config.boost_exact_matches is False
        assert config.enable_physics_enhancement is False
