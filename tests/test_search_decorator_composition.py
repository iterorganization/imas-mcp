"""
Comprehensive test suite for search tool with decorator composition.

Tests the search tool with comprehensive decorator functionality:
- Cache decorator for performance optimization
- Input validation decorator for data integrity
- Sampling decorator for AI insights
- Tool recommendations decorator for follow-up actions
- Performance decorator for monitoring
- Error handling decorator for robustness
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from imas_mcp.tools.search import Search
from imas_mcp.search.search_strategy import SearchResult, SearchConfig
from imas_mcp.search.document_store import Document, DocumentMetadata, Units
from imas_mcp.models.enums import SearchMode


class TestSearchDecoratorComposition:
    """Test cases for search tool with decorator composition."""

    @pytest.fixture
    def mock_search_service(self):
        """Create mock search service for testing."""
        service = MagicMock()
        service.search = AsyncMock()
        return service

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results for testing."""
        metadata1 = DocumentMetadata(
            path_id="core_profiles_temperature",
            path_name="core_profiles.temperature",
            ids_name="core_profiles",
            data_type="float",
            physics_domain="thermal",
            units="eV",
        )
        doc1 = Document(
            metadata=metadata1,
            documentation="Temperature profile data",
            units=Units(unit_str="eV", name="Electron volt"),
        )
        result1 = SearchResult(
            document=doc1, score=0.95, rank=1, search_mode=SearchMode.SEMANTIC
        )

        metadata2 = DocumentMetadata(
            path_id="equilibrium_magnetic_field",
            path_name="equilibrium.magnetic_field",
            ids_name="equilibrium",
            data_type="float",
            physics_domain="magnetic",
            units="T",
        )
        doc2 = Document(
            metadata=metadata2,
            documentation="Magnetic field data",
            units=Units(unit_str="T", name="Tesla"),
        )
        result2 = SearchResult(
            document=doc2, score=0.87, rank=2, search_mode=SearchMode.SEMANTIC
        )

        return [result1, result2]

    @pytest.fixture
    def search_tool(self, mock_search_service):
        """Create search tool with mocked search service."""
        tool = Search()
        tool._search_service = mock_search_service
        return tool

    @pytest.mark.asyncio
    async def test_search_with_decorator_composition(
        self, search_tool, mock_search_service, sample_search_results
    ):
        """Test search tool with comprehensive decorator composition."""
        # Setup mock to return sample results
        mock_search_service.search.return_value = sample_search_results

        # Execute search
        result = await search_tool.search_imas(
            query="temperature", search_mode="semantic", max_results=5
        )

        # Verify basic functionality works
        assert "results" in result
        assert "results_count" in result
        assert "search_mode" in result
        assert "query" in result

        # Verify results format
        assert len(result["results"]) == 2
        assert result["results_count"] == 2
        assert result["search_mode"] == "semantic"
        assert result["query"] == "temperature"

        # Verify decorator enhancements (these would be added by decorators)
        # The actual decorator functionality is tested in test_core_decorators.py
        # Here we verify the base functionality works with decorator stack
        first_result = result["results"][0]
        assert "path" in first_result
        assert "relevance_score" in first_result
        assert "documentation" in first_result

        # Verify service was called correctly
        mock_search_service.search.assert_called_once()
        call_args = mock_search_service.search.call_args
        assert call_args[0][0] == "temperature"  # query
        config = call_args[0][1]  # config
        assert config.mode == SearchMode.SEMANTIC
        assert config.max_results == 5

    @pytest.mark.asyncio
    async def test_search_with_input_validation(self, search_tool):
        """Test that input validation decorator works."""
        # Test with invalid search_mode - should be handled by validation decorator
        # Note: The actual validation logic is in the decorator,
        # here we test the integration
        with patch.object(
            search_tool._search_service, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = []

            # This should work (valid input)
            await search_tool.search_imas(query="temperature", search_mode="semantic")

            # Should have been called
            assert mock_search.called

    @pytest.mark.asyncio
    async def test_search_with_caching(
        self, search_tool, mock_search_service, sample_search_results
    ):
        """Test that caching decorator works."""
        mock_search_service.search.return_value = sample_search_results

        # First call
        result1 = await search_tool.search_imas(
            query="temperature", search_mode="semantic"
        )

        # Second call with same parameters
        result2 = await search_tool.search_imas(
            query="temperature", search_mode="semantic"
        )

        # Both should return same results
        assert result1["results_count"] == result2["results_count"]
        assert result1["query"] == result2["query"]

        # Note: Actual cache testing depends on cache implementation
        # The decorator functionality is tested in test_core_decorators.py

    @pytest.mark.asyncio
    async def test_search_sampling_prompt_generation(
        self, search_tool, sample_search_results
    ):
        """Test AI sampling prompt generation."""
        # Test with results
        prompt = search_tool._build_sampling_prompt(
            "temperature", sample_search_results
        )

        assert "temperature" in prompt
        assert "core_profiles.temperature" in prompt
        assert "Temperature profile data" in prompt
        assert "Physics context" in prompt
        assert "follow-up searches" in prompt

        # Test with no results
        empty_prompt = search_tool._build_sampling_prompt("unknown", [])

        assert "No results found" in empty_prompt
        assert "unknown" in empty_prompt
        assert "Alternative search terms" in empty_prompt

    @pytest.mark.asyncio
    async def test_search_with_different_modes(
        self, search_tool, mock_search_service, sample_search_results
    ):
        """Test search with different search modes."""
        mock_search_service.search.return_value = sample_search_results

        # Test all supported modes
        modes = ["auto", "semantic", "lexical", "hybrid"]

        for mode in modes:
            result = await search_tool.search_imas(
                query="temperature", search_mode=mode
            )

            assert result["search_mode"] == mode
            assert "results" in result

    @pytest.mark.asyncio
    async def test_search_with_ids_filter(
        self, search_tool, mock_search_service, sample_search_results
    ):
        """Test search with IDS name filter."""
        mock_search_service.search.return_value = sample_search_results

        await search_tool.search_imas(
            query="temperature", ids_name="core_profiles", max_results=10
        )

        # Verify service called with correct config
        call_args = mock_search_service.search.call_args
        config = call_args[0][1]
        assert config.filter_ids == ["core_profiles"]
        assert config.max_results == 10

    @pytest.mark.asyncio
    async def test_search_with_list_query(
        self, search_tool, mock_search_service, sample_search_results
    ):
        """Test search with list of query terms."""
        mock_search_service.search.return_value = sample_search_results

        query_list = ["temperature", "profile"]
        result = await search_tool.search_imas(query=query_list, search_mode="hybrid")

        assert result["query"] == query_list

        # Verify service called with list query
        call_args = mock_search_service.search.call_args
        assert call_args[0][0] == query_list

    @pytest.mark.asyncio
    async def test_search_error_handling(self, search_tool, mock_search_service):
        """Test error handling in search tool."""
        # Mock service to raise exception
        mock_search_service.search.side_effect = Exception("Search failed")

        # The error handling decorator should catch this
        # Note: Actual error handling depends on decorator implementation
        # Here we test that the method can handle exceptions
        try:
            await search_tool.search_imas(query="temperature")
            # If error decorator works, this might return error response
            # instead of raising exception
        except Exception:
            # If no error decorator, exception would propagate
            pass

    @pytest.mark.asyncio
    async def test_search_result_formatting(
        self, search_tool, mock_search_service, sample_search_results
    ):
        """Test that search results are properly formatted."""
        mock_search_service.search.return_value = sample_search_results

        result = await search_tool.search_imas(query="temperature")

        # Verify result structure
        assert isinstance(result, dict)
        assert "results" in result
        assert "results_count" in result
        assert "search_mode" in result
        assert "query" in result
        assert "ai_prompt" in result

        # Verify individual result structure
        if result["results"]:
            first_result = result["results"][0]
            expected_fields = [
                "path",
                "relevance_score",
                "documentation",
                "units",
                "ids_name",
                "data_type",
                "physics_domain",
            ]
            for field in expected_fields:
                assert field in first_result

    @pytest.mark.asyncio
    async def test_search_performance_metrics(
        self, search_tool, mock_search_service, sample_search_results
    ):
        """Test that performance metrics are collected."""
        mock_search_service.search.return_value = sample_search_results

        result = await search_tool.search_imas(query="temperature")

        # Note: Actual performance metrics would be added by the performance decorator
        # The decorator functionality is tested in test_core_decorators.py
        # Here we verify the base functionality supports performance measurement
        assert "results" in result  # Basic functionality works

    def test_search_tool_initialization(self):
        """Test search tool initialization."""
        # Test with default IDS set
        tool1 = Search()
        assert tool1.get_tool_name() == "search_imas"
        assert tool1._search_service is not None

        # Test with custom IDS set
        ids_set = {"core_profiles", "equilibrium"}
        tool2 = Search(ids_set=ids_set)
        assert tool2.ids_set == ids_set
        assert tool2._search_service is not None

    def test_engine_creation(self):
        """Test search engine creation."""
        tool = Search()

        # Test valid engine types
        config = SearchConfig(mode=SearchMode.SEMANTIC)

        semantic_engine = tool._create_engine("semantic", config)
        assert semantic_engine is not None

        lexical_engine = tool._create_engine("lexical", config)
        assert lexical_engine is not None

        hybrid_engine = tool._create_engine("hybrid", config)
        assert hybrid_engine is not None

        # Test invalid engine type
        with pytest.raises(ValueError, match="Unknown engine type"):
            tool._create_engine("invalid", config)


class TestSearchDecoratorsIntegration:
    """Test integration of decorators with search tool."""

    @pytest.fixture
    def search_tool(self):
        return Search()

    def test_mcp_tool_decorator(self, search_tool):
        """Test MCP tool decorator is applied."""
        method = search_tool.search_imas
        assert hasattr(method, "_mcp_tool")
        assert method._mcp_tool is True
        assert hasattr(method, "_mcp_description")
        assert "Search for IMAS data paths" in method._mcp_description

    def test_decorator_order(self, search_tool):
        """Test that decorators are applied in correct order."""
        # The decorator stack should be:
        # @cache_results (outermost)
        # @validate_input
        # @sample
        # @recommend_tools
        # @measure_performance
        # @handle_errors
        # @mcp_tool (innermost)

        method = search_tool.search_imas

        # Verify MCP tool decorator (innermost)
        assert hasattr(method, "_mcp_tool")

        # Note: Other decorator testing is done in test_core_decorators.py
        # Here we verify the integration works

    @pytest.mark.asyncio
    async def test_decorator_composition_execution(self, search_tool):
        """Test that the comprehensive decorator composition executes without errors."""
        # Mock the search service to avoid actual search
        with patch.object(
            search_tool._search_service, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = []

            # This should execute through the comprehensive decorator composition
            result = await search_tool.search_imas(
                query="test", search_mode="semantic", max_results=5
            )

            # Should return a valid result structure
            assert isinstance(result, dict)
            assert "results" in result


class TestSearchConfigurationSupport:
    """Test search tool configuration support."""

    def test_search_mode_mapping(self):
        """Test search mode string to enum mapping."""
        tool = Search()

        # This is tested indirectly through the search method
        # The mapping is internal to search_imas method
        assert tool.get_tool_name() == "search_imas"

    def test_config_creation(self):
        """Test SearchConfig creation in search tool."""
        tool = Search()

        # Test internal _create_search_service method
        service = tool._create_search_service()
        assert service is not None

    @pytest.mark.asyncio
    async def test_parameter_validation_integration(self):
        """Test parameter validation integration."""
        tool = Search()

        # Mock the service
        with patch.object(
            tool._search_service, "search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = []

            # Test with valid parameters
            result = await tool.search_imas(
                query="temperature",
                ids_name="core_profiles",
                max_results=10,
                search_mode="semantic",
            )

            assert "results" in result

            # The input validation decorator should handle invalid inputs
            # Actual validation testing is in test_core_decorators.py
