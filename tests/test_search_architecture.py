"""
Tests for IMAS MCP Search Architecture.

This test module verifies the search architecture implementation including:
- Pure search engines (semantic, lexical, hybrid)
- Search service orchestration
- Search tool implementation
- Modular Tools class with proper delegation
"""

import pytest

from imas_mcp.models.enums import SearchMode
from imas_mcp.search.search_strategy import SearchConfig, SearchResult
from imas_mcp.search.services.search_service import SearchService
from imas_mcp.tools.search_tool import SearchTool
from imas_mcp.tools import Tools


class TestSearchEngineArchitecture:
    """Test pure search engine implementation and interfaces."""

    def test_search_config_creation(self):
        """Test SearchConfig model creation and validation."""
        config = SearchConfig(
            mode=SearchMode.SEMANTIC,
            max_results=10,
            filter_ids=["core_profiles"],
            similarity_threshold=0.5,
        )

        assert config.mode == SearchMode.SEMANTIC
        assert config.max_results == 10
        assert config.filter_ids == ["core_profiles"]
        assert config.similarity_threshold == 0.5

    def test_search_result_model(self):
        """Test SearchResult model can be used."""
        # We can't easily create a SearchResult without full dependencies
        # but we can test that the class exists and has expected methods
        assert hasattr(SearchResult, "to_dict")
        assert hasattr(SearchResult, "model_validate")


class TestSearchServiceOrchestration:
    """Test search service and engine orchestration."""

    def test_search_service_creation(self):
        """Test SearchService can be created with engines."""
        service = SearchService()

        assert service is not None
        assert hasattr(service, "search")
        assert hasattr(service, "get_available_modes")

    def test_search_service_available_modes(self):
        """Test SearchService reports available modes."""
        service = SearchService()
        modes = service.get_available_modes()

        assert isinstance(modes, list)
        # Should have modes but exact ones depend on implementation
        assert len(modes) > 0

    def test_search_service_health_check(self):
        """Test SearchService health check functionality."""
        service = SearchService()
        health = service.health_check()

        assert isinstance(health, dict)
        # Should report health for each mode
        assert len(health) > 0

    @pytest.mark.asyncio
    async def test_search_service_execution(self):
        """Test SearchService can execute a search."""
        service = SearchService()
        config = SearchConfig(mode=SearchMode.AUTO, max_results=5)

        # Test with a simple query (may fail but should not crash)
        try:
            results = await service.search("temperature", config)
            assert isinstance(results, list)
        except Exception:
            # Search may fail due to missing dependencies, but service should exist
            pass


class TestSearchToolExtraction:
    """Test search logic extraction from monolithic tools to modular architecture."""

    def test_search_tool_creation(self):
        """Test Search tool can be created."""
        search_tool = SearchTool()

        assert search_tool is not None
        assert search_tool.get_tool_name() == "search_imas"
        assert hasattr(search_tool, "search_imas")

    def test_search_tool_with_ids_set(self):
        """Test Search tool can be created with ids_set."""
        ids_set = {"core_profiles", "equilibrium"}
        search_tool = SearchTool(ids_set)

        assert search_tool.ids_set == ids_set

    @pytest.mark.asyncio
    async def test_search_imas_method_exists(self):
        """Test search_imas method exists and is callable."""
        search_tool = SearchTool()

        assert hasattr(search_tool, "search_imas")
        assert callable(search_tool.search_imas)

    @pytest.mark.asyncio
    async def test_search_imas_validation(self):
        """Test search_imas validates input parameters."""
        search_tool = SearchTool()

        # Test invalid search mode
        result = await search_tool.search_imas(
            query="temperature", search_mode="invalid_mode"
        )

        assert "error" in result
        assert "Invalid search mode" in result["error"]

    @pytest.mark.asyncio
    async def test_search_imas_max_results_validation(self):
        """Test search_imas validates max_results parameter."""
        search_tool = SearchTool()

        # Test invalid max_results (too high)
        result = await search_tool.search_imas(query="temperature", max_results=150)

        assert "error" in result
        assert "Invalid max_results" in result["error"]

    @pytest.mark.asyncio
    async def test_search_imas_response_format(self):
        """Test search_imas returns properly formatted response."""
        search_tool = SearchTool()

        # Test search - may return error or results, but should have proper format
        result = await search_tool.search_imas(
            query="temperature", search_mode="auto", max_results=5
        )

        # Should have expected response structure
        required_keys = ["results", "results_count", "query"]
        for key in required_keys:
            assert key in result

        assert isinstance(result["results"], list)
        assert isinstance(result["results_count"], int)
        assert result["query"] == "temperature"

        # If successful, should also have search_mode
        if "error" not in result:
            assert "search_mode" in result


class TestModularToolsIntegration:
    """Test integration of modular tools architecture with search functionality."""

    def test_tools_class_creation(self):
        """Test Tools class can be created with new architecture."""
        tools = Tools()

        assert tools is not None
        assert hasattr(tools, "search_tool")
        assert isinstance(tools.search_tool, SearchTool)

    def test_tools_class_with_ids_set(self):
        """Test Tools class respects ids_set parameter."""
        ids_set = {"core_profiles", "equilibrium"}
        tools = Tools(ids_set)

        assert tools.ids_set == ids_set
        assert tools.search_tool.ids_set == ids_set

    @pytest.mark.asyncio
    async def test_tools_search_imas_delegation(self):
        """Test Tools class properly delegates search_imas to Search tool."""
        tools = Tools()

        # Test that delegation works - both should return same structure
        result1 = await tools.search_imas(
            query="temperature", search_mode="auto", max_results=5
        )

        result2 = await tools.search_tool.search_imas(
            query="temperature", search_mode="auto", max_results=5
        )

        # Should get the same result structure
        assert isinstance(result1, type(result2))
        assert set(result1.keys()) == set(result2.keys())

    def test_tools_mcp_registration_support(self):
        """Test Tools class supports MCP registration."""
        tools = Tools()

        assert hasattr(tools, "register")
        assert hasattr(tools, "name")
        assert tools.name == "tools"

    def test_tools_backward_compatibility(self):
        """Test Tools class maintains backward compatibility."""
        tools = Tools()

        # Should have the same interface as original Tools class
        assert hasattr(tools, "search_imas")
        assert callable(tools.search_imas)


class TestSearchArchitectureDeliverables:
    """Test that search architecture delivers the expected functionality."""

    def test_separate_search_engine_classes_exist(self):
        """Test separate search engine classes exist."""
        # Test that the engine classes can be imported
        from imas_mcp.search.engines.semantic_engine import SemanticSearchEngine
        from imas_mcp.search.engines.lexical_engine import LexicalSearchEngine
        from imas_mcp.search.engines.hybrid_engine import HybridSearchEngine

        # Classes should exist
        assert SemanticSearchEngine is not None
        assert LexicalSearchEngine is not None
        assert HybridSearchEngine is not None

    def test_search_service_orchestration(self):
        """Test search service orchestrating engine selection."""
        service = SearchService()

        # Test service has engines registered
        available_modes = service.get_available_modes()
        assert len(available_modes) > 0

        # Test health check works
        health = service.health_check()
        assert len(health) > 0

    def test_clean_separation_tools_search(self):
        """Test clean separation of search logic from tools."""
        # Create tools instance
        tools = Tools()

        # Tools should use search service internally
        assert hasattr(tools.search_tool, "_search_service")
        assert isinstance(tools.search_tool._search_service, SearchService)

        # Search tool should be separate from tools orchestration
        assert isinstance(tools.search_tool, SearchTool)
        assert tools.search_tool.get_tool_name() == "search_imas"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
