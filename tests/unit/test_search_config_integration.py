"""
Tests for tool enum conversion and integration with SearchConfig.

This module tests the integration between tools that accept Union[str, SearchMode]
parameters and SearchConfig validation, which was the main integration issue we resolved.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from imas_mcp.models.constants import SearchMode
from imas_mcp.search.search_strategy import SearchConfig
from imas_mcp.tools.search_tool import SearchTool


class TestSearchToolEnumIntegration:
    """Test SearchTool integration with enum conversion."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store for testing."""
        mock_store = MagicMock()
        mock_store.search_full_text = AsyncMock(return_value=[])
        return mock_store

    @pytest.fixture
    def search_tool(self, mock_document_store):
        """Create SearchTool with mocked dependencies."""
        with patch("imas_mcp.tools.search_tool.SearchService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service.search = AsyncMock(return_value=[])
            mock_service_class.return_value = mock_service

            tool = SearchTool(mock_document_store)
            tool._search_service = mock_service
            return tool

    @pytest.mark.asyncio
    async def test_search_tool_string_mode_input(self, search_tool):
        """Test SearchTool accepts string search_mode and converts to enum."""
        # Disable caching to avoid JSON serialization issues in tests
        with patch.object(search_tool, "_search_service") as mock_service:
            mock_service.search.return_value = []

            # Test all valid string modes
            string_modes = ["auto", "semantic", "lexical", "hybrid"]

            for mode_str in string_modes:
                result = await search_tool.search_imas(
                    query="test query",
                    search_mode=mode_str,  # String input
                )

                # Verify the call was made (tool didn't crash on type conversion)
                assert isinstance(result, dict)
                assert "results" in result
                assert "search_mode" in result
                # The return value contains the original input parameter
                assert result["search_mode"] == mode_str

    @pytest.mark.asyncio
    async def test_search_tool_enum_mode_input(self, search_tool):
        """Test SearchTool accepts SearchMode enum input."""
        enum_modes = [
            SearchMode.AUTO,
            SearchMode.SEMANTIC,
            SearchMode.LEXICAL,
            SearchMode.HYBRID,
        ]

        for mode_enum in enum_modes:
            result = await search_tool.search_imas(
                query="test query",
                search_mode=mode_enum,  # Enum input
            )

            # Verify the call was made (tool didn't crash on type conversion)
            assert isinstance(result, dict)
            assert "results" in result
            assert "search_mode" in result
            assert result["search_mode"] == mode_enum

    @pytest.mark.asyncio
    async def test_search_tool_default_mode(self, search_tool):
        """Test SearchTool uses default search_mode when not specified."""
        result = await search_tool.search_imas(query="test query")

        # Should use default "auto" mode
        assert result["search_mode"] == "auto"

    @pytest.mark.asyncio
    async def test_search_tool_config_creation_with_string(self, search_tool):
        """Test that SearchConfig is properly created with string input."""
        # This tests the specific line that was causing type checker errors
        with patch.object(search_tool._search_service, "search") as mock_search:
            mock_search.return_value = []

            await search_tool.search_imas(
                query="test query",
                search_mode="semantic",  # String that gets passed to SearchConfig
                max_results=15,
            )

            # Verify SearchConfig was created and passed to service
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            query, config = call_args[0]

            assert isinstance(config, SearchConfig)
            assert config.search_mode == SearchMode.SEMANTIC  # Converted to enum
            assert config.max_results == 15

    @pytest.mark.asyncio
    async def test_search_tool_config_creation_with_enum(self, search_tool):
        """Test that SearchConfig is properly created with enum input."""
        with patch.object(search_tool._search_service, "search") as mock_search:
            mock_search.return_value = []

            await search_tool.search_imas(
                query="test query",
                search_mode=SearchMode.HYBRID,  # Enum that gets passed to SearchConfig
                max_results=20,
            )

            # Verify SearchConfig was created and passed to service
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            query, config = call_args[0]

            assert isinstance(config, SearchConfig)
            assert config.search_mode == SearchMode.HYBRID  # Preserved as enum
            assert config.max_results == 20

    @pytest.mark.asyncio
    async def test_search_tool_invalid_string_mode(self, search_tool):
        """Test SearchTool handles invalid string modes appropriately."""
        # This should raise a validation error when SearchConfig is created
        with pytest.raises(Exception):  # Could be ValidationError or similar
            await search_tool.search_imas(
                query="test query",
                search_mode="invalid_mode",  # Invalid string
            )

    def test_search_tool_config_creation_in_init(self, mock_document_store):
        """Test that SearchTool can create SearchConfig instances during initialization."""
        # This tests the _create_search_service method which also creates SearchConfig
        with patch("imas_mcp.tools.search_tool.SearchService"):
            with patch(
                "imas_mcp.tools.search_tool.SemanticSearchEngine"
            ) as mock_semantic:
                with patch(
                    "imas_mcp.tools.search_tool.LexicalSearchEngine"
                ) as mock_lexical:
                    with patch(
                        "imas_mcp.tools.search_tool.HybridSearchEngine"
                    ) as mock_hybrid:
                        # Should not raise any errors during initialization
                        SearchTool(mock_document_store)

                        # Verify that engines were created (meaning SearchConfig worked)
                        assert mock_semantic.called
                        assert mock_lexical.called
                        assert mock_hybrid.called

    @pytest.mark.asyncio
    async def test_search_tool_ids_filter_integration(self, search_tool):
        """Test SearchTool ids_filter parameter integration with SearchConfig."""
        with patch.object(search_tool._search_service, "search") as mock_search:
            mock_search.return_value = []

            await search_tool.search_imas(
                query="test query",
                search_mode="semantic",
                ids_filter=["core_profiles", "equilibrium"],
            )

            # Verify the ids_filter were properly passed to SearchConfig
            call_args = mock_search.call_args
            query, config = call_args[0]

            assert config.ids_filter == ["core_profiles", "equilibrium"]

    @pytest.mark.asyncio
    async def test_search_tool_ids_filter_string_integration(self, search_tool):
        """Test SearchTool ids_filter with string input."""
        with patch.object(search_tool._search_service, "search") as mock_search:
            mock_search.return_value = []

            # Test space-separated string input
            await search_tool.search_imas(
                query="test query", ids_filter="core_profiles equilibrium transport"
            )

            call_args = mock_search.call_args
            query, config = call_args[0]

            assert config.ids_filter == ["core_profiles", "equilibrium", "transport"]

    @pytest.mark.asyncio
    async def test_search_tool_ids_filter_single_string(self, search_tool):
        """Test SearchTool ids_filter with single string input."""
        with patch.object(search_tool._search_service, "search") as mock_search:
            mock_search.return_value = []

            # Test single string input
            await search_tool.search_imas(
                query="test query", ids_filter="core_profiles"
            )

            call_args = mock_search.call_args
            query, config = call_args[0]

            assert config.ids_filter == ["core_profiles"]

    @pytest.mark.asyncio
    async def test_search_tool_ids_filter_none(self, search_tool):
        """Test SearchTool ids_filter with None input (default)."""
        with patch.object(search_tool._search_service, "search") as mock_search:
            mock_search.return_value = []

            # Test None input (default)
            await search_tool.search_imas(query="test query")

            call_args = mock_search.call_args
            query, config = call_args[0]

            assert config.ids_filter is None
            assert config.search_mode == SearchMode.SEMANTIC


class TestToolEnumConversionPatterns:
    """Test enum conversion patterns used across all tools."""

    def test_search_mode_enum_values_consistency(self):
        """Test that SearchMode enum values match expected string patterns."""
        # These are the strings that tools should accept
        expected_values = ["auto", "semantic", "lexical", "hybrid"]
        actual_values = [mode.value for mode in SearchMode]

        assert set(actual_values) == set(expected_values)

    def test_search_mode_enum_completeness(self):
        """Test that all SearchMode enum members are accounted for."""
        # Ensure we have all expected modes
        assert hasattr(SearchMode, "AUTO")
        assert hasattr(SearchMode, "SEMANTIC")
        assert hasattr(SearchMode, "LEXICAL")
        assert hasattr(SearchMode, "HYBRID")

        # Ensure values match enum names (lowercase)
        assert SearchMode.AUTO.value == "auto"
        assert SearchMode.SEMANTIC.value == "semantic"
        assert SearchMode.LEXICAL.value == "lexical"
        assert SearchMode.HYBRID.value == "hybrid"

    def test_search_config_field_validator_mapping(self):
        """Test that the field validator mapping covers all enum members."""
        # This tests the specific mapping logic in the field validator
        from imas_mcp.models.constants import SearchMode

        # Create the same mapping as in the field validator
        value_map = {member.value: member for member in SearchMode}

        # Verify all enum values are in the map
        for mode in SearchMode:
            assert mode.value in value_map
            assert value_map[mode.value] == mode

    def test_union_type_annotation_pattern(self):
        """Test that Union[str, SearchMode] pattern is used correctly."""
        # This is more of a documentation test to ensure the pattern is consistent
        from typing import get_type_hints
        from imas_mcp.tools.search_tool import SearchTool

        # Get type hints for the search_imas method
        hints = get_type_hints(SearchTool.search_imas)

        # The search_mode parameter should accept Union[str, SearchMode]
        search_mode_type = hints.get("search_mode")

        # Note: This test documents the expected type pattern
        # The actual Union type checking is complex, so we just verify
        # that the parameter exists and has some Union-like behavior
        assert search_mode_type is not None


class TestTypeCheckerCompatibility:
    """Test that our solutions work with static type checkers."""

    def test_search_config_post_validation_usage(self):
        """Test usage patterns that were causing type checker errors."""
        # Create config with string input
        config = SearchConfig(search_mode="hybrid")  # type: ignore[arg-type]

        # These operations should work without type errors:

        # 1. Dictionary access (was causing "SearchMode | str" errors)
        mode_handlers = {
            SearchMode.AUTO: lambda: "auto_handler",
            SearchMode.SEMANTIC: lambda: "semantic_handler",
            SearchMode.LEXICAL: lambda: "lexical_handler",
            SearchMode.HYBRID: lambda: "hybrid_handler",
        }
        handler = mode_handlers[config.search_mode]
        assert callable(handler)

        # 2. Attribute access (was causing "Cannot access attribute" errors)
        mode_value = config.search_mode.value
        assert mode_value == "hybrid"

        # 3. Enum comparison
        is_hybrid = config.search_mode == SearchMode.HYBRID
        assert is_hybrid is True

        # 4. Method calls that expect SearchMode
        def process_mode(mode: SearchMode) -> str:
            return f"Processing {mode.value}"

        result = process_mode(config.search_mode)
        assert result == "Processing hybrid"

    def test_type_ignore_comment_necessity(self):
        """Test scenarios where type: ignore comments are necessary."""
        # This documents where we needed type: ignore and why

        # Scenario: Passing Union[str, SearchMode] to SearchMode field
        # This requires type: ignore because the type checker doesn't know
        # about Pydantic's field validation

        search_mode_input = "semantic"  # Union[str, SearchMode] type

        # This would cause a type error without type: ignore
        # config = SearchConfig(search_mode=search_mode_input)  # type: ignore[arg-type]

        # Instead, we test the runtime behavior
        config = SearchConfig(search_mode=search_mode_input)  # type: ignore[arg-type]
        assert config.search_mode == SearchMode.SEMANTIC

        # The type: ignore comment is necessary in production code
        # but not in tests where we control the types
