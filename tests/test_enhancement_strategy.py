"""
Unit tests for selective AI enhancement strategy.

Tests the enhancement decision engine that determines when to apply AI sampling
to tool results based on tool type, query complexity, and result characteristics.
"""

import pytest

from imas_mcp.search.enhancement_strategy import (
    EnhancementDecisionEngine,
    EnhancementConfig,
    EnhancementMode,
    ToolType,
    TOOL_ENHANCEMENT_CONFIG,
    should_enhance_result,
    get_enhancement_engine,
    configure_enhancement,
)


class TestEnhancementDecisionEngine:
    """Test the enhancement decision engine."""

    def test_init_with_default_config(self):
        """Test engine initialization with default config."""
        engine = EnhancementDecisionEngine()
        assert engine.config is not None

    def test_init_with_custom_config(self):
        """Test engine initialization with custom config."""
        config = EnhancementConfig()
        engine = EnhancementDecisionEngine(config)
        assert engine.config == config

    def test_should_enhance_never_mode(self):
        """Test that NEVER mode never enhances."""
        config = EnhancementConfig(mode=EnhancementMode.NEVER)
        engine = EnhancementDecisionEngine(config)

        result = engine.should_enhance("test_tool", "query", {"results": []})
        assert result is False

    def test_should_enhance_always_mode(self):
        """Test that ALWAYS mode always enhances."""
        config = EnhancementConfig(mode=EnhancementMode.ALWAYS)
        engine = EnhancementDecisionEngine(config)

        result = engine.should_enhance("test_tool", "query", {"results": []})
        assert result is True

    def test_classify_tool_search(self):
        """Test tool classification for search tools."""
        engine = EnhancementDecisionEngine()
        assert engine._classify_tool("search_imas") == ToolType.SEARCH

    def test_classify_tool_explanation(self):
        """Test tool classification for explanation tools."""
        engine = EnhancementDecisionEngine()
        assert engine._classify_tool("explain_concept") == ToolType.EXPLANATION

    def test_classify_tool_analysis(self):
        """Test tool classification for analysis tools."""
        engine = EnhancementDecisionEngine()
        assert engine._classify_tool("analyze_ids_structure") == ToolType.ANALYSIS

    def test_classify_tool_unknown(self):
        """Test tool classification for unknown tools defaults to search."""
        engine = EnhancementDecisionEngine()
        assert engine._classify_tool("unknown_tool") == ToolType.SEARCH

    def test_has_errors_detection(self):
        """Test error detection in results."""
        engine = EnhancementDecisionEngine()

        # Test result with error
        error_result = {"error": "Something went wrong"}
        assert engine._has_errors(error_result) is True

        # Test result with error status
        status_error_result = {"status": "error", "message": "Failed"}
        assert engine._has_errors(status_error_result) is True

        # Test clean result
        clean_result = {"results": [{"id": "test"}]}
        assert engine._has_errors(clean_result) is False

    def test_is_empty_result_detection(self):
        """Test empty result detection."""
        engine = EnhancementDecisionEngine()

        # Test empty results list
        empty_result = {"results": []}
        assert engine._is_empty_result(empty_result) is True

        # Test results with content
        full_result = {"results": [{"id": "test"}]}
        assert engine._is_empty_result(full_result) is False

        # Test non-dict result
        assert engine._is_empty_result("test") is False

    def test_get_result_count(self):
        """Test result count extraction."""
        engine = EnhancementDecisionEngine()

        # Test with results list
        result_with_list = {"results": [{"id": "1"}, {"id": "2"}]}
        assert engine._get_result_count(result_with_list) == 2

        # Test with hits list (SearchResponse format)
        result_with_hits = {"hits": [{"id": "1"}]}
        assert engine._get_result_count(result_with_hits) == 1

        # Test with count field
        result_with_count = {"count": 5}
        assert engine._get_result_count(result_with_count) == 5

        # Test with data list (also counts as results)
        result_with_data = {"data": [{"id": "test"}]}
        assert engine._get_result_count(result_with_data) == 1

        # Test with no count info (non-dict returns 1 if truthy)
        result_no_count = {"other": "test"}
        assert engine._get_result_count(result_no_count) == 1


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_enhancement_engine_singleton(self):
        """Test that get_enhancement_engine returns singleton."""
        engine1 = get_enhancement_engine()
        engine2 = get_enhancement_engine()
        assert engine1 is engine2

    def test_should_enhance_result_convenience(self):
        """Test convenience function delegates to engine."""
        result = should_enhance_result("search_imas", "test query", {"results": []})
        assert isinstance(result, bool)

    def test_configure_enhancement(self):
        """Test global configuration."""
        new_config = EnhancementConfig()
        configure_enhancement(new_config)

        # Verify new engine uses new config
        engine = get_enhancement_engine()
        assert engine.config == new_config


class TestToolEnhancementConfig:
    """Test the global tool enhancement configuration."""

    def test_global_config_exists(self):
        """Test that global config exists and has expected attributes."""
        assert TOOL_ENHANCEMENT_CONFIG is not None
        assert hasattr(TOOL_ENHANCEMENT_CONFIG, "mode")
        assert hasattr(TOOL_ENHANCEMENT_CONFIG, "tool_specific_settings")

    def test_default_config_mode(self):
        """Test default configuration mode."""
        assert TOOL_ENHANCEMENT_CONFIG.mode == EnhancementMode.SMART


@pytest.mark.parametrize(
    "tool_name,tool_type",
    [
        ("search_imas", ToolType.SEARCH),
        ("explain_concept", ToolType.EXPLANATION),
        ("explore_identifiers", ToolType.IDENTIFIERS),
        ("get_overview", ToolType.OVERVIEW),
        ("analyze_ids_structure", ToolType.ANALYSIS),
    ],
)
def test_tool_classification(tool_name, tool_type):
    """Parametrized test for tool classification."""
    engine = EnhancementDecisionEngine()
    assert engine._classify_tool(tool_name) == tool_type
