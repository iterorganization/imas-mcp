"""
Unit tests for AI Enhancement Strategy.

Tests the simplified AI enhancement strategy that determines when AI processing
should be applied based on tool type, context, and request parameters using
clear switch-case like logic.
"""

import pytest
from unittest.mock import Mock

from imas_mcp.search.ai_sampler import (
    EnhancementDecisionEngine,
    TOOL_ENHANCEMENT_CONFIG,
    EnhancementStrategy,
    ToolCategory,
)
from imas_mcp.search.tool_suggestions import suggest_follow_up_tools


class TestEnhancementConfiguration:
    """Test AI enhancement strategy configuration and logic."""

    def test_tool_enhancement_config_mapping(self):
        """Test that all expected tools have enhancement configurations defined."""
        expected_tools = [
            "search_imas",
            "explain_concept",
            "get_overview",
            "analyze_ids_structure",
            "explore_relationships",
            "explore_identifiers",
            "export_ids",
            "export_physics_domain",
        ]

        for tool in expected_tools:
            assert tool in TOOL_ENHANCEMENT_CONFIG
            config = TOOL_ENHANCEMENT_CONFIG[tool]
            assert "strategy" in config
            assert "category" in config
            assert isinstance(config["strategy"], EnhancementStrategy)
            assert isinstance(config["category"], ToolCategory)

    def test_strategy_values(self):
        """Test specific strategy values for each tool."""
        assert (
            TOOL_ENHANCEMENT_CONFIG["search_imas"]["strategy"]
            == EnhancementStrategy.CONDITIONAL
        )
        assert (
            TOOL_ENHANCEMENT_CONFIG["explain_concept"]["strategy"]
            == EnhancementStrategy.ALWAYS
        )
        assert (
            TOOL_ENHANCEMENT_CONFIG["get_overview"]["strategy"]
            == EnhancementStrategy.ALWAYS
        )
        assert (
            TOOL_ENHANCEMENT_CONFIG["analyze_ids_structure"]["strategy"]
            == EnhancementStrategy.CONDITIONAL
        )
        assert (
            TOOL_ENHANCEMENT_CONFIG["explore_relationships"]["strategy"]
            == EnhancementStrategy.CONDITIONAL
        )
        assert (
            TOOL_ENHANCEMENT_CONFIG["explore_identifiers"]["strategy"]
            == EnhancementStrategy.NEVER
        )
        assert (
            TOOL_ENHANCEMENT_CONFIG["export_ids"]["strategy"]
            == EnhancementStrategy.CONDITIONAL
        )
        assert (
            TOOL_ENHANCEMENT_CONFIG["export_physics_domain"]["strategy"]
            == EnhancementStrategy.CONDITIONAL
        )

    def test_backward_compatibility_mapping(self):
        """Test that backward compatibility mapping works correctly."""
        assert (
            TOOL_ENHANCEMENT_CONFIG["search_imas"]["strategy"]
            == EnhancementStrategy.CONDITIONAL
        )
        assert (
            TOOL_ENHANCEMENT_CONFIG["explain_concept"]["strategy"]
            == EnhancementStrategy.ALWAYS
        )
        assert (
            TOOL_ENHANCEMENT_CONFIG["explore_identifiers"]["strategy"]
            == EnhancementStrategy.NEVER
        )


class TestEnhancementDecisionEngine:
    """Test the main AI enhancement decision engine."""

    def test_no_context_returns_false(self):
        """Test that no AI enhancement is applied when context is None."""
        result = EnhancementDecisionEngine.should_enhance(
            "search_imas",
            ("plasma temperature",),
            {"search_mode": "comprehensive"},
            ctx=None,
        )
        assert result is False

    def test_never_strategy_returns_false(self):
        """Test that 'never' strategy always returns False."""
        mock_ctx = Mock()
        result = EnhancementDecisionEngine.should_enhance(
            "explore_identifiers", ("temperature",), {"scope": "all"}, ctx=mock_ctx
        )
        assert result is False

    def test_always_strategy_returns_true(self):
        """Test that 'always' strategy returns True when context is available."""
        mock_ctx = Mock()
        result = EnhancementDecisionEngine.should_enhance(
            "explain_concept",
            ("plasma temperature",),
            {"detail_level": "intermediate"},
            ctx=mock_ctx,
        )
        assert result is True

    def test_conditional_strategy_delegates_to_specific_function(self):
        """Test that conditional strategy calls the appropriate evaluation function."""
        mock_ctx = Mock()

        # Test search_imas conditional logic
        result = EnhancementDecisionEngine.should_enhance(
            "search_imas",
            ("plasma temperature",),
            {"search_mode": "comprehensive"},
            ctx=mock_ctx,
        )
        assert result is True  # comprehensive mode should enable AI

        result = EnhancementDecisionEngine.should_enhance(
            "search_imas", ("plasma",), {"search_mode": "fast"}, ctx=mock_ctx
        )
        assert result is False  # fast mode should not enable AI

    def test_unknown_tool_defaults_to_true(self):
        """Test that unknown tools default to AI enhancement."""
        mock_ctx = Mock()
        result = EnhancementDecisionEngine.should_enhance(
            "unknown_tool", (), {}, ctx=mock_ctx
        )
        assert result is True


class TestSearchEnhancement:
    """Test search-specific AI enhancement logic."""

    def test_comprehensive_mode_enables_ai(self):
        """Test that comprehensive search mode enables AI."""
        assert (
            EnhancementDecisionEngine._should_enhance_search(
                (), {"search_mode": "comprehensive"}
            )
            is True
        )
        assert (
            EnhancementDecisionEngine._should_enhance_search(
                (), {"search_mode": "semantic"}
            )
            is True
        )

    def test_fast_mode_disables_ai(self):
        """Test that fast search mode disables AI."""
        assert (
            EnhancementDecisionEngine._should_enhance_search(
                (), {"search_mode": "fast"}
            )
            is False
        )
        assert (
            EnhancementDecisionEngine._should_enhance_search(
                (), {"search_mode": "lexical"}
            )
            is False
        )

    def test_complex_queries_enable_ai(self):
        """Test that complex queries enable AI enhancement."""
        # Multiple terms in list
        assert (
            EnhancementDecisionEngine._should_enhance_search(
                (), {"query": ["plasma", "temperature", "density"]}
            )
            is True
        )

        # Boolean operators
        assert (
            EnhancementDecisionEngine._should_enhance_search(
                (), {"query": "plasma AND temperature"}
            )
            is True
        )
        assert (
            EnhancementDecisionEngine._should_enhance_search(
                (), {"query": "plasma OR density"}
            )
            is True
        )
        assert (
            EnhancementDecisionEngine._should_enhance_search(
                (), {"query": "NOT equilibrium"}
            )
            is True
        )

        # Long queries
        assert (
            EnhancementDecisionEngine._should_enhance_search(
                (), {"query": "plasma temperature profile analysis"}
            )
            is True
        )

    def test_simple_queries_disable_ai(self):
        """Test that simple queries don't trigger AI enhancement."""
        # Simple single term
        assert (
            EnhancementDecisionEngine._should_enhance_search((), {"query": "plasma"})
            is False
        )

        # Simple two terms
        assert (
            EnhancementDecisionEngine._should_enhance_search(
                (), {"query": "plasma temperature"}
            )
            is False
        )

        # Small list
        assert (
            EnhancementDecisionEngine._should_enhance_search(
                (), {"query": ["plasma", "temp"]}
            )
            is False
        )

    def test_high_max_results_enables_ai(self):
        """Test that high max_results enables AI enhancement."""
        assert (
            EnhancementDecisionEngine._should_enhance_search((), {"max_results": 20})
            is True
        )
        assert (
            EnhancementDecisionEngine._should_enhance_search((), {"max_results": 10})
            is False
        )


class TestStructureAnalysisEnhancement:
    """Test structure analysis AI enhancement logic."""

    def test_complex_ids_enable_ai(self):
        """Test that complex IDS names enable AI enhancement."""
        complex_ids = [
            "core_profiles",
            "equilibrium",
            "transport",
            "edge_profiles",
            "mhd",
            "disruption",
            "pellets",
            "wall",
            "ec_launchers",
        ]

        for ids_name in complex_ids:
            assert (
                EnhancementDecisionEngine._should_enhance_structure_analysis(
                    (), {"ids_name": ids_name}
                )
                is True
            )

    def test_simple_ids_disable_ai(self):
        """Test that simple IDS names don't trigger AI enhancement."""
        assert (
            EnhancementDecisionEngine._should_enhance_structure_analysis(
                (), {"ids_name": "simple_ids"}
            )
            is False
        )
        assert (
            EnhancementDecisionEngine._should_enhance_structure_analysis(
                (), {"ids_name": "test"}
            )
            is False
        )


class TestRelationshipEnhancement:
    """Test relationship exploration AI enhancement logic."""

    def test_complex_relationship_types_enable_ai(self):
        """Test that complex relationship types enable AI."""
        # Specific complex types enable AI
        complex_types = ["physics", "measurement_dependencies"]

        for rel_type in complex_types:
            assert (
                EnhancementDecisionEngine._should_enhance_relationships(
                    (), {"relationship_type": rel_type}
                )
                is True
            )

        # "all" type is now considered too general and doesn't enable AI by itself
        assert (
            EnhancementDecisionEngine._should_enhance_relationships(
                (), {"relationship_type": "all"}
            )
            is False
        )

    def test_deep_analysis_enables_ai(self):
        """Test that deep relationship analysis enables AI."""
        assert (
            EnhancementDecisionEngine._should_enhance_relationships(
                (), {"max_depth": 3}
            )
            is True
        )
        assert (
            EnhancementDecisionEngine._should_enhance_relationships(
                (), {"max_depth": 2}
            )
            is False
        )

    def test_complex_physics_paths_enable_ai(self):
        """Test that complex physics paths enable AI enhancement."""
        complex_paths = [
            "core_profiles/time_slice/profiles",
            "transport/model/diffusion",
            "equilibrium/time_slice",
            "mhd/fluctuations",
            "disruption/disruption_warning",
        ]

        for path in complex_paths:
            assert (
                EnhancementDecisionEngine._should_enhance_relationships(
                    (), {"path": path}
                )
                is True
            )

    def test_simple_paths_disable_ai(self):
        """Test that simple paths don't trigger AI enhancement."""
        assert (
            EnhancementDecisionEngine._should_enhance_relationships(
                (), {"path": "simple/path"}
            )
            is False
        )


class TestBulkExportEnhancement:
    """Test bulk export AI enhancement logic."""

    def test_enhanced_format_enables_ai(self):
        """Test that enhanced format enables AI."""
        assert (
            EnhancementDecisionEngine._should_enhance_bulk_export(
                (), {"output_format": "enhanced"}
            )
            is True
        )

    def test_raw_format_disables_ai(self):
        """Test that raw format disables AI."""
        assert (
            EnhancementDecisionEngine._should_enhance_bulk_export(
                (), {"output_format": "raw"}
            )
            is False
        )
        assert (
            EnhancementDecisionEngine._should_enhance_bulk_export(
                (), {"output_format": "structured"}
            )
            is False
        )

    def test_multiple_ids_enable_ai(self):
        """Test that multiple IDS enable AI enhancement."""
        assert (
            EnhancementDecisionEngine._should_enhance_bulk_export(
                (), {"ids_list": ["ids1", "ids2", "ids3", "ids4"]}
            )
            is True
        )
        # Two IDS alone don't enable AI unless other conditions are met
        assert (
            EnhancementDecisionEngine._should_enhance_bulk_export(
                (), {"ids_list": ["ids1", "ids2"]}
            )
            is False
        )

    def test_full_analysis_enables_ai(self):
        """Test that full analysis options enable AI."""
        # Need both relationships, physics context AND 3+ IDS
        assert (
            EnhancementDecisionEngine._should_enhance_bulk_export(
                (),
                {
                    "include_relationships": True,
                    "include_physics_context": True,
                    "ids_list": ["ids1", "ids2", "ids3"],  # Need at least 3 IDS
                },
            )
            is True
        )

        # Just the flags without enough IDS don't enable AI
        assert (
            EnhancementDecisionEngine._should_enhance_bulk_export(
                (),
                {
                    "include_relationships": True,
                    "include_physics_context": True,
                    "ids_list": ["ids1", "ids2"],  # Only 2 IDS - not enough
                },
            )
            is False
        )

        assert (
            EnhancementDecisionEngine._should_enhance_bulk_export(
                (), {"include_relationships": False, "include_physics_context": False}
            )
            is False
        )


class TestPhysicsDomainEnhancement:
    """Test physics domain export AI enhancement logic."""

    def test_comprehensive_analysis_enables_ai(self):
        """Test that comprehensive analysis enables AI."""
        assert (
            EnhancementDecisionEngine._should_enhance_physics_domain(
                (), {"analysis_depth": "comprehensive"}
            )
            is True
        )

    def test_focused_analysis_disables_ai(self):
        """Test that focused analysis doesn't trigger AI."""
        assert (
            EnhancementDecisionEngine._should_enhance_physics_domain(
                (), {"analysis_depth": "focused"}
            )
            is False
        )
        assert (
            EnhancementDecisionEngine._should_enhance_physics_domain(
                (), {"analysis_depth": "overview"}
            )
            is False
        )

    def test_cross_domain_analysis_enables_ai(self):
        """Test that cross-domain analysis enables AI."""
        assert (
            EnhancementDecisionEngine._should_enhance_physics_domain(
                (), {"include_cross_domain": True}
            )
            is True
        )
        assert (
            EnhancementDecisionEngine._should_enhance_physics_domain(
                (), {"include_cross_domain": False}
            )
            is False
        )

    def test_large_exports_enable_ai(self):
        """Test that large exports enable AI enhancement."""
        assert (
            EnhancementDecisionEngine._should_enhance_physics_domain(
                (), {"max_paths": 25}
            )
            is True
        )
        assert (
            EnhancementDecisionEngine._should_enhance_physics_domain(
                (), {"max_paths": 15}
            )
            is False
        )


class TestToolSuggestions:
    """Test tool suggestion generation (simplified for new architecture)."""

    def test_tool_suggestions_backward_compatibility(self):
        """Test that suggest_follow_up_tools function exists for backward compatibility."""
        search_results = {
            "results": [
                {
                    "path": "core_profiles/time_slice/profiles_1d",
                    "ids_name": "core_profiles",
                },
            ],
        }

        # Should not raise exception and return a list
        suggestions = suggest_follow_up_tools(search_results, "search_imas")
        assert isinstance(suggestions, list)

    def test_empty_results_no_suggestions(self):
        """Test that empty results don't generate suggestions."""
        empty_results = {"results": []}

        suggestions = suggest_follow_up_tools(empty_results, "search_imas")
        assert isinstance(suggestions, list)

    def test_error_handling_in_suggestions(self):
        """Test that suggestion generation handles errors gracefully."""
        malformed_results = {"invalid": "data"}

        # Should not raise exception
        suggestions = suggest_follow_up_tools(malformed_results, "unknown_tool")
        assert isinstance(suggestions, list)


@pytest.mark.parametrize(
    "tool_name,strategy",
    [
        ("search_imas", "conditional"),
        ("explain_concept", "always"),
        ("explore_identifiers", "never"),
    ],
)
def test_strategy_consistency(tool_name, strategy):
    """Parametrized test for strategy consistency."""
    assert TOOL_ENHANCEMENT_CONFIG[tool_name]["strategy"].value == strategy


@pytest.mark.parametrize(
    "search_mode,expected",
    [
        ("comprehensive", True),
        ("semantic", True),
        ("fast", False),
        ("lexical", False),
        ("auto", False),
    ],
)
def test_search_mode_enhancement(search_mode, expected):
    """Parametrized test for search mode AI enhancement."""
    result = EnhancementDecisionEngine._should_enhance_search(
        (), {"search_mode": search_mode}
    )
    assert result == expected


@pytest.mark.parametrize(
    "output_format,expected",
    [
        ("enhanced", True),
        ("raw", False),
        ("structured", False),
    ],
)
def test_export_format_enhancement(output_format, expected):
    """Parametrized test for export format AI enhancement."""
    result = EnhancementDecisionEngine._should_enhance_bulk_export(
        (), {"output_format": output_format}
    )
    assert result == expected
