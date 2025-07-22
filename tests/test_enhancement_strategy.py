"""
Unit tests for AI Enhancement Strategy.

Tests the selective AI enhancement strategy that determines when AI processing
should be applied based on tool type, context, and request parameters.
"""

import pytest
from unittest.mock import Mock

from imas_mcp.search.ai_enhancement_strategy import (
    AI_ENHANCEMENT_STRATEGY,
    should_apply_ai_enhancement,
    suggest_follow_up_tools,
    _should_enhance_search,
    _should_enhance_structure_analysis,
    _should_enhance_relationships,
    _should_enhance_bulk_export,
    _should_enhance_physics_domain,
)


class TestAIEnhancementStrategy:
    """Test AI enhancement strategy configuration and logic."""

    def test_ai_enhancement_strategy_mapping(self):
        """Test that all expected tools have enhancement strategies defined."""
        expected_tools = [
            "search_imas",
            "explain_concept",
            "get_overview",
            "analyze_ids_structure",
            "explore_relationships",
            "explore_identifiers",
            "export_ids_bulk",
            "export_physics_domain",
        ]

        for tool in expected_tools:
            assert tool in AI_ENHANCEMENT_STRATEGY
            assert AI_ENHANCEMENT_STRATEGY[tool] in ["always", "conditional", "never"]

    def test_strategy_values(self):
        """Test specific strategy values for each tool."""
        assert AI_ENHANCEMENT_STRATEGY["search_imas"] == "conditional"
        assert AI_ENHANCEMENT_STRATEGY["explain_concept"] == "always"
        assert AI_ENHANCEMENT_STRATEGY["get_overview"] == "always"
        assert AI_ENHANCEMENT_STRATEGY["analyze_ids_structure"] == "conditional"
        assert AI_ENHANCEMENT_STRATEGY["explore_relationships"] == "conditional"
        assert AI_ENHANCEMENT_STRATEGY["explore_identifiers"] == "never"
        assert AI_ENHANCEMENT_STRATEGY["export_ids_bulk"] == "conditional"
        assert AI_ENHANCEMENT_STRATEGY["export_physics_domain"] == "conditional"


class TestShouldApplyAIEnhancement:
    """Test the main AI enhancement decision function."""

    def test_no_context_returns_false(self):
        """Test that no AI enhancement is applied when context is None."""
        result = should_apply_ai_enhancement(
            "search_imas",
            ("plasma temperature",),
            {"search_mode": "comprehensive"},
            ctx=None,
        )
        assert result is False

    def test_never_strategy_returns_false(self):
        """Test that 'never' strategy always returns False."""
        mock_ctx = Mock()
        result = should_apply_ai_enhancement(
            "explore_identifiers", ("temperature",), {"scope": "all"}, ctx=mock_ctx
        )
        assert result is False

    def test_always_strategy_returns_true(self):
        """Test that 'always' strategy returns True when context is available."""
        mock_ctx = Mock()
        result = should_apply_ai_enhancement(
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
        result = should_apply_ai_enhancement(
            "search_imas",
            ("plasma temperature",),
            {"search_mode": "comprehensive"},
            ctx=mock_ctx,
        )
        assert result is True  # comprehensive mode should enable AI

        result = should_apply_ai_enhancement(
            "search_imas", ("plasma",), {"search_mode": "fast"}, ctx=mock_ctx
        )
        assert result is False  # fast mode should not enable AI

    def test_unknown_tool_defaults_to_true(self):
        """Test that unknown tools default to AI enhancement."""
        mock_ctx = Mock()
        result = should_apply_ai_enhancement("unknown_tool", (), {}, ctx=mock_ctx)
        assert result is True


class TestSearchEnhancement:
    """Test search-specific AI enhancement logic."""

    def test_comprehensive_mode_enables_ai(self):
        """Test that comprehensive search mode enables AI."""
        assert _should_enhance_search((), {"search_mode": "comprehensive"}) is True
        assert _should_enhance_search((), {"search_mode": "semantic"}) is True

    def test_fast_mode_disables_ai(self):
        """Test that fast search mode disables AI."""
        assert _should_enhance_search((), {"search_mode": "fast"}) is False
        assert _should_enhance_search((), {"search_mode": "lexical"}) is False

    def test_complex_queries_enable_ai(self):
        """Test that complex queries enable AI enhancement."""
        # Multiple terms in list
        assert (
            _should_enhance_search((), {"query": ["plasma", "temperature", "density"]})
            is True
        )

        # Boolean operators
        assert _should_enhance_search((), {"query": "plasma AND temperature"}) is True
        assert _should_enhance_search((), {"query": "plasma OR density"}) is True
        assert _should_enhance_search((), {"query": "NOT equilibrium"}) is True

        # Long queries
        assert (
            _should_enhance_search((), {"query": "plasma temperature profile analysis"})
            is True
        )

    def test_simple_queries_disable_ai(self):
        """Test that simple queries don't trigger AI enhancement."""
        # Simple single term
        assert _should_enhance_search((), {"query": "plasma"}) is False

        # Simple two terms
        assert _should_enhance_search((), {"query": "plasma temperature"}) is False

        # Small list
        assert _should_enhance_search((), {"query": ["plasma", "temp"]}) is False

    def test_high_max_results_enables_ai(self):
        """Test that high max_results enables AI enhancement."""
        assert _should_enhance_search((), {"max_results": 20}) is True
        assert _should_enhance_search((), {"max_results": 10}) is False


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
                _should_enhance_structure_analysis((), {"ids_name": ids_name}) is True
            )

    def test_simple_ids_disable_ai(self):
        """Test that simple IDS names don't trigger AI enhancement."""
        assert (
            _should_enhance_structure_analysis((), {"ids_name": "simple_ids"}) is False
        )
        assert _should_enhance_structure_analysis((), {"ids_name": "test"}) is False


class TestRelationshipEnhancement:
    """Test relationship exploration AI enhancement logic."""

    def test_complex_relationship_types_enable_ai(self):
        """Test that complex relationship types enable AI."""
        # Specific complex types enable AI
        complex_types = ["physics", "measurement_dependencies"]

        for rel_type in complex_types:
            assert (
                _should_enhance_relationships((), {"relationship_type": rel_type})
                is True
            )

        # "all" type is now considered too general and doesn't enable AI by itself
        assert _should_enhance_relationships((), {"relationship_type": "all"}) is False

    def test_deep_analysis_enables_ai(self):
        """Test that deep relationship analysis enables AI."""
        assert _should_enhance_relationships((), {"max_depth": 3}) is True
        assert _should_enhance_relationships((), {"max_depth": 2}) is False

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
            assert _should_enhance_relationships((), {"path": path}) is True

    def test_simple_paths_disable_ai(self):
        """Test that simple paths don't trigger AI enhancement."""
        assert _should_enhance_relationships((), {"path": "simple/path"}) is False


class TestBulkExportEnhancement:
    """Test bulk export AI enhancement logic."""

    def test_enhanced_format_enables_ai(self):
        """Test that enhanced format enables AI."""
        assert _should_enhance_bulk_export((), {"output_format": "enhanced"}) is True

    def test_raw_format_disables_ai(self):
        """Test that raw format disables AI."""
        assert _should_enhance_bulk_export((), {"output_format": "raw"}) is False
        assert _should_enhance_bulk_export((), {"output_format": "structured"}) is False

    def test_multiple_ids_enable_ai(self):
        """Test that multiple IDS enable AI enhancement."""
        assert (
            _should_enhance_bulk_export(
                (), {"ids_list": ["ids1", "ids2", "ids3", "ids4"]}
            )
            is True
        )
        # Two IDS alone don't enable AI unless other conditions are met
        assert _should_enhance_bulk_export((), {"ids_list": ["ids1", "ids2"]}) is False

    def test_full_analysis_enables_ai(self):
        """Test that full analysis options enable AI."""
        # Need both relationships, physics context AND 3+ IDS
        assert (
            _should_enhance_bulk_export(
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
            _should_enhance_bulk_export(
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
            _should_enhance_bulk_export(
                (), {"include_relationships": False, "include_physics_context": False}
            )
            is False
        )


class TestPhysicsDomainEnhancement:
    """Test physics domain export AI enhancement logic."""

    def test_comprehensive_analysis_enables_ai(self):
        """Test that comprehensive analysis enables AI."""
        assert (
            _should_enhance_physics_domain((), {"analysis_depth": "comprehensive"})
            is True
        )

    def test_focused_analysis_disables_ai(self):
        """Test that focused analysis doesn't trigger AI."""
        assert (
            _should_enhance_physics_domain((), {"analysis_depth": "focused"}) is False
        )
        assert (
            _should_enhance_physics_domain((), {"analysis_depth": "overview"}) is False
        )

    def test_cross_domain_analysis_enables_ai(self):
        """Test that cross-domain analysis enables AI."""
        assert (
            _should_enhance_physics_domain((), {"include_cross_domain": True}) is True
        )
        assert (
            _should_enhance_physics_domain((), {"include_cross_domain": False}) is False
        )

    def test_large_exports_enable_ai(self):
        """Test that large exports enable AI enhancement."""
        assert _should_enhance_physics_domain((), {"max_paths": 25}) is True
        assert _should_enhance_physics_domain((), {"max_paths": 15}) is False


class TestToolSuggestions:
    """Test tool suggestion generation."""

    def test_search_follow_up_suggestions(self):
        """Test follow-up suggestions for search results."""
        search_results = {
            "results": [
                {
                    "path": "core_profiles/time_slice/profiles_1d",
                    "ids_name": "core_profiles",
                },
                {
                    "path": "equilibrium/time_slice/profiles_1d",
                    "ids_name": "equilibrium",
                },
                {"path": "transport/model/diffusion", "ids_name": "transport"},
            ],
            "physics_matches": [{"concept": "plasma temperature"}],
        }

        suggestions = suggest_follow_up_tools(search_results, "search_imas")

        assert len(suggestions) > 0
        assert any("explain_concept" in s["tool"] for s in suggestions)
        assert any("explore_relationships" in s["tool"] for s in suggestions)

        # Check suggestion structure
        for suggestion in suggestions:
            assert "tool" in suggestion
            assert "reason" in suggestion
            assert "sample_call" in suggestion

    def test_concept_follow_up_suggestions(self):
        """Test follow-up suggestions for concept explanations."""
        concept_results = {
            "concept": "plasma temperature",
            "related_paths": [
                "core_profiles/time_slice/profiles_1d/t_e",
                "edge_profiles/time_slice/ggd/electrons/temperature",
            ],
        }

        suggestions = suggest_follow_up_tools(concept_results, "explain_concept")

        assert len(suggestions) > 0
        assert any("search_imas" in s["tool"] for s in suggestions)

    def test_structure_follow_up_suggestions(self):
        """Test follow-up suggestions for structure analysis."""
        structure_results = {
            "ids_name": "core_profiles",
            "total_paths": 15,
            "path_patterns": {"profiles_1d": 8, "profiles_2d": 4},
        }

        suggestions = suggest_follow_up_tools(
            structure_results, "analyze_ids_structure"
        )

        assert len(suggestions) > 0

    def test_relationship_follow_up_suggestions(self):
        """Test follow-up suggestions for relationship exploration."""
        relationship_results = {
            "path": "core_profiles/time_slice",
            "related_paths": [
                {"ids_name": "equilibrium", "path": "equilibrium/time_slice"},
                {"ids_name": "transport", "path": "transport/model"},
            ],
            "analysis": {"cross_ids_paths": 2},
            "physics_relationships": {"concepts": [{"concept": "plasma equilibrium"}]},
        }

        suggestions = suggest_follow_up_tools(
            relationship_results, "explore_relationships"
        )

        assert len(suggestions) > 0

    def test_bulk_export_follow_up_suggestions(self):
        """Test follow-up suggestions for bulk export."""
        bulk_results = {
            "ids_data": {
                "core_profiles": {"physics_domains": ["core_plasma"]},
                "equilibrium": {"physics_domains": ["equilibrium", "core_plasma"]},
            }
        }

        suggestions = suggest_follow_up_tools(bulk_results, "export_ids_bulk")

        assert len(suggestions) > 0

    def test_empty_results_no_suggestions(self):
        """Test that empty results don't generate suggestions."""
        empty_results = {"results": []}

        suggestions = suggest_follow_up_tools(empty_results, "search_imas")

        # Should handle gracefully, may return empty list or basic suggestions
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
    assert AI_ENHANCEMENT_STRATEGY[tool_name] == strategy


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
    result = _should_enhance_search((), {"search_mode": search_mode})
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
    result = _should_enhance_bulk_export((), {"output_format": output_format})
    assert result == expected
