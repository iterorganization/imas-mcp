"""
Tests for decorator functionality enhancements.

This module tests that the query_hints and tool_hints decorators
properly populate the SearchResult fields.
"""

import pytest

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.result_models import SearchResult
from imas_mcp.models.suggestion_models import SearchSuggestion, ToolSuggestion
from imas_mcp.search.decorators.query_hints import (
    apply_query_hints,
    generate_search_query_hints,
    query_hints,
)
from imas_mcp.search.decorators.tool_hints import (
    apply_tool_hints,
    generate_search_tool_hints,
    tool_hints,
)
from imas_mcp.search.search_strategy import SearchHit


class TestQueryHintsDecorator:
    """Test query hints decorator functionality."""

    def test_generate_search_query_hints_with_results(self):
        """Test query hint generation for successful searches."""
        # Create a SearchResult with some hits
        hits = [
            SearchHit(
                path="core_profiles/profiles_1d/temperature",
                score=0.95,
                rank=0,
                physics_domain="core_transport",
                documentation="Electron temperature profile",
                data_type="FLT_1D",
                units="eV",
                ids_name="core_profiles",
            ),
            SearchHit(
                path="equilibrium/time_slice/boundary/temperature",
                score=0.87,
                rank=1,
                physics_domain="equilibrium",
                documentation="Boundary temperature",
                data_type="FLT_1D",
                units="eV",
                ids_name="equilibrium",
            ),
        ]

        result = SearchResult(
            query="temperature",
            hits=hits,
            search_mode=SearchMode.SEMANTIC,
        )

        hints = generate_search_query_hints(result)

        # Should generate hints for successful search
        assert len(hints) > 0
        assert all(isinstance(hint, SearchSuggestion) for hint in hints)

        # Should suggest related IDS or physics domains
        hint_suggestions = [hint.suggestion for hint in hints]
        assert any("core_profiles" in suggestion for suggestion in hint_suggestions)

    def test_generate_search_query_hints_no_results(self):
        """Test query hint generation for failed searches."""
        result = SearchResult(
            query="nonexistent_term",
            hits=[],
            search_mode=SearchMode.SEMANTIC,
        )

        hints = generate_search_query_hints(result)

        # Should generate alternative suggestions for empty results
        assert len(hints) > 0
        hint_suggestions = [hint.suggestion for hint in hints]

        # Should suggest broader search terms
        assert any("*" in suggestion for suggestion in hint_suggestions)

    def test_apply_query_hints(self):
        """Test that apply_query_hints properly sets the field."""
        result = SearchResult(query="test", hits=[])

        # Initially should have empty hints
        assert result.query_hints == []

        # Apply hints
        enhanced_result = apply_query_hints(result, max_hints=3)

        # Should now have hints
        assert len(enhanced_result.query_hints) <= 3
        assert all(
            isinstance(hint, SearchSuggestion) for hint in enhanced_result.query_hints
        )

    def test_query_hints_decorator(self):
        """Test the query_hints decorator function."""

        @query_hints(max_hints=2)
        async def mock_search_function() -> SearchResult:
            return SearchResult(query="temperature", hits=[])

        # Test the decorator
        result = mock_search_function()
        assert len(result.query_hints) <= 2


class TestToolHintsDecorator:
    """Test tool hints decorator functionality."""

    def test_generate_search_tool_hints_with_results(self):
        """Test tool hint generation for successful searches."""
        hits = [
            SearchHit(
                path="core_profiles/profiles_1d/temperature",
                score=0.95,
                rank=0,
                physics_domain="core_transport",
                documentation="Electron temperature profile",
                data_type="FLT_1D",
                units="eV",
                ids_name="core_profiles",
            ),
            SearchHit(
                path="equilibrium/time_slice/boundary/temperature",
                score=0.87,
                rank=1,
                physics_domain="equilibrium",
                documentation="Boundary temperature",
                data_type="FLT_1D",
                units="eV",
                ids_name="equilibrium",
            ),
        ]

        result = SearchResult(
            query="temperature",
            hits=hits,
            search_mode=SearchMode.SEMANTIC,
        )

        hints = generate_search_tool_hints(result)

        # Should generate tool suggestions
        assert len(hints) > 0
        assert all(isinstance(hint, ToolSuggestion) for hint in hints)

        # Should suggest relevant tools
        tool_names = [hint.tool_name for hint in hints]
        assert "explore_relationships" in tool_names
        assert "analyze_ids_structure" in tool_names

    def test_generate_search_tool_hints_no_results(self):
        """Test tool hint generation for failed searches."""
        result = SearchResult(
            query="nonexistent_term",
            hits=[],
            search_mode=SearchMode.SEMANTIC,
        )

        hints = generate_search_tool_hints(result)

        # Should generate discovery tool suggestions
        assert len(hints) > 0
        tool_names = [hint.tool_name for hint in hints]
        assert "get_overview" in tool_names
        assert "explore_identifiers" in tool_names
        assert "explain_concept" in tool_names

    def test_apply_tool_hints(self):
        """Test that apply_tool_hints properly sets the field."""
        result = SearchResult(query="test", hits=[])

        # Initially should have empty hints
        assert result.tool_hints == []

        # Apply hints
        enhanced_result = apply_tool_hints(result, max_hints=3)

        # Should now have hints
        assert len(enhanced_result.tool_hints) <= 3
        assert all(
            isinstance(hint, ToolSuggestion) for hint in enhanced_result.tool_hints
        )

    def test_tool_hints_decorator(self):
        """Test the tool_hints decorator function."""

        @tool_hints(max_hints=2)
        async def mock_search_function() -> SearchResult:
            return SearchResult(query="temperature", hits=[])

        # Test the decorator
        result = mock_search_function()
        assert len(result.tool_hints) <= 2

    def test_tool_hints_many_results_suggests_export(self):
        """Test that many results suggest export tools."""
        # Create many hits to trigger export suggestion
        hits = []
        for i in range(10):
            hits.append(
                SearchHit(
                    path=f"core_profiles/profiles_1d/field_{i}",
                    score=0.8,
                    rank=i,
                    physics_domain="core_transport",
                    documentation=f"Field {i}",
                    data_type="FLT_1D",
                    units="unit",
                    ids_name="core_profiles",
                )
            )

        result = SearchResult(
            query="profiles",
            hits=hits,
            search_mode=SearchMode.SEMANTIC,
        )

        hints = generate_search_tool_hints(result)
        tool_names = [hint.tool_name for hint in hints]

        # Should suggest export tools for many results
        assert "export_ids" in tool_names


class TestHintQuality:
    """Test the quality and relevance of generated hints."""

    def test_psi_search_specific_hints(self):
        """Test that psi search generates appropriate physics hints."""
        hits = [
            SearchHit(
                path="equilibrium/time_slice/profiles_1d/psi_norm",
                score=0.9,
                rank=0,
                physics_domain="flux_surfaces",
                documentation="Normalized poloidal flux",
                data_type="FLT_1D",
                units="1",
                ids_name="equilibrium",
            )
        ]

        result = SearchResult(
            query="psi",
            hits=hits,
            search_mode=SearchMode.HYBRID,
        )

        # Test query hints
        query_hints = generate_search_query_hints(result)
        query_suggestions = [hint.suggestion for hint in query_hints]

        # Should suggest flux-related terms
        assert any(
            "equilibrium" in suggestion.lower() for suggestion in query_suggestions
        )

        # Test tool hints
        tool_hints = generate_search_tool_hints(result)
        tool_names = [hint.tool_name for hint in tool_hints]

        # Should suggest analyzing equilibrium structure
        assert "analyze_ids_structure" in tool_names
        # Should suggest exploring relationships
        assert "explore_relationships" in tool_names

    def test_hint_confidence_scoring(self):
        """Test that hints have appropriate confidence scores."""
        result = SearchResult(query="temperature", hits=[])

        hints = generate_search_query_hints(result)

        # All hints should have confidence scores
        for hint in hints:
            assert hint.confidence is not None
            assert 0.0 <= hint.confidence <= 1.0

    def test_hint_deduplication(self):
        """Test that hints don't contain duplicates."""
        hits = [
            SearchHit(
                path="core_profiles/profiles_1d/temperature",
                score=0.95,
                rank=0,
                physics_domain="core_transport",
                documentation="Temperature",
                data_type="FLT_1D",
                units="eV",
                ids_name="core_profiles",
            )
        ]

        result = SearchResult(
            query="temperature",
            hits=hits,
            search_mode=SearchMode.SEMANTIC,
        )

        query_hints = generate_search_query_hints(result)
        tool_hints = generate_search_tool_hints(result)

        # Check for duplicate query suggestions
        query_suggestions = [hint.suggestion for hint in query_hints]
        assert len(query_suggestions) == len(set(query_suggestions))

        # Check for duplicate tool suggestions
        tool_names = [hint.tool_name for hint in tool_hints]
        assert len(tool_names) == len(set(tool_names))


class TestDecoratorErrorHandling:
    """Test error handling in decorator functions."""

    def test_query_hints_error_handling(self):
        """Test that query hints decorator handles errors gracefully."""
        # Create a result that might cause issues
        result = SearchResult(query=None, hits=[])

        # Should not raise an exception
        enhanced_result = apply_query_hints(result)

        # Should have empty hints instead of crashing
        assert enhanced_result.query_hints == []

    def test_tool_hints_error_handling(self):
        """Test that tool hints decorator handles errors gracefully."""
        # Create a result that might cause issues
        result = SearchResult(query=None, hits=[])

        # Should not raise an exception
        enhanced_result = apply_tool_hints(result)

        # Should have empty hints instead of crashing
        assert enhanced_result.tool_hints == []
