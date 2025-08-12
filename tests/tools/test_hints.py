"""
Tests for hint decorator functionality across all MCP tools.

This module tests that query_hints and tool_hints decorators properly
populate the hint fields for all decorated tools.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from imas_mcp.models.constants import DetailLevel, SearchMode
from imas_mcp.models.result_models import (
    ConceptResult,
    IDSExport,
    SearchResult,
    StructureResult,
)
from imas_mcp.models.suggestion_models import SearchSuggestion, ToolSuggestion
from imas_mcp.search.decorators.query_hints import (
    apply_query_hints,
    generate_generic_query_hints,
)
from imas_mcp.search.decorators.tool_hints import (
    apply_tool_hints,
    generate_generic_tool_hints,
)
from imas_mcp.search.search_strategy import SearchHit


class TestToolHintsForAllTools:
    """Test tool hints functionality for all MCP tools."""

    def test_tool_hints_for_search_result(self):
        """Test tool hints generation for SearchResult."""
        search_result = SearchResult(
            hits=[
                SearchHit(
                    path="core_profiles/profiles_1d/temperature",
                    score=0.95,
                    rank=0,
                    search_mode=SearchMode.SEMANTIC,
                    physics_domain="core_transport",
                    documentation="Electron temperature profile",
                    data_type="FLT_1D",
                    units="eV",
                    ids_name="core_profiles",
                )
            ],
            query="temperature",
            search_mode=SearchMode.SEMANTIC,
        )

        enhanced_result = apply_tool_hints(search_result, max_hints=3)

        # Should have tool hints populated
        assert hasattr(enhanced_result, "tool_hints")
        assert isinstance(enhanced_result.tool_hints, list)
        assert len(enhanced_result.tool_hints) > 0
        assert all(
            isinstance(hint, ToolSuggestion) for hint in enhanced_result.tool_hints
        )

    def test_tool_hints_for_concept_result(self):
        """Test tool hints generation for ConceptResult."""
        concept_result = ConceptResult(
            concept="poloidal flux",
            explanation="The poloidal flux in tokamak physics...",
            detail_level=DetailLevel.INTERMEDIATE,
            related_topics=["magnetic equilibrium", "flux surfaces"],
            nodes=[],
            physics_domains=["equilibrium", "flux_surfaces"],
            query="poloidal flux",
            search_mode=SearchMode.SEMANTIC,
            max_results=10,
            ids_filter=None,
        )

        enhanced_result = apply_tool_hints(concept_result, max_hints=3)

        # Should have tool hints populated
        assert hasattr(enhanced_result, "tool_hints")
        assert isinstance(enhanced_result.tool_hints, list)
        assert len(enhanced_result.tool_hints) > 0

        # Should suggest relevant tools for concept exploration
        tool_names = [hint.tool_name for hint in enhanced_result.tool_hints]
        assert "search_imas" in tool_names
        assert "explore_relationships" in tool_names

    def test_tool_hints_for_structure_result(self):
        """Test tool hints generation for StructureResult."""
        structure_result = StructureResult(
            ids_name="equilibrium",
            description="Equilibrium data structure",
            structure={"profiles_1d": 5, "profiles_2d": 3, "time_slice": 10},
            sample_paths=["equilibrium/time_slice/profiles_1d/psi"],
            max_depth=3,
            query="equilibrium structure",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
            physics_domains=["equilibrium"],
        )

        enhanced_result = apply_tool_hints(structure_result, max_hints=2)

        # Should have tool hints populated
        assert hasattr(enhanced_result, "tool_hints")
        assert isinstance(enhanced_result.tool_hints, list)
        assert len(enhanced_result.tool_hints) > 0

        # Should suggest export and search tools for structure results
        tool_names = [hint.tool_name for hint in enhanced_result.tool_hints]
        assert "export_ids" in tool_names
        assert "search_imas" in tool_names

    def test_tool_hints_for_export_result(self):
        """Test tool hints generation for export results."""
        export_result = IDSExport(
            ids_names=["core_profiles", "equilibrium"],
            include_physics=True,
            include_relationships=True,
            data={},
            metadata={},
            query="export data",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
        )

        enhanced_result = apply_tool_hints(export_result, max_hints=2)

        # Should have tool hints populated
        assert hasattr(enhanced_result, "tool_hints")
        assert isinstance(enhanced_result.tool_hints, list)
        assert len(enhanced_result.tool_hints) > 0

        # Should suggest analysis tools for export results
        tool_names = [hint.tool_name for hint in enhanced_result.tool_hints]
        assert (
            "analyze_ids_structure" in tool_names
            or "explore_relationships" in tool_names
        )

    def test_generate_generic_tool_hints(self):
        """Test the generic tool hint generator directly."""
        # Test ConceptResult hints
        concept_result = ConceptResult(
            concept="test_concept",
            explanation="Test explanation",
            detail_level=DetailLevel.BASIC,
            related_topics=[],
            nodes=[],
            physics_domains=[],
            query="test",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
        )

        hints = generate_generic_tool_hints(concept_result)
        assert len(hints) > 0
        assert all(isinstance(hint, ToolSuggestion) for hint in hints)

        # Test StructureResult hints
        structure_result = StructureResult(
            ids_name="test_ids",
            description="Test description",
            structure={},
            sample_paths=[],
            max_depth=0,
            query="test",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
            physics_domains=[],
        )

        hints = generate_generic_tool_hints(structure_result)
        assert len(hints) > 0
        assert all(isinstance(hint, ToolSuggestion) for hint in hints)


class TestQueryHintsForAllTools:
    """Test query hints functionality for all MCP tools."""

    def test_query_hints_for_search_result(self):
        """Test query hints generation for SearchResult."""
        search_result = SearchResult(
            hits=[
                SearchHit(
                    path="equilibrium/time_slice/profiles_1d/psi",
                    score=0.9,
                    rank=0,
                    search_mode=SearchMode.SEMANTIC,
                    physics_domain="flux_surfaces",
                    documentation="Poloidal flux profile",
                    data_type="FLT_1D",
                    units="Wb",
                    ids_name="equilibrium",
                )
            ],
            query="psi",
            search_mode=SearchMode.SEMANTIC,
        )

        enhanced_result = apply_query_hints(search_result, max_hints=3)

        # Should have query hints populated
        assert hasattr(enhanced_result, "query_hints")
        assert isinstance(enhanced_result.query_hints, list)
        assert len(enhanced_result.query_hints) > 0
        assert all(
            isinstance(hint, SearchSuggestion) for hint in enhanced_result.query_hints
        )

    def test_query_hints_for_concept_result(self):
        """Test query hints generation for ConceptResult."""
        concept_result = ConceptResult(
            concept="magnetic field",
            explanation="The magnetic field in tokamak physics...",
            detail_level=DetailLevel.ADVANCED,
            related_topics=["equilibrium", "transport"],
            nodes=[],
            physics_domains=["equilibrium"],
            query="magnetic field",
            search_mode=SearchMode.SEMANTIC,
            max_results=10,
            ids_filter=None,
        )

        enhanced_result = apply_query_hints(concept_result, max_hints=2)

        # Should have query hints populated
        assert hasattr(enhanced_result, "query_hints")
        assert isinstance(enhanced_result.query_hints, list)
        assert len(enhanced_result.query_hints) > 0

        # Should suggest related search terms
        suggestions = [hint.suggestion for hint in enhanced_result.query_hints]
        assert any("magnetic field" in suggestion for suggestion in suggestions)

    def test_query_hints_for_structure_result(self):
        """Test query hints generation for StructureResult."""
        structure_result = StructureResult(
            ids_name="transport",
            description="Transport data structure",
            structure={"model": 2, "coefficients": 3},
            sample_paths=["transport/model/turbulence"],
            max_depth=2,
            query="transport structure",
            search_mode=SearchMode.LEXICAL,
            max_results=None,
            ids_filter=None,
            physics_domains=["transport"],
        )

        enhanced_result = apply_query_hints(structure_result, max_hints=2)

        # Should have query hints populated
        assert hasattr(enhanced_result, "query_hints")
        assert isinstance(enhanced_result.query_hints, list)
        assert len(enhanced_result.query_hints) > 0

        # Should suggest IDS-specific search terms
        suggestions = [hint.suggestion for hint in enhanced_result.query_hints]
        assert any("transport" in suggestion for suggestion in suggestions)

    def test_generate_generic_query_hints(self):
        """Test the generic query hint generator directly."""
        # Test ConceptResult hints
        concept_result = ConceptResult(
            concept="density",
            explanation="Test explanation",
            detail_level=DetailLevel.BASIC,
            related_topics=[],
            nodes=[],
            physics_domains=[],
            query="density",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
        )

        hints = generate_generic_query_hints(concept_result)
        assert len(hints) > 0
        assert all(isinstance(hint, SearchSuggestion) for hint in hints)

        # Should suggest concept-related searches
        suggestions = [hint.suggestion for hint in hints]
        assert any("density" in suggestion for suggestion in suggestions)

        # Test StructureResult hints
        structure_result = StructureResult(
            ids_name="core_profiles",
            description="Test description",
            structure={},
            sample_paths=[],
            max_depth=0,
            query="core_profiles",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
            physics_domains=[],
        )

        hints = generate_generic_query_hints(structure_result)
        assert len(hints) > 0
        assert all(isinstance(hint, SearchSuggestion) for hint in hints)

        # Should suggest IDS-related searches
        suggestions = [hint.suggestion for hint in hints]
        assert any("core_profiles" in suggestion for suggestion in suggestions)


class TestHintFieldInheritance:
    """Test that all ToolResult subclasses have the hint fields."""

    def test_search_result_has_hint_fields(self):
        """Test that SearchResult has hint fields."""
        result = SearchResult(
            hits=[],
            query="test",
            search_mode=SearchMode.AUTO,
        )

        assert hasattr(result, "query_hints")
        assert hasattr(result, "tool_hints")
        assert isinstance(result.query_hints, list)
        assert isinstance(result.tool_hints, list)

    def test_concept_result_has_hint_fields(self):
        """Test that ConceptResult has hint fields."""
        result = ConceptResult(
            concept="test",
            explanation="test explanation",
            detail_level=DetailLevel.BASIC,
            related_topics=[],
            nodes=[],
            physics_domains=[],
            query="test",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
        )

        assert hasattr(result, "query_hints")
        assert hasattr(result, "tool_hints")
        assert isinstance(result.query_hints, list)
        assert isinstance(result.tool_hints, list)

    def test_structure_result_has_hint_fields(self):
        """Test that StructureResult has hint fields."""
        result = StructureResult(
            ids_name="test",
            description="test description",
            structure={},
            sample_paths=[],
            max_depth=0,
            query="test",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
            physics_domains=[],
        )

        assert hasattr(result, "query_hints")
        assert hasattr(result, "tool_hints")
        assert isinstance(result.query_hints, list)
        assert isinstance(result.tool_hints, list)

    def test_export_result_has_hint_fields(self):
        """Test that export results have hint fields."""
        result = IDSExport(
            ids_names=["test"],
            include_physics=True,
            include_relationships=True,
            data={},
            metadata={},
            query="test",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
        )

        assert hasattr(result, "query_hints")
        assert hasattr(result, "tool_hints")
        assert isinstance(result.query_hints, list)
        assert isinstance(result.tool_hints, list)


class TestDecoratorIntegration:
    """Test that decorators work properly on actual tool methods."""

    @pytest.fixture
    def mock_tool_with_decorators(self):
        """Create a mock tool with hint decorators applied."""
        from imas_mcp.search.decorators.query_hints import query_hints
        from imas_mcp.search.decorators.tool_hints import tool_hints

        class MockTool:
            @query_hints(max_hints=2)
            @tool_hints(max_hints=2)
            async def mock_search_method(self) -> SearchResult:
                return SearchResult(
                    hits=[
                        SearchHit(
                            path="test/path",
                            score=0.8,
                            rank=0,
                            search_mode=SearchMode.SEMANTIC,
                            documentation="Test documentation",
                            ids_name="test_ids",
                        )
                    ],
                    query="test query",
                    search_mode=SearchMode.SEMANTIC,
                )

            @tool_hints(max_hints=2)
            async def mock_concept_method(self) -> ConceptResult:
                return ConceptResult(
                    concept="test concept",
                    explanation="test explanation",
                    detail_level=DetailLevel.BASIC,
                    related_topics=[],
                    nodes=[],
                    physics_domains=[],
                    query="test",
                    search_mode=SearchMode.AUTO,
                    max_results=None,
                    ids_filter=None,
                )

        return MockTool()

    @pytest.mark.asyncio
    async def test_decorated_search_method(self, mock_tool_with_decorators):
        """Test that decorated search method populates hints."""
        result = await mock_tool_with_decorators.mock_search_method()

        # Decorators should have populated the hint fields
        assert hasattr(result, "query_hints")
        assert hasattr(result, "tool_hints")

        # The result object should be valid
        assert result.query == "test query"
        assert len(result.hits) == 1

    @pytest.mark.asyncio
    async def test_decorated_concept_method(self, mock_tool_with_decorators):
        """Test that decorated concept method populates hints."""
        result = await mock_tool_with_decorators.mock_concept_method()

        # Tool hints decorator should have populated hints
        assert hasattr(result, "tool_hints")

        # The result object should be valid
        assert result.concept == "test concept"
        assert result.detail_level == DetailLevel.BASIC
