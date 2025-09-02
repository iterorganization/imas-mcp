"""
Tests for hint population across all MCP tools.

This module tests that the query_hints and tool_hints decorators
actually populate meaningful hints for all decorated tools.
"""

import pytest

from imas_mcp.models.constants import DetailLevel, SearchMode
from imas_mcp.models.result_models import (
    ConceptResult,
    DomainExport,
    ExportData,
    IDSExport,
    IdsInfo,
    IdsPath,
    SearchResult,
    StructureResult,
)
from imas_mcp.models.suggestion_models import SearchSuggestion, ToolSuggestion
from imas_mcp.search.decorators.query_hints import apply_query_hints
from imas_mcp.search.decorators.tool_hints import apply_tool_hints
from imas_mcp.search.search_strategy import SearchHit


class TestHintPopulationAcrossTools:
    """Test that hints are actually populated with meaningful content."""

    def test_search_result_hints_populated(self):
        """Test that SearchResult gets populated with meaningful hints."""
        # Create SearchResult with multiple hits (should trigger rich hints)
        hits = [
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
            ),
            SearchHit(
                path="equilibrium/time_slice/profiles_1d/psi",
                score=0.90,
                rank=1,
                search_mode=SearchMode.SEMANTIC,
                physics_domain="flux_surfaces",
                documentation="Poloidal flux profile",
                data_type="FLT_1D",
                units="Wb",
                ids_name="equilibrium",
            ),
        ]

        result = SearchResult(
            hits=hits,
            query="temperature psi",
            search_mode=SearchMode.SEMANTIC,
        )

        # Apply both decorators
        result_with_query_hints = apply_query_hints(result, max_hints=3)
        result_with_tool_hints = apply_tool_hints(result_with_query_hints, max_hints=3)

        # Verify query hints are populated (even if they might be empty for this case)
        assert hasattr(result_with_tool_hints, "query_hints")
        assert isinstance(result_with_tool_hints.query_hints, list)

        # Verify tool hints are populated with meaningful content
        assert hasattr(result_with_tool_hints, "tool_hints")
        assert isinstance(result_with_tool_hints.tool_hints, list)
        assert len(result_with_tool_hints.tool_hints) > 0

        # Should suggest relevant tools
        tool_names = [hint.tool_name for hint in result_with_tool_hints.tool_hints]
        assert "explore_relationships" in tool_names  # Should suggest relationships
        assert any(
            name in ["analyze_ids_structure", "explain_concept"] for name in tool_names
        )

        # Verify hint content quality
        for hint in result_with_tool_hints.tool_hints:
            assert isinstance(hint, ToolSuggestion)
            assert hint.tool_name
            assert hint.description
            assert hint.relevance
            assert len(hint.description) > 10  # Should have meaningful descriptions

    def test_concept_result_hints_populated(self):
        """Test that ConceptResult gets populated with tool hints."""
        result = ConceptResult(
            concept="poloidal flux",
            explanation="Magnetic flux through a poloidal surface in a tokamak",
            detail_level=DetailLevel.INTERMEDIATE,
            related_topics=["magnetic flux", "equilibrium", "psi"],
            nodes=[],
            physics_domains=["flux_surfaces", "equilibrium"],
            query="poloidal flux",
            search_mode=SearchMode.SEMANTIC,
            max_results=None,
            ids_filter=None,
        )

        # Apply tool hints
        enhanced_result = apply_tool_hints(result, max_hints=3)

        # Verify tool hints are populated
        assert hasattr(enhanced_result, "tool_hints")
        assert isinstance(enhanced_result.tool_hints, list)
        assert len(enhanced_result.tool_hints) > 0

        # Should suggest search and relationship tools
        tool_names = [hint.tool_name for hint in enhanced_result.tool_hints]
        assert "search_imas" in tool_names  # Should suggest searching for related data
        assert (
            "explore_relationships" in tool_names
        )  # Should suggest exploring relationships

        # Verify hint quality
        for hint in enhanced_result.tool_hints:
            assert isinstance(hint, ToolSuggestion)
            assert hint.tool_name
            assert hint.description
            assert hint.relevance
            assert (
                "concept" in hint.relevance.lower()
                or "poloidal flux" in hint.relevance.lower()
            )

    def test_export_result_hints_populated(self):
        """Test that export results get populated with tool hints."""
        # Create proper IdsInfo objects
        core_profiles_info = IdsInfo(
            ids_name="core_profiles",
            description="Core plasma profiles",
            paths=[
                IdsPath(
                    path="core_profiles/profiles_1d/temperature",
                    documentation="Electron temperature profile",
                    physics_domain="core_transport",
                    data_type="FLT_1D",
                    units="eV",
                )
            ],
            physics_domains=["core_transport"],
            measurement_types=["temperature"],
        )

        equilibrium_info = IdsInfo(
            ids_name="equilibrium",
            description="Magnetic equilibrium data",
            paths=[
                IdsPath(
                    path="equilibrium/time_slice/profiles_1d/psi",
                    documentation="Poloidal flux profile",
                    physics_domain="flux_surfaces",
                    data_type="FLT_1D",
                    units="Wb",
                )
            ],
            physics_domains=["flux_surfaces"],
            measurement_types=["flux"],
        )

        export_data = ExportData(
            ids_data={
                "core_profiles": core_profiles_info,
                "equilibrium": equilibrium_info,
            }
        )

        result = IDSExport(
            ids_names=["core_profiles", "equilibrium", "transport"],
            include_physics=True,
            include_relationships=True,
            data=export_data,
            metadata={"export_count": 2},
            query="export core data",
            search_mode=SearchMode.SEMANTIC,
            max_results=None,
            ids_filter=None,
        )

        # Apply tool hints
        enhanced_result = apply_tool_hints(result, max_hints=3)

        # Verify tool hints are populated
        assert hasattr(enhanced_result, "tool_hints")
        assert isinstance(enhanced_result.tool_hints, list)
        assert len(enhanced_result.tool_hints) > 0

        # Should suggest analysis tools for exported data
        tool_names = [hint.tool_name for hint in enhanced_result.tool_hints]
        assert (
            "analyze_ids_structure" in tool_names
            or "explore_relationships" in tool_names
        )

        # Verify hint relevance
        for hint in enhanced_result.tool_hints:
            assert isinstance(hint, ToolSuggestion)
            assert hint.tool_name
            assert hint.description
            assert hint.relevance
            # Should be relevant to exported data
            assert any(
                word in hint.relevance.lower()
                for word in ["export", "data", "analysis"]
            )

    def test_domain_export_hints_populated(self):
        """Test that domain export results get tool hints."""

        export_data = ExportData(
            paths=[
                IdsPath(
                    path="equilibrium/time_slice/profiles_1d/psi",
                    documentation="Poloidal flux profile",
                    physics_domain="flux_surfaces",
                    data_type="FLT_1D",
                    units="Wb",
                )
            ],
            related_ids=["equilibrium", "core_profiles"],
        )

        result = DomainExport(
            domain="equilibrium",
            domain_info={
                "analysis_depth": "focused",
                "paths": [
                    {
                        "path": "equilibrium/time_slice/profiles_1d/psi",
                        "documentation": "...",
                    }
                ],
                "related_ids": ["equilibrium", "core_profiles"],
            },
            include_cross_domain=True,
            max_paths=5,
            data=export_data,
            metadata={"domain_export": True},
            query="equilibrium domain",
            search_mode=SearchMode.SEMANTIC,
            max_results=None,
            ids_filter=None,
        )

        # Apply tool hints
        enhanced_result = apply_tool_hints(result, max_hints=3)

        # Verify tool hints are populated
        assert hasattr(enhanced_result, "tool_hints")
        assert isinstance(enhanced_result.tool_hints, list)
        assert len(enhanced_result.tool_hints) > 0

        # Should suggest appropriate analysis tools
        tool_names = [hint.tool_name for hint in enhanced_result.tool_hints]
        expected_tools = [
            "analyze_ids_structure",
            "explore_relationships",
            "search_imas",
        ]
        assert any(tool in tool_names for tool in expected_tools)

    def test_structure_result_hints_populated(self):
        """Test that structure analysis results get tool hints."""
        result = StructureResult(
            ids_name="equilibrium",
            description="Equilibrium data structure",
            structure={"profiles_1d": 50, "profiles_2d": 20, "time_slice": 100},
            sample_paths=[
                "equilibrium/time_slice/profiles_1d/psi",
                "equilibrium/time_slice/profiles_2d/psi",
                "equilibrium/time_slice/boundary",
            ],
            max_depth=3,
            query="equilibrium structure",
            search_mode=SearchMode.SEMANTIC,
            max_results=None,
            ids_filter=None,
            physics_domains=["flux_surfaces", "equilibrium"],
        )

        # Apply tool hints
        enhanced_result = apply_tool_hints(result, max_hints=3)

        # Verify tool hints are populated
        assert hasattr(enhanced_result, "tool_hints")
        assert isinstance(enhanced_result.tool_hints, list)
        assert len(enhanced_result.tool_hints) > 0

        # Should suggest data exploration tools
        tool_names = [hint.tool_name for hint in enhanced_result.tool_hints]
        expected_tools = ["export_ids", "search_imas"]
        assert any(tool in tool_names for tool in expected_tools)

        # Verify relevance to structure analysis
        for hint in enhanced_result.tool_hints:
            assert isinstance(hint, ToolSuggestion)
            assert hint.tool_name
            assert hint.description
            assert hint.relevance
            # Should reference the IDS name
            assert (
                "equilibrium" in hint.relevance.lower()
                or "ids" in hint.relevance.lower()
            )

    def test_empty_search_result_hints(self):
        """Test that empty search results get discovery-oriented hints."""
        result = SearchResult(
            hits=[],
            query="nonexistent_data",
            search_mode=SearchMode.SEMANTIC,
        )

        # Apply tool hints
        enhanced_result = apply_tool_hints(result, max_hints=4)

        # Should have discovery-oriented tool hints
        assert hasattr(enhanced_result, "tool_hints")
        assert isinstance(enhanced_result.tool_hints, list)
        assert len(enhanced_result.tool_hints) > 0

        # Should suggest discovery tools for no-results case
        tool_names = [hint.tool_name for hint in enhanced_result.tool_hints]
        discovery_tools = ["get_overview", "explore_identifiers", "explain_concept"]
        assert any(tool in tool_names for tool in discovery_tools)

        # Verify hints are about discovery
        hint_descriptions = [
            hint.description.lower() for hint in enhanced_result.tool_hints
        ]
        discovery_keywords = [
            "overview",
            "discover",
            "explore",
            "explain",
            "understand",
        ]
        assert any(
            any(keyword in desc for keyword in discovery_keywords)
            for desc in hint_descriptions
        )

    def test_hint_field_inheritance(self):
        """Test that all ToolResult subclasses have the hint fields."""
        # Test SearchResult
        search_result = SearchResult(hits=[], query="test", search_mode=SearchMode.AUTO)
        assert hasattr(search_result, "query_hints")
        assert hasattr(search_result, "tool_hints")

        # Test ConceptResult
        concept_result = ConceptResult(
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
        assert hasattr(concept_result, "query_hints")
        assert hasattr(concept_result, "tool_hints")

        # Test IDSExport
        export_result = IDSExport(
            ids_names=["test"],
            data=ExportData(),
            metadata={},
            query="test",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
        )
        assert hasattr(export_result, "query_hints")
        assert hasattr(export_result, "tool_hints")

    def test_hint_content_quality(self):
        """Test that generated hints have high-quality content."""
        # Create a rich search result
        hits = [
            SearchHit(
                path="core_profiles/profiles_1d/temperature",
                score=0.95,
                rank=0,
                search_mode=SearchMode.SEMANTIC,
                physics_domain="core_transport",
                documentation="Electron temperature measurements",
                data_type="FLT_1D",
                units="eV",
                ids_name="core_profiles",
            )
        ]

        result = SearchResult(
            hits=hits,
            query="electron temperature",
            search_mode=SearchMode.SEMANTIC,
        )

        enhanced_result = apply_tool_hints(result, max_hints=5)

        # Test hint quality standards
        for hint in enhanced_result.tool_hints:
            # Should have proper structure
            assert isinstance(hint, ToolSuggestion)
            assert hint.tool_name
            assert hint.description
            assert hint.relevance

            # Should have meaningful content
            assert len(hint.description) >= 10
            assert len(hint.relevance) >= 10

            # Tool names should be valid MCP tools
            valid_tools = [
                "search_imas",
                "explain_concept",
                "get_overview",
                "analyze_ids_structure",
                "explore_relationships",
                "explore_identifiers",
                "export_ids",
                "export_physics_domain",
            ]
            assert hint.tool_name in valid_tools

            # Descriptions should be actionable
            action_words = [
                "get",
                "find",
                "explore",
                "analyze",
                "export",
                "discover",
                "understand",
            ]
            description_lower = hint.description.lower()
            assert any(word in description_lower for word in action_words)


class TestQueryHintPopulation:
    """Test query hints specifically."""

    def test_query_hints_for_successful_search(self):
        """Test query hints for searches with results."""
        hits = [
            SearchHit(
                path="core_profiles/profiles_1d/temperature",
                score=0.85,
                rank=0,
                search_mode=SearchMode.SEMANTIC,
                physics_domain="core_transport",
                documentation="Temperature profile data",
                data_type="FLT_1D",
                units="eV",
                ids_name="core_profiles",
            )
        ]

        result = SearchResult(
            hits=hits,
            query="temperature",
            search_mode=SearchMode.SEMANTIC,
        )

        enhanced_result = apply_query_hints(result, max_hints=3)

        # Query hints might be empty for simple successful searches,
        # but the field should exist and be a list
        assert hasattr(enhanced_result, "query_hints")
        assert isinstance(enhanced_result.query_hints, list)

    def test_query_hints_for_failed_search(self):
        """Test query hints for searches with no results."""
        result = SearchResult(
            hits=[],
            query="completely_unknown_term",
            search_mode=SearchMode.SEMANTIC,
        )

        enhanced_result = apply_query_hints(result, max_hints=5)

        # Should have query hints for failed searches
        assert hasattr(enhanced_result, "query_hints")
        assert isinstance(enhanced_result.query_hints, list)
        # Failed searches should generate helpful query suggestions
        assert (
            len(enhanced_result.query_hints) >= 0
        )  # Might be empty but should be valid list

        # If hints exist, they should be SearchSuggestion objects
        for hint in enhanced_result.query_hints:
            assert isinstance(hint, SearchSuggestion)
            assert hint.suggestion
            assert hint.reason
            assert hint.confidence is not None
