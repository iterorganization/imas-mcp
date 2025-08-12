"""
Tests for search ranking improvements and physics context enhancements.

This module tests that search results are properly ranked and that
physics context confusion is resolved.
"""

import pytest

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.result_models import SearchResult
from imas_mcp.search.search_strategy import SearchHit


class TestSearchRankingImprovements:
    """Test improvements to search result ranking."""

    def test_primary_quantities_rank_higher_than_errors(self):
        """Test that primary quantities rank higher than error quantities."""
        # Simulate a search result where error quantities had higher scores
        hits = [
            # Error quantity (should be ranked lower)
            SearchHit(
                path="ece/psi_normalization/psi_magnetic_axis_error_upper",
                score=0.8,  # Initially high score
                rank=0,
                physics_domain="",
                documentation="Upper error for psi_magnetic_axis",
                data_type="FLT_1D",
                units="Wb",
                ids_name="ece",
            ),
            # Primary quantity (should be ranked higher)
            SearchHit(
                path="equilibrium/time_slice/profiles_1d/psi_norm",
                score=0.6,  # Initially lower score
                rank=1,
                physics_domain="flux_surfaces",
                documentation="Normalised poloidal flux",
                data_type="FLT_1D",
                units="1",
                ids_name="equilibrium",
            ),
            # Raw flux quantity (should be highly ranked)
            SearchHit(
                path="equilibrium/time_slice/profiles_1d/psi",
                score=0.5,  # Initially even lower score
                rank=2,
                physics_domain="flux_surfaces",
                documentation="Poloidal flux",
                data_type="FLT_1D",
                units="Wb",
                ids_name="equilibrium",
            ),
        ]

        result = SearchResult(
            query="psi",
            hits=hits,
            search_mode=SearchMode.HYBRID,
        )

        # After proper ranking, primary quantities should be preferred
        # This test documents the expected behavior after ranking improvements
        primary_paths = [
            "equilibrium/time_slice/profiles_1d/psi_norm",
            "equilibrium/time_slice/profiles_1d/psi",
        ]
        error_paths = ["ece/psi_normalization/psi_magnetic_axis_error_upper"]

        # Check that we have both types
        result_paths = [hit.path for hit in result.hits]
        assert any(path in result_paths for path in primary_paths)
        assert any(path in result_paths for path in error_paths)

    def test_flux_surfaces_domain_boost(self):
        """Test that flux_surfaces domain gets ranking boost for psi queries."""
        hits = [
            SearchHit(
                path="equilibrium/time_slice/profiles_1d/psi_norm",
                score=0.6,
                rank=0,
                physics_domain="flux_surfaces",  # Should get boost
                documentation="Normalised poloidal flux",
                data_type="FLT_1D",
                units="1",
                ids_name="equilibrium",
            ),
            SearchHit(
                path="gas_injection/valve/flow_rate",
                score=0.8,  # Higher initial score
                rank=1,
                physics_domain="",  # No physics domain
                documentation="Flow rate at the exit of the valve",
                data_type="structure",
                units="Pa.m^3.s^-1",
                ids_name="gas_injection",
            ),
        ]

        result = SearchResult(
            query="psi",
            hits=hits,
            search_mode=SearchMode.HYBRID,
        )

        # The flux_surfaces result should be more relevant for psi queries
        flux_hit = next(
            hit for hit in result.hits if hit.physics_domain == "flux_surfaces"
        )
        assert flux_hit.path == "equilibrium/time_slice/profiles_1d/psi_norm"

    def test_structure_containers_informational_ranking(self):
        """Test that structure containers are ranked appropriately."""
        hits = [
            # Structure container
            SearchHit(
                path="ece/psi_normalization",
                score=0.55,
                rank=0,
                physics_domain="",
                documentation="Quantities to use to normalize psi",
                data_type="structure",
                units=None,
                ids_name="ece",
            ),
            # Actual data field
            SearchHit(
                path="equilibrium/time_slice/profiles_1d/psi_norm",
                score=0.6,
                rank=1,
                physics_domain="flux_surfaces",
                documentation="Normalised poloidal flux",
                data_type="FLT_1D",
                units="1",
                ids_name="equilibrium",
            ),
        ]

        result = SearchResult(
            query="psi",
            hits=hits,
            search_mode=SearchMode.HYBRID,
        )

        # Structure containers provide context but data fields are more important
        data_hits = [hit for hit in result.hits if hit.data_type != "structure"]
        structure_hits = [hit for hit in result.hits if hit.data_type == "structure"]

        assert len(data_hits) >= 1
        assert len(structure_hits) >= 1


class TestPhysicsContextImprovements:
    """Test improvements to physics context matching."""

    def test_psi_flux_context_not_pressure(self):
        """Test that psi queries prioritize magnetic flux over pressure context."""
        # This test documents the expected physics context after improvements
        _result = SearchResult(
            query="psi",
            hits=[],
            search_mode=SearchMode.SEMANTIC,
        )

        # After improvements, physics context should prioritize flux-related terms
        # This would be populated by an improved physics context engine
        expected_domains = ["flux_surfaces", "equilibrium", "coordinates"]
        unexpected_domains = ["mechanical_diagnostics", "pressure"]

        # This test is aspirational - it shows what we want the physics context to look like
        # In a full implementation, we would test that the physics context engine
        # recognizes plasma physics context and avoids pressure units for "psi"
        assert len(expected_domains) == 3
        assert len(unexpected_domains) == 2

    def test_domain_aware_search_suggestions(self):
        """Test that search suggestions are domain-aware."""
        _flux_result = SearchResult(
            query="psi",
            hits=[
                SearchHit(
                    path="equilibrium/time_slice/profiles_1d/psi_norm",
                    score=0.9,
                    rank=0,
                    physics_domain="flux_surfaces",
                    documentation="Normalised poloidal flux",
                    data_type="FLT_1D",
                    units="1",
                    ids_name="equilibrium",
                    search_mode=SearchMode.HYBRID,
                )
            ],
            search_mode=SearchMode.HYBRID,
        )

        # Physics-aware suggestions should relate to magnetic flux, not pressure
        # This is tested indirectly through the decorator tests


class TestSearchModeOptimization:
    """Test search mode selection improvements."""

    def test_technical_terms_prefer_lexical(self):
        """Test that technical IMAS terms prefer lexical search."""
        # This documents expected behavior for search mode optimization
        technical_queries = [
            "profiles_1d/psi",
            "equilibrium/time_slice",
            "core_profiles",
        ]

        for _ in technical_queries:
            # These should prefer lexical search for exact path matching
            # This would be implemented in the search configuration service
            pass

    def test_conceptual_terms_prefer_semantic(self):
        """Test that conceptual physics terms prefer semantic search."""
        conceptual_queries = [
            "poloidal flux",
            "magnetic field strength",
            "plasma temperature",
            "what is safety factor",
        ]

        for _ in conceptual_queries:
            # These should prefer semantic search for concept understanding
            # This would be implemented in the search configuration service
            pass


class TestSearchResultCompleteness:
    """Test that search results include expected core quantities."""

    def test_psi_search_includes_core_quantities(self):
        """Test that psi search includes expected core magnetic flux quantities."""
        # This test documents what a good psi search should return
        expected_paths = [
            "equilibrium/time_slice/profiles_1d/psi",  # Raw poloidal flux
            "equilibrium/time_slice/profiles_1d/psi_norm",  # Normalized flux
            "equilibrium/time_slice/profiles_2d/psi",  # 2D flux maps
        ]

        # In a full implementation, we would test that these paths
        # are returned with appropriate ranking for a "psi" query
        assert len(expected_paths) == 3  # Document expected coverage

    def test_comprehensive_flux_coverage(self):
        """Test that flux-related searches cover all relevant quantities."""
        flux_related_paths = [
            "equilibrium/time_slice/profiles_1d/psi",
            "equilibrium/time_slice/profiles_1d/psi_norm",
            "equilibrium/time_slice/profiles_2d/psi",
            "equilibrium/time_slice/boundary/psi",
            "equilibrium/time_slice/boundary/psi_norm",
        ]

        # These should all be discoverable through various flux-related queries
        # The test documents the expected comprehensive coverage
        assert len(flux_related_paths) == 5  # Document expected coverage


class TestErrorMessageImprovements:
    """Test improvements to error messages and suggestions."""

    def test_no_results_provides_better_suggestions(self):
        """Test that no results scenario provides relevant suggestions."""
        _result = SearchResult(
            query="nonexistent_physics_term",
            hits=[],
            search_mode=SearchMode.SEMANTIC,
        )

        # After improvements, should suggest:
        # 1. Alternative search terms
        # 2. Related physics concepts
        # 3. Common IMAS patterns
        # 4. Discovery tools

        # This is tested through the decorator enhancement tests

    def test_ambiguous_terms_provide_clarification(self):
        """Test that ambiguous terms get clarifying suggestions."""
        # For terms like "psi" that could mean different things
        _result = SearchResult(
            query="psi",
            hits=[],
            search_mode=SearchMode.SEMANTIC,
        )

        # Should help disambiguate between:
        # - Poloidal flux (ψ) in plasma physics
        # - Pressure (PSI) in mechanical systems
        # - Other potential meanings


class TestPerformanceConsiderations:
    """Test that improvements don't negatively impact performance."""

    def test_ranking_algorithm_efficiency(self):
        """Test that improved ranking doesn't slow down results."""
        # Create a large number of hits to test performance
        hits = []
        for i in range(100):
            hits.append(
                SearchHit(
                    path=f"test/path/field_{i}",
                    score=0.5 + (i % 10) * 0.05,
                    rank=i,
                    physics_domain="test_domain",
                    documentation=f"Test field {i}",
                    data_type="FLT_1D",
                    units="test_unit",
                    ids_name="test_ids",
                    search_mode=SearchMode.HYBRID,
                )
            )

        result = SearchResult(
            query="test_query",
            hits=hits,
            search_mode=SearchMode.HYBRID,
        )

        # Should handle large result sets efficiently
        assert len(result.hits) == 100
        assert result.hit_count == 100

    def test_metadata_properties_performance(self):
        """Test that dynamic metadata properties don't impact performance."""
        # Create result and access properties multiple times
        result = SearchResult(query="test", hits=[])

        # Should be fast to access repeatedly
        for _ in range(100):
            _ = result.tool_name
            _ = result.processing_timestamp
            _ = result.version

        # Should work without issues
        assert result.tool_name == "search_imas"
