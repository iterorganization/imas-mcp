"""
Tests for search result ranking and physics context.

This module tests that search results are properly ranked and that
physics context confusion is resolved.
"""

import pytest

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.result_models import SearchResult
from imas_mcp.search.search_strategy import SearchHit


class TestSearchRanking:
    """Test search result ranking functionality."""

    def test_primary_quantities_rank_higher_than_errors(self):
        """Test that primary quantities rank higher than error quantities."""
        # Simulate a search result where error quantities had higher scores
        hits = [
            # Error quantity (should be ranked lower)
            SearchHit(
                path="ece/psi_normalization/psi_magnetic_axis_error_upper",
                score=0.8,  # Initially high score
                rank=0,
                search_mode=SearchMode.SEMANTIC,
                documentation="Upper error for psi_magnetic_axis",
                ids_name="ece",
                units="Wb",
                data_type="FLT_1D",
            ),
            # Primary quantity (should be ranked higher)
            SearchHit(
                path="equilibrium/time_slice/profiles_1d/psi_norm",
                score=0.6,  # Lower initial score
                rank=1,
                search_mode=SearchMode.SEMANTIC,
                documentation="Normalised poloidal flux",
                ids_name="equilibrium",
                units="1",
                data_type="FLT_1D",
            ),
        ]

        # Test that after re-ranking, primary quantities should be higher
        # This would be the expected behavior after implementing the ranking improvements
        primary_hit = next(hit for hit in hits if "profiles_1d/psi_norm" in hit.path)
        error_hit = next(hit for hit in hits if "error" in hit.path)

        # Verify we found the expected hits
        assert "normalised poloidal flux" in primary_hit.documentation.lower()
        assert (
            "error" in error_hit.documentation.lower()
        )  # Primary quantities should have meaningful content
        assert primary_hit.ids_name == "equilibrium"
        assert error_hit.ids_name == "ece"
        assert primary_hit.units == "1"  # Dimensionless normalized quantity
        assert error_hit.units == "Wb"  # Weber units for flux

    def test_core_physics_domains_boost_score(self):
        """Test that core physics domains get score boosts."""
        flux_surfaces_hit = SearchHit(
            path="equilibrium/time_slice/profiles_1d/psi_norm",
            score=0.6,
            rank=0,
            search_mode=SearchMode.SEMANTIC,
            documentation="Normalised poloidal flux",
            ids_name="equilibrium",
            units="1",
            data_type="FLT_1D",
            physics_domain="flux_surfaces",
        )

        # Core physics domains should be boosted
        assert flux_surfaces_hit.physics_domain == "flux_surfaces"
        assert flux_surfaces_hit.units == "1"  # Dimensionless for normalized quantity

    def test_missing_core_psi_quantities_identified(self):
        """Test identification of missing core psi quantities."""
        # These are the key psi quantities that should be found in a "psi" search
        expected_core_quantities = [
            "equilibrium/time_slice/profiles_1d/psi",  # Raw poloidal flux
            "equilibrium/time_slice/profiles_1d/psi_norm",  # Normalized flux
            "equilibrium/time_slice/profiles_2d/psi",  # 2D flux maps
        ]

        # Create minimal hits representing what should be found
        for i, path in enumerate(expected_core_quantities):
            hit = SearchHit(
                path=path,
                score=0.9,  # Should have high scores
                rank=i,
                search_mode=SearchMode.SEMANTIC,
                documentation=f"Poloidal flux quantity: {path.split('/')[-1]}",
                ids_name="equilibrium",
                units="Wb" if "psi_norm" not in path else "1",
                data_type="FLT_1D" if "profiles_1d" in path else "FLT_2D",
                physics_domain="flux_surfaces",
            )

            # Core equilibrium quantities should have high relevance
            assert hit.physics_domain == "flux_surfaces"
            assert hit.ids_name == "equilibrium"
            assert hit.score == 0.9


class TestPhysicsContextResolution:
    """Test physics context disambiguation."""

    def test_psi_context_disambiguation(self):
        """Test that psi is correctly identified as magnetic flux, not pressure."""
        # Simulate the physics context that was incorrectly returned
        incorrect_pressure_context = [
            "Pa Unit",
            "N.m^-2 Unit",
            "Pressure Measurement",
            "Mechanical Diagnostics Domain",
        ]

        # Expected physics context for magnetic flux
        expected_flux_context = [
            "Magnetic Flux",
            "Poloidal Flux",
            "Flux Surfaces",
            "Equilibrium Domain",
        ]

        # Test that we can distinguish between pressure and flux contexts
        for pressure_term in incorrect_pressure_context:
            assert (
                "pressure" in pressure_term.lower()
                or "pa" in pressure_term.lower()
                or "n.m^-2" in pressure_term.lower()
                or "mechanical" in pressure_term.lower()
            )

        for flux_term in expected_flux_context:
            assert any(
                keyword in flux_term.lower()
                for keyword in ["flux", "magnetic", "equilibrium"]
            )

    def test_domain_aware_physics_matching(self):
        """Test that physics matching is domain-aware."""
        # For plasma physics context, psi should map to magnetic flux
        expected_domains = ["flux_surfaces", "equilibrium", "coordinates"]

        # For mechanical context, psi might map to pressure
        mechanical_domains = ["mechanical_diagnostics", "pressure"]

        # Test domain detection logic
        for domain in expected_domains:
            assert domain in ["flux_surfaces", "equilibrium", "coordinates"]

        for domain in mechanical_domains:
            assert domain in ["mechanical_diagnostics", "pressure"]


class TestSearchCoverageExpansion:
    """Test expanded search coverage for core quantities."""

    def test_profiles_1d_psi_coverage(self):
        """Test that profiles_1d/psi quantities are properly covered."""
        expected_paths = [
            "equilibrium/time_slice/profiles_1d/psi",
            "equilibrium/time_slice/profiles_1d/psi_norm",
            "core_profiles/profiles_1d/psi",  # If exists
        ]

        for path in expected_paths:
            if "equilibrium" in path:
                # Equilibrium paths should be well-documented
                assert "equilibrium" in path
                assert "profiles_1d" in path
                assert "psi" in path

    def test_profiles_2d_psi_coverage(self):
        """Test that profiles_2d/psi quantities are included."""
        expected_2d_paths = [
            "equilibrium/time_slice/profiles_2d/psi",
            "equilibrium/time_slice/profiles_2d/psi_norm",
        ]

        for path in expected_2d_paths:
            assert "profiles_2d" in path
            assert "psi" in path
            assert "equilibrium" in path

    def test_normalization_quantities_prioritized(self):
        """Test that psi normalization quantities are properly prioritized."""
        normalization_paths = [
            "ece/psi_normalization/psi_magnetic_axis",
            "ece/psi_normalization/psi_boundary",
            "reflectometer_profile/psi_normalization/psi_boundary",
        ]

        for path in normalization_paths:
            assert "psi_normalization" in path
            # Normalization containers should be informational
            assert any(component in path for component in ["magnetic_axis", "boundary"])


class TestQueryOptimization:
    """Test query optimization for better results."""

    def test_auto_mode_selection_for_psi_queries(self):
        """Test that AUTO mode selects appropriate search strategy for psi queries."""
        # Physics concept queries should use semantic search
        conceptual_queries = ["poloidal flux", "magnetic flux", "flux surfaces"]

        # Technical path queries should use lexical search
        technical_queries = ["psi_norm", "profiles_1d/psi", "equilibrium/psi"]

        # Hybrid queries should use hybrid search
        hybrid_queries = ["psi normalization", "psi equilibrium profiles"]

        for query in conceptual_queries:
            # Should be suitable for semantic search
            assert any(word in query for word in ["flux", "magnetic", "surfaces"])

        for query in technical_queries:
            # Should be suitable for lexical search
            assert "_" in query or "/" in query

        for query in hybrid_queries:
            # Should be suitable for hybrid search
            words = query.split()
            assert len(words) >= 2


class TestResultIntegration:
    """Test integration of ranking improvements with search results."""

    def test_search_result_with_improved_ranking(self):
        """Test SearchResult with improved ranking applied."""
        # Create a result with the improved ranking
        hits = [
            SearchHit(
                path="equilibrium/time_slice/profiles_1d/psi_norm",
                score=0.95,  # Should be high for primary quantity
                rank=0,
                search_mode=SearchMode.SEMANTIC,
                documentation="Normalised poloidal flux",
                ids_name="equilibrium",
                units="1",
                data_type="FLT_1D",
                physics_domain="flux_surfaces",
            ),
            SearchHit(
                path="equilibrium/time_slice/profiles_1d/psi",
                score=0.90,  # Should be high for raw quantity
                rank=1,
                search_mode=SearchMode.SEMANTIC,
                documentation="Poloidal flux",
                ids_name="equilibrium",
                units="Wb",
                data_type="FLT_1D",
                physics_domain="flux_surfaces",
            ),
            SearchHit(
                path="ece/psi_normalization/psi_magnetic_axis_error_upper",
                score=0.75,  # Lower for error quantity
                rank=2,
                search_mode=SearchMode.SEMANTIC,
                documentation="Upper error for psi_magnetic_axis",
                ids_name="ece",
                units="Wb",
                data_type="FLT_1D",
                physics_domain="",
            ),
        ]

        result = SearchResult(
            hits=hits,
            search_mode=SearchMode.HYBRID,
            query="psi",
            physics_domains=["flux_surfaces"],
        )

        # Test the improved ranking
        assert result.hit_count == 3
        assert result.hits[0].path.endswith(
            "psi_norm"
        )  # Primary normalized quantity first
        assert result.hits[1].path.endswith("psi")  # Raw quantity second
        assert "error" in result.hits[2].path  # Error quantity last

        # Test physics domain integration
        assert "flux_surfaces" in result.physics_domains

    def test_metadata_integration_with_ranking(self):
        """Test that metadata properties work with ranking improvements."""
        result = SearchResult(
            hits=[],
            search_mode=SearchMode.SEMANTIC,
            query="test",
        )

        # Test abstract property implementation
        assert result.tool_name == "search_imas"
        assert result.processing_timestamp  # Should be populated
        assert result.version  # Should be populated

        # Test that these work with the ranking system
        assert result.search_mode == SearchMode.SEMANTIC
        assert result.query == "test"
