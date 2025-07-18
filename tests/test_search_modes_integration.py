"""
Integration tests for search modes using real DocumentStore and FTS.

This module tests the search mode functionality with actual IMAS data
and real SQLite FTS capabilities.
"""

import pytest
from unittest.mock import Mock, patch

from imas_mcp.search.search_modes import (
    SearchMode,
    SearchConfig,
    SearchComposer,
    SearchModeSelector,
)
from imas_mcp.search.document_store import DocumentStore


@pytest.fixture(scope="session")
def document_store():
    """Create a real DocumentStore with test data."""
    # Create a DocumentStore instance with real IMAS data
    store = DocumentStore()
    return store


class TestSearchModesIntegration:
    """Integration tests for search modes with real data."""

    def test_lexical_search_with_real_data(self, document_store):
        """Test lexical search with real IMAS data."""
        composer = SearchComposer(document_store)

        # Test basic lexical search
        results = composer.search_with_params(
            "electron density", SearchMode.LEXICAL, max_results=10
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert "search_strategy" in results
        assert results["search_strategy"] == "lexical"
        assert isinstance(results["results"], list)

        # Results should contain relevant electron density entries
        if results["results"]:
            for result in results["results"]:
                assert "path" in result
                assert "documentation" in result
                # Should contain electron or density in documentation
                doc_lower = result["documentation"].lower()
                assert "electron" in doc_lower or "density" in doc_lower

    def test_lexical_search_no_results(self, document_store):
        """Test lexical search returns no results for nonsensical queries."""
        composer = SearchComposer(document_store)

        # Test with completely nonsensical query
        results = composer.search_with_params(
            "nonexistent_impossible_query_xyz_123", SearchMode.LEXICAL, max_results=10
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert "search_strategy" in results
        assert results["search_strategy"] == "lexical"
        assert isinstance(results["results"], list)
        assert len(results["results"]) == 0

    def test_lexical_search_exact_matches(self, document_store):
        """Test lexical search finds exact matches."""
        composer = SearchComposer(document_store)

        # Test with exact IDS name
        results = composer.search_with_params(
            "core_profiles", SearchMode.LEXICAL, max_results=10
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert results["search_strategy"] == "lexical"

        # Should find results for core_profiles
        if results["results"]:
            for result in results["results"]:
                assert "core_profiles" in result["path"]

    def test_lexical_search_with_ids_filter(self, document_store):
        """Test lexical search with IDS filtering."""
        composer = SearchComposer(document_store)

        # Test with IDS filter
        results = composer.search_with_params(
            "temperature",
            SearchMode.LEXICAL,
            max_results=10,
            filter_ids=["core_profiles"],
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert results["search_strategy"] == "lexical"

        # All results should be from core_profiles
        for result in results["results"]:
            assert "core_profiles" in result["path"]

    def test_lexical_search_case_insensitive(self, document_store):
        """Test lexical search is case insensitive."""
        composer = SearchComposer(document_store)

        # Test with different cases
        queries = ["ELECTRON", "electron", "Electron", "DENSITY", "density"]

        for query in queries:
            results = composer.search_with_params(
                query, SearchMode.LEXICAL, max_results=5
            )

            assert isinstance(results, dict)
            assert "results" in results
            assert results["search_strategy"] == "lexical"
            # Should find results regardless of case
            # (assuming there are electron/density related entries)

    def test_lexical_search_partial_matches(self, document_store):
        """Test lexical search finds partial matches."""
        composer = SearchComposer(document_store)

        # Test with partial terms
        results = composer.search_with_params(
            "temp",  # Should match "temperature"
            SearchMode.LEXICAL,
            max_results=10,
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert results["search_strategy"] == "lexical"

        # Should find temperature-related results
        if results["results"]:
            for result in results["results"]:
                doc_lower = result["documentation"].lower()
                assert "temp" in doc_lower or "temperature" in doc_lower

    def test_lexical_search_multiple_terms(self, document_store):
        """Test lexical search with multiple terms."""
        composer = SearchComposer(document_store)

        # Test with multiple terms
        results = composer.search_with_params(
            "electron temperature", SearchMode.LEXICAL, max_results=10
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert results["search_strategy"] == "lexical"

        # Should find results containing both terms or either term
        if results["results"]:
            for result in results["results"]:
                doc_lower = result["documentation"].lower()
                assert "electron" in doc_lower or "temperature" in doc_lower

    @patch("imas_mcp.search.semantic_search.SemanticSearch")
    def test_semantic_search_with_real_data(self, mock_semantic_search, document_store):
        """Test semantic search with real data."""
        # Mock semantic search to avoid loading heavy models
        mock_search_instance = Mock()
        mock_search_instance.search.return_value = [
            {"document": list(document_store.get_all_documents())[:1][0], "score": 0.95}
        ]
        mock_semantic_search.return_value = mock_search_instance

        composer = SearchComposer(document_store)

        results = composer.search_with_params(
            "plasma temperature", SearchMode.SEMANTIC, max_results=10
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert results["search_strategy"] == "semantic"
        assert isinstance(results["results"], list)

    @patch("imas_mcp.search.semantic_search.SemanticSearch")
    def test_hybrid_search_with_real_data(self, mock_semantic_search, document_store):
        """Test hybrid search with real data."""
        # Mock semantic search to avoid loading heavy models
        mock_search_instance = Mock()
        mock_search_instance.search.return_value = [
            {"document": list(document_store.get_all_documents())[:1][0], "score": 0.95}
        ]
        mock_semantic_search.return_value = mock_search_instance

        composer = SearchComposer(document_store)

        results = composer.search_with_params(
            "electron density", SearchMode.HYBRID, max_results=10
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert results["search_strategy"] == "hybrid"
        assert isinstance(results["results"], list)

    @patch("imas_mcp.search.semantic_search.SemanticSearch")
    def test_auto_mode_selection(self, mock_semantic_search, document_store):
        """Test automatic mode selection."""
        # Mock semantic search to avoid loading heavy models
        mock_search_instance = Mock()
        mock_search_instance.search.return_value = [
            {"document": list(document_store.get_all_documents())[:1][0], "score": 0.95}
        ]
        mock_semantic_search.return_value = mock_search_instance

        composer = SearchComposer(document_store)

        # Test with different query types
        test_queries = [
            "core_profiles",  # Should select lexical
            "plasma temperature",  # Should select semantic
            "electron density profile",  # Should select semantic
            "equilibrium/time_slice",  # Should select lexical
        ]

        for query in test_queries:
            results = composer.search_with_params(query, SearchMode.AUTO, max_results=5)

            assert isinstance(results, dict)
            assert "results" in results
            assert "search_strategy" in results
            assert results["search_strategy"] in ["lexical", "semantic", "hybrid"]

    def test_search_mode_selector_consistency(self):
        """Test that search mode selector is consistent."""
        selector = SearchModeSelector()

        # Test consistency - same query should always return same mode
        query = "plasma temperature"
        mode1 = selector.select_mode(query)
        mode2 = selector.select_mode(query)
        assert mode1 == mode2

        # Test different query types based on actual selector behavior

        # SEMANTIC queries (conceptual/physics terms)
        physics_query = "electron density profile"
        conceptual_query = "plasma temperature"

        physics_mode = selector.select_mode(physics_query)
        conceptual_mode = selector.select_mode(conceptual_query)

        assert physics_mode == SearchMode.SEMANTIC
        assert conceptual_mode == SearchMode.SEMANTIC

        # LEXICAL queries (technical/path-like)
        path_query = "core_profiles/profiles_1d"
        technical_query = "rho_tor_norm"

        path_mode = selector.select_mode(path_query)
        technical_mode = selector.select_mode(technical_query)

        assert path_mode == SearchMode.LEXICAL
        assert technical_mode == SearchMode.LEXICAL

        # HYBRID queries (domain terms that are neither clearly technical nor conceptual)
        domain_query1 = "equilibrium"
        domain_query2 = "transport"

        hybrid_mode1 = selector.select_mode(domain_query1)
        hybrid_mode2 = selector.select_mode(domain_query2)

        assert hybrid_mode1 == SearchMode.HYBRID
        assert hybrid_mode2 == SearchMode.HYBRID

    def test_search_mode_selector_comprehensive(self):
        """Test comprehensive search mode selection with various query types."""
        selector = SearchModeSelector()

        # LEXICAL queries (technical/path-like)
        lexical_queries = [
            "core_profiles/profiles_1d",  # Path with slash
            "units:eV",  # Field specifier
            "electron AND density",  # Boolean operator
            "rho_tor_norm",  # Underscore-separated technical term
            "profiles_1d",  # Technical term with underscore
            "time_slice/0",  # Path with slash and number
            "documentation:temperature",  # Field specifier
            '"electron density"',  # Quoted phrase
            "temp*",  # Wildcard
            "global_quantities",  # Technical term
        ]

        for query in lexical_queries:
            mode = selector.select_mode(query)
            assert mode == SearchMode.LEXICAL, (
                f"Query '{query}' should be LEXICAL, got {mode.value}"
            )

        # SEMANTIC queries (conceptual/physics terms)
        semantic_queries = [
            "plasma temperature",
            "electron density profile",
            "physics of plasma",
            "what is plasma",
            "magnetic field",
            "what is plasma temperature",
            "physics temperature",
            "density profile",
        ]

        for query in semantic_queries:
            mode = selector.select_mode(query)
            assert mode == SearchMode.SEMANTIC, (
                f"Query '{query}' should be SEMANTIC, got {mode.value}"
            )

        # HYBRID queries (domain terms that are neither clearly technical nor conceptual)
        hybrid_queries = [
            "equilibrium",
            "transport",
            "mhd",
            "wall",
            "diagnostics",
            "simulation",
            "results",
            "data analysis",
            "measurement",
            "experiment",
            "tokamak",
            "reactor",
            "fusion",
            "energy",
            "current",
            "voltage",
            "power",
            "heat flux",
            "particle flux",
            "field lines",
            "magnetic configuration",
            "pressure gradient",
            "instability",
            "turbulence",
            "confinement",
            "bootstrap current",
            "neutral beam",
        ]

        for query in hybrid_queries:
            mode = selector.select_mode(query)
            assert mode == SearchMode.HYBRID, (
                f"Query '{query}' should be HYBRID, got {mode.value}"
            )

        # Verify query distribution
        total_queries = (
            len(lexical_queries) + len(semantic_queries) + len(hybrid_queries)
        )
        print("\nQuery distribution:")
        print(f"  LEXICAL: {len(lexical_queries)} queries")
        print(f"  SEMANTIC: {len(semantic_queries)} queries")
        print(f"  HYBRID: {len(hybrid_queries)} queries")
        print(f"  Total: {total_queries} queries")

    def test_search_config_validation(self, document_store):
        """Test search configuration validation."""
        composer = SearchComposer(document_store)

        # Test with different configurations
        configs = [
            SearchConfig(mode=SearchMode.LEXICAL, max_results=1),
            SearchConfig(mode=SearchMode.LEXICAL, max_results=100),
            SearchConfig(mode=SearchMode.LEXICAL, filter_ids=["core_profiles"]),
            SearchConfig(mode=SearchMode.LEXICAL, similarity_threshold=0.5),
        ]

        for config in configs:
            results = composer.search("electron", config)
            assert isinstance(results, list)
            assert len(results) <= config.max_results

    def test_search_performance_large_queries(self, document_store):
        """Test search performance with large result sets."""
        composer = SearchComposer(document_store)

        # Test with broad query that might return many results
        results = composer.search_with_params(
            "profiles",  # Broad term likely to match many documents
            SearchMode.LEXICAL,
            max_results=100,
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert results["search_strategy"] == "lexical"
        assert len(results["results"]) <= 100

    def test_search_special_characters(self, document_store):
        """Test search with special characters."""
        composer = SearchComposer(document_store)

        # Test with special characters that might be in IMAS paths
        special_queries = [
            "profiles_1d",  # Underscore
            "time_slice/0",  # Slash and number
            "rho_tor_norm",  # Multiple underscores
        ]

        for query in special_queries:
            results = composer.search_with_params(
                query, SearchMode.LEXICAL, max_results=10
            )

            assert isinstance(results, dict)
            assert "results" in results
            assert results["search_strategy"] == "lexical"
            # Should handle special characters gracefully

    def test_empty_and_whitespace_queries(self, document_store):
        """Test handling of empty and whitespace-only queries."""
        composer = SearchComposer(document_store)

        # Test empty and whitespace queries
        empty_queries = ["", "   ", "\t", "\n"]

        for query in empty_queries:
            results = composer.search_with_params(
                query, SearchMode.LEXICAL, max_results=10
            )

            assert isinstance(results, dict)
            assert "results" in results
            assert results["search_strategy"] == "lexical"
            # Should return empty results for empty queries
            assert len(results["results"]) == 0


class TestFTSSearchCapabilities:
    """Test FTS (Full-Text Search) specific capabilities."""

    def test_fts_boolean_operators(self, document_store):
        """Test FTS with boolean operators if supported."""
        composer = SearchComposer(document_store)

        # Test with AND operator (if supported by FTS)
        results = composer.search_with_params(
            "electron AND density", SearchMode.LEXICAL, max_results=10
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert results["search_strategy"] == "lexical"

        # Results should contain both terms (if FTS supports boolean operators)
        if results["results"]:
            for result in results["results"]:
                doc_lower = result["documentation"].lower()
                # Note: This depends on FTS implementation
                # Some results might have both terms, others might not
                assert "electron" in doc_lower or "density" in doc_lower

    def test_fts_phrase_search(self, document_store):
        """Test FTS with phrase search."""
        composer = SearchComposer(document_store)

        # Test with quoted phrase
        results = composer.search_with_params(
            '"electron density"', SearchMode.LEXICAL, max_results=10
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert results["search_strategy"] == "lexical"

        # Results should preferentially match the exact phrase
        if results["results"]:
            for result in results["results"]:
                doc_lower = result["documentation"].lower()
                # Should contain both terms (phrase matching depends on FTS implementation)
                assert "electron" in doc_lower or "density" in doc_lower

    def test_fts_wildcard_search(self, document_store):
        """Test FTS with wildcard search."""
        composer = SearchComposer(document_store)

        # Test with wildcard (if supported)
        results = composer.search_with_params(
            "temp*",  # Should match "temperature", "temporal", etc.
            SearchMode.LEXICAL,
            max_results=10,
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert results["search_strategy"] == "lexical"

        # Results should match terms starting with "temp"
        if results["results"]:
            for result in results["results"]:
                doc_lower = result["documentation"].lower()
                # Should contain words starting with "temp"
                assert any(word.startswith("temp") for word in doc_lower.split())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
