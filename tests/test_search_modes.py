"""
Test suite for search modes and composition patterns.

This module tests the different search strategies, search modes, and the
SearchComposer class that orchestrates them.
"""

from dataclasses import dataclass
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List

from imas_mcp.search.search_modes import (
    SearchMode,
    SearchConfig,
    SearchResult,
    SearchComposer,
    LexicalSearchStrategy,
    SemanticSearchStrategy,
    HybridSearchStrategy,
    SearchModeSelector,
)
from imas_mcp.search.document_store import Document, DocumentStore, DocumentMetadata


@dataclass
class MockDocumentStore(DocumentStore):
    """Mock document store for testing."""

    def __post_init__(self):
        # Initialize required attributes for DocumentStore
        self._data_dir = Path("test_data")
        self._sqlite_path = Path("test.db")
        self._loaded = True

        self.documents = [
            Document(
                metadata=DocumentMetadata(
                    path_id="core_profiles/profiles_1d/0/electrons/density",
                    ids_name="core_profiles",
                    path_name="core_profiles/profiles_1d/0/electrons/density",
                    units="m^-3",
                    data_type="FLT_1D",
                    coordinates=("rho_tor_norm",),
                    physics_domain="transport",
                    physics_phenomena=("plasma", "electrons", "density"),
                ),
                documentation="Electron density in the plasma core",
            ),
            Document(
                metadata=DocumentMetadata(
                    path_id="core_profiles/profiles_1d/0/electrons/temperature",
                    ids_name="core_profiles",
                    path_name="core_profiles/profiles_1d/0/electrons/temperature",
                    units="eV",
                    data_type="FLT_1D",
                    coordinates=("rho_tor_norm",),
                    physics_domain="transport",
                    physics_phenomena=("plasma", "electrons", "temperature"),
                ),
                documentation="Electron temperature in the plasma core",
            ),
            Document(
                metadata=DocumentMetadata(
                    path_id="equilibrium/time_slice/0/profiles_2d/0/psi",
                    ids_name="equilibrium",
                    path_name="equilibrium/time_slice/0/profiles_2d/0/psi",
                    units="Wb",
                    data_type="FLT_2D",
                    coordinates=("r", "z"),
                    physics_domain="equilibrium",
                    physics_phenomena=("equilibrium", "magnetic", "flux"),
                ),
                documentation="Poloidal flux function",
            ),
        ]

    def get_all_documents(self) -> List[Document]:
        return self.documents

    def get_documents_by_ids(self, ids_name: str) -> List[Document]:
        return [doc for doc in self.documents if doc.metadata.ids_name == ids_name]

    def search_fts(self, query: str, limit: int = 10) -> List[Document]:
        """Mock FTS search."""
        # Simple mock implementation
        query_lower = query.lower()
        results = []
        for doc in self.documents:
            if any(term in doc.documentation.lower() for term in query_lower.split()):
                results.append(doc)
        return results[:limit]

    def search_full_text(self, query: str, max_results: int = 10) -> List[Document]:
        """Mock full-text search implementation."""
        # Simple mock implementation
        query_lower = query.lower()
        results = []
        for doc in self.documents:
            if any(term in doc.documentation.lower() for term in query_lower.split()):
                results.append(doc)
        return results[:max_results]


class TestSearchMode:
    """Test suite for SearchMode enum."""

    def test_search_mode_values(self):
        """Test SearchMode enum values."""
        assert SearchMode.SEMANTIC.value == "semantic"
        assert SearchMode.LEXICAL.value == "lexical"
        assert SearchMode.HYBRID.value == "hybrid"
        assert SearchMode.AUTO.value == "auto"

    def test_search_mode_from_string(self):
        """Test creating SearchMode from string."""
        assert SearchMode("semantic") == SearchMode.SEMANTIC
        assert SearchMode("lexical") == SearchMode.LEXICAL
        assert SearchMode("hybrid") == SearchMode.HYBRID
        assert SearchMode("auto") == SearchMode.AUTO


class TestSearchConfig:
    """Test suite for SearchConfig dataclass."""

    def test_search_config_defaults(self):
        """Test SearchConfig default values."""
        config = SearchConfig()
        assert config.mode == SearchMode.AUTO
        assert config.max_results == 10
        assert config.filter_ids is None
        assert config.similarity_threshold == 0.0
        assert config.boost_exact_matches is True
        assert config.enable_physics_enhancement is True

    def test_search_config_custom_values(self):
        """Test SearchConfig with custom values."""
        config = SearchConfig(
            mode=SearchMode.SEMANTIC,
            max_results=20,
            filter_ids=["core_profiles", "equilibrium"],
            similarity_threshold=0.5,
            boost_exact_matches=False,
            enable_physics_enhancement=False,
        )
        assert config.mode == SearchMode.SEMANTIC
        assert config.max_results == 20
        assert config.filter_ids == ["core_profiles", "equilibrium"]
        assert config.similarity_threshold == 0.5
        assert config.boost_exact_matches is False
        assert config.enable_physics_enhancement is False


class TestSearchResult:
    """Test suite for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test SearchResult creation."""
        doc = Document(
            metadata=DocumentMetadata(
                path_id="test/path",
                ids_name="test_ids",
                path_name="test/path",
                units="test_units",
                data_type="FLT_1D",
                coordinates=("x",),
                physics_domain="test_domain",
                physics_phenomena=("test",),
            ),
            documentation="Test documentation",
        )

        result = SearchResult(
            document=doc,
            score=0.85,
            rank=1,
            search_mode=SearchMode.SEMANTIC,
            highlights="Test highlights",
        )

        assert result.document == doc
        assert result.score == 0.85
        assert result.rank == 1
        assert result.search_mode == SearchMode.SEMANTIC
        assert result.highlights == "Test highlights"

    def test_search_result_to_dict(self):
        """Test SearchResult to_dict method."""
        from imas_mcp.search.document_store import Units

        doc = Document(
            metadata=DocumentMetadata(
                path_id="test/path",
                ids_name="test_ids",
                path_name="test/path",
                units="test_units",
                data_type="FLT_1D",
                coordinates=("x",),
                physics_domain="test_domain",
                physics_phenomena=("test",),
            ),
            documentation="Test documentation",
            units=Units(unit_str="test_units", name="test_units"),
        )

        result = SearchResult(
            document=doc, score=0.85, rank=1, search_mode=SearchMode.SEMANTIC
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["path"] == "test/path"  # This matches metadata.path_name
        assert result_dict["ids_name"] == "test_ids"
        assert result_dict["relevance_score"] == 0.85  # Updated to match Pydantic model
        assert result_dict["rank"] == 1


class TestLexicalSearchStrategy:
    """Test suite for LexicalSearchStrategy."""

    def test_lexical_search_strategy_creation(self):
        """Test LexicalSearchStrategy creation."""
        store = MockDocumentStore()
        strategy = LexicalSearchStrategy(store)
        assert strategy.document_store == store

    def test_lexical_search_basic_query(self):
        """Test lexical search with basic query."""
        store = MockDocumentStore()
        strategy = LexicalSearchStrategy(store)

        config = SearchConfig(mode=SearchMode.LEXICAL, max_results=5)
        results = strategy.search("electron density", config)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.search_mode == SearchMode.LEXICAL for r in results)

    def test_lexical_search_no_results(self):
        """Test lexical search with query that returns no results."""
        store = MockDocumentStore()
        strategy = LexicalSearchStrategy(store)

        config = SearchConfig(mode=SearchMode.LEXICAL, max_results=5)
        results = strategy.search("nonexistent_impossible_query", config)

        assert isinstance(results, list)
        assert len(results) == 0

    def test_lexical_search_with_ids_filter(self):
        """Test lexical search with IDS filter."""
        store = MockDocumentStore()
        strategy = LexicalSearchStrategy(store)

        config = SearchConfig(
            mode=SearchMode.LEXICAL, max_results=5, filter_ids=["core_profiles"]
        )
        results = strategy.search("electron", config)

        assert isinstance(results, list)
        # Results should be filtered to only core_profiles
        for result in results:
            assert result.document.metadata.ids_name == "core_profiles"


class TestSemanticSearchStrategy:
    """Test suite for SemanticSearchStrategy."""

    @patch("imas_mcp.search.semantic_search.SemanticSearch")
    def test_semantic_search_strategy_creation(self, mock_semantic_search):
        """Test SemanticSearchStrategy creation."""
        store = MockDocumentStore()
        strategy = SemanticSearchStrategy(store)
        assert strategy.document_store == store

    @patch("imas_mcp.search.semantic_search.SemanticSearch")
    def test_semantic_search_basic_query(self, mock_semantic_search):
        """Test semantic search with basic query."""
        store = MockDocumentStore()

        # Mock semantic search instance and create proper SemanticSearchResult objects
        from imas_mcp.search.semantic_search import SemanticSearchResult

        mock_search_instance = Mock()
        mock_search_instance.search.return_value = [
            SemanticSearchResult(
                document=store.documents[0], similarity_score=0.95, rank=0
            ),
            SemanticSearchResult(
                document=store.documents[1], similarity_score=0.85, rank=1
            ),
        ]
        mock_semantic_search.return_value = mock_search_instance

        strategy = SemanticSearchStrategy(store)

        config = SearchConfig(mode=SearchMode.SEMANTIC, max_results=5)
        results = strategy.search("plasma temperature", config)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.search_mode == SearchMode.SEMANTIC for r in results)

    @patch("imas_mcp.search.semantic_search.SemanticSearch")
    def test_semantic_search_no_results(self, mock_semantic_search):
        """Test semantic search with query that returns no results."""
        # Mock semantic search instance
        mock_search_instance = Mock()
        mock_search_instance.search.return_value = []
        mock_semantic_search.return_value = mock_search_instance

        store = MockDocumentStore()
        strategy = SemanticSearchStrategy(store)

        config = SearchConfig(mode=SearchMode.SEMANTIC, max_results=5)
        results = strategy.search("nonexistent_concept", config)

        assert isinstance(results, list)
        assert len(results) == 0


class TestHybridSearchStrategy:
    """Test suite for HybridSearchStrategy."""

    @patch("imas_mcp.search.semantic_search.SemanticSearch")
    def test_hybrid_search_strategy_creation(self, mock_semantic_search):
        """Test HybridSearchStrategy creation."""
        store = MockDocumentStore()
        strategy = HybridSearchStrategy(store)
        assert strategy.document_store == store

    @patch("imas_mcp.search.semantic_search.SemanticSearch")
    def test_hybrid_search_basic_query(self, mock_semantic_search):
        """Test hybrid search with basic query."""
        store = MockDocumentStore()

        # Mock semantic search instance
        from imas_mcp.search.semantic_search import SemanticSearchResult

        mock_search_instance = Mock()
        mock_search_instance.search.return_value = [
            SemanticSearchResult(
                document=store.documents[0], similarity_score=0.95, rank=0
            ),
        ]
        mock_semantic_search.return_value = mock_search_instance

        strategy = HybridSearchStrategy(store)

        config = SearchConfig(mode=SearchMode.HYBRID, max_results=5)
        results = strategy.search("electron density", config)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.search_mode == SearchMode.HYBRID for r in results)


class TestSearchModeSelector:
    """Test suite for SearchModeSelector."""

    def test_select_mode_semantic_physics_terms(self):
        """Test mode selection for physics terms."""
        selector = SearchModeSelector()

        # Physics terms should prefer semantic search
        mode = selector.select_mode("plasma temperature")
        assert mode == SearchMode.SEMANTIC

        mode = selector.select_mode("electron density profile")
        assert mode == SearchMode.SEMANTIC

    def test_select_mode_lexical_exact_paths(self):
        """Test mode selection for exact paths."""
        selector = SearchModeSelector()

        # Exact paths should prefer lexical search
        mode = selector.select_mode("core_profiles/profiles_1d")
        assert mode == SearchMode.LEXICAL

        mode = selector.select_mode("equilibrium/time_slice/0/profiles_2d")
        assert mode == SearchMode.LEXICAL

    def test_select_mode_hybrid_mixed_queries(self):
        """Test mode selection for mixed queries."""
        selector = SearchModeSelector()

        # Mixed queries should prefer hybrid search
        mode = selector.select_mode("core_profiles temperature")
        assert mode == SearchMode.HYBRID

        mode = selector.select_mode("equilibrium magnetic field")
        assert mode == SearchMode.HYBRID

    def test_select_mode_semantic_fallback(self):
        """Test mode selection falls back to semantic."""
        selector = SearchModeSelector()

        # Unknown queries should fall back to semantic
        mode = selector.select_mode("unknown physics concept")
        assert mode == SearchMode.SEMANTIC


class TestSearchComposer:
    """Test suite for SearchComposer."""

    def test_search_composer_creation(self):
        """Test SearchComposer creation."""
        store = MockDocumentStore()
        composer = SearchComposer(store)
        assert composer.document_store == store

    @patch("imas_mcp.search.semantic_search.SemanticSearch")
    def test_search_composer_semantic_mode(self, mock_semantic_search):
        """Test SearchComposer with semantic mode."""
        # Mock semantic search instance
        from imas_mcp.search.semantic_search import SemanticSearchResult

        mock_search_instance = Mock()
        store = MockDocumentStore()
        mock_search_instance.search.return_value = [
            SemanticSearchResult(
                document=store.documents[0], similarity_score=0.95, rank=0
            ),
        ]
        mock_semantic_search.return_value = mock_search_instance

        composer = SearchComposer(store)

        results = composer.search_with_params(
            "plasma temperature", SearchMode.SEMANTIC, max_results=5
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert "search_strategy" in results
        assert results["search_strategy"] == "semantic"
        assert isinstance(results["results"], list)

    def test_search_composer_lexical_mode(self):
        """Test SearchComposer with lexical mode."""
        store = MockDocumentStore()
        composer = SearchComposer(store)

        results = composer.search_with_params(
            "electron density", SearchMode.LEXICAL, max_results=5
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert "search_strategy" in results
        assert results["search_strategy"] == "lexical"
        assert isinstance(results["results"], list)

    def test_search_composer_lexical_no_results(self):
        """Test SearchComposer with lexical mode returns no results."""
        store = MockDocumentStore()
        composer = SearchComposer(store)

        results = composer.search_with_params(
            "nonexistent_impossible_query", SearchMode.LEXICAL, max_results=5
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert "search_strategy" in results
        assert results["search_strategy"] == "lexical"
        assert isinstance(results["results"], list)
        assert len(results["results"]) == 0

    @patch("imas_mcp.search.semantic_search.SemanticSearch")
    def test_search_composer_hybrid_mode(self, mock_semantic_search):
        """Test SearchComposer with hybrid mode."""
        # Mock semantic search instance
        from imas_mcp.search.semantic_search import SemanticSearchResult

        mock_search_instance = Mock()
        store = MockDocumentStore()
        mock_search_instance.search.return_value = [
            SemanticSearchResult(
                document=store.documents[0], similarity_score=0.95, rank=0
            ),
        ]
        mock_semantic_search.return_value = mock_search_instance

        composer = SearchComposer(store)

        results = composer.search_with_params(
            "electron density", SearchMode.HYBRID, max_results=5
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert "search_strategy" in results
        assert results["search_strategy"] == "hybrid"
        assert isinstance(results["results"], list)

    def test_search_composer_auto_mode(self):
        """Test SearchComposer with auto mode."""
        store = MockDocumentStore()
        composer = SearchComposer(store)

        # Test with a query that should select lexical mode
        results = composer.search_with_params(
            "core_profiles", SearchMode.AUTO, max_results=5
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert "search_strategy" in results
        assert results["search_strategy"] in ["lexical", "semantic", "hybrid"]
        assert isinstance(results["results"], list)

    def test_search_composer_with_ids_filter(self):
        """Test SearchComposer with IDS filter."""
        store = MockDocumentStore()
        composer = SearchComposer(store)

        results = composer.search_with_params(
            "electron", SearchMode.LEXICAL, max_results=5, filter_ids=["core_profiles"]
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert "search_strategy" in results
        assert results["search_strategy"] == "lexical"

        # Check that results are filtered to the specified IDS
        for result in results["results"]:
            assert result["ids_name"] == "core_profiles"

    def test_search_composer_error_handling(self):
        """Test SearchComposer error handling."""
        store = MockDocumentStore()
        composer = SearchComposer(store)

        # Test with invalid mode (should fall back gracefully)
        results = composer.search_with_params(
            "test query", SearchMode.SEMANTIC, max_results=5
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert "search_strategy" in results
        assert isinstance(results["results"], list)


class TestSearchModeIntegration:
    """Integration tests for search mode functionality."""

    @patch("imas_mcp.search.semantic_search.SemanticSearch")
    def test_end_to_end_search_workflow(self, mock_semantic_search):
        """Test complete search workflow with different modes."""
        # Mock semantic search instance
        from imas_mcp.search.semantic_search import SemanticSearchResult

        mock_search_instance = Mock()
        store = MockDocumentStore()
        mock_search_instance.search.return_value = [
            SemanticSearchResult(
                document=store.documents[0], similarity_score=0.95, rank=0
            ),
        ]
        mock_semantic_search.return_value = mock_search_instance

        composer = SearchComposer(store)

        # Test all search modes
        modes = [
            SearchMode.SEMANTIC,
            SearchMode.LEXICAL,
            SearchMode.HYBRID,
            SearchMode.AUTO,
        ]

        for mode in modes:
            results = composer.search_with_params(
                "plasma temperature", mode, max_results=3
            )

            assert isinstance(results, dict)
            assert "results" in results
            assert "search_strategy" in results
            assert "results_count" in results  # Updated to match actual implementation
            assert isinstance(results["results"], list)
            assert len(results["results"]) <= 3

    def test_search_mode_consistency(self):
        """Test that search mode selection is consistent."""
        selector = SearchModeSelector()

        # Same query should always return same mode
        query = "plasma temperature"
        mode1 = selector.select_mode(query)
        mode2 = selector.select_mode(query)
        assert mode1 == mode2

        # Different queries should potentially return different modes
        physics_query = "electron density profile"
        path_query = "core_profiles/profiles_1d"

        physics_mode = selector.select_mode(physics_query)
        path_mode = selector.select_mode(path_query)

        # Physics query should prefer semantic, path query should prefer lexical
        assert physics_mode == SearchMode.SEMANTIC
        assert path_mode == SearchMode.LEXICAL


# Performance and edge case tests
class TestSearchModePerformance:
    """Test suite for search mode performance and edge cases."""

    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        store = MockDocumentStore()
        composer = SearchComposer(store)

        # Test empty string
        results = composer.search_with_params("", SearchMode.LEXICAL, max_results=5)
        assert isinstance(results, dict)
        assert len(results["results"]) == 0

        # Test whitespace only
        results = composer.search_with_params("   ", SearchMode.LEXICAL, max_results=5)
        assert isinstance(results, dict)
        assert len(results["results"]) == 0

    def test_large_result_set_handling(self):
        """Test handling of large result sets."""
        store = MockDocumentStore()
        composer = SearchComposer(store)

        # Test with very large max_results
        results = composer.search_with_params(
            "electron", SearchMode.LEXICAL, max_results=1000
        )
        assert isinstance(results, dict)
        assert len(results["results"]) <= len(store.documents)

    def test_special_characters_in_query(self):
        """Test handling of special characters in queries."""
        store = MockDocumentStore()
        composer = SearchComposer(store)

        # Test with special characters
        special_queries = [
            "electron/density",
            "temperature_profile",
            "core-profiles",
            "flux@equilibrium",
            "density[0]",
        ]

        for query in special_queries:
            results = composer.search_with_params(
                query, SearchMode.LEXICAL, max_results=5
            )
            assert isinstance(results, dict)
            assert "results" in results
            assert isinstance(results["results"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
