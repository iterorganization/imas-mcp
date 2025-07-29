"""
Tests for search engine implementations.

This module tests the SemanticSearchEngine, LexicalSearchEngine, and HybridSearchEngine
implementations to ensure they properly implement the SearchEngine interface.
"""

import pytest
from unittest.mock import MagicMock, patch

# Mock semantic search before importing engines
with patch.dict(
    "sys.modules",
    {
        "imas_mcp.search.semantic_search": MagicMock(),
    },
):
    # Import after mocking
    from imas_mcp.search.engines.semantic_engine import SemanticSearchEngine
    from imas_mcp.search.engines.lexical_engine import LexicalSearchEngine
    from imas_mcp.search.engines.hybrid_engine import HybridSearchEngine
    from imas_mcp.search.engines.base_engine import SearchEngineError
    from imas_mcp.search.search_strategy import SearchConfig, SearchResult
    from imas_mcp.search.document_store import Document, DocumentMetadata
    from imas_mcp.models.constants import SearchMode


@pytest.fixture
def mock_document_store():
    """Create a mock document store for testing."""
    store = MagicMock()
    store.ids_set = {"core_profiles", "equilibrium"}

    # Mock document for testing
    mock_metadata = DocumentMetadata(
        path_name="core_profiles/profiles_1d/electrons/temperature",
        path_id="test_path_1",
        ids_name="core_profiles",
        data_type="temperature",
        physics_domain="plasma_core",
    )

    mock_document = Document(
        metadata=mock_metadata,
        documentation="Temperature profile for electrons",
        units=None,
    )

    store.get_all_documents.return_value = [mock_document]
    store.search_full_text.return_value = [mock_document]

    return store


@pytest.fixture
def search_config():
    """Create a standard search configuration for testing."""
    return SearchConfig(
        mode=SearchMode.SEMANTIC,
        max_results=5,
        filter_ids=None,
        similarity_threshold=0.0,
    )


class TestSemanticSearchEngine:
    """Test cases for SemanticSearchEngine."""

    def test_semantic_engine_initialization(self, mock_document_store):
        """Test that semantic engine can be initialized."""
        engine = SemanticSearchEngine(mock_document_store)

        assert engine is not None
        assert engine.name == "semantic"
        assert engine.get_engine_type() == "semantic"
        assert engine.document_store == mock_document_store

    @patch("imas_mcp.search.engines.semantic_engine.SemanticSearch", create=True)
    @patch("imas_mcp.search.engines.semantic_engine.SemanticSearchConfig", create=True)
    async def test_semantic_search_execution(
        self, mock_config_class, mock_search_class, mock_document_store, search_config
    ):
        """Test semantic search execution with mocked semantic search."""
        # Create proper document
        from imas_mcp.search.document_store import Document, DocumentMetadata

        doc_metadata = DocumentMetadata(
            path_name="core_profiles/profiles_1d/electrons/temperature",
            path_id="test_path_1",
            ids_name="core_profiles",
            data_type="temperature",
            physics_domain="plasma_core",
        )

        test_document = Document(
            metadata=doc_metadata,
            documentation="Temperature profile for electrons",
            units=None,
        )

        # Setup mocks
        mock_semantic_result = MagicMock()
        mock_semantic_result.document = test_document
        mock_semantic_result.similarity_score = 0.85

        mock_search_instance = MagicMock()
        mock_search_instance.search.return_value = [mock_semantic_result]
        mock_search_class.return_value = mock_search_instance

        # Test search execution
        engine = SemanticSearchEngine(mock_document_store)

        # Directly set the semantic_search property
        engine._semantic_search = mock_search_instance

        results = await engine.search("temperature", search_config)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].score == 0.85
        assert results[0].search_mode == SearchMode.SEMANTIC
        assert results[0].rank == 0

    async def test_semantic_search_invalid_query(
        self, mock_document_store, search_config
    ):
        """Test semantic search with invalid query."""
        engine = SemanticSearchEngine(mock_document_store)

        with pytest.raises(SearchEngineError) as exc_info:
            await engine.search("", search_config)

        assert "Invalid query" in str(exc_info.value)

    def test_semantic_query_suitability(self, mock_document_store):
        """Test semantic engine query suitability detection."""
        engine = SemanticSearchEngine(mock_document_store)

        # Conceptual queries should be suitable
        assert engine.is_suitable_for_query("what is temperature")
        assert engine.is_suitable_for_query("explain plasma physics")
        assert engine.is_suitable_for_query("describe magnetic field")

        # Technical queries should be less suitable
        assert not engine.is_suitable_for_query("core_profiles/temperature")
        assert not engine.is_suitable_for_query("AND OR NOT")

    def test_semantic_health_status(self, mock_document_store):
        """Test semantic engine health status."""
        engine = SemanticSearchEngine(mock_document_store)

        # Mock the _semantic_search attribute instead of property
        engine._semantic_search = MagicMock()

        health = engine.get_health_status()

        assert health["status"] == "healthy"
        assert health["engine_type"] == "semantic"
        assert health["document_count"] == 1
        assert health["semantic_search_available"] is True


class TestLexicalSearchEngine:
    """Test cases for LexicalSearchEngine."""

    def test_lexical_engine_initialization(self, mock_document_store):
        """Test that lexical engine can be initialized."""
        engine = LexicalSearchEngine(mock_document_store)

        assert engine is not None
        assert engine.name == "lexical"
        assert engine.get_engine_type() == "lexical"
        assert engine.document_store == mock_document_store

    async def test_lexical_search_execution(self, mock_document_store, search_config):
        """Test lexical search execution."""
        engine = LexicalSearchEngine(mock_document_store)
        results = await engine.search("temperature", search_config)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].score == 1.0  # First result gets score 1.0
        assert results[0].search_mode == SearchMode.LEXICAL
        assert results[0].rank == 0

        # Verify document store was called
        mock_document_store.search_full_text.assert_called_once()

    async def test_lexical_search_with_ids_filter(
        self, mock_document_store, search_config
    ):
        """Test lexical search with IDS filtering."""
        search_config.filter_ids = ["core_profiles"]

        engine = LexicalSearchEngine(mock_document_store)
        await engine.search("temperature", search_config)

        # Check that query was modified to include IDS filter
        call_args = mock_document_store.search_full_text.call_args[0]
        query_used = call_args[0]
        assert "ids_name:core_profiles" in query_used

    async def test_lexical_search_invalid_query(
        self, mock_document_store, search_config
    ):
        """Test lexical search with invalid query."""
        engine = LexicalSearchEngine(mock_document_store)

        with pytest.raises(SearchEngineError) as exc_info:
            await engine.search("", search_config)

        assert "Invalid query" in str(exc_info.value)

    def test_lexical_query_suitability(self, mock_document_store):
        """Test lexical engine query suitability detection."""
        engine = LexicalSearchEngine(mock_document_store)

        # Technical queries should be suitable
        assert engine.is_suitable_for_query("core_profiles/temperature")
        assert engine.is_suitable_for_query("profiles_1d AND temperature")
        assert engine.is_suitable_for_query('"exact phrase"')
        assert engine.is_suitable_for_query("temperature_profile")

        # Simple technical terms should also be suitable now
        assert engine.is_suitable_for_query("temperature")  # Technical term
        assert engine.is_suitable_for_query("time")  # Technical term

        # Pure conceptual queries should be less suitable
        assert not engine.is_suitable_for_query("what is plasma")

    def test_lexical_fts_query_preparation(self, mock_document_store):
        """Test FTS query preparation."""
        engine = LexicalSearchEngine(mock_document_store)

        prepared = engine.prepare_query_for_fts("  temperature  ")
        assert prepared == "temperature"

    def test_lexical_health_status(self, mock_document_store):
        """Test lexical engine health status."""
        engine = LexicalSearchEngine(mock_document_store)
        health = engine.get_health_status()

        assert health["status"] == "healthy"
        assert health["engine_type"] == "lexical"
        assert health["document_count"] == 1
        assert health["fts5_available"] is True


class TestHybridSearchEngine:
    """Test cases for HybridSearchEngine."""

    def test_hybrid_engine_initialization(self, mock_document_store):
        """Test that hybrid engine can be initialized."""
        engine = HybridSearchEngine(mock_document_store)

        assert engine is not None
        assert engine.name == "hybrid"
        assert engine.get_engine_type() == "hybrid"
        assert engine.document_store == mock_document_store
        assert isinstance(engine.semantic_engine, SemanticSearchEngine)
        assert isinstance(engine.lexical_engine, LexicalSearchEngine)

    @patch("imas_mcp.search.engines.semantic_engine.SemanticSearch", create=True)
    @patch("imas_mcp.search.engines.semantic_engine.SemanticSearchConfig", create=True)
    async def test_hybrid_search_execution(
        self, mock_config_class, mock_search_class, mock_document_store, search_config
    ):
        """Test hybrid search execution combining both engines."""
        # Create proper document
        from imas_mcp.search.document_store import Document, DocumentMetadata

        doc_metadata = DocumentMetadata(
            path_name="core_profiles/profiles_1d/electrons/temperature",
            path_id="test_path_1",
            ids_name="core_profiles",
            data_type="temperature",
            physics_domain="plasma_core",
        )

        test_document = Document(
            metadata=doc_metadata,
            documentation="Temperature profile for electrons",
            units=None,
        )

        # Setup semantic search mock
        mock_semantic_result = MagicMock()
        mock_semantic_result.document = test_document
        mock_semantic_result.similarity_score = 0.80

        mock_search_instance = MagicMock()
        mock_search_instance.search.return_value = [mock_semantic_result]
        mock_search_class.return_value = mock_search_instance

        # Mock lexical search results too
        mock_document_store.search_full_text.return_value = [test_document]

        # Test hybrid search
        engine = HybridSearchEngine(mock_document_store)

        # Directly set the semantic_search property for the semantic engine
        engine.semantic_engine._semantic_search = mock_search_instance

        results = await engine.search("temperature", search_config)

        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert results[0].search_mode == SearchMode.HYBRID
        # Score should be boosted from combination logic
        assert results[0].score > 0.80

    async def test_hybrid_search_deduplication(
        self, mock_document_store, search_config
    ):
        """Test that hybrid search properly deduplicates results from both engines."""
        # Create proper document
        from imas_mcp.search.document_store import Document, DocumentMetadata

        doc_metadata = DocumentMetadata(
            path_name="core_profiles/profiles_1d/electrons/temperature",
            path_id="test_path_1",
            ids_name="core_profiles",
            data_type="temperature",
            physics_domain="plasma_core",
        )

        test_document = Document(
            metadata=doc_metadata,
            documentation="Temperature profile for electrons",
            units=None,
        )

        with patch(
            "imas_mcp.search.engines.semantic_engine.SemanticSearch", create=True
        ):
            engine = HybridSearchEngine(mock_document_store)

            # Mock semantic results
            mock_semantic_result = MagicMock()
            mock_semantic_result.document = test_document
            mock_semantic_result.similarity_score = 0.75

            mock_search_instance = MagicMock()
            mock_search_instance.search.return_value = [mock_semantic_result]
            engine.semantic_engine._semantic_search = mock_search_instance

            # Mock lexical results with same document
            mock_document_store.search_full_text.return_value = [test_document]

            # Both engines return the same document
            # Hybrid should deduplicate and boost the score
            results = await engine.search("temperature", search_config)

            # Should have only one result (deduplicated)
            assert len(results) == 1
            assert results[0].search_mode == SearchMode.HYBRID

    async def test_hybrid_search_invalid_query(
        self, mock_document_store, search_config
    ):
        """Test hybrid search with invalid query."""
        engine = HybridSearchEngine(mock_document_store)

        with pytest.raises(SearchEngineError) as exc_info:
            await engine.search("", search_config)

        assert "Invalid query" in str(exc_info.value)

    def test_hybrid_query_suitability(self, mock_document_store):
        """Test hybrid engine query suitability detection."""
        engine = HybridSearchEngine(mock_document_store)

        # Hybrid should be suitable for all queries as fallback
        assert engine.is_suitable_for_query("general temperature search")
        assert engine.is_suitable_for_query("what is temperature")  # conceptual
        assert engine.is_suitable_for_query("core_profiles/temp")  # technical
        assert engine.is_suitable_for_query("random query")  # fallback

    def test_hybrid_combination_strategy(self, mock_document_store):
        """Test hybrid combination strategy selection."""
        engine = HybridSearchEngine(mock_document_store)

        # Technical queries should use lexical-dominant strategy
        strategy = engine.get_combination_strategy("core_profiles/temperature")
        assert strategy == "lexical_dominant"

        # Pure conceptual queries without technical terms should use semantic-dominant strategy
        strategy = engine.get_combination_strategy("how does physics work")
        assert strategy == "semantic_dominant"

        # Mixed queries may favor either based on technical content
        strategy = engine.get_combination_strategy("what is equilibrium")
        assert strategy in ["lexical_dominant", "semantic_dominant"]  # Could be either

        # Balanced queries should use balanced strategy (when neither engine claims it)
        strategy = engine.get_combination_strategy("random data search")
        assert strategy == "balanced"

    def test_hybrid_health_status(self, mock_document_store):
        """Test hybrid engine health status."""
        with patch(
            "imas_mcp.search.engines.semantic_engine.SemanticSearch", create=True
        ):
            engine = HybridSearchEngine(mock_document_store)
            health = engine.get_health_status()

            assert health["status"] == "healthy"
            assert health["engine_type"] == "hybrid"
            assert "semantic_engine" in health
            assert "lexical_engine" in health


class TestSearchEngineIntegration:
    """Integration tests for search engines."""

    def test_all_engines_implement_interface(self, mock_document_store):
        """Test that all engines properly implement the SearchEngine interface."""
        engines = [
            SemanticSearchEngine(mock_document_store),
            LexicalSearchEngine(mock_document_store),
            HybridSearchEngine(mock_document_store),
        ]

        for engine in engines:
            # All engines should have required methods
            assert hasattr(engine, "search")
            assert hasattr(engine, "get_engine_type")
            assert hasattr(engine, "validate_query")
            assert hasattr(engine, "normalize_query")

            # All engines should have unique type identifiers
            assert engine.get_engine_type() in ["semantic", "lexical", "hybrid"]

    async def test_engine_error_handling(self, mock_document_store, search_config):
        """Test that engines properly handle and propagate errors."""
        # Mock document store to raise exception
        mock_document_store.search_full_text.side_effect = Exception("Database error")

        engine = LexicalSearchEngine(mock_document_store)

        with pytest.raises(SearchEngineError) as exc_info:
            await engine.search("test", search_config)

        assert "Lexical search failed" in str(exc_info.value)
        assert "Database error" in str(exc_info.value)

    def test_engine_query_validation_consistency(self, mock_document_store):
        """Test that all engines validate queries consistently."""
        engines = [
            SemanticSearchEngine(mock_document_store),
            LexicalSearchEngine(mock_document_store),
            HybridSearchEngine(mock_document_store),
        ]

        test_queries = [
            ("valid query", True),
            ("", False),
            ("   ", False),
            (["valid", "list"], True),
            ([], False),
            (["valid", ""], True),  # Mixed list should be valid
            (None, False),
        ]

        for query, expected_valid in test_queries:
            for engine in engines:
                if query is not None:
                    result = engine.validate_query(query)
                    assert result == expected_valid, (
                        f"Engine {engine.get_engine_type()} failed on query: {query}"
                    )
