"""Tests for hybrid search engine.

These tests focus on the non-async methods and helper functions of the
HybridSearchEngine to avoid conflicts with session-scoped mocks.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.models.constants import SearchMode
from imas_codex.search.document_store import Document, DocumentMetadata, DocumentStore
from imas_codex.search.engines.hybrid_engine import HybridSearchEngine
from imas_codex.search.search_strategy import SearchConfig, SearchMatch, SearchResponse


def create_mock_document(
    path_id: str, ids_name: str = "core_profiles", documentation: str = ""
) -> Document:
    """Create a mock document for testing."""
    metadata = DocumentMetadata(
        path_id=path_id,
        ids_name=ids_name,
        path_name=path_id.split("/")[-1],
        units="m",
        data_type="float",
    )
    return Document(
        metadata=metadata,
        documentation=documentation or f"Documentation for {path_id}",
        relationships={},
        raw_data={},
    )


def create_search_match(doc: Document, score: float, rank: int) -> SearchMatch:
    """Create a SearchMatch for testing."""
    return SearchMatch(
        document=doc, score=score, rank=rank, search_mode=SearchMode.LEXICAL
    )


@pytest.fixture
def mock_document_store():
    """Create a mock document store."""
    store = MagicMock(spec=DocumentStore)
    store.ids_set = {"core_profiles", "equilibrium"}
    store.get_all_documents.return_value = []
    store.get_available_ids.return_value = ["core_profiles", "equilibrium"]
    store.search_full_text.return_value = []
    return store


@pytest.fixture
def mock_semantic_engine():
    """Create a mock semantic engine."""
    engine = MagicMock()
    engine.search = AsyncMock()
    engine.is_suitable_for_query = MagicMock(return_value=True)
    engine.get_health_status = MagicMock(
        return_value={
            "status": "healthy",
            "engine_type": "semantic",
            "document_count": 10,
            "ids_set": ["core_profiles"],
        }
    )
    return engine


@pytest.fixture
def mock_lexical_engine():
    """Create a mock lexical engine."""
    engine = MagicMock()
    engine.search = AsyncMock()
    engine.is_suitable_for_query = MagicMock(return_value=True)
    engine.get_health_status = MagicMock(
        return_value={"status": "healthy", "engine_type": "lexical"}
    )
    return engine


@pytest.fixture
def hybrid_engine(mock_document_store, mock_semantic_engine, mock_lexical_engine):
    """Create a hybrid search engine for testing."""
    engine = HybridSearchEngine(mock_document_store)
    engine.semantic_engine = mock_semantic_engine
    engine.lexical_engine = mock_lexical_engine
    return engine


class TestHybridSearchEngine:
    """Tests for HybridSearchEngine helper methods and non-async functionality."""

    def test_get_engine_type(self, hybrid_engine):
        """Test engine type identifier."""
        assert hybrid_engine.get_engine_type() == "hybrid"

    def test_is_suitable_for_query_both_suitable(
        self, hybrid_engine, mock_semantic_engine, mock_lexical_engine
    ):
        """Test suitability when both engines are suitable."""
        mock_semantic_engine.is_suitable_for_query.return_value = True
        mock_lexical_engine.is_suitable_for_query.return_value = True

        assert hybrid_engine.is_suitable_for_query("plasma temperature") is True

    def test_is_suitable_for_query_neither_suitable(
        self, hybrid_engine, mock_semantic_engine, mock_lexical_engine
    ):
        """Test suitability when neither engine is suitable."""
        mock_semantic_engine.is_suitable_for_query.return_value = False
        mock_lexical_engine.is_suitable_for_query.return_value = False

        assert hybrid_engine.is_suitable_for_query("random query") is True

    def test_is_suitable_for_query_mixed(
        self, hybrid_engine, mock_semantic_engine, mock_lexical_engine
    ):
        """Test suitability for mixed queries."""
        mock_semantic_engine.is_suitable_for_query.return_value = True
        mock_lexical_engine.is_suitable_for_query.return_value = False

        assert (
            hybrid_engine.is_suitable_for_query("describe plasma equilibrium") is True
        )

    def test_get_combination_strategy_lexical_dominant(
        self, hybrid_engine, mock_lexical_engine, mock_semantic_engine
    ):
        """Test combination strategy selection for technical queries."""
        mock_lexical_engine.is_suitable_for_query.return_value = True
        mock_semantic_engine.is_suitable_for_query.return_value = False

        strategy = hybrid_engine.get_combination_strategy("core_profiles/temperature")

        assert strategy == "lexical_dominant"

    def test_get_combination_strategy_semantic_dominant(
        self, hybrid_engine, mock_lexical_engine, mock_semantic_engine
    ):
        """Test combination strategy selection for conceptual queries."""
        mock_lexical_engine.is_suitable_for_query.return_value = False
        mock_semantic_engine.is_suitable_for_query.return_value = True

        strategy = hybrid_engine.get_combination_strategy("what is plasma")

        assert strategy == "semantic_dominant"

    def test_get_combination_strategy_balanced(
        self, hybrid_engine, mock_lexical_engine, mock_semantic_engine
    ):
        """Test combination strategy selection for balanced queries."""
        mock_lexical_engine.is_suitable_for_query.return_value = False
        mock_semantic_engine.is_suitable_for_query.return_value = False

        strategy = hybrid_engine.get_combination_strategy("general query")

        assert strategy == "balanced"

    def test_get_health_status(
        self, hybrid_engine, mock_semantic_engine, mock_lexical_engine
    ):
        """Test health status reporting."""
        mock_semantic_engine.get_health_status.return_value = {
            "status": "healthy",
            "engine_type": "semantic",
            "document_count": 100,
            "ids_set": ["core_profiles"],
        }
        mock_lexical_engine.get_health_status.return_value = {
            "status": "healthy",
            "engine_type": "lexical",
        }

        status = hybrid_engine.get_health_status()

        assert status["status"] == "healthy"
        assert status["engine_type"] == "hybrid"
        assert "semantic_engine" in status
        assert "lexical_engine" in status

    def test_get_health_status_unhealthy(
        self, hybrid_engine, mock_semantic_engine, mock_lexical_engine
    ):
        """Test health status when one engine is unhealthy."""
        mock_semantic_engine.get_health_status.return_value = {
            "status": "unhealthy",
            "error": "Semantic failed",
        }
        mock_lexical_engine.get_health_status.return_value = {"status": "healthy"}

        status = hybrid_engine.get_health_status()

        assert status["status"] == "unhealthy"

    def test_get_health_status_exception(
        self, hybrid_engine, mock_semantic_engine, mock_lexical_engine
    ):
        """Test health status when exception occurs."""
        mock_semantic_engine.get_health_status.side_effect = Exception("Error")

        status = hybrid_engine.get_health_status()

        assert status["status"] == "unhealthy"
        assert "error" in status
