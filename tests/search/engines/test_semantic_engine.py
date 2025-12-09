"""Tests for semantic search engine.

These tests focus on the non-async methods and helper functions of the
SemanticSearchEngine to avoid conflicts with session-scoped mocks.
"""

from unittest.mock import MagicMock, patch

import pytest

from imas_mcp.models.constants import SearchMode
from imas_mcp.search.document_store import Document, DocumentMetadata, DocumentStore
from imas_mcp.search.engines.semantic_engine import SemanticSearchEngine
from imas_mcp.search.search_strategy import SearchConfig


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


@pytest.fixture
def mock_document_store():
    """Create a mock document store."""
    store = MagicMock(spec=DocumentStore)
    store.ids_set = {"core_profiles", "equilibrium"}
    store.get_all_documents.return_value = []
    return store


@pytest.fixture
def semantic_engine(mock_document_store):
    """Create a semantic search engine for testing."""
    return SemanticSearchEngine(mock_document_store)


class TestSemanticSearchEngine:
    """Tests for SemanticSearchEngine helper methods and non-async functionality."""

    def test_get_engine_type(self, semantic_engine):
        """Test engine type identifier."""
        assert semantic_engine.get_engine_type() == "semantic"

    def test_is_suitable_for_query_conceptual(self, semantic_engine):
        """Test suitability for conceptual queries."""
        assert (
            semantic_engine.is_suitable_for_query("what is plasma temperature") is True
        )
        assert semantic_engine.is_suitable_for_query("explain tokamak physics") is True
        assert (
            semantic_engine.is_suitable_for_query("how does magnetic confinement work")
            is True
        )

    def test_is_suitable_for_query_natural_language(self, semantic_engine):
        """Test suitability for natural language queries."""
        assert (
            semantic_engine.is_suitable_for_query("describe plasma equilibrium") is True
        )
        assert semantic_engine.is_suitable_for_query("meaning of safety factor") is True

    def test_is_suitable_for_query_technical(self, semantic_engine):
        """Test suitability for technical queries returns False."""
        assert (
            semantic_engine.is_suitable_for_query("core_profiles/temperature") is False
        )
        assert semantic_engine.is_suitable_for_query("rho_tor_norm") is False

    def test_is_suitable_for_query_with_physics_terms(self, semantic_engine):
        """Test suitability for queries with physics terms."""
        assert semantic_engine.is_suitable_for_query("plasma physics concept") is True
        assert semantic_engine.is_suitable_for_query("magnetic field theory") is True

    def test_get_health_status(self, semantic_engine, mock_document_store):
        """Test health status reporting."""
        mock_semantic = MagicMock()
        semantic_engine._semantic_search = mock_semantic
        mock_document_store.get_all_documents.return_value = [
            create_mock_document("test/path")
        ]

        status = semantic_engine.get_health_status()

        assert status["status"] == "healthy"
        assert status["engine_type"] == "semantic"
        assert status["semantic_search_available"] is True

    def test_get_health_status_unhealthy(self, semantic_engine, mock_document_store):
        """Test health status when unhealthy."""
        mock_document_store.get_all_documents.side_effect = Exception("Test error")
        semantic_engine._semantic_search = None  # Ensure lazy init will be tried

        with patch.object(
            SemanticSearchEngine,
            "semantic_search",
            new_callable=lambda: property(
                lambda self: (_ for _ in ()).throw(Exception("Init error"))
            ),
        ):
            status = semantic_engine.get_health_status()

            assert status["status"] == "unhealthy"
            assert "error" in status

    def test_lazy_initialization(self, semantic_engine, mock_document_store):
        """Test that semantic search is lazily initialized."""
        assert semantic_engine._semantic_search is None
