"""Tests for lexical search engine.

These tests focus on the non-async methods and helper functions of the
LexicalSearchEngine to avoid conflicts with session-scoped mocks.
"""

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.models.constants import SearchMode
from imas_codex.search.document_store import Document, DocumentMetadata, DocumentStore
from imas_codex.search.engines.lexical_engine import LexicalSearchEngine
from imas_codex.search.search_strategy import SearchConfig


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
    store.get_available_ids.return_value = ["core_profiles", "equilibrium", "wall"]
    store.get_document.return_value = None
    store.search_full_text.return_value = []
    store.get_all_documents.return_value = []
    return store


@pytest.fixture
def lexical_engine(mock_document_store):
    """Create a lexical search engine for testing."""
    return LexicalSearchEngine(mock_document_store)


class TestLexicalSearchEngine:
    """Tests for LexicalSearchEngine helper methods and non-async functionality."""

    def test_extract_ids_from_path(self, lexical_engine, mock_document_store):
        """Test IDS extraction from path-like queries."""
        mock_document_store.get_available_ids.return_value = [
            "core_profiles",
            "equilibrium",
        ]

        result = lexical_engine._extract_ids_from_path(
            "core_profiles/profiles_1d/temperature"
        )
        assert result == "core_profiles"

        result = lexical_engine._extract_ids_from_path("temperature")
        assert result is None

        result = lexical_engine._extract_ids_from_path("invalid_ids/path")
        assert result is None

    def test_is_full_imas_path(self, lexical_engine, mock_document_store):
        """Test IMAS path detection."""
        mock_document_store.get_available_ids.return_value = [
            "core_profiles",
            "equilibrium",
        ]

        assert (
            lexical_engine._is_full_imas_path("core_profiles/profiles_1d/temperature")
            is True
        )
        assert lexical_engine._is_full_imas_path("temperature") is False
        assert lexical_engine._is_full_imas_path("invalid/path") is False

    def test_escape_path_for_fts(self, lexical_engine):
        """Test FTS path escaping."""
        escaped = lexical_engine._escape_path_for_fts(
            "core_profiles/profiles_1d/temperature"
        )

        assert '"' in escaped
        assert "/" not in escaped

    def test_is_suitable_for_query_explicit_operators(self, lexical_engine):
        """Test suitability detection with explicit operators."""
        assert lexical_engine.is_suitable_for_query("units:eV") is True
        assert lexical_engine.is_suitable_for_query("temperature AND density") is True
        assert lexical_engine.is_suitable_for_query('"exact phrase"') is True

    def test_is_suitable_for_query_path_like(self, lexical_engine):
        """Test suitability detection for path-like queries."""
        assert lexical_engine.is_suitable_for_query("profiles_1d/temperature") is True
        assert lexical_engine.is_suitable_for_query("time_slice") is True

    def test_is_suitable_for_query_technical_terms(self, lexical_engine):
        """Test suitability detection for technical terms."""
        assert lexical_engine.is_suitable_for_query("core_profiles") is True
        assert lexical_engine.is_suitable_for_query("rho_tor_norm") is True

    def test_is_suitable_for_query_conceptual(self, lexical_engine):
        """Test suitability for conceptual queries returns False."""
        assert lexical_engine.is_suitable_for_query("explain plasma physics") is False

    def test_get_engine_type(self, lexical_engine):
        """Test engine type identifier."""
        assert lexical_engine.get_engine_type() == "lexical"

    def test_prepare_query_for_fts(self, lexical_engine):
        """Test FTS query preparation."""
        prepared = lexical_engine.prepare_query_for_fts("  temperature  ")
        assert prepared == "temperature"

    def test_get_health_status(self, lexical_engine, mock_document_store):
        """Test health status reporting."""
        mock_document_store.get_all_documents.return_value = [
            create_mock_document("test/path")
        ]
        mock_document_store.search_full_text.return_value = []

        status = lexical_engine.get_health_status()

        assert status["status"] == "healthy"
        assert status["engine_type"] == "lexical"
        assert status["fts5_available"] is True

    def test_get_health_status_unhealthy(self, lexical_engine, mock_document_store):
        """Test health status when unhealthy."""
        mock_document_store.get_all_documents.side_effect = Exception("Test error")

        status = lexical_engine.get_health_status()

        assert status["status"] == "unhealthy"
        assert "error" in status

    def test_enhance_config_with_path_intelligence(
        self, lexical_engine, mock_document_store
    ):
        """Test config enhancement with path intelligence."""
        mock_document_store.get_available_ids.return_value = [
            "core_profiles",
            "equilibrium",
        ]

        config = SearchConfig(max_results=10)
        enhanced = lexical_engine._enhance_config_with_path_intelligence(
            "core_profiles/temperature", config
        )

        assert enhanced.ids_filter == ["core_profiles"]

    def test_enhance_config_respects_existing_filter(
        self, lexical_engine, mock_document_store
    ):
        """Test that existing IDS filter is respected."""
        config = SearchConfig(max_results=10, ids_filter=["equilibrium"])
        enhanced = lexical_engine._enhance_config_with_path_intelligence(
            "core_profiles/temperature", config
        )

        assert enhanced.ids_filter == ["equilibrium"]

    @pytest.mark.asyncio
    async def test_try_exact_path_search_not_full_path(
        self, lexical_engine, mock_document_store
    ):
        """Test exact path search returns None for non-path queries."""
        mock_document_store.get_available_ids.return_value = ["core_profiles"]

        config = SearchConfig(max_results=10)
        result = lexical_engine._try_exact_path_search("temperature", config)

        assert result is None

    @pytest.mark.asyncio
    async def test_try_exact_path_search_fts_fallback(
        self, lexical_engine, mock_document_store
    ):
        """Test exact path search falls back to FTS when direct lookup fails."""
        mock_document_store.get_available_ids.return_value = [
            "core_profiles",
            "equilibrium",
        ]
        mock_document_store.get_document.return_value = None
        mock_docs = [create_mock_document("core_profiles/profiles_1d/temperature")]
        mock_document_store.search_full_text.return_value = mock_docs

        config = SearchConfig(max_results=10)
        result = lexical_engine._try_exact_path_search(
            "core_profiles/profiles_1d/temperature", config
        )

        assert result is not None
        assert len(result.hits) == 1
