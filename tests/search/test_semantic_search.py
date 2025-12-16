"""Tests for search/semantic_search.py module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from imas_codex.search.document_store import Document, DocumentMetadata
from imas_codex.search.semantic_search import (
    SemanticSearch,
    SemanticSearchConfig,
    SemanticSearchResult,
)


class TestSemanticSearchResult:
    """Tests for the SemanticSearchResult class."""

    def test_path_id_property(self):
        """path_id property returns document path."""
        metadata = DocumentMetadata(
            path_id="test/path",
            ids_name="test_ids",
            path_name="path",
        )
        doc = Document(metadata=metadata, documentation="Test doc")
        result = SemanticSearchResult(document=doc, similarity_score=0.95, rank=0)

        assert result.path_id == "test/path"

    def test_ids_name_property(self):
        """ids_name property returns document IDS name."""
        metadata = DocumentMetadata(
            path_id="test/path",
            ids_name="test_ids",
            path_name="path",
        )
        doc = Document(metadata=metadata, documentation="Test doc")
        result = SemanticSearchResult(document=doc, similarity_score=0.95, rank=0)

        assert result.ids_name == "test_ids"


class TestSemanticSearchConfig:
    """Tests for the SemanticSearchConfig class."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = SemanticSearchConfig()

        assert config.default_top_k == 10
        assert config.similarity_threshold == 0.0
        assert config.ids_set is None

    def test_custom_values(self):
        """Config accepts custom values."""
        config = SemanticSearchConfig(
            default_top_k=20,
            similarity_threshold=0.5,
            ids_set={"equilibrium"},
        )

        assert config.default_top_k == 20
        assert config.similarity_threshold == 0.5
        assert config.ids_set == {"equilibrium"}


class TestSemanticSearch:
    """Tests for the SemanticSearch class."""

    def test_requires_embeddings(self):
        """SemanticSearch requires embeddings to be provided."""
        with pytest.raises(ValueError, match="Embeddings instance must be provided"):
            SemanticSearch(embeddings=None)

    def test_compute_similarities_with_empty_matrix(self):
        """_compute_similarities handles empty matrix."""
        mock_embeddings = MagicMock()
        mock_embeddings.get_embeddings_matrix.return_value = np.array([])
        mock_embeddings.get_path_ids.return_value = []

        search = SemanticSearch(embeddings=mock_embeddings)
        result = search._compute_similarities(np.zeros(384), np.array([]))

        assert len(result) == 0

    def test_get_document_count(self):
        """get_document_count returns document store count."""
        mock_embeddings = MagicMock()
        search = SemanticSearch(embeddings=mock_embeddings)

        # Document store is mocked in conftest
        count = search.get_document_count()
        assert isinstance(count, int)
