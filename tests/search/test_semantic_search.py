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

    def test_get_embeddings_and_ids(self):
        """_get_embeddings_and_ids retrieves matrix and path IDs."""
        mock_embeddings = MagicMock()
        mock_embeddings.get_embeddings_matrix.return_value = np.random.randn(5, 384)
        mock_embeddings.get_path_ids.return_value = [
            "path1",
            "path2",
            "path3",
            "path4",
            "path5",
        ]

        search = SemanticSearch(embeddings=mock_embeddings)
        matrix, path_ids = search._get_embeddings_and_ids()

        assert matrix.shape == (5, 384)
        assert len(path_ids) == 5

    def test_get_embeddings_and_ids_no_embeddings(self):
        """_get_embeddings_and_ids raises error when no embeddings."""
        mock_embeddings = MagicMock()
        search = SemanticSearch(embeddings=mock_embeddings)
        search.embeddings = None  # Clear embeddings

        with pytest.raises(RuntimeError, match="Embeddings component not set"):
            search._get_embeddings_and_ids()

    def test_compute_similarities_with_data(self):
        """_compute_similarities computes dot product correctly."""
        mock_embeddings = MagicMock()
        search = SemanticSearch(embeddings=mock_embeddings)

        query = np.array([1.0, 0.0, 0.0])
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])

        result = search._compute_similarities(query, matrix)

        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.5])

    def test_get_candidate_indices_basic(self):
        """_get_candidate_indices returns top indices above threshold."""
        mock_embeddings = MagicMock()
        mock_store = MagicMock()
        mock_store.get_document.return_value = None

        search = SemanticSearch(embeddings=mock_embeddings, document_store=mock_store)

        similarities = np.array([0.3, 0.8, 0.1, 0.9, 0.5])
        path_ids = ["p1", "p2", "p3", "p4", "p5"]

        indices = search._get_candidate_indices(
            similarities,
            max_candidates=3,
            similarity_threshold=0.2,
            ids_filter=None,
            path_ids=path_ids,
        )

        # Should return top 3 above threshold: indices 3 (0.9), 1 (0.8), 4 (0.5)
        assert 3 in indices  # highest score
        assert 1 in indices  # second highest
        assert 4 in indices  # third highest
        assert 2 not in indices  # below threshold

    def test_get_candidate_indices_with_ids_filter(self):
        """_get_candidate_indices respects IDS filter."""
        mock_embeddings = MagicMock()
        mock_store = MagicMock()

        # Create documents with different IDS names
        def get_doc(path_id):
            if path_id in ["p1", "p3"]:
                metadata = DocumentMetadata(
                    path_id=path_id, ids_name="equilibrium", path_name="path"
                )
                return Document(metadata=metadata, documentation="Test")
            return None

        mock_store.get_document.side_effect = get_doc

        search = SemanticSearch(embeddings=mock_embeddings, document_store=mock_store)

        similarities = np.array([0.9, 0.8, 0.7])
        path_ids = ["p1", "p2", "p3"]

        indices = search._get_candidate_indices(
            similarities,
            max_candidates=5,
            similarity_threshold=0.0,
            ids_filter=["equilibrium"],
            path_ids=path_ids,
        )

        # Only p1 and p3 match the IDS filter
        assert 0 in indices  # p1 matches
        assert 2 in indices  # p3 matches
        assert 1 not in indices  # p2 doesn't match

    def test_apply_hybrid_boost(self):
        """_apply_hybrid_boost boosts scores for FTS matches."""
        mock_embeddings = MagicMock()
        mock_store = MagicMock()

        # Mock FTS results
        fts_metadata = DocumentMetadata(
            path_id="matched_path", ids_name="test", path_name="path"
        )
        fts_doc = Document(metadata=fts_metadata, documentation="Test")
        mock_store.search_full_text.return_value = [fts_doc]

        search = SemanticSearch(embeddings=mock_embeddings, document_store=mock_store)

        # Create test results
        matched_metadata = DocumentMetadata(
            path_id="matched_path", ids_name="test", path_name="path"
        )
        matched_doc = Document(metadata=matched_metadata, documentation="Test")
        matched_result = SemanticSearchResult(
            document=matched_doc, similarity_score=0.8, rank=0
        )

        unmatched_metadata = DocumentMetadata(
            path_id="unmatched_path", ids_name="test", path_name="path2"
        )
        unmatched_doc = Document(metadata=unmatched_metadata, documentation="Test2")
        unmatched_result = SemanticSearchResult(
            document=unmatched_doc, similarity_score=0.8, rank=1
        )

        results = search._apply_hybrid_boost(
            "query", [matched_result, unmatched_result]
        )

        # Matched should be boosted by 1.1
        assert results[0].similarity_score == pytest.approx(0.88)  # 0.8 * 1.1
        assert results[1].similarity_score == pytest.approx(0.8)  # unchanged

    def test_search_no_embeddings_raises(self):
        """search raises error when embeddings not configured."""
        mock_embeddings = MagicMock()
        search = SemanticSearch(embeddings=mock_embeddings)
        search.embeddings = None

        with pytest.raises(RuntimeError, match="Embeddings not configured"):
            search.search("test query")

    def test_search_basic(self):
        """search returns results with correct structure."""
        mock_embeddings = MagicMock()
        mock_embeddings.encode_texts.return_value = np.array([[1.0, 0.0, 0.0]])
        mock_embeddings.get_embeddings_matrix.return_value = np.array(
            [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]
        )
        mock_embeddings.get_path_ids.return_value = ["path1", "path2"]

        mock_store = MagicMock()
        metadata = DocumentMetadata(path_id="path1", ids_name="test", path_name="path")
        doc = Document(metadata=metadata, documentation="Test doc")
        mock_store.get_document.return_value = doc
        mock_store.search_full_text.return_value = []

        search = SemanticSearch(embeddings=mock_embeddings, document_store=mock_store)
        results = search.search("test query", top_k=2, hybrid_search=False)

        assert len(results) <= 2
        for result in results:
            assert isinstance(result, SemanticSearchResult)

    def test_search_with_defaults(self):
        """search uses config defaults when not specified."""
        config = SemanticSearchConfig(default_top_k=5, similarity_threshold=0.1)
        mock_embeddings = MagicMock()
        mock_embeddings.encode_texts.return_value = np.array([[1.0, 0.0, 0.0]])
        mock_embeddings.get_embeddings_matrix.return_value = np.array([[0.9, 0.1, 0.0]])
        mock_embeddings.get_path_ids.return_value = ["path1"]

        mock_store = MagicMock()
        mock_store.get_document.return_value = None
        mock_store.search_full_text.return_value = []

        search = SemanticSearch(
            config=config, embeddings=mock_embeddings, document_store=mock_store
        )
        search.search("test query")

        # Should work with defaults

    def test_search_similar_documents(self):
        """search_similar_documents finds related documents."""
        mock_embeddings = MagicMock()
        mock_embeddings.encode_texts.return_value = np.array([[1.0, 0.0, 0.0]])
        mock_embeddings.get_embeddings_matrix.return_value = np.array(
            [[0.95, 0.05, 0.0], [0.1, 0.9, 0.0]]
        )
        mock_embeddings.get_path_ids.return_value = ["path1", "path2"]

        mock_store = MagicMock()
        metadata1 = DocumentMetadata(
            path_id="path1", ids_name="test", path_name="path1"
        )
        doc1 = Document(
            metadata=metadata1, documentation="Source doc for embedding text generation"
        )
        metadata2 = DocumentMetadata(
            path_id="path2", ids_name="test", path_name="path2"
        )
        doc2 = Document(metadata=metadata2, documentation="Similar doc")

        def get_doc(path_id):
            return doc1 if path_id == "path1" else doc2

        mock_store.get_document.side_effect = get_doc
        mock_store.search_full_text.return_value = []

        search = SemanticSearch(embeddings=mock_embeddings, document_store=mock_store)
        results = search.search_similar_documents("path1", top_k=1)

        # Should find documents similar to path1's embedding text
        assert isinstance(results, list)

    def test_search_similar_documents_not_found(self):
        """search_similar_documents returns empty for missing document."""
        mock_embeddings = MagicMock()
        mock_store = MagicMock()
        mock_store.get_document.return_value = None

        search = SemanticSearch(embeddings=mock_embeddings, document_store=mock_store)
        results = search.search_similar_documents("nonexistent")

        assert results == []

    def test_batch_search_basic(self):
        """batch_search processes multiple queries."""
        mock_embeddings = MagicMock()
        mock_embeddings.encode_texts.return_value = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        mock_embeddings.get_embeddings_matrix.return_value = np.array([[0.9, 0.1, 0.0]])
        mock_embeddings.get_path_ids.return_value = ["path1"]

        mock_store = MagicMock()
        metadata = DocumentMetadata(path_id="path1", ids_name="test", path_name="path")
        doc = Document(metadata=metadata, documentation="Test")
        mock_store.get_document.return_value = doc

        search = SemanticSearch(embeddings=mock_embeddings, document_store=mock_store)
        results = search.batch_search(["query1", "query2"], top_k=5)

        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_batch_search_empty_matrix(self):
        """batch_search handles empty embeddings matrix."""
        mock_embeddings = MagicMock()
        mock_embeddings.get_embeddings_matrix.return_value = np.array([])
        mock_embeddings.get_path_ids.return_value = []

        search = SemanticSearch(embeddings=mock_embeddings)
        results = search.batch_search(["query1", "query2"])

        assert results == [[], []]

    def test_batch_search_no_embeddings_raises(self):
        """batch_search raises error when embeddings not configured."""
        mock_embeddings = MagicMock()
        search = SemanticSearch(embeddings=mock_embeddings)
        search.embeddings = None

        with pytest.raises(RuntimeError, match="Embeddings not configured"):
            search.batch_search(["query"])

    def test_get_embeddings_info_not_initialized(self):
        """get_embeddings_info handles uninitialized state."""
        mock_embeddings = MagicMock()
        mock_embeddings._embedding_manager = None

        search = SemanticSearch(embeddings=mock_embeddings)
        info = search.get_embeddings_info()

        assert info == {"status": "not_initialized"}

    def test_get_embeddings_info_with_manager(self):
        """get_embeddings_info returns cache info from manager."""
        mock_embeddings = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_cache_info.return_value = {"cached": True, "count": 100}
        mock_embeddings._embedding_manager = mock_manager

        search = SemanticSearch(embeddings=mock_embeddings)
        info = search.get_embeddings_info()

        assert info == {"cached": True, "count": 100}

    def test_cache_status_not_initialized(self):
        """cache_status handles uninitialized state."""
        mock_embeddings = MagicMock()
        mock_embeddings._embedding_manager = None

        search = SemanticSearch(embeddings=mock_embeddings)
        status = search.cache_status()

        assert status == {"status": "not_initialized"}

    def test_cache_status_with_manager(self):
        """cache_status returns info from manager."""
        mock_embeddings = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_cache_info.return_value = {"status": "ready"}
        mock_embeddings._embedding_manager = mock_manager

        search = SemanticSearch(embeddings=mock_embeddings)
        status = search.cache_status()

        assert status == {"status": "ready"}

    def test_list_cache_files_not_initialized(self):
        """list_cache_files handles uninitialized state."""
        mock_embeddings = MagicMock()
        mock_embeddings._embedding_manager = None

        search = SemanticSearch(embeddings=mock_embeddings)
        files = search.list_cache_files()

        assert files == []

    def test_list_cache_files_with_manager(self):
        """list_cache_files returns files from manager."""
        mock_embeddings = MagicMock()
        mock_manager = MagicMock()
        mock_manager.list_cache_files.return_value = [{"file": "cache.pkl"}]
        mock_embeddings._embedding_manager = mock_manager

        search = SemanticSearch(embeddings=mock_embeddings)
        files = search.list_cache_files()

        assert files == [{"file": "cache.pkl"}]

    def test_cleanup_old_caches_not_initialized(self):
        """cleanup_old_caches handles uninitialized state."""
        mock_embeddings = MagicMock()
        mock_embeddings._embedding_manager = None

        search = SemanticSearch(embeddings=mock_embeddings)
        removed = search.cleanup_old_caches()

        assert removed == 0

    def test_cleanup_old_caches_with_manager(self):
        """cleanup_old_caches calls manager."""
        mock_embeddings = MagicMock()
        mock_manager = MagicMock()
        mock_manager.cleanup_old_caches.return_value = 3
        mock_embeddings._embedding_manager = mock_manager

        search = SemanticSearch(embeddings=mock_embeddings)
        removed = search.cleanup_old_caches(keep_count=2)

        assert removed == 3
        mock_manager.cleanup_old_caches.assert_called_once_with(2)

    def test_list_all_cache_files_no_dir(self):
        """list_all_cache_files handles missing directory."""
        with MagicMock() as mock_accessor:
            mock_accessor.embeddings_dir.exists.return_value = False

            # Should return empty list when dir doesn't exist
            result = SemanticSearch.list_all_cache_files()
            assert isinstance(result, list)

    def test_cleanup_all_old_caches(self):
        """cleanup_all_old_caches removes old files."""
        # Static method - just verify it returns an int
        result = SemanticSearch.cleanup_all_old_caches(keep_count=3)
        assert isinstance(result, int)

    def test_rebuild_embeddings_raises(self):
        """rebuild_embeddings raises NotImplementedError."""
        mock_embeddings = MagicMock()
        search = SemanticSearch(embeddings=mock_embeddings)

        with pytest.raises(NotImplementedError):
            search.rebuild_embeddings()
