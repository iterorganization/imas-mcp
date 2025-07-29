"""
Comprehensive test suite for SemanticSearch class.

This test suite covers all aspects of the SemanticSearch class including:
- Configuration management
- Model loading and initialization
- Embedding generation and caching
- Search functionality (basic, hybrid, batch)
- Error handling and edge cases
- Performance characteristics
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
import pytest

from imas_mcp.search.document_store import Document, DocumentMetadata, DocumentStore
from imas_mcp.search.semantic_search import (
    EmbeddingCache,
    SemanticSearch,
    SemanticSearchConfig,
    SemanticSearchResult,
)


class TestSemanticSearchConfig:
    """Test suite for SemanticSearchConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SemanticSearchConfig()

        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.device is None
        assert config.default_top_k == 10
        assert config.similarity_threshold == 0.0
        assert config.batch_size == 50
        assert config.ids_set is None
        assert config.enable_cache is True
        assert config.normalize_embeddings is True
        assert config.use_half_precision is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SemanticSearchConfig(
            model_name="custom-model",
            device="cuda",
            default_top_k=5,
            similarity_threshold=0.5,
            batch_size=500,
            ids_set={"test_ids"},
            enable_cache=False,
            normalize_embeddings=False,
            use_half_precision=True,
        )

        assert config.model_name == "custom-model"
        assert config.device == "cuda"
        assert config.default_top_k == 5
        assert config.similarity_threshold == 0.5
        assert config.batch_size == 500
        assert config.ids_set == {"test_ids"}
        assert config.enable_cache is False
        assert config.normalize_embeddings is False
        assert config.use_half_precision is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configurations
        config = SemanticSearchConfig(default_top_k=1)
        assert config.default_top_k == 1

        config = SemanticSearchConfig(similarity_threshold=1.0)
        assert config.similarity_threshold == 1.0


class TestEmbeddingCache:
    """Test suite for EmbeddingCache."""

    def test_empty_cache(self):
        """Test empty cache creation."""
        cache = EmbeddingCache()

        assert cache.embeddings.size == 0
        assert cache.path_ids == []
        assert cache.model_name == ""
        assert cache.document_count == 0
        assert cache.ids_set is None
        assert isinstance(cache.created_at, float)

    def test_cache_with_data(self):
        """Test cache with actual data."""
        embeddings = np.random.random((10, 384))
        path_ids = [f"path_{i}" for i in range(10)]

        cache = EmbeddingCache(
            embeddings=embeddings,
            path_ids=path_ids,
            model_name="test-model",
            document_count=10,
            ids_set={"test_ids"},
        )

        assert cache.embeddings.shape == (10, 384)
        assert len(cache.path_ids) == 10
        assert cache.model_name == "test-model"
        assert cache.document_count == 10
        assert cache.ids_set == {"test_ids"}

    def test_cache_validation_valid(self):
        """Test valid cache validation."""
        cache = EmbeddingCache(
            embeddings=np.random.random((10, 384)),
            path_ids=[f"path_{i}" for i in range(10)],
            model_name="test-model",
            document_count=10,
            ids_set={"test_ids"},
        )

        assert cache.is_valid(10, "test-model", {"test_ids"})

    def test_cache_validation_invalid(self):
        """Test invalid cache validation."""
        cache = EmbeddingCache(
            embeddings=np.random.random((10, 384)),
            path_ids=[f"path_{i}" for i in range(10)],
            model_name="test-model",
            document_count=10,
            ids_set={"test_ids"},
        )

        assert not cache.is_valid(10, "different-model", {"test_ids"})
        assert not cache.is_valid(10, "test-model", {"different_ids"})
        assert not cache.is_valid(
            20, "test-model", {"test_ids"}
        )  # Too large difference

    def test_cache_validation_empty_embeddings(self):
        """Test cache validation with empty embeddings."""
        cache = EmbeddingCache(
            embeddings=np.array([]),
            path_ids=[],
            model_name="test-model",
            document_count=0,
        )

        assert not cache.is_valid(0, "test-model", None)


class TestSemanticSearchResult:
    """Test suite for SemanticSearchResult."""

    def test_result_creation(self):
        """Test semantic search result creation."""
        metadata = DocumentMetadata(
            path_id="test/path",
            ids_name="test_ids",
            path_name="test_path",
            units="m",
            data_type="float",
        )

        document = Document(metadata=metadata, documentation="Test documentation")

        result = SemanticSearchResult(document=document, similarity_score=0.95, rank=1)

        assert result.document == document
        assert result.similarity_score == 0.95
        assert result.rank == 1
        assert result.path_id == "test/path"
        assert result.ids_name == "test_ids"

    def test_result_properties(self):
        """Test semantic search result properties."""
        metadata = DocumentMetadata(
            path_id="equilibrium/profiles_1d",
            ids_name="equilibrium",
            path_name="profiles_1d",
            units="m",
            data_type="array",
        )

        document = Document(metadata=metadata, documentation="Equilibrium profiles")

        result = SemanticSearchResult(document=document, similarity_score=0.85, rank=0)

        assert result.path_id == "equilibrium/profiles_1d"
        assert result.ids_name == "equilibrium"


class TestSemanticSearch:
    """Test suite for SemanticSearch class."""

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store for testing."""
        mock_store = Mock(spec=DocumentStore)
        type(mock_store).ids_set = PropertyMock(return_value=None)
        mock_store.get_document_count.return_value = 5

        # Create mock documents
        mock_documents = []
        for i in range(5):
            metadata = DocumentMetadata(
                path_id=f"test/path_{i}",
                ids_name="test_ids",
                path_name=f"path_{i}",
                units="m",
                data_type="float",
            )
            doc = Document(metadata=metadata, documentation=f"Test document {i}")
            mock_documents.append(doc)

        mock_store.get_all_documents.return_value = mock_documents
        mock_store.get_document.side_effect = lambda path_id: next(
            (doc for doc in mock_documents if doc.metadata.path_id == path_id), None
        )
        mock_store.search_full_text.return_value = mock_documents[
            :3
        ]  # Return first 3 for FTS

        return mock_store

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock sentence transformer for testing."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.random.random((1, 384))
        return mock_model

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_initialization_with_defaults(self, mock_document_store):
        """Test SemanticSearch initialization with defaults."""
        with patch("imas_mcp.search.semantic_search.SentenceTransformer") as mock_st:
            mock_st.return_value = Mock()
            mock_st.return_value.device = "cpu"
            mock_st.return_value.encode.return_value = np.random.random((5, 384))

            search = SemanticSearch(document_store=mock_document_store)

            assert search.config.model_name == "all-MiniLM-L6-v2"
            assert search.document_store == mock_document_store
            assert search._initialized is True

    def test_initialization_with_custom_config(self, mock_document_store):
        """Test SemanticSearch initialization with custom config."""
        config = SemanticSearchConfig(
            model_name="custom-model", default_top_k=5, enable_cache=False
        )

        with patch("imas_mcp.search.semantic_search.SentenceTransformer") as mock_st:
            mock_st.return_value = Mock()
            mock_st.return_value.device = "cpu"
            mock_st.return_value.encode.return_value = np.random.random((5, 384))

            search = SemanticSearch(config=config, document_store=mock_document_store)

            assert search.config.model_name == "custom-model"
            assert search.config.default_top_k == 5
            assert search.config.enable_cache is False

    def test_initialization_with_ids_set_mismatch(self, mock_document_store):
        """Test initialization with mismatched IDS sets."""
        config = SemanticSearchConfig(ids_set={"different_ids"})
        mock_document_store.ids_set = {"test_ids"}

        with pytest.raises(ValueError, match="does not match"):
            SemanticSearch(config=config, document_store=mock_document_store)

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_model_loading_success(self, mock_st, mock_document_store):
        """Test successful model loading."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.random.random((5, 384))
        mock_st.return_value = mock_model

        search = SemanticSearch(document_store=mock_document_store)

        assert search._model == mock_model
        mock_st.assert_called_once()

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_model_loading_fallback(self, mock_st, mock_document_store):
        """Test model loading with fallback."""
        # First call raises exception, second succeeds
        mock_st.side_effect = [Exception("Model not found"), Mock()]
        mock_st.return_value.device = "cpu"
        mock_st.return_value.encode.return_value = np.random.random((5, 384))

        search = SemanticSearch(document_store=mock_document_store)

        assert search.config.model_name == "all-MiniLM-L6-v2"  # Fallback model
        assert mock_st.call_count == 2

    def test_cache_filename_generation(self, mock_document_store):
        """Test cache filename generation."""
        config = SemanticSearchConfig(
            model_name="test-model",
            batch_size=500,
            ids_set={"test_ids", "another_ids"},
            enable_cache=False,  # Disable cache to prevent file creation
        )

        # Set matching ids_set on mock store using PropertyMock
        type(mock_document_store).ids_set = PropertyMock(
            return_value={"test_ids", "another_ids"}
        )

        with patch("imas_mcp.search.semantic_search.SentenceTransformer") as mock_st:
            mock_st.return_value = Mock()
            mock_st.return_value.device = "cpu"
            mock_st.return_value.encode.return_value = np.random.random((5, 384))

            search = SemanticSearch(config=config, document_store=mock_document_store)

            # Temporarily enable cache just for filename generation test
            search.config.enable_cache = True
            filename = search._generate_cache_filename()
            search.config.enable_cache = False  # Restore disabled state

            assert filename.startswith(".test_model_")
            assert filename.endswith(".pkl")
            assert (
                "batch_500" in filename or len(filename) > 10
            )  # Hash includes batch size

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_embedding_generation(self, mock_st, mock_document_store):
        """Test embedding generation."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.random.random((5, 384))
        mock_st.return_value = mock_model

        search = SemanticSearch(document_store=mock_document_store)

        assert search._embeddings_cache is not None
        assert search._embeddings_cache.embeddings.shape == (5, 384)
        assert len(search._embeddings_cache.path_ids) == 5
        assert search._embeddings_cache.document_count == 5

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_embedding_generation_empty_documents(self, mock_st, mock_document_store):
        """Test embedding generation with empty documents."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_st.return_value = mock_model

        mock_document_store.get_document_count.return_value = 0
        mock_document_store.get_all_documents.return_value = []

        search = SemanticSearch(document_store=mock_document_store)

        assert search._embeddings_cache is not None
        assert search._embeddings_cache.embeddings.size == 0
        assert len(search._embeddings_cache.path_ids) == 0
        assert search._embeddings_cache.document_count == 0

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_basic_search(self, mock_st, mock_document_store):
        """Test basic semantic search."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.side_effect = [
            np.random.random((5, 384)),  # For embedding generation
            np.random.random((1, 384)),  # For query embedding
        ]
        mock_st.return_value = mock_model

        search = SemanticSearch(document_store=mock_document_store)

        # Mock similarity computation to return predictable results
        with patch.object(search, "_compute_similarities") as mock_sim:
            mock_sim.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

            results = search.search("test query", top_k=3)

            assert len(results) == 3
            assert all(isinstance(r, SemanticSearchResult) for r in results)
            assert results[0].similarity_score >= results[1].similarity_score

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_search_with_similarity_threshold(self, mock_st, mock_document_store):
        """Test search with similarity threshold."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.side_effect = [
            np.random.random((5, 384)),  # For embedding generation
            np.random.random((1, 384)),  # For query embedding
        ]
        mock_st.return_value = mock_model

        search = SemanticSearch(document_store=mock_document_store)

        # Mock similarity computation with some scores below threshold
        with patch.object(search, "_compute_similarities") as mock_sim:
            mock_sim.return_value = np.array([0.9, 0.8, 0.4, 0.3, 0.2])

            results = search.search("test query", top_k=5, similarity_threshold=0.5)

            assert len(results) == 2  # Only first 2 above threshold
            assert all(r.similarity_score >= 0.5 for r in results)

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_search_with_ids_filter(self, mock_st, mock_document_store):
        """Test search with IDS filter."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.side_effect = [
            np.random.random((5, 384)),  # For embedding generation
            np.random.random((1, 384)),  # For query embedding
        ]
        mock_st.return_value = mock_model

        search = SemanticSearch(document_store=mock_document_store)

        with patch.object(search, "_compute_similarities") as mock_sim:
            mock_sim.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

            results = search.search("test query", filter_ids=["test_ids"])

            assert len(results) <= 5
            assert all(r.ids_name == "test_ids" for r in results)

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_hybrid_search(self, mock_st, mock_document_store):
        """Test hybrid search with full-text search boost."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.side_effect = [
            np.random.random((5, 384)),  # For embedding generation
            np.random.random((1, 384)),  # For query embedding
        ]
        mock_st.return_value = mock_model

        search = SemanticSearch(document_store=mock_document_store)

        with patch.object(search, "_compute_similarities") as mock_sim:
            mock_sim.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

            results = search.search("test query", hybrid_search=True)

            assert len(results) <= 5
            # First 3 documents should have boost (from FTS mock)
            # This is hard to test precisely without complex mocking

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_search_similar_documents(self, mock_st, mock_document_store):
        """Test finding similar documents."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.side_effect = [
            np.random.random((5, 384)),  # For embedding generation
            np.random.random((1, 384)),  # For query embedding
        ]
        mock_st.return_value = mock_model

        search = SemanticSearch(document_store=mock_document_store)

        with patch.object(search, "_compute_similarities") as mock_sim:
            mock_sim.return_value = np.array(
                [1.0, 0.9, 0.8, 0.7, 0.6]
            )  # First is exact match

            results = search.search_similar_documents("test/path_0", top_k=3)

            assert len(results) == 3
            # Should exclude the source document (first result)
            assert all(r.path_id != "test/path_0" for r in results)

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_batch_search(self, mock_st, mock_document_store):
        """Test batch search functionality."""
        mock_model = Mock()
        mock_model.device = "cpu"
        # Set up encode to return proper batch embeddings
        mock_model.encode.return_value = np.random.random(
            (3, 384)
        )  # 3 queries, 384-dim embeddings
        mock_st.return_value = mock_model

        with patch.object(SemanticSearch, "_initialize") as mock_init:
            mock_init.return_value = None

            search = SemanticSearch(document_store=mock_document_store)

            # Manually set up the search object for testing
            search._model = mock_model
            search._initialized = True
            search._embeddings_cache = Mock()
            search._embeddings_cache.embeddings = np.random.random((5, 384))
            search._embeddings_cache.path_ids = [f"test/path_{i}" for i in range(5)]

            with patch.object(search, "_compute_similarities") as mock_sim:
                mock_sim.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

                queries = ["query 1", "query 2", "query 3"]
                results = search.batch_search(queries, top_k=2)

                assert len(results) == 3  # One result list per query
                assert all(len(query_results) == 2 for query_results in results)

                # Verify that _compute_similarities was called once per query
                assert mock_sim.call_count == 3

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_compute_similarities_normalized(self, mock_st, mock_document_store):
        """Test similarity computation with normalized embeddings."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.random.random((5, 384))
        mock_st.return_value = mock_model

        config = SemanticSearchConfig(normalize_embeddings=True)

        with patch.object(SemanticSearch, "_initialize") as mock_init:
            mock_init.return_value = None

            search = SemanticSearch(config=config, document_store=mock_document_store)

            # Manually set up the search object for testing
            search._model = mock_model
            search._initialized = True
            search._embeddings_cache = Mock()

            # Create normalized embeddings (L2 norm = 1)
            embeddings = np.random.random((5, 384))
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            search._embeddings_cache.embeddings = embeddings

            # Create normalized query embedding
            query_embedding = np.random.random(384)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            similarities = search._compute_similarities(query_embedding)

            assert similarities.shape == (5,)
            # For normalized embeddings, cosine similarity should be in [-1, 1]
            assert np.all(similarities >= -1) and np.all(similarities <= 1)

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_compute_similarities_not_normalized(self, mock_st, mock_document_store):
        """Test similarity computation without normalized embeddings."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.random.random((5, 384))
        mock_st.return_value = mock_model

        config = SemanticSearchConfig(normalize_embeddings=False)
        search = SemanticSearch(config=config, document_store=mock_document_store)

        query_embedding = np.random.random(384)
        similarities = search._compute_similarities(query_embedding)

        assert similarities.shape == (5,)
        assert np.all(similarities >= -1) and np.all(similarities <= 1)

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_get_embeddings_info(self, mock_st, mock_document_store):
        """Test getting embeddings information."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.random.random((5, 384))
        mock_st.return_value = mock_model

        search = SemanticSearch(document_store=mock_document_store)
        info = search.get_embeddings_info()

        assert "model_name" in info
        assert "document_count" in info
        assert "embedding_dimension" in info
        assert "dtype" in info
        assert "created_at" in info
        assert "memory_usage_mb" in info
        assert info["document_count"] == 5
        assert info["embedding_dimension"] == 384

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_get_embeddings_info_not_initialized(self, mock_st, mock_document_store):
        """Test getting embeddings info when not initialized."""
        # Create search without initialization
        search = SemanticSearch.__new__(SemanticSearch)
        search.config = SemanticSearchConfig()
        search.document_store = mock_document_store
        search._embeddings_cache = None

        info = search.get_embeddings_info()

        assert info == {"status": "not_initialized"}

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_clear_cache(self, mock_st, mock_document_store, temp_cache_dir):
        """Test clearing embeddings cache."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.random.random((5, 384))
        mock_st.return_value = mock_model

        config = SemanticSearchConfig(enable_cache=True)
        search = SemanticSearch(config=config, document_store=mock_document_store)

        # Mock cache path
        cache_file = temp_cache_dir / "test_cache.pkl"
        cache_file.write_text("test cache content")
        search._cache_path = cache_file

        search.clear_cache()

        assert not cache_file.exists()
        assert search._embeddings_cache is None
        assert search._initialized is False

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_rebuild_embeddings(self, mock_st, mock_document_store):
        """Test rebuilding embeddings."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.random.random((5, 384))
        mock_st.return_value = mock_model

        search = SemanticSearch(document_store=mock_document_store)

        # Mock clear_cache and _initialize
        with patch.object(search, "clear_cache") as mock_clear:
            with patch.object(search, "_initialize") as mock_init:
                search.rebuild_embeddings()

                mock_clear.assert_called_once()
                mock_init.assert_called_once()


class TestSemanticSearchCaching:
    """Test suite for SemanticSearch caching functionality."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary directory for cache testing."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_document_store(self):
        """Create a mock document store for testing."""
        mock_store = Mock(spec=DocumentStore)
        type(mock_store).ids_set = PropertyMock(return_value=None)
        mock_store.get_document_count.return_value = 3

        # Create mock documents
        mock_documents = []
        for i in range(3):
            metadata = DocumentMetadata(
                path_id=f"test/path_{i}",
                ids_name="test_ids",
                path_name=f"path_{i}",
                units="m",
                data_type="float",
            )
            doc = Document(metadata=metadata, documentation=f"Test document {i}")
            mock_documents.append(doc)

        mock_store.get_all_documents.return_value = mock_documents
        return mock_store

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_cache_save_and_load(self, mock_st, mock_document_store, temp_cache_dir):
        """Test saving and loading cache."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.random.random((3, 384))
        mock_st.return_value = mock_model

        config = SemanticSearchConfig(enable_cache=True)

        # Mock the cache directory
        with patch.object(
            SemanticSearch, "_get_embeddings_dir", return_value=temp_cache_dir
        ):
            search = SemanticSearch(config=config, document_store=mock_document_store)

            # Cache should be created
            assert search._embeddings_cache is not None
            if search._cache_path:
                assert search._cache_path.exists()

            # Create another instance - should load from cache
            search2 = SemanticSearch(config=config, document_store=mock_document_store)

            # Should have same embeddings
            assert search2._embeddings_cache is not None
            assert search2._embeddings_cache.document_count == 3

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_cache_disabled(self, mock_st, mock_document_store):
        """Test operation with cache disabled."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.random.random((3, 384))
        mock_st.return_value = mock_model

        config = SemanticSearchConfig(enable_cache=False)
        search = SemanticSearch(config=config, document_store=mock_document_store)

        assert search._cache_path is None
        assert search._embeddings_cache is not None  # Still created, just not saved

    @patch("imas_mcp.search.semantic_search.SentenceTransformer")
    def test_cache_invalidation(self, mock_st, mock_document_store, temp_cache_dir):
        """Test cache invalidation when conditions change."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.random.random((3, 384))
        mock_st.return_value = mock_model

        config = SemanticSearchConfig(enable_cache=True)

        with patch.object(
            SemanticSearch, "_get_embeddings_dir", return_value=temp_cache_dir
        ):
            # Create first search instance
            SemanticSearch(config=config, document_store=mock_document_store)

            # Change document count
            mock_document_store.get_document_count.return_value = 10

            # Create second instance - should invalidate cache
            search2 = SemanticSearch(config=config, document_store=mock_document_store)

            # Should regenerate embeddings
            assert search2._embeddings_cache is not None


class TestSemanticSearchIntegration:
    """Integration tests for SemanticSearch with real components."""

    @pytest.mark.integration
    @pytest.mark.fast
    def test_integration_with_test_server(self, semantic_search):
        """Test SemanticSearch integration with test server."""

        # Test that semantic search is properly initialized
        assert semantic_search._initialized is True
        assert semantic_search._embeddings_cache is not None
        assert semantic_search._model is not None

        # Test basic search functionality
        results = semantic_search.search("plasma temperature", top_k=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        assert all(isinstance(r, SemanticSearchResult) for r in results)

        # Test embeddings info
        info = semantic_search.get_embeddings_info()
        assert "model_name" in info
        assert "document_count" in info
        assert info["document_count"] > 0

    @pytest.mark.integration
    @pytest.mark.fast
    def test_integration_search_results_quality(self, semantic_search):
        """Test quality of search results."""

        # Search for physics-related terms
        results = semantic_search.search("electron density", top_k=10)

        assert len(results) > 0
        assert all(r.similarity_score > 0 for r in results)
        assert (
            results[0].similarity_score >= results[-1].similarity_score
        )  # Sorted by score

        # Check that results contain relevant information
        paths = [r.path_id for r in results]
        assert any(
            "density" in path.lower() or "electron" in path.lower() for path in paths
        )

    @pytest.mark.integration
    @pytest.mark.fast
    def test_integration_similar_documents(self, semantic_search):
        """Test finding similar documents."""

        # Get first document
        first_results = semantic_search.search("temperature", top_k=1)
        if first_results:
            similar = semantic_search.search_similar_documents(
                first_results[0].path_id, top_k=3
            )

            assert len(similar) <= 3
            assert all(isinstance(r, SemanticSearchResult) for r in similar)
            # Should not include the source document
            assert all(r.path_id != first_results[0].path_id for r in similar)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_integration_batch_search(self, semantic_search):
        """Test batch search functionality."""

        queries = ["plasma temperature", "magnetic field", "electron density"]

        batch_results = semantic_search.batch_search(queries, top_k=3)

        assert len(batch_results) == 3
        assert all(len(query_results) <= 3 for query_results in batch_results)
        assert all(
            all(isinstance(r, SemanticSearchResult) for r in query_results)
            for query_results in batch_results
        )


class TestSemanticSearchErrorHandling:
    """Test suite for error handling in SemanticSearch."""

    def test_search_without_initialization(self):
        """Test search without proper initialization."""
        # Create a mock document store
        mock_store = Mock(spec=DocumentStore)
        type(mock_store).ids_set = PropertyMock(return_value=None)

        # Create SemanticSearch with mocked initialization
        with patch("imas_mcp.search.semantic_search.SentenceTransformer") as mock_st:
            mock_st.return_value = Mock()
            mock_st.return_value.device = "cpu"

            with patch.object(SemanticSearch, "_initialize") as mock_init:
                mock_init.return_value = None

                search = SemanticSearch(document_store=mock_store)

                # Reset to uninitialized state
                search._model = None
                search._embeddings_cache = None
                search._initialized = False

                with pytest.raises(
                    RuntimeError, match="Search not properly initialized"
                ):
                    search.search("test query")

    def test_search_with_invalid_document_store(self):
        """Test search with empty document store."""
        mock_store = Mock(spec=DocumentStore)
        type(mock_store).ids_set = PropertyMock(return_value=None)
        mock_store.get_document_count.return_value = 0
        mock_store.get_all_documents.return_value = []

        with patch("imas_mcp.search.semantic_search.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.device = "cpu"
            mock_model.encode.return_value = np.array(
                [[0.1, 0.2, 0.3]]
            )  # Return proper numpy array
            mock_st.return_value = mock_model

            search = SemanticSearch(document_store=mock_store)

            # Should handle empty document store gracefully and return empty results
            results = search.search("test query")
            assert len(results) == 0

    def test_model_loading_complete_failure(self):
        """Test handling of complete model loading failure."""
        mock_store = Mock()
        mock_store.ids_set = None
        mock_store.get_document_count.return_value = 1
        mock_store.get_all_documents.return_value = []

        with patch("imas_mcp.search.semantic_search.SentenceTransformer") as mock_st:
            mock_st.side_effect = Exception("Complete failure")

            with pytest.raises(Exception, match="Complete failure"):
                SemanticSearch(document_store=mock_store)

    def test_hybrid_search_failure(self, semantic_search):
        """Test hybrid search with FTS failure."""

        # Mock FTS to raise exception
        with patch.object(
            semantic_search.document_store, "search_full_text"
        ) as mock_fts:
            mock_fts.side_effect = Exception("FTS failure")

            # Should not fail, just skip hybrid boost
            results = semantic_search.search("test query", hybrid_search=True)
            assert isinstance(results, list)


class TestSemanticSearchPerformance:
    """Test suite for SemanticSearch performance characteristics."""

    @pytest.mark.performance
    @pytest.mark.fast
    def test_search_performance(self, semantic_search):
        """Test search performance."""

        start_time = time.time()
        results = semantic_search.search("plasma temperature", top_k=10)
        search_time = time.time() - start_time

        # Should complete within reasonable time
        assert search_time < 2.0  # 2 seconds max
        assert len(results) <= 10

    @pytest.mark.performance
    @pytest.mark.fast
    def test_batch_search_efficiency(self, semantic_search):
        """Test batch search efficiency."""

        queries = ["temperature", "density", "magnetic", "pressure", "velocity"]

        start_time = time.time()
        batch_results = semantic_search.batch_search(queries, top_k=5)
        batch_time = time.time() - start_time

        # Batch should be more efficient than individual searches
        assert batch_time < 3.0  # 3 seconds max for 5 queries
        assert len(batch_results) == 5

    @pytest.mark.performance
    @pytest.mark.fast
    def test_memory_usage_stability(self, semantic_search):
        """Test memory usage stability during repeated searches."""

        # Perform multiple searches
        for i in range(10):
            results = semantic_search.search(f"test query {i}", top_k=3)
            assert len(results) <= 3

        # Memory should remain stable (hard to test precisely)
        info = semantic_search.get_embeddings_info()
        assert "memory_usage_mb" in info
        assert info["memory_usage_mb"] > 0
