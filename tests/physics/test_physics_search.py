"""
Unit tests for physics search module.

Tests physics semantic search functionality including embedding generation,
caching, and search operations using mocked dependencies.
"""

import pickle
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from imas_mcp.core.data_model import PhysicsDomain
from imas_mcp.core.physics_domains import DomainCharacteristics
from imas_mcp.models.constants import ComplexityLevel, ConceptType
from imas_mcp.models.physics_models import (
    EmbeddingDocument,
    SemanticResult,
)
from imas_mcp.search.physics_search import (
    PhysicsEmbeddingCache,
    PhysicsSemanticSearch,
    get_physics_search,
    search_physics_concepts,
    build_physics_embeddings,
)


class TestPhysicsEmbeddingDocument:
    """Test cases for EmbeddingDocument model."""

    def test_document_creation_valid(self):
        """Test creating a valid physics embedding document."""
        doc = EmbeddingDocument(
            concept_id="domain:equilibrium",
            concept_type=ConceptType.DOMAIN,
            domain_name="equilibrium",
            title="Equilibrium Domain",
            description="Magnetohydrodynamic equilibrium modeling",
            content="Physics domain: equilibrium | Description: MHD equilibrium",
            metadata={"complexity_level": "intermediate"},
        )

        assert doc.concept_id == "domain:equilibrium"
        assert doc.concept_type == ConceptType.DOMAIN
        assert doc.domain_name == "equilibrium"
        assert doc.title == "Equilibrium Domain"
        assert doc.description == "Magnetohydrodynamic equilibrium modeling"
        assert doc.metadata["complexity_level"] == "intermediate"

    def test_document_frozen(self):
        """Test that document is immutable due to frozen config."""
        doc = EmbeddingDocument(
            concept_id="test",
            concept_type=ConceptType.DOMAIN,
            domain_name="test",
            title="Test",
            description="Test description",
            content="Test content",
        )

        # Pydantic frozen models raise ValidationError, not AttributeError
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError):
            doc.concept_id = "new_id"

    def test_document_default_metadata(self):
        """Test document with default empty metadata."""
        doc = EmbeddingDocument(
            concept_id="test",
            concept_type=ConceptType.DOMAIN,
            domain_name="test",
            title="Test",
            description="Test description",
            content="Test content",
        )

        assert doc.metadata == {}


class TestPhysicsSemanticResult:
    """Test cases for SemanticResult model."""

    def test_result_creation(self):
        """Test creating a physics semantic result."""
        doc = EmbeddingDocument(
            concept_id="domain:equilibrium",
            concept_type=ConceptType.DOMAIN,
            domain_name="equilibrium",
            title="Equilibrium Domain",
            description="Test description",
            content="Test content",
        )

        result = SemanticResult(
            document=doc,
            similarity_score=0.85,
            rank=0,
        )

        assert result.document == doc
        assert result.similarity_score == 0.85
        assert result.rank == 0
        assert result.concept_id == "domain:equilibrium"
        assert result.domain_name == "equilibrium"

    def test_result_properties(self):
        """Test result properties delegate to document correctly."""
        doc = EmbeddingDocument(
            concept_id="phenomenon:transport:diffusion",
            concept_type=ConceptType.PHENOMENON,
            domain_name="transport",
            title="Diffusion",
            description="Test description",
            content="Test content",
        )

        result = SemanticResult(
            document=doc,
            similarity_score=0.75,
            rank=1,
        )

        assert result.concept_id == "phenomenon:transport:diffusion"
        assert result.domain_name == "transport"


class TestPhysicsEmbeddingCache:
    """Test cases for PhysicsEmbeddingCache."""

    def test_cache_creation_empty(self):
        """Test creating an empty cache."""
        cache = PhysicsEmbeddingCache()

        assert cache.size == 0
        assert len(cache.documents) == 0
        assert len(cache.concept_ids) == 0
        assert cache.model_name == ""
        assert cache.is_valid()

    def test_cache_creation_with_data(self):
        """Test creating cache with data."""
        doc = EmbeddingDocument(
            concept_id="test",
            concept_type=ConceptType.DOMAIN,
            domain_name="test",
            title="Test",
            description="Test description",
            content="Test content",
        )

        embeddings = np.array([[0.1, 0.2, 0.3]])
        cache = PhysicsEmbeddingCache(
            embeddings=embeddings,
            documents=[doc],
            concept_ids=["test"],
            model_name="test-model",
            created_at=time.time(),
        )

        assert cache.size == 1
        assert len(cache.documents) == 1
        assert len(cache.concept_ids) == 1
        assert cache.model_name == "test-model"
        assert cache.is_valid()

    def test_cache_validation_invalid(self):
        """Test cache validation with inconsistent data."""
        doc = EmbeddingDocument(
            concept_id="test",
            concept_type=ConceptType.DOMAIN,
            domain_name="test",
            title="Test",
            description="Test description",
            content="Test content",
        )

        # Mismatched sizes
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        cache = PhysicsEmbeddingCache(
            embeddings=embeddings,
            documents=[doc],  # Only one document for two embeddings
            concept_ids=["test"],
            model_name="test-model",
        )

        assert not cache.is_valid()


class TestPhysicsSemanticSearch:
    """Test cases for PhysicsSemanticSearch class."""

    @pytest.fixture
    def mock_domain_accessor(self):
        """Mock domain accessor with test data."""
        accessor = Mock()

        # Mock domain characteristics
        domain_data = DomainCharacteristics(
            description="Magnetohydrodynamic equilibrium modeling",
            primary_phenomena=["magnetic_confinement", "pressure_balance"],
            typical_units=["T", "Pa", "m"],
            measurement_methods=["magnetic_diagnostics", "pressure_gauges"],
            related_domains=["transport", "mhd"],
            complexity_level=ComplexityLevel.INTERMEDIATE,
        )

        accessor.get_all_domains.return_value = {PhysicsDomain.EQUILIBRIUM}
        accessor.get_domain_info.return_value = domain_data

        return accessor

    @pytest.fixture
    def mock_unit_accessor(self):
        """Mock unit accessor with test data."""
        accessor = Mock()

        accessor.get_all_unit_contexts.return_value = {
            "T": "Tesla - magnetic field strength",
            "Pa": "Pascal - pressure unit",
        }
        accessor.get_category_for_unit.return_value = "magnetic_field"
        accessor.get_domains_for_unit.return_value = [PhysicsDomain.EQUILIBRIUM]

        return accessor

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock sentence transformer model."""
        model = Mock()

        # Mock encode method to return deterministic embeddings
        def mock_encode(texts, **kwargs):
            return np.array([[0.1, 0.2, 0.3] for _ in texts])

        model.encode = mock_encode
        return model

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    @patch("imas_mcp.search.physics_search.SentenceTransformer")
    def test_init_default_params(
        self, mock_st_class, mock_unit_class, mock_domain_class
    ):
        """Test PhysicsSemanticSearch initialization with default parameters."""
        search = PhysicsSemanticSearch()

        assert search.model_name == "all-MiniLM-L6-v2"
        assert search.device == "cpu"
        assert search.enable_cache is True
        assert search._model is None
        assert search._cache is None

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    @patch("imas_mcp.search.physics_search.SentenceTransformer")
    def test_init_custom_params(
        self, mock_st_class, mock_unit_class, mock_domain_class
    ):
        """Test PhysicsSemanticSearch initialization with custom parameters."""
        cache_dir = Path("/tmp/test_cache")

        search = PhysicsSemanticSearch(
            model_name="custom-model",
            device="cuda",
            enable_cache=False,
            cache_dir=cache_dir,
        )

        assert search.model_name == "custom-model"
        assert search.device == "cuda"
        assert search.enable_cache is False
        assert search.cache_dir == cache_dir

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    @patch("imas_mcp.search.physics_search.SentenceTransformer")
    def test_load_model(
        self,
        mock_st_class,
        mock_unit_class,
        mock_domain_class,
        mock_sentence_transformer,
    ):
        """Test loading sentence transformer model."""
        mock_st_class.return_value = mock_sentence_transformer

        search = PhysicsSemanticSearch()
        search._load_model()

        assert search._model is not None
        mock_st_class.assert_called_once_with(
            "all-MiniLM-L6-v2",
            device="cpu",
            cache_folder="/home/runner/work/imas-mcp/imas-mcp/imas_mcp/resources/embeddings/models",
            local_files_only=True,
        )

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    def test_compute_content_hash(
        self, mock_unit_class, mock_domain_class, mock_domain_accessor
    ):
        """Test content hash computation."""
        mock_domain_class.return_value = mock_domain_accessor

        search = PhysicsSemanticSearch()
        domain_set = {PhysicsDomain.EQUILIBRIUM}

        hash_result = search._compute_content_hash(domain_set)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex digest length

        # Test hash consistency
        hash_result2 = search._compute_content_hash(domain_set)
        assert hash_result == hash_result2

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    def test_create_physics_documents(
        self,
        mock_unit_class,
        mock_domain_class,
        mock_domain_accessor,
        mock_unit_accessor,
    ):
        """Test creating physics embedding documents."""
        mock_domain_class.return_value = mock_domain_accessor
        mock_unit_class.return_value = mock_unit_accessor

        search = PhysicsSemanticSearch()
        documents = search._create_physics_documents()

        assert len(documents) > 0

        # Check that we have different types of documents
        doc_types = {doc.concept_type for doc in documents}
        assert "domain" in doc_types
        assert "phenomenon" in doc_types
        assert "measurement_method" in doc_types
        assert "unit" in doc_types

        # Validate document structure
        for doc in documents:
            assert isinstance(doc, EmbeddingDocument)
            assert doc.concept_id is not None
            assert doc.concept_type in [
                ConceptType.DOMAIN,
                ConceptType.PHENOMENON,
                ConceptType.MEASUREMENT_METHOD,
                ConceptType.UNIT,
            ]
            assert doc.domain_name is not None
            assert doc.title is not None
            assert doc.description is not None
            assert doc.content is not None

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    @patch("imas_mcp.search.physics_search.SentenceTransformer")
    def test_build_embeddings_fresh(
        self,
        mock_st_class,
        mock_unit_class,
        mock_domain_class,
        mock_domain_accessor,
        mock_unit_accessor,
        mock_sentence_transformer,
    ):
        """Test building embeddings from scratch."""
        mock_domain_class.return_value = mock_domain_accessor
        mock_unit_class.return_value = mock_unit_accessor
        mock_st_class.return_value = mock_sentence_transformer

        search = PhysicsSemanticSearch(enable_cache=False)
        search.build_embeddings()

        assert search._cache is not None
        assert search._cache.size > 0
        assert search._cache.model_name == "all-MiniLM-L6-v2"
        assert search._cache.is_valid()

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    @patch("imas_mcp.search.physics_search.SentenceTransformer")
    def test_search_basic(
        self,
        mock_st_class,
        mock_unit_class,
        mock_domain_class,
        mock_domain_accessor,
        mock_unit_accessor,
        mock_sentence_transformer,
    ):
        """Test basic search functionality."""
        mock_domain_class.return_value = mock_domain_accessor
        mock_unit_class.return_value = mock_unit_accessor
        mock_st_class.return_value = mock_sentence_transformer

        search = PhysicsSemanticSearch(enable_cache=False)
        search.build_embeddings()

        results = search.search("magnetic field", max_results=5)

        assert isinstance(results, list)
        assert all(isinstance(result, SemanticResult) for result in results)
        assert len(results) <= 5

        for result in results:
            assert 0.0 <= result.similarity_score <= 1.0
            assert result.rank >= 0

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    @patch("imas_mcp.search.physics_search.SentenceTransformer")
    def test_search_with_filters(
        self,
        mock_st_class,
        mock_unit_class,
        mock_domain_class,
        mock_domain_accessor,
        mock_unit_accessor,
        mock_sentence_transformer,
    ):
        """Test search with concept type and domain filters."""
        mock_domain_class.return_value = mock_domain_accessor
        mock_unit_class.return_value = mock_unit_accessor
        mock_st_class.return_value = mock_sentence_transformer

        search = PhysicsSemanticSearch(enable_cache=False)
        search.build_embeddings()

        # Test with concept type filter
        results = search.search("test", concept_types=["domain"])

        for result in results:
            assert result.document.concept_type == ConceptType.DOMAIN

        # Test with domain filter
        results = search.search("test", domains=["equilibrium"])

        for result in results:
            assert result.document.domain_name == "equilibrium"

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    @patch("imas_mcp.search.physics_search.SentenceTransformer")
    def test_search_no_results(
        self,
        mock_st_class,
        mock_unit_class,
        mock_domain_class,
        mock_domain_accessor,
        mock_unit_accessor,
        mock_sentence_transformer,
    ):
        """Test search with no matching results."""
        mock_domain_class.return_value = mock_domain_accessor
        mock_unit_class.return_value = mock_unit_accessor

        # Mock to return very low similarities
        def mock_encode(texts, **kwargs):
            return np.array([[0.0, 0.0, 0.0] for _ in texts])

        mock_sentence_transformer.encode = mock_encode
        mock_st_class.return_value = mock_sentence_transformer

        search = PhysicsSemanticSearch(enable_cache=False)
        search.build_embeddings()

        results = search.search("nonexistent", min_similarity=0.5)

        assert len(results) == 0

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    @patch("imas_mcp.search.physics_search.SentenceTransformer")
    def test_find_similar_concepts(
        self,
        mock_st_class,
        mock_unit_class,
        mock_domain_class,
        mock_domain_accessor,
        mock_unit_accessor,
        mock_sentence_transformer,
    ):
        """Test finding similar concepts."""
        mock_domain_class.return_value = mock_domain_accessor
        mock_unit_class.return_value = mock_unit_accessor
        mock_st_class.return_value = mock_sentence_transformer

        search = PhysicsSemanticSearch(enable_cache=False)
        search.build_embeddings()

        # Get a concept ID from the cache
        if search._cache and search._cache.concept_ids:
            concept_id = search._cache.concept_ids[0]
            results = search.find_similar_concepts(concept_id)

            assert isinstance(results, list)
            assert all(isinstance(result, SemanticResult) for result in results)

            # Original concept should not be in results
            for result in results:
                assert result.concept_id != concept_id

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    @patch("imas_mcp.search.physics_search.SentenceTransformer")
    def test_get_concept_by_id(
        self,
        mock_st_class,
        mock_unit_class,
        mock_domain_class,
        mock_domain_accessor,
        mock_unit_accessor,
        mock_sentence_transformer,
    ):
        """Test getting concept by ID."""
        mock_domain_class.return_value = mock_domain_accessor
        mock_unit_class.return_value = mock_unit_accessor
        mock_st_class.return_value = mock_sentence_transformer

        search = PhysicsSemanticSearch(enable_cache=False)
        search.build_embeddings()

        if search._cache and search._cache.concept_ids:
            concept_id = search._cache.concept_ids[0]
            document = search.get_concept_by_id(concept_id)

            assert document is not None
            assert isinstance(document, EmbeddingDocument)
            assert document.concept_id == concept_id

        # Test with non-existent ID
        document = search.get_concept_by_id("nonexistent:id")
        assert document is None

    def test_cache_save_load_cycle(self, tmp_path):
        """Test saving and loading cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create a test document and cache
        doc = EmbeddingDocument(
            concept_id="test",
            concept_type=ConceptType.DOMAIN,
            domain_name="test",
            title="Test",
            description="Test description",
            content="Test content",
        )

        embeddings = np.array([[0.1, 0.2, 0.3]])
        original_cache = PhysicsEmbeddingCache(
            embeddings=embeddings,
            documents=[doc],
            concept_ids=["test"],
            model_name="test-model",
            created_at=time.time(),
        )

        # Save cache manually
        cache_path = cache_dir / "test_cache.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(original_cache, f)

        # Load cache manually
        with open(cache_path, "rb") as f:
            loaded_cache = pickle.load(f)

        assert loaded_cache.size == original_cache.size
        assert loaded_cache.model_name == original_cache.model_name
        assert loaded_cache.concept_ids == original_cache.concept_ids
        assert np.array_equal(loaded_cache.embeddings, original_cache.embeddings)


class TestPhysicsSearchGlobalFunctions:
    """Test cases for global functions."""

    @patch("imas_mcp.search.physics_search.PhysicsSemanticSearch")
    def test_get_physics_search_singleton(self, mock_search_class):
        """Test that get_physics_search returns singleton instance."""
        mock_instance = Mock()
        mock_search_class.return_value = mock_instance

        # Reset global instance
        import imas_mcp.search.physics_search

        imas_mcp.search.physics_search._physics_search = None

        instance1 = get_physics_search()
        instance2 = get_physics_search()

        assert instance1 is instance2
        mock_search_class.assert_called_once()

    @patch("imas_mcp.search.physics_search.get_physics_search")
    def test_search_physics_concepts_function(self, mock_get_search):
        """Test search_physics_concepts convenience function."""
        mock_search_instance = Mock()
        mock_results = [Mock()]
        mock_search_instance.search.return_value = mock_results
        mock_get_search.return_value = mock_search_instance

        results = search_physics_concepts("test query", max_results=5)

        assert results == mock_results
        mock_search_instance.search.assert_called_once_with("test query", max_results=5)

    @patch("imas_mcp.search.physics_search.get_physics_search")
    def test_build_physics_embeddings_function(self, mock_get_search):
        """Test build_physics_embeddings convenience function."""
        mock_search_instance = Mock()
        mock_get_search.return_value = mock_search_instance

        build_physics_embeddings(force_rebuild=True)

        mock_search_instance.build_embeddings.assert_called_once_with(
            force_rebuild=True
        )


class TestPhysicsSearchEdgeCases:
    """Test edge cases and error conditions."""

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    def test_no_domains_available(self, mock_unit_class, mock_domain_class):
        """Test behavior when no domains are available."""
        mock_domain_accessor = Mock()
        mock_domain_accessor.get_all_domains.return_value = set()
        mock_domain_class.return_value = mock_domain_accessor

        mock_unit_accessor = Mock()
        mock_unit_accessor.get_all_unit_contexts.return_value = {}
        mock_unit_class.return_value = mock_unit_accessor

        search = PhysicsSemanticSearch(enable_cache=False)
        documents = search._create_physics_documents()

        assert len(documents) == 0

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    @patch("imas_mcp.search.physics_search.SentenceTransformer")
    def test_search_with_empty_cache(
        self, mock_st_class, mock_unit_class, mock_domain_class
    ):
        """Test search behavior with empty cache."""
        # Setup mocks to return empty data
        mock_domain_accessor = Mock()
        mock_domain_accessor.get_all_domains.return_value = set()
        mock_domain_class.return_value = mock_domain_accessor

        mock_unit_accessor = Mock()
        mock_unit_accessor.get_all_unit_contexts.return_value = {}
        mock_unit_class.return_value = mock_unit_accessor

        mock_transformer = Mock()
        mock_transformer.encode.return_value = np.array([])
        mock_st_class.return_value = mock_transformer

        search = PhysicsSemanticSearch(enable_cache=False)
        search.build_embeddings()

        results = search.search("test query")
        assert len(results) == 0

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    def test_invalid_cache_handling(self, mock_unit_class, mock_domain_class, tmp_path):
        """Test handling of invalid cache files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create invalid cache file
        cache_path = cache_dir / "invalid_cache.pkl"
        with open(cache_path, "w") as f:
            f.write("invalid pickle data")

        search = PhysicsSemanticSearch(cache_dir=cache_dir, enable_cache=True)

        # Should handle invalid cache gracefully
        loaded = search._load_cache()
        assert not loaded
        assert search._cache is None


class TestPhysicsSearchPerformance:
    """Performance and efficiency tests."""

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    @patch("imas_mcp.search.physics_search.SentenceTransformer")
    def test_large_document_set_handling(
        self, mock_st_class, mock_unit_class, mock_domain_class
    ):
        """Test handling of large document sets."""
        # Create mock that returns many domains
        mock_domain_accessor = Mock()
        large_domain_set = {
            PhysicsDomain.EQUILIBRIUM,
            PhysicsDomain.TRANSPORT,
            PhysicsDomain.MHD,
            PhysicsDomain.HEATING,
        }
        mock_domain_accessor.get_all_domains.return_value = large_domain_set

        # Mock domain info with many phenomena and methods
        domain_data = DomainCharacteristics(
            description="Test domain",
            primary_phenomena=[f"phenomenon_{i}" for i in range(20)],
            typical_units=[f"unit_{i}" for i in range(10)],
            measurement_methods=[f"method_{i}" for i in range(15)],
            related_domains=["test"],
            complexity_level=ComplexityLevel.ADVANCED,
        )
        mock_domain_accessor.get_domain_info.return_value = domain_data
        mock_domain_class.return_value = mock_domain_accessor

        # Mock unit accessor with many units
        mock_unit_accessor = Mock()
        unit_contexts = {f"unit_{i}": f"Description {i}" for i in range(50)}
        mock_unit_accessor.get_all_unit_contexts.return_value = unit_contexts
        mock_unit_accessor.get_category_for_unit.return_value = "test_category"
        mock_unit_accessor.get_domains_for_unit.return_value = [
            PhysicsDomain.EQUILIBRIUM
        ]
        mock_unit_class.return_value = mock_unit_accessor

        # Mock sentence transformer
        def mock_encode(texts, **kwargs):
            return np.random.rand(len(texts), 384)  # Typical embedding dimension

        mock_transformer = Mock()
        mock_transformer.encode = mock_encode
        mock_st_class.return_value = mock_transformer

        search = PhysicsSemanticSearch(enable_cache=False)
        search.build_embeddings()

        # Should handle large document set efficiently
        assert search._cache is not None
        assert search._cache.size > 100  # Should have many documents

        # Search should still work efficiently
        results = search.search("test query", max_results=10)
        assert len(results) <= 10

    @patch("imas_mcp.search.physics_search.DomainAccessor")
    @patch("imas_mcp.search.physics_search.UnitAccessor")
    @patch("imas_mcp.search.physics_search.SentenceTransformer")
    def test_multiple_searches_efficiency(
        self, mock_st_class, mock_unit_class, mock_domain_class
    ):
        """Test that multiple searches are efficient (model loaded once)."""
        # Setup mocks
        mock_domain_accessor = Mock()
        domain_data = DomainCharacteristics(
            description="Test domain",
            primary_phenomena=["phenomenon1"],
            typical_units=["unit1"],
            measurement_methods=["method1"],
            related_domains=["test"],
            complexity_level=ComplexityLevel.BASIC,
        )
        mock_domain_accessor.get_all_domains.return_value = {PhysicsDomain.EQUILIBRIUM}
        mock_domain_accessor.get_domain_info.return_value = domain_data
        mock_domain_class.return_value = mock_domain_accessor

        mock_unit_accessor = Mock()
        mock_unit_accessor.get_all_unit_contexts.return_value = {"T": "Tesla"}
        mock_unit_accessor.get_category_for_unit.return_value = "test_category"
        mock_unit_accessor.get_domains_for_unit.return_value = [
            PhysicsDomain.EQUILIBRIUM
        ]
        mock_unit_class.return_value = mock_unit_accessor

        mock_sentence_transformer = Mock()
        mock_sentence_transformer.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st_class.return_value = mock_sentence_transformer

        search = PhysicsSemanticSearch(enable_cache=False)
        search.build_embeddings()

        # Perform multiple searches
        for i in range(5):
            results = search.search(f"query {i}")
            assert isinstance(results, list)

        # SentenceTransformer should only be created once
        mock_st_class.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
