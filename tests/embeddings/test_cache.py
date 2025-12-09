"""Tests for embeddings/cache.py module."""

import json

import numpy as np

from imas_mcp.embeddings.cache import EmbeddingCache


class TestEmbeddingCache:
    """Tests for the EmbeddingCache class."""

    def test_initialization(self):
        """Cache initializes with provided values."""
        cache = EmbeddingCache(
            embeddings=np.zeros((10, 384)),
            path_ids=["path_" + str(i) for i in range(10)],
            model_name="test-model",
            document_count=10,
        )

        assert cache.document_count == 10
        assert cache.model_name == "test-model"
        assert len(cache.path_ids) == 10

    def test_validation_document_count_mismatch(self):
        """Validation fails when document count differs."""
        cache = EmbeddingCache(
            embeddings=np.zeros((10, 384)),
            path_ids=["path_" + str(i) for i in range(10)],
            model_name="test-model",
            document_count=10,
        )

        is_valid, reason = cache.validate_with_reason(5, "test-model")

        assert is_valid is False
        assert "Document count mismatch" in reason

    def test_validation_model_mismatch(self):
        """Validation fails when model name differs."""
        cache = EmbeddingCache(
            embeddings=np.zeros((10, 384)),
            path_ids=["path_" + str(i) for i in range(10)],
            model_name="test-model",
            document_count=10,
        )

        is_valid, reason = cache.validate_with_reason(10, "different-model")

        assert is_valid is False
        assert "Model name mismatch" in reason

    def test_validation_empty_embeddings(self):
        """Validation fails when embeddings are empty."""
        cache = EmbeddingCache(
            embeddings=np.array([]),
            path_ids=["path_0"],
            model_name="test-model",
            document_count=1,
        )

        is_valid, reason = cache.validate_with_reason(1, "test-model")

        assert is_valid is False
        assert "no embeddings" in reason

    def test_validation_empty_path_ids(self):
        """Validation fails when path IDs are empty."""
        cache = EmbeddingCache(
            embeddings=np.zeros((1, 384)),
            path_ids=[],
            model_name="test-model",
            document_count=1,
        )

        is_valid, reason = cache.validate_with_reason(1, "test-model")

        assert is_valid is False
        assert "no path IDs" in reason

    def test_validation_ids_set_mismatch(self):
        """Validation fails when IDS set differs."""
        cache = EmbeddingCache(
            embeddings=np.zeros((10, 384)),
            path_ids=["path_" + str(i) for i in range(10)],
            model_name="test-model",
            document_count=10,
            ids_set={"equilibrium"},
        )

        is_valid, reason = cache.validate_with_reason(
            10, "test-model", current_ids_set={"core_profiles"}
        )

        assert is_valid is False
        assert "IDS set mismatch" in reason

    def test_validation_valid_cache(self):
        """Validation passes for valid cache."""
        cache = EmbeddingCache(
            embeddings=np.zeros((10, 384)),
            path_ids=["path_" + str(i) for i in range(10)],
            model_name="test-model",
            document_count=10,
            ids_set={"equilibrium"},
        )

        is_valid, reason = cache.validate_with_reason(
            10, "test-model", current_ids_set={"equilibrium"}
        )

        assert is_valid is True
        assert reason == "Cache is valid"

    def test_is_valid_convenience_method(self):
        """is_valid method returns boolean correctly."""
        cache = EmbeddingCache(
            embeddings=np.zeros((10, 384)),
            path_ids=["path_" + str(i) for i in range(10)],
            model_name="test-model",
            document_count=10,
        )

        assert cache.is_valid(10, "test-model") is True
        assert cache.is_valid(5, "test-model") is False

    def test_compute_source_hash_deterministic(self, tmp_path):
        """Source content hash is deterministic."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()

        catalog = {"metadata": {"version": "4.0.0"}, "ids_catalog": {}}
        (source_dir / "ids_catalog.json").write_text(json.dumps(catalog))

        detailed = source_dir / "detailed"
        detailed.mkdir()
        (detailed / "test.json").write_text('{"test": true}')

        cache = EmbeddingCache()
        hash1 = cache._compute_source_content_hash(source_dir)
        hash2 = cache._compute_source_content_hash(source_dir)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length

    def test_dd_version_validation(self, tmp_path):
        """Validation fails when DD version changes."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        catalog = {"metadata": {"version": "4.0.0"}}
        (source_dir / "ids_catalog.json").write_text(json.dumps(catalog))

        cache = EmbeddingCache(
            embeddings=np.zeros((1, 384)),
            path_ids=["path_0"],
            model_name="test-model",
            document_count=1,
            dd_version="3.9.0",
        )

        is_valid, reason = cache.validate_with_reason(
            1, "test-model", source_data_dir=source_dir
        )

        assert is_valid is False
        assert "Data Dictionary version changed" in reason
