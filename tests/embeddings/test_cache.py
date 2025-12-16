"""Tests for embeddings/cache.py module."""

import json

import numpy as np

from imas_codex.embeddings.cache import EmbeddingCache


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

    def test_dd_version_missing_requires_rebuild(self, tmp_path):
        """Validation fails when cache has no DD version."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        catalog = {"metadata": {"version": "4.0.0"}}
        (source_dir / "ids_catalog.json").write_text(json.dumps(catalog))

        cache = EmbeddingCache(
            embeddings=np.zeros((1, 384)),
            path_ids=["path_0"],
            model_name="test-model",
            document_count=1,
            dd_version=None,  # No version stored
        )

        is_valid, reason = cache.validate_with_reason(
            1, "test-model", source_data_dir=source_dir
        )

        assert is_valid is False
        assert "rebuild" in reason.lower()

    def test_source_content_hash_validation(self, tmp_path):
        """Validation fails when source content hash differs."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        catalog = {"metadata": {"version": "4.0.0"}}
        (source_dir / "ids_catalog.json").write_text(json.dumps(catalog))

        cache = EmbeddingCache(
            embeddings=np.zeros((1, 384)),
            path_ids=["path_0"],
            model_name="test-model",
            document_count=1,
            dd_version="4.0.0",
            source_content_hash="different_hash",
        )

        is_valid, reason = cache.validate_with_reason(
            1, "test-model", source_data_dir=source_dir
        )

        assert is_valid is False
        assert "hash" in reason.lower()

    def test_has_modified_source_files(self, tmp_path):
        """_has_modified_source_files detects modified files."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        (source_dir / "ids_catalog.json").write_text('{"test": true}')

        cache = EmbeddingCache(created_at=0)  # Old creation time

        assert cache._has_modified_source_files(source_dir) is True

    def test_has_modified_source_files_no_modifications(self, tmp_path):
        """_has_modified_source_files returns False for old files."""
        import time

        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        (source_dir / "ids_catalog.json").write_text('{"test": true}')

        cache = EmbeddingCache(created_at=time.time() + 1000)

        assert cache._has_modified_source_files(source_dir) is False

    def test_get_modified_source_files_catalog(self, tmp_path):
        """_get_modified_source_files detects modified catalog."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        (source_dir / "ids_catalog.json").write_text('{"test": true}')

        cache = EmbeddingCache(created_at=0)

        modified = cache._get_modified_source_files(source_dir)

        assert "ids_catalog.json" in modified

    def test_get_modified_source_files_detailed(self, tmp_path):
        """_get_modified_source_files detects modified detailed files."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        detailed = source_dir / "detailed"
        detailed.mkdir()
        (detailed / "test.json").write_text('{"test": true}')

        cache = EmbeddingCache(created_at=0)

        modified = cache._get_modified_source_files(source_dir)

        assert any("test.json" in f for f in modified)

    def test_compute_source_content_hash_includes_ids_set(self, tmp_path):
        """Source hash includes IDS set when configured."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        (source_dir / "ids_catalog.json").write_text('{"metadata": {}}')

        cache1 = EmbeddingCache(ids_set={"equilibrium"})
        cache2 = EmbeddingCache(ids_set={"core_profiles"})

        hash1 = cache1._compute_source_content_hash(source_dir)
        hash2 = cache2._compute_source_content_hash(source_dir)

        assert hash1 != hash2

    def test_compute_source_content_hash_excludes_generation_date(self, tmp_path):
        """Source hash excludes dynamic generation_date metadata."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()

        # Create catalog with different generation dates
        catalog1 = {"metadata": {"version": "4.0.0", "generation_date": "2024-01-01"}}
        catalog2 = {"metadata": {"version": "4.0.0", "generation_date": "2024-12-01"}}

        (source_dir / "ids_catalog.json").write_text(json.dumps(catalog1))
        cache = EmbeddingCache()
        hash1 = cache._compute_source_content_hash(source_dir)

        (source_dir / "ids_catalog.json").write_text(json.dumps(catalog2))
        hash2 = cache._compute_source_content_hash(source_dir)

        assert hash1 == hash2

    def test_get_max_source_mtime(self, tmp_path):
        """_get_max_source_mtime returns maximum modification time."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        (source_dir / "ids_catalog.json").write_text('{"test": true}')
        detailed = source_dir / "detailed"
        detailed.mkdir()
        (detailed / "test.json").write_text('{"test": true}')

        cache = EmbeddingCache()
        max_mtime = cache._get_max_source_mtime(source_dir)

        assert max_mtime > 0

    def test_update_source_metadata(self, tmp_path):
        """update_source_metadata updates all source metadata fields."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        catalog = {"metadata": {"version": "4.0.0"}}
        (source_dir / "ids_catalog.json").write_text(json.dumps(catalog))

        cache = EmbeddingCache()
        cache.update_source_metadata(source_dir)

        assert cache.source_content_hash != ""
        assert cache.source_max_mtime > 0
        assert cache.dd_version == "4.0.0"

    def test_get_current_dd_version(self, tmp_path):
        """_get_current_dd_version extracts version from catalog."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        catalog = {"metadata": {"version": "4.1.0"}}
        (source_dir / "ids_catalog.json").write_text(json.dumps(catalog))

        cache = EmbeddingCache()
        version = cache._get_current_dd_version(source_dir)

        assert version == "4.1.0"

    def test_get_current_dd_version_missing_file(self, tmp_path):
        """_get_current_dd_version returns None for missing catalog."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()

        cache = EmbeddingCache()
        version = cache._get_current_dd_version(source_dir)

        assert version is None

    def test_get_current_dd_version_no_version_field(self, tmp_path):
        """_get_current_dd_version returns None when version field missing."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        (source_dir / "ids_catalog.json").write_text('{"metadata": {}}')

        cache = EmbeddingCache()
        version = cache._get_current_dd_version(source_dir)

        assert version is None

    def test_validation_passes_with_matching_version(self, tmp_path):
        """Validation passes when DD version matches."""
        source_dir = tmp_path / "schemas"
        source_dir.mkdir()
        catalog = {"metadata": {"version": "4.0.0"}}
        (source_dir / "ids_catalog.json").write_text(json.dumps(catalog))

        cache = EmbeddingCache(
            embeddings=np.zeros((1, 384)),
            path_ids=["path_0"],
            model_name="test-model",
            document_count=1,
            dd_version="4.0.0",
        )
        # Also set source_content_hash to match
        cache.source_content_hash = cache._compute_source_content_hash(source_dir)

        is_valid, reason = cache.validate_with_reason(
            1, "test-model", source_data_dir=source_dir
        )

        assert is_valid is True
