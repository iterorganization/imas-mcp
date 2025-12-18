"""Tests for label_cache.py - content-addressed label caching."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from imas_codex.clusters.label_cache import (
    CachedLabel,
    LabelCache,
    compute_cluster_hash,
)


class TestComputeClusterHash:
    """Tests for compute_cluster_hash function."""

    def test_hash_is_deterministic(self):
        """Same paths should produce same hash."""
        paths = ["a/b/c", "d/e/f", "g/h/i"]
        hash1 = compute_cluster_hash(paths)
        hash2 = compute_cluster_hash(paths)
        assert hash1 == hash2

    def test_hash_is_order_independent(self):
        """Hash should be same regardless of path order."""
        paths1 = ["a/b/c", "d/e/f", "g/h/i"]
        paths2 = ["g/h/i", "a/b/c", "d/e/f"]
        hash1 = compute_cluster_hash(paths1)
        hash2 = compute_cluster_hash(paths2)
        assert hash1 == hash2

    def test_different_paths_produce_different_hash(self):
        """Different paths should produce different hashes."""
        hash1 = compute_cluster_hash(["a/b/c", "d/e/f"])
        hash2 = compute_cluster_hash(["x/y/z", "m/n/o"])
        assert hash1 != hash2

    def test_hash_length(self):
        """Hash should be 16 characters (truncated SHA256)."""
        hash_val = compute_cluster_hash(["a/b/c"])
        assert len(hash_val) == 16

    def test_empty_paths(self):
        """Empty paths should produce a valid hash."""
        hash_val = compute_cluster_hash([])
        assert len(hash_val) == 16


class TestCachedLabel:
    """Tests for CachedLabel dataclass."""

    def test_cached_label_creation(self):
        """Test creating a CachedLabel."""
        label = CachedLabel(
            label="Temperature Profiles",
            description="A cluster of temperature paths",
            model="gpt-4",
            created_at="2025-01-01T00:00:00",
        )
        assert label.label == "Temperature Profiles"
        assert label.description == "A cluster of temperature paths"
        assert label.model == "gpt-4"
        assert label.created_at == "2025-01-01T00:00:00"


class TestLabelCache:
    """Tests for LabelCache class."""

    @pytest.fixture
    def cache_file(self, tmp_path):
        """Create a temporary cache file path."""
        return tmp_path / "test_label_cache.db"

    @pytest.fixture
    def json_file(self, tmp_path):
        """Create a temporary JSON file path for persistence."""
        return tmp_path / "test_labels.json"

    @pytest.fixture
    def cache(self, cache_file, json_file):
        """Create a LabelCache instance with isolated file paths."""
        return LabelCache(cache_file, json_file)

    def test_initialization_creates_file(self, cache_file, json_file):
        """Test that initialization creates the database file."""
        LabelCache(cache_file, json_file)
        assert cache_file.exists()

    def test_get_label_nonexistent(self, cache):
        """Test getting a label that doesn't exist."""
        result = cache.get_label(["nonexistent/path"], model="test-model")
        assert result is None

    def test_set_and_get_label(self, cache):
        """Test setting and getting a label."""
        paths = ["a/b/c", "d/e/f"]
        cache.set_label(
            paths=paths,
            label="Test Label",
            description="Test Description",
            model="test-model",
        )

        result = cache.get_label(paths, model="test-model")
        assert result is not None
        assert result.label == "Test Label"
        assert result.description == "Test Description"
        assert result.model == "test-model"

    def test_get_label_without_model_filter(self, cache):
        """Test getting a label without model filter."""
        paths = ["x/y/z"]
        cache.set_label(
            paths=paths,
            label="Any Model Label",
            description="Desc",
            model="some-model",
        )

        result = cache.get_label(paths, model=None)  # No model filter
        assert result is not None
        assert result.label == "Any Model Label"

    def test_get_label_with_wrong_model(self, cache):
        """Test getting a label with wrong model filter."""
        paths = ["x/y/z"]
        cache.set_label(
            paths=paths,
            label="Model Specific",
            description="Desc",
            model="model-a",
        )

        result = cache.get_label(paths, model="model-b")
        assert result is None

    def test_set_label_returns_hash(self, cache):
        """Test that set_label returns the path hash."""
        paths = ["a/b", "c/d"]
        path_hash = cache.set_label(
            paths=paths,
            label="Label",
            description="Desc",
            model="test",
        )

        expected_hash = compute_cluster_hash(paths)
        assert path_hash == expected_hash

    def test_get_many_all_cached(self, cache):
        """Test get_many when all clusters are cached."""
        clusters = [
            {"id": 0, "paths": ["a/b"]},
            {"id": 1, "paths": ["c/d"]},
        ]

        # Cache all labels
        cache.set_label(["a/b"], "Label 0", "Desc 0", model="test")
        cache.set_label(["c/d"], "Label 1", "Desc 1", model="test")

        cached, uncached = cache.get_many(clusters, model="test")

        assert len(cached) == 2
        assert len(uncached) == 0
        assert 0 in cached
        assert 1 in cached
        assert cached[0].label == "Label 0"
        assert cached[1].label == "Label 1"

    def test_get_many_none_cached(self, cache):
        """Test get_many when no clusters are cached."""
        clusters = [
            {"id": 0, "paths": ["new/path/1"]},
            {"id": 1, "paths": ["new/path/2"]},
        ]

        cached, uncached = cache.get_many(clusters, model="test")

        assert len(cached) == 0
        assert len(uncached) == 2

    def test_get_many_partial_cached(self, cache):
        """Test get_many when some clusters are cached."""
        clusters = [
            {"id": 0, "paths": ["cached/path"]},
            {"id": 1, "paths": ["uncached/path"]},
        ]

        cache.set_label(["cached/path"], "Cached Label", "Desc", model="test")

        cached, uncached = cache.get_many(clusters, model="test")

        assert len(cached) == 1
        assert len(uncached) == 1
        assert 0 in cached
        assert uncached[0]["id"] == 1

    def test_set_many(self, cache):
        """Test setting multiple labels at once."""
        labels = [
            (["a/b"], "Label A", "Desc A"),
            (["c/d"], "Label B", "Desc B"),
            (["e/f"], "Label C", "Desc C"),
        ]

        count = cache.set_many(labels, model="test")

        assert count == 3
        assert cache.get_label(["a/b"], "test").label == "Label A"
        assert cache.get_label(["c/d"], "test").label == "Label B"
        assert cache.get_label(["e/f"], "test").label == "Label C"

    def test_get_stats(self, cache):
        """Test getting cache statistics."""
        cache.set_label(["a/b"], "Label 1", "Desc 1", model="model-a")
        cache.set_label(["c/d"], "Label 2", "Desc 2", model="model-a")
        cache.set_label(["e/f"], "Label 3", "Desc 3", model="model-b")

        stats = cache.get_stats()

        assert stats["total_labels"] == 3
        assert stats["by_model"]["model-a"] == 2
        assert stats["by_model"]["model-b"] == 1
        assert "cache_file" in stats
        assert "cache_size_mb" in stats

    def test_clear_all(self, cache):
        """Test clearing all cached labels."""
        cache.set_label(["a/b"], "Label 1", "Desc 1", model="model-a")
        cache.set_label(["c/d"], "Label 2", "Desc 2", model="model-b")

        count = cache.clear()

        assert count == 2
        assert cache.get_stats()["total_labels"] == 0

    def test_clear_by_model(self, cache):
        """Test clearing labels for a specific model."""
        cache.set_label(["a/b"], "Label 1", "Desc 1", model="model-a")
        cache.set_label(["c/d"], "Label 2", "Desc 2", model="model-a")
        cache.set_label(["e/f"], "Label 3", "Desc 3", model="model-b")

        count = cache.clear(model="model-a")

        assert count == 2
        stats = cache.get_stats()
        assert stats["total_labels"] == 1
        assert stats["by_model"]["model-b"] == 1

    def test_update_existing_label(self, cache):
        """Test updating an existing label (upsert)."""
        paths = ["a/b/c"]
        cache.set_label(paths, "Original Label", "Original Desc", model="test")
        cache.set_label(paths, "Updated Label", "Updated Desc", model="test")

        result = cache.get_label(paths, model="test")
        assert result.label == "Updated Label"
        assert result.description == "Updated Desc"

    def test_persistence(self, cache_file, json_file):
        """Test that labels persist across cache instances."""
        # Create and populate cache
        cache1 = LabelCache(cache_file, json_file)
        cache1.set_label(["persist/test"], "Persisted", "Desc", model="test")

        # Create new cache instance with same file
        cache2 = LabelCache(cache_file, json_file)
        result = cache2.get_label(["persist/test"], model="test")

        assert result is not None
        assert result.label == "Persisted"


class TestLabelCacheExportImport:
    """Tests for export/import functionality."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a LabelCache instance with temp files."""
        return LabelCache(tmp_path / "test_cache.db", tmp_path / "test_labels.json")

    @pytest.fixture
    def export_file(self, tmp_path):
        """Create a temporary export file path."""
        return tmp_path / "exported_labels.json"

    def test_export_labels_empty(self, cache, export_file):
        """Test exporting when cache is empty."""
        result = cache.export_labels(export_file)

        assert result == {}
        assert export_file.exists()

    def test_export_labels_with_data(self, cache, export_file):
        """Test exporting labels to JSON in slim format."""
        cache.set_label(["a/b/c", "d/e/f"], "Label 1", "Desc 1", model="test-model")
        cache.set_label(["x/y/z"], "Label 2", "Desc 2", model="test-model")

        result = cache.export_labels(export_file)

        assert len(result) == 2
        # Check that all entries have slim format: [label, description]
        for _path_hash, entry in result.items():
            assert isinstance(entry, list)
            assert len(entry) == 2
            assert isinstance(entry[0], str)  # label
            assert isinstance(entry[1], str)  # description

    def test_export_overwrites_existing(self, cache, export_file):
        """Test that export overwrites existing file content (slim format)."""
        import json

        # Write existing content (will be overwritten)
        existing = {
            "existing_hash": ["Existing Label", "Existing Desc"],
        }
        with export_file.open("w") as f:
            json.dump(existing, f)

        # Add new labels to cache
        cache.set_label(["new/path"], "New Label", "New Desc", model="new-model")

        result = cache.export_labels(export_file)

        # Should only have new entry (export overwrites, not merges)
        assert len(result) == 1
        assert list(result.values())[0] == ["New Label", "New Desc"]

    def test_import_labels_empty(self, cache):
        """Test importing empty data."""
        count = cache.import_labels({})
        assert count == 0

    def test_import_labels(self, cache):
        """Test importing labels from dict."""
        data = {
            "hash1": {
                "label": "Imported Label 1",
                "description": "Imported Desc 1",
                "model": "import-model",
                "created_at": "2025-01-01T00:00:00",
                "paths": ["a/b/c"],
            },
            "hash2": {
                "label": "Imported Label 2",
                "description": "Imported Desc 2",
                "model": "import-model",
                "created_at": "2025-01-01T00:00:00",
                "paths": ["x/y/z"],
            },
        }

        count = cache.import_labels(data)

        assert count == 2
        stats = cache.get_stats()
        assert stats["total_labels"] == 2

    def test_import_skips_existing(self, cache):
        """Test that import skips existing hashes (additive only)."""
        # Add a label first
        cache.set_label(["a/b/c"], "Original", "Original Desc", model="original")
        original_hash = compute_cluster_hash(["a/b/c"])

        # Try to import with same hash
        data = {
            original_hash: {
                "label": "Should Not Override",
                "description": "Should Not Override",
                "model": "import-model",
                "paths": ["a/b/c"],
            },
        }

        count = cache.import_labels(data)

        assert count == 0  # Nothing imported, hash already exists
        result = cache.get_label(["a/b/c"])
        assert result.label == "Original"  # Original preserved

    def test_export_import_roundtrip(self, cache, export_file, tmp_path):
        """Test full roundtrip: export from one cache, import to another."""
        # Populate first cache
        cache.set_label(["a/b"], "Label A", "Desc A", model="test")
        cache.set_label(["c/d"], "Label B", "Desc B", model="test")

        # Export
        cache.export_labels(export_file)

        # Create new empty cache
        import json

        cache2 = LabelCache(tmp_path / "new_cache.db", tmp_path / "new_labels.json")
        with export_file.open() as f:
            data = json.load(f)

        count = cache2.import_labels(data)

        assert count == 2
        # Verify labels are accessible
        result = cache2.get_label(["a/b"])
        assert result is not None
        assert result.label == "Label A"
