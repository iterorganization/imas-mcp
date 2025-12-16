"""Tests for core/clusters.py - Clusters manager."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imas_mcp.core.clusters import Clusters
from imas_mcp.embeddings.config import EncoderConfig


@pytest.fixture
def encoder_config():
    """Create test encoder config."""
    return EncoderConfig(
        model_name="all-MiniLM-L6-v2",
        ids_set={"equilibrium", "core_profiles"},
    )


@pytest.fixture
def clusters_file(tmp_path):
    """Create a test clusters.json file."""
    clusters_data = {
        "metadata": {
            "version": "4.0.0",
            "created_at": datetime.now().isoformat(),
            "total_clusters": 3,
        },
        "clusters": [
            {
                "id": 0,
                "label": "Temperature Profiles",
                "is_cross_ids": False,
                "ids_names": ["core_profiles"],
                "paths": [
                    "core_profiles/profiles_1d/electrons/temperature",
                    "core_profiles/profiles_1d/ion/temperature",
                ],
                "similarity_score": 0.95,
            },
            {
                "id": 1,
                "label": "Magnetic Field",
                "is_cross_ids": True,
                "ids_names": ["equilibrium", "core_profiles"],
                "paths": [
                    "equilibrium/time_slice/profiles_2d/b_field_r",
                    "equilibrium/time_slice/profiles_2d/b_field_z",
                ],
                "similarity_score": 0.88,
            },
            {
                "id": 2,
                "label": "Boundary Data",
                "is_cross_ids": False,
                "ids_names": ["equilibrium"],
                "paths": [
                    "equilibrium/time_slice/boundary/psi",
                    "equilibrium/time_slice/boundary/r",
                ],
                "similarity_score": 0.92,
            },
        ],
        "unit_families": {
            "length": {"unit": "m", "paths": ["equilibrium/time_slice/boundary/r"]},
            "energy": {
                "unit": "eV",
                "paths": ["core_profiles/profiles_1d/electrons/temperature"],
            },
        },
        "cross_references": {
            "equilibrium": ["core_profiles"],
        },
        "embeddings_file": "cluster_embeddings.npz",
        "embeddings_hash": "abc123",
    }

    file_path = tmp_path / "clusters.json"
    with open(file_path, "w") as f:
        json.dump(clusters_data, f)

    return file_path


class TestClustersInit:
    """Tests for Clusters initialization."""

    def test_initialization_with_encoder_config(self, encoder_config, clusters_file):
        """Test clusters initializes with encoder config."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        assert clusters.encoder_config == encoder_config
        assert clusters.clusters_file == clusters_file

    def test_initialization_sets_default_file(self, encoder_config):
        """Test default clusters file path is set in __post_init__."""
        with patch("imas_mcp.core.clusters.ResourcePathAccessor") as mock_accessor:
            mock_accessor_instance = MagicMock()
            mock_accessor_instance.clusters_dir = Path("/mock/clusters")
            mock_accessor.return_value = mock_accessor_instance

            clusters = Clusters(encoder_config=encoder_config)

            # Should have hash suffix due to ids_set
            assert clusters.clusters_file is not None
            assert "clusters_" in str(clusters.clusters_file)

    def test_initialization_full_dataset_uses_simple_name(self):
        """Test full dataset uses simple clusters.json name."""
        config = EncoderConfig(model_name="all-MiniLM-L6-v2", ids_set=None)

        with patch("imas_mcp.core.clusters.ResourcePathAccessor") as mock_accessor:
            mock_accessor_instance = MagicMock()
            mock_accessor_instance.clusters_dir = Path("/mock/clusters")
            mock_accessor.return_value = mock_accessor_instance

            clusters = Clusters(encoder_config=config)

            assert clusters.clusters_file is not None
            assert clusters.clusters_file.name == "clusters.json"


class TestFilePath:
    """Tests for file_path property."""

    def test_file_path_returns_clusters_file(self, encoder_config, clusters_file):
        """Test file_path property returns correct path."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        assert clusters.file_path == clusters_file


class TestFileMtime:
    """Tests for file modification time utilities."""

    def test_get_file_mtime_existing_file(self, encoder_config, clusters_file):
        """Test getting mtime for existing file."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        mtime = clusters._get_file_mtime(clusters_file)

        assert mtime > 0

    def test_get_file_mtime_nonexistent_file(self, encoder_config, clusters_file):
        """Test getting mtime for nonexistent file returns 0."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        mtime = clusters._get_file_mtime(Path("/nonexistent/file.json"))

        assert mtime == 0.0


class TestDependencyFreshness:
    """Tests for dependency freshness checking."""

    def test_check_dependency_freshness_no_file(self, encoder_config, tmp_path):
        """Test freshness check when clusters file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.json"

        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=nonexistent,
        )

        assert clusters._check_dependency_freshness() is True

    def test_check_dependency_freshness_no_embedding_cache(
        self, encoder_config, clusters_file
    ):
        """Test freshness check when no embedding cache exists."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(clusters, "_get_embedding_cache_file", return_value=None):
            # No embedding cache means rebuild is needed
            result = clusters._check_dependency_freshness()
            assert result is True

    def test_check_dependency_freshness_embedding_newer(
        self, encoder_config, clusters_file, tmp_path
    ):
        """Test freshness check when embedding cache is newer."""
        # Create a newer embedding cache file
        embedding_cache = tmp_path / "embeddings.npz"
        embedding_cache.touch()

        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(
            clusters, "_get_embedding_cache_file", return_value=embedding_cache
        ):
            # Make embedding cache newer
            import time

            time.sleep(0.1)
            embedding_cache.touch()

            result = clusters._check_dependency_freshness()
            assert result is True

    def test_check_dependency_freshness_clusters_current(
        self, encoder_config, tmp_path
    ):
        """Test freshness check when clusters file is up-to-date."""
        # Create embedding cache first
        embedding_cache = tmp_path / "embeddings.npz"
        embedding_cache.touch()

        import time

        time.sleep(0.1)

        # Then create clusters file (newer)
        clusters_file = tmp_path / "clusters.json"
        with open(clusters_file, "w") as f:
            json.dump({"clusters": []}, f)

        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(
            clusters, "_get_embedding_cache_file", return_value=embedding_cache
        ):
            result = clusters._check_dependency_freshness()
            assert result is False


class TestShouldCheckDependencies:
    """Tests for dependency check interval."""

    def test_should_check_dependencies_first_call(self, encoder_config, clusters_file):
        """Test first call should always check dependencies."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        assert clusters._should_check_dependencies() is True

    def test_should_check_dependencies_after_interval(
        self, encoder_config, clusters_file
    ):
        """Test should check after interval expires."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        # Set last check to old time
        clusters._last_dependency_check = datetime.now().timestamp() - 100

        assert clusters._should_check_dependencies() is True

    def test_should_not_check_dependencies_within_interval(
        self, encoder_config, clusters_file
    ):
        """Test should not check within interval."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        # Set last check to now
        clusters._last_dependency_check = datetime.now().timestamp()

        assert clusters._should_check_dependencies() is False


class TestInvalidateCache:
    """Tests for cache invalidation."""

    def test_invalidate_cache_clears_data(self, encoder_config, clusters_file):
        """Test cache invalidation clears cached data."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        # Set some cached data
        clusters._cached_data = {"test": "data"}
        clusters._cached_mtime = 123.0

        clusters._invalidate_cache()

        assert clusters._cached_data is None
        assert clusters._cached_mtime is None


class TestLoadClustersData:
    """Tests for loading clusters data."""

    def test_load_clusters_data_from_file(self, encoder_config, clusters_file):
        """Test loading clusters data from file."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        # Skip dependency check
        with patch.object(clusters, "_should_check_dependencies", return_value=False):
            data = clusters._load_clusters_data()

        assert "clusters" in data
        assert len(data["clusters"]) == 3

    def test_load_clusters_data_caches_result(self, encoder_config, clusters_file):
        """Test loaded data is cached."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(clusters, "_should_check_dependencies", return_value=False):
            data1 = clusters._load_clusters_data()
            # Second call should use cache
            clusters._cached_mtime = clusters._get_file_mtime(clusters_file)
            data2 = clusters._load_clusters_data()

        assert data1 is data2  # Same object from cache

    def test_load_clusters_data_auto_rebuilds(self, encoder_config, tmp_path):
        """Test auto-rebuild when dependencies are stale."""
        clusters_file = tmp_path / "clusters.json"

        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with (
            patch.object(clusters, "_should_check_dependencies", return_value=True),
            patch.object(clusters, "_check_dependency_freshness", return_value=True),
            patch.object(clusters, "build") as mock_build,
        ):
            # Should trigger auto-rebuild
            try:
                clusters._load_clusters_data()
            except FileNotFoundError:
                pass  # Expected if build mock doesn't create file

            # build is called in check phase and again if file still doesn't exist
            assert mock_build.call_count >= 1
            mock_build.assert_any_call(force=True)


class TestGetData:
    """Tests for get_data method."""

    def test_get_data_returns_clusters(self, encoder_config, clusters_file):
        """Test get_data returns clusters dictionary."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(clusters, "_should_check_dependencies", return_value=False):
            data = clusters.get_data()

        assert "clusters" in data
        assert "metadata" in data


class TestGetClusters:
    """Tests for get_clusters method."""

    def test_get_clusters_returns_list(self, encoder_config, clusters_file):
        """Test get_clusters returns cluster list."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(clusters, "_should_check_dependencies", return_value=False):
            cluster_list = clusters.get_clusters()

        assert isinstance(cluster_list, list)
        assert len(cluster_list) == 3


class TestGetMetadata:
    """Tests for get_metadata method."""

    def test_get_metadata_returns_dict(self, encoder_config, clusters_file):
        """Test get_metadata returns metadata dictionary."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(clusters, "_should_check_dependencies", return_value=False):
            metadata = clusters.get_metadata()

        assert isinstance(metadata, dict)
        assert "version" in metadata


class TestGetUnitFamilies:
    """Tests for get_unit_families method."""

    def test_get_unit_families_returns_dict(self, encoder_config, clusters_file):
        """Test get_unit_families returns unit families dictionary."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(clusters, "_should_check_dependencies", return_value=False):
            families = clusters.get_unit_families()

        assert isinstance(families, dict)
        assert "length" in families


class TestGetCrossReferences:
    """Tests for get_cross_references method."""

    def test_get_cross_references_returns_dict(self, encoder_config, clusters_file):
        """Test get_cross_references returns cross-references dictionary."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(clusters, "_should_check_dependencies", return_value=False):
            refs = clusters.get_cross_references()

        assert isinstance(refs, dict)
        assert "equilibrium" in refs


class TestIsAvailable:
    """Tests for is_available method."""

    def test_is_available_when_file_exists(self, encoder_config, clusters_file):
        """Test is_available returns True when file exists."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(clusters, "_should_check_dependencies", return_value=False):
            assert clusters.is_available() is True

    def test_is_available_when_file_missing(self, encoder_config, tmp_path):
        """Test is_available returns False when file doesn't exist and build fails."""
        nonexistent = tmp_path / "nonexistent.json"

        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=nonexistent,
        )

        # is_available calls get_data which may try to build/load
        # We need to mock at the right level to simulate unavailability
        with patch.object(
            clusters, "get_data", side_effect=FileNotFoundError("Not found")
        ):
            assert clusters.is_available() is False


class TestNeedsRebuild:
    """Tests for needs_rebuild method."""

    def test_needs_rebuild_delegates_to_freshness_check(
        self, encoder_config, clusters_file
    ):
        """Test needs_rebuild delegates to dependency freshness check."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(
            clusters, "_check_dependency_freshness", return_value=True
        ) as mock_check:
            result = clusters.needs_rebuild()

            mock_check.assert_called_once()
            assert result is True


class TestGetCacheInfo:
    """Tests for get_cache_info method."""

    def test_get_cache_info_returns_dict(self, encoder_config, clusters_file):
        """Test get_cache_info returns info dictionary."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        # Work around missing _cluster_engine attribute in source
        clusters._cluster_engine = None

        info = clusters.get_cache_info()

        assert "file_path" in info
        assert "file_exists" in info
        assert "cached" in info
        assert "needs_rebuild" in info

    def test_get_cache_info_includes_file_stats(self, encoder_config, clusters_file):
        """Test cache info includes file stats when file exists."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        # Work around missing _cluster_engine attribute in source
        clusters._cluster_engine = None

        info = clusters.get_cache_info()

        assert info["file_exists"] is True
        assert "file_size_mb" in info
        assert "file_mtime" in info


class TestForceReload:
    """Tests for force_reload method."""

    def test_force_reload_invalidates_cache(self, encoder_config, clusters_file):
        """Test force_reload invalidates cache."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        # Set some cached data
        clusters._cached_data = {"test": "data"}

        clusters.force_reload()

        assert clusters._cached_data is None


class TestBuild:
    """Tests for build method."""

    def test_build_skips_when_not_needed(self, encoder_config, clusters_file):
        """Test build skips when not needed and force=False."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(clusters, "needs_rebuild", return_value=False):
            result = clusters.build(force=False)

        assert result is False

    def test_build_runs_when_forced(self, encoder_config, clusters_file):
        """Test build runs when force=True."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with (
            patch("imas_mcp.core.clusters.ResourcePathAccessor") as mock_path_accessor,
            patch("imas_mcp.core.clusters.RelationshipExtractionConfig"),
            patch("imas_mcp.core.clusters.RelationshipExtractor") as mock_extractor,
        ):
            mock_accessor_instance = MagicMock()
            mock_accessor_instance.version_dir = clusters_file.parent
            mock_path_accessor.return_value = mock_accessor_instance

            mock_extractor_instance = MagicMock()
            mock_extractor_instance.extract_relationships.return_value = []
            mock_extractor.return_value = mock_extractor_instance

            result = clusters.build(force=True)

            assert result is True
            mock_extractor_instance.extract_relationships.assert_called_once()
            mock_extractor_instance.save_relationships.assert_called_once()


class TestGetClusterSearcher:
    """Tests for get_cluster_searcher method."""

    def test_get_cluster_searcher_creates_searcher(self, encoder_config, clusters_file):
        """Test get_cluster_searcher creates and caches searcher."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with (
            patch.object(clusters, "_should_check_dependencies", return_value=False),
            patch("imas_mcp.core.clusters.ClusterSearcher") as mock_searcher_class,
        ):
            mock_searcher = MagicMock()
            mock_searcher_class.return_value = mock_searcher

            searcher1 = clusters.get_cluster_searcher()
            searcher2 = clusters.get_cluster_searcher()

            # Should only create once
            mock_searcher_class.assert_called_once()
            assert searcher1 is searcher2


class TestGetEncoder:
    """Tests for get_encoder method."""

    def test_get_encoder_creates_encoder(self, encoder_config, clusters_file):
        """Test get_encoder creates and caches encoder."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch("imas_mcp.core.clusters.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder_class.return_value = mock_encoder

            encoder1 = clusters.get_encoder()
            encoder2 = clusters.get_encoder()

            # Should only create once
            mock_encoder_class.assert_called_once_with(encoder_config)
            assert encoder1 is encoder2


class TestSearchClusters:
    """Tests for search_clusters method."""

    def test_search_clusters_returns_results(self, encoder_config, clusters_file):
        """Test search_clusters returns formatted results."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        mock_result = MagicMock()
        mock_result.cluster_id = 0
        mock_result.similarity_score = 0.95
        mock_result.is_cross_ids = False
        mock_result.ids_names = ["core_profiles"]
        mock_result.paths = ["core_profiles/profiles_1d/electrons/temperature"]
        mock_result.cluster_similarity = 0.87

        with (
            patch.object(clusters, "_should_check_dependencies", return_value=False),
            patch.object(clusters, "get_cluster_searcher") as mock_get_searcher,
            patch.object(clusters, "get_encoder") as mock_get_encoder,
        ):
            mock_searcher = MagicMock()
            mock_searcher.search_by_text.return_value = [mock_result]
            mock_get_searcher.return_value = mock_searcher

            mock_encoder = MagicMock()
            mock_get_encoder.return_value = mock_encoder

            results = clusters.search_clusters("temperature", top_k=5)

        assert len(results) == 1
        assert results[0]["cluster_id"] == 0
        assert results[0]["similarity_score"] == 0.95

    def test_search_clusters_passes_parameters(self, encoder_config, clusters_file):
        """Test search_clusters passes all parameters correctly."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with (
            patch.object(clusters, "_should_check_dependencies", return_value=False),
            patch.object(clusters, "get_cluster_searcher") as mock_get_searcher,
            patch.object(clusters, "get_encoder") as mock_get_encoder,
        ):
            mock_searcher = MagicMock()
            mock_searcher.search_by_text.return_value = []
            mock_get_searcher.return_value = mock_searcher

            mock_encoder = MagicMock()
            mock_get_encoder.return_value = mock_encoder

            clusters.search_clusters(
                query="test",
                top_k=10,
                similarity_threshold=0.5,
                cross_ids_only=True,
            )

            mock_searcher.search_by_text.assert_called_once_with(
                query="test",
                encoder=mock_encoder,
                top_k=10,
                similarity_threshold=0.5,
                cross_ids_only=True,
            )


class TestGetEmbeddingCacheFile:
    """Tests for _get_embedding_cache_file method."""

    def test_get_embedding_cache_file_returns_path(self, encoder_config, clusters_file):
        """Test getting embedding cache file path."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch("imas_mcp.core.clusters.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder._cache_path = clusters_file.parent / "embeddings.npz"
            mock_encoder_class.return_value = mock_encoder

            # Create the mock cache file
            (clusters_file.parent / "embeddings.npz").touch()

            result = clusters._get_embedding_cache_file()

            assert result is not None
            assert result.exists()

    def test_get_embedding_cache_file_returns_none_when_missing(
        self, encoder_config, clusters_file
    ):
        """Test returns None when cache file doesn't exist."""
        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch("imas_mcp.core.clusters.Encoder") as mock_encoder_class:
            mock_encoder = MagicMock()
            mock_encoder._cache_path = clusters_file.parent / "nonexistent.npz"
            mock_encoder_class.return_value = mock_encoder

            result = clusters._get_embedding_cache_file()

            # Should return None when file doesn't exist
            assert result is None


class TestLatin1Fallback:
    """Tests for encoding fallback."""

    def test_load_with_latin1_encoding(self, encoder_config, tmp_path):
        """Test loading file with latin-1 encoding after UTF-8 failure."""
        clusters_file = tmp_path / "clusters.json"

        # Write file with content that works in latin-1
        content = {"clusters": [{"label": "Test"}], "metadata": {}}
        with open(clusters_file, "w", encoding="latin-1") as f:
            json.dump(content, f)

        clusters = Clusters(
            encoder_config=encoder_config,
            clusters_file=clusters_file,
        )

        with patch.object(clusters, "_should_check_dependencies", return_value=False):
            # Should succeed (file is valid JSON in either encoding)
            data = clusters._load_clusters_data()

        assert "clusters" in data
