"""Tests for clustering.py - multi-membership clustering functionality."""

import numpy as np
import pytest

from imas_codex.clusters.clustering import (
    EmbeddingClusterer,
    RelationshipBuilder,
    _compute_cluster_centroid,
    _compute_cluster_similarity,
)
from imas_codex.clusters.config import RelationshipExtractionConfig
from imas_codex.clusters.models import ClusterInfo
from imas_codex.embeddings.config import EncoderConfig


class TestComputeClusterSimilarity:
    """Tests for _compute_cluster_similarity function."""

    def test_single_item_cluster_returns_perfect_similarity(self):
        """Single item cluster should have perfect similarity."""
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        similarity = _compute_cluster_similarity([0], embeddings)
        assert similarity == 1.0

    def test_identical_vectors_have_perfect_similarity(self):
        """Identical vectors should have perfect similarity."""
        embeddings = np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        )
        similarity = _compute_cluster_similarity([0, 1, 2], embeddings)
        assert abs(similarity - 1.0) < 1e-5

    def test_orthogonal_vectors_have_zero_similarity(self):
        """Orthogonal vectors should have near-zero similarity."""
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        similarity = _compute_cluster_similarity([0, 1], embeddings)
        assert abs(similarity) < 0.01

    def test_similar_vectors_have_high_similarity(self):
        """Similar vectors should have high similarity."""
        embeddings = np.array(
            [[1.0, 0.1, 0.0], [1.0, 0.0, 0.0], [1.0, 0.05, 0.0]], dtype=np.float32
        )
        similarity = _compute_cluster_similarity([0, 1, 2], embeddings)
        assert similarity > 0.9

    def test_similarity_is_clamped_to_valid_range(self):
        """Similarity should be clamped to [0, 1]."""
        embeddings = np.random.randn(5, 10).astype(np.float32)
        similarity = _compute_cluster_similarity([0, 1, 2, 3, 4], embeddings)
        assert 0.0 <= similarity <= 1.0


class TestComputeClusterCentroid:
    """Additional tests for _compute_cluster_centroid."""

    def test_zero_norm_centroid_handling(self):
        """Test handling of zero-norm centroid (opposite vectors)."""
        embeddings = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
        centroid = _compute_cluster_centroid([0, 1], embeddings)
        # Mean is [0, 0, 0], normalization should handle this gracefully
        assert len(centroid) == 3

    def test_centroid_subset_of_indices(self):
        """Test computing centroid for subset of embeddings."""
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # index 0
                [0.0, 1.0, 0.0],  # index 1
                [0.0, 0.0, 1.0],  # index 2
                [1.0, 1.0, 0.0],  # index 3
            ],
            dtype=np.float32,
        )
        centroid = _compute_cluster_centroid([0, 3], embeddings)
        # Mean of [1,0,0] and [1,1,0] = [1, 0.5, 0], normalized
        expected_mean = np.array([1.0, 0.5, 0.0])
        expected_norm = expected_mean / np.linalg.norm(expected_mean)
        np.testing.assert_array_almost_equal(
            centroid, expected_norm.tolist(), decimal=5
        )


class TestEmbeddingClusterer:
    """Tests for EmbeddingClusterer class with HDBSCAN."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RelationshipExtractionConfig(
            encoder_config=EncoderConfig(),
            min_cluster_size=2,
            min_samples=2,
            cluster_selection_method="eom",
        )

    @pytest.fixture
    def clusterer(self, config):
        """Create clusterer instance."""
        return EmbeddingClusterer(config)

    def test_initialization(self, config):
        """Test clusterer initialization."""
        clusterer = EmbeddingClusterer(config)
        assert clusterer.config == config

    def test_initialization_with_cluster_selection_method(self, config):
        """Test clusterer initialization with different selection methods."""
        clusterer_eom = EmbeddingClusterer(config, cluster_selection_method="eom")
        assert clusterer_eom.cluster_selection_method == "eom"

        clusterer_leaf = EmbeddingClusterer(config, cluster_selection_method="leaf")
        assert clusterer_leaf.cluster_selection_method == "leaf"

    def test_extract_ids_name(self, clusterer):
        """Test IDS name extraction from paths."""
        assert (
            clusterer._extract_ids_name("equilibrium/time_slice/psi") == "equilibrium"
        )
        assert (
            clusterer._extract_ids_name("core_profiles/profiles_1d/te")
            == "core_profiles"
        )

    def test_cluster_embeddings_empty(self, clusterer):
        """Test clustering with empty input."""
        embeddings = np.array([]).reshape(0, 384)
        clusters, memberships, stats = clusterer.cluster_embeddings(embeddings, [], {})
        assert "error" in stats

    def test_cluster_embeddings_creates_clusters(self, config):
        """Test that clustering produces clusters from similar embeddings."""
        clusterer = EmbeddingClusterer(config)
        np.random.seed(42)

        # Create embeddings with clear cluster structure - must be well separated
        # Use unit vectors to ensure clear separation for HDBSCAN
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # cluster 1
                [0.99, 0.01, 0.0],  # cluster 1
                [0.98, 0.02, 0.0],  # cluster 1
                [0.0, 1.0, 0.0],  # cluster 2
                [0.01, 0.99, 0.0],  # cluster 2
                [0.02, 0.98, 0.0],  # cluster 2
            ],
            dtype=np.float32,
        )

        paths = [
            "ids_a/field1",
            "ids_a/field2",
            "ids_b/field1",
            "ids_c/field1",
            "ids_c/field2",
            "ids_c/field3",
        ]
        filtered_paths = {p: {"ids": p.split("/")[0]} for p in paths}

        clusters, memberships, stats = clusterer.cluster_embeddings(
            embeddings, paths, filtered_paths
        )

        # HDBSCAN should find at least one cluster
        assert len(clusters) >= 1
        assert len(memberships) == len(paths)

    def test_cluster_derives_cross_ids_correctly(self, config):
        """Test that is_cross_ids is derived from cluster membership."""
        clusterer = EmbeddingClusterer(config)

        # Create identical embeddings from different IDS
        base = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        embeddings = np.vstack([base, base, base]).astype(np.float32)

        paths = ["ids_a/field", "ids_b/field", "ids_c/field"]
        filtered_paths = {p: {"ids": p.split("/")[0]} for p in paths}

        clusters, memberships, stats = clusterer.cluster_embeddings(
            embeddings, paths, filtered_paths
        )

        # Should create a cross-IDS cluster
        if clusters:
            cluster = clusters[0]
            assert cluster.is_cross_ids is True
            assert len(cluster.ids_names) == 3

    def test_calculate_statistics(self, clusterer):
        """Test statistics calculation."""
        from imas_codex.clusters.models import PathMembership

        clusters = [
            ClusterInfo(
                id=0,
                similarity_score=0.9,
                size=3,
                is_cross_ids=True,
                ids_names=["a", "b"],
                paths=["a/x", "a/y", "b/z"],
            ),
            ClusterInfo(
                id=1,
                similarity_score=0.85,
                size=2,
                is_cross_ids=False,
                ids_names=["a"],
                paths=["a/m", "a/n"],
            ),
        ]
        path_index = {
            "a/x": PathMembership(cross_ids_cluster=0, intra_ids_cluster=None),
            "a/m": PathMembership(cross_ids_cluster=None, intra_ids_cluster=1),
            "a/p": PathMembership(cross_ids_cluster=None, intra_ids_cluster=None),
        }

        stats = clusterer._calculate_statistics(clusters, path_index)

        assert "cross_ids_clustering" in stats
        assert "intra_ids_clustering" in stats
        assert "multi_membership_paths" in stats
        assert "isolated_paths" in stats
        assert stats["isolated_paths"] == 1  # a/p has no cluster


class TestRelationshipBuilder:
    """Tests for RelationshipBuilder class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RelationshipExtractionConfig(encoder_config=EncoderConfig())

    @pytest.fixture
    def builder(self, config):
        """Create builder instance."""
        return RelationshipBuilder(config)

    def test_build_path_index_empty(self, builder):
        """Test building path index with no clusters."""
        result = builder.build_path_index({})
        assert result["path_to_cluster"] == {}
        assert result["cluster_to_paths"] == {}

    def test_build_path_index_with_clusters(self, builder):
        """Test building path index with clusters."""
        cluster_infos = {
            0: ClusterInfo(
                id=0,
                similarity_score=0.9,
                size=2,
                is_cross_ids=True,
                ids_names=["a", "b"],
                paths=["a/x", "b/y"],
            ),
            1: ClusterInfo(
                id=1,
                similarity_score=0.8,
                size=2,
                is_cross_ids=False,
                ids_names=["a"],
                paths=["a/m", "a/n"],
            ),
        }

        result = builder.build_path_index(cluster_infos)

        assert result["path_to_cluster"]["a/x"] == 0
        assert result["path_to_cluster"]["b/y"] == 0
        assert result["path_to_cluster"]["a/m"] == 1
        assert result["cluster_to_paths"][0] == ["a/x", "b/y"]
        assert result["cluster_to_paths"][1] == ["a/m", "a/n"]
