"""Tests for clustering.py - multi-membership clustering functionality."""

import numpy as np
import pytest

from imas_mcp.clusters.clustering import (
    EmbeddingClusterer,
    RelationshipBuilder,
    _compute_cluster_centroid,
    _compute_cluster_similarity,
)
from imas_mcp.clusters.config import RelationshipExtractionConfig
from imas_mcp.clusters.models import ClusterInfo
from imas_mcp.embeddings.config import EncoderConfig


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
    """Tests for EmbeddingClusterer class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RelationshipExtractionConfig(
            encoder_config=EncoderConfig(),
            cross_ids_eps=0.3,
            cross_ids_min_samples=2,
            intra_ids_eps=0.2,
            intra_ids_min_samples=2,
        )

    @pytest.fixture
    def clusterer(self, config):
        """Create clusterer instance."""
        return EmbeddingClusterer(config)

    def test_initialization(self, config):
        """Test clusterer initialization."""
        clusterer = EmbeddingClusterer(config)
        assert clusterer.config == config

    def test_extract_concept_from_path(self, clusterer):
        """Test concept extraction from paths."""
        path = "equilibrium/time_slice/boundary/psi"
        concept = clusterer._extract_concept(path)
        # Should extract last 2 meaningful components
        assert "psi" in concept or "boundary" in concept

    def test_extract_concept_skips_time_slice(self, clusterer):
        """Test that time_slice is filtered from concepts."""
        path = "core_profiles/time_slice/profiles_1d/temperature"
        concept = clusterer._extract_concept(path)
        assert "time_slice" not in concept

    def test_get_cross_ids_candidates_empty_list(self, clusterer):
        """Test cross-IDS candidates with empty input."""
        candidates = clusterer._get_cross_ids_candidates([], {})
        assert len(candidates) == 0

    def test_get_cross_ids_candidates_finds_shared_concepts(self, clusterer):
        """Test finding paths with shared concepts across IDS."""
        path_list = [
            "equilibrium/profiles_1d/temperature",
            "core_profiles/profiles_1d/temperature",
            "equilibrium/profiles_1d/density",
            "core_profiles/profiles_1d/density",
        ]
        filtered_paths = {path: {} for path in path_list}
        candidates = clusterer._get_cross_ids_candidates(path_list, filtered_paths)
        # Temperature and density appear in multiple IDS
        assert len(candidates) > 0

    def test_build_unified_path_index(self, clusterer):
        """Test building unified path index."""
        path_list = ["a/b/c", "d/e/f", "g/h/i"]
        cross_memberships = {"a/b/c": 0}
        intra_memberships = {"d/e/f": 1000}

        path_index = clusterer._build_unified_path_index(
            path_list, cross_memberships, intra_memberships
        )

        assert "a/b/c" in path_index
        assert path_index["a/b/c"].cross_ids_cluster == 0
        assert path_index["a/b/c"].intra_ids_cluster is None
        assert path_index["d/e/f"].intra_ids_cluster == 1000

    def test_calculate_statistics(self, clusterer):
        """Test statistics calculation."""
        from imas_mcp.clusters.models import PathMembership

        cross_clusters = [
            ClusterInfo(
                id=0,
                similarity_score=0.9,
                size=3,
                is_cross_ids=True,
                ids_names=["a", "b"],
                paths=["a/x", "a/y", "b/z"],
            )
        ]
        intra_clusters = [
            ClusterInfo(
                id=1000,
                similarity_score=0.85,
                size=2,
                is_cross_ids=False,
                ids_names=["a"],
                paths=["a/m", "a/n"],
            )
        ]
        path_index = {
            "a/x": PathMembership(cross_ids_cluster=0, intra_ids_cluster=None),
            "a/m": PathMembership(cross_ids_cluster=None, intra_ids_cluster=1000),
            "a/p": PathMembership(cross_ids_cluster=None, intra_ids_cluster=None),
        }

        stats = clusterer._calculate_statistics(
            cross_clusters, intra_clusters, path_index
        )

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
