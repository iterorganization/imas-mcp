"""Tests for cluster search functionality using centroid embeddings."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from imas_codex.clusters import ClusterSearcher, ClusterSearchResult
from imas_codex.clusters.clustering import (
    _compute_cluster_centroid,
    _compute_cluster_similarity,
)
from imas_codex.clusters.models import ClusterInfo


class TestCentroidComputation:
    """Tests for centroid computation."""

    def test_compute_centroid_single_point(self):
        """Centroid of single point is the point itself."""
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        centroid = _compute_cluster_centroid([0], embeddings)

        assert len(centroid) == 3
        np.testing.assert_array_almost_equal(centroid, [1.0, 0.0, 0.0])

    def test_compute_centroid_multiple_points(self):
        """Centroid is the normalized mean of points."""
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        centroid = _compute_cluster_centroid([0, 1], embeddings)

        # Mean is [0.5, 0.5, 0.0], normalized
        expected = np.array([0.5, 0.5, 0.0])
        expected = expected / np.linalg.norm(expected)

        np.testing.assert_array_almost_equal(centroid, expected.tolist())

    def test_centroid_is_normalized(self):
        """Centroid should be L2-normalized."""
        embeddings = np.array(
            [
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
            dtype=np.float32,
        )
        centroid = _compute_cluster_centroid([0, 1], embeddings)

        norm = np.linalg.norm(centroid)
        assert abs(norm - 1.0) < 1e-6

    def test_centroid_unnormalized(self):
        """Can compute unnormalized centroid."""
        embeddings = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        centroid = _compute_cluster_centroid([0, 1], embeddings, normalize=False)

        expected = [1.0, 1.0, 0.0]
        np.testing.assert_array_almost_equal(centroid, expected)


class TestClusterInfoWithCentroid:
    """Tests for ClusterInfo model with centroid field."""

    def test_cluster_info_with_centroid(self):
        """ClusterInfo should accept centroid field."""
        cluster = ClusterInfo(
            id="test-uuid-001",
            similarity_score=0.9,
            size=2,
            is_cross_ids=True,
            ids_names=["core_profiles", "equilibrium"],
            paths=["core_profiles/a", "equilibrium/b"],
            centroid=[0.1, 0.2, 0.3],
            scope="global",
        )

        assert cluster.centroid == [0.1, 0.2, 0.3]
        assert cluster.id == "test-uuid-001"
        assert cluster.is_cross_ids

    def test_cluster_info_without_centroid(self):
        """ClusterInfo should work without centroid."""
        cluster = ClusterInfo(
            id="test-uuid-002",
            similarity_score=0.9,
            size=2,
            is_cross_ids=False,
            ids_names=["core_profiles"],
            paths=["core_profiles/a", "core_profiles/b"],
            scope="global",
        )

        assert cluster.centroid is None
        assert cluster.id == "test-uuid-002"


class TestClusterSearcher:
    """Tests for ClusterSearcher."""

    @pytest.fixture
    def sample_clusters(self):
        """Create sample clusters (without embedded centroids - now in .npz)."""
        return [
            {
                "id": "test-uuid-000",
                "similarity_score": 0.9,
                "is_cross_ids": True,
                "ids_names": ["core_profiles", "equilibrium"],
                "paths": ["core_profiles/density", "equilibrium/psi"],
                "scope": "global",
            },
            {
                "id": "test-uuid-001",
                "similarity_score": 0.85,
                "is_cross_ids": False,
                "ids_names": ["core_profiles"],
                "paths": ["core_profiles/temperature", "core_profiles/pressure"],
                "scope": "global",
            },
            {
                "id": "test-uuid-002",
                "similarity_score": 0.8,
                "is_cross_ids": True,
                "ids_names": ["equilibrium", "mhd"],
                "paths": ["equilibrium/boundary", "mhd/stability"],
                "scope": "global",
            },
        ]

    @pytest.fixture
    def sample_embeddings_file(self, tmp_path):
        """Create a temporary .npz file with sample embeddings."""
        embeddings_file = tmp_path / "cluster_embeddings.npz"

        # Centroids: unit vectors in X, Y, Z directions
        centroids = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        # UUIDs stored as object array
        centroid_cluster_ids = np.array(
            ["test-uuid-000", "test-uuid-001", "test-uuid-002"], dtype=object
        )

        # No label embeddings for basic tests
        label_embeddings = np.array([], dtype=np.float32)
        label_cluster_ids = np.array([], dtype=object)

        np.savez_compressed(
            embeddings_file,
            centroids=centroids,
            centroid_cluster_ids=centroid_cluster_ids,
            label_embeddings=label_embeddings,
            label_cluster_ids=label_cluster_ids,
        )

        return embeddings_file

    def test_searcher_initialization(self, sample_clusters, sample_embeddings_file):
        """ClusterSearcher should load centroids matrix from .npz file."""
        searcher = ClusterSearcher(
            clusters=sample_clusters, embeddings_file=sample_embeddings_file
        )

        # Force load embeddings (normally lazy)
        searcher._load_embeddings()

        assert searcher.centroids is not None
        assert searcher.centroids.shape == (3, 3)
        assert len(searcher.cluster_ids) == 3

    def test_searcher_empty_clusters(self):
        """ClusterSearcher handles empty cluster list."""
        searcher = ClusterSearcher(clusters=[])

        assert searcher.centroids is None
        assert len(searcher.cluster_ids) == 0

    def test_search_finds_closest_cluster(
        self, sample_clusters, sample_embeddings_file
    ):
        """Search returns cluster with most similar centroid."""
        searcher = ClusterSearcher(
            clusters=sample_clusters, embeddings_file=sample_embeddings_file
        )

        # Query in X direction should find cluster 0
        query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        results = searcher.search(query, top_k=1)

        assert len(results) == 1
        assert results[0].cluster_id == "test-uuid-000"

    def test_search_respects_top_k(self, sample_clusters, sample_embeddings_file):
        """Search respects top_k limit."""
        searcher = ClusterSearcher(
            clusters=sample_clusters, embeddings_file=sample_embeddings_file
        )

        query = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        results = searcher.search(query, top_k=2)

        assert len(results) <= 2

    def test_search_respects_threshold(self, sample_clusters, sample_embeddings_file):
        """Search respects similarity threshold."""
        searcher = ClusterSearcher(
            clusters=sample_clusters, embeddings_file=sample_embeddings_file
        )

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = searcher.search(query, top_k=10, similarity_threshold=0.99)

        # Only cluster 0 should match with high threshold
        assert len(results) == 1
        assert results[0].cluster_id == "test-uuid-000"

    def test_search_cross_ids_only(self, sample_clusters, sample_embeddings_file):
        """Search can filter to cross-IDS clusters only."""
        searcher = ClusterSearcher(
            clusters=sample_clusters, embeddings_file=sample_embeddings_file
        )

        # Query in Y direction (cluster 1 is intra-IDS)
        query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        results = searcher.search(
            query,
            top_k=10,
            similarity_threshold=0.0,
            cross_ids_only=True,
        )

        # Cluster 1 is not cross-IDS, so it should be excluded
        for result in results:
            assert result.is_cross_ids

    def test_search_result_fields(self, sample_clusters, sample_embeddings_file):
        """Search results have expected fields."""
        searcher = ClusterSearcher(
            clusters=sample_clusters, embeddings_file=sample_embeddings_file
        )

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = searcher.search(query, top_k=1)

        result = results[0]
        assert isinstance(result, ClusterSearchResult)
        assert isinstance(result.cluster_id, str)  # UUID string
        assert isinstance(result.similarity_score, float)
        assert isinstance(result.is_cross_ids, bool)
        assert isinstance(result.ids_names, list)
        assert isinstance(result.paths, list)

    def test_get_similar_clusters(self, sample_clusters, sample_embeddings_file):
        """Can find clusters similar to a given cluster."""
        searcher = ClusterSearcher(
            clusters=sample_clusters, embeddings_file=sample_embeddings_file
        )

        # Get clusters similar to cluster 0
        similar = searcher.get_similar_clusters(
            cluster_id="test-uuid-000",
            top_k=2,
            similarity_threshold=0.0,
        )

        # Should not include cluster 0 itself
        for result in similar:
            assert result.cluster_id != "test-uuid-000"

    def test_clusters_without_centroids(self):
        """Searcher handles clusters without embeddings file."""
        clusters = [
            {
                "id": "test-uuid-000",
                "similarity_score": 0.9,
                "is_cross_ids": True,
                "ids_names": ["core_profiles"],
                "paths": ["core_profiles/a"],
                "scope": "global",
            },
        ]

        # No embeddings file provided
        searcher = ClusterSearcher(clusters=clusters)

        # Embeddings not loaded yet (lazy), should be None
        assert searcher.centroids is None
        results = searcher.search(np.array([1.0, 0.0, 0.0]))
        assert len(results) == 0
