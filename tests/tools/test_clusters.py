"""
Tests for cluster search functionality.

This module tests the clusters tool which provides semantic search
over related IMAS paths using LLM-generated labels.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from imas_mcp.clusters.search import ClusterSearcher, ClusterSearchResult
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.tools.clusters_tool import ClustersTool
from tests.conftest import STANDARD_TEST_IDS_SET


@pytest.fixture(scope="function")
def document_store():
    """Provide a fresh DocumentStore instance for each test."""
    return DocumentStore(ids_set=STANDARD_TEST_IDS_SET)


@pytest.fixture(scope="function")
def clusters_tool(document_store):
    """Provide a fresh ClustersTool instance for each test."""
    return ClustersTool(document_store)


class TestClusterSearcher:
    """Test cluster search functionality."""

    def test_searcher_with_empty_clusters(self):
        """Test searcher handles empty cluster list."""
        searcher = ClusterSearcher(clusters=[])
        assert searcher.centroids is None
        assert searcher.cluster_ids == []

    def test_searcher_with_mock_clusters(self, tmp_path):
        """Test searcher with mock cluster data and .npz embeddings file."""
        mock_clusters = [
            {
                "id": 0,
                "label": "Test Cluster",
                "description": "A test cluster",
                "is_cross_ids": True,
                "ids_names": ["core_profiles", "equilibrium"],
                "paths": ["core_profiles/test/path", "equilibrium/test/path"],
                "similarity_score": 0.95,
            }
        ]

        # Create temporary .npz file with embeddings
        embeddings_file = tmp_path / "cluster_embeddings.npz"
        centroids = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        centroid_cluster_ids = np.array([0], dtype=np.int32)
        label_embeddings = np.array([], dtype=np.float32)
        label_cluster_ids = np.array([], dtype=np.int32)

        np.savez_compressed(
            embeddings_file,
            centroids=centroids,
            centroid_cluster_ids=centroid_cluster_ids,
            label_embeddings=label_embeddings,
            label_cluster_ids=label_cluster_ids,
        )

        searcher = ClusterSearcher(
            clusters=mock_clusters, embeddings_file=embeddings_file
        )
        # Force load embeddings
        searcher._load_embeddings()

        assert searcher.centroids is not None
        assert len(searcher.cluster_ids) == 1

    def test_search_by_path(self):
        """Test path-based cluster lookup."""
        mock_clusters = [
            {
                "id": 0,
                "paths": ["core_profiles/test/path"],
                "ids_names": ["core_profiles"],
                "is_cross_ids": False,
                "similarity_score": 0.9,
            }
        ]
        searcher = ClusterSearcher(clusters=mock_clusters)
        results = searcher.search_by_path("core_profiles/test/path")
        assert len(results) == 1
        assert results[0].cluster_id == 0


class TestClustersTool:
    """Test the main clusters tool functionality."""

    @pytest.mark.asyncio
    async def test_tool_instantiation(self, clusters_tool):
        """Test that the tool can be instantiated."""
        assert clusters_tool is not None
        assert hasattr(clusters_tool, "search_imas_clusters")

    @pytest.mark.asyncio
    async def test_path_query(self, clusters_tool):
        """Test path-based query."""
        result = await clusters_tool.search_imas_clusters(
            query="core_profiles/profiles_1d/electrons/density",
        )
        # Should return result or error
        assert result is not None

    @pytest.mark.asyncio
    async def test_natural_language_query(self, clusters_tool):
        """Test natural language query."""
        result = await clusters_tool.search_imas_clusters(
            query="electron temperature",
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_semantic_query(self, clusters_tool):
        """Test semantic query."""
        result = await clusters_tool.search_imas_clusters(
            query="magnetic field",
        )
        assert result is not None
