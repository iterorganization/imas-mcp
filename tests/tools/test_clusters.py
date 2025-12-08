"""
Tests for cluster search functionality.

This module tests the clusters tool which provides semantic search
over related IMAS paths using LLM-generated labels.
"""

import pytest

from imas_mcp.clusters.search import ClusterSearcher, ClusterSearchResult
from imas_mcp.physics.relationship_engine import (
    RelationshipStrength,
    SemanticRelationshipAnalyzer,
)
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


class TestRelationshipStrength:
    """Test relationship strength classification system."""

    def test_strength_categories(self):
        """Test strength category constants."""
        assert RelationshipStrength.VERY_STRONG == 0.9
        assert RelationshipStrength.STRONG == 0.7
        assert RelationshipStrength.MODERATE == 0.5
        assert RelationshipStrength.WEAK == 0.3
        assert RelationshipStrength.VERY_WEAK == 0.1

    def test_get_category_classification(self):
        """Test category classification from strength values."""
        assert RelationshipStrength.get_category(0.95) == "very_strong"
        assert RelationshipStrength.get_category(0.75) == "strong"
        assert RelationshipStrength.get_category(0.55) == "moderate"
        assert RelationshipStrength.get_category(0.35) == "weak"
        assert RelationshipStrength.get_category(0.15) == "very_weak"

    def test_boundary_conditions(self):
        """Test boundary conditions for strength classification."""
        assert RelationshipStrength.get_category(0.9) == "very_strong"
        assert RelationshipStrength.get_category(0.7) == "strong"
        assert RelationshipStrength.get_category(0.5) == "moderate"
        assert RelationshipStrength.get_category(0.3) == "weak"
        assert RelationshipStrength.get_category(0.89) == "strong"


class TestSemanticRelationshipAnalyzer:
    """Test semantic analysis capabilities."""

    def test_analyzer_instantiation(self):
        """Test that semantic analyzer can be instantiated."""
        analyzer = SemanticRelationshipAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, "analyze_concept")

    def test_concept_analysis(self):
        """Test physics concept analysis functionality."""
        analyzer = SemanticRelationshipAnalyzer()
        result = analyzer.analyze_concept("core_profiles/profiles_1d/electrons/density")
        assert result is not None
        assert isinstance(result, dict)


class TestClusterSearcher:
    """Test cluster search functionality."""

    def test_searcher_with_empty_clusters(self):
        """Test searcher handles empty cluster list."""
        searcher = ClusterSearcher(clusters=[])
        assert searcher.centroids is None
        assert searcher.cluster_ids == []

    def test_searcher_with_mock_clusters(self):
        """Test searcher with mock cluster data."""
        mock_clusters = [
            {
                "id": 0,
                "label": "Test Cluster",
                "description": "A test cluster",
                "is_cross_ids": True,
                "ids_names": ["core_profiles", "equilibrium"],
                "paths": ["core_profiles/test/path", "equilibrium/test/path"],
                "centroid": [0.1, 0.2, 0.3],
                "similarity_score": 0.95,
            }
        ]
        searcher = ClusterSearcher(clusters=mock_clusters)
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
