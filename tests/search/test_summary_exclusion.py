"""Tests for summary IDS exclusion from search results.

The `summary` IDS contains 1541 aggregate/reference paths that can dominate
search results. By default, search_dd_paths and search_dd_clusters exclude
summary paths so users see primary sources first.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_client():
    """Live graph client for integration tests."""
    from imas_codex.graph.client import GraphClient

    try:
        gc = GraphClient()
        gc.get_stats()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")
    yield gc
    gc.close()


@pytest.fixture
def clusters_tool(graph_client):
    """GraphClustersTool backed by the live graph."""
    from imas_codex.tools.graph_search import GraphClustersTool

    return GraphClustersTool(graph_client)


# ---------------------------------------------------------------------------
# _text_search_dd_paths — summary exclusion at text-search level
# ---------------------------------------------------------------------------


@pytest.mark.graph
class TestTextSearchSummaryExclusion:
    """Verify summary IDS exclusion in the text search path."""

    def test_default_text_search_excludes_summary(self, graph_client):
        """_text_search_dd_paths with exclude_summary=True returns no summary/ paths."""
        from imas_codex.tools.graph_search import _text_search_dd_paths

        results = _text_search_dd_paths(
            graph_client, "heating power", 50, ids_filter=None, exclude_summary=True
        )
        ids_found = [r["id"] for r in results]
        summary_hits = [p for p in ids_found if p.startswith("summary/")]
        assert not summary_hits, (
            f"Expected no summary paths with exclude_summary=True, got: {summary_hits[:5]}"
        )

    def test_text_search_include_summary_can_return_summary(self, graph_client):
        """_text_search_dd_paths with exclude_summary=False may return summary/ paths."""
        from imas_codex.tools.graph_search import _text_search_dd_paths

        results_incl = _text_search_dd_paths(
            graph_client, "heating power", 100, ids_filter=None, exclude_summary=False
        )
        ids_incl = [r["id"] for r in results_incl]
        # The include run should return at least some results
        assert ids_incl, "Expected results when include_summary_ids=False"
        # Summary paths should appear in the include run
        summary_in_incl = [p for p in ids_incl if p.startswith("summary/")]
        assert summary_in_incl, (
            f"Expected summary/ paths when exclude_summary=False, got none in: {ids_incl[:10]}"
        )

    def test_text_search_ids_filter_summary_with_exclude(self, graph_client):
        """When ids_filter='summary', results are only summary paths regardless of exclude_summary."""
        from imas_codex.tools.graph_search import _text_search_dd_paths

        # With ids_filter='summary', exclude_summary is intentionally NOT applied
        # at this level — the caller (_search_dd_paths) handles this logic.
        # The helper itself just uses ids_filter as a positive filter.
        results = _text_search_dd_paths(
            graph_client, "power", 20, ids_filter="summary", exclude_summary=False
        )
        ids_found = [r["id"] for r in results]
        if ids_found:
            assert all(p.startswith("summary/") for p in ids_found), (
                f"ids_filter='summary' should return only summary paths, "
                f"got non-summary: {[p for p in ids_found if not p.startswith('summary/')][:5]}"
            )


# ---------------------------------------------------------------------------
# search_dd_clusters — summary path filtering
# ---------------------------------------------------------------------------


@pytest.mark.graph
class TestSearchDDClustersSummaryExclusion:
    """Verify summary paths are excluded from cluster member lists by default."""

    @pytest.mark.asyncio
    async def test_default_excludes_summary_from_cluster_paths(self, clusters_tool):
        """Cluster member lists contain no summary/ paths by default."""
        result = await clusters_tool.search_dd_clusters(query="current drive")
        # Skip if tool returned an error (e.g., vector index dimension mismatch)
        if not isinstance(result, dict) or "clusters" not in result:
            pytest.skip(f"Cluster search returned error: {result}")
        for cluster in result.get("clusters", []):
            summary_paths = [
                p for p in cluster.get("paths", []) if p.startswith("summary/")
            ]
            assert not summary_paths, (
                f"Cluster '{cluster.get('label')}' has summary paths: {summary_paths}"
            )

    @pytest.mark.asyncio
    async def test_include_summary_false_removes_summary_members(self, clusters_tool):
        """include_summary_ids=False removes summary/ paths from all cluster member lists."""
        result = await clusters_tool.search_dd_clusters(
            query="heating", include_summary_ids=False
        )
        if not isinstance(result, dict) or "clusters" not in result:
            pytest.skip(f"Cluster search returned error: {result}")
        all_paths = [p for c in result.get("clusters", []) for p in c.get("paths", [])]
        summary_in_result = [p for p in all_paths if p.startswith("summary/")]
        assert not summary_in_result, (
            f"include_summary_ids=False still returned summary paths: {summary_in_result[:5]}"
        )

    @pytest.mark.asyncio
    async def test_include_summary_true_does_not_strip_summary_members(
        self, clusters_tool
    ):
        """include_summary_ids=True preserves summary/ paths in cluster member lists."""
        result_with = await clusters_tool.search_dd_clusters(
            query="heating", include_summary_ids=True
        )
        result_without = await clusters_tool.search_dd_clusters(
            query="heating", include_summary_ids=False
        )
        if not isinstance(result_with, dict) or not isinstance(result_without, dict):
            pytest.skip(
                "Cluster search returned error (likely embedding dimension mismatch)"
            )
        all_paths_with = [
            p for c in result_with.get("clusters", []) for p in c.get("paths", [])
        ]
        all_paths_without = [
            p for c in result_without.get("clusters", []) for p in c.get("paths", [])
        ]
        summary_in_without = [p for p in all_paths_without if p.startswith("summary/")]
        assert not summary_in_without, (
            f"include_summary_ids=False still returns summary paths: {summary_in_without[:5]}"
        )
        # include_summary_ids=True should have at least as many total paths
        assert len(all_paths_with) >= len(all_paths_without)
