"""Plan 39 Phase 0 (c) tests — slim ``cluster_search`` sync helper.

The fan-out catalog calls :func:`imas_codex.graph.dd_search.cluster_search`
directly.  These tests lock down the public signature, the path-vs-text
branching, and the :class:`ClusterHit` shape.

The full MCP-tool semantics
(:class:`imas_codex.tools.graph_search.GraphClustersTool.search_dd_clusters`)
are covered by ``tests/tools/test_cluster_search.py`` and are unaffected
by Phase 0 — the slim helper is additive.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch


def test_cluster_search_signature() -> None:
    """``cluster_search(gc, query, *, scope=None, k=8, dd_version=None)``."""
    from imas_codex.graph.dd_search import cluster_search

    sig = inspect.signature(cluster_search)
    params = sig.parameters
    assert "gc" in params
    assert "query" in params
    assert params["scope"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["scope"].default is None
    assert params["k"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["k"].default == 8
    assert params["dd_version"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["dd_version"].default is None


def test_cluster_search_is_sync() -> None:
    from imas_codex.graph.dd_search import cluster_search

    assert not inspect.iscoroutinefunction(cluster_search)


def test_cluster_search_empty_query_returns_empty() -> None:
    from imas_codex.graph.dd_search import cluster_search

    gc = MagicMock()
    assert cluster_search(gc, "") == []
    assert cluster_search(gc, "   ") == []
    gc.query.assert_not_called()


def test_cluster_search_path_branch() -> None:
    """Query containing ``/`` and no whitespace → path lookup branch."""
    from imas_codex.graph.dd_search import ClusterHit, cluster_search

    gc = MagicMock()
    gc.query.return_value = [
        {
            "id": "cluster_42",
            "label": "electron temperature",
            "description": "Te clusters",
            "scope": "global",
            "ids_names": ["core_profiles", "summary"],
            "paths": ["core_profiles/profiles_1d/electrons/temperature"],
        }
    ]

    result = cluster_search(gc, "core_profiles/profiles_1d/electrons/temperature", k=5)

    cypher = gc.query.call_args[0][0]
    # Path branch matches IMASNode by id, no vector index call.
    assert "{id: $path}" in cypher
    assert "db.index.vector.queryNodes" not in cypher
    assert len(result) == 1
    assert isinstance(result[0], ClusterHit)
    assert result[0].id == "cluster_42"
    assert result[0].score == 1.0
    assert result[0].ids_names == ["core_profiles", "summary"]


@patch("imas_codex.graph.dd_search._embed", return_value=[0.1] * 768)
def test_cluster_search_semantic_branch(mock_embed) -> None:
    """Whitespace-bearing query → vector-index semantic branch."""
    from imas_codex.graph.dd_search import ClusterHit, cluster_search

    gc = MagicMock()
    gc.query.return_value = [
        {
            "id": "cluster_99",
            "label": "ion temperature",
            "description": "",
            "scope": "global",
            "ids_names": ["core_profiles"],
            "paths": [],
            "score": 0.81,
        }
    ]

    result = cluster_search(gc, "ion temperature profile", k=8)

    cypher = gc.query.call_args[0][0]
    assert "cluster_label_embedding" in cypher
    assert len(result) == 1
    assert isinstance(result[0], ClusterHit)
    assert result[0].score == 0.81


@patch("imas_codex.graph.dd_search._embed", return_value=[0.1] * 768)
def test_cluster_search_scope_filter_in_cypher(mock_embed) -> None:
    from imas_codex.graph.dd_search import cluster_search

    gc = MagicMock()
    gc.query.return_value = []
    cluster_search(gc, "ion temperature profile", scope="domain")
    cypher = gc.query.call_args[0][0]
    assert "c.scope = $scope" in cypher
    kwargs = gc.query.call_args.kwargs
    assert kwargs["scope"] == "domain"
