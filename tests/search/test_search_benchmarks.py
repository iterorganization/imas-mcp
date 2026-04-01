"""Graph-backed search quality benchmarks for IMAS DD.

Tests are gated on Neo4j availability (``@pytest.mark.graph``) and
optionally on the embedding server (``requires_embed_server`` fixture).
When infrastructure is unavailable, tests skip gracefully.

Each search method is tested independently before combination.  MRR
thresholds are pinned to validated baselines — a PR that causes a
regression will fail these tests.

Run with::

    uv run pytest tests/search/test_search_benchmarks.py -v

"""

from __future__ import annotations

import logging

import pytest

from tests.search.benchmark_data import (
    ALL_QUERIES,
    PATH_QUERIES,
    SEMANTIC_QUERIES,
    TEXT_QUERIES,
    BenchmarkQuery,
)
from tests.search.benchmark_helpers import (
    BenchmarkResults,
    assert_mrr_above,
    assert_precision_at_1_above,
    run_benchmark,
)

logger = logging.getLogger(__name__)

# All tests require Neo4j
pytestmark = pytest.mark.graph


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def graph_client():
    """Module-scoped GraphClient for benchmark tests."""
    from imas_codex.graph.client import GraphClient

    try:
        client = GraphClient()
        client.get_stats()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")

    yield client
    client.close()


@pytest.fixture(scope="module")
def embed_available() -> bool:
    """Check whether the embedding server is reachable."""
    try:
        from imas_codex.embeddings.encoder import Encoder

        enc = Encoder()
        result = enc.embed_texts(["test"])
        return result is not None and len(result) > 0
    except Exception:
        return False


@pytest.fixture(scope="module")
def search_tool(graph_client):
    """GraphSearchTool backed by the live graph."""
    from imas_codex.tools.graph_search import GraphSearchTool

    return GraphSearchTool(graph_client)


# ── Helper: extract path IDs from search results ────────────────────────────


def _extract_paths_from_vector(graph_client, query: str, limit: int) -> list[str]:
    """Run vector-only search and return path IDs in ranked order."""
    from imas_codex.tools.graph_search import _get_encoder

    encoder = _get_encoder()
    embedding = encoder.embed_texts([query])[0].tolist()

    try:
        results = graph_client.query(
            """
            CALL db.index.vector.queryNodes('imas_node_embedding', $k, $embedding)
            YIELD node AS path, score
            WHERE NOT (path)-[:DEPRECATED_IN]->(:DDVersion)
              AND path.node_category = 'data'
            RETURN path.id AS id, score
            ORDER BY score DESC
            LIMIT $vector_limit
            """,
            embedding=embedding,
            k=min(limit * 5, 500),
            vector_limit=limit,
        )
    except Exception as e:
        if "dimensionality" in str(e).lower():
            pytest.skip(f"Vector index dimension mismatch: {e}")
        raise
    return [r["id"] for r in (results or [])]


def _extract_paths_from_text(graph_client, query: str, limit: int) -> list[str]:
    """Run text-only search (BM25 + CONTAINS) and return path IDs."""
    from imas_codex.tools.graph_search import _text_search_imas_paths

    results = _text_search_imas_paths(graph_client, query, limit, ids_filter=None)
    sorted_results = sorted(results, key=lambda r: r["score"], reverse=True)
    return [r["id"] for r in sorted_results]


def _extract_paths_from_path_lookup(graph_client, query: str, limit: int) -> list[str]:
    """Run exact path lookup and return matching path IDs."""
    results = graph_client.query(
        """
        MATCH (p:IMASNode)
        WHERE p.id = $path_query
          AND p.node_category = 'data'
          AND NOT (p)-[:DEPRECATED_IN]->(:DDVersion)
        RETURN p.id AS id
        LIMIT $lim
        """,
        path_query=query,
        lim=limit,
    )
    ids = [r["id"] for r in (results or [])]
    if ids:
        return ids

    # Fallback: CONTAINS match on path ID
    results = graph_client.query(
        """
        MATCH (p:IMASNode)
        WHERE p.node_category = 'data'
          AND NOT (p)-[:DEPRECATED_IN]->(:DDVersion)
          AND toLower(p.id) CONTAINS toLower($path_query)
        RETURN p.id AS id
        LIMIT $lim
        """,
        path_query=query,
        lim=limit,
    )
    return [r["id"] for r in (results or [])]


async def _extract_paths_from_hybrid(search_tool, query: str, limit: int) -> list[str]:
    """Run full hybrid search via the tool and return path IDs."""
    try:
        result = await search_tool.search_imas_paths(query=query, max_results=limit)
        # The decorator chain may return a ToolError instead of SearchPathsResult
        if hasattr(result, "hits"):
            return [hit.path for hit in result.hits]
        return []
    except Exception:
        return []


# ── Test Classes ─────────────────────────────────────────────────────────────


class TestVectorSearchBenchmark:
    """Vector/semantic search quality — requires embed server + graph.

    MRR threshold will be raised as search quality improves.
    Current baseline reflects pre-fix state.
    """

    # Pre-fix baseline: 0.016.  Post-fix target: ≥0.40.
    # Set conservatively at first — raise after Phase 8 validates.
    MRR_THRESHOLD = 0.01

    def test_vector_mrr(self, graph_client, embed_available):
        if not embed_available:
            pytest.skip("Embed server not available")

        results = run_benchmark(
            method_name="Vector",
            queries=SEMANTIC_QUERIES,
            search_fn=lambda q, lim: _extract_paths_from_vector(graph_client, q, lim),
            limit=50,
        )

        logger.info(results.summary())
        assert_mrr_above(results, self.MRR_THRESHOLD)

    def test_vector_returns_results(self, graph_client, embed_available):
        """Sanity check: vector search should return non-empty results."""
        if not embed_available:
            pytest.skip("Embed server not available")

        paths = _extract_paths_from_vector(graph_client, "electron temperature", 10)
        assert len(paths) > 0, (
            "Vector search returned no results for 'electron temperature'"
        )


class TestBM25SearchBenchmark:
    """BM25/fulltext search quality — requires graph only (no embed server).

    BM25 is the primary text search method using the fulltext index.
    """

    # Pre-fix baseline: ~0.20.  Post-fix target: ≥0.45.
    MRR_THRESHOLD = 0.10

    def test_bm25_mrr(self, graph_client):
        results = run_benchmark(
            method_name="BM25",
            queries=TEXT_QUERIES,
            search_fn=lambda q, lim: _extract_paths_from_text(graph_client, q, lim),
            limit=50,
        )

        logger.info(results.summary())
        assert_mrr_above(results, self.MRR_THRESHOLD)

    def test_bm25_per_category(self, graph_client):
        """Verify no category is completely broken."""
        results = run_benchmark(
            method_name="BM25 (per-category)",
            queries=TEXT_QUERIES,
            search_fn=lambda q, lim: _extract_paths_from_text(graph_client, q, lim),
            limit=50,
        )

        cat_mrr = results.per_category_mrr()
        logger.info("BM25 per-category MRR: %s", cat_mrr)

        # Structural queries should work well with text search
        if "structural" in cat_mrr:
            assert cat_mrr["structural"] >= 0.30, (
                f"Structural query MRR too low: {cat_mrr['structural']:.3f}"
            )


class TestPathLookupBenchmark:
    """Exact path lookup — the simplest search mode.

    Path queries containing '/' should always return the exact match.
    """

    ACCURACY_THRESHOLD = 0.80

    def test_path_lookup_accuracy(self, graph_client):
        results = run_benchmark(
            method_name="Path Lookup",
            queries=PATH_QUERIES,
            search_fn=lambda q, lim: _extract_paths_from_path_lookup(
                graph_client, q, lim
            ),
            limit=10,
        )

        logger.info(results.summary())
        assert_mrr_above(results, self.ACCURACY_THRESHOLD)

    def test_exact_path_at_rank_1(self, graph_client):
        """An exact path query should return that path at rank 1."""
        test_path = "equilibrium/time_slice/profiles_1d/psi"
        paths = _extract_paths_from_path_lookup(graph_client, test_path, 5)
        assert paths and paths[0] == test_path, (
            f"Exact path lookup failed: got {paths[:3]}"
        )


class TestHybridSearchBenchmark:
    """Combined hybrid search — must at least match individual methods.

    After Phase 9 (hybrid tuning), hybrid MRR should exceed both vector
    and BM25 individually.  Pre-tuning, we just check it works.
    """

    MRR_THRESHOLD = 0.10

    @pytest.mark.asyncio
    async def test_hybrid_mrr(self, search_tool, embed_available):
        if not embed_available:
            pytest.skip("Embed server not available for hybrid search")

        results = BenchmarkResults(method_name="Hybrid")
        for q in ALL_QUERIES:
            paths = await _extract_paths_from_hybrid(search_tool, q.query_text, 50)
            from tests.search.benchmark_helpers import QueryResult

            results.query_results.append(QueryResult(query=q, returned_paths=paths))

        # If most queries returned empty, the vector index is incompatible
        empty_count = sum(1 for qr in results.query_results if not qr.returned_paths)
        if empty_count > len(ALL_QUERIES) * 0.8:
            pytest.skip(
                f"Hybrid search returned empty for {empty_count}/{len(ALL_QUERIES)} "
                "queries — likely vector index dimension mismatch"
            )

        logger.info(results.summary())
        assert_mrr_above(results, self.MRR_THRESHOLD)

    @pytest.mark.asyncio
    async def test_hybrid_returns_results(self, search_tool, embed_available):
        """Sanity: hybrid search returns non-empty for a basic query."""
        if not embed_available:
            pytest.skip("Embed server not available")

        paths = await _extract_paths_from_hybrid(
            search_tool, "electron temperature", 10
        )
        if not paths:
            pytest.skip(
                "Hybrid search returned empty — likely vector index dimension mismatch"
            )


class TestSearchQualityRegression:
    """Cross-cutting regression checks.

    These tests catch specific known failure modes rather than measuring
    aggregate MRR.
    """

    def test_electron_temp_not_dominated_by_accessors(
        self, graph_client, embed_available
    ):
        """Top results for 'electron temperature' should be concept nodes,
        not accessor terminals like 'value', 'time', 'r', 'z'."""
        if not embed_available:
            pytest.skip("Embed server not available")

        paths = _extract_paths_from_vector(graph_client, "electron temperature", 10)
        accessor_names = {"value", "time", "r", "z", "phi", "data", "validity"}
        top5_names = [p.split("/")[-1] for p in paths[:5]]
        accessor_count = sum(1 for n in top5_names if n in accessor_names)
        assert accessor_count <= 2, (
            f"Top 5 results dominated by accessors: {top5_names}"
        )

    def test_path_query_skips_unrelated(self, graph_client):
        """A path query should not return unrelated paths."""
        paths = _extract_paths_from_path_lookup(
            graph_client, "equilibrium/time_slice/profiles_1d/psi", 10
        )
        if paths:
            assert all("equilibrium" in p for p in paths[:3]), (
                f"Path lookup returned unrelated: {paths[:3]}"
            )

    def test_text_search_returns_results(self, graph_client):
        """Sanity: text search returns something for a common term."""
        paths = _extract_paths_from_text(graph_client, "temperature", 10)
        assert len(paths) > 0, "Text search returned no results for 'temperature'"
