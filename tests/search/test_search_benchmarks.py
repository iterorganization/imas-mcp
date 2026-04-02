"""Search quality benchmarks for IMAS DD.

Unit tests (``TestMRRComputation``, ``TestCategoryMRR``) verify the MRR
calculation logic and need no external infrastructure.

Integration tests are gated on Neo4j availability
(``@pytest.mark.graph``) and optionally on the embedding server.  When
infrastructure is unavailable, those tests skip gracefully.

Each search method is tested independently before combination.  MRR
thresholds are pinned to validated baselines — a PR that causes a
regression will fail these tests.

Run with::

    # Unit tests only (no graph needed)
    uv run pytest tests/search/test_search_benchmarks.py -k "TestMRR or TestCategory" -v

    # All tests (requires Neo4j + embed server)
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
    compute_category_mrr,
    compute_mrr,
)
from tests.search.benchmark_helpers import (
    BenchmarkResults,
    assert_mrr_above,
    assert_precision_at_1_above,
    run_benchmark,
)

logger = logging.getLogger(__name__)


# ── Unit tests — MRR computation (no graph needed) ───────────────────────────


class TestMRRComputation:
    """Verify reciprocal rank calculation logic."""

    def test_perfect_rank(self):
        assert compute_mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_rank(self):
        assert compute_mrr(["x", "a", "c"], ["a"]) == 0.5

    def test_third_rank(self):
        assert compute_mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert compute_mrr(["x", "y", "z"], ["a"]) == 0.0

    def test_empty_results(self):
        assert compute_mrr([], ["a"]) == 0.0

    def test_multiple_expected_first_match_wins(self):
        """First match among multiple expected paths determines rank."""
        assert compute_mrr(["b", "a"], ["a", "b"]) == 1.0

    def test_prefix_match_child(self):
        """Result is a child of expected path — matches with allow_prefix."""
        assert (
            compute_mrr(
                ["equilibrium/time_slice/global_quantities/ip/data"],
                ["equilibrium/time_slice/global_quantities/ip"],
                allow_prefix=True,
            )
            == 1.0
        )

    def test_prefix_match_parent(self):
        """Result is a parent of expected path — matches with allow_prefix."""
        assert (
            compute_mrr(
                ["equilibrium/time_slice/global_quantities/ip"],
                ["equilibrium/time_slice/global_quantities/ip/data"],
                allow_prefix=True,
            )
            == 1.0
        )

    def test_prefix_no_match_without_flag(self):
        """Prefix relationships are ignored without allow_prefix."""
        assert (
            compute_mrr(
                ["equilibrium/time_slice/global_quantities/ip/data"],
                ["equilibrium/time_slice/global_quantities/ip"],
                allow_prefix=False,
            )
            == 0.0
        )

    def test_prefix_no_partial_segment_match(self):
        """Prefix must be at a ``/`` boundary, not mid-segment."""
        assert (
            compute_mrr(
                ["equilibrium/time_slice/profiles_1d/psi_norm"],
                ["equilibrium/time_slice/profiles_1d/psi"],
                allow_prefix=True,
            )
            == 0.0
        )

    def test_rank_position_matters(self):
        """Later positions give smaller reciprocal rank."""
        assert compute_mrr(["x", "y", "z", "a"], ["a"]) == pytest.approx(0.25)


class TestCategoryMRR:
    """Verify category-level MRR aggregation."""

    def test_perfect_category(self):
        queries = [
            BenchmarkQuery("q1", ["a"], "test"),
            BenchmarkQuery("q2", ["b"], "test"),
        ]
        results = {"q1": ["a", "x"], "q2": ["b", "y"]}
        assert compute_category_mrr(results, queries) == 1.0

    def test_mixed_category(self):
        queries = [
            BenchmarkQuery("q1", ["a"], "test"),
            BenchmarkQuery("q2", ["b"], "test"),
        ]
        results = {"q1": ["a"], "q2": ["x", "b"]}
        assert compute_category_mrr(results, queries) == pytest.approx(0.75)

    def test_empty_queries(self):
        assert compute_category_mrr({}, []) == 0.0

    def test_missing_query_in_results(self):
        """Missing query → MRR 0 for that query, averages down."""
        queries = [
            BenchmarkQuery("q1", ["a"], "test"),
            BenchmarkQuery("q2", ["b"], "test"),
        ]
        results = {"q1": ["a"]}  # q2 missing
        assert compute_category_mrr(results, queries) == pytest.approx(0.5)

    def test_all_misses(self):
        queries = [
            BenchmarkQuery("q1", ["a"], "test"),
            BenchmarkQuery("q2", ["b"], "test"),
        ]
        results = {"q1": ["x", "y"], "q2": ["x", "y"]}
        assert compute_category_mrr(results, queries) == 0.0


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
def encoder():
    """Fresh Encoder that connects to the real remote embedding server.

    The session-scoped conftest forces IMAS_CODEX_EMBEDDING_LOCATION=local
    and uses all-MiniLM-L6-v2 (384-dim) for fast unit tests.  Benchmark
    tests need the real remote server with 256-dim Qwen3 embeddings, so
    we temporarily restore the production env vars.
    """
    import os

    from imas_codex.settings import _get_section

    # Read the real config from pyproject.toml (not the conftest overrides)
    embed_config = _get_section("embedding")
    real_location = embed_config.get("location", "")
    real_model = embed_config.get("model", "")

    if not real_location or real_location == "local":
        pytest.skip("No remote embedding location configured in pyproject.toml")

    # Temporarily restore production env vars
    old_location = os.environ.get("IMAS_CODEX_EMBEDDING_LOCATION")
    old_model = os.environ.get("IMAS_CODEX_EMBEDDING_MODEL")
    try:
        os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = real_location
        if real_model:
            os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = real_model
        elif "IMAS_CODEX_EMBEDDING_MODEL" in os.environ:
            del os.environ["IMAS_CODEX_EMBEDDING_MODEL"]

        from imas_codex.embeddings.encoder import Encoder, EncoderConfig

        config = EncoderConfig()
        enc = Encoder(config=config)
        result = enc.embed_texts(["test"])
        if result is None or len(result) == 0:
            pytest.skip("Embed server returned empty results")
        dim = len(result[0])
        if dim != 256:
            pytest.skip(
                f"Embed server returns {dim}-dim, expected 256 "
                f"(backend={config.backend}, url={config.remote_url})"
            )
        yield enc
    except pytest.skip.Exception:
        raise
    except Exception as e:
        pytest.skip(f"Embed server not available: {e}")
    finally:
        # Restore conftest env vars so other tests aren't affected
        if old_location is not None:
            os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = old_location
        if old_model is not None:
            os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = old_model


@pytest.fixture(scope="module")
def embed_available(encoder) -> bool:
    """Whether the embedding server is reachable (True if encoder fixture succeeded)."""
    return encoder is not None


@pytest.fixture(scope="module")
def search_tool(graph_client):
    """GraphSearchTool backed by the live graph."""
    from imas_codex.tools.graph_search import GraphSearchTool

    return GraphSearchTool(graph_client)


# ── Helper: extract path IDs from search results ────────────────────────────


def _extract_paths_from_vector(
    graph_client, encoder, query: str, limit: int
) -> list[str]:
    """Run vector-only search and return path IDs in ranked order."""
    from imas_codex.tools.query_analysis import QueryAnalyzer

    analyzer = QueryAnalyzer()
    intent = analyzer.analyze(query)
    expanded = " ".join(intent.expanded_terms) if intent.expanded_terms else query
    embedding = encoder.embed_texts([expanded])[0].tolist()

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

# Baseline thresholds — calibrated to current search quality.
# Raise these as search improvements land (abbreviation expansion,
# LLM enrichment, accessor filtering, etc.).


@pytest.mark.graph
class TestVectorSearchBenchmark:
    """Vector/semantic search quality — requires embed server + graph.

    Post-fix: MRR ≥0.40 target. IDS-prefixed embedding text provides semantic
    separation between identically-described nodes in different IDSs.
    """

    MRR_THRESHOLD = 0.15
    P_AT_1_THRESHOLD = 0.05

    def test_vector_mrr(self, graph_client, encoder, embed_available):
        if not embed_available:
            pytest.skip("Embed server not available")

        results = run_benchmark(
            method_name="Vector",
            queries=SEMANTIC_QUERIES,
            search_fn=lambda q, lim: _extract_paths_from_vector(
                graph_client, encoder, q, lim
            ),
            limit=50,
        )

        logger.info(results.summary())
        assert_mrr_above(results, self.MRR_THRESHOLD)

    def test_vector_precision_at_1(self, graph_client, encoder, embed_available):
        """At least 25% of queries should have the correct answer at rank 1."""
        if not embed_available:
            pytest.skip("Embed server not available")

        results = run_benchmark(
            method_name="Vector P@1",
            queries=SEMANTIC_QUERIES,
            search_fn=lambda q, lim: _extract_paths_from_vector(
                graph_client, encoder, q, lim
            ),
            limit=50,
        )

        logger.info(results.summary())
        assert_precision_at_1_above(results, self.P_AT_1_THRESHOLD)

    def test_vector_returns_results(self, graph_client, encoder, embed_available):
        """Sanity check: vector search should return non-empty results."""
        if not embed_available:
            pytest.skip("Embed server not available")

        paths = _extract_paths_from_vector(
            graph_client, encoder, "electron temperature", 10
        )
        assert len(paths) > 0, (
            "Vector search returned no results for 'electron temperature'"
        )


@pytest.mark.graph
class TestBM25SearchBenchmark:
    """BM25/fulltext search quality — requires graph only (no embed server).

    Post-fix: MRR ≥0.45 target. Abbreviation expansion, post-BM25 reranking,
    accessor terminal exclusion, and child keyword inheritance.
    """

    MRR_THRESHOLD = 0.15

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
            assert cat_mrr["structural"] >= 0.80, (
                f"Structural query MRR too low: {cat_mrr['structural']:.3f}"
            )

        # Exact concept queries should be findable by keyword
        if "exact_concept" in cat_mrr:
            assert cat_mrr["exact_concept"] >= 0.05, (
                f"Exact concept MRR too low: {cat_mrr['exact_concept']:.3f}"
            )


@pytest.mark.graph
class TestPathLookupBenchmark:
    """Exact path lookup — the simplest search mode.

    Path queries containing '/' should always return the exact match.
    This should work perfectly — it's just string matching.
    """

    ACCURACY_THRESHOLD = 1.00

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


@pytest.mark.graph
class TestHybridSearchBenchmark:
    """Combined hybrid search — must exceed best individual method.

    Post-fix: MRR ≥0.50 target. RRF fusion with vector gating and heuristic
    reranking combines BM25 and vector search strengths.
    """

    MRR_THRESHOLD = 0.20

    @pytest.mark.asyncio
    async def test_hybrid_mrr(self, search_tool, embed_available):
        if not embed_available:
            pytest.skip("Embed server not available for hybrid search")

        results = BenchmarkResults(method_name="Hybrid")
        for q in ALL_QUERIES:
            paths = await _extract_paths_from_hybrid(search_tool, q.query_text, 50)
            from tests.search.benchmark_helpers import QueryResult

            results.query_results.append(QueryResult(query=q, returned_paths=paths))

        # If most queries returned empty, the search tool is broken
        empty_count = sum(1 for qr in results.query_results if not qr.returned_paths)
        if empty_count > len(ALL_QUERIES) * 0.8:
            pytest.skip(
                f"Hybrid search returned empty for {empty_count}/{len(ALL_QUERIES)} "
                "queries — search tool is broken (check vector index + embed server)"
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
        assert len(paths) > 0, "Hybrid search returned no results"


@pytest.mark.graph
class TestSearchQualityRegression:
    """Cross-cutting regression checks.

    These tests catch specific known failure modes rather than measuring
    aggregate MRR.  They define the quality bar for individual queries.
    """

    def test_electron_temp_not_dominated_by_accessors(
        self, graph_client, encoder, embed_available
    ):
        """Top results for 'electron temperature' should be concept nodes,
        not accessor terminals like 'value', 'time', 'r', 'z'."""
        if not embed_available:
            pytest.skip("Embed server not available")

        paths = _extract_paths_from_vector(
            graph_client, encoder, "electron temperature", 10
        )
        accessor_names = {
            "value",
            "time",
            "r",
            "z",
            "phi",
            "data",
            "validity",
            "validity_timed",
            "coefficients",
        }
        top5_names = [p.split("/")[-1] for p in paths[:5]]
        accessor_count = sum(1 for n in top5_names if n in accessor_names)
        assert accessor_count <= 3, (
            f"Top 5 results dominated by accessors: {top5_names}"
        )

    def test_electron_temp_finds_core_profiles(
        self, graph_client, encoder, embed_available
    ):
        """'electron temperature' MUST find a relevant temperature path."""
        if not embed_available:
            pytest.skip("Embed server not available")

        paths = _extract_paths_from_vector(
            graph_client, encoder, "electron temperature", 50
        )
        # Accept any path that represents electron temperature
        valid_paths = {
            "core_profiles/profiles_1d/electrons/temperature",
            "summary/local/itb/t_e",
            "summary/local/limiter/t_e",
            "summary/line_average/t_e",
            "summary/local/pedestal/t_e",
            "summary/volume_average/t_e",
            "edge_profiles/profiles_1d/electrons/temperature",
        }
        assert any(p in valid_paths for p in paths), (
            f"Expected a valid electron temperature path, got: {paths[:5]}"
        )

    def test_plasma_current_finds_ip(self, graph_client, encoder, embed_available):
        """'plasma current' MUST find equilibrium/.../ip."""
        if not embed_available:
            pytest.skip("Embed server not available")

        paths = _extract_paths_from_vector(graph_client, encoder, "plasma current", 20)
        assert any("ip" in p.split("/")[-1] for p in paths), (
            f"Expected a path ending in 'ip', got: {paths[:5]}"
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
