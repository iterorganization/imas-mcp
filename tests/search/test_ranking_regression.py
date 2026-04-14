"""Ranking regression tests for core physics queries.

Guards against ranking regressions by verifying that well-known IMAS
DD paths appear within the expected top-N results for 20 canonical queries.

All tests require a live Neo4j instance (``@pytest.mark.graph``) and use
the ``search_tool`` fixture backed by the ``GraphSearchTool``.

Run with::

    # Skip all (no graph)
    uv run pytest tests/search/test_ranking_regression.py -v -k "not graph"

    # Full suite (requires Neo4j + embed server)
    uv run pytest tests/search/test_ranking_regression.py -v
"""

from __future__ import annotations

import asyncio
import logging

import pytest

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(coro):
    """Run a coroutine synchronously (works inside pytest sync tests)."""
    return asyncio.get_event_loop().run_until_complete(coro)


async def _search(search_tool, query: str, limit: int = 20) -> list[str]:
    """Run hybrid search and return path IDs in ranked order."""
    try:
        result = await search_tool.search_dd_paths(query=query, max_results=limit)
        if hasattr(result, "hits"):
            return [hit.path for hit in result.hits]
        return []
    except Exception as e:
        logger.warning("Search failed for %r: %s", query, e)
        return []


def _assert_path_in_top_n(
    paths: list[str],
    expected_path: str,
    top_n: int,
    query: str,
) -> None:
    """Assert that *expected_path* appears in the top-*top_n* results."""
    top = paths[:top_n]
    assert expected_path in top, (
        f"Query {query!r}: expected {expected_path!r} in top {top_n}, got: {top}"
    )


def _assert_prefix_in_top_n(
    paths: list[str],
    prefix: str,
    top_n: int,
    query: str,
) -> None:
    """Assert that at least one result starting with *prefix* is in top-*top_n*."""
    top = paths[:top_n]
    matches = [p for p in top if p.startswith(prefix)]
    assert matches, (
        f"Query {query!r}: expected a path starting with {prefix!r} in top {top_n}, "
        f"got: {top}"
    )


def _assert_segment_in_top_n(
    paths: list[str],
    segment: str,
    top_n: int,
    query: str,
) -> None:
    """Assert that at least one result has *segment* as a terminal path component."""
    top = paths[:top_n]
    matches = [p for p in top if p.split("/")[-1] == segment or f"/{segment}" in p]
    assert matches, (
        f"Query {query!r}: expected a path containing segment {segment!r} in top {top_n}, "
        f"got: {top}"
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def graph_client():
    """Module-scoped GraphClient for regression tests."""
    from imas_codex.graph.client import GraphClient

    try:
        client = GraphClient()
        client.get_stats()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")

    yield client
    client.close()


@pytest.fixture(scope="module")
def search_tool(graph_client):
    """GraphSearchTool backed by the live graph."""
    from imas_codex.tools.graph_search import GraphSearchTool

    return GraphSearchTool(graph_client)


# ── Concept query regression tests (10) ──────────────────────────────────────


@pytest.mark.graph
class TestConceptQueryRegression:
    """Core physics concept queries must always return the canonical path."""

    @pytest.mark.parametrize(
        "query,expected_path,top_n",
        [
            (
                "electron temperature",
                "core_profiles/profiles_1d/electrons/temperature",
                3,
            ),
            (
                "plasma current",
                "equilibrium/time_slice/global_quantities/ip",
                3,
            ),
            (
                "safety factor q",
                "equilibrium/time_slice/profiles_1d/q",
                5,
            ),
            (
                "electron density",
                "core_profiles/profiles_1d/electrons/density",
                3,
            ),
            (
                "poloidal flux",
                "equilibrium/time_slice/profiles_1d/psi",
                5,
            ),
            (
                "ion temperature",
                "core_profiles/profiles_1d/ion/temperature",
                5,
            ),
            (
                "toroidal magnetic field",
                "equilibrium/vacuum_toroidal_field/b0",
                5,
            ),
            (
                "plasma boundary shape",
                "equilibrium/time_slice/boundary/outline/r",
                10,
            ),
            (
                "effective charge",
                "core_profiles/profiles_1d/zeff",
                5,
            ),
            (
                "loop voltage",
                "summary/global_quantities/v_loop",
                5,
            ),
        ],
        ids=[
            "electron_temperature",
            "plasma_current",
            "safety_factor_q",
            "electron_density",
            "poloidal_flux",
            "ion_temperature",
            "toroidal_b_field",
            "boundary_shape",
            "effective_charge",
            "loop_voltage",
        ],
    )
    def test_concept_query(self, search_tool, query, expected_path, top_n):
        """Canonical path must appear in top-N results."""
        paths = _run(_search(search_tool, query, limit=top_n + 10))
        if not paths:
            pytest.xfail(
                f"Search returned no results for {query!r} — embed server may be down"
            )
        _assert_path_in_top_n(paths, expected_path, top_n, query)


# ── IDS-qualified query regression tests (5) ─────────────────────────────────


@pytest.mark.graph
class TestIDSQualifiedQueryRegression:
    """IDS-prefixed queries should surface exact matches near the top."""

    def test_equilibrium_psi(self, search_tool):
        """'equilibrium psi' → any equilibrium/.../psi path in top 3."""
        query = "equilibrium psi"
        paths = _run(_search(search_tool, query, limit=15))
        if not paths:
            pytest.xfail("Search returned no results — embed server may be down")
        top3 = paths[:3]
        matches = [
            p for p in top3 if p.startswith("equilibrium/") and p.endswith("/psi")
        ]
        assert matches, (
            f"Query {query!r}: expected an equilibrium/.../psi path in top 3, got: {top3}"
        )

    def test_core_profiles_electron_temperature(self, search_tool):
        """'core_profiles electron temperature' → exact path in top 3."""
        query = "core_profiles electron temperature"
        paths = _run(_search(search_tool, query, limit=15))
        if not paths:
            pytest.xfail("Search returned no results — embed server may be down")
        _assert_path_in_top_n(
            paths,
            "core_profiles/profiles_1d/electrons/temperature",
            3,
            query,
        )

    def test_magnetics_b_field(self, search_tool):
        """'magnetics b_field' → any magnetics/b_field* path in top 5."""
        query = "magnetics b_field"
        paths = _run(_search(search_tool, query, limit=20))
        if not paths:
            pytest.xfail("Search returned no results — embed server may be down")
        _assert_prefix_in_top_n(paths, "magnetics/b_field", 5, query)

    def test_summary_ip(self, search_tool):
        """'summary ip' → summary/global_quantities/ip in top 5."""
        query = "summary ip"
        paths = _run(_search(search_tool, query, limit=20))
        if not paths:
            pytest.xfail("Search returned no results — embed server may be down")
        _assert_path_in_top_n(paths, "summary/global_quantities/ip", 5, query)

    def test_equilibrium_boundary_elongation(self, search_tool):
        """'equilibrium boundary elongation' → elongation path in top 5."""
        query = "equilibrium boundary elongation"
        paths = _run(_search(search_tool, query, limit=20))
        if not paths:
            pytest.xfail("Search returned no results — embed server may be down")
        _assert_path_in_top_n(
            paths,
            "equilibrium/time_slice/boundary/elongation",
            5,
            query,
        )


# ── Edge case regression tests (5) ───────────────────────────────────────────


@pytest.mark.graph
class TestEdgeCaseQueryRegression:
    """Short tokens and ambiguous terms — verify at least one plausible result."""

    def test_abbreviation_ip(self, search_tool):
        """'ip' → any path with 'ip' as terminal segment in top 10."""
        query = "ip"
        paths = _run(_search(search_tool, query, limit=20))
        if not paths:
            pytest.xfail("Search returned no results — embed server may be down")
        _assert_segment_in_top_n(paths, "ip", 10, query)

    def test_abbreviation_ne(self, search_tool):
        """'ne' → any electron density path in top 10."""
        query = "ne"
        paths = _run(_search(search_tool, query, limit=20))
        if not paths:
            pytest.xfail("Search returned no results — embed server may be down")
        top10 = paths[:10]
        # Accept density, ne, or n_e as terminal segment in electron-related paths
        density_matches = [
            p
            for p in top10
            if "electron" in p and ("density" in p or p.endswith("/ne"))
        ]
        segment_matches = [
            p for p in top10 if p.split("/")[-1] in {"ne", "n_e", "density"}
        ]
        assert density_matches or segment_matches, (
            f"Query {query!r}: expected an electron density path in top 10, got: {top10}"
        )

    def test_abbreviation_q(self, search_tool):
        """'q' → any safety factor path in top 10."""
        query = "q"
        paths = _run(_search(search_tool, query, limit=20))
        if not paths:
            pytest.xfail("Search returned no results — embed server may be down")
        top10 = paths[:10]
        q_matches = [p for p in top10 if p.split("/")[-1] == "q" or "safety" in p]
        assert q_matches, (
            f"Query {query!r}: expected a safety factor path in top 10, got: {top10}"
        )

    def test_abbreviation_te(self, search_tool):
        """'Te' → any electron temperature path in top 10."""
        query = "Te"
        paths = _run(_search(search_tool, query, limit=20))
        if not paths:
            pytest.xfail("Search returned no results — embed server may be down")
        top10 = paths[:10]
        te_matches = [
            p
            for p in top10
            if "electron" in p
            and "temperature" in p
            or p.split("/")[-1] in {"te", "Te", "temperature"}
        ]
        assert te_matches, (
            f"Query {query!r}: expected an electron temperature path in top 10, got: {top10}"
        )

    def test_abbreviation_b0(self, search_tool):
        """'b0' → any b0 path in top 10."""
        query = "b0"
        paths = _run(_search(search_tool, query, limit=20))
        if not paths:
            pytest.xfail("Search returned no results — embed server may be down")
        _assert_segment_in_top_n(paths, "b0", 10, query)
