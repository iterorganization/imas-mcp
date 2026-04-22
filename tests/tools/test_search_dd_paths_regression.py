"""Regression tests for search_dd_paths MCP tool after hybrid_dd_search extraction.

Verifies that the MCP tool's SearchPathsResult contains expected
stable substrings and structural properties — byte-identical matching
is too brittle given score/rank variability across graph states.

Tests require a live Neo4j connection.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.graph


@pytest.fixture(scope="module", autouse=True)
def _use_production_embedder():
    """Override embedding config to match graph vector indexes."""
    prior = {
        "IMAS_CODEX_EMBEDDING_MODEL": os.environ.get("IMAS_CODEX_EMBEDDING_MODEL"),
        "IMAS_CODEX_EMBEDDING_LOCATION": os.environ.get(
            "IMAS_CODEX_EMBEDDING_LOCATION"
        ),
        "IMAS_CODEX_EMBEDDING_DIMENSION": os.environ.get(
            "IMAS_CODEX_EMBEDDING_DIMENSION"
        ),
    }
    os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = "Qwen/Qwen3-Embedding-0.6B"
    os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = "titan"
    os.environ["IMAS_CODEX_EMBEDDING_DIMENSION"] = "256"
    yield
    for key, value in prior.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="module")
def search_tool(_use_production_embedder):
    """Live GraphSearchTool instance."""
    from imas_codex.llm.server import _get_imas_tools

    try:
        tools = _get_imas_tools(semantic_search=True)
    except Exception as e:
        pytest.skip(f"Could not initialise tools: {e}")
    return tools.search_tool


# ---------------------------------------------------------------------------
# Structural properties of SearchPathsResult
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_has_hits_and_summary(search_tool):
    """SearchPathsResult must have hits list and summary dict."""
    res = await search_tool.search_dd_paths(query="electron temperature", max_results=5)
    assert hasattr(res, "hits"), f"Result missing 'hits': {type(res)}"
    assert hasattr(res, "summary"), f"Result missing 'summary': {type(res)}"
    assert isinstance(res.hits, list)
    assert isinstance(res.summary, dict)


@pytest.mark.asyncio
async def test_result_summary_has_query_key(search_tool):
    """Summary must echo back the query string."""
    res = await search_tool.search_dd_paths(query="plasma current", max_results=5)
    assert "query" in res.summary
    # query is normalized (path notation), but should contain the concept
    assert (
        "plasma" in res.summary["query"].lower()
        or "current" in res.summary["query"].lower()
    )


@pytest.mark.asyncio
async def test_hits_contain_known_paths(search_tool):
    """For well-known concepts, expected paths should appear in results."""
    res = await search_tool.search_dd_paths(
        query="electron temperature", max_results=20
    )
    paths = [h.path for h in res.hits]
    # core_profiles electron temperature is canonical
    assert any("core_profiles" in p and "electron" in p for p in paths), (
        f"Expected core_profiles/*/electron* path in results: {paths[:5]}"
    )


@pytest.mark.asyncio
async def test_ids_filter_propagates(search_tool):
    """ids_filter must restrict all hits to the specified IDS."""
    res = await search_tool.search_dd_paths(
        query="temperature", ids_filter="equilibrium", max_results=10
    )
    for hit in res.hits:
        assert hit.ids_name == "equilibrium", (
            f"Hit from wrong IDS: {hit.ids_name} ({hit.path})"
        )


@pytest.mark.asyncio
async def test_physics_domains_populated(search_tool):
    """physics_domains list should be populated for broad queries."""
    res = await search_tool.search_dd_paths(query="magnetic field", max_results=20)
    assert hasattr(res, "physics_domains")
    assert isinstance(res.physics_domains, list)
    # Magnetic field hits should include at least one domain
    if res.hits:
        assert len(res.physics_domains) >= 1


@pytest.mark.asyncio
async def test_search_mode_echoed(search_tool):
    """search_mode in summary should match the requested mode."""
    res = await search_tool.search_dd_paths(
        query="safety factor", max_results=5, search_mode="auto"
    )
    # SearchMode enum str representation
    assert "auto" in res.summary.get("search_mode", "").lower()
