"""Live-graph regression tests for DD search quality improvements.

Covers:
- find_related_dd_paths unit_companions returns cross-domain results
  for psi (regression from a stray physics_domain filter).
- find_related_dd_paths exposes a cocos_kin section for COCOS-tagged paths.
- search_dd_paths supports cocos_transformation_type filtering.
- list_dd_paths supports cocos_transformation_type filtering.
- search_formatters renders the new COCOS line, node_category badges,
  and cocos_kin section.

All tests require a live Neo4j connection; they are skipped automatically
when the graph is unreachable via the top-level pytest_collection_modifyitems
hook (pytest.mark.graph marker).
"""

from __future__ import annotations

import os

import pytest

from imas_codex.llm.search_formatters import (
    format_path_context_report,
    format_search_dd_report,
)
from imas_codex.llm.server import _get_imas_tools

pytestmark = pytest.mark.graph


@pytest.fixture(scope="module", autouse=True)
def _use_production_embedder():
    """Override the session autouse MiniLM fixture with production config.

    The graph's vector indexes are built with the Qwen 256-dim embedder.
    Tests in this file hit the live graph via semantic search, so they must
    use the same embedder; otherwise Neo4j raises a dimensionality error.
    """
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
def live_tools(_use_production_embedder):
    return _get_imas_tools(semantic_search=True)


# ---------------------------------------------------------------------------
# find_related_dd_paths — Feature A regressions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unit_companions_cross_domain_for_psi(live_tools):
    """psi unit_companions must return results (regression: physics_domain filter
    suppressed all companions when the source path lived in a domain with no
    other Wb fields)."""
    res = await live_tools.path_context_tool.find_related_dd_paths(
        path="equilibrium/time_slice/profiles_1d/psi",
        relationship_types="unit",
        max_results=10,
    )
    companions = res["sections"].get("unit_companions", [])
    assert companions, "unit_companions should not be empty for psi"
    # All companions should carry Wb units (implicit — unit relationship filter)
    assert len(companions) >= 3


@pytest.mark.asyncio
async def test_cocos_kin_populated_for_ip(live_tools):
    """magnetics/ip is ip_like; cocos_kin section should expose other ip_like
    paths from across the DD."""
    res = await live_tools.path_context_tool.find_related_dd_paths(
        path="magnetics/ip",
        relationship_types="cocos",
        max_results=10,
    )
    cocos_kin = res["sections"].get("cocos_kin", [])
    assert cocos_kin, "cocos_kin should be populated for magnetics/ip"
    for entry in cocos_kin:
        assert entry.get("cocos_type") == "ip_like"


@pytest.mark.asyncio
async def test_cocos_kin_absent_when_source_has_no_cocos(live_tools):
    """Paths without cocos_label_transformation should produce an empty
    cocos_kin section (not an error)."""
    res = await live_tools.path_context_tool.find_related_dd_paths(
        path="equilibrium/time_slice/profiles_1d/psi",
        relationship_types="cocos",
        max_results=10,
    )
    # psi itself has no cocos_label_transformation (coordinate), so kin is empty
    assert res["sections"].get("cocos_kin", []) == []


# ---------------------------------------------------------------------------
# search_dd_paths / list_dd_paths — Feature B filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_dd_paths_cocos_filter(live_tools):
    """cocos_transformation_type filter must restrict hits to that COCOS type."""
    res = await live_tools.search_tool.search_dd_paths(
        query="poloidal flux psi",
        max_results=50,
        cocos_transformation_type="psi_like",
    )
    assert hasattr(res, "hits"), f"search returned ToolError: {res}"
    assert res.hits, "expected at least one psi_like hit for 'poloidal flux psi'"
    for h in res.hits:
        assert h.cocos_label_transformation == "psi_like"


@pytest.mark.asyncio
async def test_list_dd_paths_cocos_filter(live_tools):
    """cocos_transformation_type filter on list_dd_paths must restrict results."""
    res = await live_tools.list_tool.list_dd_paths(
        paths="equilibrium",
        leaf_only=True,
        max_paths=500,
        cocos_transformation_type="psi_like",
    )
    assert res.results, "list_dd_paths should return at least one result container"
    # All returned paths (if any) must be psi_like
    for container in res.results:
        for p in container.paths:
            # Access the underlying cocos label via raw dict if available
            cocos = getattr(p, "cocos_label_transformation", None)
            if cocos is not None:
                assert cocos == "psi_like"


# ---------------------------------------------------------------------------
# Formatter snapshot checks
# ---------------------------------------------------------------------------


def test_format_search_dd_report_includes_cocos_line():
    """The search report must render a COCOS line when the hit carries a
    cocos_label_transformation."""
    from imas_codex.models.result_models import SearchPathsResult
    from imas_codex.search.search_strategy import SearchHit

    hit = SearchHit(
        path="magnetics/ip",
        ids_name="magnetics",
        documentation="Plasma current.",
        units="A",
        data_type="FLT_0D",
        physics_domain="magnetics",
        cocos_label_transformation="ip_like",
        keywords=["plasma current"],
        node_category="quantity",
        score=0.95,
        rank=1,
        search_mode="auto",
    )
    result = SearchPathsResult(
        hits=[hit],
        query="plasma current",
        search_mode="auto",
    )
    out = format_search_dd_report(result)
    assert "COCOS: ip_like" in out


def test_format_path_context_report_renders_cocos_kin_and_badges():
    """format_path_context_report must render node_category badges on
    cluster_siblings/unit_companions and a cocos_kin section."""
    ctx = {
        "focus": {
            "path": "magnetics/ip",
            "ids": "magnetics",
            "documentation": "Plasma current",
            "units": "A",
            "physics_domain": "magnetics",
            "cocos_label_transformation": "ip_like",
        },
        "sections": {
            "cluster_siblings": [
                {
                    "path": "core_profiles/global_quantities/ip",
                    "ids": "core_profiles",
                    "cluster": "plasma_current",
                    "physics_domain": "core_transport",
                    "node_category": "quantity",
                }
            ],
            "unit_companions": [
                {
                    "path": "pulse_schedule/flux_control/i_plasma/reference",
                    "ids": "pulse_schedule",
                    "unit": "A",
                    "units": "A",
                    "physics_domain": "control",
                    "node_category": "quantity",
                }
            ],
            "cocos_kin": [
                {
                    "path": "core_profiles/global_quantities/current_bootstrap",
                    "ids": "core_profiles",
                    "cocos_type": "ip_like",
                    "doc": "Bootstrap current.",
                    "node_category": "quantity",
                }
            ],
        },
    }
    out = format_path_context_report(ctx)
    assert "ip_like" in out
    # node_category badge rendered somewhere
    assert "quantity" in out
    # cocos_kin section header rendered
    assert "cocos" in out.lower()
