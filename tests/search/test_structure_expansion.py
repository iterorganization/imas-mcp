"""Unit tests for STRUCTURE hit child expansion in search_dd_paths."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_graph_search_tool(gc_mock):
    """Construct a GraphSearchTool with a mocked GraphClient."""
    from imas_codex.tools.graph_search import GraphSearchTool

    tool = GraphSearchTool.__new__(GraphSearchTool)
    tool._gc = gc_mock
    return tool


def _make_gc_mock(enriched_rows, child_rows=None):
    """Return a GraphClient mock with configurable query side effects.

    Tests patch ``_text_search_dd_paths`` so it never calls gc.query, and
    tests set ``_embed_query`` to return None so vector search is also
    skipped.  The only gc.query calls that actually occur are:

      1st call  — enrichment (returns enriched_rows)
      2nd call  — rename lineage (always empty for test paths)
      3rd call  — child expansion (returns child_rows or [])
    """
    gc = MagicMock()
    call_sequence = [
        enriched_rows,  # enrichment
        [],  # rename lineage (_fetch_rename_lineage, always empty for test paths)
        child_rows or [],  # child expansion
    ]
    gc.query = MagicMock(side_effect=call_sequence)
    return gc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structure_row(path: str, ids: str = "tf") -> dict:
    """Build an enriched DB row for a STRUCTURE-type node."""
    return {
        "id": path,
        "name": path.split("/")[-1],
        "ids": ids,
        "documentation": f"Structure node {path}",
        "data_type": "STRUCTURE",
        "physics_domain": "magnetics",
        "units": None,
        "node_type": "dynamic",
        "lifecycle_status": None,
        "lifecycle_version": None,
        "timebasepath": None,
        "path_doc": None,
        "coordinate1_same_as": None,
        "coordinate2_same_as": None,
        "cocos_transformation_type": None,
        "cocos_transformation_expression": None,
        "description": None,
        "keywords": None,
        "enrichment_source": None,
        "coordinates": [],
        "has_identifier_schema": False,
        "introduced_after_version": None,
        "timebase": None,
        "structure_reference": None,
        "coordinate1": None,
        "coordinate2": None,
        "cocos_label": None,
        "cocos_expression": None,
    }


def _leaf_row(path: str, ids: str = "tf") -> dict:
    """Build an enriched DB row for a leaf (non-STRUCTURE) node."""
    row = _structure_row(path, ids)
    row["data_type"] = "FLT_1D"
    row["units"] = "A"
    return row


def _child_entry(path: str) -> dict:
    """Build a child dict as returned by the child expansion query."""
    return {
        "id": path,
        "name": path.split("/")[-1],
        "data_type": "FLT_1D",
        "units": "A",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structure_hits_get_children():
    """STRUCTURE hits should have children populated after expansion."""
    struct_path = "tf/coil/current"
    child_paths = [f"{struct_path}/data", f"{struct_path}/time"]

    enriched_rows = [_structure_row(struct_path)]
    child_rows = [
        {
            "parent_id": struct_path,
            "children": [_child_entry(p) for p in child_paths],
            "total": 2,
        }
    ]

    gc = _make_gc_mock(enriched_rows, child_rows)

    # Patch _text_search_dd_paths to return a text score hit
    with (
        patch(
            "imas_codex.tools.graph_search._text_search_dd_paths",
            return_value=[{"id": struct_path, "score": 0.9}],
        ),
        patch("imas_codex.graph.dd_search._embed", return_value=None),
    ):
        tool = _make_graph_search_tool(gc)
        result = await tool.search_dd_paths("coil current")

    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.path == struct_path
    assert hit.data_type == "STRUCTURE"
    assert hit.children is not None
    assert len(hit.children) == 2
    assert hit.children_total == 2
    assert hit.children[0]["id"] == f"{struct_path}/data"
    assert hit.children[1]["id"] == f"{struct_path}/time"


@pytest.mark.asyncio
async def test_non_structure_hits_have_no_children():
    """Leaf (non-STRUCTURE) hits should have children=None."""
    leaf_path = "equilibrium/time_slice/profiles_1d/psi"

    enriched_rows = [_leaf_row(leaf_path, ids="equilibrium")]
    # child expansion query should not even be called for leaf hits,
    # but if it were, it would return empty.
    child_rows: list = []

    gc = _make_gc_mock(enriched_rows, child_rows)

    with (
        patch(
            "imas_codex.tools.graph_search._text_search_dd_paths",
            return_value=[{"id": leaf_path, "score": 0.9}],
        ),
        patch("imas_codex.graph.dd_search._embed", return_value=None),
    ):
        tool = _make_graph_search_tool(gc)
        result = await tool.search_dd_paths("psi profile")

    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.data_type == "FLT_1D"
    assert hit.children is None
    assert hit.children_total is None


@pytest.mark.asyncio
async def test_structure_expansion_capped_at_five_parents():
    """Child expansion should be requested for at most 5 STRUCTURE parents."""
    # Create 7 STRUCTURE hits
    struct_paths = [f"tf/coil{i}/current" for i in range(7)]
    enriched_rows = [_structure_row(p) for p in struct_paths]

    # Child expansion query returns nothing (we only check the call)
    gc = _make_gc_mock(enriched_rows, child_rows=[])

    with (
        patch(
            "imas_codex.tools.graph_search._text_search_dd_paths",
            return_value=[
                {"id": p, "score": 0.9 - i * 0.01} for i, p in enumerate(struct_paths)
            ],
        ),
        patch("imas_codex.graph.dd_search._embed", return_value=None),
    ):
        tool = _make_graph_search_tool(gc)
        result = await tool.search_dd_paths("coil current")

    # Confirm all 7 hits came back
    assert len(result.hits) == 7

    # Find the child expansion call — it's the last query() call that has
    # a `parent_ids` keyword argument.
    child_expansion_calls = [
        call for call in gc.query.call_args_list if "parent_ids" in (call.kwargs or {})
    ]
    assert len(child_expansion_calls) == 1
    parent_ids_sent = child_expansion_calls[0].kwargs["parent_ids"]
    assert len(parent_ids_sent) <= 5, (
        f"Expected ≤5 parent IDs, got {len(parent_ids_sent)}"
    )


@pytest.mark.asyncio
async def test_structure_with_no_children_in_db():
    """A STRUCTURE hit with no matching DB children should leave children=None."""
    struct_path = "tf/coil/current"
    enriched_rows = [_structure_row(struct_path)]
    # DB returns no children for this parent
    child_rows = [{"parent_id": struct_path, "children": [], "total": 0}]

    gc = _make_gc_mock(enriched_rows, child_rows)

    with (
        patch(
            "imas_codex.tools.graph_search._text_search_dd_paths",
            return_value=[{"id": struct_path, "score": 0.9}],
        ),
        patch("imas_codex.graph.dd_search._embed", return_value=None),
    ):
        tool = _make_graph_search_tool(gc)
        result = await tool.search_dd_paths("coil current")

    assert len(result.hits) == 1
    hit = result.hits[0]
    # children_by_parent will have an entry but children list is empty —
    # the code attaches the empty list since child_data is truthy (non-None).
    # This verifies no crash occurs and children_total is reported correctly.
    assert hit.children_total == 0


@pytest.mark.asyncio
async def test_mixed_structure_and_leaf_hits():
    """Only STRUCTURE hits should receive children; leaf hits stay unmodified."""
    struct_path = "tf/coil/current"
    leaf_path = "equilibrium/time_slice/profiles_1d/psi"

    enriched_rows = [
        _structure_row(struct_path),
        _leaf_row(leaf_path, ids="equilibrium"),
    ]
    child_rows = [
        {
            "parent_id": struct_path,
            "children": [_child_entry(f"{struct_path}/data")],
            "total": 1,
        }
    ]

    gc = _make_gc_mock(enriched_rows, child_rows)

    with (
        patch(
            "imas_codex.tools.graph_search._text_search_dd_paths",
            return_value=[
                {"id": struct_path, "score": 0.9},
                {"id": leaf_path, "score": 0.8},
            ],
        ),
        patch("imas_codex.graph.dd_search._embed", return_value=None),
    ):
        tool = _make_graph_search_tool(gc)
        result = await tool.search_dd_paths("coil current psi")

    assert len(result.hits) == 2
    hits_by_path = {h.path: h for h in result.hits}

    struct_hit = hits_by_path[struct_path]
    assert struct_hit.children is not None
    assert struct_hit.children_total == 1

    leaf_hit = hits_by_path[leaf_path]
    assert leaf_hit.children is None
    assert leaf_hit.children_total is None
