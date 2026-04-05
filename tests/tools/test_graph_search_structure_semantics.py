"""Direct tests for graph-backed structure/leaf semantics."""

from unittest.mock import MagicMock

import pytest

from imas_codex.tools.graph_search import (
    GraphStructureTool,
    _leaf_data_type_clause,
    _text_search_dd_paths,
)


def test_leaf_data_type_clause_uses_uppercase_structure_types():
    clause = _leaf_data_type_clause("p")

    assert clause == (
        "p.data_type IS NOT NULL AND NOT (p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])"
    )


def test_text_search_fallback_uses_uppercase_structure_types():
    gc = MagicMock()

    def side_effect(cypher, **kwargs):
        if "db.index.fulltext.queryNodes" in cypher:
            raise Exception("Index not found")
        return []

    gc.query.side_effect = side_effect

    _text_search_dd_paths(gc, "plasma current", 10, None)

    fallback_calls = [
        call.args[0]
        for call in gc.query.call_args_list
        if "MATCH (p:IMASNode)" in call.args[0]
    ]
    assert fallback_calls
    assert any(
        "NOT (p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])" in cypher
        for cypher in fallback_calls
    )
    assert not any("<> 'structure'" in cypher for cypher in fallback_calls)


@pytest.mark.asyncio
async def test_analyze_dd_structure_uses_uppercase_structure_types():
    gc = MagicMock()
    gc.query.side_effect = [
        [{"total_paths": 0, "leaf_count": 0, "max_depth": 0, "avg_depth": 0}],
        [],
        [],
        [],
        [],
    ]
    tool = GraphStructureTool(gc)

    await tool.analyze_dd_structure("equilibrium")

    metrics_query = gc.query.call_args_list[0].args[0]
    assert "nullIf" in metrics_query
    assert "IN ['STRUCTURE', 'STRUCT_ARRAY']" in metrics_query
    assert "<> 'structure'" not in metrics_query


@pytest.mark.asyncio
async def test_export_imas_ids_leaf_filter_uses_uppercase_structure_types():
    gc = MagicMock()
    gc.query.return_value = []
    tool = GraphStructureTool(gc)

    await tool.export_imas_ids("equilibrium", leaf_only=True)

    query = gc.query.call_args.args[0]
    assert "NOT (p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])" in query
    assert "<> 'structure'" not in query
