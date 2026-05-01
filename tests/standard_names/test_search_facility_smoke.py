"""Regression tests for plan-40 SN search facility bug fixes.

Surfaced via live-graph smoke run against the running equilibrium pilot
(see ``scripts/sn_search_smoke.py`` history). Each test pins the
expected behaviour of one targeted fix.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Bug A: _search_standard_names("") used to run a full vector search and
# return embedding-noise hits. Now early-returns a helpful prompt.
# ---------------------------------------------------------------------------


class TestSearchEmptyQueryEarlyReturn:
    def test_empty_query_no_filters_returns_prompt(self):
        from imas_codex.llm.sn_tools import _search_standard_names

        gc = MagicMock()
        out = _search_standard_names("", gc=gc)
        assert "No query provided" in out
        # Must not have hit Cypher
        gc.query.assert_not_called()

    def test_whitespace_query_no_filters_returns_prompt(self):
        from imas_codex.llm.sn_tools import _search_standard_names

        gc = MagicMock()
        out = _search_standard_names("   ", gc=gc)
        assert "No query provided" in out
        gc.query.assert_not_called()

    def test_empty_query_with_filter_proceeds(self):
        """Empty query is fine if a grammar filter narrows the space."""
        from imas_codex.llm.sn_tools import _search_standard_names

        gc = MagicMock()
        gc.query.return_value = []
        out = _search_standard_names("", physical_base="magnetic_flux", gc=gc)
        # Should not be the "no query" message — should run the filter path.
        assert "No query provided" not in out


# ---------------------------------------------------------------------------
# Bug B: segment_filters used typed grammar edges (HAS_PHYSICAL_BASE,
# HAS_SUBJECT) that are unpopulated for open-vocab segments. Now matches
# bare-name columns directly.
# ---------------------------------------------------------------------------


class TestSegmentFilterUsesBareColumns:
    def test_backing_segment_filter_queries_bare_column(self):
        from imas_codex.standard_names.search import _segment_filter_search

        gc = MagicMock()
        gc.query.return_value = []
        _segment_filter_search(
            gc,
            query="ignored",
            k=5,
            segment_filters={"physical_base": "magnetic_flux"},
            kind=None,
            pipeline_status=None,
            cocos_type=None,
        )
        cypher = gc.query.call_args.args[0]
        # Must NOT use the typed-edge MATCH; must use bare column
        assert "HAS_PHYSICAL_BASE" not in cypher
        assert "GrammarToken" not in cypher
        assert "sn.physical_base" in cypher

    def test_mcp_segment_filter_queries_bare_column(self):
        from imas_codex.llm.sn_tools import _segment_filter_search_standard_names

        gc = MagicMock()
        gc.query.return_value = []
        _segment_filter_search_standard_names(
            gc, query="x", k=5, segment_filters={"subject": "electron"}
        )
        cypher = gc.query.call_args.args[0]
        assert "HAS_SUBJECT" not in cypher
        assert "GrammarToken" not in cypher
        assert "sn.subject" in cypher

    def test_unknown_segment_skipped(self):
        from imas_codex.llm.sn_tools import _segment_filter_search_standard_names

        gc = MagicMock()
        gc.query.return_value = []
        # Only an unknown segment → no WHERE clause to build → empty list
        result = _segment_filter_search_standard_names(
            gc, query="x", k=5, segment_filters={"bogus": "v"}
        )
        assert result == []
        gc.query.assert_not_called()


# ---------------------------------------------------------------------------
# Bug C: _find_related_standard_names("missing_name") used to return
# "No related names found" — indistinguishable from a real anchor with
# no relations. Now reports the missing anchor and points to check.
# ---------------------------------------------------------------------------


class TestFindRelatedMissingAnchor:
    def test_missing_anchor_reports_not_found(self):
        from imas_codex.llm.sn_tools import _find_related_standard_names

        gc = MagicMock()
        # First query (existence check) returns empty
        gc.query.return_value = []
        out = _find_related_standard_names("nonexistent_xxx", gc=gc)
        assert "not found in the catalogue" in out
        assert "check_standard_names" in out
        # Existence-check is a single Cypher call; backing must not run.
        assert gc.query.call_count == 1

    def test_existing_anchor_invokes_backing(self, monkeypatch):
        from imas_codex.llm import sn_tools

        gc = MagicMock()
        gc.query.return_value = [{"id": "plasma_current"}]

        called = {}

        def _fake_backing(name, **kw):
            called["name"] = name
            return {"Unit Companions": [{"name": "x", "description": "y"}]}

        monkeypatch.setattr(sn_tools, "_find_related_backing", _fake_backing)
        out = sn_tools._find_related_standard_names("plasma_current", gc=gc)
        assert called["name"] == "plasma_current"
        assert "Unit Companions" in out


# ---------------------------------------------------------------------------
# Tokenisation behaviour pin (avoid silent regression from the smoke run)
# ---------------------------------------------------------------------------


def test_tokenise_query_drops_punctuation_and_stopwords():
    from imas_codex.standard_names.grammar_query import tokenise_query

    assert tokenise_query("x_component_of_magnetic_field") == [
        "x",
        "component",
        "magnetic",
        "field",
    ]
    assert tokenise_query("electron_temperature_at_outboard_midplane") == [
        "electron",
        "temperature",
        "outboard",
        "midplane",
    ]
    # punctuation-only query → empty
    assert tokenise_query("...!!!") == []
    # whitespace → empty
    assert tokenise_query("   \t\n") == []


# ---------------------------------------------------------------------------
# Live graph smoke: keep these as pytest-skippable without a graph.
# ---------------------------------------------------------------------------


def _graph_available() -> bool:
    try:
        from imas_codex.graph.client import GraphClient

        gc = GraphClient()
        try:
            rows = gc.query("MATCH (sn:StandardName) RETURN count(sn) AS c LIMIT 1")
            return bool(rows and rows[0].get("c"))
        finally:
            gc.close()
    except Exception:
        return False


@pytest.mark.skipif(not _graph_available(), reason="no live graph with StandardNames")
class TestLiveGraphSmoke:
    def test_segment_filter_returns_results_on_live_graph(self):
        """End-to-end: open-vocab physical_base filter must surface SNs."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.search import search_standard_names

        gc = GraphClient()
        try:
            # Pick a physical_base known to exist with multiple members.
            rows = gc.query(
                "MATCH (sn:StandardName) WHERE sn.physical_base IS NOT NULL "
                "WITH sn.physical_base AS pb, count(*) AS n "
                "WHERE n >= 2 RETURN pb LIMIT 1"
            )
            if not rows:
                pytest.skip("no multi-member physical_base in graph")
            pb = rows[0]["pb"]
            results = search_standard_names(
                "anything", segment_filters={"physical_base": pb}, gc=gc
            )
            assert results, f"segment filter physical_base={pb} returned 0 hits"
            assert all(r.get("physical_base") == pb for r in results)
        finally:
            gc.close()
