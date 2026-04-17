"""Tests for RENAMED_TO / RENAMED_FROM lineage surfaced by search_dd_paths
and fetch_dd_paths.

Uses mock graph clients so no live Neo4j connection is needed.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from imas_codex.core.data_model import IdsNode
from imas_codex.llm.search_formatters import (
    format_fetch_paths_report,
    format_search_dd_report,
)
from imas_codex.models.result_models import FetchPathsResult, SearchPathsResult
from imas_codex.search.search_strategy import SearchHit
from imas_codex.tools.graph_search import GraphPathTool, _fetch_rename_lineage

# ---------------------------------------------------------------------------
# Unit tests for _fetch_rename_lineage helper
# ---------------------------------------------------------------------------


class TestFetchRenameLineage:
    """Tests for the _fetch_rename_lineage batch helper."""

    def _make_gc(self, query_return):
        gc = MagicMock()
        gc.query = MagicMock(return_value=query_return)
        return gc

    def test_returns_empty_for_no_paths(self):
        gc = self._make_gc([])
        result = _fetch_rename_lineage(gc, [])
        assert result == {}
        gc.query.assert_not_called()

    def test_single_renamed_to(self):
        """Path renamed to a successor — only renamed_to populated."""
        gc = self._make_gc(
            [
                {
                    "id": "magnetics/b_field_tor_probe/field",
                    "rename_version": "3.39.0",
                    "renamed_from": [],
                    "renamed_to": ["magnetics/b_field_phi_probe/field"],
                }
            ]
        )
        result = _fetch_rename_lineage(gc, ["magnetics/b_field_tor_probe/field"])
        assert "magnetics/b_field_tor_probe/field" in result
        lin = result["magnetics/b_field_tor_probe/field"]
        assert lin["renamed_to"] == ["magnetics/b_field_phi_probe/field"]
        assert lin["renamed_from"] == []
        assert lin["rename_version"] == "3.39.0"

    def test_single_renamed_from(self):
        """Path is the successor of a renamed predecessor."""
        gc = self._make_gc(
            [
                {
                    "id": "magnetics/b_field_phi_probe/field",
                    "rename_version": None,
                    "renamed_from": ["magnetics/b_field_tor_probe/field"],
                    "renamed_to": [],
                }
            ]
        )
        result = _fetch_rename_lineage(gc, ["magnetics/b_field_phi_probe/field"])
        lin = result["magnetics/b_field_phi_probe/field"]
        assert lin["renamed_from"] == ["magnetics/b_field_tor_probe/field"]
        assert lin["renamed_to"] == []
        assert lin["rename_version"] is None

    def test_multi_hop_chain(self):
        """Middle node of A→B→C chain has both predecessors and successors."""
        gc = self._make_gc(
            [
                {
                    "id": "node_b",
                    "rename_version": "4.0.0",
                    "renamed_from": ["node_a"],
                    "renamed_to": ["node_c"],
                }
            ]
        )
        result = _fetch_rename_lineage(gc, ["node_b"])
        lin = result["node_b"]
        assert "node_a" in lin["renamed_from"]
        assert "node_c" in lin["renamed_to"]
        assert lin["rename_version"] == "4.0.0"

    def test_no_history_omitted(self):
        """Paths without any rename history are not included in the result."""
        gc = self._make_gc([])  # empty — no rename rows
        result = _fetch_rename_lineage(gc, ["equilibrium/time_slice/profiles_1d/psi"])
        assert result == {}

    def test_graph_error_returns_empty(self):
        """Graph query errors are caught and an empty dict is returned."""
        gc = MagicMock()
        gc.query.side_effect = RuntimeError("Neo4j unreachable")
        result = _fetch_rename_lineage(gc, ["some/path"])
        assert result == {}

    def test_null_path_entries_filtered(self):
        """None values in renamed_from/renamed_to are stripped."""
        gc = self._make_gc(
            [
                {
                    "id": "some/path",
                    "rename_version": None,
                    "renamed_from": [None, "real/predecessor"],
                    "renamed_to": [None],
                }
            ]
        )
        result = _fetch_rename_lineage(gc, ["some/path"])
        lin = result["some/path"]
        assert lin["renamed_from"] == ["real/predecessor"]
        assert lin["renamed_to"] == []


# ---------------------------------------------------------------------------
# Integration-style tests for format_search_dd_report
# ---------------------------------------------------------------------------


class TestFormatSearchDdReportRenameLineage:
    """Formatter must render rename lineage lines under each matching path."""

    def _make_hit(self, path, renamed_from=None, renamed_to=None, rename_version=None):
        lineage = None
        if renamed_from is not None or renamed_to is not None:
            lineage = {
                "renamed_from": renamed_from or [],
                "renamed_to": renamed_to or [],
                "rename_version": rename_version,
            }
        return SearchHit(
            path=path,
            ids_name=path.split("/")[0],
            documentation="Test documentation.",
            score=0.9,
            rank=1,
            search_mode="auto",
            rename_lineage=lineage,
        )

    def test_renamed_to_rendered(self):
        hit = self._make_hit(
            "magnetics/b_field_tor_probe/field",
            renamed_to=["magnetics/b_field_phi_probe/field"],
            rename_version="3.39.0",
        )
        result = SearchPathsResult(
            hits=[hit], query="b field probe", search_mode="auto"
        )
        out = format_search_dd_report(result)
        assert "↳ renamed to: magnetics/b_field_phi_probe/field" in out
        assert "(in v3.39.0)" in out

    def test_renamed_from_rendered(self):
        hit = self._make_hit(
            "magnetics/b_field_phi_probe/field",
            renamed_from=["magnetics/b_field_tor_probe/field"],
        )
        result = SearchPathsResult(
            hits=[hit], query="b field probe", search_mode="auto"
        )
        out = format_search_dd_report(result)
        assert "↳ renamed from: magnetics/b_field_tor_probe/field" in out

    def test_no_lineage_no_annotation(self):
        """Paths without rename history must not emit any lineage lines."""
        hit = self._make_hit("equilibrium/time_slice/profiles_1d/psi")
        result = SearchPathsResult(
            hits=[hit], query="poloidal flux", search_mode="auto"
        )
        out = format_search_dd_report(result)
        assert "↳ renamed" not in out

    def test_renamed_to_without_version(self):
        """When rename_version is None, the suffix is omitted."""
        hit = self._make_hit(
            "old/path",
            renamed_to=["new/path"],
            rename_version=None,
        )
        result = SearchPathsResult(hits=[hit], query="test", search_mode="auto")
        out = format_search_dd_report(result)
        assert "↳ renamed to: new/path" in out
        assert "in v" not in out

    def test_multi_hop_chain_all_shown(self):
        """All predecessors and successors are rendered, not just immediate ones."""
        hit = self._make_hit(
            "node_b",
            renamed_from=["node_a"],
            renamed_to=["node_c", "node_d"],
            rename_version="4.0.0",
        )
        result = SearchPathsResult(hits=[hit], query="test", search_mode="auto")
        out = format_search_dd_report(result)
        assert "↳ renamed from: node_a" in out
        assert "↳ renamed to: node_c" in out
        assert "↳ renamed to: node_d" in out


# ---------------------------------------------------------------------------
# Integration-style tests for format_fetch_paths_report
# ---------------------------------------------------------------------------


class TestFormatFetchPathsReportRenameLineage:
    """Formatter must render rename lineage in RENAMED FROM/TO lines."""

    def _make_node(self, path, renamed_from=None, renamed_to=None, rename_version=None):
        lineage = None
        if renamed_from is not None or renamed_to is not None:
            lineage = {
                "renamed_from": renamed_from or [],
                "renamed_to": renamed_to or [],
                "rename_version": rename_version,
            }
        node = IdsNode(
            path=path,
            ids_name=path.split("/")[0],
            documentation="Test documentation.",
            rename_lineage=lineage,
        )
        return node

    def test_renamed_to_rendered(self):
        node = self._make_node(
            "magnetics/b_field_tor_probe/field",
            renamed_to=["magnetics/b_field_phi_probe/field"],
            rename_version="3.39.0",
        )
        result = FetchPathsResult(nodes=[node], not_found_paths=[], deprecated_paths=[])
        out = format_fetch_paths_report(result)
        assert "RENAMED TO: magnetics/b_field_phi_probe/field" in out
        assert "(in v3.39.0)" in out

    def test_renamed_from_rendered(self):
        node = self._make_node(
            "magnetics/b_field_phi_probe/field",
            renamed_from=["magnetics/b_field_tor_probe/field"],
        )
        result = FetchPathsResult(nodes=[node], not_found_paths=[], deprecated_paths=[])
        out = format_fetch_paths_report(result)
        assert "RENAMED FROM: magnetics/b_field_tor_probe/field" in out

    def test_no_lineage_no_annotation(self):
        node = self._make_node("equilibrium/time_slice/profiles_1d/psi")
        result = FetchPathsResult(nodes=[node], not_found_paths=[], deprecated_paths=[])
        out = format_fetch_paths_report(result)
        assert "RENAMED" not in out

    def test_renamed_to_without_version(self):
        node = self._make_node("old/path", renamed_to=["new/path"])
        result = FetchPathsResult(nodes=[node], not_found_paths=[], deprecated_paths=[])
        out = format_fetch_paths_report(result)
        assert "RENAMED TO: new/path" in out
        assert "in v" not in out


# ---------------------------------------------------------------------------
# Tests for fetch_dd_paths graph integration
# ---------------------------------------------------------------------------


class TestFetchDdPathsRenameAttachment:
    """fetch_dd_paths must attach rename lineage to returned IdsNode objects."""

    def _make_gc_for_fetch(self, main_return, rename_return):
        gc = MagicMock()

        def query_side_effect(cypher, **kwargs):
            # Distinguish the main path fetch from the lineage batch query
            if "RENAMED_TO*1..10" in cypher:
                return rename_return
            return main_return

        gc.query = MagicMock(side_effect=query_side_effect)
        return gc

    @pytest.mark.asyncio
    async def test_rename_lineage_attached_when_present(self):
        main_row = [
            {
                "id": "magnetics/b_field_tor_probe/field",
                "name": "field",
                "ids": "magnetics",
                "documentation": "Toroidal probe field.",
                "data_type": "FLT_0D",
                "node_type": "dynamic",
                "physics_domain": "magnetics",
                "ndim": 0,
                "structure_path": None,
                "lifecycle_status": "obsolescent",
                "lifecycle_version": "3.39.0",
                "cocos_label": None,
                "cocos_expression": None,
                "coordinate1": None,
                "coordinate2": None,
                "timebase": None,
                "description": None,
                "keywords": None,
                "enrichment_source": None,
                "units": "T",
                "cluster_labels": [],
                "coordinates": [],
                "identifier_schema_name": None,
                "identifier_schema_documentation": None,
                "identifier_schema_options": None,
                "introduced_after_version": "3.0.0",
                "version_changes": [],
            }
        ]
        rename_row = [
            {
                "id": "magnetics/b_field_tor_probe/field",
                "rename_version": "3.39.0",
                "renamed_from": [],
                "renamed_to": ["magnetics/b_field_phi_probe/field"],
            }
        ]
        gc = self._make_gc_for_fetch(main_row, rename_row)
        tool = GraphPathTool(gc)
        result = await tool.fetch_dd_paths("magnetics/b_field_tor_probe/field")

        assert result.nodes
        node = result.nodes[0]
        assert node.rename_lineage is not None
        assert node.rename_lineage["renamed_to"] == [
            "magnetics/b_field_phi_probe/field"
        ]

    @pytest.mark.asyncio
    async def test_rename_lineage_absent_when_none(self):
        main_row = [
            {
                "id": "equilibrium/time_slice/profiles_1d/psi",
                "name": "psi",
                "ids": "equilibrium",
                "documentation": "Poloidal flux.",
                "data_type": "FLT_1D",
                "node_type": "dynamic",
                "physics_domain": "equilibrium",
                "ndim": 1,
                "structure_path": None,
                "lifecycle_status": "active",
                "lifecycle_version": None,
                "cocos_label": "psi_like",
                "cocos_expression": None,
                "coordinate1": None,
                "coordinate2": None,
                "timebase": None,
                "description": None,
                "keywords": None,
                "enrichment_source": None,
                "units": "Wb",
                "cluster_labels": [],
                "coordinates": [],
                "identifier_schema_name": None,
                "identifier_schema_documentation": None,
                "identifier_schema_options": None,
                "introduced_after_version": None,
                "version_changes": [],
            }
        ]
        # Empty rename result → no lineage
        gc = self._make_gc_for_fetch(main_row, [])
        tool = GraphPathTool(gc)
        result = await tool.fetch_dd_paths("equilibrium/time_slice/profiles_1d/psi")

        assert result.nodes
        node = result.nodes[0]
        assert node.rename_lineage is None
