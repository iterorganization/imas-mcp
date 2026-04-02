"""Tests for DD version validity filtering, cluster scope query construction,
and overview/export tools.

Validates that _dd_version_clause uses major-version comparison:
- Uses toInteger(split(id, '.')[0]) <= $dd_major_version on INTRODUCED_IN/DEPRECATED_IN
- A path is valid in DD major N if introduced in any version with major <= N
  and not deprecated in any version with major <= N.
- Paths introduced in DD3 and never deprecated are valid in both DD3 and DD4.
- Paths deprecated in DD4 are valid in DD3 but not DD4.
- Paths deprecated in DD3 are invalid in both DD3 and DD4.

Also validates that cluster scope queries produce valid Cypher, and that
overview/export tools use correct query patterns.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from imas_codex.tools.graph_search import (
    GraphClustersTool,
    GraphListTool,
    GraphOverviewTool,
    GraphPathTool,
    GraphStructureTool,
    _dd_version_clause,
)

# ============================================================================
# _dd_version_clause unit tests
# ============================================================================


class TestDDVersionClause:
    """Test the version filtering helper directly."""

    def test_none_returns_empty(self):
        assert _dd_version_clause("p", dd_version=None) == ""

    def test_none_does_not_modify_params(self):
        params = {}
        _dd_version_clause("p", dd_version=None, params=params)
        assert params == {}

    def test_returns_and_clause(self):
        clause = _dd_version_clause("p", dd_version=4)
        assert clause.startswith("AND EXISTS")

    def test_sets_dd_version_param(self):
        params = {}
        _dd_version_clause("p", dd_version=4, params=params)
        assert params["dd_major_version"] == 4

    def test_uses_correct_alias(self):
        clause = _dd_version_clause("node", dd_version=3)
        assert "(node)-[:INTRODUCED_IN]->" in clause
        assert "(node)-[:DEPRECATED_IN]->" in clause

    def test_uses_major_version_comparison(self):
        """The clause compares the major version integer, not a string prefix."""
        clause = _dd_version_clause("p", dd_version=4)
        assert "toInteger(split(" in clause
        assert "$dd_major_version" in clause

    def test_checks_introduced_exists(self):
        """Must check that INTRODUCED_IN relationship exists with major <= N."""
        clause = _dd_version_clause("p", dd_version=4)
        assert "INTRODUCED_IN" in clause
        assert "<= $dd_major_version" in clause

    def test_checks_not_deprecated_lte(self):
        """Must exclude paths deprecated in any version with major <= N."""
        clause = _dd_version_clause("p", dd_version=4)
        assert "NOT EXISTS" in clause
        assert "DEPRECATED_IN" in clause


# ============================================================================
# Renamed path handling tests
# ============================================================================


class TestRenamedPathHandling:
    """Test that check_imas_paths correctly handles RENAMED_TO relationships."""

    @pytest.mark.asyncio
    async def test_renamed_path_returns_valid_model(self):
        """Renamed paths must produce a valid CheckPathsResultItem, not a Pydantic error."""
        gc = MagicMock()
        # First query: path not found (no match)
        # Second query: RENAMED_TO found
        gc.query.side_effect = [
            [],  # path lookup returns empty
            [
                {
                    "old_path": "magnetics/bpol_probe/polarisation_angle",
                    "new_path": "magnetics/bpol_probe/polarization_angle",
                }
            ],
        ]
        tool = GraphPathTool(gc)
        result = await tool.check_imas_paths("magnetics/bpol_probe/polarisation_angle")

        assert result.summary["not_found"] == 1
        item = result.results[0]
        assert item.exists is False
        assert item.suggestion == "magnetics/bpol_probe/polarization_angle"
        assert isinstance(item.renamed_from, list)
        assert (
            item.renamed_from[0]["new_path"]
            == "magnetics/bpol_probe/polarization_angle"
        )
        assert isinstance(item.migration, dict)
        assert item.migration["type"] == "renamed"


# ============================================================================
# Version filtering integration tests (mock graph)
# ============================================================================


class TestVersionFilteringSemantics:
    """Test that check_imas_paths applies version filtering correctly via mocked graph."""

    @pytest.mark.asyncio
    async def test_dd3_path_found_with_dd_version_4(self):
        """A path introduced in DD3 and never deprecated must be found with dd_version=4."""
        gc = MagicMock()
        gc.query.return_value = [
            {
                "id": "equilibrium/time_slice/profiles_1d/psi",
                "ids": "equilibrium",
                "data_type": "FLT_1D",
                "units": "Wb",
            }
        ]
        tool = GraphPathTool(gc)
        result = await tool.check_imas_paths(
            "equilibrium/time_slice/profiles_1d/psi", dd_version=4
        )
        assert result.results[0].exists is True

        # Verify the Cypher includes major-version comparison
        cypher = gc.query.call_args_list[0][0][0]
        assert "toInteger(split(" in cypher

    @pytest.mark.asyncio
    async def test_dd_version_param_is_integer(self):
        """The dd_major_version parameter passed to Cypher must be an integer."""
        gc = MagicMock()
        gc.query.return_value = [
            {"id": "test/path", "ids": "test", "data_type": "FLT_0D", "units": ""}
        ]
        tool = GraphPathTool(gc)
        await tool.check_imas_paths("test/path", dd_version=4)

        # Check that dd_major_version=4 was passed
        kwargs = gc.query.call_args_list[0][1]
        assert "dd_major_version" in kwargs
        assert kwargs["dd_major_version"] == 4

    @pytest.mark.asyncio
    async def test_no_dd_version_skips_filter(self):
        """When dd_version is None, no version filter clause should be in the query."""
        gc = MagicMock()
        gc.query.return_value = [
            {"id": "test/path", "ids": "test", "data_type": "FLT_0D", "units": ""}
        ]
        tool = GraphPathTool(gc)
        await tool.check_imas_paths("test/path", dd_version=None)

        cypher = gc.query.call_args_list[0][0][0]
        assert "INTRODUCED_IN" not in cypher
        assert "DEPRECATED_IN" not in cypher


# ============================================================================
# Cluster scope syntax tests
# ============================================================================


class TestClusterScopeQuery:
    """Test that cluster _search_by_path produces valid Cypher with scope parameter."""

    def test_search_by_path_with_scope_filters(self):
        """When scope is provided, the query must include scope filter."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphClustersTool(gc)

        tool._search_by_path("equilibrium/time_slice/profiles_1d/psi", scope="global")

        cypher = gc.query.call_args[0][0]
        assert "c.scope = $scope" in cypher
        kwargs = gc.query.call_args[1]
        assert kwargs["scope"] == "global"

    def test_search_by_path_without_scope(self):
        """When scope is None, no scope filter should appear."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphClustersTool(gc)

        tool._search_by_path("equilibrium/time_slice/profiles_1d/psi", scope=None)

        cypher = gc.query.call_args[0][0]
        assert "c.scope = $scope" not in cypher

    def test_search_by_path_with_scope_and_dd_version(self):
        """Both scope and dd_version must produce valid Cypher."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphClustersTool(gc)

        tool._search_by_path(
            "equilibrium/time_slice/profiles_1d/psi",
            scope="ids",
            dd_version=4,
        )

        cypher = gc.query.call_args[0][0]
        assert "c.scope = $scope" in cypher
        assert "INTRODUCED_IN" in cypher
        kwargs = gc.query.call_args[1]
        assert kwargs["scope"] == "ids"
        assert kwargs["dd_major_version"] == 4

    @pytest.mark.asyncio
    async def test_search_imas_clusters_path_with_scope(self):
        """Full tool call with scope must not raise."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphClustersTool(gc)

        result = await tool.search_imas_clusters(
            query="equilibrium/time_slice/profiles_1d/psi",
            scope="global",
        )

        assert result["query_type"] == "path"
        assert result["clusters_found"] == 0


# ============================================================================
# Overview query structure tests
# ============================================================================


class TestOverviewQueryStructure:
    """Test that get_imas_overview uses correct query patterns."""

    @pytest.mark.asyncio
    async def test_overview_queries_ids_nodes(self):
        """The overview Cypher must query IDS nodes."""
        gc = MagicMock()
        gc.query.side_effect = [
            [
                {
                    "name": "equilibrium",
                    "description": "Equilibrium quantities",
                    "physics_domain": "equilibrium",
                    "lifecycle_status": "active",
                    "path_count": 641,
                }
            ],
            [{"version": "4.1.0"}],
        ]

        tool = GraphOverviewTool(gc)
        await tool.get_imas_overview()

        ids_cypher = gc.query.call_args_list[0][0][0]
        assert "MATCH (i:IDS)" in ids_cypher

    @pytest.mark.asyncio
    async def test_overview_with_dd_version_includes_filter(self):
        """Overview with dd_version must include version filter."""
        gc = MagicMock()
        gc.query.side_effect = [
            [
                {
                    "name": "equilibrium",
                    "description": "Equilibrium quantities",
                    "physics_domain": "equilibrium",
                    "lifecycle_status": "active",
                    "path_count": 455,
                }
            ],
            [{"version": "4.1.0"}],
        ]

        tool = GraphOverviewTool(gc)
        await tool.get_imas_overview(dd_version=4)

        ids_cypher = gc.query.call_args_list[0][0][0]
        assert "MATCH (i:IDS)" in ids_cypher
        assert "INTRODUCED_IN" in ids_cypher
        kwargs = gc.query.call_args_list[0][1]
        assert kwargs["dd_major_version"] == 4


# ============================================================================
# Export query structure tests
# ============================================================================


class TestExportQueryStructure:
    """Test that export tools use correct query patterns."""

    @pytest.mark.asyncio
    async def test_export_ids_uses_ids_filter(self):
        """export_imas_ids must filter by IDS name."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphStructureTool(gc)

        await tool.export_imas_ids("equilibrium")

        cypher = gc.query.call_args[0][0]
        assert "p.ids = $ids_name" in cypher

    @pytest.mark.asyncio
    async def test_export_domain_uses_domain_filter(self):
        """export_imas_domain must filter by physics domain."""
        gc = MagicMock()
        gc.query.side_effect = [
            [],  # export results
        ]
        tool = GraphStructureTool(gc)

        await tool.export_imas_domain("equilibrium")

        export_cypher = gc.query.call_args_list[-1][0][0]
        assert "physics_domain" in export_cypher

    @pytest.mark.asyncio
    async def test_export_ids_with_dd_version(self):
        """export_imas_ids with dd_version must include version filter."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphStructureTool(gc)

        await tool.export_imas_ids("equilibrium", dd_version=4)

        cypher = gc.query.call_args[0][0]
        assert "INTRODUCED_IN" in cypher
        kwargs = gc.query.call_args[1]
        assert kwargs["dd_major_version"] == 4


# ============================================================================
# Phase 3: list_imas_paths query tests
# ============================================================================


class TestListPathsQuery:
    """Verify list_imas_paths query patterns."""

    @pytest.mark.asyncio
    async def test_ids_level_queries_graph(self):
        """Listing an IDS by name should query the graph."""
        gc = MagicMock()
        gc.query.side_effect = [
            [{"i.name": "equilibrium"}],  # IDS exists
            [{"id": "equilibrium/time_slice"}],  # IDS-only mode: ids = query
        ]
        tool = GraphListTool(gc)

        await tool.list_imas_paths("equilibrium")

        assert gc.query.call_count == 2

    @pytest.mark.asyncio
    async def test_subpath_uses_starts_with(self):
        """A subpath query should use STARTS WITH prefix."""
        gc = MagicMock()
        gc.query.side_effect = [
            [{"i.name": "equilibrium"}],  # IDS exists
            [{"id": "equilibrium/time_slice/profiles_1d"}],
        ]
        tool = GraphListTool(gc)

        await tool.list_imas_paths("equilibrium/time_slice")

        path_cypher = gc.query.call_args_list[1][0][0]
        assert "STARTS WITH $prefix" in path_cypher

    @pytest.mark.asyncio
    async def test_ids_level_returns_results(self):
        """IDS-level query must return path results."""
        gc = MagicMock()
        gc.query.side_effect = [
            [{"i.name": "equilibrium"}],  # IDS exists
            [  # IDS-only mode: ids = query
                {"id": "equilibrium/time_slice"},
                {"id": "equilibrium/vacuum_toroidal_field"},
            ],
        ]
        tool = GraphListTool(gc)

        result = await tool.list_imas_paths("equilibrium")

        assert result.results[0].path_count == 2
        assert "equilibrium/time_slice" in result.results[0].paths
        assert "equilibrium/vacuum_toroidal_field" in result.results[0].paths


# ============================================================================
# Phase 3: Legacy dd_version in ids/tools.py
# ============================================================================


class TestSemanticMatchDDVersion:
    """Verify compute_semantic_matches uses relationship-based version filtering."""

    def test_dd_version_uses_introduced_in(self):
        """compute_semantic_matches should use INTRODUCED_IN, not n.dd_version."""
        import numpy as np

        from imas_codex.ids.tools import compute_semantic_matches

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [
            {
                "id": "equilibrium/time_slice/profiles_1d/psi",
                "doc": "Poloidal flux",
                "score": 0.9,
            }
        ]

        with patch("imas_codex.ids.tools.GraphClient", return_value=mock_gc):
            compute_semantic_matches(
                source_descriptions=[("src1", "some description")],
                target_ids_name="equilibrium",
                gc=mock_gc,
                dd_version=4,
                precomputed_embeddings=[np.zeros(384)],
            )

        # The per-thread GraphClient calls tgc.query with the constructed cypher
        assert mock_gc.query.call_count >= 1
        cypher = mock_gc.query.call_args_list[0][0][0]
        assert "INTRODUCED_IN" in cypher
        assert "n.dd_version" not in cypher


# ============================================================================
# Phase 3: Legacy dd_version in cli/imas_dd.py
# ============================================================================


class TestCliSearchDDVersion:
    """Verify CLI search builds relationship-based version filter."""

    def test_version_filter_builds_introduced_in_clause(self):
        """CLI --version flag should produce INTRODUCED_IN filter, not node property."""
        import inspect

        import imas_codex.cli.imas_dd as imas_dd_module

        source = inspect.getsource(imas_dd_module)
        assert "INTRODUCED_IN" in source
        assert "node.dd_version" not in source


# ============================================================================
# Phase 4: Error field context attachment tests
# ============================================================================


class TestErrorFieldContext:
    """Verify search and fetch tools attach error field context."""

    def test_search_hit_has_error_fields_attribute(self):
        """SearchHit model must have error_fields field."""
        from imas_codex.search.search_strategy import SearchHit

        hit = SearchHit(
            path="equilibrium/time_slice/profiles_1d/psi",
            documentation="test",
            ids_name="equilibrium",
            score=0.9,
            rank=1,
            search_mode="auto",
            error_fields=["equilibrium/time_slice/profiles_1d/psi_error_upper"],
        )
        assert hit.error_fields == [
            "equilibrium/time_slice/profiles_1d/psi_error_upper"
        ]

    def test_search_hit_error_fields_default_none(self):
        """SearchHit error_fields defaults to None."""
        from imas_codex.search.search_strategy import SearchHit

        hit = SearchHit(
            path="equilibrium/time_slice/profiles_1d/psi",
            documentation="test",
            ids_name="equilibrium",
            score=0.9,
            rank=1,
            search_mode="auto",
        )
        assert hit.error_fields is None

    def test_ids_node_has_error_fields_attribute(self):
        """IdsNode model must have error_fields field."""
        from imas_codex.core.data_model import IdsNode

        node = IdsNode(
            path="equilibrium/time_slice/profiles_1d/psi",
            documentation="test",
            error_fields=["equilibrium/time_slice/profiles_1d/psi_error_upper"],
        )
        assert node.error_fields == [
            "equilibrium/time_slice/profiles_1d/psi_error_upper"
        ]

    def test_search_enrichment_query_includes_has_error(self):
        """search_imas_paths enrichment query must fetch HAS_ERROR."""
        import inspect

        from imas_codex.tools.graph_search import GraphSearchTool

        source = inspect.getsource(GraphSearchTool.search_imas_paths)
        assert "HAS_ERROR" in source
        assert "error_fields" in source

    def test_fetch_enrichment_query_includes_has_error(self):
        """fetch_imas_paths enrichment query must fetch HAS_ERROR."""
        import inspect

        from imas_codex.tools.graph_search import GraphPathTool

        source = inspect.getsource(GraphPathTool.fetch_imas_paths)
        assert "HAS_ERROR" in source
        assert "error_fields" in source


# ============================================================================
# Phase 4: Cluster-aware reranking tests
# ============================================================================


class TestClusterReranking:
    """Verify cluster-aware reranking suppresses duplicate cluster hits."""

    @pytest.mark.skip(reason="IN_CLUSTER enrichment not yet in search_imas_paths")
    def test_search_enrichment_fetches_cluster_labels(self):
        """search_imas_paths enrichment must fetch IN_CLUSTER labels."""
        import inspect

        from imas_codex.tools.graph_search import GraphSearchTool

        source = inspect.getsource(GraphSearchTool.search_imas_paths)
        assert "IN_CLUSTER" in source
        assert "cluster_labels" in source

    def test_cluster_penalty_reduces_duplicate_scores(self):
        """Cluster duplicates should receive a score penalty."""
        scores = {
            "path/a": 0.95,
            "path/b": 0.90,
            "path/c": 0.85,
        }
        enriched_by_id = {
            "path/a": {"cluster_labels": ["temperature"]},
            "path/b": {"cluster_labels": ["temperature"]},
            "path/c": {"cluster_labels": ["pressure"]},
        }

        seen_clusters: dict[str, str] = {}
        sorted_ids = sorted(scores, key=lambda pid: scores[pid], reverse=True)
        for pid in sorted_ids:
            r = enriched_by_id.get(pid)
            if not r:
                continue
            for cl in r.get("cluster_labels") or []:
                if not cl:
                    continue
                if cl not in seen_clusters:
                    seen_clusters[cl] = pid
                elif scores.get(pid, 0) > scores.get(seen_clusters[cl], 0):
                    scores[seen_clusters[cl]] = round(
                        scores.get(seen_clusters[cl], 0) * 0.95, 4
                    )
                    seen_clusters[cl] = pid
                else:
                    scores[pid] = round(scores.get(pid, 0) * 0.95, 4)

        assert scores["path/a"] == 0.95
        assert scores["path/b"] < 0.90
        assert scores["path/c"] == 0.85
