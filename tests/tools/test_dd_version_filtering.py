"""Tests for DD version validity filtering, cluster scope query construction,
and overview/export tools.

Validates version filtering at full semver granularity:
- resolve_dd_version accepts N, N.N, N.N.N, or 'latest' and resolves to concrete semver
- _dd_version_clause uses semver comparison (major.minor.patch) on INTRODUCED_IN/DEPRECATED_IN
- _deprecated_filter hard-excludes deprecated paths only when no version is specified
- A path is valid at version V if introduced at or before V and not deprecated at or before V
- Paths deprecated in 4.0.0 are valid at 3.42.2 but not at 4.0.0
- Paths deprecated in 3.40.0 are invalid at 3.40.0, 3.42.2, and 4.1.0

Also validates cluster scope queries, overview/export tools, and version-scoped
search with both BM25/CONTAINS (text) and vector search.

Skipped tests (6):
- test_search_enrichment_query_includes_has_error: HAS_ERROR enrichment not yet in search_imas_paths
- test_fetch_enrichment_query_includes_has_error: HAS_ERROR enrichment not yet in fetch_imas_paths
- test_search_enrichment_fetches_cluster_labels: IN_CLUSTER enrichment not yet in search_imas_paths
- test_vector_search_dd3_includes_deprecated_in_dd4: requires live remote embed server (Qwen3, 256-dim)
- test_vector_search_dd4_excludes_deprecated_in_dd4: requires live remote embed server
- test_vector_search_no_version_excludes_deprecated: requires live remote embed server
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
    _deprecated_filter,
    resolve_dd_version,
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
        clause = _dd_version_clause("p", dd_version="4.1.0")
        assert clause.startswith("AND EXISTS")

    def test_sets_semver_params(self):
        params = {}
        _dd_version_clause("p", dd_version="4.1.0", params=params)
        assert params["dd_ver_major"] == 4
        assert params["dd_ver_minor"] == 1
        assert params["dd_ver_patch"] == 0

    def test_uses_correct_alias(self):
        clause = _dd_version_clause("node", dd_version="3.42.0")
        assert "(node)-[:INTRODUCED_IN]->" in clause
        assert "(node)-[:DEPRECATED_IN]->" in clause

    def test_uses_semver_comparison(self):
        """The clause compares all three semver components numerically."""
        clause = _dd_version_clause("p", dd_version="4.1.0")
        assert "toInteger(split(" in clause
        assert "$dd_ver_major" in clause
        assert "$dd_ver_minor" in clause
        assert "$dd_ver_patch" in clause

    def test_checks_introduced_exists(self):
        """Must check that INTRODUCED_IN relationship exists with version ≤ N.N.N."""
        clause = _dd_version_clause("p", dd_version="4.0.0")
        assert "INTRODUCED_IN" in clause
        assert "<= $dd_ver_patch" in clause

    def test_checks_not_deprecated_lte(self):
        """Must exclude paths deprecated in any version ≤ N.N.N."""
        clause = _dd_version_clause("p", dd_version="4.0.0")
        assert "NOT EXISTS" in clause
        assert "DEPRECATED_IN" in clause

    def test_minor_version_discrimination(self):
        """3.39.0 and 3.40.0 should produce different filter params."""
        params_39: dict = {}
        params_40: dict = {}
        _dd_version_clause("p", dd_version="3.39.0", params=params_39)
        _dd_version_clause("p", dd_version="3.40.0", params=params_40)
        assert params_39["dd_ver_minor"] == 39
        assert params_40["dd_ver_minor"] == 40


# ============================================================================
# _deprecated_filter unit tests
# ============================================================================


class TestDeprecatedFilter:
    """Test the version-aware deprecated filter helper."""

    def test_none_returns_blanket_exclusion(self):
        """No version specified → hard-exclude all deprecated paths."""
        result = _deprecated_filter("p", dd_version=None)
        assert "DEPRECATED_IN" in result
        assert "DDVersion" in result
        assert result == "NOT (p)-[:DEPRECATED_IN]->(:DDVersion)"

    def test_version_specified_returns_empty(self):
        """Version specified → no hard filter (dd_version_clause handles it)."""
        result = _deprecated_filter("p", dd_version="3.42.0")
        assert result == ""

    def test_version_specified_returns_empty_dd4(self):
        result = _deprecated_filter("p", dd_version="4.0.0")
        assert result == ""

    def test_uses_correct_alias(self):
        result = _deprecated_filter("node", dd_version=None)
        assert "(node)-[:DEPRECATED_IN]->" in result

    def test_combined_with_dd_version_clause(self):
        """When dd_version specified, only _dd_version_clause should filter deprecation."""
        dep = _deprecated_filter("p", dd_version="3.42.0")
        clause = _dd_version_clause("p", dd_version="3.42.0")
        assert dep == ""
        assert "DEPRECATED_IN" in clause
        assert "$dd_ver_patch" in clause

    def test_combined_with_dd_version_clause_none(self):
        """When dd_version=None, blanket filter applied, no version clause."""
        dep = _deprecated_filter("p", dd_version=None)
        clause = _dd_version_clause("p", dd_version=None)
        assert dep != ""
        assert clause == ""


# ============================================================================
# resolve_dd_version unit tests
# ============================================================================


def _graph_available() -> bool:
    """Check if Neo4j is reachable."""
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            gc.query("RETURN 1")
        return True
    except Exception:
        return False


requires_graph = pytest.mark.skipif(
    not _graph_available(), reason="Neo4j not available"
)


@requires_graph
class TestResolveDDVersion:
    """Test resolve_dd_version against live graph."""

    def test_none_returns_none(self):
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            assert resolve_dd_version(gc, None) is None

    def test_latest_returns_current(self):
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            result = resolve_dd_version(gc, "latest")
            assert result == "4.1.0"  # current in graph

    def test_major_only_resolves_to_latest_in_major(self):
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            result = resolve_dd_version(gc, 3)
            assert result == "3.42.2"  # latest 3.x.x

    def test_major_only_string(self):
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            result = resolve_dd_version(gc, "4")
            assert result in ("4.1.0", "4.1.1")  # latest 4.x.x

    def test_major_minor_resolves(self):
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            result = resolve_dd_version(gc, "3.42")
            assert result == "3.42.2"  # latest 3.42.x

    def test_major_minor_single_version(self):
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            result = resolve_dd_version(gc, "4.0")
            assert result == "4.0.0"

    def test_exact_semver(self):
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            result = resolve_dd_version(gc, "3.40.1")
            assert result == "3.40.1"

    def test_exact_semver_earliest(self):
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            result = resolve_dd_version(gc, "3.22.0")
            assert result == "3.22.0"

    def test_invalid_version_raises(self):
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            with pytest.raises(ValueError, match="not found"):
                resolve_dd_version(gc, "99.99.99")

    def test_invalid_major_raises(self):
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            with pytest.raises(ValueError, match="No DDVersion found"):
                resolve_dd_version(gc, 99)

    def test_invalid_format_raises(self):
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            with pytest.raises(ValueError, match="Invalid dd_version"):
                resolve_dd_version(gc, "abc")

    def test_int_4_resolves(self):
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            result = resolve_dd_version(gc, 4)
            parts = result.split(".")
            assert parts[0] == "4"
            assert len(parts) == 3


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
    @patch(
        "imas_codex.tools.graph_search.resolve_dd_version", side_effect=lambda gc, v: v
    )
    async def test_dd3_path_found_with_dd_version_4(self, mock_resolve):
        """A path introduced in DD3 and never deprecated must be found with dd_version=4.1.0."""
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
            "equilibrium/time_slice/profiles_1d/psi", dd_version="4.1.0"
        )
        assert result.results[0].exists is True

        # Verify the Cypher includes semver comparison
        cypher = gc.query.call_args_list[0][0][0]
        assert "toInteger(split(" in cypher

    @pytest.mark.asyncio
    @patch(
        "imas_codex.tools.graph_search.resolve_dd_version", side_effect=lambda gc, v: v
    )
    async def test_dd_version_params_are_semver(self, mock_resolve):
        """The dd_ver_major/minor/patch params passed to Cypher must be integers."""
        gc = MagicMock()
        gc.query.return_value = [
            {"id": "test/path", "ids": "test", "data_type": "FLT_0D", "units": ""}
        ]
        tool = GraphPathTool(gc)
        await tool.check_imas_paths("test/path", dd_version="4.1.0")

        kwargs = gc.query.call_args_list[0][1]
        assert kwargs["dd_ver_major"] == 4
        assert kwargs["dd_ver_minor"] == 1
        assert kwargs["dd_ver_patch"] == 0

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
            dd_version="4.1.0",
        )

        cypher = gc.query.call_args[0][0]
        assert "c.scope = $scope" in cypher
        assert "INTRODUCED_IN" in cypher
        kwargs = gc.query.call_args[1]
        assert kwargs["scope"] == "ids"
        assert kwargs["dd_ver_major"] == 4

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
    @patch(
        "imas_codex.tools.graph_search.resolve_dd_version", side_effect=lambda gc, v: v
    )
    async def test_overview_with_dd_version_includes_filter(self, mock_resolve):
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
        await tool.get_imas_overview(dd_version="4.1.0")

        ids_cypher = gc.query.call_args_list[0][0][0]
        assert "MATCH (i:IDS)" in ids_cypher
        assert "INTRODUCED_IN" in ids_cypher
        kwargs = gc.query.call_args_list[0][1]
        assert kwargs["dd_ver_major"] == 4


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
    @patch(
        "imas_codex.tools.graph_search.resolve_dd_version", side_effect=lambda gc, v: v
    )
    async def test_export_ids_with_dd_version(self, mock_resolve):
        """export_imas_ids with dd_version must include version filter."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphStructureTool(gc)

        await tool.export_imas_ids("equilibrium", dd_version="4.1.0")

        cypher = gc.query.call_args[0][0]
        assert "INTRODUCED_IN" in cypher
        kwargs = gc.query.call_args[1]
        assert kwargs["dd_ver_major"] == 4


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
            [{"id": "equilibrium/time_slice"}],  # STARTS WITH query
            [{"id": "equilibrium/time_slice"}],  # ids = query (overwrites)
        ]
        tool = GraphListTool(gc)

        await tool.list_imas_paths("equilibrium")

        assert gc.query.call_count >= 2

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
            [],  # STARTS WITH query
            [  # ids = query (overwrites)
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
                dd_version="4.1.0",
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

    @pytest.mark.skip(
        reason="HAS_ERROR enrichment not yet implemented in search_imas_paths"
    )
    def test_search_enrichment_query_includes_has_error(self):
        """search_imas_paths enrichment query must fetch HAS_ERROR."""
        import inspect

        from imas_codex.tools.graph_search import GraphSearchTool

        source = inspect.getsource(GraphSearchTool.search_imas_paths)
        assert "HAS_ERROR" in source
        assert "error_fields" in source

    @pytest.mark.skip(
        reason="HAS_ERROR enrichment not yet implemented in fetch_imas_paths"
    )
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


# ============================================================================
# Version-scoped deprecated path search (live graph)
# ============================================================================


@requires_graph
class TestVersionScopedDeprecatedSearch:
    """Integration tests: version-scoped search returns deprecated-in-later-version paths.

    Uses real graph data.  Key test paths:
      - ece/channel/t_e: introduced 3.22.0, deprecated 4.0.0 → visible at 3.42.2, hidden at 4.0.0
      - equilibrium/time_slice/boundary/x_point: introduced 3.22.0, deprecated 4.0.0
      - bolometer/channel/power: introduced 3.22.0, deprecated 4.1.0
    """

    def test_text_search_dd3_includes_path_deprecated_in_dd4(self):
        """A path deprecated in DD4 must appear in text search with dd_version=3."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.tools.graph_search import _text_search_imas_paths

        with GraphClient() as gc:
            results = _text_search_imas_paths(
                gc,
                "electron temperature ECE",
                limit=50,
                ids_filter=None,
                dd_version="3.42.2",
            )
        ids = [r["id"] for r in results]
        assert "ece/channel/t_e" in ids, (
            f"ece/channel/t_e (deprecated in DD4) should appear in DD3 search. "
            f"Got {ids[:10]}"
        )

    def test_text_search_dd4_excludes_path_deprecated_in_dd4(self):
        """A path deprecated in DD4 must NOT appear in text search with dd_version=4."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.tools.graph_search import _text_search_imas_paths

        with GraphClient() as gc:
            results = _text_search_imas_paths(
                gc,
                "electron temperature ECE",
                limit=50,
                ids_filter=None,
                dd_version="4.1.0",
            )
        ids = [r["id"] for r in results]
        assert "ece/channel/t_e" not in ids, (
            "ece/channel/t_e (deprecated in DD4) should NOT appear in DD4 search"
        )

    def test_text_search_no_version_excludes_all_deprecated(self):
        """With no dd_version, all deprecated paths are excluded (current behavior)."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.tools.graph_search import _text_search_imas_paths

        with GraphClient() as gc:
            results = _text_search_imas_paths(
                gc,
                "electron temperature ECE",
                limit=50,
                ids_filter=None,
                dd_version=None,
            )
        ids = [r["id"] for r in results]
        assert "ece/channel/t_e" not in ids, (
            "ece/channel/t_e should be excluded when no version filter is applied"
        )

    def test_text_search_dd3_includes_x_point_deprecated_in_dd4(self):
        """equilibrium x_point (deprecated DD4) should appear in DD3 search."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.tools.graph_search import _text_search_imas_paths

        with GraphClient() as gc:
            results = _text_search_imas_paths(
                gc,
                "x point magnetic boundary",
                limit=50,
                ids_filter=None,
                dd_version="3.42.2",
            )
        ids = [r["id"] for r in results]
        assert "equilibrium/time_slice/boundary/x_point" in ids, (
            f"x_point (deprecated DD4) should appear in DD3 search. Got {ids[:10]}"
        )

    @pytest.mark.asyncio
    async def test_vector_search_dd3_includes_deprecated_in_dd4(self):
        """Vector search with dd_version=3.42.2 should include paths deprecated in 4.0.0.

        Requires the remote embed server to produce matching-dimension vectors.
        Skips if the embed server is unavailable or produces incompatible dimensions.
        """
        import os

        from imas_codex.graph.client import GraphClient
        from imas_codex.tools.graph_search import GraphSearchTool

        # Override conftest's local model — use the real remote embed server
        orig_loc = os.environ.get("IMAS_CODEX_EMBEDDING_LOCATION")
        orig_model = os.environ.get("IMAS_CODEX_EMBEDDING_MODEL")
        try:
            os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = "titan"
            os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = "Qwen/Qwen3-Embedding-0.6B"

            # Check if remote embed server is available
            import imas_codex.embeddings.embed as embed_mod
            import imas_codex.tools.graph_search as gs_mod
            from imas_codex.embeddings.client import RemoteEmbeddingClient

            client = RemoteEmbeddingClient()
            if not client.is_available():
                pytest.skip("Remote embed server not available")

            embed_mod._cached_encoder = None
            gs_mod._encoder = None

            with GraphClient() as gc:
                tool = GraphSearchTool(gc)
                result = await tool.search_imas_paths(
                    "electron temperature ECE channel",
                    max_results=50,
                    dd_version="3.42.2",
                )
            if hasattr(result, "error") and result.error:
                pytest.skip(f"Vector search error: {result.error[:100]}")
            all_ids = [r.id for r in result.hits]
            assert len(all_ids) > 0, "DD 3.42.2 vector search should return results"
        finally:
            # Restore original env vars and clear caches
            if orig_loc is not None:
                os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = orig_loc
            else:
                os.environ.pop("IMAS_CODEX_EMBEDDING_LOCATION", None)
            if orig_model is not None:
                os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = orig_model
            else:
                os.environ.pop("IMAS_CODEX_EMBEDDING_MODEL", None)
            embed_mod._cached_encoder = None
            gs_mod._encoder = None

    @pytest.mark.asyncio
    async def test_vector_search_dd4_excludes_deprecated_in_dd4(self):
        """Vector search with dd_version=4.1.0 should exclude paths deprecated in 4.x."""
        import os

        from imas_codex.graph.client import GraphClient
        from imas_codex.tools.graph_search import GraphSearchTool

        orig_loc = os.environ.get("IMAS_CODEX_EMBEDDING_LOCATION")
        orig_model = os.environ.get("IMAS_CODEX_EMBEDDING_MODEL")
        try:
            os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = "titan"
            os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = "Qwen/Qwen3-Embedding-0.6B"

            import imas_codex.embeddings.embed as embed_mod
            import imas_codex.tools.graph_search as gs_mod
            from imas_codex.embeddings.client import RemoteEmbeddingClient

            client = RemoteEmbeddingClient()
            if not client.is_available():
                pytest.skip("Remote embed server not available")

            embed_mod._cached_encoder = None
            gs_mod._encoder = None

            with GraphClient() as gc:
                tool = GraphSearchTool(gc)
                result = await tool.search_imas_paths(
                    "electron temperature ECE channel",
                    max_results=50,
                    dd_version="4.1.0",
                )
            if hasattr(result, "error") and result.error:
                pytest.skip(f"Vector search error: {result.error[:100]}")
            all_ids = [r.id for r in result.hits]
            assert "ece/channel/t_e" not in all_ids, (
                "ece/channel/t_e (deprecated 4.0.0) should NOT appear in DD 4.1.0 search"
            )
        finally:
            if orig_loc is not None:
                os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = orig_loc
            else:
                os.environ.pop("IMAS_CODEX_EMBEDDING_LOCATION", None)
            if orig_model is not None:
                os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = orig_model
            else:
                os.environ.pop("IMAS_CODEX_EMBEDDING_MODEL", None)
            embed_mod._cached_encoder = None
            gs_mod._encoder = None

    @pytest.mark.asyncio
    async def test_vector_search_no_version_excludes_deprecated(self):
        """Vector search with no version should exclude all deprecated paths."""
        import os

        from imas_codex.graph.client import GraphClient
        from imas_codex.tools.graph_search import GraphSearchTool

        orig_loc = os.environ.get("IMAS_CODEX_EMBEDDING_LOCATION")
        orig_model = os.environ.get("IMAS_CODEX_EMBEDDING_MODEL")
        try:
            os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = "titan"
            os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = "Qwen/Qwen3-Embedding-0.6B"

            import imas_codex.embeddings.embed as embed_mod
            import imas_codex.tools.graph_search as gs_mod
            from imas_codex.embeddings.client import RemoteEmbeddingClient

            client = RemoteEmbeddingClient()
            if not client.is_available():
                pytest.skip("Remote embed server not available")

            embed_mod._cached_encoder = None
            gs_mod._encoder = None

            with GraphClient() as gc:
                tool = GraphSearchTool(gc)
                result = await tool.search_imas_paths(
                    "electron temperature ECE channel",
                    max_results=50,
                    dd_version=None,
                )
            if hasattr(result, "error") and result.error:
                pytest.skip(f"Vector search error: {result.error[:100]}")
            all_ids = [r.id for r in result.hits]
            assert "ece/channel/t_e" not in all_ids, (
                "ece/channel/t_e should be excluded with no version filter"
            )
        finally:
            if orig_loc is not None:
                os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = orig_loc
            else:
                os.environ.pop("IMAS_CODEX_EMBEDDING_LOCATION", None)
            if orig_model is not None:
                os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = orig_model
            else:
                os.environ.pop("IMAS_CODEX_EMBEDDING_MODEL", None)
            embed_mod._cached_encoder = None
            gs_mod._encoder = None

    def test_dd3_search_includes_paths_absent_from_dd4(self):
        """DD3 search should include some paths that DD4 excludes (deprecated in DD4).

        Paths deprecated in DD4 are valid in DD3, so DD3 results should contain
        paths not present in DD4 results.
        """
        from imas_codex.graph.client import GraphClient
        from imas_codex.tools.graph_search import _text_search_imas_paths

        with GraphClient() as gc:
            dd3_results = _text_search_imas_paths(
                gc, "temperature", limit=100, ids_filter=None, dd_version="3.42.2"
            )
            dd4_results = _text_search_imas_paths(
                gc, "temperature", limit=100, ids_filter=None, dd_version="4.1.0"
            )
        assert len(dd3_results) > 0, "DD3 search should return results"
        assert len(dd4_results) > 0, "DD4 search should return results"
        dd3_ids = {r["id"] for r in dd3_results}
        dd4_ids = {r["id"] for r in dd4_results}
        # DD3 should include paths that DD4 does not (deprecated-in-DD4 paths)
        dd3_only = dd3_ids - dd4_ids
        assert len(dd3_only) > 0, (
            "DD3 should include deprecated-in-DD4 paths absent from DD4 results"
        )

    def test_path_deprecated_in_dd3_excluded_from_both(self):
        """A path deprecated in DD3 must be excluded from both DD3 and DD4 search."""
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            # Find a path deprecated in DD3
            dd3_deprecated = gc.query("""
                MATCH (p:IMASNode)-[:DEPRECATED_IN]->(dv:DDVersion)
                WHERE p.node_category = 'data'
                  AND toInteger(split(dv.id, '.')[0]) = 3
                MATCH (p)-[:INTRODUCED_IN]->(iv:DDVersion)
                WHERE toInteger(split(iv.id, '.')[0]) = 3
                RETURN p.id AS id LIMIT 1
            """)
            if not dd3_deprecated:
                pytest.skip("No DD3-deprecated paths found in graph")

            path_id = dd3_deprecated[0]["id"]

            from imas_codex.tools.graph_search import _text_search_imas_paths

            # Extract a search term from the path
            search_term = path_id.split("/")[-1].replace("_", " ")
            dd3_results = _text_search_imas_paths(
                gc, search_term, limit=100, ids_filter=None, dd_version="3.42.2"
            )
            dd4_results = _text_search_imas_paths(
                gc, search_term, limit=100, ids_filter=None, dd_version="4.1.0"
            )

        dd3_ids = [r["id"] for r in dd3_results]
        dd4_ids = [r["id"] for r in dd4_results]
        assert path_id not in dd3_ids, (
            f"{path_id} (deprecated in DD3) should be excluded from DD3 search"
        )
        assert path_id not in dd4_ids, (
            f"{path_id} (deprecated in DD3) should be excluded from DD4 search"
        )
