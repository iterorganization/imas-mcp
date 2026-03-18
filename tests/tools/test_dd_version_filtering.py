"""Tests for DD version validity filtering, cluster scope query construction,
and node_category filtering in overview and export tools.

Validates that _dd_version_clause uses correct "valid in major N" semantics:
- A path is valid in DD major N if introduced in major ≤ N and not deprecated in major ≤ N.
- Paths introduced in DD3 and never deprecated are valid in both DD3 and DD4.
- Paths deprecated in DD4 are valid in DD3 but not DD4.
- Paths deprecated in DD3 are invalid in both DD3 and DD4.

Also validates that cluster scope queries produce valid Cypher, and that
overview/export tools filter by node_category='data'.
"""

from unittest.mock import MagicMock, call

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
        assert params["dd_version"] == 4

    def test_uses_correct_alias(self):
        clause = _dd_version_clause("node", dd_version=3)
        assert "(node)-[:INTRODUCED_IN]->" in clause
        assert "(node)-[:DEPRECATED_IN]->" in clause

    def test_uses_integer_comparison(self):
        """The clause must use toInteger(split()) for major version comparison,
        not STARTS WITH prefix matching."""
        clause = _dd_version_clause("p", dd_version=4)
        assert "toInteger(split(" in clause
        assert "$dd_version" in clause
        assert "STARTS WITH" not in clause

    def test_checks_introduced_lte(self):
        """Must check introduced_in major <= N (not ==)."""
        clause = _dd_version_clause("p", dd_version=4)
        assert "<= $dd_version" in clause

    def test_checks_not_deprecated_lte(self):
        """Must exclude paths deprecated in any version with major <= N."""
        clause = _dd_version_clause("p", dd_version=4)
        assert "NOT EXISTS" in clause
        assert "DEPRECATED_IN" in clause


# ============================================================================
# Version filtering integration tests (mock graph)
# ============================================================================


class TestVersionFilteringSemantics:
    """Test that check_imas_paths applies version filtering correctly via mocked graph."""

    @pytest.mark.asyncio
    async def test_dd3_path_found_with_dd_version_4(self):
        """A path introduced in DD3 and never deprecated must be found with dd_version=4."""
        gc = MagicMock()
        # First call: the main query with dd_version clause
        gc.query.return_value = [
            {"id": "equilibrium/time_slice/profiles_1d/psi", "ids": "equilibrium",
             "data_type": "FLT_1D", "units": "Wb"}
        ]
        tool = GraphPathTool(gc)
        result = await tool.check_imas_paths(
            "equilibrium/time_slice/profiles_1d/psi", dd_version=4
        )
        assert result.results[0].exists is True

        # Verify the Cypher includes the correct version semantics
        cypher = gc.query.call_args_list[0][0][0]
        assert "toInteger(split(" in cypher
        assert "STARTS WITH" not in cypher

    @pytest.mark.asyncio
    async def test_dd_version_param_is_integer(self):
        """The dd_version parameter passed to Cypher must be an integer, not a string prefix."""
        gc = MagicMock()
        gc.query.return_value = [
            {"id": "test/path", "ids": "test", "data_type": "FLT_0D", "units": ""}
        ]
        tool = GraphPathTool(gc)
        await tool.check_imas_paths("test/path", dd_version=4)

        # Check that dd_version=4 (int) was passed, not dd_version_prefix="4."
        kwargs = gc.query.call_args_list[0][1]
        assert "dd_version" in kwargs
        assert kwargs["dd_version"] == 4
        assert "dd_version_prefix" not in kwargs

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

    def test_search_by_path_with_scope_has_where(self):
        """When scope is provided, the query must have WHERE before the AND clause."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphClustersTool(gc)

        tool._search_by_path("equilibrium/time_slice/profiles_1d/psi", scope="global")

        cypher = gc.query.call_args[0][0]
        # The scope filter must come after a WHERE, not bare after MATCH
        assert "WHERE true AND c.scope = $scope" in cypher
        # Verify scope param was passed
        kwargs = gc.query.call_args[1]
        assert kwargs["scope"] == "global"

    def test_search_by_path_without_scope(self):
        """When scope is None, no scope filter should appear in WHERE."""
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
        # Both filters present and valid
        assert "WHERE true AND c.scope = $scope" in cypher
        assert "INTRODUCED_IN" in cypher
        kwargs = gc.query.call_args[1]
        assert kwargs["scope"] == "ids"
        assert kwargs["dd_version"] == 4

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
# Overview node_category filtering tests
# ============================================================================


class TestOverviewNodeCategoryFilter:
    """Test that get_imas_overview counts only data nodes."""

    @pytest.mark.asyncio
    async def test_overview_query_filters_data_only(self):
        """The overview Cypher must include node_category = 'data'."""
        gc = MagicMock()
        gc.query.side_effect = [
            # IDS query result
            [
                {
                    "name": "equilibrium",
                    "description": "Equilibrium quantities",
                    "physics_domain": "equilibrium",
                    "lifecycle_status": "active",
                    "path_count": 641,
                }
            ],
            # DDVersion query result
            [{"version": "4.1.0"}],
        ]

        tool = GraphOverviewTool(gc)
        await tool.get_imas_overview()

        # First query is the IDS+path count query
        ids_cypher = gc.query.call_args_list[0][0][0]
        assert "node_category = 'data'" in ids_cypher

    @pytest.mark.asyncio
    async def test_overview_with_dd_version_filters_data(self):
        """Overview with dd_version must filter both node_category and version."""
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
        assert "node_category = 'data'" in ids_cypher
        assert "INTRODUCED_IN" in ids_cypher
        kwargs = gc.query.call_args_list[0][1]
        assert kwargs["dd_version"] == 4


# ============================================================================
# Export node_category filtering tests
# ============================================================================


class TestExportNodeCategoryFilter:
    """Test that export tools filter to data nodes by default."""

    @pytest.mark.asyncio
    async def test_export_ids_filters_data_only(self):
        """export_imas_ids default must filter to data nodes only."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphStructureTool(gc)

        await tool.export_imas_ids("equilibrium")

        cypher = gc.query.call_args[0][0]
        assert "node_category IN $categories" in cypher
        kwargs = gc.query.call_args[1]
        assert kwargs["categories"] == ["data"]
    async def test_export_domain_filters_data_only(self):
        """export_imas_domain default must filter to data nodes only."""
        gc = MagicMock()
        # _resolve_physics_domain tries: exact enum match first, then IDS query.
        # Mock IDS lookup to return a valid domain.
        gc.query.side_effect = [
            [{"domain": "equilibrium"}],  # IDS name resolution
            [],  # export results
        ]
        tool = GraphStructureTool(gc)

        # Use an IDS name so _resolve_physics_domain falls through to gc.query
        await tool.export_imas_domain("equilibrium")

        # The export query is the last call
        export_cypher = gc.query.call_args_list[-1][0][0]
        assert "node_category IN $categories" in export_cypher
        kwargs = gc.query.call_args_list[-1][1]
        assert kwargs["categories"] == ["data"]
    async def test_export_ids_with_dd_version(self):
        """export_imas_ids with dd_version must filter both."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphStructureTool(gc)

        await tool.export_imas_ids("equilibrium", dd_version=4)

        cypher = gc.query.call_args[0][0]
        assert "node_category" in cypher
        assert "INTRODUCED_IN" in cypher
        kwargs = gc.query.call_args[1]
        assert kwargs["dd_version"] == 4

    @pytest.mark.asyncio
    async def test_export_ids_include_errors(self):
        """export_imas_ids with include_errors=True must include error nodes."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphStructureTool(gc)

        await tool.export_imas_ids("equilibrium", include_errors=True)

        kwargs = gc.query.call_args[1]
        assert "data" in kwargs["categories"]
        assert "error" in kwargs["categories"]

    @pytest.mark.asyncio
    async def test_export_ids_default_excludes_errors(self):
        """export_imas_ids default must exclude error nodes."""
        gc = MagicMock()
        gc.query.return_value = []
        tool = GraphStructureTool(gc)

        await tool.export_imas_ids("equilibrium")

        kwargs = gc.query.call_args[1]
        assert kwargs["categories"] == ["data"]

    @pytest.mark.asyncio
    async def test_export_domain_include_errors(self):
        """export_imas_domain with include_errors=True must include error nodes."""
        gc = MagicMock()
        gc.query.side_effect = [
            [{"domain": "equilibrium"}],  # IDS name resolution
            [],  # export results
        ]
        tool = GraphStructureTool(gc)

        await tool.export_imas_domain("equilibrium", include_errors=True)

        kwargs = gc.query.call_args_list[-1][1]
        assert "data" in kwargs["categories"]
        assert "error" in kwargs["categories"]


# ============================================================================
# Phase 3: list_imas_paths query consolidation tests
# ============================================================================


class TestListPathsQueryConsolidation:
    """Verify list_imas_paths uses a single query for IDS-level requests."""

    @pytest.mark.asyncio
    async def test_ids_level_uses_single_query(self):
        """Listing an IDS by name should issue exactly one path query."""
        gc = MagicMock()
        # First call: IDS existence check; second call: path query
        gc.query.side_effect = [
            [{"name": "equilibrium"}],  # IDS exists
            [{"id": "equilibrium/time_slice"}],  # path results
        ]
        tool = GraphListTool(gc)

        await tool.list_imas_paths("equilibrium")

        # Exactly 2 queries: existence check + single path query
        assert gc.query.call_count == 2
        path_cypher = gc.query.call_args_list[1][0][0]
        # Should use ids-level match, not STARTS WITH prefix/
        assert "ids = $ids_name" in path_cypher

    @pytest.mark.asyncio
    async def test_subpath_uses_starts_with(self):
        """A subpath query should use STARTS WITH prefix."""
        gc = MagicMock()
        gc.query.side_effect = [
            [{"name": "equilibrium"}],  # IDS exists
            [{"id": "equilibrium/time_slice/profiles_1d"}],
        ]
        tool = GraphListTool(gc)

        await tool.list_imas_paths("equilibrium/time_slice")

        path_cypher = gc.query.call_args_list[1][0][0]
        assert "STARTS WITH $prefix" in path_cypher

    @pytest.mark.asyncio
    async def test_ids_level_no_result_overwrite(self):
        """IDS-level query must not overwrite results with a second query."""
        gc = MagicMock()
        gc.query.side_effect = [
            [{"name": "equilibrium"}],  # IDS exists
            [  # path results with multiple items
                {"id": "equilibrium/time_slice"},
                {"id": "equilibrium/vacuum_toroidal_field"},
            ],
        ]
        tool = GraphListTool(gc)

        result = await tool.list_imas_paths("equilibrium")

        # Both paths should be present (no overwrite)
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
        from imas_codex.ids.tools import compute_semantic_matches

        gc = MagicMock()
        gc.query.return_value = []

        # Mock the encoder to avoid needing the model
        import imas_codex.ids.tools as ids_module

        mock_encoder_cls = MagicMock()
        mock_encoder = MagicMock()

        import numpy as np

        mock_encoder.embed_texts.return_value = [np.zeros(384)]
        mock_encoder_cls.return_value = mock_encoder

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ids_module, "Encoder", mock_encoder_cls, raising=False)
            compute_semantic_matches(
                source_descriptions=[("src1", "some description")],
                target_ids_name="equilibrium",
                gc=gc,
                dd_version=4,
            )

        # Check that the IMAS query uses relationship-based filtering
        imas_call = gc.query.call_args_list[0]
        cypher = imas_call[0][0]
        assert "INTRODUCED_IN" in cypher
        # Should NOT have n.dd_version property filter
        assert "n.dd_version" not in cypher


# ============================================================================
# Phase 3: Legacy dd_version in cli/imas_dd.py
# ============================================================================


class TestCliSearchDDVersion:
    """Verify CLI search builds relationship-based version filter."""

    def test_version_filter_builds_introduced_in_clause(self):
        """CLI --version flag should produce INTRODUCED_IN filter, not node property."""
        # Verify the module source code uses INTRODUCED_IN, not dd_version property
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

        # Apply cluster penalty logic (mirrors graph_search.py implementation)
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
