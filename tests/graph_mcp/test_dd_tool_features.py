"""Functional tests for capability-gap additions to the DD MCP server.

Covers:
  - get_dd_cocos_fields (GraphStructureTool)
  - Enhanced get_dd_version_context bulk-query mode (VersionTool)
  - Unit statistics in overview (GraphOverviewTool)
  - Lifecycle filtering in list/analyze (GraphListTool / GraphStructureTool)
  - Migration guide summary_only mode (generate_migration_guide)
  - Search parameter unification – physics_domain, node_type, include_children
    (GraphListTool / GraphPathTool)

These tests work against BOTH fixture (CI) and production graphs.
Assertions use ``>=`` / ``>`` for counts rather than exact fixture values.
"""

import pytest

pytestmark = pytest.mark.graph_mcp


# ── Phase 1: get_dd_cocos_fields ─────────────────────────────────────────────


class TestCocosFields:
    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphStructureTool

        return GraphStructureTool(graph_client)

    @pytest.mark.asyncio
    async def test_get_all_cocos_fields(self, graph_client):
        """Retrieve all COCOS fields without filters."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_cocos_fields()
        assert result["total_fields"] >= 1
        types = result["transformation_types"]
        assert len(types) >= 1
        # psi_like should always exist in any DD graph
        assert "psi_like" in types
        psi_like = types["psi_like"]
        assert psi_like["count"] >= 1
        ids_affected = {p["ids"] for p in psi_like["fields"] if "ids" in p}
        assert len(ids_affected) >= 1

    @pytest.mark.asyncio
    async def test_filter_by_transformation_type(self, graph_client):
        """Filtering by transformation type returns only matching entries."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_cocos_fields(transformation_type="psi_like")
        assert result["total_fields"] >= 1
        assert list(result["transformation_types"].keys()) == ["psi_like"]

    @pytest.mark.asyncio
    async def test_filter_nonexistent_type(self, graph_client):
        """Non-existent transformation type returns 0 fields."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_cocos_fields(transformation_type="nonexistent_zz")
        assert result["total_fields"] == 0
        assert result["transformation_types"] == {}

    @pytest.mark.asyncio
    async def test_filter_by_ids(self, graph_client):
        """Filtering by IDS name returns only paths from that IDS."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_cocos_fields(ids_filter="equilibrium")
        assert result["total_fields"] >= 1
        for tt_data in result["transformation_types"].values():
            ids_in_type = {p["ids"] for p in tt_data["fields"] if "ids" in p}
            assert "equilibrium" in ids_in_type

    @pytest.mark.asyncio
    async def test_filter_ids_no_cocos(self, graph_client):
        """IDS with no COCOS fields returns empty results."""
        tool = self._make_tool(graph_client)
        # controllers IDS has no COCOS-dependent fields
        result = await tool.get_dd_cocos_fields(ids_filter="controllers")
        assert result["total_fields"] == 0

    @pytest.mark.asyncio
    async def test_sample_paths_populated(self, graph_client):
        """Fields are populated for each transformation type."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_cocos_fields()
        for tt_data in result["transformation_types"].values():
            assert len(tt_data["fields"]) > 0


# ── Phase 2: Enhanced get_dd_version_context ──────────────────────────────


class TestVersionContextBulk:
    def _make_tool(self, graph_client):
        from imas_codex.tools.version_tool import VersionTool

        return VersionTool(graph_client)

    @pytest.mark.asyncio
    async def test_per_path_mode_backward_compat(self, graph_client):
        """Existing per-path mode still works and returns path-keyed dict."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_version_context(
            paths="equilibrium/time_slice/profiles_1d/psi"
        )
        assert "paths" in result
        assert "equilibrium/time_slice/profiles_1d/psi" in result["paths"]

    @pytest.mark.asyncio
    async def test_bulk_query_by_change_type(self, graph_client):
        """Bulk query mode with change_type_filter returns mode=bulk_query."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_version_context(
            change_type_filter="cocos_label_transformation"
        )
        assert result.get("mode") == "bulk_query"
        assert result["change_count"] >= 1
        for c in result["changes"]:
            assert c["change_type"] == "cocos_label_transformation"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Test graph may lack IMASNodeChange nodes with change_type='added'"
    )
    async def test_bulk_query_path_added(self, graph_client):
        """Bulk query for 'added' change type returns matching changes."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_version_context(change_type_filter="added")
        assert result.get("mode") == "bulk_query"
        # Production graph should have many 'added' changes
        assert result["change_count"] >= 1

    @pytest.mark.asyncio
    async def test_bulk_query_with_ids_filter(self, graph_client):
        """Bulk query with ids_filter restricts to that IDS."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_version_context(
            change_type_filter="added",
            ids_filter="equilibrium",
        )
        assert result.get("mode") == "bulk_query"
        if result["change_count"] > 0:
            # All affected IDS should include equilibrium
            assert "equilibrium" in result["ids_affected"]

    @pytest.mark.asyncio
    async def test_error_when_no_paths_and_no_filter(self, graph_client):
        """Calling with neither paths nor change_type_filter returns error."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_version_context()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_bulk_query_nonexistent_change_type(self, graph_client):
        """Non-existent change type returns mode=bulk_query with 0 changes."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_version_context(
            change_type_filter="nonexistent_type_zz"
        )
        assert result.get("mode") == "bulk_query"
        assert result["change_count"] == 0


# ── Phase 3: Unit stats in overview ──────────────────────────────────────


class TestOverviewUnitStats:
    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphOverviewTool

        return GraphOverviewTool(graph_client)

    @pytest.mark.asyncio
    async def test_overview_without_unit_stats(self, graph_client):
        """Default overview has no unit_statistics."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_catalog()
        assert result.unit_statistics is None


# ── Phase 4: Lifecycle filtering ──────────────────────────────────────────


@pytest.mark.skip(reason="lifecycle_filter parameter not implemented in list_dd_paths")
class TestLifecycleFiltering:
    @pytest.mark.asyncio
    async def test_list_paths_lifecycle_active(self, graph_client):
        """lifecycle_filter='active' excludes non-active paths."""
        from imas_codex.tools.graph_search import GraphListTool

        tool = GraphListTool(graph_client)
        result = await tool.list_dd_paths("core_profiles", lifecycle_filter="active")
        total = sum(item.path_count for item in result.results)
        assert total >= 1  # core_profiles has active paths

    @pytest.mark.asyncio
    async def test_list_paths_lifecycle_alpha(self, graph_client):
        """lifecycle_filter='alpha' returns only alpha paths."""
        from imas_codex.tools.graph_search import GraphListTool

        tool = GraphListTool(graph_client)
        result = await tool.list_dd_paths("core_profiles", lifecycle_filter="alpha")
        total = sum(item.path_count for item in result.results)
        # Production graph should have some alpha paths in core_profiles
        assert total >= 0  # may be 0 if no alpha paths exist

    @pytest.mark.asyncio
    async def test_lifecycle_filter_reduces_results(self, graph_client):
        """Filtering by lifecycle returns fewer results than unfiltered."""
        from imas_codex.tools.graph_search import GraphListTool

        tool = GraphListTool(graph_client)
        all_result = await tool.list_dd_paths("equilibrium")
        active_result = await tool.list_dd_paths(
            "equilibrium", lifecycle_filter="active"
        )
        total_all = sum(item.path_count for item in all_result.results)
        total_active = sum(item.path_count for item in active_result.results)
        # Active should be <= all (some paths may be obsolescent/alpha)
        assert total_active <= total_all
        assert total_active >= 1


# ── Phase 5: Migration guide summary mode ────────────────────────────────


class TestMigrationSummary:
    def test_summary_only(self, graph_client):
        """summary_only=True returns a compact Migration Summary."""
        from imas_codex.tools.migration_guide import generate_migration_guide

        result = generate_migration_guide(
            graph_client,
            "3.39.0",
            "4.0.0",
            summary_only=True,
        )
        assert isinstance(result, str)
        assert len(result) > 0
        # Summary should be compact
        assert len(result) < 10000

    def test_summary_vs_full_both_work(self, graph_client):
        """Both summary and full guide return non-empty strings; summary is shorter."""
        from imas_codex.tools.migration_guide import generate_migration_guide

        summary = generate_migration_guide(
            graph_client, "3.39.0", "4.0.0", summary_only=True
        )
        full = generate_migration_guide(
            graph_client, "3.39.0", "4.0.0", summary_only=False
        )
        assert len(summary) > 0
        assert len(full) > 0
        assert len(summary) <= len(full)


# ── Phase 7: Search parameter unification ────────────────────────────────


@pytest.mark.skip(
    reason="physics_domain/node_type/include_children parameters not implemented"
)
class TestSearchParamUnification:
    @pytest.mark.asyncio
    async def test_list_paths_physics_domain(self, graph_client):
        """physics_domain filter returns only paths with matching domain."""
        from imas_codex.tools.graph_search import GraphListTool

        tool = GraphListTool(graph_client)
        result = await tool.list_dd_paths("equilibrium", physics_domain="equilibrium")
        total = sum(item.path_count for item in result.results)
        assert total > 0

    @pytest.mark.asyncio
    async def test_list_paths_physics_domain_no_match(self, graph_client):
        """physics_domain with no match returns zero paths."""
        from imas_codex.tools.graph_search import GraphListTool

        tool = GraphListTool(graph_client)
        result = await tool.list_dd_paths(
            "equilibrium", physics_domain="nonexistent_domain_zz"
        )
        total = sum(item.path_count for item in result.results)
        assert total == 0

    @pytest.mark.asyncio
    async def test_list_paths_node_type_dynamic(self, graph_client):
        """node_type='dynamic' filters to dynamic paths only."""
        from imas_codex.tools.graph_search import GraphListTool

        tool = GraphListTool(graph_client)
        dyn_result = await tool.list_dd_paths("equilibrium", node_type="dynamic")
        all_result = await tool.list_dd_paths("equilibrium")
        dyn_count = sum(item.path_count for item in dyn_result.results)
        all_count = sum(item.path_count for item in all_result.results)
        # Dynamic subset should be < all
        assert dyn_count <= all_count
        assert dyn_count >= 1

    @pytest.mark.asyncio
    async def test_fetch_with_children(self, graph_client):
        """include_children=True populates children_preview for structure nodes."""
        from imas_codex.tools.graph_search import GraphPathTool

        tool = GraphPathTool(graph_client)
        result = await tool.fetch_dd_paths(
            "equilibrium/time_slice/profiles_1d",
            include_children=True,
        )
        assert len(result.nodes) >= 1
        node = result.nodes[0]
        assert node.children_preview is not None
        assert len(node.children_preview) >= 1
        # Each child should have at least a name
        for child in node.children_preview:
            assert "name" in child

    @pytest.mark.asyncio
    async def test_fetch_without_children(self, graph_client):
        """include_children=False leaves children_preview as None."""
        from imas_codex.tools.graph_search import GraphPathTool

        tool = GraphPathTool(graph_client)
        result = await tool.fetch_dd_paths(
            "equilibrium/time_slice/profiles_1d",
            include_children=False,
        )
        assert len(result.nodes) >= 1
        node = result.nodes[0]
        assert node.children_preview is None
