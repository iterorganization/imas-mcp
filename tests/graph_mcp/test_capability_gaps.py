"""Functional tests for capability-gap additions to the DD MCP server.

Covers:
  - get_cocos_fields (GraphStructureTool)
  - Enhanced get_dd_version_context bulk-query mode (VersionTool)
  - Unit statistics in overview (GraphOverviewTool)
  - Lifecycle filtering in list/analyze (GraphListTool / GraphStructureTool)
  - Migration guide summary_only mode (generate_migration_guide)
  - Search parameter unification – physics_domain, node_type, include_children
    (GraphListTool / GraphPathTool)

Tests require a live Neo4j connection loaded with the session fixture graph.
All tests are auto-skipped when Neo4j is unavailable via conftest's
``pytest_collection_modifyitems`` hook.
"""

import pytest

pytestmark = pytest.mark.graph_mcp


# ── Phase 1: get_cocos_fields ─────────────────────────────────────────────


class TestCocosFields:
    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphStructureTool

        return GraphStructureTool(graph_client)

    @pytest.mark.asyncio
    async def test_get_all_cocos_fields(self, graph_client):
        """Retrieve all COCOS fields without filters."""
        tool = self._make_tool(graph_client)
        result = await tool.get_cocos_fields()
        assert result["total_fields"] == 3  # 3 psi_like paths in fixture
        assert result["transformation_type_count"] >= 1
        types = result["transformation_types"]
        assert len(types) >= 1
        psi_like = next(t for t in types if t["type"] == "psi_like")
        assert psi_like["field_count"] == 3
        assert "equilibrium" in psi_like["ids_affected"]

    @pytest.mark.asyncio
    async def test_filter_by_transformation_type(self, graph_client):
        """Filtering by transformation type returns only matching entries."""
        tool = self._make_tool(graph_client)
        result = await tool.get_cocos_fields(transformation_type="psi_like")
        assert result["total_fields"] == 3
        for t in result["transformation_types"]:
            assert t["type"] == "psi_like"

    @pytest.mark.asyncio
    async def test_filter_nonexistent_type(self, graph_client):
        """Non-existent transformation type returns 0 fields."""
        tool = self._make_tool(graph_client)
        result = await tool.get_cocos_fields(transformation_type="nonexistent")
        assert result["total_fields"] == 0
        assert result["transformation_types"] == []

    @pytest.mark.asyncio
    async def test_filter_by_ids(self, graph_client):
        """Filtering by IDS name returns only paths from that IDS."""
        tool = self._make_tool(graph_client)
        result = await tool.get_cocos_fields(ids_filter="equilibrium")
        assert result["total_fields"] == 3  # all COCOS paths are in equilibrium
        for t in result["transformation_types"]:
            assert "equilibrium" in t["ids_affected"]

    @pytest.mark.asyncio
    async def test_filter_ids_no_cocos(self, graph_client):
        """IDS with no COCOS fields returns empty results."""
        tool = self._make_tool(graph_client)
        result = await tool.get_cocos_fields(ids_filter="core_profiles")
        assert result["total_fields"] == 0

    @pytest.mark.asyncio
    async def test_sample_paths_populated(self, graph_client):
        """sample_paths are populated and capped at 10."""
        tool = self._make_tool(graph_client)
        result = await tool.get_cocos_fields()
        for t in result["transformation_types"]:
            assert len(t["sample_paths"]) > 0
            assert len(t["sample_paths"]) <= 10  # max 10 samples


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
    async def test_bulk_query_added(self, graph_client):
        """Bulk query for 'added' change type returns matching changes."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_version_context(change_type_filter="added")
        assert result.get("mode") == "bulk_query"
        assert result["change_count"] >= 1

    @pytest.mark.asyncio
    async def test_bulk_query_with_version_range(self, graph_client):
        """Bulk query with version range includes only changes within range."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_version_context(
            change_type_filter="cocos_label_transformation",
            from_version="3.42.0",
            to_version="4.0.0",
        )
        assert result.get("mode") == "bulk_query"
        for c in result["changes"]:
            assert c["version"] <= "4.0.0"

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
            change_type_filter="nonexistent_type"
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
        result = await tool.get_imas_overview()
        assert result.unit_statistics is None

    @pytest.mark.asyncio
    async def test_overview_with_unit_stats(self, graph_client):
        """Overview with include_unit_stats=True returns unit distribution."""
        tool = self._make_tool(graph_client)
        result = await tool.get_imas_overview(include_unit_stats=True)
        assert result.unit_statistics is not None
        assert "top_units" in result.unit_statistics
        assert len(result.unit_statistics["top_units"]) > 0
        # Check at least one known unit from fixture is present
        unit_names = [u["unit"] for u in result.unit_statistics["top_units"]]
        assert any(u in unit_names for u in ("Pa", "eV", "T.m^2", "m^-3", "m"))


# ── Phase 4: Lifecycle filtering ──────────────────────────────────────────


class TestLifecycleFiltering:
    @pytest.mark.asyncio
    async def test_list_paths_lifecycle_active(self, graph_client):
        """lifecycle_filter='active' excludes alpha paths."""
        from imas_codex.tools.graph_search import GraphListTool

        tool = GraphListTool(graph_client)
        result = await tool.list_imas_paths("core_profiles", lifecycle_filter="active")
        all_paths = [path for item in result.results for path in item.paths]
        # The alpha pressure path must not appear
        assert not any("pressure" in p for p in all_paths)

    @pytest.mark.asyncio
    async def test_list_paths_lifecycle_alpha(self, graph_client):
        """lifecycle_filter='alpha' returns only alpha paths."""
        from imas_codex.tools.graph_search import GraphListTool

        tool = GraphListTool(graph_client)
        result = await tool.list_imas_paths("core_profiles", lifecycle_filter="alpha")
        all_paths = [path for item in result.results for path in item.paths]
        # Must include at least one path and all paths should be the pressure path
        assert len(all_paths) >= 1
        assert all("pressure" in p for p in all_paths)

    @pytest.mark.asyncio
    async def test_analyze_structure_has_lifecycle(self, graph_client):
        """analyze_imas_structure includes lifecycle_distribution."""
        from imas_codex.tools.graph_search import GraphStructureTool

        tool = GraphStructureTool(graph_client)
        result = await tool.analyze_imas_structure("equilibrium")
        assert "lifecycle_distribution" in result
        statuses = {ld["status"] for ld in result["lifecycle_distribution"]}
        assert "active" in statuses


# ── Phase 5: Migration guide summary mode ────────────────────────────────


class TestMigrationSummary:
    def test_summary_only(self, graph_client):
        """summary_only=True returns a compact Migration Summary."""
        from imas_codex.tools.migration_guide import generate_migration_guide

        result = generate_migration_guide(
            graph_client,
            "3.42.0",
            "4.0.0",
            summary_only=True,
        )
        assert "Migration Summary" in result
        assert "Total changes" in result or "total_changes" in result.lower()
        # Summary should be compact
        assert len(result) < 5000

    def test_summary_contains_change_types(self, graph_client):
        """Summary includes change-type breakdown."""
        from imas_codex.tools.migration_guide import generate_migration_guide

        result = generate_migration_guide(
            graph_client,
            "3.42.0",
            "4.0.0",
            summary_only=True,
        )
        # Fixture has 'added' and 'cocos_label_transformation' changes
        assert "Change Type" in result or "change_type" in result.lower()

    def test_summary_vs_full_both_work(self, graph_client):
        """Both summary and full guide return non-empty strings; summary is shorter."""
        from imas_codex.tools.migration_guide import generate_migration_guide

        summary = generate_migration_guide(
            graph_client, "3.42.0", "4.0.0", summary_only=True
        )
        full = generate_migration_guide(
            graph_client, "3.42.0", "4.0.0", summary_only=False
        )
        assert len(summary) > 0
        assert len(full) > 0
        assert len(summary) < len(full)


# ── Phase 7: Search parameter unification ────────────────────────────────


class TestSearchParamUnification:
    @pytest.mark.asyncio
    async def test_list_paths_physics_domain(self, graph_client):
        """physics_domain filter returns only paths with matching domain."""
        from imas_codex.tools.graph_search import GraphListTool

        tool = GraphListTool(graph_client)
        result = await tool.list_imas_paths("equilibrium", physics_domain="equilibrium")
        total = sum(item.path_count for item in result.results)
        assert total > 0  # equilibrium IDS has equilibrium-domain paths

    @pytest.mark.asyncio
    async def test_list_paths_physics_domain_no_match(self, graph_client):
        """physics_domain with no match returns zero paths."""
        from imas_codex.tools.graph_search import GraphListTool

        tool = GraphListTool(graph_client)
        result = await tool.list_imas_paths("equilibrium", physics_domain="transport")
        total = sum(item.path_count for item in result.results)
        assert total == 0  # equilibrium IDS has no transport paths

    @pytest.mark.asyncio
    async def test_list_paths_node_type_leaf(self, graph_client):
        """node_type='leaf' excludes structure nodes."""
        from imas_codex.tools.graph_search import GraphListTool

        tool = GraphListTool(graph_client)
        result = await tool.list_imas_paths("equilibrium", node_type="leaf")
        all_paths = [path for item in result.results for path in item.paths]
        # The profiles_1d structure node must not appear as a leaf result
        assert not any(p.endswith("profiles_1d") for p in all_paths)

    @pytest.mark.asyncio
    async def test_fetch_with_children(self, graph_client):
        """include_children=True populates children_preview for structure nodes."""
        from imas_codex.tools.graph_search import GraphPathTool

        tool = GraphPathTool(graph_client)
        result = await tool.fetch_imas_paths(
            "equilibrium/time_slice/profiles_1d",
            include_children=True,
        )
        assert len(result.nodes) == 1
        node = result.nodes[0]
        assert node.children_preview is not None
        assert len(node.children_preview) >= 2  # psi and pressure
        child_names = [c["name"] for c in node.children_preview]
        assert "psi" in child_names

    @pytest.mark.asyncio
    async def test_fetch_without_children(self, graph_client):
        """include_children=False leaves children_preview as None."""
        from imas_codex.tools.graph_search import GraphPathTool

        tool = GraphPathTool(graph_client)
        result = await tool.fetch_imas_paths(
            "equilibrium/time_slice/profiles_1d",
            include_children=False,
        )
        assert len(result.nodes) == 1
        node = result.nodes[0]
        assert node.children_preview is None
