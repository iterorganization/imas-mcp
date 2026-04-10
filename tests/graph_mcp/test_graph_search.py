"""Tests for graph-backed search tool implementations.

Tests verify that graph-backed tools correctly query Neo4j and return
properly structured result models, matching the same interfaces as
file-backed tools.

Skipped on production graphs (uses fixture-specific assertions).
"""

import pytest

from tests.graph_mcp.conftest import CLUSTERS, IDS_NODES, IMAS_PATHS

pytestmark = [pytest.mark.graph_mcp, pytest.mark.fixture_only]


# ── GraphPathTool tests ──────────────────────────────────────────────────


class TestGraphPathTool:
    """Tests for GraphPathTool (check + fetch)."""

    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphPathTool

        return GraphPathTool(graph_client)

    @pytest.mark.asyncio
    async def test_check_existing_path(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.check_dd_paths("equilibrium/time_slice/profiles_1d/psi")
        assert result.summary["found"] == 1
        assert result.results[0].exists is True
        assert result.results[0].data_type == "FLT_1D"

    @pytest.mark.asyncio
    async def test_check_nonexistent_path(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.check_dd_paths("equilibrium/nonexistent/path")
        assert result.summary["found"] == 0
        assert result.results[0].exists is False

    @pytest.mark.asyncio
    async def test_check_with_ids_prefix(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.check_dd_paths(
            "time_slice/profiles_1d/psi", ids="equilibrium"
        )
        assert result.summary["found"] == 1
        assert result.results[0].ids_name == "equilibrium"

    @pytest.mark.asyncio
    async def test_check_multiple_paths(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.check_dd_paths(
            "equilibrium/time_slice/profiles_1d/psi core_profiles/profiles_1d/electrons/temperature"
        )
        assert result.summary["total"] == 2
        assert result.summary["found"] == 2

    @pytest.mark.asyncio
    async def test_check_mixed_found_notfound(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.check_dd_paths(
            [
                "equilibrium/time_slice/profiles_1d/psi",
                "fake/nonexistent/path",
            ]
        )
        assert result.summary["found"] == 1
        assert result.summary["not_found"] == 1

    @pytest.mark.asyncio
    async def test_fetch_existing_path(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths("equilibrium/time_slice/profiles_1d/psi")
        assert len(result.nodes) == 1
        node = result.nodes[0]
        assert node.path == "equilibrium/time_slice/profiles_1d/psi"
        assert node.documentation == "Poloidal magnetic flux profile"
        assert node.data_type == "FLT_1D"

    @pytest.mark.asyncio
    async def test_fetch_with_cluster_labels(self, graph_client):
        """Paths in clusters should return cluster labels."""
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths("equilibrium/time_slice/boundary/psi")
        assert len(result.nodes) == 1
        node = result.nodes[0]
        # This path is in cluster_equilibrium_boundary
        assert node.cluster_labels is not None
        assert len(node.cluster_labels) > 0
        assert all(isinstance(label, str) for label in node.cluster_labels)

    @pytest.mark.asyncio
    async def test_fetch_nonexistent_path(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths("fake/nonexistent/thing")
        assert len(result.nodes) == 0
        assert len(result.not_found_paths) == 1

    @pytest.mark.asyncio
    async def test_fetch_multiple_paths(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths(
            [
                "equilibrium/time_slice/profiles_1d/psi",
                "core_profiles/profiles_1d/electrons/temperature",
            ]
        )
        assert len(result.nodes) == 2
        paths = {n.path for n in result.nodes}
        assert "equilibrium/time_slice/profiles_1d/psi" in paths
        assert "core_profiles/profiles_1d/electrons/temperature" in paths


# ── GraphListTool tests ──────────────────────────────────────────────────


class TestGraphListTool:
    """Tests for GraphListTool."""

    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphListTool

        return GraphListTool(graph_client)

    @pytest.mark.asyncio
    async def test_list_ids(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.list_dd_paths("equilibrium")
        assert len(result.results) == 1
        item = result.results[0]
        assert item.path_count > 0
        # All fixture equilibrium paths should be listed
        eq_count = sum(1 for p in IMAS_PATHS if p["ids_name"] == "equilibrium")
        assert item.path_count == eq_count

    @pytest.mark.asyncio
    async def test_list_nonexistent_ids(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.list_dd_paths("nonexistent_ids")
        assert result.results[0].error is not None

    @pytest.mark.asyncio
    async def test_list_multiple_ids(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.list_dd_paths("equilibrium core_profiles")
        assert len(result.results) == 2
        assert result.summary["total_paths"] > 0

    @pytest.mark.asyncio
    async def test_list_with_path_prefix(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.list_dd_paths("equilibrium/time_slice/boundary")
        assert len(result.results) == 1
        item = result.results[0]
        # Only boundary paths should be returned
        assert item.path_count > 0
        for p in item.paths:
            assert "boundary" in p


# ── GraphOverviewTool tests ──────────────────────────────────────────────


class TestGraphOverviewTool:
    """Tests for GraphOverviewTool."""

    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphOverviewTool

        return GraphOverviewTool(graph_client)

    @pytest.mark.asyncio
    async def test_overview_returns_all_ids(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_catalog()
        assert len(result.available_ids) == len(IDS_NODES)
        for ids in IDS_NODES:
            assert ids["name"] in result.available_ids

    @pytest.mark.asyncio
    async def test_overview_has_statistics(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_catalog()
        assert len(result.ids_statistics) > 0
        assert "equilibrium" in result.ids_statistics

    @pytest.mark.asyncio
    async def test_overview_has_dd_version(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_catalog()
        assert result.dd_version == "4.1.0"

    @pytest.mark.asyncio
    async def test_overview_has_mcp_tools(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_catalog()
        assert "search_dd_paths" in result.mcp_tools
        # query_imas_graph was removed in the unified server cleanup

    @pytest.mark.asyncio
    async def test_overview_has_physics_domains(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_catalog()
        assert len(result.physics_domains) > 0


# ── GraphClustersTool tests ──────────────────────────────────────────────


class TestGraphClustersTool:
    """Tests for GraphClustersTool (path-based lookup only, no embeddings)."""

    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphClustersTool

        return GraphClustersTool(graph_client)

    @pytest.mark.asyncio
    async def test_search_by_path(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.search_dd_clusters("equilibrium/time_slice/boundary/psi")
        assert result["query_type"] == "path"
        assert result["clusters_found"] >= 1
        # Should find cluster_equilibrium_boundary
        labels = [c["label"] for c in result["clusters"]]
        assert "Equilibrium Boundary" in labels

    @pytest.mark.asyncio
    async def test_search_by_path_not_in_cluster(self, graph_client):
        """Paths not in any cluster return 0 results."""
        tool = self._make_tool(graph_client)
        result = await tool.search_dd_clusters("equilibrium/time_slice/profiles_1d/psi")
        assert result["query_type"] == "path"
        assert result["clusters_found"] == 0

    @pytest.mark.asyncio
    async def test_cluster_paths_populated(self, graph_client):
        """Clusters should include member path IDs."""
        tool = self._make_tool(graph_client)
        result = await tool.search_dd_clusters(
            "core_profiles/profiles_1d/electrons/temperature"
        )
        assert result["clusters_found"] >= 1
        cluster = result["clusters"][0]
        assert len(cluster["paths"]) > 0

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.search_dd_clusters("")
        assert "error" in result


# ── GraphIdentifiersTool tests ───────────────────────────────────────────


class TestGraphIdentifiersTool:
    """Tests for GraphIdentifiersTool."""

    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphIdentifiersTool

        return GraphIdentifiersTool(graph_client)

    @pytest.mark.asyncio
    async def test_get_identifiers(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_identifiers()
        assert len(result.schemas) >= 1

    @pytest.mark.asyncio
    async def test_identifiers_have_options(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_identifiers()
        # Our fixture has boundary_type with 2 options
        schema = result.schemas[0]
        assert schema["option_count"] >= 1

    @pytest.mark.asyncio
    async def test_identifiers_with_query_filter(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_identifiers(query="boundary")
        assert len(result.schemas) >= 1

    @pytest.mark.asyncio
    async def test_identifiers_no_match(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_identifiers(query="zzz_nomatch_zzz")
        assert len(result.schemas) == 0


# ── Tools integration tests ─────────────────────────────────────────────


class TestToolsGraphMode:
    """Test Tools class in graph-native mode."""

    def _make_tools(self, graph_client):
        from imas_codex.tools import Tools

        return Tools(graph_client=graph_client)

    def test_graph_mode_registers_all_tools(self, graph_client):
        tools = self._make_tools(graph_client)
        names = tools.get_registered_tool_names()
        # Should have all graph-backed tools (query_imas_graph and get_dd_graph_schema were removed)
        expected = {
            "search_dd_paths",
            "check_dd_paths",
            "fetch_dd_paths",
            "list_dd_paths",
            "get_dd_catalog",
            "search_dd_clusters",
            "get_dd_identifiers",
            "get_dd_versions",
        }
        assert expected.issubset(set(names))

    def test_graph_mode_no_document_store(self, graph_client):
        """In graph mode, document_store should not be created."""
        tools = self._make_tools(graph_client)
        assert not hasattr(tools, "document_store")

    @pytest.mark.asyncio
    async def test_delegation_check_paths(self, graph_client):
        tools = self._make_tools(graph_client)
        result = await tools.path_tool.check_dd_paths(
            "equilibrium/time_slice/profiles_1d/psi"
        )
        assert result.summary["found"] == 1

    @pytest.mark.asyncio
    async def test_delegation_fetch_paths(self, graph_client):
        tools = self._make_tools(graph_client)
        result = await tools.path_tool.fetch_dd_paths(
            "equilibrium/time_slice/profiles_1d/psi"
        )
        assert len(result.nodes) == 1

    @pytest.mark.asyncio
    async def test_delegation_list_paths(self, graph_client):
        tools = self._make_tools(graph_client)
        result = await tools.list_tool.list_dd_paths("equilibrium")
        assert result.results[0].path_count > 0

    @pytest.mark.asyncio
    async def test_delegation_overview(self, graph_client):
        tools = self._make_tools(graph_client)
        result = await tools.overview_tool.get_dd_catalog()
        assert len(result.available_ids) > 0

    @pytest.mark.asyncio
    async def test_delegation_identifiers(self, graph_client):
        tools = self._make_tools(graph_client)
        result = await tools.identifiers_tool.get_dd_identifiers()
        assert len(result.schemas) >= 1


# ── T1: fetch_dd_paths enrichment tests ────────────────────────────────


class TestFetchImasPathsEnrichment:
    """Tests for fetch_dd_paths with identifier schemas and version history."""

    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphPathTool

        return GraphPathTool(graph_client)

    @pytest.mark.asyncio
    async def test_fetch_includes_identifier_schema(self, graph_client):
        """Path with IdentifierSchema should populate identifier_schema."""
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths("equilibrium/time_slice/boundary/type")
        assert len(result.nodes) == 1
        node = result.nodes[0]
        assert node.identifier_schema is not None
        assert node.identifier_schema.schema_path == "boundary_type"

    @pytest.mark.asyncio
    async def test_fetch_no_identifier_schema(self, graph_client):
        """Path without IdentifierSchema should have None."""
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths("equilibrium/time_slice/profiles_1d/psi")
        node = result.nodes[0]
        assert node.identifier_schema is None

    @pytest.mark.asyncio
    async def test_fetch_version_history_enabled(self, graph_client):
        """include_version_history=True should populate version_changes."""
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths(
            "core_profiles/profiles_1d/electrons/pressure",
            include_version_history=True,
        )
        assert len(result.nodes) == 1
        node = result.nodes[0]
        assert node.version_changes is not None
        assert len(node.version_changes) >= 1
        change = node.version_changes[0]
        assert "version" in change
        assert "type" in change

    @pytest.mark.asyncio
    async def test_fetch_version_history_disabled(self, graph_client):
        """include_version_history=False should leave version_changes as None."""
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths(
            "core_profiles/profiles_1d/electrons/pressure",
            include_version_history=False,
        )
        node = result.nodes[0]
        assert node.version_changes is None

    @pytest.mark.asyncio
    async def test_fetch_version_history_no_changes(self, graph_client):
        """Path with no IMASNodeChange should return None even with flag."""
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths(
            "core_profiles/profiles_1d/ion/temperature",
            include_version_history=True,
        )
        node = result.nodes[0]
        # No changes attached to this path, so should be None
        assert node.version_changes is None


# ── T2: search_dd_clusters listing mode tests ──────────────────────────


class TestSearchImaClustersListingMode:
    """Tests for search_dd_clusters with IDS listing mode."""

    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphClustersTool

        return GraphClustersTool(graph_client)

    @pytest.mark.asyncio
    async def test_list_clusters_by_ids(self, graph_client):
        """No query + ids_filter should list all clusters for that IDS."""
        tool = self._make_tool(graph_client)
        result = await tool.search_dd_clusters(
            ids_filter="equilibrium",
        )
        assert result["query_type"] == "ids_listing"
        assert result["clusters_found"] >= 1
        labels = [c["label"] for c in result["clusters"]]
        assert "Equilibrium Boundary" in labels

    @pytest.mark.asyncio
    async def test_list_clusters_section_only(self, graph_client):
        """section_only=True should filter to IDS-scoped clusters."""
        tool = self._make_tool(graph_client)
        result = await tool.search_dd_clusters(
            ids_filter="equilibrium",
            section_only=True,
        )
        assert result["query_type"] == "ids_listing"
        assert result["section_only"] is True
        # All returned clusters should have IDS-level paths
        for c in result["clusters"]:
            assert any("/" in p for p in c["paths"])

    @pytest.mark.asyncio
    async def test_list_clusters_no_ids_no_query(self, graph_client):
        """No query and no ids_filter returns error."""
        tool = self._make_tool(graph_client)
        result = await tool.search_dd_clusters()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_clusters_includes_paths(self, graph_client):
        """Listed clusters should include member paths."""
        tool = self._make_tool(graph_client)
        result = await tool.search_dd_clusters(
            ids_filter="core_profiles",
        )
        assert result["clusters_found"] >= 1
        cluster = result["clusters"][0]
        assert len(cluster["paths"]) > 0


# ── T3: get_dd_version_context tests ─────────────────────────────────────


class TestGetDDVersionContext:
    """Tests for VersionTool.get_dd_version_context."""

    @pytest.fixture
    def version_tool(self, graph_client):
        from imas_codex.tools.version_tool import VersionTool

        return VersionTool(graph_client)

    @pytest.mark.anyio
    async def test_path_with_changes(self, version_tool):
        """Path with IMASNodeChange should return change data."""
        result = await version_tool.get_dd_version_context(
            "core_profiles/profiles_1d/electrons/pressure"
        )
        assert result["total_paths"] == 1
        path_data = result["paths"]["core_profiles/profiles_1d/electrons/pressure"]
        assert path_data["change_count"] >= 1
        assert len(path_data["changes"]) >= 1

    @pytest.mark.anyio
    async def test_path_without_changes(self, version_tool):
        """Path with no changes should have 0 change_count."""
        result = await version_tool.get_dd_version_context(
            "core_profiles/profiles_1d/ion/temperature"
        )
        path_data = result["paths"]["core_profiles/profiles_1d/ion/temperature"]
        assert path_data["change_count"] == 0
        assert path_data["changes"] == []

    @pytest.mark.anyio
    async def test_multiple_paths(self, version_tool):
        result = await version_tool.get_dd_version_context(
            [
                "core_profiles/profiles_1d/electrons/pressure",
                "core_profiles/profiles_1d/ion/temperature",
            ]
        )
        assert result["total_paths"] == 2
        assert result["paths_with_changes"] == 1
        assert (
            "core_profiles/profiles_1d/ion/temperature"
            in result["paths_without_changes"]
        )

    @pytest.mark.anyio
    async def test_nonexistent_path(self, version_tool):
        result = await version_tool.get_dd_version_context("fake/path")
        assert result["total_paths"] == 1
        assert "fake/path" in result["not_found"]

    @pytest.mark.anyio
    async def test_version_context_reports_change_diagnostics(self, version_tool):
        result = await version_tool.get_dd_version_context(
            "core_profiles/profiles_1d/electrons/pressure"
        )
        assert result["paths_found"] == ["core_profiles/profiles_1d/electrons/pressure"]
        assert result["graph_change_nodes_seen"] >= 1

    @pytest.mark.anyio
    async def test_empty_paths(self, version_tool):
        result = await version_tool.get_dd_version_context("")
        assert "error" in result


# ── _common_path_prefix tests ────────────────────────────────────────────


class TestCommonPathPrefix:
    """Tests for the _common_path_prefix helper."""

    def test_common_prefix(self):
        from imas_codex.tools.graph_search import _common_path_prefix

        result = _common_path_prefix(
            [
                "equilibrium/time_slice/boundary/psi",
                "equilibrium/time_slice/boundary/psi_norm",
                "equilibrium/time_slice/boundary/type",
            ]
        )
        assert result == "equilibrium/time_slice/boundary"

    def test_no_common_prefix(self):
        from imas_codex.tools.graph_search import _common_path_prefix

        result = _common_path_prefix(
            [
                "equilibrium/time_slice/profiles_1d/psi",
                "core_profiles/profiles_1d/electrons/temperature",
            ]
        )
        assert result == ""

    def test_single_path(self):
        from imas_codex.tools.graph_search import _common_path_prefix

        result = _common_path_prefix(["equilibrium/time_slice/profiles_1d/psi"])
        assert result == "equilibrium/time_slice/profiles_1d/psi"

    def test_empty_list(self):
        from imas_codex.tools.graph_search import _common_path_prefix

        result = _common_path_prefix([])
        assert result == ""


# ── Cluster path lookup tests ────────────────────────────────────────────


class TestClusterPathLookup:
    """Tests for cluster search by specific path."""

    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphClustersTool

        return GraphClustersTool(graph_client)

    @pytest.mark.asyncio
    async def test_path_in_cluster(self, graph_client):
        """Path that is a cluster member should return its cluster."""
        tool = self._make_tool(graph_client)
        result = await tool.search_dd_clusters(
            query="equilibrium/time_slice/boundary/psi"
        )
        assert result["query_type"] == "path"
        assert result["clusters_found"] >= 1
        labels = [c["label"] for c in result["clusters"]]
        assert "Equilibrium Boundary" in labels

    @pytest.mark.asyncio
    async def test_path_not_in_cluster(self, graph_client):
        """Path not in any cluster should return 0 clusters."""
        tool = self._make_tool(graph_client)
        result = await tool.search_dd_clusters(
            query="equilibrium/time_slice/profiles_1d/psi"
        )
        assert result["query_type"] == "path"
        assert result["clusters_found"] == 0


# ── Domain resolution tests ──────────────────────────────────────────────


class TestResolveDomain:
    """Tests for _resolve_physics_domain helper."""

    def test_exact_match(self, graph_client):
        from imas_codex.tools.graph_search import _resolve_physics_domain

        domains, method = _resolve_physics_domain(graph_client, "transport")
        assert domains == ["transport"]
        assert method == "exact"

    def test_ids_name_resolution(self, graph_client):
        """IDS name 'equilibrium' resolves to its physics_domain."""
        from imas_codex.tools.graph_search import _resolve_physics_domain

        domains, method = _resolve_physics_domain(graph_client, "equilibrium")
        # In fixture, equilibrium IDS has physics_domain='magnetics'
        # But 'equilibrium' is also a valid PhysicsDomain enum value,
        # so it matches exact first
        assert "equilibrium" in domains
        assert method == "exact"

    def test_ids_name_not_in_enum(self, graph_client):
        """IDS name that isn't a PhysicsDomain resolves via graph."""
        from imas_codex.tools.graph_search import _resolve_physics_domain

        # core_profiles is not a PhysicsDomain enum value
        domains, method = _resolve_physics_domain(graph_client, "core_profiles")
        assert domains == ["transport"]
        assert method == "ids_name:core_profiles"

    def test_substring_match(self, graph_client):
        from imas_codex.tools.graph_search import _resolve_physics_domain

        domains, method = _resolve_physics_domain(graph_client, "auxiliary")
        assert "auxiliary_heating" in domains
        assert method == "substring:auxiliary"

    def test_no_match(self, graph_client):
        from imas_codex.tools.graph_search import _resolve_physics_domain

        domains, method = _resolve_physics_domain(
            graph_client, "nonexistent_domain_xyz"
        )
        assert domains == []
        assert method == "no_match"

    def test_category_expansion(self, graph_client):
        from imas_codex.tools.graph_search import _resolve_physics_domain

        domains, method = _resolve_physics_domain(graph_client, "diagnostics")
        assert method == "category:diagnostics"
        assert "magnetic_field_diagnostics" in domains
        assert "radiation_measurement_diagnostics" in domains


# ── Domain export tests ──────────────────────────────────────────────────


class TestExportDomain:
    """Tests for export_dd_domain with domain resolution."""

    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphStructureTool

        return GraphStructureTool(graph_client)

    @pytest.mark.asyncio
    async def test_export_exact_domain(self, graph_client):
        """Exact domain name should return paths."""
        tool = self._make_tool(graph_client)
        # 'equilibrium' is stored on equilibrium paths in fixtures
        result = await tool.export_dd_domain(domain="equilibrium")
        assert result["total_paths"] > 0
        assert "equilibrium" in result["resolved_domains"]

    @pytest.mark.asyncio
    async def test_export_ids_name(self, graph_client):
        """IDS name should resolve and export domain paths."""
        tool = self._make_tool(graph_client)
        result = await tool.export_dd_domain(domain="core_profiles")
        assert result["total_paths"] > 0
        assert result["resolution"] == "ids_name:core_profiles"
        assert "transport" in result["resolved_domains"]

    @pytest.mark.asyncio
    async def test_export_no_match(self, graph_client):
        """No-match domain should return error."""
        tool = self._make_tool(graph_client)
        result = await tool.export_dd_domain(domain="nonexistent_xyz")
        assert result["total_paths"] == 0
        assert "error" in result


# ── Phase 4: Fetch metadata parity ──────────────────────────────────────


class TestFetchMetadataParity:
    """Verify fetch_dd_paths returns introduced_after_version and version_changes."""

    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphPathTool

        return GraphPathTool(graph_client)

    @pytest.mark.asyncio
    async def test_introduced_after_version_populated(self, graph_client):
        """Paths with INTRODUCED_IN should expose introduced_after_version."""
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths("equilibrium/time_slice/profiles_1d/psi")
        node = result.nodes[0]
        assert node.introduced_after_version == "3.42.0"

    @pytest.mark.asyncio
    async def test_introduced_version_newer_path(self, graph_client):
        """A path introduced in 4.0.0 should reflect that version."""
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths(
            "core_profiles/profiles_1d/electrons/pressure"
        )
        node = result.nodes[0]
        assert node.introduced_after_version == "4.0.0"

    @pytest.mark.asyncio
    async def test_version_changes_included(self, graph_client):
        """include_version_history=True should return version_changes."""
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths(
            "core_profiles/profiles_1d/electrons/pressure",
            include_version_history=True,
        )
        node = result.nodes[0]
        assert node.version_changes is not None
        assert len(node.version_changes) > 0
        change = node.version_changes[0]
        assert change["version"] == "4.0.0"

    @pytest.mark.asyncio
    async def test_version_changes_empty_without_flag(self, graph_client):
        """Without include_version_history, version_changes should be None."""
        tool = self._make_tool(graph_client)
        result = await tool.fetch_dd_paths(
            "core_profiles/profiles_1d/electrons/pressure",
            include_version_history=False,
        )
        node = result.nodes[0]
        assert node.version_changes is None


# ── Phase 5: Identifier search with enrichment ──────────────────────────


class TestIdentifierSearch:
    """Verify identifier search uses enriched descriptions and keywords."""

    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphIdentifiersTool

        return GraphIdentifiersTool(graph_client)

    @pytest.mark.asyncio
    async def test_list_all_identifiers(self, graph_client):
        """No query should return all schemas."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_identifiers()
        assert result.analytics["total_schemas"] >= 1
        schema = result.schemas[0]
        assert schema["path"] == "boundary_type"

    @pytest.mark.asyncio
    async def test_keyword_match_description(self, graph_client):
        """Query matching LLM description should return schema."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_identifiers(query="topology")
        assert result.analytics["total_schemas"] >= 1
        names = [s["path"] for s in result.schemas]
        assert "boundary_type" in names

    @pytest.mark.asyncio
    async def test_keyword_match_keywords_field(self, graph_client):
        """Query matching keywords list should return schema."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_identifiers(query="separatrix")
        assert result.analytics["total_schemas"] >= 1
        names = [s["path"] for s in result.schemas]
        assert "boundary_type" in names

    @pytest.mark.asyncio
    async def test_description_preferred(self, graph_client):
        """Description should contain LLM-enriched content when available."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_identifiers()
        schema = result.schemas[0]
        assert "topology" in schema["description"]

    @pytest.mark.asyncio
    async def test_no_match(self, graph_client):
        """Query with no match should return empty."""
        tool = self._make_tool(graph_client)
        result = await tool.get_dd_identifiers(query="zzz_nonexistent_xyz")
        assert result.analytics["total_schemas"] == 0
