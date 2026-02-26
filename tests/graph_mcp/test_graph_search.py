"""Tests for graph-backed search tool implementations.

Tests verify that graph-backed tools correctly query Neo4j and return
properly structured result models, matching the same interfaces as
file-backed tools.
"""

import pytest

from tests.graph_mcp.conftest import CLUSTERS, IDS_NODES, IMAS_PATHS

pytestmark = pytest.mark.graph_mcp


# ── GraphPathTool tests ──────────────────────────────────────────────────


class TestGraphPathTool:
    """Tests for GraphPathTool (check + fetch)."""

    def _make_tool(self, graph_client):
        from imas_codex.tools.graph_search import GraphPathTool

        return GraphPathTool(graph_client)

    @pytest.mark.asyncio
    async def test_check_existing_path(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.check_imas_paths("equilibrium/time_slice/profiles_1d/psi")
        assert result.summary["found"] == 1
        assert result.results[0].exists is True
        assert result.results[0].data_type == "FLT_1D"

    @pytest.mark.asyncio
    async def test_check_nonexistent_path(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.check_imas_paths("equilibrium/nonexistent/path")
        assert result.summary["found"] == 0
        assert result.results[0].exists is False

    @pytest.mark.asyncio
    async def test_check_with_ids_prefix(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.check_imas_paths(
            "time_slice/profiles_1d/psi", ids="equilibrium"
        )
        assert result.summary["found"] == 1
        assert result.results[0].ids_name == "equilibrium"

    @pytest.mark.asyncio
    async def test_check_multiple_paths(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.check_imas_paths(
            "equilibrium/time_slice/profiles_1d/psi core_profiles/profiles_1d/electrons/temperature"
        )
        assert result.summary["total"] == 2
        assert result.summary["found"] == 2

    @pytest.mark.asyncio
    async def test_check_mixed_found_notfound(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.check_imas_paths(
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
        result = await tool.fetch_imas_paths("equilibrium/time_slice/profiles_1d/psi")
        assert len(result.nodes) == 1
        node = result.nodes[0]
        assert node.path == "equilibrium/time_slice/profiles_1d/psi"
        assert node.documentation == "Poloidal magnetic flux profile"
        assert node.data_type == "FLT_1D"

    @pytest.mark.asyncio
    async def test_fetch_with_cluster_labels(self, graph_client):
        """Paths in clusters should return cluster labels."""
        tool = self._make_tool(graph_client)
        result = await tool.fetch_imas_paths("equilibrium/time_slice/boundary/psi")
        assert len(result.nodes) == 1
        node = result.nodes[0]
        # This path is in cluster_equilibrium_boundary
        assert node.cluster_labels is not None
        assert len(node.cluster_labels) > 0

    @pytest.mark.asyncio
    async def test_fetch_nonexistent_path(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.fetch_imas_paths("fake/nonexistent/thing")
        assert len(result.nodes) == 0
        assert len(result.not_found_paths) == 1

    @pytest.mark.asyncio
    async def test_fetch_multiple_paths(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.fetch_imas_paths(
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
        result = await tool.list_imas_paths("equilibrium")
        assert len(result.results) == 1
        item = result.results[0]
        assert item.path_count > 0
        # All fixture equilibrium paths should be listed
        eq_count = sum(1 for p in IMAS_PATHS if p["ids_name"] == "equilibrium")
        assert item.path_count == eq_count

    @pytest.mark.asyncio
    async def test_list_nonexistent_ids(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.list_imas_paths("nonexistent_ids")
        assert result.results[0].error is not None

    @pytest.mark.asyncio
    async def test_list_multiple_ids(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.list_imas_paths("equilibrium core_profiles")
        assert len(result.results) == 2
        assert result.summary["total_paths"] > 0

    @pytest.mark.asyncio
    async def test_list_with_path_prefix(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.list_imas_paths("equilibrium/time_slice/boundary")
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
        result = await tool.get_imas_overview()
        assert len(result.available_ids) == len(IDS_NODES)
        for ids in IDS_NODES:
            assert ids["name"] in result.available_ids

    @pytest.mark.asyncio
    async def test_overview_has_statistics(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_imas_overview()
        assert len(result.ids_statistics) > 0
        assert "equilibrium" in result.ids_statistics

    @pytest.mark.asyncio
    async def test_overview_with_query_filter(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_imas_overview(query="equilibrium")
        assert "equilibrium" in result.available_ids
        # core_profiles should be filtered out
        assert "core_profiles" not in result.available_ids

    @pytest.mark.asyncio
    async def test_overview_has_dd_version(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_imas_overview()
        assert result.dd_version == "4.1.0"

    @pytest.mark.asyncio
    async def test_overview_has_mcp_tools(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_imas_overview()
        assert "search_imas_paths" in result.mcp_tools
        assert "query_imas_graph" in result.mcp_tools

    @pytest.mark.asyncio
    async def test_overview_has_physics_domains(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_imas_overview()
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
        result = await tool.search_imas_clusters("equilibrium/time_slice/boundary/psi")
        assert result["query_type"] == "path"
        assert result["clusters_found"] >= 1
        # Should find cluster_equilibrium_boundary
        labels = [c["label"] for c in result["clusters"]]
        assert "Equilibrium Boundary" in labels

    @pytest.mark.asyncio
    async def test_search_by_path_not_in_cluster(self, graph_client):
        """Paths not in any cluster return 0 results."""
        tool = self._make_tool(graph_client)
        result = await tool.search_imas_clusters(
            "equilibrium/time_slice/profiles_1d/psi"
        )
        assert result["query_type"] == "path"
        assert result["clusters_found"] == 0

    @pytest.mark.asyncio
    async def test_cluster_paths_populated(self, graph_client):
        """Clusters should include member path IDs."""
        tool = self._make_tool(graph_client)
        result = await tool.search_imas_clusters(
            "core_profiles/profiles_1d/electrons/temperature"
        )
        assert result["clusters_found"] >= 1
        cluster = result["clusters"][0]
        assert len(cluster["paths"]) > 0

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.search_imas_clusters("")
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
        result = await tool.get_imas_identifiers()
        assert len(result.schemas) >= 1

    @pytest.mark.asyncio
    async def test_identifiers_have_options(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_imas_identifiers()
        # Our fixture has boundary_type with 2 options
        schema = result.schemas[0]
        assert schema["option_count"] >= 1

    @pytest.mark.asyncio
    async def test_identifiers_with_query_filter(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_imas_identifiers(query="boundary")
        assert len(result.schemas) >= 1

    @pytest.mark.asyncio
    async def test_identifiers_no_match(self, graph_client):
        tool = self._make_tool(graph_client)
        result = await tool.get_imas_identifiers(query="zzz_nomatch_zzz")
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
        # Should have all 10 tools (6 graph-backed + cypher + schema + version)
        expected = {
            "search_imas_paths",
            "check_imas_paths",
            "fetch_imas_paths",
            "list_imas_paths",
            "get_imas_overview",
            "search_imas_clusters",
            "get_imas_identifiers",
            "query_imas_graph",
            "get_dd_graph_schema",
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
        result = await tools.check_imas_paths("equilibrium/time_slice/profiles_1d/psi")
        assert result.summary["found"] == 1

    @pytest.mark.asyncio
    async def test_delegation_fetch_paths(self, graph_client):
        tools = self._make_tools(graph_client)
        result = await tools.fetch_imas_paths("equilibrium/time_slice/profiles_1d/psi")
        assert len(result.nodes) == 1

    @pytest.mark.asyncio
    async def test_delegation_list_paths(self, graph_client):
        tools = self._make_tools(graph_client)
        result = await tools.list_imas_paths("equilibrium")
        assert result.results[0].path_count > 0

    @pytest.mark.asyncio
    async def test_delegation_overview(self, graph_client):
        tools = self._make_tools(graph_client)
        result = await tools.get_imas_overview()
        assert len(result.available_ids) > 0

    @pytest.mark.asyncio
    async def test_delegation_identifiers(self, graph_client):
        tools = self._make_tools(graph_client)
        result = await tools.get_imas_identifiers()
        assert len(result.schemas) >= 1
