"""Integration tests for mapping context enrichment against the live graph.

These tests query the real production graph (via tunnel) to validate the
enrichment tools (T1-T3) against actual DD data. They skip if Neo4j is
not reachable.

Connection auto-routes via GraphClient() — no NEO4J_URI override needed.
When running off-site, the SSH tunnel is established automatically by
resolve_neo4j() (port 17687 for the 'iter' location).
"""

import pytest


def _make_graph_client():
    """Create a GraphClient using default auto-routing resolution."""
    from imas_codex.graph.client import GraphClient

    return GraphClient()


@pytest.fixture(scope="module")
def live_client():
    """Module-scoped GraphClient for live graph integration tests."""
    try:
        gc = _make_graph_client()
        gc.get_stats()
    except Exception as e:
        pytest.skip(f"Live graph not available: {e}")
    yield gc
    gc.close()


# ── T1: fetch_dd_paths enrichment ──────────────────────────────────────


class TestFetchEnrichmentLive:
    """Test fetch_dd_paths identifier schemas + version history on live data."""

    def _make_tool(self, client):
        from imas_codex.tools.graph_search import GraphPathTool

        return GraphPathTool(client)

    @pytest.mark.asyncio
    async def test_fetch_returns_identifier_schema(self, live_client):
        """An occurrence_type path should have an IdentifierSchema."""
        tool = self._make_tool(live_client)
        # occurrence_type paths have HAS_IDENTIFIER_SCHEMA in the live graph
        result = await tool.fetch_dd_paths("summary/ids_properties/occurrence_type")
        if not result.nodes:
            pytest.skip("Path not found in graph")
        node = result.nodes[0]
        assert node.identifier_schema is not None
        assert node.identifier_schema.schema_path

    @pytest.mark.asyncio
    async def test_fetch_leaf_no_identifier_schema(self, live_client):
        """A normal leaf path should not have an IdentifierSchema."""
        tool = self._make_tool(live_client)
        result = await tool.fetch_dd_paths("equilibrium/time_slice/profiles_1d/psi")
        if not result.nodes:
            pytest.skip("Path not found in graph")
        node = result.nodes[0]
        assert node.identifier_schema is None

    @pytest.mark.asyncio
    async def test_fetch_version_history_populates(self, live_client):
        """include_version_history=True should populate version_changes."""
        tool = self._make_tool(live_client)
        # Fetch a path likely to have changes
        result = await tool.fetch_dd_paths(
            "equilibrium/time_slice/profiles_1d/psi",
            include_version_history=True,
        )
        if not result.nodes:
            pytest.skip("Path not found in graph")
        # We can't assert changes exist (depends on data), but the field
        # should be a list or None, not raise
        node = result.nodes[0]
        assert node.version_changes is None or isinstance(node.version_changes, list)

    @pytest.mark.asyncio
    async def test_fetch_version_history_disabled_is_none(self, live_client):
        """include_version_history=False should leave version_changes None."""
        tool = self._make_tool(live_client)
        result = await tool.fetch_dd_paths(
            "equilibrium/time_slice/profiles_1d/psi",
            include_version_history=False,
        )
        if not result.nodes:
            pytest.skip("Path not found in graph")
        assert result.nodes[0].version_changes is None

    @pytest.mark.asyncio
    async def test_fetch_multiple_with_enrichment(self, live_client):
        """Fetching multiple paths should populate enrichment for each."""
        tool = self._make_tool(live_client)
        result = await tool.fetch_dd_paths(
            [
                "summary/ids_properties/occurrence_type",
                "equilibrium/time_slice/profiles_1d/psi",
            ],
            include_version_history=True,
        )
        assert len(result.nodes) == 2
        # occurrence_type should have identifier schema
        occ = [n for n in result.nodes if n.path.endswith("occurrence_type")]
        if occ:
            assert occ[0].identifier_schema is not None


# ── T2: search_dd_clusters listing mode ────────────────────────────────


class TestClustersListingModeLive:
    """Test search_dd_clusters IDS listing mode on live data."""

    def _make_tool(self, client):
        from imas_codex.tools.graph_search import GraphClustersTool

        return GraphClustersTool(client)

    @pytest.mark.asyncio
    async def test_list_by_ids_returns_clusters(self, live_client):
        """Listing clusters for equilibrium should find some."""
        tool = self._make_tool(live_client)
        result = await tool.search_dd_clusters(
            ids_filter="equilibrium",
        )
        assert result["query_type"] == "ids_listing"
        assert result["clusters_found"] >= 1

    @pytest.mark.asyncio
    async def test_list_by_ids_section_only(self, live_client):
        """section_only=True should filter to section-level clusters."""
        tool = self._make_tool(live_client)
        result = await tool.search_dd_clusters(
            ids_filter="equilibrium",
            section_only=True,
        )
        assert result["query_type"] == "ids_listing"
        assert result["section_only"] is True
        # All paths should contain '/' (IDS-scoped)
        for c in result["clusters"]:
            assert any("/" in p for p in c["paths"])

    @pytest.mark.asyncio
    async def test_list_no_query_no_ids_error(self, live_client):
        """No query and no ids_filter should return an error."""
        tool = self._make_tool(live_client)
        result = await tool.search_dd_clusters()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_path_lookup_still_works(self, live_client):
        """Existing path-based lookup should still work."""
        tool = self._make_tool(live_client)
        result = await tool.search_dd_clusters("equilibrium/time_slice/boundary/psi")
        assert result["query_type"] == "path"


# ── T3: get_dd_version_context ───────────────────────────────────────────


class TestVersionContextLive:
    """Test VersionTool.get_dd_version_context on live data."""

    @pytest.fixture
    def version_tool(self, live_client):
        from imas_codex.tools.version_tool import VersionTool

        return VersionTool(live_client)

    @pytest.mark.anyio
    async def test_known_path(self, version_tool):
        """Query version context for a known DD path."""
        result = await version_tool.get_dd_version_context(
            "equilibrium/time_slice/profiles_1d/psi"
        )
        assert result["total_paths"] == 1
        assert "equilibrium/time_slice/profiles_1d/psi" in result["paths"]
        assert result["not_found"] == []

    @pytest.mark.anyio
    async def test_nonexistent_path(self, version_tool):
        result = await version_tool.get_dd_version_context("fake/nonexistent/thing")
        assert "fake/nonexistent/thing" in result["not_found"]

    @pytest.mark.anyio
    async def test_multiple_paths(self, version_tool):
        result = await version_tool.get_dd_version_context(
            [
                "equilibrium/time_slice/profiles_1d/psi",
                "core_profiles/profiles_1d/electrons/temperature",
            ]
        )
        assert result["total_paths"] == 2
        assert len(result["paths"]) == 2

    @pytest.mark.anyio
    async def test_empty_input(self, version_tool):
        result = await version_tool.get_dd_version_context("")
        assert "error" in result


# ── Tools delegation integration ─────────────────────────────────────────


class TestToolsDelegationLive:
    """Test Tools class delegation of new methods on live graph."""

    def _make_tools(self, client):
        from imas_codex.tools import Tools

        return Tools(graph_client=client)

    @pytest.mark.asyncio
    async def test_fetch_with_version_history(self, live_client):
        tools = self._make_tools(live_client)
        result = await tools.path_tool.fetch_dd_paths(
            "equilibrium/time_slice/profiles_1d/psi",
            include_version_history=True,
        )
        assert len(result.nodes) == 1

    @pytest.mark.asyncio
    async def test_get_dd_version_context(self, live_client):
        tools = self._make_tools(live_client)
        result = await tools.version_tool.get_dd_version_context(
            "equilibrium/time_slice/profiles_1d/psi"
        )
        assert result["total_paths"] == 1

    def test_registered_tools_include_version_context(self, live_client):
        tools = self._make_tools(live_client)
        names = tools.get_registered_tool_names()
        assert "get_dd_version_context" in names
