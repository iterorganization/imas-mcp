"""Tests for graceful degradation: Tier 2 DD tools work without embeddings.

Verifies that graph-only DD tools (check_imas_paths, list_imas_paths, etc.)
can initialize when embedding warmup fails, while search tools (search_imas,
search_imas_clusters, find_related_imas_paths) still require full warmup.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level singletons between tests."""
    import imas_codex.llm.server as srv

    original = {
        "_warmup_applied": srv._warmup_applied,
        "_graph_warmup_applied": srv._graph_warmup_applied,
        "_imas_tools_instance": srv._imas_tools_instance,
        "_graph_client": srv._graph_client,
        "GraphClient": srv.GraphClient,
        "get_schema": srv.get_schema,
        "to_cypher_props": srv.to_cypher_props,
    }
    yield
    srv._warmup_applied = original["_warmup_applied"]
    srv._graph_warmup_applied = original["_graph_warmup_applied"]
    srv._imas_tools_instance = original["_imas_tools_instance"]
    srv._graph_client = original["_graph_client"]
    srv.GraphClient = original["GraphClient"]
    srv.get_schema = original["get_schema"]
    srv.to_cypher_props = original["to_cypher_props"]


@pytest.fixture()
def mock_graph_warmup():
    """Patch warmup.graph() to return mock objects without Neo4j."""
    mock_gc_cls = MagicMock(name="GraphClient")
    mock_gc_instance = MagicMock(name="GraphClient()")
    mock_gc_cls.from_profile.return_value = mock_gc_instance

    graph_ns = {
        "GraphClient": mock_gc_cls,
        "get_schema": MagicMock(name="get_schema"),
        "to_cypher_props": MagicMock(name="to_cypher_props"),
    }
    with patch("imas_codex.llm.server.warmup") as mock_warmup:
        mock_warmup.graph.return_value = graph_ns
        yield mock_warmup, mock_gc_cls, mock_gc_instance


# ---------------------------------------------------------------------------
# _require_graph tests
# ---------------------------------------------------------------------------


class TestRequireGraph:
    """Tests for _require_graph() — graph-only warmup."""

    def test_populates_graph_globals(self, mock_graph_warmup):
        """_require_graph() should populate GraphClient, get_schema, to_cypher_props."""
        import imas_codex.llm.server as srv

        srv._graph_warmup_applied = False
        srv.GraphClient = None
        srv.get_schema = None
        srv.to_cypher_props = None

        srv._require_graph()

        assert srv.GraphClient is not None
        assert srv.get_schema is not None
        assert srv.to_cypher_props is not None
        assert srv._graph_warmup_applied is True

    def test_does_not_call_embeddings(self, mock_graph_warmup):
        """_require_graph() must NOT call warmup.embeddings()."""
        import imas_codex.llm.server as srv

        mock_warmup = mock_graph_warmup[0]
        srv._graph_warmup_applied = False

        srv._require_graph()

        mock_warmup.graph.assert_called_once()
        mock_warmup.embeddings.assert_not_called()
        mock_warmup.discovery.assert_not_called()
        mock_warmup.remote.assert_not_called()

    def test_idempotent(self, mock_graph_warmup):
        """Second call should be a no-op (fast path)."""
        import imas_codex.llm.server as srv

        mock_warmup = mock_graph_warmup[0]
        srv._graph_warmup_applied = False

        srv._require_graph()
        srv._require_graph()

        mock_warmup.graph.assert_called_once()

    def test_full_warmup_sets_graph_flag(self):
        """_require_warmup() should also set _graph_warmup_applied."""
        # Verify the flag is set by inspecting the source
        # (we can't easily run _require_warmup without a real server)
        import inspect

        import imas_codex.llm.server as srv

        source = inspect.getsource(srv._require_warmup)
        assert "_graph_warmup_applied = True" in source


# ---------------------------------------------------------------------------
# _get_graph_client tests
# ---------------------------------------------------------------------------


class TestGetGraphClient:
    """Tests for _get_graph_client() singleton."""

    def test_creates_client_from_profile(self, mock_graph_warmup):
        """Should create GraphClient.from_profile() on first call."""
        import imas_codex.llm.server as srv

        _, mock_gc_cls, mock_gc_instance = mock_graph_warmup
        srv._graph_warmup_applied = False
        srv._graph_client = None

        result = srv._get_graph_client()

        mock_gc_cls.from_profile.assert_called_once()
        assert result is mock_gc_instance

    def test_singleton_behavior(self, mock_graph_warmup):
        """Second call returns same instance without creating a new one."""
        import imas_codex.llm.server as srv

        _, mock_gc_cls, mock_gc_instance = mock_graph_warmup
        srv._graph_warmup_applied = False
        srv._graph_client = None

        first = srv._get_graph_client()
        second = srv._get_graph_client()

        assert first is second
        mock_gc_cls.from_profile.assert_called_once()

    def test_thread_safety(self, mock_graph_warmup):
        """Multiple threads should all get the same instance."""
        import imas_codex.llm.server as srv

        _, mock_gc_cls, mock_gc_instance = mock_graph_warmup
        srv._graph_warmup_applied = False
        srv._graph_client = None

        results = []
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            results.append(srv._get_graph_client())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert all(r is mock_gc_instance for r in results)


# ---------------------------------------------------------------------------
# _get_imas_tools with graph_only tests
# ---------------------------------------------------------------------------


class TestGetImasToolsGraphOnly:
    """Tests for _get_imas_tools(graph_only=True)."""

    def test_graph_only_skips_full_warmup(self, mock_graph_warmup):
        """graph_only=True should not call _require_warmup()."""
        import imas_codex.llm.server as srv

        mock_warmup = mock_graph_warmup[0]
        srv._graph_warmup_applied = False
        srv._graph_client = None
        srv._imas_tools_instance = None

        with patch("imas_codex.llm.server._require_warmup") as mock_full:
            with patch("imas_codex.tools.Tools") as MockTools:
                MockTools.return_value = MagicMock()
                srv._get_imas_tools(graph_only=True)

            mock_full.assert_not_called()
            mock_warmup.embeddings.assert_not_called()

    def test_graph_only_uses_standalone_client(self, mock_graph_warmup):
        """graph_only=True should use _get_graph_client(), not _get_repl()."""
        import imas_codex.llm.server as srv

        srv._graph_warmup_applied = False
        srv._graph_client = None
        srv._imas_tools_instance = None

        with patch("imas_codex.llm.server._get_repl") as mock_repl:
            with patch("imas_codex.tools.Tools") as MockTools:
                MockTools.return_value = MagicMock()
                srv._get_imas_tools(graph_only=True)

            mock_repl.assert_not_called()

    def test_full_warmup_when_not_graph_only(self, mock_graph_warmup):
        """Default (graph_only=False) should call _require_warmup()."""
        import imas_codex.llm.server as srv

        srv._imas_tools_instance = None

        with patch("imas_codex.llm.server._require_warmup") as mock_full:
            with patch("imas_codex.llm.server._get_repl") as mock_repl:
                mock_repl.return_value = {"gc": MagicMock()}
                with patch("imas_codex.tools.Tools") as MockTools:
                    MockTools.return_value = MagicMock()
                    srv._get_imas_tools(graph_only=False)

            mock_full.assert_called_once()


# ---------------------------------------------------------------------------
# Tier classification tests
# ---------------------------------------------------------------------------

# Tier 1 tools: require embeddings (search_imas, search_imas_clusters,
#               find_related_imas_paths)
# Tier 2 tools: graph-only (check_imas_paths, fetch_imas_paths, list_imas_paths,
#               fetch_error_fields, get_imas_overview, get_imas_identifiers,
#               get_dd_versions, get_dd_version_context, analyze_imas_structure,
#               export_imas_ids, export_imas_domain)

TIER_1_TOOLS = {"search_imas", "search_imas_clusters", "find_related_imas_paths"}
TIER_2_TOOLS = {
    "check_imas_paths",
    "fetch_imas_paths",
    "list_imas_paths",
    "fetch_error_fields",
    "get_imas_overview",
    "get_imas_identifiers",
    "get_dd_versions",
    "get_dd_version_context",
    "analyze_imas_structure",
    "export_imas_ids",
    "export_imas_domain",
}


class TestToolTierClassification:
    """Verify that tool handlers call the correct warmup path.

    This is a source-level test: it inspects the registered tool functions
    to confirm Tier 2 tools pass graph_only=True and Tier 1 tools do not.
    """

    def test_tier2_tools_use_graph_only(self):
        """All Tier 2 promoted tools should call _get_imas_tools(graph_only=True)."""
        import inspect

        from imas_codex.llm.server import AgentsServer

        server = AgentsServer(dd_only=True)
        components = server.mcp._local_provider._components

        for tool_name in TIER_2_TOOLS:
            # FastMCP appends '@' to tool keys
            key = f"tool:{tool_name}@"
            assert key in components, f"Tool {tool_name} not registered"

            fn = components[key].fn
            source = inspect.getsource(fn)
            assert "graph_only=True" in source, (
                f"Tier 2 tool '{tool_name}' should call "
                f"_get_imas_tools(graph_only=True)"
            )

    def test_tier1_tools_do_not_use_graph_only(self):
        """Tier 1 search tools should NOT pass graph_only=True."""
        import inspect

        from imas_codex.llm.server import AgentsServer

        server = AgentsServer(dd_only=True)
        components = server.mcp._local_provider._components

        for tool_name in TIER_1_TOOLS:
            key = f"tool:{tool_name}@"
            assert key in components, f"Tool {tool_name} not registered"

            fn = components[key].fn
            source = inspect.getsource(fn)
            # search_imas calls _get_imas_tools() without graph_only
            assert "graph_only=True" not in source, (
                f"Tier 1 tool '{tool_name}' should NOT use graph_only=True"
            )
