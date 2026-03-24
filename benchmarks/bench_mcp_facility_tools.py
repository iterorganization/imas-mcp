"""Facility tool benchmarks (MCP layer).

Measures end-to-end latency of facility-scoped tools (signals, docs,
code, fetch) and the graph schema tool.  Facility tools gracefully
skip when no facility data is present in the graph.
"""

from __future__ import annotations

from benchmarks.conftest_bench import SEARCH_QUERIES, run_tool


class FacilityToolBenchmarks:
    """Benchmark suite for facility-scoped MCP tools.

    Requires facility data in the loaded graph dump.  If no Facility
    nodes are present, ``setup()`` raises ``NotImplementedError`` and
    ASV skips the entire class.
    """

    timeout = 120

    def setup(self):
        """Check for facility data; skip entire suite if absent."""
        from benchmarks.conftest_bench import _fixture

        result = _fixture.graph_client.query("MATCH (f:Facility) RETURN count(f) AS n")
        if not result or result[0]["n"] == 0:
            raise NotImplementedError("No facility data in graph")

        # Determine first available facility for benchmarks
        fac = _fixture.graph_client.query(
            "MATCH (f:Facility) RETURN f.id AS id LIMIT 1"
        )
        self.facility = fac[0]["id"]

        # Warmup
        run_tool(
            "search_signals",
            {"query": "warmup", "facility": self.facility, "k": 1},
        )

    # -- search_signals ------------------------------------------------------

    def time_search_signals(self):
        """Hybrid signal search with enrichment."""
        run_tool(
            "search_signals",
            {
                "query": SEARCH_QUERIES["simple"],
                "facility": self.facility,
                "k": 10,
            },
        )

    # -- signal_analytics ----------------------------------------------------

    def time_signal_analytics(self):
        """Aggregate signal counts."""
        run_tool("signal_analytics", {"facility": self.facility})

    # -- search_docs ---------------------------------------------------------

    def time_search_docs(self):
        """Wiki/doc hybrid search."""
        run_tool(
            "search_docs",
            {
                "query": SEARCH_QUERIES["simple"],
                "facility": self.facility,
                "k": 10,
            },
        )

    # -- search_code ---------------------------------------------------------

    def time_search_code(self):
        """Code hybrid search."""
        run_tool(
            "search_code",
            {
                "query": SEARCH_QUERIES["simple"],
                "facility": self.facility,
                "k": 10,
            },
        )

    # -- get_discovery_context -----------------------------------------------

    def time_discovery_context(self):
        """Coverage/gap analysis."""
        run_tool("get_discovery_context", {"facility": self.facility})


class GraphSchemaToolBenchmarks:
    """Benchmark get_graph_schema with various scopes.

    Does NOT require facility data — works with any loaded graph.
    """

    timeout = 60

    def setup(self):
        """Warmup."""
        run_tool("get_graph_schema")

    def time_graph_schema_overview(self):
        """Schema overview (default scope)."""
        run_tool("get_graph_schema")

    def time_graph_schema_signals(self):
        """Signals task scope."""
        run_tool("get_graph_schema", {"scope": "signals"})

    def time_graph_schema_imas(self):
        """IMAS task scope."""
        run_tool("get_graph_schema", {"scope": "imas"})
