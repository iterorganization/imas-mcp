"""Search tool benchmarks (MCP layer).

Measures end-to-end latency of search_dd_paths, search_dd_clusters,
and get_dd_path_context through the FastMCP client against a
real GHCR graph dump.
"""

from __future__ import annotations

from benchmarks.conftest_bench import IMAS_PATHS, SEARCH_QUERIES, run_tool


class SearchToolBenchmarks:
    """Benchmark suite for IMAS search tools."""

    timeout = 120

    def setup(self):
        """Warmup: trigger lazy server init + embedding model load."""
        run_tool("search_dd_paths", {"query": "warmup", "k": 1})

    # -- search_dd_paths ---------------------------------------------------------

    def time_search_imas_basic(self):
        """Baseline hybrid search."""
        run_tool("search_dd_paths", {"query": SEARCH_QUERIES["simple"], "k": 10})

    def time_search_imas_filtered(self):
        """IDS-filtered search."""
        run_tool(
            "search_dd_paths",
            {
                "query": SEARCH_QUERIES["simple"],
                "ids_filter": "core_profiles",
                "k": 10,
            },
        )

    def time_search_imas_with_version(self):
        """Version enrichment overhead."""
        run_tool(
            "search_dd_paths",
            {
                "query": SEARCH_QUERIES["simple"],
                "include_version_context": True,
                "k": 10,
            },
        )

    def time_search_imas_large_k(self):
        """Scaling with result count."""
        run_tool("search_dd_paths", {"query": SEARCH_QUERIES["simple"], "k": 100})

    def time_search_imas_complex_query(self):
        """Multi-word semantic query."""
        run_tool("search_dd_paths", {"query": SEARCH_QUERIES["complex"], "k": 10})

    # -- search_dd_clusters ------------------------------------------------

    def time_search_clusters_semantic(self):
        """Cluster centroid search."""
        run_tool(
            "search_dd_clusters",
            {"query": SEARCH_QUERIES["physics_specific"]},
        )

    def time_search_clusters_filtered(self):
        """Filtered cluster search."""
        run_tool(
            "search_dd_clusters",
            {"query": SEARCH_QUERIES["simple"], "ids_filter": "equilibrium"},
        )

    def time_search_clusters_by_path(self):
        """Path-based cluster lookup."""
        run_tool(
            "search_dd_clusters",
            {"query": IMAS_PATHS["leaf"]},
        )

    # -- get_dd_path_context -----------------------------------------------

    def time_path_context_all(self):
        """Full cross-IDS analysis."""
        run_tool(
            "get_dd_path_context",
            {"path": IMAS_PATHS["leaf"]},
        )

    def time_path_context_cluster(self):
        """Cluster-only relationships."""
        run_tool(
            "get_dd_path_context",
            {"path": IMAS_PATHS["branch"]},
        )

    # -- memory --------------------------------------------------------------

    def peakmem_search_imas_basic(self):
        """Memory footprint for basic search."""
        run_tool("search_dd_paths", {"query": SEARCH_QUERIES["simple"], "k": 10})

    def peakmem_search_imas_large_k(self):
        """Memory scaling with result count."""
        run_tool("search_dd_paths", {"query": SEARCH_QUERIES["simple"], "k": 100})
