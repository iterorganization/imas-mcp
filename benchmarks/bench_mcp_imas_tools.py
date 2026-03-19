"""IMAS Data Dictionary tool benchmarks (MCP layer).

Measures end-to-end latency of the 11 IMAS DD read tools through
the FastMCP client against a real GHCR graph dump.
"""

from __future__ import annotations

from benchmarks.conftest_bench import IDS_NAMES, IMAS_PATHS, run_tool


class IMASToolBenchmarks:
    """Benchmark suite for IMAS Data Dictionary tools."""

    timeout = 120

    def setup(self):
        """Warmup: trigger lazy server init."""
        run_tool("get_dd_versions")

    # -- check_imas_paths ----------------------------------------------------

    def time_check_paths_single(self):
        """Single path validation."""
        run_tool("check_imas_paths", {"paths": IMAS_PATHS["leaf"]})

    def time_check_paths_batch(self):
        """Batch validation (10 comma-separated paths)."""
        paths = ",".join(
            [
                "core_profiles/profiles_1d/electrons/temperature",
                "core_profiles/profiles_1d/electrons/density",
                "core_profiles/profiles_1d/ion/temperature",
                "equilibrium/time_slice/profiles_1d/psi",
                "equilibrium/time_slice/profiles_1d/q",
                "equilibrium/time_slice/global_quantities/ip",
                "equilibrium/time_slice/global_quantities/magnetic_axis/r",
                "magnetics/flux_loop/flux/data",
                "wall/description_2d/limiter/unit/outline/r",
                "pf_active/coil/current/data",
            ]
        )
        run_tool("check_imas_paths", {"paths": paths})

    # -- fetch_imas_paths ----------------------------------------------------

    def time_fetch_paths_single(self):
        """Full path documentation."""
        run_tool("fetch_imas_paths", {"paths": IMAS_PATHS["leaf"]})

    def time_fetch_paths_with_history(self):
        """Version history enrichment."""
        run_tool(
            "fetch_imas_paths",
            {"paths": IMAS_PATHS["leaf"], "include_version_history": True},
        )

    # -- fetch_error_fields --------------------------------------------------

    def time_fetch_error_fields(self):
        """Error field traversal."""
        run_tool(
            "fetch_error_fields",
            {"path": "core_profiles/profiles_1d/electrons/temperature"},
        )

    # -- list_imas_paths -----------------------------------------------------

    def time_list_paths_ids(self):
        """Full IDS enumeration."""
        run_tool("list_imas_paths", {"paths": IDS_NAMES["large"]})

    def time_list_paths_subtree(self):
        """Subtree enumeration."""
        run_tool("list_imas_paths", {"paths": IMAS_PATHS["branch"]})

    def time_list_paths_leaf_only(self):
        """Leaf filtering."""
        run_tool(
            "list_imas_paths",
            {"paths": IDS_NAMES["small"], "leaf_only": True},
        )

    # -- get_imas_overview ---------------------------------------------------

    def time_overview_all(self):
        """Full IDS summary scan."""
        run_tool("get_imas_overview", {"ids": IDS_NAMES["large"]})

    def time_overview_filtered(self):
        """Filtered overview with vector search."""
        run_tool("get_imas_overview", {"ids": IDS_NAMES["domain"]})

    # -- get_imas_identifiers ------------------------------------------------

    def time_identifiers(self):
        """Identifier schema search."""
        run_tool("get_imas_identifiers", {"ids": IDS_NAMES["small"]})

    # -- analyze_imas_structure ----------------------------------------------

    def time_structure_analysis(self):
        """Hierarchical analysis."""
        run_tool("analyze_imas_structure", {"ids": IDS_NAMES["large"]})

    # -- export --------------------------------------------------------------

    def time_export_ids(self):
        """Full IDS export."""
        run_tool("export_imas_ids", {"ids_list": IDS_NAMES["small"]})

    def time_export_domain(self):
        """Domain export."""
        run_tool("export_imas_domain", {"domain": IDS_NAMES["domain"]})

    # -- DD version context --------------------------------------------------

    def time_dd_version_context(self):
        """Version change history."""
        run_tool("get_dd_version_context", {"dd_version": 4})

    def time_dd_versions(self):
        """Version metadata."""
        run_tool("get_dd_versions")

    # -- memory --------------------------------------------------------------

    def peakmem_export_ids(self):
        """Export memory footprint."""
        run_tool("export_imas_ids", {"ids_list": IDS_NAMES["large"]})
