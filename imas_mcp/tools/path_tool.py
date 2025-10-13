"""
Path tool implementation.

Provides both fast validation and rich data retrieval for IMAS paths.
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.result_models import IdsPathResult
from imas_mcp.search.decorators import handle_errors, mcp_tool, measure_performance

from .base import BaseTool

logger = logging.getLogger(__name__)


class PathTool(BaseTool):
    """Tool for IMAS path validation and data retrieval."""

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "path_tool"

    @measure_performance(include_metrics=False, slow_threshold=0.1)
    @handle_errors(fallback=None)
    @mcp_tool(
        "Fast batch validation of IMAS paths. "
        "paths: space-delimited string or list of paths. "
        "ids: optional IDS name to prefix all paths (e.g., ids='equilibrium' + paths='time_slice/boundary/psi')"
    )
    async def check_imas_paths(
        self,
        paths: str | list[str],
        ids: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Check if one or more exact IMAS paths exist in the data dictionary.

        Fast validation tool for batch path existence checking without search overhead.
        Directly accesses the data dictionary for immediate results.

        Args:
            paths: One or more IMAS paths to validate. Accepts either:
                  - Space-delimited string: "time_slice/boundary/psi time_slice/boundary/psi_norm"
                  - List of paths: ["time_slice/boundary/psi", "profiles_1d/electrons/temperature"]
                  - Full paths: "equilibrium/time_slice/boundary/psi" (if ids not specified)
            ids: Optional IDS name to prefix to all paths. When specified, paths don't need
                 the IDS prefix. Example: ids="equilibrium", paths="time_slice/boundary/psi"
                 becomes "equilibrium/time_slice/boundary/psi"

        Returns:
            Dictionary with structured validation results:
            - summary: {"total": int, "found": int, "not_found": int, "invalid": int}
            - results: List of path results, each containing:
              - path: The queried path (with IDS prefix)
              - exists: Boolean indicating if path was found
              - ids_name: IDS name if path exists
              - data_type: Data type if available (optional)
              - units: Physical units if available (optional)
              - error: Error message if path format is invalid (optional)

        Examples:
            Single path (string):
                check_imas_paths("equilibrium/time_slice/boundary/psi")
                → {"summary": {"total": 1, "found": 1, "not_found": 0, "invalid": 0},
                   "results": [{"path": "equilibrium/time_slice/boundary/psi", "exists": true, "ids_name": "equilibrium"}]}

            Multiple paths with ids prefix (ensemble checking):
                check_imas_paths("time_slice/boundary/psi time_slice/boundary/psi_norm time_slice/boundary/type", ids="equilibrium")
                → {"summary": {"total": 3, "found": 3, "not_found": 0, "invalid": 0},
                   "results": [
                     {"path": "equilibrium/time_slice/boundary/psi", "exists": true, "ids_name": "equilibrium"},
                     {"path": "equilibrium/time_slice/boundary/psi_norm", "exists": true, "ids_name": "equilibrium"},
                     {"path": "equilibrium/time_slice/boundary/type", "exists": true, "ids_name": "equilibrium"}
                   ]}

            Multiple paths (list):
                check_imas_paths(["time_slice/boundary/psi", "time_slice/boundary/psi_norm"], ids="equilibrium")

        Note:
            This tool is optimized for exact path validation. For discovering paths
            or searching by concept, use search_imas instead.
        """
        # Convert paths to list if provided as space-delimited string
        if isinstance(paths, str):
            paths_list = paths.split()
        else:
            paths_list = paths

        # Initialize counters and results
        results = []
        found_count = 0
        not_found_count = 0
        invalid_count = 0

        # Validate each path
        for path in paths_list:
            path = path.strip()

            # Prefix with IDS name if provided and path doesn't already start with it
            if ids:
                # Check if path already has IDS prefix
                if not path.startswith(f"{ids}/"):
                    # Add IDS prefix
                    path = f"{ids}/{path}"

            # Check for invalid format
            if "/" not in path:
                invalid_count += 1
                results.append(
                    {
                        "path": path,
                        "exists": False,
                        "error": "Invalid format - must contain '/'",
                    }
                )
                continue

            # Access data dictionary directly via document store
            try:
                document = self.document_store.get_document(path)

                if document and document.metadata:
                    found_count += 1
                    metadata = document.metadata
                    result = {
                        "path": path,
                        "exists": True,
                        "ids_name": metadata.ids_name,
                    }

                    # Add optional fields if available (for detailed view)
                    if metadata.data_type:
                        result["data_type"] = metadata.data_type
                    if metadata.units:
                        result["units"] = metadata.units

                    results.append(result)
                    logger.debug(f"Path validation: {path} - exists")
                else:
                    not_found_count += 1
                    results.append(
                        {
                            "path": path,
                            "exists": False,
                        }
                    )
                    logger.debug(f"Path validation: {path} - not found")

            except Exception as e:
                invalid_count += 1
                logger.error(f"Error validating path {path}: {e}")
                results.append(
                    {
                        "path": path,
                        "exists": False,
                        "error": str(e),
                    }
                )

        # Build summary
        summary = {
            "total": len(paths_list),
            "found": found_count,
            "not_found": not_found_count,
            "invalid": invalid_count,
        }

        logger.info(f"Batch path validation: {found_count}/{len(paths_list)} found")
        return {"summary": summary, "results": results}

    @measure_performance(include_metrics=False, slow_threshold=0.2)
    @handle_errors(fallback=None)
    @mcp_tool(
        "Retrieve full IMAS path data with documentation. "
        "paths: space-delimited string or list of paths. "
        "ids: optional IDS name to prefix all paths (e.g., ids='equilibrium' + paths='time_slice/boundary/psi')"
    )
    async def fetch_imas_paths(
        self,
        paths: str | list[str],
        ids: str | None = None,
        ctx: Context | None = None,
    ) -> IdsPathResult:
        """
        Retrieve full data for one or more IMAS paths including documentation and metadata.

        Rich data retrieval tool for fetching complete path information with documentation,
        units, data types, and physics context. Returns structured IdsNode objects.

        Args:
            paths: One or more IMAS paths to retrieve. Accepts either:
                  - Space-delimited string: "time_slice/boundary/psi time_slice/boundary/psi_norm"
                  - List of paths: ["time_slice/boundary/psi", "profiles_1d/electrons/temperature"]
                  - Full paths: "equilibrium/time_slice/boundary/psi" (if ids not specified)
            ids: Optional IDS name to prefix to all paths. When specified, paths don't need
                 the IDS prefix. Example: ids="equilibrium", paths="time_slice/boundary/psi"
                 becomes "equilibrium/time_slice/boundary/psi"
            ctx: FastMCP context for potential future enhancements

        Returns:
            IdsPathResult containing:
            - nodes: List of IdsNode objects with complete documentation and metadata
            - summary: Statistics about the retrieval operation
            - physics_context: Aggregated physics domain information

        Examples:
            Single path retrieval:
                fetch_imas_paths("equilibrium/time_slice/boundary/psi")
                → IdsPathResult with one IdsNode containing full documentation

            Multiple paths with ids prefix:
                fetch_imas_paths("time_slice/boundary/psi time_slice/boundary/psi_norm", ids="equilibrium")
                → IdsPathResult with multiple IdsNode objects

            List of paths:
                fetch_imas_paths(["time_slice/boundary/psi", "profiles_1d/electrons/temperature"], ids="equilibrium")

        Note:
            For fast existence checking without documentation overhead, use check_imas_paths instead.
            For discovering paths by concept, use search_imas.
        """
        # Convert paths to list if provided as space-delimited string
        if isinstance(paths, str):
            paths_list = paths.split()
        else:
            paths_list = paths

        # Initialize tracking
        nodes = []
        found_count = 0
        not_found_count = 0
        invalid_count = 0
        physics_domains = set()

        # Retrieve each path
        for path in paths_list:
            path = path.strip()

            # Prefix with IDS name if provided and path doesn't already start with it
            if ids:
                if not path.startswith(f"{ids}/"):
                    path = f"{ids}/{path}"

            # Check for invalid format
            if "/" not in path:
                invalid_count += 1
                logger.warning(f"Invalid path format (no '/'): {path}")
                continue

            # Retrieve document from document store
            try:
                document = self.document_store.get_document(path)

                if document and document.metadata:
                    found_count += 1
                    metadata = document.metadata

                    # Collect physics domain
                    if metadata.physics_domain:
                        physics_domains.add(metadata.physics_domain)

                    # Use document.to_datapath() to get complete IdsNode with all fields
                    node = document.to_datapath()
                    nodes.append(node)
                    logger.debug(f"Path retrieved: {path}")
                else:
                    not_found_count += 1
                    logger.debug(f"Path not found: {path}")

            except Exception as e:
                invalid_count += 1
                logger.error(f"Error retrieving path {path}: {e}")

        # Build summary
        summary = {
            "total_requested": len(paths_list),
            "retrieved": found_count,
            "not_found": not_found_count,
            "invalid": invalid_count,
            "physics_domains": sorted(physics_domains),
        }

        # Build result
        result = IdsPathResult(
            nodes=nodes,
            summary=summary,
            query=" ".join(paths_list[:3]) + ("..." if len(paths_list) > 3 else ""),
            search_mode=SearchMode.LEXICAL,  # Direct lookup, not search
            physics_domains=sorted(physics_domains),
        )

        logger.info(
            f"Path retrieval completed: {found_count}/{len(paths_list)} retrieved"
        )
        return result
