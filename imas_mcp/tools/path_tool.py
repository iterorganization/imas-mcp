"""
Path validation tool implementation.

Fast existence checking for IMAS paths without search overhead.
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp.search.decorators import handle_errors, mcp_tool, measure_performance

from .base import BaseTool

logger = logging.getLogger(__name__)


class ValidateTool(BaseTool):
    """Tool for fast IMAS path validation."""

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "check_ids_paths"

    @measure_performance(include_metrics=False, slow_threshold=0.1)
    @handle_errors(fallback=None)
    @mcp_tool(
        "Fast batch validation of IMAS paths. "
        "paths: space-delimited string or list of paths. "
        "ids: optional IDS name to prefix all paths (e.g., ids='equilibrium' + paths='time_slice/boundary/psi')"
    )
    async def check_ids_paths(
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
                check_ids_paths("equilibrium/time_slice/boundary/psi")
                → {"summary": {"total": 1, "found": 1, "not_found": 0, "invalid": 0},
                   "results": [{"path": "equilibrium/time_slice/boundary/psi", "exists": true, "ids_name": "equilibrium"}]}

            Multiple paths with ids prefix (ensemble checking):
                check_ids_paths("time_slice/boundary/psi time_slice/boundary/psi_norm time_slice/boundary/type", ids="equilibrium")
                → {"summary": {"total": 3, "found": 3, "not_found": 0, "invalid": 0},
                   "results": [
                     {"path": "equilibrium/time_slice/boundary/psi", "exists": true, "ids_name": "equilibrium"},
                     {"path": "equilibrium/time_slice/boundary/psi_norm", "exists": true, "ids_name": "equilibrium"},
                     {"path": "equilibrium/time_slice/boundary/type", "exists": true, "ids_name": "equilibrium"}
                   ]}

            Multiple paths (list):
                check_ids_paths(["time_slice/boundary/psi", "time_slice/boundary/psi_norm"], ids="equilibrium")

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
