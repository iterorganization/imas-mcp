"""
Path tool implementation.

Provides both fast validation and rich data retrieval for IMAS paths,
with migration suggestions for deprecated paths and rename history.
"""

import logging

from fastmcp import Context

from imas_mcp.mappings import PathMap, get_path_map
from imas_mcp.models.constants import SearchMode
from imas_mcp.models.result_models import (
    CheckPathsResult,
    CheckPathsResultItem,
    DeprecatedPathInfo,
    ExcludedPathInfo,
    FetchPathsResult,
)
from imas_mcp.search.decorators import (
    handle_errors,
    mcp_tool,
    measure_performance,
)
from imas_mcp.search.document_store import DocumentStore

from .base import BaseTool

logger = logging.getLogger(__name__)


class PathTool(BaseTool):
    """Tool for IMAS path validation and data retrieval."""

    def __init__(
        self,
        document_store: DocumentStore | None = None,
        path_map: PathMap | None = None,
    ):
        """
        Initialize PathTool.

        Args:
            document_store: Optional DocumentStore instance.
            path_map: Optional PathMap for testing. Uses singleton if None.
        """
        super().__init__(document_store=document_store)
        self._path_map = path_map

    @property
    def path_map(self) -> PathMap:
        """Get the path map (lazy loaded)."""
        if self._path_map is None:
            self._path_map = get_path_map()
        return self._path_map

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
    ) -> CheckPathsResult:
        """
        Check if one or more exact IMAS paths exist in the data dictionary.

        Fast validation tool for batch path existence checking without search overhead.
        Directly accesses the data dictionary for immediate results. Returns migration
        suggestions for deprecated paths and rename history for current paths.

        Args:
            paths: One or more IMAS paths to validate. Accepts either:
                  - Space-delimited string: "time_slice/boundary/psi time_slice/boundary/psi_norm"
                  - List of paths: ["time_slice/boundary/psi", "profiles_1d/electrons/temperature"]
                  - Full paths: "equilibrium/time_slice/boundary/psi" (if ids not specified)
            ids: Optional IDS name to prefix to all paths. When specified, paths don't need
                 the IDS prefix. Example: ids="equilibrium", paths="time_slice/boundary/psi"
                 becomes "equilibrium/time_slice/boundary/psi"

        Returns:
            CheckPathsResult with:
            - summary: Counts (total, found, not_found, invalid)
            - results: List of CheckPathsResultItem for each path

        Examples:
            Path exists (current):
                check_imas_paths("equilibrium/time_slice/constraints/b_field_pol_probe")
                → CheckPathsResult with exists=True, rename history

            Path deprecated (has migration):
                check_imas_paths("equilibrium/time_slice/constraints/bpol_probe")
                → CheckPathsResult with exists=False, migration info

        Note:
            This tool is optimized for exact path validation. For discovering paths
            or searching by concept, use search_imas instead.
        """
        # Convert paths to list if provided as space-delimited string
        if isinstance(paths, str):
            paths_list = paths.split()
        else:
            paths_list = list(paths)

        # Initialize counters and results
        results: list[CheckPathsResultItem] = []
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
                    CheckPathsResultItem(
                        path=path,
                        exists=False,
                        error="Invalid format - must contain '/'",
                    )
                )
                continue

            # Access data dictionary directly via document store
            try:
                document = self.document_store.get_document(path)

                if document and document.metadata:
                    found_count += 1
                    metadata = document.metadata

                    # Build rename history if available
                    renamed_from = None
                    rename_history = self.path_map.get_rename_history(path)
                    if rename_history:
                        renamed_from = [
                            {
                                "old_path": entry.old_path,
                                "deprecated_in": entry.deprecated_in,
                            }
                            for entry in rename_history
                        ]

                    results.append(
                        CheckPathsResultItem(
                            path=path,
                            exists=True,
                            ids_name=metadata.ids_name,
                            data_type=metadata.data_type,
                            units=metadata.units,
                            renamed_from=renamed_from,
                        )
                    )
                    logger.debug(f"Path validation: {path} - exists")
                else:
                    not_found_count += 1
                    migration_info = None
                    excluded_info = None

                    # Check for migration suggestion
                    migration = self.path_map.get_mapping(path)
                    if migration:
                        migration_info = {
                            "new_path": migration.new_path,
                            "deprecated_in": migration.deprecated_in,
                            "last_valid_version": migration.last_valid_version,
                        }
                        # Check if the new path is excluded from index
                        if migration.new_path:
                            exclusion_reason = self.path_map.get_exclusion_reason(
                                migration.new_path
                            )
                            if exclusion_reason:
                                migration_info["new_path_excluded"] = True
                                migration_info["exclusion_reason"] = (
                                    self.path_map.get_exclusion_description(
                                        exclusion_reason
                                    )
                                )
                    else:
                        # Not deprecated - check if path is excluded from index
                        exclusion_reason = self.path_map.get_exclusion_reason(path)
                        if exclusion_reason:
                            excluded_info = {
                                "reason_key": exclusion_reason,
                                "reason": self.path_map.get_exclusion_description(
                                    exclusion_reason
                                ),
                            }

                    results.append(
                        CheckPathsResultItem(
                            path=path,
                            exists=False,
                            migration=migration_info,
                            excluded=excluded_info,
                        )
                    )
                    logger.debug(f"Path validation: {path} - not found")

            except Exception as e:
                invalid_count += 1
                logger.error(f"Error validating path {path}: {e}")
                results.append(
                    CheckPathsResultItem(
                        path=path,
                        exists=False,
                        error=str(e),
                    )
                )

        # Build summary
        summary = {
            "total": len(paths_list),
            "found": found_count,
            "not_found": not_found_count,
            "invalid": invalid_count,
        }

        logger.info(f"Batch path validation: {found_count}/{len(paths_list)} found")
        return CheckPathsResult(summary=summary, results=results)

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
    ) -> FetchPathsResult:
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
            FetchPathsResult containing:
            - nodes: List of IdsNode objects with complete documentation and metadata
            - summary: Statistics about the retrieval operation
            - physics_context: Aggregated physics domain information

        Examples:
            Single path retrieval:
                fetch_imas_paths("equilibrium/time_slice/boundary/psi")
                → FetchPathsResult with one IdsNode containing full documentation

            Multiple paths with ids prefix:
                fetch_imas_paths("time_slice/boundary/psi time_slice/boundary/psi_norm", ids="equilibrium")
                → FetchPathsResult with multiple IdsNode objects

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
            paths_list = list(paths)

        # Initialize tracking
        nodes = []
        deprecated_paths: list[DeprecatedPathInfo] = []
        excluded_paths: list[ExcludedPathInfo] = []
        found_count = 0
        not_found_count = 0
        deprecated_count = 0
        excluded_count = 0
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
                    # Path not found - check for migration
                    migration = self.path_map.get_mapping(path)
                    if migration:
                        deprecated_count += 1
                        # Check if new path is excluded
                        new_path_excluded = False
                        exclusion_reason = None
                        if migration.new_path:
                            reason_key = self.path_map.get_exclusion_reason(
                                migration.new_path
                            )
                            if reason_key:
                                new_path_excluded = True
                                exclusion_reason = (
                                    self.path_map.get_exclusion_description(reason_key)
                                )
                        deprecated_paths.append(
                            DeprecatedPathInfo(
                                path=path,
                                new_path=migration.new_path,
                                deprecated_in=migration.deprecated_in,
                                last_valid_version=migration.last_valid_version,
                                new_path_excluded=new_path_excluded,
                                exclusion_reason=exclusion_reason,
                            )
                        )
                        logger.debug(f"Path deprecated: {path} -> {migration.new_path}")
                    else:
                        # Not deprecated - check if path is excluded from index
                        reason_key = self.path_map.get_exclusion_reason(path)
                        if reason_key:
                            excluded_count += 1
                            excluded_paths.append(
                                ExcludedPathInfo(
                                    path=path,
                                    reason_key=reason_key,
                                    reason_description=self.path_map.get_exclusion_description(
                                        reason_key
                                    ),
                                )
                            )
                            logger.debug(f"Path excluded: {path} ({reason_key})")
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
            "deprecated": deprecated_count,
            "excluded": excluded_count,
            "not_found": not_found_count,
            "invalid": invalid_count,
            "physics_domains": sorted(physics_domains),
        }

        # Build result
        result = FetchPathsResult(
            nodes=nodes,
            deprecated_paths=deprecated_paths,
            excluded_paths=excluded_paths,
            summary=summary,
            query=" ".join(paths_list[:3]) + ("..." if len(paths_list) > 3 else ""),
            search_mode=SearchMode.LEXICAL,  # Direct lookup, not search
            physics_domains=sorted(physics_domains),
        )

        logger.info(
            f"Path retrieval completed: {found_count}/{len(paths_list)} retrieved"
        )
        return result
