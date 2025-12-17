"""
Path tool implementation.

Provides both fast validation and rich data retrieval for IMAS paths,
with migration suggestions for deprecated paths and rename history.
"""

import logging
from typing import TYPE_CHECKING

from fastmcp import Context

from imas_codex.mappings import PathMap, get_path_map
from imas_codex.models.constants import SearchMode
from imas_codex.models.result_models import (
    CheckPathsResult,
    CheckPathsResultItem,
    DeprecatedPathInfo,
    ExcludedPathInfo,
    FetchPathsResult,
    NotFoundPathInfo,
)
from imas_codex.search.decorators import (
    handle_errors,
    mcp_tool,
    measure_performance,
)
from imas_codex.search.document_store import DocumentStore
from imas_codex.search.fuzzy_matcher import suggest_correction

from .base import BaseTool
from .utils import normalize_paths_input

if TYPE_CHECKING:
    from imas_codex.clusters.search import ClusterSearcher

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
        self._cluster_searcher: ClusterSearcher | None = None

    @property
    def path_map(self) -> PathMap:
        """Get the path map (lazy loaded)."""
        if self._path_map is None:
            self._path_map = get_path_map()
        return self._path_map

    @property
    def cluster_searcher(self) -> "ClusterSearcher | None":
        """Get the cluster searcher (lazy loaded)."""
        if self._cluster_searcher is None:
            try:
                from imas_codex.core.clusters import Clusters

                clusters = Clusters()
                if clusters.is_available():
                    from imas_codex.clusters.search import ClusterSearcher

                    self._cluster_searcher = ClusterSearcher(
                        clusters=clusters.get_clusters()
                    )
            except Exception as e:
                logger.debug(f"Cluster searcher not available: {e}")
        return self._cluster_searcher

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "path_tool"

    def _get_valid_ids_names(self) -> list[str]:
        """Get list of valid IDS names from document store."""
        return self.document_store._get_available_ids()

    def _get_valid_paths(self) -> list[str]:
        """Get list of valid paths from document store."""
        # Ensure documents are loaded before accessing the index
        self.document_store._ensure_loaded()
        return list(self.document_store._index.by_path_id.keys())

    def _get_suggestion_for_path(self, path: str) -> str | None:
        """Get a typo suggestion for a not-found path."""
        try:
            valid_ids = self._get_valid_ids_names()
            valid_paths = self._get_valid_paths()
            return suggest_correction(path, valid_ids, valid_paths)
        except Exception as e:
            logger.debug(f"Error getting suggestion for path {path}: {e}")
            return None

    @measure_performance(include_metrics=False, slow_threshold=0.1)
    @handle_errors(fallback=None)
    @mcp_tool(
        "Fast batch validation of IMAS paths. "
        "paths: Space-delimited paths (e.g., 'equilibrium/time_slice/boundary/psi core_profiles/profiles_1d/electrons/temperature'). "
        "ids: Optional IDS prefix for relative paths (e.g., ids='equilibrium', paths='time_slice/boundary/psi'). "
        "Returns: existence status, data type, units, and typo suggestions for not-found paths."
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
        # Normalize paths to list using utility function
        paths_list = normalize_paths_input(paths)

        # Handle empty input with helpful error message
        if not paths_list:
            return CheckPathsResult(
                summary={"total": 0, "found": 0, "not_found": 0, "invalid": 0},
                results=[],
                error="No paths provided. Examples:\n"
                "  check_imas_paths('equilibrium/time_slice/boundary/psi')\n"
                "  check_imas_paths('time_slice/boundary/psi', ids='equilibrium')",
            )

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

                    # Get suggestion for not-found paths (only if not deprecated/excluded)
                    suggestion = None
                    if not migration_info and not excluded_info:
                        suggestion = self._get_suggestion_for_path(path)

                    results.append(
                        CheckPathsResultItem(
                            path=path,
                            exists=False,
                            migration=migration_info,
                            excluded=excluded_info,
                            suggestion=suggestion,
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
        "paths: Space-delimited paths (e.g., 'equilibrium/time_slice/global_quantities/ip'). "
        "ids: Optional IDS prefix for relative paths. "
        "Returns: documentation, units, coordinates, data type, and cluster labels. "
        "Lists not-found paths with typo suggestions."
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
        units, data types, and cluster labels. Returns structured IdsNode objects enriched
        with LLM-generated cluster labels describing the physics context.

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
            - nodes: List of IdsNode objects with documentation, metadata, and cluster_labels
            - summary: Statistics about the retrieval operation
            - physics_domains: Aggregated physics domain information

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
        # Normalize paths to list using utility function
        paths_list = normalize_paths_input(paths)

        # Handle empty input with helpful error message
        if not paths_list:
            return FetchPathsResult(
                nodes=[],
                deprecated_paths=[],
                excluded_paths=[],
                summary={
                    "total_requested": 0,
                    "retrieved": 0,
                    "deprecated": 0,
                    "excluded": 0,
                    "not_found": 0,
                    "invalid": 0,
                    "physics_domains": [],
                },
                query="",
                search_mode=SearchMode.LEXICAL,
                physics_domains=[],
                error="No paths provided. Examples:\n"
                "  fetch_imas_paths('equilibrium/time_slice/global_quantities/ip')\n"
                "  fetch_imas_paths('time_slice/boundary/psi', ids='equilibrium')",
            )

        # Initialize tracking
        nodes = []
        deprecated_paths: list[DeprecatedPathInfo] = []
        excluded_paths: list[ExcludedPathInfo] = []
        not_found_paths: list[NotFoundPathInfo] = []
        found_count = 0
        not_found_count = 0
        deprecated_count = 0
        excluded_count = 0
        invalid_count = 0
        physics_domains = set()

        # Get cluster searcher for enriching paths with cluster labels
        searcher = self.cluster_searcher

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
                not_found_paths.append(
                    NotFoundPathInfo(
                        path=path,
                        reason="invalid_format",
                        suggestion="Path must contain '/' (e.g., 'equilibrium/time_slice').",
                    )
                )
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

                    # Enrich with cluster labels if available
                    if searcher:
                        cluster_labels = searcher.get_cluster_labels_for_path(path)
                        if cluster_labels:
                            node.cluster_labels = cluster_labels

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
                            # Get suggestion for typo correction
                            suggestion = self._get_suggestion_for_path(path)
                            not_found_paths.append(
                                NotFoundPathInfo(
                                    path=path,
                                    reason="path_not_exists",
                                    suggestion=suggestion,
                                )
                            )
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
            not_found_paths=not_found_paths,
            summary=summary,
            query=" ".join(paths_list[:3]) + ("..." if len(paths_list) > 3 else ""),
            search_mode=SearchMode.LEXICAL,  # Direct lookup, not search
            physics_domains=sorted(physics_domains),
        )

        logger.info(
            f"Path retrieval completed: {found_count}/{len(paths_list)} retrieved"
        )
        return result
