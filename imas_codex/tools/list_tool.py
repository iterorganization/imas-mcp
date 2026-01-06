"""
List tool implementation for lightweight path enumeration.

Provides minimal-overhead path listing with multiple output formats optimized for LLM consumption.
"""

import logging
from typing import Any

from fastmcp import Context

from imas_codex.models.error_models import ToolError
from imas_codex.models.request_models import ListPathsInput
from imas_codex.models.result_models import ListPathsResult, ListPathsResultItem
from imas_codex.search.decorators import (
    handle_errors,
    mcp_tool,
    measure_performance,
    validate_input,
)

from .base import BaseTool

logger = logging.getLogger(__name__)


class ListTool(BaseTool):
    """Tool for lightweight IMAS path listing."""

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "list_tool"

    def _build_json_tree(self, paths: list[str]) -> str:
        """
        Build JSON-formatted tree structure from paths.

        Args:
            paths: List of full paths (e.g., "equilibrium/time_slice/boundary/psi")

        Returns:
            JSON-formatted string with hierarchical structure
        """
        import json

        if not paths:
            return "{}"

        # Build tree structure
        tree: dict[str, Any] = {}

        for path in paths:
            parts = path.split("/")
            current = tree

            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]

        return json.dumps(tree, indent=2)

    def _build_dict_tree(self, paths: list[str]) -> dict[str, Any]:
        """
        Build dictionary tree structure from paths.

        Args:
            paths: List of full paths (e.g., "equilibrium/time_slice/boundary/psi")

        Returns:
            Nested dictionary with hierarchical structure
        """
        if not paths:
            return {}

        # Build tree structure
        tree: dict[str, Any] = {}

        for path in paths:
            parts = path.split("/")
            current = tree

            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]

        return tree

    def _build_yaml_tree(self, paths: list[str], show_leaf_only: bool = False) -> str:
        """
        Build YAML-formatted tree structure from paths.

        Args:
            paths: List of full paths (e.g., "equilibrium/time_slice/boundary/psi")
            show_leaf_only: If True, only show leaf nodes (no intermediate structure)

        Returns:
            YAML-formatted string with clean indentation (no visual tree characters)
        """
        if not paths:
            return ""

        # Build tree structure
        tree: dict[str, Any] = {}

        for path in paths:
            parts = path.split("/")
            current = tree

            for i, part in enumerate(parts):
                is_leaf = i == len(parts) - 1

                if part not in current:
                    if show_leaf_only and not is_leaf:
                        # Skip intermediate nodes in leaf-only mode
                        current = current.setdefault(part, {})
                    else:
                        current[part] = {}

                if not show_leaf_only or is_leaf:
                    current = current[part]

        # Convert tree to YAML format (clean indentation, no markers)
        def tree_to_yaml(node: dict, indent: int = 0) -> list[str]:
            """Convert tree dictionary to YAML-style indented lines."""
            lines = []
            items = sorted(node.items())

            for key, children in items:
                # Add current node
                lines.append("  " * indent + key)

                # Recursively add children
                if isinstance(children, dict) and children:
                    lines.extend(tree_to_yaml(children, indent + 1))

            return lines

        yaml_lines = tree_to_yaml(tree)
        return "\n".join(yaml_lines)

    def _build_flat_list(self, paths: list[str]) -> list[str]:
        """Build simple flat list of paths."""
        return sorted(paths)

    def _filter_paths_by_prefix(
        self, all_paths: list[str], prefix_filter: str | None
    ) -> list[str]:
        """
        Filter paths to only include those under a specific prefix.

        Args:
            all_paths: All available paths for the IDS
            prefix_filter: Prefix to filter by (e.g., "time_slice/global_quantities")

        Returns:
            Filtered list of paths
        """
        if not prefix_filter:
            return all_paths

        # Ensure prefix doesn't have leading/trailing slashes for consistent matching
        prefix_filter = prefix_filter.strip("/")

        filtered = []
        for path in all_paths:
            # Check if path starts with the prefix
            # Handle both exact matches and children
            if path == prefix_filter or path.startswith(prefix_filter + "/"):
                filtered.append(path)

        return filtered

    def _extract_paths_from_documents(
        self,
        documents: list,
        ids_name: str,
        include_ids_prefix: bool,
        leaf_only: bool,
    ) -> list[str]:
        """
        Extract path strings from documents.

        Args:
            documents: List of document objects
            ids_name: Name of the IDS
            include_ids_prefix: Whether to include IDS name in paths
            leaf_only: Whether to include only leaf nodes

        Returns:
            List of path strings
        """
        paths = []

        for doc in documents:
            path = doc.metadata.path_name

            # Remove IDS prefix if requested
            if not include_ids_prefix and path.startswith(f"{ids_name}/"):
                path = path[len(ids_name) + 1 :]

            # Filter leaf nodes if requested
            if leaf_only:
                # Check if this is a leaf by seeing if no other path starts with this path + "/"
                is_leaf = not any(
                    d.metadata.path_name.startswith(doc.metadata.path_name + "/")
                    for d in documents
                    if d != doc
                )
                if not is_leaf:
                    continue

            paths.append(path)

        return sorted(paths)

    @measure_performance(include_metrics=False, slow_threshold=0.5)
    @validate_input(schema=ListPathsInput)
    @handle_errors(fallback=None)
    @mcp_tool(
        "List data paths in IDS with minimal overhead (paths only, no descriptions). "
        "paths (required): Space-separated IDS names or path prefixes (e.g., 'equilibrium', 'magnetics/flux_loop'). "
        "format: 'yaml' (default, token-efficient tree), 'list' (flat array), 'json', 'dict'. "
        "leaf_only: True returns only data fields, not intermediate structure nodes. "
        "max_paths: Limit output size. "
        "Use for structure exploration; use fetch_imas_paths for documentation."
    )
    async def list_imas_paths(
        self,
        paths: str,
        format: str = "yaml",
        leaf_only: bool = False,
        include_ids_prefix: bool = True,
        max_paths: int | None = None,
        ctx: Context | None = None,
    ) -> ListPathsResult | ToolError:
        """
        List all data paths in one or more IDS or path prefixes with minimal overhead.

        Lightweight enumeration tool optimized for LLM consumption. Returns only path
        strings without documentation, units, or other metadata. Supports multiple
        output formats and can list entire IDS or filter by path prefix.

        Args:
            paths: Space-delimited IDS names or path prefixes. Examples:
                  - IDS names: "equilibrium magnetics thomson_scattering"
                  - Path prefixes: "equilibrium/time_slice/global_quantities"
                  - Mixed: "equilibrium/time_slice magnetics/flux_loop"
                  Each token is processed independently - if it contains '/', treated as
                  path prefix; otherwise treated as IDS name.

            format: Output format for paths (default: "yaml"):
                  - "yaml": YAML-style indented tree (DEFAULT - most token-efficient, human-friendly)
                  - "list": Array of path strings, one per element (easy parsing, copy-paste ready)
                  - "json": JSON string tree (structured text format)
                  - "dict": Structured dictionary tree (best for programmatic use)

            leaf_only: If True, return only leaf nodes (actual data paths), not
                      intermediate structure nodes (default: False)

            include_ids_prefix: If True, include IDS name in paths (default: True).
                               Example: True → "equilibrium/time_slice/..."
                                       False → "time_slice/..."

            max_paths: Maximum number of paths to return per IDS/prefix to prevent
                      overwhelming output (default: format-dependent)
                      - dict format: default 5000 (native objects, most efficient)
                      - list format: default 3000 (simple text)
                      - json format: default 5000 (30-40% token reduction)
                      - yaml format: default 5000 (40-50% token reduction)

            ctx: FastMCP context for potential future enhancements

        Returns:
            ListPathsResult containing:
            - format: The output format used
            - results: List of ListPathsResultItem for each queried IDS/prefix
            - summary: Overall statistics across all queries

        Examples:
            List multiple IDS in YAML format (default, most token-efficient):
                list_imas_paths("equilibrium magnetics")
                → Returns YAML indented tree for both IDS

            List specific subtree:
                list_imas_paths("equilibrium/time_slice/global_quantities")
                → Returns only paths under that prefix

            Simple array of leaf paths only:
                list_imas_paths("equilibrium", format="list", leaf_only=True, include_ids_prefix=False)
                → Returns:
                  ["time_slice/boundary/psi", "time_slice/boundary/psi_norm", ...]

            Multiple prefixes:
                list_imas_paths("equilibrium/time_slice magnetics/flux_loop")
                → Returns paths for both specific subtrees

        Note:
            This tool is optimized for minimal token usage. For full documentation
            and metadata, use fetch_imas_paths. For path discovery by concept,
            use search_imas.
        """
        # Input validation is handled by @validate_input decorator
        # Parse input - split by spaces
        queries = paths.strip().split()

        if not queries:
            return self._create_error_response(
                "No IDS or path prefix specified. "
                "Provide IDS names like 'equilibrium magnetics' or path prefixes like "
                "'equilibrium/time_slice'. Use get_imas_overview() to see available IDS.",
                paths,
            )

        # After validation, max_paths should be set to a default if None
        # This is a safety check since validation should handle it
        if max_paths is None:
            # Default based on format
            format_defaults = {
                "yaml": 5000,
                "list": 3000,
                "json": 5000,
                "dict": 5000,
            }
            max_paths = format_defaults.get(format, 5000)

        results = []
        total_paths = 0
        total_truncated = 0

        for query in queries:
            query = query.strip()
            if not query:
                continue

            try:
                # Determine if this is an IDS name or path prefix
                if "/" in query:
                    # Path prefix - extract IDS name and filter
                    parts = query.split("/", 1)
                    ids_name = parts[0]
                    prefix_filter = parts[1] if len(parts) > 1 else None
                else:
                    # Just an IDS name
                    ids_name = query
                    prefix_filter = None

                # Validate IDS exists
                valid_ids, invalid_ids = await self.documents.validate_ids([ids_name])
                if not valid_ids:
                    results.append(
                        ListPathsResultItem(
                            query=query,
                            error=f"IDS '{ids_name}' not found",
                            path_count=0,
                        )
                    )
                    continue

                # Get all documents for this IDS
                ids_documents = await self.documents.get_documents_safe(ids_name)

                if not ids_documents:
                    results.append(
                        ListPathsResultItem(
                            query=query,
                            error=f"No paths found in IDS '{ids_name}'",
                            path_count=0,
                        )
                    )
                    continue

                # Extract paths from documents
                all_paths = self._extract_paths_from_documents(
                    ids_documents, ids_name, include_ids_prefix, leaf_only
                )

                # Apply prefix filter if specified
                if prefix_filter:
                    # For prefix filtering, we need to adjust the filter based on include_ids_prefix
                    if include_ids_prefix:
                        filter_prefix = f"{ids_name}/{prefix_filter}"
                    else:
                        filter_prefix = prefix_filter

                    all_paths = self._filter_paths_by_prefix(all_paths, filter_prefix)

                # Check if we need to truncate
                truncated = len(all_paths) > max_paths
                paths_to_show = all_paths[:max_paths]

                # Build result using Pydantic model
                result = ListPathsResultItem(
                    query=query,
                    path_count=len(all_paths),
                    truncated_to=max_paths if truncated else None,
                )

                if truncated:
                    total_truncated += 1

                # Generate output based on format
                if format == "dict":
                    result.paths = self._build_dict_tree(paths_to_show)
                elif format == "json":
                    result.paths = self._build_json_tree(paths_to_show)
                elif format == "yaml":
                    result.paths = self._build_yaml_tree(paths_to_show, leaf_only)
                elif format == "list":
                    result.paths = self._build_flat_list(paths_to_show)

                results.append(result)
                total_paths += len(all_paths)

                logger.info(
                    f"Listed {len(all_paths)} paths for '{query}' "
                    f"(format={format}, leaf_only={leaf_only})"
                )

            except Exception as e:
                logger.error(f"Error listing paths for '{query}': {e}")
                results.append(
                    ListPathsResultItem(
                        query=query,
                        error=str(e),
                        path_count=0,
                    )
                )

        # Build summary
        summary = {
            "total_queries": len(queries),
            "successful_queries": len([r for r in results if not r.error]),
            "total_paths": total_paths,
            "format": format,
            "leaf_only": leaf_only,
            "truncated_results": total_truncated,
        }

        return ListPathsResult(
            format=format,
            results=results,
            summary=summary,
        )
