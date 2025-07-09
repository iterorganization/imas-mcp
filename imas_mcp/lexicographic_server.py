"""
DEPRECATED: This module is deprecated and will be removed in a future version.

The lexicographic server has been replaced by the AI-enhanced server (server.py).
Please use the new server.py module instead, which provides the same functionality
with enhanced AI capabilities and better performance.

For migration guidance, see the documentation or contact the development team.
"""

import warnings

# Standard library imports
import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Annotated, List, Union, Optional, Set

import nest_asyncio

# Third-party imports
from fastmcp import FastMCP
from pydantic import Field

# Local imports
from imas_mcp.lexicographic_search import LexicographicSearch

# apply nest_asyncio to allow nested event loops
# This is necessary for Jupyter notebooks and some other environments
# that don't support nested event loops by default.
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Server:
    """IMAS MCP Server with configurable search index management."""

    ids_set: Optional[Set[str]] = None
    auto_build: bool = False
    mcp: FastMCP = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the MCP server after dataclass initialization."""
        warnings.warn(
            "The lexicographic_server module is deprecated. "
            "Please use server.py (AI-enhanced server) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.mcp = FastMCP("IMAS")
        self._register_tools()

    @cached_property
    def lexicographic_index(self) -> LexicographicSearch:
        """Return a lexicographic search index instance."""
        logger.info(
            f"Initializing search index with ids_set: {self.ids_set}, auto_build: {self.auto_build}"
        )
        index = LexicographicSearch(ids_set=self.ids_set, auto_build=self.auto_build)

        if len(index) == 0:
            raise ValueError(
                "Lexicographic index is empty. This could be due to:\n"
                "1. auto_build=False and no pre-built index exists\n"
                "2. Invalid ids_set specified\n"
                "3. Data dictionary not properly installed\n"
                "Try setting auto_build=True or check your ids_set configuration."
            )

        return index

    def _register_tools(self):
        """Register all MCP tools with the server."""

        # Register lexicographic index tools
        self.mcp.tool(self.ids_names)
        self.mcp.tool(self.ids_info)
        self.mcp.tool(self.search_by_keywords)
        self.mcp.tool(self.search_by_exact_path)
        self.mcp.tool(self.search_by_path_prefix)
        self.mcp.tool(self.filter_search_results)
        self.mcp.tool(self.get_index_stats)
        self.mcp.tool(self.get_ids_structure)
        self.mcp.tool(self.get_common_units)

        # Register semantic search tools
        # TODO register semantic search tools once implemented

    def ids_names(self) -> List[str]:
        """Return a list of IDS names available in the Data Dictionary.

        Returns:
            A list of IMAS IDS (Interface Data Structure) names that can be
            searched and queried through this server.
        """
        return self.lexicographic_index.ids_names

    def ids_info(self) -> dict[str, Union[str, int, List[str]]]:
        """Return high-level information about the IMAS Data Dictionary.

        Returns:
            A dictionary containing metadata about the Data Dictionary including
            version, total IDS count, and available IDS names.
        """
        return {
            "version": str(self.lexicographic_index.dd_version),
            "total_ids_count": len(self.lexicographic_index.ids_names),
            "available_ids": self.lexicographic_index.ids_names,
            "index_type": self.lexicographic_index.index_prefix,
            "total_documents": len(self.lexicographic_index),
        }

    def search_by_keywords(
        self,
        query_str: Annotated[
            str,
            Field(
                description="Natural language search query. Supports field prefixes "
                "(e.g., 'documentation:plasma'), wildcards (e.g., 'core*'), "
                "boolean operators (AND, OR, NOT), and phrases in quotes."
            ),
        ],
        page_size: Annotated[
            int,
            Field(
                default=10,
                ge=1,
                le=100,
                description="Maximum number of results to return (1-100)",
            ),
        ] = 10,
        page: Annotated[
            int,
            Field(default=1, ge=1, description="Page number for pagination (1-based)"),
        ] = 1,
        enable_fuzzy: Annotated[
            bool,
            Field(
                default=False,
                description="Enable fuzzy matching for typos and approximate matches",
            ),
        ] = False,
        search_fields: Annotated[
            Union[List[str], None],
            Field(
                default=None,
                description="List of fields to search in (defaults to documentation and path_segments)",
            ),
        ] = None,
        sort_by: Annotated[
            Union[str, None],
            Field(default=None, description="Field name to sort results by"),
        ] = None,
        sort_reverse: Annotated[
            bool, Field(default=False, description="Whether to reverse the sort order")
        ] = False,
    ) -> List[dict]:
        """Search the IMAS Data Dictionary by keywords.

        Performs full-text search across IMAS Data Dictionary documentation and paths.
        Supports advanced query syntax including field-specific searches, wildcards,
        boolean operators, and fuzzy matching.

        Args:
            query_str: Search query with optional field prefixes and operators
            page_size: Number of results per page (1-100)
            page: Page number for pagination
            enable_fuzzy: Enable fuzzy matching for typos
            search_fields: Fields to search in (defaults to documentation and path_segments)
            sort_by: Field to sort results by
            sort_reverse: Whether to reverse sort order

        Returns:
            List of search results containing path, documentation, units, and metadata.

        Examples:
            - Basic search: "plasma current"
            - Field-specific: "documentation:temperature ids:core_profiles"
            - Wildcards: "core_profiles/prof*"
            - Boolean: "density AND NOT temperature"            - Phrases: "ion temperature"
        """
        if search_fields is None:
            search_fields = ["documentation", "path_segments"]

        results = self.lexicographic_index.search_by_keywords(
            query_str=query_str,
            page_size=page_size,
            page=page,
            fuzzy=enable_fuzzy,
            search_fields=search_fields,
            sort_by=sort_by,
            sort_reverse=sort_reverse,
        )

        return [result.model_dump() for result in results]

    def search_by_exact_path(
        self,
        path_value: Annotated[
            str,
            Field(
                description="The exact IDS path to retrieve (e.g., 'core_profiles/profiles_1d/temperature')"
            ),
        ],
    ) -> Union[dict, None]:
        """Return documentation and metadata for an exact IDS path lookup.

        Performs an exact path match to retrieve a specific entry from the IMAS
        Data Dictionary. Useful when you know the precise path you want to query.

        Args:
            path_value: The exact path of the document to retrieve

        Returns:
            A dictionary containing the search result with path, documentation,
            units, and metadata if found, otherwise None.

        Examples:
            - "core_profiles/profiles_1d/temperature"
            - "equilibrium/time_slice/boundary/outline/r"
            - "pf_active/coil/name" """
        result = self.lexicographic_index.search_by_exact_path(path_value)
        return result.model_dump() if result else None

    def search_by_path_prefix(
        self,
        path_prefix: Annotated[
            str,
            Field(
                description="The path prefix to search for (e.g., 'core_profiles/profiles_1d')"
            ),
        ],
        page_size: Annotated[
            int,
            Field(
                default=10,
                ge=1,
                le=100,
                description="Maximum number of results to return (1-100)",
            ),
        ] = 10,
        page: Annotated[
            int,
            Field(default=1, ge=1, description="Page number for pagination (1-based)"),
        ] = 1,
        sort_by: Annotated[
            Union[str, None],
            Field(default=None, description="Field name to sort results by"),
        ] = None,
        sort_reverse: Annotated[
            bool, Field(default=False, description="Whether to reverse the sort order")
        ] = False,
    ) -> List[dict]:
        """Return all entries matching a given IDS path prefix.

        Searches for all paths that start with the specified prefix. Useful for
        exploring the hierarchical structure of IMAS data and finding all
        sub-elements under a particular path.

        Args:
            path_prefix: The prefix of the path to search for
            page_size: Number of results per page (1-100)
            page: Page number for pagination
            sort_by: Field to sort results by
            sort_reverse: Whether to reverse sort order

        Returns:
            List of search results containing all paths that match the prefix.

        Examples:
            - "core_profiles" - Returns all core_profiles paths
            - "core_profiles/profiles_1d" - Returns all 1D profile data paths
            - "equilibrium/time_slice" - Returns all equilibrium time slice paths
        """
        results = self.lexicographic_index.search_by_path_prefix(
            path_prefix=path_prefix,
            page_size=page_size,
            page=page,
            sort_by=sort_by,
            sort_reverse=sort_reverse,
        )

        return [result.model_dump() for result in results]

    def filter_search_results(
        self,
        search_query: Annotated[
            str, Field(description="Initial search query to get base results to filter")
        ],
        filters: Annotated[
            dict[str, str],
            Field(
                description="Dictionary of field names and values to filter by "
                "(e.g., {'ids_name': 'core_profiles', 'units': 'm'})"
            ),
        ],
        enable_regex: Annotated[
            bool,
            Field(
                default=True,
                description="Enable regex pattern matching in filter values",
            ),
        ] = True,
        page_size: Annotated[
            int,
            Field(
                default=10,
                ge=1,
                le=100,
                description="Maximum number of results to return (1-100)",
            ),
        ] = 10,
    ) -> List[dict]:
        """Filter search results based on field values with optional regex support.

        First performs a keyword search, then filters the results based on specific
        field criteria. Supports both exact matching and regex pattern matching for
        advanced filtering capabilities.

        Args:
            search_query: Initial search query to get base results
            filters: Dictionary where keys are field names (path, documentation,
                    units, ids_name) and values are the desired filter criteria
            enable_regex: Enable regex pattern matching for filter values
            page_size: Maximum number of filtered results to return

        Returns:
            List of filtered search results that match all specified criteria.

        Examples:
            - Basic filter: search_query="temperature", filters={"ids_name": "core_profiles"}
            - Regex filter: search_query="density", filters={"path": ".*profiles_1d.*"}
            - Multi-field: search_query="current", filters={"units": "A", "ids_name": "pf_active"}
        """  # First get initial results from keyword search
        initial_results = self.lexicographic_index.search_by_keywords(
            query_str=search_query,
            page_size=100,  # Get more results initially to have enough to filter
            page=1,
        )  # Then filter those results
        filtered_results = self.lexicographic_index.filter_search_results(
            search_results=initial_results,
            filters=filters,
            regex=enable_regex,
        )

        # Limit to requested page size
        limited_results = filtered_results[:page_size]

        return [result.model_dump() for result in limited_results]

    def get_index_stats(self) -> dict[str, Union[str, int, List[str]]]:
        """Return statistics and metadata about the search index.

        Provides information about the current state of the search index including
        document counts, index configuration, and available fields.

        Returns:
            Dictionary containing index statistics and metadata including document
            count, index name, schema fields, and configuration details."""
        schema_fields = list(self.lexicographic_index.resolved_schema.names())

        return {
            "total_documents": len(self.lexicographic_index),
            "index_name": self.lexicographic_index.indexname or "unknown",
            "index_type": self.lexicographic_index.index_prefix,
            "data_dictionary_version": str(self.lexicographic_index.dd_version),
            "schema_fields": schema_fields,
            "index_directory": str(self.lexicographic_index.dirname),
            "available_ids_count": len(self.lexicographic_index.ids_names),
        }

    def get_ids_structure(
        self,
        ids_name: Annotated[
            str,
            Field(
                description="Name of the IDS to explore (e.g., 'core_profiles', 'equilibrium')"
            ),
        ],
        max_depth: Annotated[
            int,
            Field(
                default=3,
                ge=1,
                le=10,
                description="Maximum depth level to explore (1-10)",
            ),
        ] = 3,
        page_size: Annotated[
            int,
            Field(
                default=50,
                ge=1,
                le=200,
                description="Maximum number of paths to return (1-200)",
            ),
        ] = 50,
    ) -> dict:
        """Explore the hierarchical structure of a specific IDS.

        Returns the hierarchical structure of an IDS showing all available paths
        up to a specified depth. Useful for understanding the organization and
        available data within a particular IDS.

        Args:
            ids_name: Name of the IDS to explore
            max_depth: Maximum depth level to traverse
            page_size: Maximum number of paths to return

        Returns:
            Dictionary containing the IDS structure with paths organized by depth
            level, total path count, and metadata.

        Examples:
            - ids_name="core_profiles", max_depth=2
            - ids_name="equilibrium", max_depth=3
            - ids_name="pf_active", max_depth=1
        """  # Get all paths for this IDS
        all_results = self.lexicographic_index.search_by_path_prefix(
            path_prefix=ids_name,
            page_size=page_size,
            page=1,
            sort_by="path",
        )  # Organize results by depth
        structure_by_depth = {}
        for result in all_results:
            path = result.path
            depth = path.count("/")

            if depth <= max_depth:
                if depth not in structure_by_depth:
                    structure_by_depth[depth] = []

                structure_by_depth[depth].append(
                    {
                        "path": path,
                        "documentation": result.documentation[:100] + "..."
                        if len(result.documentation) > 100
                        else result.documentation,
                        "units": result.units,
                        "depth": depth,
                    }
                )

        return {
            "ids_name": ids_name,
            "max_depth_explored": max_depth,
            "total_paths_found": len(all_results),
            "structure_by_depth": structure_by_depth,
            "depth_summary": {
                str(depth): len(paths) for depth, paths in structure_by_depth.items()
            },
        }

    def get_common_units(self) -> dict:
        """Get a summary of the most commonly used units in the Data Dictionary.

        Analyzes all indexed documents to provide statistics on unit usage across
        the IMAS Data Dictionary. Useful for understanding the measurement systems
        and units used throughout IMAS.

        Returns:
            Dictionary containing unit usage statistics including most common units,
            total unique units count, and unit categories.
        """  # Get a large sample of results to analyze units
        sample_results = self.lexicographic_index.search_by_keywords(
            query_str="*",  # Match everything
            page_size=100,
            page=1,
        )

        # Count unit occurrences
        unit_counts = {}
        for result in sample_results:
            unit = result.units
            if unit and unit != "none":
                unit_counts[unit] = unit_counts.get(unit, 0) + 1

        # Sort by frequency
        sorted_units = sorted(unit_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_unique_units": len(unit_counts),
            "most_common_units": [
                {"unit": unit, "count": count} for unit, count in sorted_units[:20]
            ],
            "all_units": list(unit_counts.keys()),
            "sample_size": len(sample_results),
            "dimensionless_count": sum(
                1
                for result in sample_results
                if result.units in ["none", "", "1", "dimensionless"]
            ),
        }

    def run(
        self, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000
    ) -> None:
        """Run the MCP server with the configured search index.

        Args:
            transport: Transport protocol to use
            host: Host to bind to (for sse and streamable-http transports)
            port: Port to bind to (for sse and streamable-http transports)
        """
        try:
            match transport:
                case "stdio":
                    self.mcp.run(transport="stdio")
                case "sse":
                    self.mcp.run(transport=transport, host=host, port=port)
                case "streamable-http":
                    self._run_http_with_health(host=host, port=port)
        except KeyboardInterrupt:
            logger.info("Stopping MCP server...")

    def _run_http_with_health(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the MCP server with streamable-http transport and add health endpoint.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        try:
            # Get the FastMCP ASGI app
            app = self.mcp.http_app()

            # Add health endpoint using Starlette routing
            from starlette.responses import JSONResponse
            from starlette.routing import Route
        except ImportError as e:
            raise ImportError(
                "HTTP transport requires additional dependencies. "
                "Install with: pip install imas-mcp[http]"
            ) from e

        async def health_endpoint(request):
            """Health check endpoint that verifies the search index is accessible."""
            try:
                # Verify the search index is working
                index_stats = self.get_index_stats()

                return JSONResponse(
                    {
                        "status": "healthy",
                        "service": "imas-mcp-server",
                        "version": self._get_version(),
                        "index_stats": {
                            "total_paths": index_stats.get("total_paths", 0),
                            "index_name": index_stats.get("index_name", "unknown"),
                        },
                        "transport": "streamable-http",
                    }
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return JSONResponse(
                    {
                        "status": "unhealthy",
                        "service": "imas-mcp-server",
                        "error": str(e),
                        "transport": "streamable-http",
                    },
                    status_code=503,
                )  # Add the health route to the existing app

        health_route = Route("/health", health_endpoint, methods=["GET"])
        app.routes.append(health_route)

        logger.info(
            f"Starting MCP server with health endpoint at http://{host}:{port}/health"
        )

        # Run with uvicorn
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "HTTP transport requires additional dependencies. "
                "Install with: pip install imas-mcp[http]"
            ) from e

        uvicorn.run(app, host=host, port=port, log_level="info")

    def _get_version(self) -> str:
        """Get the package version."""
        try:
            import importlib.metadata

            return importlib.metadata.version("imas-mcp-server")
        except Exception:
            return "unknown"


if __name__ == "__main__":
    # Default server creation for basic testing
    server = Server(auto_build=True)
    server.run()
