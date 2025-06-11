# filepath: c:\Users\mcintos\Code\imas-mcp-server\imas_mcp_server\mcp_imas.py
# Standard library imports
import logging
from typing import Annotated, List, Union

import click
import nest_asyncio

# Third-party imports
from fastmcp import FastMCP
from pydantic import Field

# Local imports
from imas_mcp_server.lexicographic_search import LexicographicSearch

# apply nest_asyncio to allow nested event loops
# This is necessary for Jupyter notebooks and some other environments
# that don't support nested event loops by default.
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize shared index instance
_search_index = LexicographicSearch()

# Initialize MCP server
mcp = FastMCP("IMAS")


@mcp.tool
def ids_names() -> List[str]:
    """Return a list of IDS names available in the Data Dictionary.

    Returns:
        A list of IMAS IDS (Interface Data Structure) names that can be
        searched and queried through this server.
    """
    return _search_index.ids_names


@mcp.tool
def ids_info() -> dict[str, Union[str, int, List[str]]]:
    """Return high-level information about the IMAS Data Dictionary.

    Returns:
        A dictionary containing metadata about the Data Dictionary including
        version, total IDS count, and available IDS names.
    """
    return {
        "version": str(_search_index.dd_version),
        "total_ids_count": len(_search_index.ids_names),
        "available_ids": _search_index.ids_names,
        "index_type": _search_index.index_prefix,
        "total_documents": len(_search_index),
    }


@mcp.tool
def search_by_keywords(
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
        int, Field(default=1, ge=1, description="Page number for pagination (1-based)")
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
        - Boolean: "density AND NOT temperature"
        - Phrases: "ion temperature"
    """
    if search_fields is None:
        search_fields = ["documentation", "path_segments"]

    results = _search_index.search_by_keywords(
        query_str=query_str,
        page_size=page_size,
        page=page,
        fuzzy=enable_fuzzy,
        search_fields=search_fields,
        sort_by=sort_by,
        sort_reverse=sort_reverse,
    )

    return [result.model_dump() for result in results]


@mcp.tool
def search_by_exact_path(
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
        - "pf_active/coil/name"
    """
    result = _search_index.search_by_exact_path(path_value)
    return result.model_dump() if result else None


@mcp.tool
def search_by_path_prefix(
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
        int, Field(default=1, ge=1, description="Page number for pagination (1-based)")
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
    results = _search_index.search_by_path_prefix(
        path_prefix=path_prefix,
        page_size=page_size,
        page=page,
        sort_by=sort_by,
        sort_reverse=sort_reverse,
    )

    return [result.model_dump() for result in results]


@mcp.tool
def filter_search_results(
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
            default=True, description="Enable regex pattern matching in filter values"
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
    """
    # First get initial results from keyword search
    initial_results = _search_index.search_by_keywords(
        query_str=search_query,
        page_size=100,  # Get more results initially to have enough to filter
        page=1,
    )

    # Then filter those results
    filtered_results = _search_index.filter_search_results(
        search_results=initial_results,
        filters=filters,
        regex=enable_regex,
    )

    # Limit to requested page size
    limited_results = filtered_results[:page_size]

    return [result.model_dump() for result in limited_results]


@mcp.tool
def get_index_stats() -> dict[str, Union[str, int, List[str]]]:
    """Return statistics and metadata about the search index.

    Provides information about the current state of the search index including
    document counts, index configuration, and available fields.

    Returns:
        Dictionary containing index statistics and metadata including document
        count, index name, schema fields, and configuration details.
    """
    schema_fields = list(_search_index.resolved_schema.names())

    return {
        "total_documents": len(_search_index),
        "index_name": _search_index.indexname or "unknown",
        "index_type": _search_index.index_prefix,
        "data_dictionary_version": str(_search_index.dd_version),
        "schema_fields": schema_fields,
        "index_directory": str(_search_index.dirname),
        "available_ids_count": len(_search_index.ids_names),
    }


@mcp.tool
def get_ids_structure(
    ids_name: Annotated[
        str,
        Field(
            description="Name of the IDS to explore (e.g., 'core_profiles', 'equilibrium')"
        ),
    ],
    max_depth: Annotated[
        int,
        Field(
            default=3, ge=1, le=10, description="Maximum depth level to explore (1-10)"
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
    """
    # Get all paths for this IDS
    all_results = _search_index.search_by_path_prefix(
        path_prefix=ids_name,
        page_size=page_size,
        page=1,
        sort_by="path",
    )

    # Organize results by depth
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


@mcp.tool
def get_common_units() -> dict:
    """Get a summary of the most commonly used units in the Data Dictionary.

    Analyzes all indexed documents to provide statistics on unit usage across
    the IMAS Data Dictionary. Useful for understanding the measurement systems
    and units used throughout IMAS.

    Returns:
        Dictionary containing unit usage statistics including most common units,
        total unique units count, and unit categories.
    """
    # Get a large sample of results to analyze units
    sample_results = _search_index.search_by_keywords(
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


@click.command()
@click.option(
    "--transport",
    default="stdio",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    help="Transport protocol to use (stdio, sse, or streamable-http)",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (for sse and streamable-http transports)",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind to (for sse and streamable-http transports)",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
def run_server(transport: str, host: str, port: int, log_level: str) -> None:
    """Run the MCP server with configurable transport options.

    Examples:
        # Run with default STDIO transport
        run-server

        # Run with SSE transport on custom host/port
        run-server --transport sse --host 0.0.0.0 --port 9000

        # Run with debug logging
        run-server --log-level DEBUG

        # Run with streamable-http transport
        run-server --transport streamable-http --port 8080
    """
    # Configure logging based on the provided level
    logging.basicConfig(level=getattr(logging, log_level))
    logger.info(f"Starting MCP server with transport={transport}")

    match transport:
        case "stdio":
            logger.info("Using STDIO transport")
        case _:
            logger.info(f"Using {transport} transport on {host}:{port}")

    try:
        # Build kwargs for mcp.run()
        match transport:
            case "stdio":
                mcp.run(transport="stdio")
            case "sse" | "streamable-http":
                mcp.run(transport=transport, host=host, port=port)
    except KeyboardInterrupt:
        logger.info("Stopping MCP server...")


if __name__ == "__main__":
    run_server()
