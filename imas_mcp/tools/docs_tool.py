"""
Documentation search tool implementation.

Provides documentation search capabilities using the docs-mcp-server proxy.
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp.models.request_models import ListDocsInput, SearchDocsInput
from imas_mcp.search.decorators import (
    cache_results,
    mcp_tool,
    measure_performance,
    validate_input,
)
from imas_mcp.services.docs_proxy_service import (
    DocsServerUnavailableError,
    LibraryNotFoundError,
)
from imas_mcp.services.docs_server_manager import DocsServerManager

logger = logging.getLogger(__name__)


class DocsTool:
    """Tool for documentation search with dependency injection."""

    def __init__(self, docs_manager: DocsServerManager):
        """Initialize the docs tool with injected docs manager.

        Args:
            docs_manager: Shared docs server manager instance
        """
        self.docs_manager = docs_manager
        self.logger = logger

    @cache_results(ttl=300, key_strategy="semantic")
    @validate_input(schema=SearchDocsInput)
    @measure_performance(include_metrics=True, slow_threshold=1.0)
    @mcp_tool(
        "Search indexed documentation libraries for specific topics, APIs, or concepts. "
        "Returns relevant documentation excerpts with URLs. "
        "Library parameter is required - use list_docs first to see available libraries. "
        "Examples: library='data-dictionary' query='time_slice equilibrium', library='imas-python' query='IDS class methods'. "
        "Supports optional version filtering for specific library versions."
    )
    async def search_docs(
        self,
        query: str,
        library: str | None = None,
        limit: int = 5,
        version: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Search indexed documentation libraries for specific topics, APIs, or concepts.

        Primary tool for finding documentation across multiple indexed libraries.
        Returns ranked results with content excerpts and source URLs.

        Args:
            query: Search query for documentation (e.g., "authentication", "configuration", "API usage")
            library: Library name to search (required - get from list_docs)
            limit: Maximum number of results to return (1-20, default: 5)
            version: Specific library version to search (optional, defaults to latest)
            ctx: FastMCP context

        Returns:
            Dictionary containing:
            - results: List of search results with content and URLs
            - count: Number of results returned
            - query, library, version: Search parameters
            - success: Boolean indicating success

        Note:
            Use list_docs tool first to discover available libraries.
            Library parameter is required for search.
        """

        # Basic parameter validation
        if not query or not query.strip():
            return {
                "error": "Query cannot be empty",
                "query": query,
                "library": library,
                "version": version,
                "validation_failed": True,
            }

        if limit < 1 or limit > 20:
            return {
                "error": "Limit must be between 1 and 20",
                "query": query,
                "library": library,
                "version": version,
                "validation_failed": True,
            }

        # The docs-mcp-server requires a library parameter
        if not library:
            # Get available libraries to help the user
            try:
                available_libraries = await self.docs_manager.proxy_list_libraries()
                return {
                    "error": "Library parameter is required for search",
                    "query": query,
                    "library": library,
                    "version": version,
                    "available_libraries": available_libraries,
                    "library_required": True,
                    "proxy_info": {
                        "method": "docs-mcp-server proxy",
                        "server_url": self.docs_manager.base_url,
                    },
                    "setup_instructions": True,
                }
            except Exception:
                return {
                    "error": "Library parameter is required for search",
                    "query": query,
                    "library": library,
                    "version": version,
                    "library_required": True,
                    "proxy_info": {
                        "method": "fallback",
                        "server_url": self.docs_manager.base_url,
                    },
                    "setup_instructions": True,
                }

        try:
            # Use proxy method for search
            result = await self.docs_manager.proxy_search_docs(
                query, library, version, limit
            )
            result["proxy_info"] = {
                "method": "docs-mcp-server proxy",
                "server_url": self.docs_manager.base_url,
            }
            return result
        except DocsServerUnavailableError as e:
            return {
                "error": str(e),
                "query": query,
                "library": library,
                "version": version,
                "setup_instructions": True,
            }
        except LibraryNotFoundError as e:
            return {
                "error": str(e),
                "query": query,
                "library": library,
                "version": version,
                "available_libraries": e.available_libraries,
                "library_not_found": True,
            }
        except Exception as e:
            # Basic error handling without IMAS-specific fallbacks
            return {
                "error": f"Search failed: {str(e)}",
                "query": query,
                "library": library,
                "version": version,
                "search_failed": True,
            }

    @cache_results(ttl=600, key_strategy="simple")
    @validate_input(schema=ListDocsInput)
    @measure_performance(include_metrics=True, slow_threshold=0.5)
    @mcp_tool(
        "List all indexed documentation libraries available for search. "
        "Returns library names that can be used with search_docs tool. "
        "Use this tool first before searching documentation to discover available libraries. "
        "Examples of returned libraries: 'data-dictionary', 'imas-python', 'testembeddings'. "
        "Returns library names only - for version-specific searches, use the version parameter in search_docs."
    )
    async def list_docs(
        self,
        library: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        List all indexed documentation libraries available for search.

        Discovery tool for finding what documentation libraries are available.
        Use before search_docs to know which libraries can be searched.

        Args:
            library: Specific library name to get info for (optional, currently unused)
            ctx: FastMCP context

        Returns:
            Dictionary containing:
            - libraries: List of available library names
            - count: Number of libraries available
            - success: Boolean indicating success
            - note: Information about version availability

        Examples:
            list_docs() → {"libraries": ["data-dictionary", "testembeddings"], "count": 2, "success": true}

        Note:
            Returns library names only. The MCP list_libraries tool does not provide
            version information. Use the version parameter in search_docs for version-specific searches.
        """

        # Basic parameter validation
        if library is not None and not library.strip():
            return {
                "error": "Library name cannot be empty",
                "library": library,
                "validation_failed": True,
            }

        try:
            if library:
                # Note: Library parameter currently unused - list_libraries MCP tool
                # does not support per-library queries
                # Return info suggesting to use search_docs for library-specific info
                return {
                    "library": library,
                    "note": "Use search_docs with this library name to explore available content",
                    "success": True,
                    "proxy_info": {
                        "method": "docs-mcp-server",
                        "server_url": self.docs_manager.base_url,
                    },
                }
            else:
                # List all available libraries using MCP proxy
                # Note: The MCP list_libraries tool only returns library names (not versions)
                libraries = await self.docs_manager.proxy_list_libraries()

                if not libraries:
                    return {
                        "error": "No libraries found in docs-mcp-server",
                        "libraries": [],
                        "count": 0,
                        "success": False,
                        "proxy_info": {
                            "method": "docs-mcp-server MCP tool",
                            "server_url": self.docs_manager.base_url,
                        },
                        "note": "The MCP list_libraries tool returns library names only. For version info, query individual libraries.",
                    }

                return {
                    "libraries": libraries,
                    "count": len(libraries),
                    "success": True,
                    "proxy_info": {
                        "method": "docs-mcp-server MCP tool",
                        "server_url": self.docs_manager.base_url,
                    },
                    "note": "This list contains library names only. For version details, query individual libraries.",
                }
        except DocsServerUnavailableError as e:
            # Graceful fallback when docs-mcp-server is not available
            return {
                "error": str(e),
                "libraries": [],
                "library": library,
                "server_status": "unavailable",
            }
        except Exception as e:
            # Catch-all for any other unexpected errors
            return {
                "error": f"Unexpected error: {str(e)}",
                "libraries": [],
                "library": library,
                "unexpected_error": True,
            }
