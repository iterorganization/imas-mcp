"""
Documentation search tool implementation.

Provides documentation search capabilities using the docs-mcp-server proxy.
"""

import logging

from fastmcp import Context

from imas_codex.models.request_models import ListDocsInput, SearchDocsInput
from imas_codex.models.result_models import (
    ListDocsResult,
    SearchDocsResult,
    SearchDocsResultItem,
)
from imas_codex.search.decorators import (
    cache_results,
    mcp_tool,
    measure_performance,
    validate_input,
)
from imas_codex.services.docs_proxy_service import (
    DocsServerUnavailableError,
    LibraryNotFoundError,
)
from imas_codex.services.docs_server_manager import DocsServerManager

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
        "Search documentation for IMAS, Python, and other indexed libraries. "
        "Use this to find code examples, API references, and guides. "
        "Returns relevant documentation excerpts with URLs. "
        "Library parameter is required - use list_imas_docs first to see available libraries. "
        "Examples: library='data-dictionary' query='time_slice equilibrium', library='imas-python' query='IDS class methods'. "
        "Supports optional version filtering for specific library versions."
    )
    async def search_imas_docs(
        self,
        query: str,
        library: str | None = None,
        limit: int = 5,
        version: str | None = None,
        ctx: Context | None = None,
    ) -> SearchDocsResult:
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
            SearchDocsResult with:
            - results: List of SearchDocsResultItem with content and URLs
            - count: Number of results returned
            - query, library, version: Search parameters
            - success: Boolean indicating success

        Note:
            Use list_imas_docs tool first to discover available libraries.
            Library parameter is required for search.
        """

        # Basic parameter validation
        if not query or not query.strip():
            return SearchDocsResult(
                query=query,
                library=library,
                version=version,
                success=False,
                error="Query cannot be empty",
            )

        if limit < 1 or limit > 20:
            return SearchDocsResult(
                query=query,
                library=library,
                version=version,
                success=False,
                error="Limit must be between 1 and 20",
            )

        # The docs-mcp-server requires a library parameter
        if not library:
            # Get available libraries to help the user
            try:
                available_libraries = await self.docs_manager.proxy_list_libraries()
                return SearchDocsResult(
                    query=query,
                    library=library,
                    version=version,
                    success=False,
                    error="Library parameter is required for search",
                    available_libraries=available_libraries,
                )
            except Exception:
                return SearchDocsResult(
                    query=query,
                    library=library,
                    version=version,
                    success=False,
                    error="Library parameter is required for search",
                )

        try:
            # Use proxy method for search
            result = await self.docs_manager.proxy_search_docs(
                query, library, version, limit
            )
            # Convert proxy result to Pydantic model
            hits = [
                SearchDocsResultItem(
                    title=r.get("title"),
                    url=r.get("url"),
                    content=r.get("content"),
                    score=r.get("score"),
                )
                for r in result.get("results", [])
            ]
            return SearchDocsResult(
                results=hits,
                count=len(hits),
                query=query,
                library=library,
                version=version,
                success=True,
            )
        except DocsServerUnavailableError as e:
            return SearchDocsResult(
                query=query,
                library=library,
                version=version,
                success=False,
                error=str(e),
            )
        except LibraryNotFoundError as e:
            return SearchDocsResult(
                query=query,
                library=library,
                version=version,
                success=False,
                error=str(e),
                available_libraries=e.available_libraries,
            )
        except Exception as e:
            return SearchDocsResult(
                query=query,
                library=library,
                version=version,
                success=False,
                error=f"Search failed: {str(e)}",
            )

    @cache_results(ttl=600, key_strategy="simple")
    @validate_input(schema=ListDocsInput)
    @measure_performance(include_metrics=True, slow_threshold=0.5)
    @mcp_tool(
        "List all available documentation libraries. "
        "Returns library names that can be used with search_imas_docs tool. "
        "Use this tool first before searching documentation to discover available libraries. "
        "Examples of returned libraries: 'data-dictionary', 'imas-python', 'testembeddings'. "
        "Returns library names only - for version-specific searches, use the version parameter in search_imas_docs."
    )
    async def list_imas_docs(
        self,
        library: str | None = None,
        ctx: Context | None = None,
    ) -> ListDocsResult:
        """
        List all indexed documentation libraries available for search.

        Discovery tool for finding what documentation libraries are available.
        Use before search_imas_docs to know which libraries can be searched.

        Args:
            library: Specific library name to get info for (optional, currently unused)
            ctx: FastMCP context

        Returns:
            ListDocsResult with:
            - libraries: List of available library names
            - count: Number of libraries available
            - success: Boolean indicating success
            - note: Information about version availability

        Examples:
            list_imas_docs() → ListDocsResult with libraries=["data-dictionary", "testembeddings"]

        Note:
            Returns library names only. The MCP list_libraries tool does not provide
            version information. Use the version parameter in search_docs for version-specific searches.
        """

        # Basic parameter validation
        if library is not None and not library.strip():
            return ListDocsResult(
                library=library,
                success=False,
                error="Library name cannot be empty",
            )

        try:
            if library:
                # Note: Library parameter currently unused - list_libraries MCP tool
                # does not support per-library queries
                # Return info suggesting to use search_docs for library-specific info

                # Ensure server is started to get base_url
                await self.docs_manager.ensure_started()

                return ListDocsResult(
                    library=library,
                    success=True,
                    note="Use search_imas_docs with this library name to explore available content",
                )
            else:
                # List all available libraries using MCP proxy
                # Note: The MCP list_libraries tool only returns library names (not versions)
                libraries = await self.docs_manager.proxy_list_libraries()

                if not libraries:
                    return ListDocsResult(
                        success=False,
                        error="No libraries found in docs-mcp-server",
                        note="The MCP list_libraries tool returns library names only.",
                    )

                return ListDocsResult(
                    libraries=libraries,
                    count=len(libraries),
                    success=True,
                    note="Library names only. Use version parameter in search_imas_docs for version-specific searches.",
                )
        except DocsServerUnavailableError as e:
            # Graceful fallback when docs-mcp-server is not available
            return ListDocsResult(
                library=library,
                success=False,
                error=str(e),
            )
        except Exception as e:
            # Catch-all for any other unexpected errors
            return ListDocsResult(
                library=library,
                success=False,
                error=f"Unexpected error: {str(e)}",
            )
