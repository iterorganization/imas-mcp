"""
Documentation search MCP tool functions.

This module contains MCP tool functions for documentation search capabilities
using only compatible decorators for generic documentation libraries.
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp.search.decorators import (
    cache_results,
    mcp_tool,
    measure_performance,
)
from imas_mcp.services.docs_proxy_service import (
    DocsProxyService,
    DocsServerUnavailableError,
    LibraryNotFoundError,
)

logger = logging.getLogger(__name__)

# Global service instance
_docs_proxy: DocsProxyService | None = None


def get_docs_proxy() -> DocsProxyService:
    """Get or create global docs proxy service instance"""
    global _docs_proxy
    if _docs_proxy is None:
        _docs_proxy = DocsProxyService()
    return _docs_proxy


@mcp_tool(
    "Search any indexed documentation library with optional version filtering. "
    "Supports multiple documentation libraries and returns comprehensive version information."
)
@cache_results(ttl=300, key_strategy="semantic")
@measure_performance(include_metrics=True, slow_threshold=1.0)
async def search_docs(
    query: str,
    library: str | None = None,
    limit: int | None = None,
    version: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Search documentation with comprehensive error handling"""

    # Basic parameter validation
    if not query or not query.strip():
        return {
            "error": "Query cannot be empty",
            "query": query,
            "library": library,
            "version": version,
            "validation_failed": True,
        }

    if limit is not None and (limit < 1 or limit > 20):
        return {
            "error": "Limit must be between 1 and 20",
            "query": query,
            "library": library,
            "version": version,
            "validation_failed": True,
        }

    # The new npx-based service requires a library parameter
    if not library:
        # Get available libraries to help the user
        try:
            docs_proxy = get_docs_proxy()
            available_libraries = await docs_proxy.list_available_libraries()
            return {
                "error": "Library parameter is required for search",
                "query": query,
                "library": library,
                "version": version,
                "available_libraries": available_libraries,
                "library_required": True,
                "setup_instructions": True,
            }
        except Exception:
            return {
                "error": "Library parameter is required for search",
                "query": query,
                "library": library,
                "version": version,
                "library_required": True,
                "setup_instructions": True,
            }

    docs_proxy = get_docs_proxy()

    try:
        return await docs_proxy.search_docs(query, library, limit, version)
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


@mcp_tool(
    "Search specifically in IMAS-Python documentation with version support. "
    "Automatically uses IMAS-Python library with physics-specific optimizations."
)
@cache_results(ttl=300, key_strategy="semantic")
@measure_performance(include_metrics=True, slow_threshold=1.0)
async def search_imas_python_docs(
    query: str,
    limit: int | None = None,
    version: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Search IMAS-Python documentation with automatic IMAS-specific handling"""

    # Basic parameter validation
    if not query or not query.strip():
        return {
            "error": "Query cannot be empty",
            "query": query,
            "library": "imas-python",
            "version": version,
            "validation_failed": True,
        }

    if limit is not None and (limit < 1 or limit > 20):
        return {
            "error": "Limit must be between 1 and 20",
            "query": query,
            "library": "imas-python",
            "version": version,
            "validation_failed": True,
        }

    docs_proxy = get_docs_proxy()

    try:
        # Use IMAS-Python as the library
        return await docs_proxy.search_docs(
            query=query,
            library="imas-python",
            limit=limit,
            version=version,
        )
    except DocsServerUnavailableError as e:
        return {
            "error": str(e),
            "query": query,
            "library": "imas-python",
            "version": version,
            "setup_instructions": True,
        }
    except Exception as e:
        return {
            "error": str(e),
            "query": query,
            "library": "imas-python",
            "version": version,
            "imas_search_failed": True,
        }


@mcp_tool(
    "List all available documentation libraries with their versions. "
    "Returns comprehensive library information including available versions."
)
@cache_results(ttl=600, key_strategy="simple")
@measure_performance(include_metrics=True, slow_threshold=0.5)
async def list_docs(
    library: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """List available documentation libraries or get versions for a specific library"""

    # Basic parameter validation
    if library is not None and not library.strip():
        return {
            "error": "Library name cannot be empty",
            "library": library,
            "versions": [],
            "validation_failed": True,
        }

    docs_proxy = get_docs_proxy()

    try:
        if library:
            # Get versions for specific library
            versions = await docs_proxy.get_library_versions(library)
            return {
                "library": library,
                "versions": versions,
                "latest": versions[0] if versions else None,
                "count": len(versions),
                "success": True,
            }
        else:
            # List all available libraries
            libraries = await docs_proxy.list_available_libraries()
            return {
                "libraries": libraries,
                "count": len(libraries),
                "success": True,
            }
    except DocsServerUnavailableError as e:
        return {
            "error": str(e),
            "setup_instructions": True,
            "libraries": [],
            "library": library,
        }
    except LibraryNotFoundError as e:
        if library:
            return {
                "error": str(e),
                "library": library,
                "available_libraries": e.available_libraries,
                "library_not_found": True,
            }
        else:
            return {
                "error": str(e),
                "setup_instructions": True,
                "libraries": [],
            }
    except Exception as e:
        if library:
            return {
                "error": f"Failed to get versions for {library}: {str(e)}",
                "library": library,
                "versions": [],
                "version_fetch_failed": True,
            }
        else:
            return {
                "error": f"Failed to list libraries: {str(e)}",
                "libraries": [],
                "listing_failed": True,
            }
