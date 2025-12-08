"""
Tool hints decorator for SearchResult enhancement.

Provides intelligent tool recommendations based on search results.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from ...models.result_models import SearchResult
from ...models.suggestion_models import ToolSuggestion

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def generate_search_tool_hints(search_result: SearchResult) -> list[ToolSuggestion]:
    """
    Generate tool hints based on SearchResult content.

    Args:
        search_result: The SearchResult to analyze

    Returns:
        List of tool suggestions
    """
    hints = []

    # Check if we have results
    if search_result.hit_count > 0:
        # Suggest exploring relationships for found paths
        hints.append(
            ToolSuggestion(
                tool_name="search_imas_clusters",
                description="Discover how these data paths connect to other IMAS structures",
                relevance=f"Explore relationships for {search_result.hit_count} found paths",
            )
        )

        # Extract unique IDS names from hits
        ids_names = set()
        domains = set()
        for hit in search_result.hits:
            # Extract IDS name (first part of path)
            if hit.path:
                parts = hit.path.split("/")
                if parts:
                    ids_names.add(parts[0])

            # Extract physics domain if available
            if hasattr(hit, "physics_domain") and hit.physics_domain:
                domains.add(hit.physics_domain)

        # Suggest listing paths for top IDS
        if ids_names:
            top_ids = list(ids_names)[:2]  # Limit to top 2
            for ids_name in top_ids:
                hints.append(
                    ToolSuggestion(
                        tool_name="list_imas_paths",
                        description=f"List all data paths in {ids_name}",
                        relevance=f"Explore the structure of {ids_name} IDS",
                    )
                )

        # Suggest fetch_imas_paths for substantial results
        if search_result.hit_count >= 5:
            hints.append(
                ToolSuggestion(
                    tool_name="fetch_imas_paths",
                    description="Get detailed documentation for found paths",
                    relevance=f"Get details for {search_result.hit_count} paths found",
                )
            )

    else:
        # No results - suggest discovery tools
        hints.extend(
            [
                ToolSuggestion(
                    tool_name="get_imas_overview",
                    description="Explore IMAS data structure and available concepts",
                    relevance="No results found - get overview of available data",
                ),
                ToolSuggestion(
                    tool_name="list_imas_identifiers",
                    description="Discover alternative search terms and data identifiers",
                    relevance="Search for related terms and identifiers",
                ),
            ]
        )

    return hints


def generate_generic_tool_hints(result: Any) -> list[ToolSuggestion]:
    """
    Generate tool hints for non-search ToolResult objects.

    Args:
        result: Any ToolResult object

    Returns:
        List of tool suggestions appropriate for the result type
    """
    hints = []

    # Determine result type and suggest appropriate tools
    result_type = type(result).__name__

    if result_type == "OverviewResult":
        # For overview, suggest search and exploration tools
        hints.extend(
            [
                ToolSuggestion(
                    tool_name="search_imas_paths",
                    description="Search for specific IMAS data paths",
                    relevance="Find specific data paths in the IMAS data dictionary",
                ),
                ToolSuggestion(
                    tool_name="list_imas_identifiers",
                    description="Explore identifier schemas and enumerations",
                    relevance="Discover valid values and identifiers",
                ),
            ]
        )

    elif result_type == "RelationshipResult":
        # For relationship results, suggest search and path listing
        hints.extend(
            [
                ToolSuggestion(
                    tool_name="fetch_imas_paths",
                    description="Get detailed documentation for related paths",
                    relevance="Get full documentation for related data paths",
                ),
                ToolSuggestion(
                    tool_name="search_imas_paths",
                    description="Search for related data paths",
                    relevance="Find additional related measurements",
                ),
            ]
        )

    elif result_type == "IdentifierResult":
        # For identifier results, suggest search and overview
        hints.extend(
            [
                ToolSuggestion(
                    tool_name="search_imas_paths",
                    description="Search for paths using these identifiers",
                    relevance="Find data paths that use these identifiers",
                ),
                ToolSuggestion(
                    tool_name="get_imas_overview",
                    description="Get overview of IMAS data structure",
                    relevance="Understand the broader context of these identifiers",
                ),
            ]
        )

    else:
        # Generic suggestions for any tool result
        hints.extend(
            [
                ToolSuggestion(
                    tool_name="search_imas_paths",
                    description="Search for related IMAS data paths",
                    relevance="Find additional related data",
                ),
                ToolSuggestion(
                    tool_name="get_imas_overview",
                    description="Get overview of IMAS data structure",
                    relevance="Explore the broader IMAS data context",
                ),
            ]
        )

    return hints


def apply_tool_hints(result: Any, max_hints: int = 4) -> Any:
    """
    Apply tool hints to any ToolResult object.

    Args:
        result: The result to enhance (must have tool_hints attribute)
        max_hints: Maximum number of hints to include

    Returns:
        Enhanced result with tool suggestions
    """
    try:
        # Check if result has tool_hints attribute (any ToolResult subclass)
        if not hasattr(result, "tool_hints"):
            logger.warning(f"Result type {type(result)} does not have tool_hints field")
            return result

        # Generate hints based on result type - check tool_name property for SearchResult
        if hasattr(result, "tool_name") and result.tool_name == "search_imas_paths":
            # SearchResult - use existing search hint generator
            hints = generate_search_tool_hints(result)
        else:
            # Other ToolResult types - generate generic tool hints
            hints = generate_generic_tool_hints(result)

        # Limit and assign hints
        limited_hints = hints[:max_hints]
        result.tool_hints = limited_hints

    except Exception as e:
        logger.warning(f"Tool hints generation failed: {e}")
        # Ensure tool_hints exists even if generation fails
        if hasattr(result, "tool_hints"):
            result.tool_hints = []

    return result


def tool_hints(max_hints: int = 4) -> Callable[[F], F]:
    """
    Decorator to add tool hints to any ToolResult object.

    Args:
        max_hints: Maximum number of tool hints to include

    Returns:
        Decorated function with tool hints applied to result
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute original function
            result = await func(*args, **kwargs)

            # Apply tool hints if result has tool_hints attribute (any ToolResult)
            # Check call-time include_hints option
            include = kwargs.get("include_hints", True)
            enabled = True
            if isinstance(include, bool):
                enabled = include
            elif isinstance(include, dict):
                enabled = include.get("tool", True)

            if enabled and hasattr(result, "tool_hints"):
                result = apply_tool_hints(result, max_hints)

            return result

        return wrapper  # type: ignore

    return decorator
