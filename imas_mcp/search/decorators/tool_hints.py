"""
Tool hints decorator for SearchResult enhancement.

Provides intelligent tool recommendations based on search results.
"""

import functools
import logging
from typing import Any, Callable, List, TypeVar

from ...models.result_models import SearchResult
from ...models.suggestion_models import ToolSuggestion

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def generate_search_tool_hints(search_result: SearchResult) -> List[ToolSuggestion]:
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
                tool_name="explore_relationships",
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

        # Suggest IDS structure analysis for top IDS
        if ids_names:
            top_ids = list(ids_names)[:2]  # Limit to top 2
            for ids_name in top_ids:
                hints.append(
                    ToolSuggestion(
                        tool_name="analyze_ids_structure",
                        description=f"Get comprehensive structural analysis of {ids_name}",
                        relevance=f"Analyze detailed structure of {ids_name} IDS",
                    )
                )

        # Suggest concept explanations for physics domains
        if domains:
            for domain in list(domains)[:2]:  # Limit to top 2
                hints.append(
                    ToolSuggestion(
                        tool_name="explain_concept",
                        description=f"Get detailed explanation of {domain} concepts",
                        relevance=f"Learn more about {domain} physics domain",
                    )
                )

        # Suggest export for substantial results
        if search_result.hit_count >= 5:
            hints.append(
                ToolSuggestion(
                    tool_name="export_ids",
                    description="Export structured data for analysis workflows",
                    relevance=f"Export data for {len(ids_names)} IDS found",
                )
            )

    else:
        # No results - suggest discovery tools
        hints.extend(
            [
                ToolSuggestion(
                    tool_name="get_overview",
                    description="Explore IMAS data structure and available concepts",
                    relevance="No results found - get overview of available data",
                ),
                ToolSuggestion(
                    tool_name="explore_identifiers",
                    description="Discover alternative search terms and data identifiers",
                    relevance="Search for related terms and identifiers",
                ),
                ToolSuggestion(
                    tool_name="explain_concept",
                    description="Get conceptual understanding and context",
                    relevance=f'Learn about "{search_result.query}" concept in fusion physics',
                ),
            ]
        )

    return hints


def apply_tool_hints(search_result: SearchResult, max_hints: int = 4) -> SearchResult:
    """
    Apply tool hints to a SearchResult.

    Args:
        search_result: The SearchResult to enhance
        max_hints: Maximum number of hints to include

    Returns:
        Enhanced SearchResult with tool suggestions
    """
    try:
        hints = generate_search_tool_hints(search_result)

        # Just use the first max_hints suggestions
        limited_hints = hints[:max_hints]

        # Add hints to search result (using base dict access)
        if hasattr(search_result, "__dict__"):
            search_result.__dict__["tool_suggestions"] = limited_hints

    except Exception as e:
        logger.warning(f"Tool hints generation failed: {e}")
        # Ensure tool_suggestions exists even if generation fails
        if hasattr(search_result, "__dict__"):
            search_result.__dict__["tool_suggestions"] = []

    return search_result


def tool_hints(max_hints: int = 4) -> Callable[[F], F]:
    """
    Decorator to add tool hints to SearchResult.

    Args:
        max_hints: Maximum number of tool hints to include

    Returns:
        Decorated function with tool hints applied to SearchResult
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute original function
            result = await func(*args, **kwargs)

            # Apply tool hints if result is SearchResult
            if isinstance(result, SearchResult):
                result = apply_tool_hints(result, max_hints)

            return result

        return wrapper  # type: ignore

    return decorator
