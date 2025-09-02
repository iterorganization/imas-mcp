"""
Tool recommendations decorator for recommending related tools.

Provides intelligent recommendations for follow-up tools based on results.
"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def analyze_search_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze search results to determine recommendation context.

    Args:
        results: List of search results

    Returns:
        Analysis context for recommendations
    """
    if not results:
        return {"result_count": 0, "domains": [], "ids_names": []}

    # Extract domains and IDS names from results
    domains = set()
    ids_names = set()
    paths = []

    for result in results:
        if isinstance(result, dict):
            # Extract path information
            path = result.get("path", "")
            if path:
                paths.append(path)

                # Extract IDS name (first part of path)
                path_parts = path.split("/")
                if path_parts:
                    ids_names.add(path_parts[0])

            # Extract physics domain if available
            domain = result.get("physics_domain")
            if domain:
                domains.add(domain)

    return {
        "result_count": len(results),
        "domains": list(domains),
        "ids_names": list(ids_names),
        "paths": paths,
    }


def generate_search_suggestions(
    query: str, context: dict[str, Any]
) -> list[dict[str, str]]:
    """
    Generate search-based tool suggestions.

    Args:
        query: Original search query
        context: Analysis context from results

    Returns:
        List of tool suggestions
    """
    suggestions = []

    # If we have results, suggest related tools
    if context["result_count"] > 0:
        # Suggest exploring relationships
        suggestions.append(
            {
                "tool": "explore_relationships",
                "reason": f"Explore data relationships for the {context['result_count']} found paths",
                "description": "Discover how these data paths connect to other IMAS structures",
            }
        )

        # If we have specific IDS names, suggest structure analysis
        if context["ids_names"]:
            for ids_name in context["ids_names"][:2]:  # Limit suggestions
                suggestions.append(
                    {
                        "tool": "analyze_ids_structure",
                        "reason": f"Analyze detailed structure of {ids_name} IDS",
                        "description": f"Get comprehensive structural analysis of {ids_name}",
                    }
                )

        # If we have physics domains, suggest concept explanation
        if context["domains"]:
            for domain in context["domains"][:2]:  # Limit suggestions
                suggestions.append(
                    {
                        "tool": "explain_concept",
                        "reason": f"Learn more about {domain} physics domain",
                        "description": f"Get detailed explanation of {domain} concepts",
                    }
                )

        # Suggest export for practical use
        if len(context["paths"]) >= 3:
            suggestions.append(
                {
                    "tool": "export_ids",
                    "reason": f"Export data for the {len(context['ids_names'])} IDS found",
                    "description": "Export structured data for use in analysis workflows",
                }
            )

    else:
        # No results - suggest broader search strategies
        suggestions.append(
            {
                "tool": "get_overview",
                "reason": "No results found - get overview of available data",
                "description": "Explore IMAS data structure and available concepts",
            }
        )

        suggestions.append(
            {
                "tool": "explore_identifiers",
                "reason": "Search for related terms and identifiers",
                "description": "Discover alternative search terms and data identifiers",
            }
        )

        # Suggest concept explanation for the query
        suggestions.append(
            {
                "tool": "explain_concept",
                "reason": f'Learn about "{query}" concept in fusion physics',
                "description": "Get conceptual understanding and context",
            }
        )

    return suggestions


def generate_concept_suggestions(concept: str) -> list[dict[str, str]]:
    """
    Generate suggestions for concept explanation results.

    Args:
        concept: The explained concept

    Returns:
        List of tool suggestions
    """
    suggestions = [
        {
            "tool": "search_imas",
            "reason": f"Find data paths related to {concept}",
            "description": f"Search for IMAS data containing {concept} measurements",
        },
        {
            "tool": "explore_identifiers",
            "reason": f"Explore identifiers and terms related to {concept}",
            "description": "Discover related concepts and terminology",
        },
    ]

    # Add domain-specific suggestions based on concept
    concept_lower = concept.lower()

    if any(term in concept_lower for term in ["temperature", "density", "pressure"]):
        suggestions.append(
            {
                "tool": "search_imas",
                "reason": "Explore core plasma profiles",
                "description": "Search for core_profiles data containing plasma parameters",
            }
        )

    if any(term in concept_lower for term in ["magnetic", "field", "equilibrium"]):
        suggestions.append(
            {
                "tool": "analyze_ids_structure",
                "reason": "Analyze equilibrium IDS structure",
                "description": "Examine magnetic equilibrium data organization",
            }
        )

    if any(term in concept_lower for term in ["transport", "flux"]):
        suggestions.append(
            {
                "tool": "explore_relationships",
                "reason": "Explore transport-related data relationships",
                "description": "Understand connections between transport phenomena",
            }
        )

    return suggestions


def generate_tool_recommendations(
    result: dict[str, Any], strategy: str = "search_based"
) -> list[dict[str, str]]:
    """
    Generate tool recommendations based on function results.

    Args:
        result: Function result to analyze
        strategy: Recommendation strategy

    Returns:
        List of tool recommendations
    """
    if "error" in result:
        # Error case - suggest diagnostic tools
        return [
            {
                "tool": "get_overview",
                "reason": "Get overview of available data and functionality",
                "description": "Explore IMAS capabilities and data structure",
            }
        ]

    # Determine suggestion type based on result structure
    if "hits" in result or "results" in result:
        # Search result
        query = result.get("query", "")
        results = result.get("hits", result.get("results", []))
        context = analyze_search_results(results)
        return generate_search_suggestions(query, context)

    elif "concept" in str(result):
        # Concept explanation result
        # Try to extract concept from result
        concept = result.get("concept", "physics concept")
        return generate_concept_suggestions(concept)

    else:
        # Generic suggestions
        return [
            {
                "tool": "search_imas",
                "reason": "Search for specific data paths",
                "description": "Find relevant IMAS data for your research",
            },
            {
                "tool": "get_overview",
                "reason": "Get overview of IMAS structure",
                "description": "Understand available data and capabilities",
            },
        ]


def recommend_tools(
    strategy: str = "search_based", max_tools: int = 4
) -> Callable[[F], F]:
    """
    Decorator to add tool recommendations to function results.

    Args:
        strategy: Recommendation strategy ("search_based", "concept_based", "generic")
        max_tools: Maximum number of tool recommendations to include (default: 4)

    Returns:
        Decorated function with tool recommendations
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute original function
            result = await func(*args, **kwargs)

            # Only add recommendations to successful results
            if isinstance(result, dict) and "error" not in result:
                recommendations = generate_tool_recommendations(result, strategy)

                # Limit recommendations to specified maximum
                result["suggestions"] = recommendations[:max_tools]

            return result

        return wrapper  # type: ignore

    return decorator
