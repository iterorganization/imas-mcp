"""
Tool Suggestions Module for IMAS MCP Server.

This module provides a decorator for adding suggested follow-up tools to MCP tool responses.
Separate from AI enhancement to maintain clear separation of concerns.
"""

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


def suggest_follow_up_tools(
    results: dict[str, Any], func_name: str
) -> list[dict[str, str]]:
    """
    Suggest relevant follow-up tools based on current results.

    Args:
        results: Results from the current tool execution
        func_name: Name of the function that generated the results

    Returns:
        List of tool suggestions with sample calls
    """
    suggestions = []

    try:
        # Tool-specific suggestions using clear switch-like logic
        if func_name == "search_imas":
            # For search results, suggest concept explanation and structure analysis
            if results.get("results"):
                suggestions.append(
                    {
                        "tool": "explain_concept",
                        "reason": "Get detailed explanation of physics concepts found in search results",
                        "sample_call": "explain_concept(concept='plasma temperature')",
                    }
                )

                # If results contain specific IDS paths, suggest structure analysis
                for result in results.get("results", [])[:3]:  # Check first 3 results
                    if "/" in result.get("path", ""):
                        ids_name = result["path"].split("/")[0]
                        suggestions.append(
                            {
                                "tool": "analyze_ids_structure",
                                "reason": f"Analyze structure of {ids_name} IDS for better understanding",
                                "sample_call": f"analyze_ids_structure(ids_name='{ids_name}')",
                            }
                        )
                        break

        elif func_name == "explain_concept":
            # After concept explanation, suggest searching for related data
            concept = results.get("concept", "")
            if concept:
                suggestions.append(
                    {
                        "tool": "search_imas",
                        "reason": f"Search for data paths related to {concept}",
                        "sample_call": f"search_imas(query='{concept}')",
                    }
                )

        elif func_name == "analyze_ids_structure":
            # After structure analysis, suggest exploring relationships
            ids_name = results.get("ids_name", "")
            if ids_name:
                suggestions.append(
                    {
                        "tool": "explore_relationships",
                        "reason": f"Explore relationships within {ids_name} IDS",
                        "sample_call": f"explore_relationships(path='{ids_name}')",
                    }
                )

        elif func_name == "get_overview":
            # After overview, suggest searching for specific topics
            suggestions.extend(
                [
                    {
                        "tool": "search_imas",
                        "reason": "Search for specific physics concepts or data paths",
                        "sample_call": "search_imas(query='plasma temperature')",
                    },
                    {
                        "tool": "explore_identifiers",
                        "reason": "Explore identifier schemas and enumeration options",
                        "sample_call": "explore_identifiers(scope='summary')",
                    },
                ]
            )

        elif func_name in ["export_ids", "export_physics_domain"]:
            # After exports, suggest analysis tools
            suggestions.append(
                {
                    "tool": "search_imas",
                    "reason": "Search for specific data within exported domains",
                    "sample_call": "search_imas(query='specific concept')",
                }
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            key = suggestion["tool"]
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)

        return unique_suggestions[:3]  # Limit to 3 suggestions

    except Exception as e:
        logger.warning(f"Error generating tool suggestions: {e}")
        return []


def tool_suggestions(func: Callable) -> Callable:
    """
    Decorator for adding suggested follow-up tools to MCP tool responses.

    Automatically adds tool suggestions to tool responses based on the response content
    and operation type. This is separate from AI enhancement and always applied regardless
    of context availability.

    Usage:
        @tool_suggestions
        async def search_imas(self, query: str, ctx: Optional[Context] = None):
            # Tool implementation returns base response
            results = self.perform_search(query)
            return {"results": results}

            # Decorator will add suggested_tools to the response
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> dict[str, Any]:
        # Call the original tool function
        base_response = await func(*args, **kwargs)

        # Always add tool suggestions if not present and response is a dict
        if isinstance(base_response, dict) and "suggested_tools" not in base_response:
            try:
                suggestions = suggest_follow_up_tools(base_response, func.__name__)
                base_response["suggested_tools"] = suggestions
            except Exception as e:
                logger.warning(f"Failed to generate tool suggestions: {e}")
                base_response["suggested_tools"] = []

        return base_response

    return wrapper
