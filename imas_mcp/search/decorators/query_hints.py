"""
Query hints decorator for SearchResult enhancement.

Provides intelligent query suggestions based on search results.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from ...models.result_models import SearchResult
from ...models.suggestion_models import SearchSuggestion

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def generate_search_query_hints(search_result: SearchResult) -> list[SearchSuggestion]:
    """
    Generate query hints based on SearchResult content.

    Args:
        search_result: The SearchResult to analyze

    Returns:
        List of search suggestions
    """
    hints = []
    # Handle query which might be a list of strings
    if isinstance(search_result.query, list):
        original_query = " ".join(search_result.query).lower()
    else:
        original_query = search_result.query.lower() if search_result.query else ""

    if search_result.hit_count > 0:
        # Extract context from successful search
        found_terms = set()
        physics_domains = set()
        ids_names = set()

        for hit in search_result.hits:
            if hit.path:
                # Extract IDS name
                parts = hit.path.split("/")
                if parts:
                    ids_names.add(parts[0])

                # Extract path terms
                path_terms = [
                    term.lower()
                    for term in hit.path.replace("/", " ").replace("_", " ").split()
                    if len(term) > 2 and term not in original_query
                ]
                found_terms.update(path_terms[:3])  # Limit terms per path

            # Extract physics domain if available
            if hasattr(hit, "physics_domain") and hit.physics_domain:
                physics_domains.add(hit.physics_domain)

        # Suggest related searches based on found IDS
        for ids_name in list(ids_names)[:3]:  # Limit suggestions
            if ids_name.lower() not in original_query:
                hints.append(
                    SearchSuggestion(
                        suggestion=f"{ids_name} measurements",
                        reason=f"Explore more {ids_name} data paths",
                        confidence=0.8,
                    )
                )

        # Suggest searches based on physics domains
        for domain in list(physics_domains)[:2]:
            if domain.lower() not in original_query:
                hints.append(
                    SearchSuggestion(
                        suggestion=f"{domain} physics",
                        reason=f"Find more {domain}-related measurements",
                        confidence=0.7,
                    )
                )

        # Suggest related terms from paths
        for term in list(found_terms)[:3]:
            if term not in original_query and len(term) > 3:
                hints.append(
                    SearchSuggestion(
                        suggestion=term,
                        reason="Term found in related paths",
                        confidence=0.6,
                    )
                )

        # Suggest more specific searches
        if search_result.hit_count > 10:
            hints.append(
                SearchSuggestion(
                    suggestion=f"{original_query} time",
                    reason="Add temporal constraints to narrow results",
                    confidence=0.5,
                )
            )
            hints.append(
                SearchSuggestion(
                    suggestion=f"{original_query} profile",
                    reason="Focus on profile measurements",
                    confidence=0.5,
                )
            )

    else:
        # No results - suggest alternative searches
        query_words = original_query.split()

        # Suggest broader terms
        if len(query_words) > 1:
            for word in query_words:
                if len(word) > 3:
                    hints.append(
                        SearchSuggestion(
                            suggestion=word,
                            reason="Try individual terms from your query",
                            confidence=0.7,
                        )
                    )

        # Suggest common physics terms
        physics_suggestions = [
            ("temperature", "Core plasma temperature measurements"),
            ("density", "Plasma density profiles"),
            ("magnetic field", "Magnetic field measurements"),
            ("pressure", "Plasma pressure data"),
            ("current", "Current density profiles"),
            ("equilibrium", "Magnetic equilibrium data"),
        ]

        for term, description in physics_suggestions:
            if term not in original_query:
                hints.append(
                    SearchSuggestion(
                        suggestion=term,
                        reason=description,
                        confidence=0.4,
                    )
                )
                break  # Just suggest one common term

        # Suggest wildcard searches
        if not any("*" in word for word in query_words):
            hints.append(
                SearchSuggestion(
                    suggestion=f"*{original_query}*",
                    reason="Use wildcards for broader matching",
                    confidence=0.6,
                )
            )

    return hints


def generate_generic_query_hints(result: Any) -> list[SearchSuggestion]:
    """
    Generate generic query hints for non-search ToolResult objects.

    Args:
        result: Any ToolResult object

    Returns:
        List of query suggestions appropriate for the result type
    """
    hints = []
    result_type = type(result).__name__

    if result_type == "ConceptResult":
        concept = getattr(result, "concept", "")
        if concept:
            hints.extend(
                [
                    SearchSuggestion(
                        suggestion=f"{concept} measurements",
                        reason="Search for measurement data related to this concept",
                        confidence=0.8,
                    ),
                    SearchSuggestion(
                        suggestion=f"{concept} profile",
                        reason="Find profile data for this concept",
                        confidence=0.7,
                    ),
                ]
            )

    elif result_type == "StructureResult":
        ids_name = getattr(result, "ids_name", "")
        if ids_name:
            hints.extend(
                [
                    SearchSuggestion(
                        suggestion=f"{ids_name} time",
                        reason="Search for time-dependent data in this IDS",
                        confidence=0.7,
                    ),
                    SearchSuggestion(
                        suggestion=f"{ids_name} profiles",
                        reason="Find profile data in this IDS",
                        confidence=0.6,
                    ),
                ]
            )

    else:
        # Generic suggestions for any result
        hints.extend(
            [
                SearchSuggestion(
                    suggestion="related measurements",
                    reason="Search for related measurement data",
                    confidence=0.5,
                ),
                SearchSuggestion(
                    suggestion="*profile*",
                    reason="Use wildcards to find profile data",
                    confidence=0.4,
                ),
            ]
        )

    return hints


def apply_query_hints(result: Any, max_hints: int = 5) -> Any:
    """
    Apply query hints to any ToolResult object.

    Args:
        result: The result to enhance (must have query_hints attribute)
        max_hints: Maximum number of hints to include

    Returns:
        Enhanced result with query suggestions
    """
    try:
        # Check if result has query_hints attribute (any ToolResult subclass)
        if not hasattr(result, "query_hints"):
            logger.warning(
                f"Result type {type(result)} does not have query_hints field"
            )
            return result

        # Generate hints - only SearchResult has the specific generator for now
        if hasattr(result, "hits") and hasattr(result, "query"):
            hints = generate_search_query_hints(result)
        else:
            # For other ToolResult types, generate basic query suggestions
            hints = generate_generic_query_hints(result)

        # Sort by confidence and limit
        sorted_hints = sorted(hints, key=lambda h: h.confidence or 0, reverse=True)
        result.query_hints = sorted_hints[:max_hints]

    except Exception as e:
        logger.warning(f"Query hints generation failed: {e}")
        # Ensure query_hints exists even if generation fails
        if hasattr(result, "query_hints"):
            result.query_hints = []

    return result


def query_hints(max_hints: int = 5) -> Callable[[F], F]:
    """
    Decorator to add query hints to any ToolResult object.

    Args:
        max_hints: Maximum number of query hints to include

    Returns:
        Decorated function with query hints applied to result
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute original function
            result = await func(*args, **kwargs)

            # Apply query hints if result has query_hints attribute (any ToolResult)
            if hasattr(result, "query_hints"):
                result = apply_query_hints(result, max_hints)

            return result

        return wrapper  # type: ignore

    return decorator
