"""
Physics hints decorator for SearchPathsResult enhancement.

Provides physics context enhancement for search results.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from ...models.result_models import SearchPathsResult
from ...services import PhysicsService

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


async def apply_physics_hints(
    search_result: SearchPathsResult,
    physics_service: PhysicsService,
) -> SearchPathsResult:
    """
    Apply physics hints to a SearchPathsResult.

    Args:
        search_result: The SearchPathsResult to enhance
        physics_service: Physics service for enhancement

    Returns:
        Enhanced SearchPathsResult with physics context
    """
    try:
        # Determine query to use for physics enhancement
        if isinstance(search_result.query, list):
            query = " ".join(search_result.query)
        else:
            query = search_result.query or ""

        if not query:
            logger.debug("No query available for physics enhancement")
            return search_result

        # Enhance with physics context
        if hasattr(physics_service, "enhance_query"):
            physics_context = await physics_service.enhance_query(query)
            if physics_context:
                # Set physics context directly on the result
                search_result.physics_context = physics_context
                logger.debug(f"Physics context enhanced for: {query}")
            else:
                logger.debug("No physics enhancement available")
        else:
            logger.debug("Physics service does not support query enhancement")

    except Exception as e:
        logger.warning(f"Physics hints enhancement failed: {e}")

    return search_result


def physics_hints() -> Callable[[F], F]:
    """
    Decorator to add physics hints to SearchPathsResult.

    Returns:
        Decorated function with physics hints applied to SearchPathsResult
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get tool instance (self) from args
            tool_instance = args[0] if args else None

            # Execute original function
            result = await func(*args, **kwargs)

            # Apply physics hints if result is SearchPathsResult and tool has physics service
            if (
                isinstance(result, SearchPathsResult)
                and tool_instance
                and hasattr(tool_instance, "physics")
            ):
                include = kwargs.get("include_hints", True)
                enabled = True
                if isinstance(include, bool):
                    enabled = include
                elif isinstance(include, dict):
                    enabled = include.get("physics", True)

                if enabled:
                    result = await apply_physics_hints(result, tool_instance.physics)

            return result

        return wrapper  # type: ignore

    return decorator
