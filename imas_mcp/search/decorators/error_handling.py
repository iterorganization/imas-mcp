"""
Error handling decorator for robust tool execution.

Provides standardized error handling, logging, and fallback responses.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from imas_mcp.models.error_models import ToolError

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


class ToolException(Exception):
    """Base exception for tool-related errors."""

    def __init__(self, message: str, query: str = "", tool_name: str = ""):
        super().__init__(message)
        self.message = message
        self.query = query
        self.tool_name = tool_name


class ValidationError(ToolException):
    """Error for input validation failures."""

    pass


class SearchError(ToolException):
    """Error for search operation failures."""

    pass


class ServiceError(ToolException):
    """Error for service operation failures."""

    pass


def create_error_response(
    error: str | Exception,
    query: str = "",
    tool_name: str = "",
    include_suggestions: bool = True,
    fallback_data: dict[str, Any] | None = None,
) -> ToolError:
    """
    Create standardized error response.

    Args:
        error: Error message or exception
        query: Original query that caused the error
        tool_name: Name of the tool that failed
        include_suggestions: Whether to include error recovery suggestions
        fallback_data: Optional fallback data to include

    Returns:
        Standardized error response as dictionary (will be converted to ToolError by caller)
    """
    if isinstance(error, Exception):
        error_message = str(error)
        error_type = type(error).__name__
    else:
        error_message = error
        error_type = "ToolError"

    suggestions = []
    context = {
        "error_type": error_type,
        "tool_name": tool_name,
        "operation": "tool_execution",
    }

    if query:
        context["query"] = query

    if include_suggestions:
        suggestions = generate_error_recovery_suggestions(
            error_message, query, tool_name
        )

    return ToolError(
        error=error_message,
        suggestions=suggestions,
        context=context,
        fallback_data=fallback_data,
        query=query,
        ai_prompt={},
        ai_response={},
    )


def generate_error_recovery_suggestions(
    error_message: str, query: str, tool_name: str
) -> list[str]:
    """
    Generate error recovery suggestions.

    Args:
        error_message: The error message
        query: Original query
        tool_name: Name of the failed tool

    Returns:
        List of recovery suggestions
    """
    suggestions = []
    error_lower = error_message.lower()

    # Common error patterns and suggestions
    if "validation" in error_lower or "invalid" in error_lower:
        suggestions.append(
            "Check input parameters - verify that all required parameters are provided and valid"
        )

        if "search_mode" in error_lower:
            suggestions.append(
                'Use valid search mode - try "auto", "semantic", "lexical", or "hybrid"'
            )

        if "max_results" in error_lower:
            suggestions.append("Adjust max_results - use a value between 1 and 100")

    elif "not found" in error_lower or "no results" in error_lower:
        suggestions.append(
            "Try broader search terms - use more general keywords or physics concepts"
        )
        suggestions.append(
            "Check spelling and terminology - verify scientific terms and IMAS-specific vocabulary"
        )

        if tool_name == "search_imas":
            suggestions.append(
                "Try different search mode - switch between semantic, lexical, or hybrid search"
            )

    elif "timeout" in error_lower or "slow" in error_lower:
        suggestions.append(
            "Reduce search scope - use more specific queries or limit max_results"
        )
        suggestions.append(
            "Try simpler query - break complex queries into simpler parts"
        )

    elif "service" in error_lower or "connection" in error_lower:
        suggestions.append(
            "Retry the operation - the service may be temporarily unavailable"
        )
        suggestions.append(
            "Check service status - verify that required services are running"
        )

    # General fallback suggestions
    if not suggestions:
        suggestions.append(
            "Try alternative approach - consider using a different tool or search strategy"
        )
        suggestions.append(
            "Get overview - use get_overview to understand available data"
        )

    return suggestions


def get_fallback_response(
    fallback_type: str, query: str = "", tool_name: str = ""
) -> dict[str, Any]:
    """
    Generate fallback response for specific error types.

    Args:
        fallback_type: Type of fallback response
        query: Original query
        tool_name: Name of the tool

    Returns:
        Fallback response
    """
    if fallback_type == "search_suggestions":
        return {
            "message": "Search failed, but here are some suggestions",
            "query": query,
            "suggestions": [
                {
                    "tool": "get_overview",
                    "reason": "Get overview of available IMAS data",
                    "description": "Explore data structure and capabilities",
                },
                {
                    "tool": "explore_identifiers",
                    "reason": "Find related terms and identifiers",
                    "description": "Discover alternative search terms",
                },
                {
                    "tool": "explain_concept",
                    "reason": f'Learn about "{query}" in fusion physics',
                    "description": "Get conceptual understanding",
                },
            ],
        }

    elif fallback_type == "concept_suggestions":
        return {
            "message": "Concept explanation failed, trying alternative approach",
            "query": query,
            "suggestions": [
                {
                    "tool": "search_imas",
                    "reason": f'Search for data related to "{query}"',
                    "description": "Find specific measurements and data paths",
                },
                {
                    "tool": "get_overview",
                    "reason": "Get general overview of IMAS concepts",
                    "description": "Explore available physics domains",
                },
            ],
        }

    else:
        return {
            "message": f"Operation failed for {tool_name}",
            "query": query,
            "fallback_type": fallback_type,
        }


def handle_errors(
    fallback: str | None = None,
    log_errors: bool = True,
    include_traceback: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to handle errors with standardized responses.

    Args:
        fallback: Type of fallback response to provide
        log_errors: Whether to log errors
        include_traceback: Whether to include traceback in logs

    Returns:
        Decorated function with error handling
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract instance and query for error context
            instance = args[0] if args else None
            query = kwargs.get("query", "")

            # Get tool name
            tool_name = ""
            if instance and hasattr(instance, "get_tool_name"):
                tool_name = instance.get_tool_name()
            elif hasattr(func, "__name__"):
                tool_name = func.__name__

            try:
                # Execute function
                return await func(*args, **kwargs)

            except (ValidationError, SearchError, ServiceError) as e:
                # Handle known tool errors
                if log_errors:
                    logger.warning(f"Tool error in {tool_name}: {e}")

                # Extract query from exception if available
                error_query = getattr(e, "query", query)

                # Get fallback data if specified
                fallback_data = None
                if fallback:
                    fallback_data = get_fallback_response(
                        fallback, error_query, tool_name
                    )

                error_response = create_error_response(
                    e, error_query, tool_name, fallback_data=fallback_data
                )

                return error_response

            except Exception as e:
                # Handle unexpected errors
                if log_errors:
                    if include_traceback:
                        logger.exception(f"Unexpected error in {tool_name}: {e}")
                    else:
                        logger.error(f"Unexpected error in {tool_name}: {e}")

                # Get fallback data if specified
                fallback_data = None
                if fallback:
                    fallback_data = get_fallback_response(fallback, query, tool_name)

                # Create proper error response with additional context
                error_response = create_error_response(
                    f"Unexpected error: {e}",
                    query,
                    tool_name,
                    fallback_data=fallback_data,
                )

                # Add error type to context
                if error_response.context is None:
                    error_response.context = {}
                error_response.context["error_type"] = type(e).__name__

                return error_response

        return wrapper  # type: ignore

    return decorator


def create_timeout_handler(timeout_seconds: float) -> Callable[[F], F]:
    """
    Create a timeout handler decorator.

    Args:
        timeout_seconds: Timeout in seconds

    Returns:
        Timeout decorator
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            import asyncio

            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), timeout=timeout_seconds
                )
            except TimeoutError as e:
                query = kwargs.get("query", "")
                tool_name = func.__name__ if hasattr(func, "__name__") else "unknown"

                raise ServiceError(
                    f"Operation timed out after {timeout_seconds} seconds",
                    query=query,
                    tool_name=tool_name,
                ) from e

        return wrapper  # type: ignore

    return decorator
