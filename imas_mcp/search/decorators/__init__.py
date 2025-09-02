"""
Search decorators package.

This package contains all the decorator implementations for cross-cutting concerns
like caching, validation, sampling, performance monitoring, and error handling.
"""

from .cache import cache_results, clear_cache, get_cache_stats
from .error_handling import (
    SearchError,
    ServiceError,
    ToolError,
    ValidationError,
    create_timeout_handler,
    handle_errors,
)

# MCP tool decorator
from .mcp_tool import mcp_tool
from .performance import get_performance_summary, measure_performance

# Note: physics_hints, query_hints, sample, and tool_hints are not imported here
# to avoid circular dependencies. Import them directly when needed.
from .tool_recommendations import generate_tool_recommendations, recommend_tools
from .validation import create_validation_schema, validate_input

__all__ = [
    # Cache decorators
    "cache_results",
    "clear_cache",
    "get_cache_stats",
    # Validation decorators
    "validate_input",
    "create_validation_schema",
    # Tool recommendation decorators
    "recommend_tools",
    "generate_tool_recommendations",
    # Performance decorators
    "measure_performance",
    "get_performance_summary",
    # Error handling decorators
    "handle_errors",
    "create_timeout_handler",
    # MCP tool decorator
    "mcp_tool",
    # Error classes
    "ToolError",
    "ValidationError",
    "SearchError",
    "ServiceError",
    # Note: physics_hints, query_hints, sample, and tool_hints are not exported
    # to avoid circular dependencies. Import them directly when needed.
]
