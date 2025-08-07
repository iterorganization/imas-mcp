"""
Search decorators package.

This package contains all the decorator implementations for cross-cutting concerns
like caching, validation, sampling, performance monitoring, and error handling.
"""

from .cache import cache_results, clear_cache, get_cache_stats
from .validation import validate_input, create_validation_schema
from .tool_recommendations import recommend_tools, generate_tool_recommendations
from .performance import measure_performance, get_performance_summary
from .error_handling import (
    handle_errors,
    create_timeout_handler,
    ToolError,
    ValidationError,
    SearchError,
    ServiceError,
)

# SearchResult enhancement decorators
from .sample import sample
from .tool_hints import tool_hints
from .query_hints import query_hints
from .physics_hints import physics_hints

# MCP tool decorator
from .mcp_tool import mcp_tool

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
    # SearchResult decorators
    "sample",
    "tool_hints",
    "query_hints",
    "physics_hints",
    # MCP tool decorator
    "mcp_tool",
    # Error classes
    "ToolError",
    "ValidationError",
    "SearchError",
    "ServiceError",
]
