"""
Search decorators package.

This package contains all the decorator implementations for cross-cutting concerns
like caching, validation, sampling, performance monitoring, and error handling.
"""

from .cache import cache_results, clear_cache, get_cache_stats
from .validation import validate_input, create_validation_schema
from .sampling import sample, build_search_sample_prompt, build_concept_sample_prompt
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

__all__ = [
    # Cache decorators
    "cache_results",
    "clear_cache",
    "get_cache_stats",
    # Validation decorators
    "validate_input",
    "create_validation_schema",
    # Sampling decorators
    "sample",
    "build_search_sample_prompt",
    "build_concept_sample_prompt",
    # Tool recommendation decorators
    "recommend_tools",
    "generate_tool_recommendations",
    # Performance decorators
    "measure_performance",
    "get_performance_summary",
    # Error handling decorators
    "handle_errors",
    "create_timeout_handler",
    "ToolError",
    "ValidationError",
    "SearchError",
    "ServiceError",
]
