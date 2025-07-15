"""
Search module for IMAS MCP server.

This module provides search handlers using the composition pattern to separate
concerns and make the intent of each search strategy clear.
"""

from .ai_enhancer import (
    ai_enhancer,
    SEARCH_EXPERT,
    EXPLANATION_EXPERT,
    OVERVIEW_EXPERT,
    STRUCTURE_EXPERT,
    RELATIONSHIP_EXPERT,
)
from .search_handler import SearchHandler, SearchRequest, SearchResponse
from .search_router import QueryFeatures, SearchRouter

__all__ = [
    "ai_enhancer",
    "SEARCH_EXPERT",
    "EXPLANATION_EXPERT",
    "OVERVIEW_EXPERT",
    "STRUCTURE_EXPERT",
    "RELATIONSHIP_EXPERT",
    "SearchHandler",
    "SearchRequest",
    "SearchResponse",
    "QueryFeatures",
    "SearchRouter",
]
