"""
Search module for IMAS MCP server.

This module provides semantic search capabilities using sentence transformers
and document storage for efficient IMAS data dictionary querying.
"""

from .cache import SearchCache
from .document_store import Document, DocumentMetadata, DocumentStore
from .search_strategy import (
    SearchConfig,
    SearchMatch,
)
from .tool_suggestions import tool_suggestions

__all__ = [
    "tool_suggestions",
    "SearchCache",
    "DocumentStore",
    "Document",
    "DocumentMetadata",
    "SearchConfig",
    "SearchMatch",
]
