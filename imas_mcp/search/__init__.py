"""
Search module for IMAS MCP server.

This module provides semantic search capabilities using sentence transformers
and document storage for efficient IMAS data dictionary querying.
"""

from .decorators.sampling import sample
from .cache import SearchCache
from .document_store import Document, DocumentMetadata, DocumentStore
from .search_strategy import (
    SearchComposer,
    SearchConfig,
    SearchResult,
)
from .tool_suggestions import tool_suggestions

__all__ = [
    "sample",
    "tool_suggestions",
    "SearchCache",
    "DocumentStore",
    "Document",
    "DocumentMetadata",
    "SearchComposer",
    "SearchConfig",
    "SearchResult",
]
