"""
Search module for IMAS MCP server.

This module provides semantic search capabilities using sentence transformers
and document storage for efficient IMAS data dictionary querying.
"""

from .ai_enhancer import ai_enhancer
from .cache import SearchCache
from .document_store import Document, DocumentMetadata, DocumentStore
from .search_modes import (
    SearchComposer,
    SearchConfig,
    SearchMode,
    SearchResult,
)
from .tool_suggestions import tool_suggestions

__all__ = [
    "ai_enhancer",
    "tool_suggestions",
    "SearchCache",
    "DocumentStore",
    "Document",
    "DocumentMetadata",
    "SearchComposer",
    "SearchConfig",
    "SearchMode",
    "SearchResult",
]
