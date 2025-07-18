"""
Search module for IMAS MCP server.

This module provides semantic search capabilities using sentence transformers
and document storage for efficient IMAS data dictionary querying.
"""

from .ai_enhancer import (
    BULK_EXPORT_EXPERT,
    EXPLANATION_EXPERT,
    OVERVIEW_EXPERT,
    PHYSICS_DOMAIN_EXPERT,
    RELATIONSHIP_EXPERT,
    SEARCH_EXPERT,
    STRUCTURE_EXPERT,
    ai_enhancer,
)
from .cache import SearchCache
from .document_store import Document, DocumentMetadata, DocumentStore
from .search_modes import (
    SearchComposer,
    SearchConfig,
    SearchMode,
    SearchResult,
)

__all__ = [
    "ai_enhancer",
    "SEARCH_EXPERT",
    "EXPLANATION_EXPERT",
    "OVERVIEW_EXPERT",
    "STRUCTURE_EXPERT",
    "RELATIONSHIP_EXPERT",
    "BULK_EXPORT_EXPERT",
    "PHYSICS_DOMAIN_EXPERT",
    "SearchCache",
    "DocumentStore",
    "Document",
    "DocumentMetadata",
    "SearchComposer",
    "SearchConfig",
    "SearchMode",
    "SearchResult",
]
