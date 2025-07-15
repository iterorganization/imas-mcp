"""
Search module for IMAS MCP server.

This module provides semantic search capabilities using sentence transformers
and document storage for efficient IMAS data dictionary querying.
"""

from .ai_enhancer import (
    ai_enhancer,
    SEARCH_EXPERT,
    EXPLANATION_EXPERT,
    OVERVIEW_EXPERT,
    STRUCTURE_EXPERT,
    RELATIONSHIP_EXPERT,
)
from .document_store import DocumentStore, Document, DocumentMetadata
from .semantic_search import SemanticSearch, SemanticSearchConfig, SemanticSearchResult

__all__ = [
    "ai_enhancer",
    "SEARCH_EXPERT",
    "EXPLANATION_EXPERT",
    "OVERVIEW_EXPERT",
    "STRUCTURE_EXPERT",
    "RELATIONSHIP_EXPERT",
    "DocumentStore",
    "Document",
    "DocumentMetadata",
    "SemanticSearch",
    "SemanticSearchConfig",
    "SemanticSearchResult",
]
