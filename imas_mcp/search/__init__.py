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
from .document_store import Document, DocumentMetadata, DocumentStore

# Don't import SemanticSearch here - make it lazy to avoid startup penalty
# from .semantic_search import SemanticSearch, SemanticSearchConfig, SemanticSearchResult

__all__ = [
    "ai_enhancer",
    "SEARCH_EXPERT",
    "EXPLANATION_EXPERT",
    "OVERVIEW_EXPERT",
    "STRUCTURE_EXPERT",
    "RELATIONSHIP_EXPERT",
    "BULK_EXPORT_EXPERT",
    "PHYSICS_DOMAIN_EXPERT",
    "DocumentStore",
    "Document",
    "DocumentMetadata",
]
