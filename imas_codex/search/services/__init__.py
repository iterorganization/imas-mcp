"""
Search services for IMAS Codex.

This package contains business logic services for search orchestration,
providing clean interfaces for search execution and result processing.
"""

from .search_service import (
    SearchRequest,
    SearchService,
    SearchServiceError,
)

__all__ = ["SearchService", "SearchServiceError", "SearchRequest"]
