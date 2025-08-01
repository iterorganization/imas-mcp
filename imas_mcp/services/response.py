"""Response building service for consistent Pydantic model construction."""

import importlib.metadata
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, TypeVar
from pydantic import BaseModel

from imas_mcp.models.response_models import SearchResponse
from imas_mcp.search.search_strategy import SearchResult
from imas_mcp.models.constants import SearchMode
from .base import BaseService

try:
    VERSION = importlib.metadata.version("imas-mcp")
except importlib.metadata.PackageNotFoundError:
    VERSION = "development"

T = TypeVar("T", bound=BaseModel)


class ResponseService(BaseService):
    """Service for building standardized responses."""

    def build_search_response(
        self,
        results: List[SearchResult],
        query: str,
        search_mode: SearchMode,
        ai_insights: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """Build SearchResponse from search results."""

        # Convert SearchResult objects to SearchHit for API response
        hits = [result.to_hit() for result in results]

        return SearchResponse(
            hits=hits,
            search_mode=search_mode,
            query=query,
            ai_insights=ai_insights or {},
        )

    def add_standard_metadata(self, response: T, tool_name: str) -> T:
        """Add standard metadata to any response."""
        if hasattr(response, "metadata"):
            metadata = getattr(response, "metadata", {})
            if metadata is None:
                metadata = {}
            metadata.update(
                {
                    "tool": tool_name,
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                    "version": VERSION,
                }
            )
            # Use setattr to update the metadata field
            setattr(response, "metadata", metadata)
        return response
