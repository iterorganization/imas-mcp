"""Response building service for consistent Pydantic model construction."""

import importlib.metadata
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.result_models import (
    GetIdentifiersResult,
    GetOverviewResult,
    SearchPathsResult,
)
from imas_mcp.search.search_strategy import SearchMatch

from .base import BaseService


def _get_version() -> str:
    """Get package version with robust fallback."""
    try:
        version = importlib.metadata.version("imas-mcp")
        if version is not None:
            return version
    except importlib.metadata.PackageNotFoundError:
        pass
    return "development"


VERSION = _get_version()

T = TypeVar("T", bound=BaseModel)


class ResponseService(BaseService):
    """Service for building standardized responses across all tool types."""

    def build_search_response(
        self,
        results: list[SearchMatch],
        query: str,
        search_mode: SearchMode,
        ids_filter: str | list[str] | None = None,
        max_results: int | None = None,
        physics_domains: list[str] | None = None,
    ) -> SearchPathsResult:
        """Build SearchPathsResult from search results with complete context."""
        # Convert SearchMatch objects to SearchHit for API response
        hits = [result.to_hit() for result in results]

        # Convert ids_filter to list if it's a string
        if isinstance(ids_filter, str):
            ids_filter = ids_filter.split()

        return SearchPathsResult(
            hits=hits,
            search_mode=search_mode,
            query=query,
            ids_filter=ids_filter,
            max_results=max_results,
            physics_domains=physics_domains or [],
        )

    def build_overview_response(
        self,
        content: str,
        available_ids: list[str],
        hits: list[Any],
        query: str | None = None,
        physics_domains: list[str] | None = None,
        ids_statistics: dict[str, Any] | None = None,
        usage_guidance: dict[str, Any] | None = None,
    ) -> GetOverviewResult:
        """Build GetOverviewResult for system overviews."""
        return GetOverviewResult(
            content=content,
            available_ids=available_ids,
            hits=hits,
            query=query or "",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
            physics_domains=physics_domains or [],
            ids_statistics=ids_statistics or {},
            usage_guidance=usage_guidance or {},
        )

    def build_identifier_response(
        self,
        schemas: list[dict[str, Any]],
        paths: list[dict[str, Any]],
        analytics: dict[str, Any],
        tool_name: str,
        query: str | None = None,
    ) -> GetIdentifiersResult:
        """Build GetIdentifiersResult for identifier exploration."""
        return GetIdentifiersResult(
            schemas=schemas,
            paths=paths,
            analytics=analytics,
            tool_name=tool_name,
            processing_timestamp=datetime.now(UTC).isoformat(),
            version=VERSION,
            query=query or "",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
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
                    "processing_timestamp": datetime.now(UTC).isoformat(),
                    "version": VERSION,
                }
            )
            # Use setattr to update the metadata field
            response.metadata = metadata
        return response
