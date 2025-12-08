"""
Search tool implementation.

This module contains the search_imas tool logic with decorators
for caching, validation, performance monitoring, and error handling.
"""

import logging

from fastmcp import Context

from imas_mcp.models.constants import ResponseProfile, SearchMode
from imas_mcp.models.request_models import SearchInput
from imas_mcp.models.result_models import SearchPathsResult
from imas_mcp.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    measure_performance,
    validate_input,
)

from .base import BaseTool

logger = logging.getLogger(__name__)


class SearchTool(BaseTool):
    """Tool for searching IMAS data paths using service composition."""

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "search_imas_paths"

    @cache_results(ttl=300, key_strategy="semantic")
    @validate_input(schema=SearchInput)
    @measure_performance(include_metrics=True, slow_threshold=1.0)
    @handle_errors(fallback="search_suggestions")
    @mcp_tool(
        "Find IMAS IDS entries using semantic and lexical search. "
        "Options: search_mode=auto|semantic|lexical|hybrid, "
        "response_profile=minimal|standard|detailed"
    )
    async def search_imas_paths(
        self,
        query: str,
        ids_filter: str | list[str] | None = None,
        max_results: int = 50,
        search_mode: str | SearchMode = "auto",
        response_profile: str | ResponseProfile = "standard",
        ctx: Context | None = None,
    ) -> SearchPathsResult:
        """
        Find IMAS data paths using semantic and lexical search capabilities.

        Primary discovery tool for locating specific measurements, physics quantities,
        or diagnostic data within the IMAS data dictionary. Returns ranked results
        with physics context and documentation.

        Args:
            query: Full IMAS path for validation, or search term/concept for discovery
            ids_filter: Limit search to specific IDS. Accepts either:
                       - Space-delimited string: "equilibrium transport core_profiles"
                       - List of IDS names: ["equilibrium", "transport"]
            max_results: Maximum number of hits to return (summary contains all matches)
            search_mode: Search strategy - "auto", "semantic", "lexical", or "hybrid"
            response_profile: Response preset - "minimal" (results only),
                            "standard" (results+physics context, default), or "detailed" (full context)
            ctx: FastMCP context

        Returns:
            SearchPathsResult with ranked data paths, documentation, and physics insights

        Note:
            For fast exact path validation, use the check_ids_path tool instead.
            That tool is optimized for existence checking without search overhead.
        """

        # Execute search - base.py now handles SearchPathsResult conversion and summary
        result = await self.execute_search(
            query=query,
            search_mode=search_mode,
            max_results=max_results,
            ids_filter=ids_filter,
        )

        # Add query and search_mode to summary if not already present
        if hasattr(result, "summary") and result.summary:
            result.summary.update({"query": query, "search_mode": str(search_mode)})

        # Apply response profile formatting if requested
        profile = str(response_profile)
        if profile == ResponseProfile.MINIMAL.value or profile == "minimal":
            # Minimal: results only, strip all extras
            result = self._format_minimal(result)

        logger.info(
            f"Search completed: {len(result.hits)} hits returned with profile {response_profile}"
        )
        return result

    def _format_minimal(self, result: SearchPathsResult) -> SearchPathsResult:
        """Format result with minimal information - results only, no extras."""
        # Keep paths and basic info but trim documentation
        for hit in result.hits:
            if hasattr(hit, "documentation") and hit.documentation:
                # Truncate documentation to first 100 characters
                hit.documentation = (
                    hit.documentation[:100] + "..."
                    if len(hit.documentation) > 100
                    else hit.documentation
                )

        # Remove physics context to save tokens
        if hasattr(result, "physics_context"):
            result.physics_context = None

        return result
