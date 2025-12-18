"""
Search tool implementation.

This module contains the search_imas tool logic with decorators
for caching, validation, performance monitoring, and error handling.
"""

import logging

from fastmcp import Context

from imas_codex.models.constants import (
    LOW_CONFIDENCE_THRESHOLD,
    VERY_LOW_CONFIDENCE_THRESHOLD,
    ResponseProfile,
    SearchMode,
)
from imas_codex.models.request_models import SearchInput
from imas_codex.models.result_models import SearchPathsResult
from imas_codex.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    measure_performance,
    validate_input,
)

from .base import BaseTool
from .utils import normalize_ids_filter, validate_query

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
        "query (required): Natural language description or physics term (e.g., 'electron temperature', 'magnetic field boundary', 'plasma current'). "
        "Common abbreviations supported: Te (electron temp), Ti (ion temp), ne (electron density), Ip (plasma current). "
        "ids_filter: Limit to specific IDS (space/comma-delimited: 'equilibrium magnetics' or 'equilibrium, core_profiles'). "
        "search_mode: 'auto' (default), 'semantic', 'lexical', or 'hybrid'. "
        "response_profile: 'minimal', 'standard' (default), or 'detailed'."
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
            ids_filter: Limit search to specific IDS. Accepts:
                       - Space-delimited string: "equilibrium transport core_profiles"
                       - Comma-delimited string: "equilibrium, transport, core_profiles"
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
        # Validate query is not empty
        is_valid, error_message = validate_query(query, "search_imas_paths")
        if not is_valid:
            helpful_error = (
                "Query cannot be empty. Provide a search term like:\n"
                "  - 'electron temperature' or 'Te' for temperature data\n"
                "  - 'magnetic field' for field measurements\n"
                "  - 'equilibrium boundary' for plasma boundary data\n"
                "Use get_imas_overview() to explore available IDS structures."
            )
            return SearchPathsResult(
                hits=[],
                summary={"error": helpful_error, "query": query or ""},
                query=query or "",
                search_mode=SearchMode.AUTO,
                physics_domains=[],
                error=helpful_error,
            )

        # Normalize ids_filter to support space/comma-delimited strings
        normalized_ids_filter = normalize_ids_filter(ids_filter)

        # Execute search - base.py now handles SearchPathsResult conversion and summary
        result = await self.execute_search(
            query=query,
            search_mode=search_mode,
            max_results=max_results,
            ids_filter=normalized_ids_filter,
        )

        # Add query and search_mode to summary if not already present
        if hasattr(result, "summary") and result.summary:
            result.summary.update({"query": query, "search_mode": str(search_mode)})

        # Check confidence based on top score and add warning if needed
        result = self._check_confidence(result, query)

        # Apply response profile formatting if requested
        profile = str(response_profile)
        if profile == ResponseProfile.MINIMAL.value or profile == "minimal":
            # Minimal: results only, strip all extras
            result = self._format_minimal(result)

        logger.info(
            f"Search completed: {len(result.hits)} hits returned with profile {response_profile}"
        )
        return result

    def _check_confidence(
        self, result: SearchPathsResult, query: str
    ) -> SearchPathsResult:
        """Check search result confidence and add warning if scores are low.

        Args:
            result: The search result to check
            query: The original query for context in warnings

        Returns:
            SearchPathsResult with confidence_warning set if needed
        """
        if not result.hits:
            return result

        # Get the maximum score from results
        max_score = max(hit.score for hit in result.hits)

        if max_score < VERY_LOW_CONFIDENCE_THRESHOLD:
            result.confidence_warning = (
                f"Very low confidence results (max score: {max_score:.2f}). "
                f"The query '{query}' may not match any IMAS concepts. "
                "Consider using get_imas_overview() to explore available data structures, "
                "or try more specific physics terms like 'electron temperature', "
                "'magnetic field', or 'plasma current'."
            )
        elif max_score < LOW_CONFIDENCE_THRESHOLD:
            result.confidence_warning = (
                f"Low confidence results (max score: {max_score:.2f}). "
                "Consider refining your search query with more specific terms. "
                "Use search_imas_clusters() to discover related concepts."
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

        return result
