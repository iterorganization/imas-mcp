"""
Search tool implementation.

This module contains the search_imas tool logic with decorators
for caching, validation, AI enhancement, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp.models.constants import ResponseProfile, SearchMode
from imas_mcp.models.request_models import SearchInput
from imas_mcp.models.result_models import SearchResult

# Import only essential decorators
from imas_mcp.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    measure_performance,
    validate_input,
)
from imas_mcp.search.decorators.hints import hints
from imas_mcp.search.decorators.sample import sample

from .base import BaseTool

logger = logging.getLogger(__name__)


class SearchTool(BaseTool):
    """Tool for searching IMAS data paths using service composition."""

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "search_imas"

    @cache_results(ttl=300, key_strategy="semantic")
    @validate_input(schema=SearchInput)
    @measure_performance(include_metrics=True, slow_threshold=1.0)
    @handle_errors(fallback="search_suggestions")
    @hints(tool_max=4, query_max=5)
    @sample(temperature=0.3, max_tokens=800)
    @mcp_tool(
        "Find IMAS IDS entries using semantic and lexical search. "
        "Options: search_mode=auto|semantic|lexical|hybrid, "
        "response_profile=minimal|standard|detailed"
    )
    async def search_imas(
        self,
        query: str,
        ids_filter: str | list[str] | None = None,
        max_results: int = 50,
        search_mode: str | SearchMode = "auto",
        response_profile: str | ResponseProfile = "standard",
        ctx: Context | None = None,
    ) -> SearchResult:
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
            response_profile: Response preset - "minimal" (results only, no hints/context),
                            "standard" (results+essential hints, default), or "detailed" (full AI enhancement)
            context: FastMCP context for LLM sampling enhancement

        Returns:
            SearchResult with ranked data paths, documentation, and physics insights

        Note:
            For fast exact path validation, use the check_ids_path tool instead.
            That tool is optimized for existence checking without search overhead.
        """

        # Execute search - base.py now handles SearchResult conversion and summary
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
        # Hints are handled in the composite decorator based on the provided mode via kwargs
        profile = str(response_profile)
        if profile == ResponseProfile.MINIMAL.value or profile == "minimal":
            # Minimal: results only, strip all extras
            result = self._format_minimal(result)
        elif profile == ResponseProfile.DETAILED.value or profile == "detailed":
            # Detailed: keep everything (default behavior from decorators)
            pass  # No formatting needed, keep full response

        logger.info(
            f"Search completed: {len(result.hits)} hits returned (of {result.summary.get('total_paths', 0)} total) with profile {response_profile}"
        )
        return result

    def build_prompt(self, prompt_type: str, tool_context: dict[str, Any]) -> str:
        """Build search-specific AI prompts."""
        if prompt_type == "search_analysis":
            return self._build_search_analysis_prompt(tool_context)
        elif prompt_type == "no_results":
            return self._build_no_results_prompt(tool_context)
        elif prompt_type == "search_context":
            return self._build_search_context_prompt(tool_context)
        return ""

    def system_prompt(self) -> str:
        """Get search tool-specific system prompt."""
        return """You are an expert IMAS (Integrated Modelling and Analysis Suite) data analyst specializing in fusion physics data discovery and interpretation. Your expertise includes:

- Deep knowledge of tokamak physics, plasma diagnostics, and fusion measurements
- Understanding of IMAS data dictionary structure and data path conventions
- Experience with plasma parameter relationships and physics contexts
- Ability to suggest relevant follow-up searches and related measurements
- Knowledge of common data access patterns and validation considerations

When analyzing search results, provide:
1. Clear physics context and significance of found data paths
2. Practical guidance for data interpretation and usage
3. Relevant cross-references to related measurements or phenomena
4. Actionable recommendations for follow-up analysis
5. Insights into data quality considerations and validation approaches

Focus on helping researchers efficiently navigate and understand IMAS data for their specific physics investigations."""

    def _build_search_analysis_prompt(self, tool_context: dict[str, Any]) -> str:
        """Build prompt for search result analysis."""
        query = tool_context.get("query", "")
        results = tool_context.get("results", [])
        max_results = tool_context.get("max_results", 3)

        if not results:
            return self._build_no_results_prompt(tool_context)

        # Limit results for prompt
        top_results = results[:max_results]

        # Build results summary
        results_summary = []
        for i, result in enumerate(top_results, 1):
            if hasattr(result, "path"):
                path = result.path
                doc = getattr(result, "documentation", "")[:100]
                score = getattr(result, "relevance_score", 0)
                results_summary.append(f"{i}. {path} (score: {score:.2f})")
                if doc:
                    results_summary.append(f"   Documentation: {doc}...")
            else:
                results_summary.append(f"{i}. {str(result)[:100]}")

        return f"""Search Results Analysis for: "{query}"

Found {len(results)} relevant paths in IMAS data dictionary.

Top results:
{chr(10).join(results_summary)}

Please provide enhanced analysis including:
1. Physics context and significance of these paths
2. Recommended follow-up searches or related concepts
3. Data usage patterns and common workflows
4. Validation considerations for these measurements
5. Brief explanation of how these paths relate to the query"""

    def _build_no_results_prompt(self, tool_context: dict[str, Any]) -> str:
        """Build prompt for when no search results are found."""
        query = tool_context.get("query", "")

        return f"""Search Query Analysis: "{query}"

No results were found for this query in the IMAS data dictionary.

Please provide:
1. Suggestions for alternative search terms or queries
2. Possible related IMAS concepts or data paths
3. Common physics contexts where this term might appear
4. Recommended follow-up searches"""

    def _build_search_context_prompt(self, tool_context: dict[str, Any]) -> str:
        """Build prompt for search mode context."""
        search_mode = tool_context.get("search_mode", "auto")
        return f"""Search mode: {search_mode}
Provide mode-specific analysis and recommendations."""

    def _format_minimal(self, result: SearchResult) -> SearchResult:
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

        # Remove all hints and context
        result.query_hints = []
        result.tool_hints = []

        # Remove physics context to save tokens
        if hasattr(result, "physics_context"):
            result.physics_context = None

        return result
