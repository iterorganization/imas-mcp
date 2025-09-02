"""
Search tool implementation.

This module contains the search_imas tool logic with decorators
for caching, validation, AI enhancement, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp.models.constants import SearchMode
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
from imas_mcp.search.decorators.physics_hints import physics_hints
from imas_mcp.search.decorators.query_hints import query_hints
from imas_mcp.search.decorators.sample import sample
from imas_mcp.search.decorators.tool_hints import tool_hints

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
    @tool_hints(max_hints=4)
    @query_hints(max_hints=5)
    @physics_hints()
    @sample(temperature=0.3, max_tokens=800)
    @mcp_tool("Find IMAS data paths using semantic and lexical search")
    async def search_imas(
        self,
        query: str,
        ids_filter: list[str] | None = None,
        max_results: int = 50,
        search_mode: str | SearchMode = "auto",
        output_mode: str = "full",
        ctx: Context | None = None,
    ) -> SearchResult:
        """
        Find IMAS data paths using semantic and lexical search capabilities.

        Primary discovery tool for locating specific measurements, physics quantities,
        or diagnostic data within the IMAS data dictionary. Returns ranked results
        with physics context and documentation.

        Args:
            query: Search term, physics concept, measurement name, or data path pattern
            ids_filter: Limit search to specific IDS (e.g., ['equilibrium', 'transport'])
            max_results: Maximum number of hits to return (summary contains all matches)
            search_mode: Search strategy - "auto", "semantic", "lexical", or "hybrid"
            output_mode: Output format - "full" or "compact"
            context: FastMCP context for LLM sampling enhancement

        Returns:
            SearchResult with ranked data paths, documentation, and physics insights
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

        # Apply output mode formatting if requested
        if output_mode == "compact":
            result = self._format_compact(result)

        logger.info(
            f"Search completed: {len(result.hits)} hits returned (of {result.summary.get('total_paths', 0)} total) in {output_mode} mode"
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

    def _format_compact(self, result: SearchResult) -> SearchResult:
        """Format result with minimal documentation for efficiency."""
        # Keep paths and basic info but trim documentation
        for hit in result.hits:
            if hasattr(hit, "documentation") and hit.documentation:
                # Truncate documentation to first 100 characters
                hit.documentation = (
                    hit.documentation[:100] + "..."
                    if len(hit.documentation) > 100
                    else hit.documentation
                )

        # Reduce hints to save space
        result.query_hints = result.query_hints[:3] if result.query_hints else []
        result.tool_hints = result.tool_hints[:3] if result.tool_hints else []

        # Keep physics context but reduce physics_matches to top 3
        if (
            hasattr(result, "physics_context")
            and result.physics_context
            and hasattr(result.physics_context, "physics_matches")
        ):
            result.physics_context.physics_matches = (
                result.physics_context.physics_matches[:3]
            )

        return result
