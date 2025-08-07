"""
Search tool implementation.

This module contains the search_imas tool logic with decorators
for caching, validation, AI enhancement, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.result_models import SearchResult
from imas_mcp.models.request_models import SearchInput

# Import only essential decorators
from imas_mcp.search.decorators import (
    cache_results,
    validate_input,
    measure_performance,
    handle_errors,
    sample,
    tool_hints,
    query_hints,
    physics_hints,
    mcp_tool,
)

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
    @mcp_tool("Search for IMAS data paths with relevance-ordered results")
    async def search_imas(
        self,
        query: str,
        ids_filter: Optional[List[str]] = None,
        max_results: int = 10,
        search_mode: Union[str, SearchMode] = "auto",
        ctx: Optional[Any] = None,
    ) -> SearchResult:
        """
        Search for IMAS data paths with relevance-ordered results.

        Uses service composition and context manager for consistent orchestration:
        - Service context manager handles pre/post processing
        - SearchService: Executes search with optimized configuration
        - Unified service pipeline for physics enhancement and AI processing

        Args:
            query: Search term(s), physics concept, symbol, or pattern
            ids_filter: Optional specific IDS to search within
            max_results: Maximum number of results to return (1-100)
            search_mode: Search mode - "auto", "semantic", "lexical", or "hybrid"
            ctx: MCP context for enhancement

        Returns:
            SearchResult with hits, metadata, and AI insights
        """

        # Execute search
        result = await self.execute_search(
            query=query,
            search_mode=search_mode,
            max_results=max_results,
            ids_filter=ids_filter,
        )

        logger.info(f"Search completed: {len(result.hits)} results returned")
        return result

    def build_prompt(self, prompt_type: str, tool_context: Dict[str, Any]) -> str:
        """Build search-specific AI prompts."""
        if prompt_type == "search_analysis":
            return self._build_search_analysis_prompt(tool_context)
        elif prompt_type == "no_results":
            return self._build_no_results_prompt(tool_context)
        elif prompt_type == "search_context":
            return self._build_search_context_prompt(tool_context)
        return ""

    def _build_search_analysis_prompt(self, tool_context: Dict[str, Any]) -> str:
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

    def _build_no_results_prompt(self, tool_context: Dict[str, Any]) -> str:
        """Build prompt for when no search results are found."""
        query = tool_context.get("query", "")

        return f"""Search Query Analysis: "{query}"

No results were found for this query in the IMAS data dictionary.

Please provide:
1. Suggestions for alternative search terms or queries
2. Possible related IMAS concepts or data paths
3. Common physics contexts where this term might appear
4. Recommended follow-up searches"""

    def _build_search_context_prompt(self, tool_context: Dict[str, Any]) -> str:
        """Build prompt for search mode context."""
        search_mode = tool_context.get("search_mode", "auto")
        return f"""Search mode: {search_mode}
Provide mode-specific analysis and recommendations."""
