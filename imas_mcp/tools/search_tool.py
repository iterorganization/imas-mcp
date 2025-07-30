"""
Search tool implementation.

This module contains the search_imas tool logic with decorators
for caching, validation, AI enhancement, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Any, List, Optional, Union

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.response_models import SearchResponse, SearchHit
from imas_mcp.search.search_strategy import SearchConfig, SearchResult
from imas_mcp.models.request_models import SearchInput

# Import all decorators
from imas_mcp.search.decorators import (
    cache_results,
    validate_input,
    sample,
    recommend_tools,
    measure_performance,
    handle_errors,
)

from .base import BaseTool

logger = logging.getLogger(__name__)


def mcp_tool(description: str):
    """Decorator to mark methods as MCP tools with descriptions."""

    def decorator(func):
        func._mcp_tool = True
        func._mcp_description = description
        return func

    return decorator


class SearchTool(BaseTool):
    """Tool for searching IMAS data paths."""

    def get_tool_name(self) -> str:
        return "search_imas"

    @cache_results(ttl=300, key_strategy="semantic")
    @validate_input(schema=SearchInput)
    @sample(temperature=0.3, max_tokens=800)
    @recommend_tools(strategy="search_based", max_tools=4)
    @measure_performance(include_metrics=True, slow_threshold=1.0)
    @handle_errors(fallback="search_suggestions")
    @mcp_tool("Search for IMAS data paths with relevance-ordered results")
    async def search_imas(
        self,
        query: Union[str, List[str]],
        ids_filter: Optional[Union[str, List[str]]] = None,
        max_results: int = 10,
        search_mode: Union[str, SearchMode] = "auto",
        ctx: Optional[Any] = None,
    ) -> SearchResponse:
        """
        Search for IMAS data paths with relevance-ordered results.

        Implementation with decorators:
        - Caching for performance optimization
        - Input validation with Pydantic schemas
        - AI sampling for enriched insights and analysis
        - Tool recommendations for follow-up actions
        - Performance monitoring and metrics collection
        - Error handling with fallback suggestions

        Search modes:
        - "auto": Automatically selects best mode based on query
        - "semantic": Concept-based search using AI embeddings
        - "lexical": Fast text-based keyword search
        - "hybrid": Combines semantic and lexical approaches

        Args:
            query: Search term(s), physics concept, symbol, or pattern
            ids_name: Optional specific IDS to search within
            max_results: Maximum number of results to return (1-100)
            search_mode: Search mode - "auto", "semantic", "lexical", or "hybrid"
            ctx: MCP context for enhancement

        Returns:
            Dictionary with search results processed by decorators:
            - results: List of matching paths with metadata
            - sample_insights: AI-generated analysis (from @sample)
            - suggestions: Follow-up tool recommendations (from @recommend_tools)
            - performance: Execution metrics (from @measure_performance)
        """
        # Create search configuration for service
        config = SearchConfig(
            search_mode=search_mode,  # type: ignore[arg-type]  # Pydantic field validator converts str to SearchMode
            max_results=max_results,
            ids_filter=ids_filter,  # type: ignore[arg-type]  # Pydantic field validator converts str/list to List[str]
            similarity_threshold=0.0,
        )

        # Execute search through service
        logger.info(
            f"Executing search: query='{query}' mode={search_mode} max_results={max_results}"
        )
        results = await self._search_service.search(query, config)

        # Build AI sampling prompt for @sample decorator
        ai_prompt = self._build_sampling_prompt(query, results)

        # Convert SearchResult objects to SearchHit objects
        hits = []
        for result in results:
            # Create SearchHit by copying SearchResult fields and adding API-specific fields
            hit = SearchHit(
                # Inherited from SearchResult
                score=result.score,
                rank=result.rank,
                search_mode=result.search_mode,
                highlights=result.highlights,
                # API-specific fields from document
                path=result.document.metadata.path_name,
                documentation=result.document.documentation,
                units=result.document.units.unit_str if result.document.units else None,
                data_type=result.document.metadata.data_type,
                ids_name=result.document.metadata.ids_name,
                physics_domain=result.document.metadata.physics_domain,
                # Exclude the internal document from API response
                document=None,
            )
            hits.append(hit)

        # Create proper SearchResponse
        response = SearchResponse(
            hits=hits,
            count=len(hits),
            search_mode=config.search_mode,  # This is already a SearchMode enum after validation
            query=query,
            ai_insights={"ai_prompt": ai_prompt},  # Used by @sample decorator
        )

        logger.info(f"Search completed: {len(results)} results returned")
        return response

    def _build_sampling_prompt(
        self, query: Union[str, List[str]], results: List[SearchResult]
    ) -> str:
        """Build AI sampling prompt based on search results."""
        query_str = query if isinstance(query, str) else " ".join(query)

        if not results:
            return f"""No results found for IMAS search: "{query_str}"

Provide helpful guidance including:
1. Alternative search terms or concepts to try
2. Common IMAS data paths that might be related
3. Physics context that might help refine the search
4. Suggestions for broader or narrower search strategies"""

        # Build prompt with top results using direct field access
        top_results = results[:3]
        results_text = "\n".join(
            [
                f"- {result.document.metadata.path_name}: {result.document.documentation[:100]}..."
                for result in top_results
            ]
        )

        return f"""Search Results Analysis for: "{query_str}"
Found {len(results)} relevant paths in IMAS data dictionary.

Top results:
{results_text}

Provide detailed analysis including:
1. Physics context and significance of these paths
2. Recommended follow-up searches or related concepts  
3. Data usage patterns and common workflows
4. Validation considerations for these measurements
5. Relationships between the found paths"""
