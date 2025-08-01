"""
Search tool implementation.

This module contains the search_imas tool logic with decorators
for caching, validation, AI enhancement, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Any, List, Optional, Union

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.response_models import SearchResponse
from imas_mcp.models.request_models import SearchInput

# Import only essential decorators
from imas_mcp.search.decorators import (
    cache_results,
    validate_input,
    measure_performance,
    handle_errors,
)

from .base import BaseTool
from imas_mcp.services.sampling import SamplingStrategy
from imas_mcp.services.tool_recommendations import RecommendationStrategy

logger = logging.getLogger(__name__)


def mcp_tool(description: str):
    """Decorator to mark methods as MCP tools with descriptions."""

    def decorator(func):
        func._mcp_tool = True
        func._mcp_description = description
        return func

    return decorator


class SearchTool(BaseTool):
    """Tool for searching IMAS data paths using service composition."""

    # Enable both services for search tool
    enable_sampling: bool = True
    enable_recommendations: bool = True

    # Use search-appropriate strategies
    sampling_strategy = SamplingStrategy.SMART
    recommendation_strategy = RecommendationStrategy.SEARCH_BASED
    max_recommended_tools: int = 5

    def get_tool_name(self) -> str:
        return "search_imas"

    @cache_results(ttl=300, key_strategy="semantic")
    @validate_input(schema=SearchInput)
    @measure_performance(include_metrics=True, slow_threshold=1.0)
    @handle_errors(fallback="search_suggestions")
    @mcp_tool("Search for IMAS data paths with relevance-ordered results")
    async def search_imas(
        self,
        query: str,
        ids_filter: Optional[List[str]] = None,
        max_results: int = 10,
        search_mode: Union[str, SearchMode] = "auto",
        ctx: Optional[Any] = None,
    ) -> SearchResponse:
        """
        Search for IMAS data paths with relevance-ordered results.

        Uses service composition for business logic:
        - SearchConfigurationService: Creates and optimizes search configuration
        - PhysicsService: Enhances queries with physics context
        - ResponseService: Builds standardized Pydantic responses

        Args:
            query: Search term(s), physics concept, symbol, or pattern
            ids_filter: Optional specific IDS to search within
            max_results: Maximum number of results to return (1-100)
            search_mode: Search mode - "auto", "semantic", "lexical", or "hybrid"
            ctx: MCP context for enhancement

        Returns:
            SearchResponse with hits, metadata, and optional AI insights
        """

        # Create search configuration using service
        config = self.search_config.create_config(
            search_mode=search_mode,
            max_results=max_results,
            ids_filter=ids_filter,
            enable_physics=True,
        )

        # Optimize configuration based on query characteristics
        config = self.search_config.optimize_for_query(query, config)

        # Execute search through existing search service
        logger.info(
            f"Executing search: query='{query}' mode={config.search_mode} max_results={max_results}"
        )
        search_results = await self._search_service.search(query, config)

        # Enhance with physics context
        physics_context = await self.physics.enhance_query(query)

        # Prepare AI insights for potential sampling
        ai_insights = {}
        if physics_context:
            ai_insights["physics_context"] = physics_context
            logger.debug(f"Physics context added for query: {query}")
        else:
            logger.debug(f"No physics context found for query: {query}")

        if not search_results:
            ai_insights["guidance"] = self._build_no_results_guidance(query)
        else:
            ai_insights["analysis_prompt"] = self._build_analysis_prompt(
                query, search_results
            )

        # Build response using service
        response = self.response.build_search_response(
            results=search_results,
            query=query,
            search_mode=config.search_mode,
            ai_insights=ai_insights,
        )

        # Apply post-processing services (sampling and recommendations)
        response = await self.apply_services(
            result=response,
            query=query,
            search_mode=config.search_mode,
            tool_name=self.get_tool_name(),
            ctx=ctx,
        )

        # Add standard metadata
        response = self.response.add_standard_metadata(response, self.get_tool_name())

        logger.info(f"Search completed: {len(search_results)} results returned")
        return response

    def _build_no_results_guidance(self, query: str) -> str:
        """Build guidance for queries with no results."""
        return f"""No results found for IMAS search: "{query}"

Provide helpful guidance including:
1. Alternative search terms or concepts to try
2. Common IMAS data paths that might be related
3. Physics context that might help refine the search
4. Suggestions for broader or narrower search strategies"""

    def _build_analysis_prompt(self, query: str, results: List[Any]) -> str:
        """Build analysis prompt for AI enhancement."""

        top_results = results[:3]
        results_text = "\n".join(
            [
                f"- {result.document.metadata.path_name}: {result.document.documentation[:100]}..."
                for result in top_results
            ]
        )

        return f"""Search Results Analysis for: "{query}"
Found {len(results)} relevant paths in IMAS data dictionary.

Top results:
{results_text}

Provide detailed analysis including:
1. Physics context and significance of these paths
2. Recommended follow-up searches or related concepts
3. Data usage patterns and common workflows
4. Validation considerations for these measurements
5. Relationships between the found paths"""
