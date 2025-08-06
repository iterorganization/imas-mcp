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

        async with self.service_context(
            operation_type="search",
            query=query,
            search_mode=search_mode,
            max_results=max_results,
            ids_filter=ids_filter,
            ctx=ctx,
        ) as service_ctx:
            # Ensure search config is available
            if not service_ctx.search_config:
                # Fallback configuration
                service_ctx.search_config = self.search_config.create_config(
                    search_mode=search_mode,
                    max_results=max_results,
                    ids_filter=ids_filter,
                )

            # Execute search with configured context
            search_results = await self._search_service.search(
                query, service_ctx.search_config
            )

            # Build response using response service directly
            response = self.response.build_search_response(
                results=search_results,
                query=query,
                search_mode=service_ctx.search_config.search_mode,
                ids_filter=ids_filter,
                max_results=max_results,
                ai_response=service_ctx.ai_response,
                ai_prompt=service_ctx.ai_prompt,
            )

            # Store result in context for post-processing
            service_ctx.result = response

            logger.info(f"Search completed: {len(search_results)} results returned")

            # Type assertion for return
            assert isinstance(service_ctx.result, SearchResult)
            return service_ctx.result

    def _build_tool_specific_prompts(
        self, tool_context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Build search-specific AI prompts."""
        prompts = {}

        if tool_context.get("search_mode"):
            prompts["search_context"] = f"""Search mode: {tool_context["search_mode"]}
Provide mode-specific analysis and recommendations."""

        return prompts
