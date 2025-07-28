"""
Search tool implementation with comprehensive decorator composition.

This module contains the search_imas tool logic with comprehensive decorators
for caching, validation, AI enhancement, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from imas_mcp.models.enums import SearchMode
from imas_mcp.search.search_strategy import SearchConfig, SearchResult
from imas_mcp.search.services.search_service import SearchService
from imas_mcp.search.engines.semantic_engine import SemanticSearchEngine
from imas_mcp.search.engines.lexical_engine import LexicalSearchEngine
from imas_mcp.search.engines.hybrid_engine import HybridSearchEngine
from imas_mcp.search.schemas.search_schemas import SearchInputSchema

# Import all decorators for Phase 4 implementation
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


class Search(BaseTool):
    """Tool for searching IMAS data paths with comprehensive decorator composition."""

    def __init__(self, ids_set: Optional[set[str]] = None):
        """Initialize search tool with search service."""
        super().__init__()
        self.ids_set = ids_set
        self._search_service = self._create_search_service()

    def _create_search_service(self) -> SearchService:
        """Create search service with appropriate engines."""
        # Create engines for each mode
        engines = {}
        for mode in [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]:
            config = SearchConfig(
                mode=mode, max_results=100
            )  # Service will limit based on request
            engine = self._create_engine(mode.value, config)
            engines[mode] = engine

        return SearchService(engines)

    def _create_engine(self, engine_type: str, config: SearchConfig):
        """Create a search engine of the specified type."""
        engine_map = {
            "semantic": SemanticSearchEngine,
            "lexical": LexicalSearchEngine,
            "hybrid": HybridSearchEngine,
        }

        if engine_type not in engine_map:
            raise ValueError(f"Unknown engine type: {engine_type}")

        engine_class = engine_map[engine_type]
        return engine_class(config)

    def get_tool_name(self) -> str:
        return "search_imas"

    @cache_results(ttl=300, key_strategy="semantic")
    @validate_input(schema=SearchInputSchema)
    @sample(temperature=0.3, max_tokens=800)
    @recommend_tools(strategy="search_based", max_tools=4)
    @measure_performance(include_metrics=True, slow_threshold=1.0)
    @handle_errors(fallback="search_suggestions")
    @mcp_tool("Search for IMAS data paths with relevance-ordered results")
    async def search_imas(
        self,
        query: Union[str, List[str]],
        ids_name: Optional[str] = None,
        max_results: int = 10,
        search_mode: str = "auto",
        ctx: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Search for IMAS data paths with relevance-ordered results.

        Implementation with comprehensive decorator composition:
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
        # Convert search mode string to enum for service
        mode_map = {
            "auto": SearchMode.AUTO,
            "semantic": SearchMode.SEMANTIC,
            "lexical": SearchMode.LEXICAL,
            "hybrid": SearchMode.HYBRID,
        }
        search_mode_enum = mode_map[search_mode]

        # Create search configuration for service
        config = SearchConfig(
            mode=search_mode_enum,
            max_results=max_results,
            filter_ids=[ids_name] if ids_name else None,
            similarity_threshold=0.0,
        )

        # Execute search through service
        logger.info(
            f"Executing search: query='{query}' mode={search_mode} max_results={max_results}"
        )
        results = await self._search_service.search(query, config)

        # Build AI sampling prompt for @sample decorator
        ai_prompt = self._build_sampling_prompt(query, results)

        # Convert to expected format with sampling prompt
        result = {
            "results": [hit.to_dict() for hit in results],
            "results_count": len(results),
            "search_mode": search_mode,
            "query": query,
            "ai_prompt": ai_prompt,  # Used by @sample decorator
        }

        logger.info(f"Search completed: {len(results)} results returned")
        return result

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

        # Build prompt with top results using to_dict() method
        top_results = results[:3]
        results_text = "\n".join(
            [
                f"- {hit.to_dict()['path']}: {hit.to_dict()['documentation'][:100]}..."
                for hit in top_results
            ]
        )

        return f"""Search Results Analysis for: "{query_str}"
Found {len(results)} relevant paths in IMAS data dictionary.

Top results:
{results_text}

Provide comprehensive analysis including:
1. Physics context and significance of these paths
2. Recommended follow-up searches or related concepts  
3. Data usage patterns and common workflows
4. Validation considerations for these measurements
5. Relationships between the found paths"""
