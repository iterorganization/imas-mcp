"""
Search tool implementation.

This module contains the search_imas tool logic extracted from the monolithic Tools class.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from imas_mcp.models.enums import SearchMode
from imas_mcp.search.search_strategy import SearchConfig, SearchResult
from imas_mcp.search.services.search_service import SearchService
from imas_mcp.search.engines.semantic_engine import SemanticSearchEngine
from imas_mcp.search.engines.lexical_engine import LexicalSearchEngine
from imas_mcp.search.engines.hybrid_engine import HybridSearchEngine
from imas_mcp.search.ai_enhancer import ai_enhancer

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
    """Tool for searching IMAS data paths."""

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

    @mcp_tool("Search for IMAS data paths with relevance-ordered results")
    @ai_enhancer(temperature=0.3, max_tokens=800)
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

        Search modes for LLM usage:
        - "auto": Automatically selects best mode based on query
        - "semantic": Concept-based search using AI embeddings
        - "lexical": Fast text-based keyword search
        - "hybrid": Combines semantic and lexical approaches

        Args:
            query: Search term(s), physics concept, symbol, or pattern
            ids_name: Optional specific IDS to search within
            max_results: Maximum number of results to return (1-100)
            search_mode: Search mode string - must be "auto", "semantic", "lexical", or "hybrid"
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary containing search results with paths, descriptions, and metadata
        """
        try:
            # Validate search mode
            if search_mode not in ["auto", "semantic", "lexical", "hybrid"]:
                return self._create_error_response(
                    f"Invalid search mode: {search_mode}. Must be auto, semantic, lexical, or hybrid.",
                    query,
                )

            # Validate max_results
            if not (1 <= max_results <= 100):
                return self._create_error_response(
                    f"Invalid max_results: {max_results}. Must be between 1 and 100.",
                    query,
                )

            # Convert search mode string to enum
            mode_map = {
                "auto": SearchMode.AUTO,
                "semantic": SearchMode.SEMANTIC,
                "lexical": SearchMode.LEXICAL,
                "hybrid": SearchMode.HYBRID,
            }
            search_mode_enum = mode_map[search_mode]

            # Create search configuration
            config = SearchConfig(
                mode=search_mode_enum,
                max_results=max_results,
                filter_ids=[ids_name] if ids_name else None,
                similarity_threshold=0.0,
            )

            # Execute search using search service
            logger.info(
                f"Executing search: query='{query}' mode={search_mode} max_results={max_results}"
            )
            results = await self._search_service.search(query, config)

            # Convert results to expected format
            response = self._format_search_response(results, search_mode, query)

            logger.info(f"Search completed: {len(results)} results returned")
            return response

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return self._create_error_response(str(e), query)

    def _format_search_response(
        self,
        results: List[SearchResult],
        search_mode: str,
        query: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """Format search results into expected response format."""
        formatted_results = []

        for result in results:
            # Use the SearchResult.to_dict() method for consistency
            result_dict = result.to_dict()

            # Ensure we have the expected fields with fallbacks
            formatted_result = {
                "path": result_dict.get("path", ""),
                "description": result_dict.get("documentation", ""),
                "relevance_score": result_dict.get("relevance_score", result.score),
                "rank": result_dict.get("rank", result.rank),
                "metadata": {
                    "ids_name": result_dict.get("ids_name", ""),
                    "data_type": result_dict.get("data_type", ""),
                    "physics_domain": result_dict.get("physics_domain", "general"),
                    "units": result_dict.get("units", ""),
                },
            }

            # Add highlights if available
            if result_dict.get("highlights"):
                formatted_result["context"] = result_dict["highlights"]

            formatted_results.append(formatted_result)

        return {
            "results": formatted_results,
            "results_count": len(formatted_results),
            "search_mode": search_mode,
            "query": query,
            "max_results": len(formatted_results),  # Actual results returned
        }

    def _create_error_response(
        self, error_message: str, query: Union[str, List[str]]
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "error": error_message,
            "query": query,
            "results": [],
            "results_count": 0,
        }
