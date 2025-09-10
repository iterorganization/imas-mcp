"""
Hybrid search engine implementation for IMAS MCP.

This module provides hybrid search capabilities that combine semantic
and lexical search for optimal results in the IMAS data dictionary.
"""

import logging

from imas_mcp.models.constants import SearchMode
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.engines.base_engine import SearchEngine, SearchEngineError
from imas_mcp.search.engines.lexical_engine import LexicalSearchEngine
from imas_mcp.search.engines.semantic_engine import SemanticSearchEngine
from imas_mcp.search.search_strategy import SearchConfig, SearchResponse

logger = logging.getLogger(__name__)


class HybridSearchEngine(SearchEngine):
    """Hybrid search engine combining semantic and lexical search.

    This engine provides the best of both worlds by combining semantic
    understanding with exact keyword matching for comprehensive search results.
    """

    def __init__(self, document_store: DocumentStore, use_rich: bool = True):
        """Initialize hybrid search engine.

        Args:
            document_store: Document store containing IMAS data
        """
        super().__init__("hybrid")
        self.document_store = document_store
        self.semantic_engine = SemanticSearchEngine(document_store, use_rich=use_rich)
        self.lexical_engine = LexicalSearchEngine(document_store)

    async def search(
        self, query: str | list[str], config: SearchConfig
    ) -> SearchResponse:
        """Execute hybrid search combining semantic and lexical results.

        Args:
            query: Search query string or list of strings
            config: Search configuration with parameters

        Returns:
            SearchResponse with limited hits and all matching paths

        Raises:
            SearchEngineError: When hybrid search execution fails
        """
        try:
            # Validate query
            if not self.validate_query(query):
                raise SearchEngineError(
                    self.name, f"Invalid query: {query}", str(query)
                )

            # Get results from both engines
            semantic_results = await self.semantic_engine.search(query, config)
            lexical_results = await self.lexical_engine.search(query, config)

            # Combine and deduplicate results
            combined_results = {}

            # Add semantic results with boosted scores
            for result in semantic_results.hits:
                path_id = result.document.metadata.path_id
                result.score *= 1.2  # Boost semantic scores
                result.search_mode = SearchMode.HYBRID  # Mark as hybrid
                combined_results[path_id] = result

            # Add lexical results, boosting if they match semantic results
            for result in lexical_results.hits:
                path_id = result.document.metadata.path_id
                if path_id in combined_results:
                    # Boost score for documents found in both searches
                    combined_results[path_id].score = (
                        combined_results[path_id].score * 0.7 + result.score * 0.3 + 0.1
                    )
                    combined_results[path_id].search_mode = SearchMode.HYBRID
                else:
                    result.search_mode = SearchMode.HYBRID  # Mark as hybrid
                    combined_results[path_id] = result

            # Sort by score and re-rank
            sorted_results = sorted(
                combined_results.values(), key=lambda x: x.score, reverse=True
            )

            # Update rankings and ensure all are marked as hybrid
            final_results = []
            for rank, result in enumerate(sorted_results[: config.max_results]):
                result.rank = rank
                result.search_mode = SearchMode.HYBRID
                final_results.append(result)

            # Log search execution
            self.log_search_execution(query, config, len(final_results))

            return SearchResponse(hits=final_results)

        except Exception as e:
            error_msg = f"Hybrid search failed: {str(e)}"
            self.logger.error(error_msg)
            raise SearchEngineError(self.name, error_msg, str(query)) from e

    def get_engine_type(self) -> str:
        """Get the type identifier for this engine."""
        return "hybrid"

    def is_suitable_for_query(self, query: str | list[str]) -> bool:
        """Check if this engine is suitable for the given query.

        Hybrid search is suitable for most queries, especially:
        - Mixed technical and conceptual queries
        - Queries that benefit from both exact matching and semantic understanding
        - General exploration queries
        - Fallback when no specific engine is ideal

        Args:
            query: Query to evaluate

        Returns:
            True if hybrid search is recommended
        """
        # Check if both engines find the query suitable
        semantic_suitable = self.semantic_engine.is_suitable_for_query(query)
        lexical_suitable = self.lexical_engine.is_suitable_for_query(query)

        # Hybrid is especially good when both engines are suitable
        if semantic_suitable and lexical_suitable:
            return True

        # Also good for general queries that don't clearly favor one approach
        if not semantic_suitable and not lexical_suitable:
            return True

        # For mixed queries, hybrid can still be beneficial even if one engine dominates
        query_str = self.normalize_query(query).lower()
        mixed_indicators = ["and", "with", "including", "about", "related to"]
        if any(indicator in query_str for indicator in mixed_indicators):
            return True

        # Default fallback - hybrid can handle any query
        return True

    def get_combination_strategy(self, query: str | list[str]) -> str:
        """Determine the best combination strategy for the query.

        Args:
            query: Query to analyze

        Returns:
            Strategy name for result combination
        """
        _ = self.normalize_query(
            query
        ).lower()  # query_str - unused, kept for potential future strategy logic

        # For technical queries, favor lexical results
        if self.lexical_engine.is_suitable_for_query(query):
            return "lexical_dominant"

        # For conceptual queries, favor semantic results
        elif self.semantic_engine.is_suitable_for_query(query):
            return "semantic_dominant"

        # For balanced queries, use equal weighting
        else:
            return "balanced"

    def get_health_status(self) -> dict:
        """Get health status of hybrid search components.

        Returns:
            Dictionary with health status information
        """
        try:
            # Get health status from both engines
            semantic_health = self.semantic_engine.get_health_status()
            lexical_health = self.lexical_engine.get_health_status()

            # Hybrid is healthy if both engines are healthy
            overall_healthy = (
                semantic_health.get("status") == "healthy"
                and lexical_health.get("status") == "healthy"
            )

            return {
                "status": "healthy" if overall_healthy else "unhealthy",
                "engine_type": self.get_engine_type(),
                "semantic_engine": semantic_health,
                "lexical_engine": lexical_health,
                "document_count": semantic_health.get("document_count", 0),
                "ids_set": semantic_health.get("ids_set", []),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "engine_type": self.get_engine_type(),
                "error": str(e),
                "semantic_engine": {"status": "unknown"},
                "lexical_engine": {"status": "unknown"},
            }
