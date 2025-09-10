"""
Semantic search engine implementation for IMAS MCP.

This module provid            results = self.semantic_search.search(
                query=query_str,
                top_k=config.max_results,
                ids_filter=config.ids_filter,
                similarity_threshold=config.similarity_threshold,
            )antic search capabilities using sentence transformers
for embedding-based similarity search in the IMAS data dictionary.
"""

import logging

from imas_mcp.models.constants import SearchMode
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.engines.base_engine import SearchEngine, SearchEngineError
from imas_mcp.search.search_strategy import (
    SearchConfig,
    SearchMatch,
    SearchResponse,
)

logger = logging.getLogger(__name__)


class SemanticSearchEngine(SearchEngine):
    """Semantic search engine using sentence transformers.

    This engine provides semantic search capabilities by using pre-trained
    sentence transformer models to find semantically similar content in
    the IMAS data dictionary.
    """

    def __init__(self, document_store: DocumentStore, use_rich: bool = True):
        """Initialize semantic search engine.

        Args:
            document_store: Document store containing IMAS data
        """
        super().__init__("semantic")
        self.document_store = document_store
        self._use_rich = use_rich
        self._semantic_search = None

    @property
    def semantic_search(self):
        """Lazy initialization of semantic search."""
        if self._semantic_search is None:
            # Local imports defer heavy model/embedding dependencies until needed
            from imas_mcp.embeddings.embeddings import Embeddings
            from imas_mcp.search.semantic_search import (
                SemanticSearch,
                SemanticSearchConfig,
            )

            # Create config that matches document store's ids_set
            config = SemanticSearchConfig(ids_set=self.document_store.ids_set)
            embeddings = Embeddings(
                document_store=self.document_store,
                ids_set=self.document_store.ids_set,
                use_rich=self._use_rich,
            )  # Synchronous initialization now
            self._semantic_search = SemanticSearch(
                config=config, document_store=self.document_store, embeddings=embeddings
            )
        return self._semantic_search

    async def search(
        self, query: str | list[str], config: SearchConfig
    ) -> SearchResponse:
        """Execute semantic search using sentence transformers.

        Args:
            query: Search query string or list of strings
            config: Search configuration with parameters

        Returns:
            SearchResponse with search hits limited to max_results

        Raises:
            SearchEngineError: When semantic search execution fails
        """
        try:
            # Validate query
            if not self.validate_query(query):
                raise SearchEngineError(
                    self.name, f"Invalid query: {query}", str(query)
                )

            # Normalize query to string format
            query_str = self.normalize_query(query)

            # Execute semantic search
            semantic_results = self.semantic_search.search(
                query=query_str,
                top_k=config.max_results,
                ids_filter=config.ids_filter,
                similarity_threshold=config.similarity_threshold,
            )

            # Convert to SearchMatch objects
            results = []
            for rank, semantic_result in enumerate(semantic_results):
                result = SearchMatch(
                    document=semantic_result.document,
                    score=semantic_result.similarity_score,
                    rank=rank,
                    search_mode=SearchMode.SEMANTIC,
                    highlights="",  # Could add semantic highlights in future
                )
                results.append(result)

            # Log search execution
            self.log_search_execution(query, config, len(results))

            return SearchResponse(hits=results)

        except Exception as e:
            error_msg = f"Semantic search failed: {str(e)}"
            self.logger.error(error_msg)
            raise SearchEngineError(self.name, error_msg, str(query)) from e

    def get_engine_type(self) -> str:
        """Get the type identifier for this engine."""
        return "semantic"

    def is_suitable_for_query(self, query: str | list[str]) -> bool:
        """Check if this engine is suitable for the given query.

        Semantic search is particularly good for:
        - Conceptual queries
        - Physics domain questions
        - Natural language descriptions

        Args:
            query: Query to evaluate

        Returns:
            True if semantic search is recommended for this query
        """
        query_str = self.normalize_query(query).lower()

        # Exclude technical path queries (these are better for lexical)
        if "/" in query_str or "_" in query_str:
            return False

        # Conceptual indicators that benefit from semantic search
        conceptual_indicators = [
            "what is",
            "how does",
            "explain",
            "describe",
            "meaning of",
            "physics",
            "plasma",
            "temperature",
            "density",
            "magnetic field",
            "understand",
            "concept",
            "theory",
        ]

        return any(indicator in query_str for indicator in conceptual_indicators)

    def get_health_status(self) -> dict:
        """Get health status of semantic search components.

        Returns:
            Dictionary with health status information
        """
        try:
            # Check if semantic search can be initialized
            _ = self.semantic_search

            # Check document store availability
            doc_count = len(self.document_store.get_all_documents())

            return {
                "status": "healthy",
                "engine_type": self.get_engine_type(),
                "document_count": doc_count,
                "semantic_search_available": True,
                "ids_set": list(self.document_store.ids_set or set()),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "engine_type": self.get_engine_type(),
                "error": str(e),
                "semantic_search_available": False,
            }
