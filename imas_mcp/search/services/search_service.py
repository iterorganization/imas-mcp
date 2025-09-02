"""
Search service for IMAS MCP.

This module provides the main search service that orchestrates different
search engines and handles search requests in a clean, testable way.
"""

import logging

from imas_mcp.models.constants import SearchMode
from imas_mcp.search.engines.base_engine import SearchEngine
from imas_mcp.search.search_strategy import SearchConfig, SearchMatch, SearchResponse

logger = logging.getLogger(__name__)


class SearchService:
    """Central search service that orchestrates different search engines.

    This service provides a clean interface for executing searches while
    abstracting the complexity of engine selection and result processing.
    It follows the service pattern for better separation of concerns.
    """

    def __init__(self, engines: dict[SearchMode, SearchEngine] | None = None):
        """Initialize search service with available engines.

        Args:
            engines: Dictionary mapping search modes to engine instances.
                    If None, will use default mock engines for testing.
        """
        self.engines = engines or self._create_default_engines()
        self.logger = logger

    def _create_default_engines(self) -> dict[SearchMode, SearchEngine]:
        """Create default engines for testing and development.

        Returns:
            Dictionary with mock engines for each search mode
        """
        from imas_mcp.search.engines.base_engine import MockSearchEngine

        mock_engine = MockSearchEngine()
        return {
            SearchMode.SEMANTIC: mock_engine,
            SearchMode.LEXICAL: mock_engine,
            SearchMode.HYBRID: mock_engine,
            SearchMode.AUTO: mock_engine,  # Will be resolved to specific mode
        }

    async def search(
        self, query: str | list[str], config: SearchConfig
    ) -> SearchResponse:
        """Execute search request with given configuration.

        Args:
            query: Search query string or list of strings
            config: Search configuration including mode, filters, limits

        Returns:
            SearchResponse with search hits limited to max_results

        Raises:
            SearchServiceError: When search execution fails
        """
        try:
            # Resolve AUTO mode to specific mode
            resolved_mode = self._resolve_search_mode(query, config)

            # Get appropriate engine
            engine = self._get_engine(resolved_mode)

            # Execute search - engines now return SearchResponse
            search_result = await engine.search(query, config)

            # Post-process hits if needed
            processed_hits = self._post_process_results(search_result.hits, config)

            # Log search execution
            self.logger.info(
                f"Search completed: mode={resolved_mode.value} "
                f"query_len={len(str(query))} hits={len(processed_hits)}"
            )

            return SearchResponse(hits=processed_hits)

        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            self.logger.error(error_msg)
            raise SearchServiceError(error_msg, query) from e

    def _resolve_search_mode(
        self, query: str | list[str], config: SearchConfig
    ) -> SearchMode:
        """Resolve AUTO search mode to specific mode based on query analysis.

        Args:
            query: Search query to analyze
            config: Search configuration

        Returns:
            Resolved search mode (never AUTO)
        """
        if config.search_mode != SearchMode.AUTO:
            return config.search_mode

        # Simple heuristic for mode selection
        # In a full implementation, this would use the SearchModeSelector
        query_str = query if isinstance(query, str) else " ".join(query)

        # Check for technical patterns
        if "/" in query_str or "_" in query_str:
            return SearchMode.LEXICAL

        # Check for conceptual patterns
        conceptual_keywords = ["what", "how", "explain", "describe", "physics"]
        if any(keyword in query_str.lower() for keyword in conceptual_keywords):
            return SearchMode.SEMANTIC

        # Default to hybrid for mixed queries
        return SearchMode.HYBRID

    def _get_engine(self, mode: SearchMode) -> SearchEngine:
        """Get search engine for specified mode.

        Args:
            mode: Search mode to get engine for

        Returns:
            SearchEngine instance for the mode

        Raises:
            SearchServiceError: When engine is not available
        """
        if mode not in self.engines:
            raise SearchServiceError(
                f"Search engine not available for mode: {mode.value}", ""
            )

        return self.engines[mode]

    def _post_process_results(
        self, results: list[SearchMatch], config: SearchConfig
    ) -> list[SearchMatch]:
        """Post-process search results based on configuration.

        Args:
            results: Raw search results from engine
            config: Search configuration with processing options

        Returns:
            Processed and filtered results
        """
        # Apply result limit
        processed = results[: config.max_results]

        # Apply similarity threshold if specified
        if config.similarity_threshold > 0:
            processed = [
                result
                for result in processed
                if result.score >= config.similarity_threshold
            ]

        # Re-rank if needed (update rank field)
        for rank, result in enumerate(processed):
            result.rank = rank

        return processed

    def get_available_modes(self) -> list[SearchMode]:
        """Get list of available search modes.

        Returns:
            List of SearchMode enums for available engines
        """
        return [mode for mode in self.engines.keys() if mode != SearchMode.AUTO]

    def register_engine(self, mode: SearchMode, engine: SearchEngine) -> None:
        """Register a new search engine for a specific mode.

        Args:
            mode: Search mode to register engine for
            engine: SearchEngine instance to register
        """
        self.engines[mode] = engine
        self.logger.info(
            f"Registered {engine.get_engine_type()} engine for {mode.value} mode"
        )

    def health_check(self) -> dict[str, bool]:
        """Check health status of all registered engines.

        Returns:
            Dictionary mapping mode names to health status
        """
        health_status = {}

        for mode, engine in self.engines.items():
            try:
                # Simple health check - engine should have required methods
                _ = engine.get_engine_type()  # Just verify method exists
                health_status[mode.value] = True
            except Exception as e:
                self.logger.warning(f"Health check failed for {mode.value} engine: {e}")
                health_status[mode.value] = False

        return health_status


class SearchServiceError(Exception):
    """Exception raised when search service operations fail."""

    def __init__(self, message: str, query: str | list[str] = ""):
        """Initialize search service error.

        Args:
            message: Error description
            query: Query that caused the error (optional)
        """
        self.query = query
        super().__init__(message)


class SearchRequest:
    """Structured search request for service interface."""

    def __init__(
        self,
        query: str | list[str],
        mode: SearchMode = SearchMode.AUTO,
        max_results: int = 10,
        ids_filter: list[str] | None = None,
        similarity_threshold: float = 0.0,
    ):
        """Initialize search request.

        Args:
            query: Search query
            mode: Search mode to use
            max_results: Maximum number of results
            ids_filter: IDS names to filter by
            similarity_threshold: Minimum similarity score
        """
        self.query = query
        self.config = SearchConfig(
            search_mode=mode,
            max_results=max_results,
            ids_filter=ids_filter,
            similarity_threshold=similarity_threshold,
        )
