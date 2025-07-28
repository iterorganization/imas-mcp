"""
Lexical search engine implementation for IMAS MCP.

This module provides full-text search capabilities using SQLite FTS5
for exact matching and keyword-based search in the IMAS data dictionary.
"""

import logging
from typing import List, Union

from imas_mcp.search.engines.base_engine import SearchEngine, SearchEngineError
from imas_mcp.search.search_strategy import SearchConfig, SearchResult
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.models.enums import SearchMode

logger = logging.getLogger(__name__)


class LexicalSearchEngine(SearchEngine):
    """Lexical search engine using SQLite FTS5.

    This engine provides full-text search capabilities for exact matching
    and keyword-based search in the IMAS data dictionary.
    """

    def __init__(self, document_store: DocumentStore):
        """Initialize lexical search engine.

        Args:
            document_store: Document store containing IMAS data
        """
        super().__init__("lexical")
        self.document_store = document_store

    async def search(
        self, query: Union[str, List[str]], config: SearchConfig
    ) -> List[SearchResult]:
        """Execute lexical search using full-text search.

        Args:
            query: Search query string or list of strings
            config: Search configuration with parameters

        Returns:
            List of SearchResult objects ordered by relevance

        Raises:
            SearchEngineError: When lexical search execution fails
        """
        try:
            # Validate query
            if not self.validate_query(query):
                raise SearchEngineError(
                    self.name, f"Invalid query: {query}", str(query)
                )

            # Convert query to string format
            query_str = self.normalize_query(query)

            # Apply IDS filtering if specified
            if config.filter_ids:
                # Add IDS filter to query
                ids_filter = " OR ".join(
                    [f"ids_name:{ids}" for ids in config.filter_ids]
                )
                query_str = f"({query_str}) AND ({ids_filter})"

            # Execute full-text search
            documents = self.document_store.search_full_text(
                query_str, max_results=config.max_results
            )

            # Convert to SearchResult objects
            results = []
            for rank, doc in enumerate(documents):
                # Calculate simple ranking score based on position
                score = 1.0 - (rank / max(len(documents), 1))

                result = SearchResult(
                    document=doc,
                    score=score,
                    rank=rank,
                    search_mode=SearchMode.LEXICAL,
                    highlights="",  # FTS5 could provide highlights in future
                )
                results.append(result)

            # Log search execution
            self.log_search_execution(query, config, len(results))

            return results

        except Exception as e:
            error_msg = f"Lexical search failed: {str(e)}"
            self.logger.error(error_msg)
            raise SearchEngineError(self.name, error_msg, str(query)) from e

    def get_engine_type(self) -> str:
        """Get the type identifier for this engine."""
        return "lexical"

    def is_suitable_for_query(self, query: Union[str, List[str]]) -> bool:
        """Check if this engine is suitable for the given query.

        Lexical search is particularly good for:
        - Exact term matching
        - Technical path queries
        - IMAS-specific terminology
        - Queries with explicit operators

        Args:
            query: Query to evaluate

        Returns:
            True if lexical search is recommended for this query
        """
        query_str = self.normalize_query(query)

        # Check for explicit technical operators
        explicit_operators = [
            "units:",
            "documentation:",
            "ids_name:",
            "path:",
            "AND",
            "OR",
            "NOT",
            '"',
            "*",
            "~",
        ]
        if any(operator in query_str for operator in explicit_operators):
            return True

        # Check for path-like queries (contains / and looks like IMAS paths)
        if "/" in query_str:
            path_indicators = [
                "profiles_1d",
                "profiles_2d",
                "time_slice",
                "global_quantities",
                "core_profiles",
                "equilibrium",
                "transport",
                "mhd",
                "wall",
            ]
            if any(indicator in query_str.lower() for indicator in path_indicators):
                return True

        # Check for IMAS-specific technical terms
        imas_technical_terms = [
            "core_profiles",
            "equilibrium",
            "transport",
            "mhd",
            "wall",
            "profiles_1d",
            "profiles_2d",
            "time_slice",
            "global_quantities",
        ]
        if any(term in query_str.lower() for term in imas_technical_terms):
            return True

        # Check for underscore-separated technical terms (common in IMAS)
        if "_" in query_str and len(query_str.split("_")) > 1:
            words = query_str.lower().split("_")
            technical_words = [
                "profiles",
                "time",
                "slice",
                "global",
                "quantities",
                "1d",
                "2d",
                "rho",
                "tor",
                "norm",
                "psi",
                "flux",
                "coord",
                "grid",
            ]
            if any(word in technical_words for word in words):
                return True

        # Simple technical terms that suggest exact matching is needed
        simple_technical_terms = ["time", "temperature", "density", "field"]
        if any(term in query_str.lower() for term in simple_technical_terms):
            return True

        return False

    def prepare_query_for_fts(self, query: str) -> str:
        """Prepare query string for FTS5 search.

        Args:
            query: Raw query string

        Returns:
            FTS5-optimized query string
        """
        # Basic FTS5 query preparation
        # In a full implementation, this would handle:
        # - Escaping special characters
        # - Converting to FTS5 query syntax
        # - Adding boost factors for exact matches

        # For now, just return the cleaned query
        return query.strip()

    def get_health_status(self) -> dict:
        """Get health status of lexical search components.

        Returns:
            Dictionary with health status information
        """
        try:
            # Check document store availability
            doc_count = len(self.document_store.get_all_documents())

            # Test a simple search to verify FTS5 functionality
            _ = self.document_store.search_full_text(
                "test", max_results=1
            )  # test_results - unused, just checking FTS5 works
            fts_available = True

            return {
                "status": "healthy",
                "engine_type": self.get_engine_type(),
                "document_count": doc_count,
                "fts5_available": fts_available,
                "ids_set": list(self.document_store.ids_set or set()),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "engine_type": self.get_engine_type(),
                "error": str(e),
                "fts5_available": False,
            }
