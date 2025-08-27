"""
Lexical search engine implementation for IMAS MCP.

T            if config.ids_filter:
                # Add IDS filter to query
                ids_filter_query = " OR ".join(
                    [f"ids_name:{ids}" for ids in config.ids_filter]
                )odule provides full-text search capabilities using SQLite FTS5
for exact matching and keyword-based search in the IMAS data dictionary.
"""

import logging
import re

from imas_mcp.models.constants import SearchMode
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.engines.base_engine import SearchEngine, SearchEngineError
from imas_mcp.search.search_strategy import SearchConfig, SearchMatch, SearchResponse

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
        self, query: str | list[str], config: SearchConfig
    ) -> SearchResponse:
        """Execute lexical search using full-text search.

        Args:
            query: Search query string or list of strings
            config: Search configuration with parameters

        Returns:
            SearchResponse with limited hits and all matching paths

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

            # Apply intelligent path parsing for automatic IDS filtering
            enhanced_config = self._enhance_config_with_path_intelligence(
                query_str, config
            )

            # Check if this is an exact path query and handle specially
            exact_path_results = self._try_exact_path_search(query_str, enhanced_config)
            if exact_path_results is not None:
                return exact_path_results

            # Apply IDS filtering if specified (including auto-detected)
            if enhanced_config.ids_filter:
                # Add IDS filter to query
                ids_filter = " OR ".join(
                    [f"ids_name:{ids}" for ids in enhanced_config.ids_filter]
                )
                query_str = f"({query_str}) AND ({ids_filter})"

            # Execute full-text search
            documents = self.document_store.search_full_text(
                query_str, max_results=enhanced_config.max_results
            )

            # Convert to SearchMatch objects
            results = []
            for rank, doc in enumerate(documents):
                # Calculate simple ranking score based on position
                score = 1.0 - (rank / max(len(documents), 1))

                result = SearchMatch(
                    document=doc,
                    score=score,
                    rank=rank,
                    search_mode=SearchMode.LEXICAL,
                    highlights="",  # FTS5 could provide highlights in future
                )
                results.append(result)

            # Log search execution
            self.log_search_execution(query, enhanced_config, len(results))

            return SearchResponse(hits=results)

        except Exception as e:
            error_msg = f"Lexical search failed: {str(e)}"
            self.logger.error(error_msg)
            raise SearchEngineError(self.name, error_msg, str(query)) from e

    def _extract_ids_from_path(self, query: str) -> str | None:
        """Extract IDS name from path-like query for intelligent filtering.

        This method recognizes path-like queries and extracts the IDS name
        from the first component, enabling automatic IDS filtering for more
        efficient lexical search. Validates against actual IDS catalog.

        Args:
            query: Search query string that might be a path

        Returns:
            IDS name if path-like query detected and valid, None otherwise
        """
        # Check if query looks like a path (contains forward slashes)
        if "/" not in query:
            return None

        # Extract first component before the first slash
        parts = query.strip().split("/")
        if len(parts) < 2:
            return None

        potential_ids = parts[0].strip()

        # Validate that it looks like a valid IDS name
        # IDS names typically contain lowercase letters, numbers, and underscores
        if not re.match(r"^[a-z][a-z0-9_]*$", potential_ids):
            return None

        # Validate against actual IDS catalog using document store
        try:
            available_ids = self.document_store.get_available_ids()
            if potential_ids not in available_ids:
                self.logger.debug(
                    f"Path intelligence: '{potential_ids}' not found in IDS catalog "
                    f"(available: {len(available_ids)} IDS)"
                )
                return None

            self.logger.debug(
                f"Path intelligence: extracted IDS '{potential_ids}' from query '{query}'"
            )
            return potential_ids

        except Exception as e:
            self.logger.warning(f"Failed to validate IDS against catalog: {e}")
            return None

    def _try_exact_path_search(
        self, query_str: str, config: SearchConfig
    ) -> SearchResponse | None:
        """Try exact path search for full IMAS path queries.

        This method detects full IMAS paths and performs direct lookup
        for exact matches, ensuring single results for valid paths.

        Args:
            query_str: Search query string
            config: Search configuration

        Returns:
            SearchResponse if exact path found, None otherwise
        """
        # Check if this looks like a full IMAS path
        if not self._is_full_imas_path(query_str):
            return None

        # Try direct document lookup first
        document = self.document_store.get_document(query_str)
        if document:
            # Found exact match
            result = SearchMatch(
                document=document,
                score=1.0,  # Perfect match
                rank=0,
                search_mode=SearchMode.LEXICAL,
                highlights="",
            )
            self.log_search_execution(query_str, config, 1)
            return SearchResponse(hits=[result])

        # Try FTS5 with quoted exact match
        try:
            # Escape the path for FTS5 and use quotes for exact matching
            escaped_query = self._escape_path_for_fts(query_str)
            documents = self.document_store.search_full_text(
                escaped_query, max_results=config.max_results
            )

            if documents:
                results = []
                for rank, doc in enumerate(documents):
                    # Prioritize exact path matches
                    score = (
                        1.0 if doc.metadata.path_id == query_str else 0.8 - (rank * 0.1)
                    )

                    result = SearchMatch(
                        document=doc,
                        score=max(score, 0.1),  # Minimum score
                        rank=rank,
                        search_mode=SearchMode.LEXICAL,
                        highlights="",
                    )
                    results.append(result)

                # Sort by score (exact matches first)
                results.sort(key=lambda x: x.score, reverse=True)
                self.log_search_execution(query_str, config, len(results))
                return SearchResponse(hits=results)

        except Exception as e:
            self.logger.debug(f"Exact path FTS search failed: {e}")

        return None

    def _is_full_imas_path(self, query_str: str) -> bool:
        """Check if query looks like a full IMAS path.

        Args:
            query_str: Query string to check

        Returns:
            True if it looks like a full IMAS path
        """
        # Must contain slashes
        if "/" not in query_str:
            return False

        # Must have at least 2 path components
        parts = query_str.split("/")
        if len(parts) < 2:
            return False

        # First part should be a valid IDS name - use the existing validation logic
        potential_ids = self._extract_ids_from_path(query_str)
        return potential_ids is not None

    def _escape_path_for_fts(self, path: str) -> str:
        """Escape IMAS path for FTS5 exact matching.

        Args:
            path: IMAS path to escape

        Returns:
            FTS5-safe escaped path query
        """
        # Use double quotes for exact phrase matching
        # Replace slashes with spaces for better FTS5 matching
        escaped = path.replace("/", " ")
        return f'"{escaped}"'

    def _enhance_config_with_path_intelligence(
        self, query_str: str, config: SearchConfig
    ) -> SearchConfig:
        """Enhance search config with intelligent path parsing.

        Automatically extracts IDS names from path-like queries and applies
        them as filters for more efficient lexical search.

        Args:
            query_str: Search query string to analyze
            config: Original search configuration

        Returns:
            Enhanced SearchConfig with intelligent IDS filtering
        """
        # If user already provided IDS filter, respect it
        if config.ids_filter:
            return config

        # Try to extract IDS from path-like query
        extracted_ids = self._extract_ids_from_path(query_str)

        if extracted_ids:
            # Create new config with extracted IDS filter
            enhanced_config = config.model_copy()
            enhanced_config.ids_filter = [extracted_ids]
            return enhanced_config

        return config

    def get_engine_type(self) -> str:
        """Get the type identifier for this engine."""
        return "lexical"

    def is_suitable_for_query(self, query: str | list[str]) -> bool:
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
