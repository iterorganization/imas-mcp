"""Search configuration service."""

from imas_mcp.models.constants import SearchMode
from imas_mcp.search.search_strategy import SearchConfig

from .base import BaseService


class SearchConfigurationService(BaseService):
    """Service for search configuration and optimization."""

    def create_config(
        self,
        search_mode: str | SearchMode = "auto",
        max_results: int = 50,
        ids_filter: str | list[str] | None = None,
    ) -> SearchConfig:
        """Create optimized search configuration."""

        # Convert string to SearchMode enum if needed
        if isinstance(search_mode, str):
            search_mode = SearchMode(search_mode)

        # Convert ids_filter to proper format
        if isinstance(ids_filter, str):
            ids_filter = [ids_filter]

        # No max_results limits for LLM optimization - removed performance warning

        return SearchConfig(
            search_mode=search_mode,
            max_results=max_results,
            ids_filter=ids_filter,
            similarity_threshold=0.0,
        )

    def optimize_for_query(
        self, query: str | list[str], base_config: SearchConfig
    ) -> SearchConfig:
        """Optimize configuration based on query characteristics.

        Only optimizes AUTO mode - explicit user choices are always respected.
        """

        # Never override explicit user choices - only optimize AUTO mode
        if base_config.search_mode != SearchMode.AUTO:
            self.logger.debug(
                f"Preserving explicit search mode: {base_config.search_mode.value}"
            )
            return base_config

        query_str = query if isinstance(query, str) else " ".join(query)

        # Only optimize when mode is AUTO - adjust search mode based on query characteristics
        if len(query_str.split()) > 5:
            # Complex queries benefit from semantic search
            base_config.search_mode = SearchMode.SEMANTIC
            self.logger.debug("AUTO mode: Selected SEMANTIC for complex query")
        elif self._has_boolean_operators(query_str):
            # Boolean queries work better with lexical search
            base_config.search_mode = SearchMode.LEXICAL
            self.logger.debug("AUTO mode: Selected LEXICAL for boolean query")
        else:
            # Default to hybrid for balanced results
            base_config.search_mode = SearchMode.HYBRID
            self.logger.debug("AUTO mode: Selected HYBRID as default")

        return base_config

    def _has_boolean_operators(self, query: str) -> bool:
        """Check if query contains boolean operators as whole words."""
        import re

        # Match boolean operators as whole words (case insensitive)
        boolean_pattern = r"\b(?:AND|OR|NOT)\b"
        return bool(re.search(boolean_pattern, query, re.IGNORECASE))
