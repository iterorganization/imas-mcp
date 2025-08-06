"""Search configuration service."""

from typing import Union, List, Optional

from imas_mcp.search.search_strategy import SearchConfig
from imas_mcp.models.constants import SearchMode
from .base import BaseService


class SearchConfigurationService(BaseService):
    """Service for search configuration and optimization."""

    def create_config(
        self,
        search_mode: Union[str, SearchMode] = "auto",
        max_results: int = 10,
        ids_filter: Optional[Union[str, List[str]]] = None,
    ) -> SearchConfig:
        """Create optimized search configuration."""

        # Convert string to SearchMode enum if needed
        if isinstance(search_mode, str):
            search_mode = SearchMode(search_mode)

        # Convert ids_filter to proper format
        if isinstance(ids_filter, str):
            ids_filter = [ids_filter]

        # Optimize max_results based on mode
        if search_mode == SearchMode.SEMANTIC and max_results > 20:
            self.logger.warning(
                f"Large result set ({max_results}) may impact semantic search performance"
            )

        return SearchConfig(
            search_mode=search_mode,
            max_results=max_results,
            ids_filter=ids_filter,
            similarity_threshold=0.0,
        )

    def optimize_for_query(
        self, query: Union[str, List[str]], base_config: SearchConfig
    ) -> SearchConfig:
        """Optimize configuration based on query characteristics."""

        query_str = query if isinstance(query, str) else " ".join(query)

        # Adjust search mode based on query complexity
        if len(query_str.split()) > 5:
            # Complex queries benefit from semantic search
            base_config.search_mode = SearchMode.SEMANTIC
        elif any(op in query_str.upper() for op in ["AND", "OR", "NOT"]):
            # Boolean queries work better with lexical search
            base_config.search_mode = SearchMode.LEXICAL

        return base_config
