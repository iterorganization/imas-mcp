"""
Search modes and composition patterns for IMAS MCP server.

This module provides different search strategies and modes for querying the
IMAS data dictionary, following composition patterns for maintainability
and extensibility.
"""

import logging
import re
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, field_validator

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.context_models import SearchParameters
from imas_mcp.search.document_store import Document, DocumentStore

logger = logging.getLogger(__name__)


class SearchConfig(SearchParameters):
    """Configuration for search operations."""

    # Override base class fields to make them required and set proper defaults
    search_mode: SearchMode = SearchMode.AUTO
    max_results: int = 50

    # Additional search-specific configuration
    similarity_threshold: float = 0.0
    boost_exact_matches: bool = True

    @field_validator("search_mode", mode="before")
    @classmethod
    def validate_search_mode(cls, v):
        """Convert string to SearchMode enum if needed.

        Accepts both string values ('auto', 'semantic', 'lexical', 'hybrid')
        and SearchMode enum instances. Always returns SearchMode enum.
        """
        if isinstance(v, str):
            # Create mapping from string values to enum members
            value_map = {member.value: member for member in SearchMode}
            if v not in value_map:
                valid_values = list(value_map.keys())
                raise ValueError(
                    f"Invalid search_mode: {v}. Valid options: {valid_values}"
                )
            return value_map[v]
        return v

    @field_validator("ids_filter", mode="before")
    @classmethod
    def validate_ids_filter(cls, v):
        """Convert string or list to list of strings for IDS filtering.

        Accepts:
        - None: No filtering
        - str: Single IDS name or space-separated IDS names
        - List[str]: List of IDS names

        Always returns Optional[List[str]].
        """
        if v is None:
            return None

        if isinstance(v, str):
            # Handle space-separated string or single IDS
            return v.split() if " " in v else [v]

        if isinstance(v, list):
            # Validate all items are strings
            if not all(isinstance(item, str) for item in v):
                raise ValueError("All items in ids_filter list must be strings")
            return v

        raise ValueError(
            f"ids_filter must be None, string, or list of strings, got {type(v)}"
        )


class SearchBase(BaseModel):
    """Base class for all search-related models with common metadata."""

    # Core search metadata shared by all search models
    score: float = Field(description="Relevance/similarity score")
    rank: int = Field(description="Rank/position in search results")
    search_mode: SearchMode = Field(description="Search mode that found this result")
    highlights: str = Field(default="", description="Highlighted text snippets")


class SearchHit(SearchBase):
    """API-friendly search hit without internal document reference."""

    # Flattened API fields from document metadata
    path: str = Field(description="Full IMAS path")
    documentation: str = Field(description="Path documentation")
    units: str | None = Field(default=None, description="Physical units")
    data_type: str | None = Field(default=None, description="Data type")
    ids_name: str = Field(description="IDS name this path belongs to")
    physics_domain: str | None = Field(default=None, description="Physics domain")


class SearchMatch(SearchBase):
    """Internal search result with document reference for search processing."""

    # Internal document reference for search processing
    document: Document = Field(description="Internal document with full metadata")

    def to_hit(self) -> SearchHit:
        """
        Convert SearchMatch to SearchHit for API responses.

        Flattens document metadata into API-friendly fields while
        excluding the internal document reference for clean API responses.

        Returns:
            SearchHit suitable for API responses
        """
        return SearchHit(
            # Inherited base fields
            score=self.score,
            rank=self.rank,
            search_mode=self.search_mode,
            highlights=self.highlights,
            # Flattened document fields for API
            path=self.document.metadata.path_name,
            documentation=self.document.documentation,
            units=self.document.units.unit_str if self.document.units else None,
            data_type=self.document.metadata.data_type,
            ids_name=self.document.metadata.ids_name,
            physics_domain=self.document.metadata.physics_domain,
        )


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store

    @abstractmethod
    def search(
        self,
        query: str | list[str],
        config: SearchConfig,
    ) -> list[SearchMatch]:
        """Execute search with given query and configuration."""
        pass

    @abstractmethod
    def get_mode(self) -> SearchMode:
        """Get the search mode for this strategy."""
        pass


class LexicalSearchStrategy(SearchStrategy):
    """Full-text search strategy using SQLite FTS5."""

    def search(
        self,
        query: str | list[str],
        config: SearchConfig,
    ) -> list[SearchMatch]:
        """Execute lexical search using full-text search."""
        # Convert query to string format
        query_str = query if isinstance(query, str) else " ".join(query)

        # Apply IDS filtering if specified
        if config.ids_filter:
            # Add IDS filter to query
            ids_filter = " OR ".join([f"ids_name:{ids}" for ids in config.ids_filter])
            query_str = f"({query_str}) AND ({ids_filter})"

        # Execute full-text search
        try:
            documents = self.document_store.search_full_text(
                query_str, max_results=config.max_results
            )

            # Convert to SearchMatch objects
            results = []
            for rank, doc in enumerate(documents):
                result = SearchMatch(
                    document=doc,
                    score=1.0 - (rank / max(len(documents), 1)),  # Simple ranking score
                    rank=rank,
                    search_mode=SearchMode.LEXICAL,
                    highlights="",  # FTS5 could provide highlights in future
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Lexical search failed: {e}")
            return []

    def get_mode(self) -> SearchMode:
        """Get the search mode for this strategy."""
        return SearchMode.LEXICAL


class SemanticSearchStrategy(SearchStrategy):
    """Semantic search strategy using sentence transformers."""

    def __init__(self, document_store: DocumentStore):
        super().__init__(document_store)
        self._semantic_search = None

    @property
    def semantic_search(self):
        """Lazy initialization of semantic search."""
        if self._semantic_search is None:
            from .semantic_search import SemanticSearch, SemanticSearchConfig

            # Create config that matches document store's ids_set
            config = SemanticSearchConfig(ids_set=self.document_store.ids_set)
            self._semantic_search = SemanticSearch(
                config=config, document_store=self.document_store
            )
        return self._semantic_search

    def search(
        self,
        query: str | list[str],
        config: SearchConfig,
    ) -> list[SearchMatch]:
        """Execute semantic search using sentence transformers."""
        # Convert query to string format
        query_str = query if isinstance(query, str) else " ".join(query)

        try:
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
                    highlights="",  # Could add semantic highlights
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def get_mode(self) -> SearchMode:
        """Get the search mode for this strategy."""
        return SearchMode.SEMANTIC


class HybridSearchStrategy(SearchStrategy):
    """Hybrid search strategy combining semantic and lexical search."""

    def __init__(self, document_store: DocumentStore):
        super().__init__(document_store)
        self.semantic_strategy = SemanticSearchStrategy(document_store)
        self.lexical_strategy = LexicalSearchStrategy(document_store)

    def search(
        self,
        query: str | list[str],
        config: SearchConfig,
    ) -> list[SearchMatch]:
        """Execute hybrid search combining semantic and lexical results."""
        # Get results from both strategies
        semantic_results = self.semantic_strategy.search(query, config)
        lexical_results = self.lexical_strategy.search(query, config)

        # Combine and deduplicate results
        combined_results = {}

        # Add semantic results with boosted scores
        for result in semantic_results:
            path_id = result.document.metadata.path_id
            result.score *= 1.2  # Boost semantic scores
            result.search_mode = SearchMode.HYBRID  # Mark as hybrid
            combined_results[path_id] = result

        # Add lexical results, boosting if they match semantic results
        for result in lexical_results:
            path_id = result.document.metadata.path_id
            if path_id in combined_results:
                # Boost score for documents found in both
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
        for rank, result in enumerate(sorted_results[: config.max_results]):
            result.rank = rank
            result.search_mode = SearchMode.HYBRID

        return sorted_results[: config.max_results]

    def get_mode(self) -> SearchMode:
        """Get the search mode for this strategy."""
        return SearchMode.HYBRID


class SearchModeSelector:
    """Intelligent search mode selection based on query characteristics."""

    def select_mode(self, query: str | list[str]) -> SearchMode:
        """Select optimal search mode based on query characteristics."""
        query_str = query if isinstance(query, str) else " ".join(query)

        # Check for mixed queries (both technical and conceptual elements)
        is_technical = self._is_technical_query(query_str)
        is_conceptual = self._is_conceptual_query(query_str)
        has_explicit_operators = self._has_explicit_technical_operators(query_str)

        # If query has explicit technical operators (AND, OR, quotes, wildcards, etc.),
        # prioritize lexical search regardless of conceptual elements
        if has_explicit_operators:
            return SearchMode.LEXICAL
        # If query has both IMAS technical terms AND conceptual terms, use hybrid
        elif is_technical and is_conceptual:
            return SearchMode.HYBRID
        elif is_technical:
            return SearchMode.LEXICAL
        elif is_conceptual:
            return SearchMode.SEMANTIC
        else:
            return SearchMode.HYBRID

    def _has_explicit_technical_operators(self, query: str) -> bool:
        """Check if query has explicit technical search operators."""
        # Explicit technical operators that indicate user wants precise search
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

        return any(operator in query for operator in explicit_operators)

    def _is_technical_query(self, query: str) -> bool:
        """Check if query is technical and benefits from exact matching."""
        # Check for path-like queries (contains / and looks like IMAS paths)
        if "/" in query:
            # Check if it looks like a full IMAS path with proper structure
            parts = query.split("/")
            if len(parts) >= 2:
                # First part should look like an IDS name
                potential_ids = parts[0].strip()
                if re.match(r"^[a-z][a-z0-9_]*$", potential_ids):
                    # Contains common IMAS path indicators
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
                        "electrons",
                        "ions",
                        "temperature",
                        "density",
                        "validity",
                    ]
                    # If it contains a slash and any common IMAS path component, treat as path query
                    if any(indicator in query.lower() for indicator in path_indicators):
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
        if any(term in query.lower() for term in imas_technical_terms):
            return True

        # Check for underscore-separated technical terms (common in IMAS)
        if "_" in query and len(query.split("_")) > 1:
            # If it's mostly technical/path-like terms, treat as technical
            words = query.lower().split("_")
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

        return False

    def _is_conceptual_query(self, query: str) -> bool:
        """Check if query is conceptual and benefits from semantic search."""
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
        ]
        return any(indicator in query.lower() for indicator in conceptual_indicators)


class SearchResponse(BaseModel):
    """Response from search engine containing search hits."""

    hits: list[SearchMatch] = Field(description="Search results limited to max_results")
