"""
Search modes and composition patterns for IMAS MCP server.

This module provides different search strategies and modes for querying the
IMAS data dictionary, following composition patterns for maintainability
and extensibility.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel
from .document_store import Document, DocumentStore

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Enumeration of available search modes."""

    SEMANTIC = "semantic"  # AI-powered semantic search using sentence transformers
    LEXICAL = "lexical"  # Traditional full-text search using SQLite FTS5
    HYBRID = "hybrid"  # Combination of semantic and lexical search
    AUTO = "auto"  # Automatically choose best mode based on query


class SearchConfig(BaseModel):
    """Configuration for search operations."""

    mode: SearchMode = SearchMode.AUTO
    max_results: int = 10
    filter_ids: Optional[List[str]] = None
    similarity_threshold: float = 0.0
    boost_exact_matches: bool = True
    enable_physics_enhancement: bool = True


class SearchResult(BaseModel):
    """Standardized search result format with clear field intentions."""

    document: Document
    score: float
    rank: int
    search_mode: SearchMode
    highlights: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary format with clear field names.

        This method provides a custom serialization that transforms internal
        field names to the expected API format (e.g., score -> relevance_score).
        This is kept for backward compatibility with existing code.
        """
        return {
            "path": self.document.metadata.path_name,
            "relevance_score": self.score,
            "documentation": self.document.documentation,
            "units": self.document.units.unit_str if self.document.units else "",
            "ids_name": self.document.metadata.ids_name,
            "data_type": self.document.metadata.data_type,
            "physics_domain": self.document.metadata.physics_domain or "general",
            "highlights": self.highlights,
            "search_mode": self.search_mode.value,
            "rank": self.rank,
        }

    @property
    def physics_domain_valid(self) -> bool:
        """Check if this result has a valid physics domain."""
        return bool(self.document.metadata.physics_domain)

    @property
    def has_units(self) -> bool:
        """Check if this result has units defined."""
        return bool(self.document.units and self.document.units.unit_str)

    def extract_measurement_context(self) -> Optional[Dict[str, str]]:
        """Extract measurement context information from documentation."""
        doc_lower = self.document.documentation.lower()
        measurement_terms = [
            "temperature",
            "density",
            "pressure",
            "magnetic",
            "current",
        ]

        matching_terms = [term for term in measurement_terms if term in doc_lower]

        if matching_terms:
            measurement_type = (
                "multiple" if len(matching_terms) > 1 else matching_terms[0]
            )
            return {
                "path": self.document.metadata.path_name,
                "measurement_type": measurement_type,
                "context": self.document.documentation[:150],
            }
        return None


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store

    @abstractmethod
    def search(
        self,
        query: Union[str, List[str]],
        config: SearchConfig,
    ) -> List[SearchResult]:
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
        query: Union[str, List[str]],
        config: SearchConfig,
    ) -> List[SearchResult]:
        """Execute lexical search using full-text search."""
        # Convert query to string format
        query_str = query if isinstance(query, str) else " ".join(query)

        # Apply IDS filtering if specified
        if config.filter_ids:
            # Add IDS filter to query
            ids_filter = " OR ".join([f"ids_name:{ids}" for ids in config.filter_ids])
            query_str = f"({query_str}) AND ({ids_filter})"

        # Execute full-text search
        try:
            documents = self.document_store.search_full_text(
                query_str, max_results=config.max_results
            )

            # Convert to SearchResult objects
            results = []
            for rank, doc in enumerate(documents):
                result = SearchResult(
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
        query: Union[str, List[str]],
        config: SearchConfig,
    ) -> List[SearchResult]:
        """Execute semantic search using sentence transformers."""
        # Convert query to string format
        query_str = query if isinstance(query, str) else " ".join(query)

        try:
            # Execute semantic search
            semantic_results = self.semantic_search.search(
                query=query_str,
                top_k=config.max_results,
                filter_ids=config.filter_ids,
                similarity_threshold=config.similarity_threshold,
            )

            # Convert to SearchResult objects
            results = []
            for rank, semantic_result in enumerate(semantic_results):
                result = SearchResult(
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
        query: Union[str, List[str]],
        config: SearchConfig,
    ) -> List[SearchResult]:
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

    def select_mode(self, query: Union[str, List[str]]) -> SearchMode:
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
            # Additional checks to ensure it's a path-like query
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


class SearchComposer:
    """Main search composition class that orchestrates different search strategies."""

    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store
        self.strategies = {
            SearchMode.LEXICAL: LexicalSearchStrategy(document_store),
            SearchMode.SEMANTIC: SemanticSearchStrategy(document_store),
            SearchMode.HYBRID: HybridSearchStrategy(document_store),
        }
        self.mode_selector = SearchModeSelector()

    def search(
        self,
        query: Union[str, List[str]],
        config: Optional[SearchConfig] = None,
    ) -> List[SearchResult]:
        """Execute search with automatic mode selection or explicit configuration."""
        if config is None:
            config = SearchConfig()

        # Auto-select mode if needed
        if config.mode == SearchMode.AUTO:
            config.mode = self.mode_selector.select_mode(query)

        # Execute search with selected strategy
        strategy = self.strategies[config.mode]
        return strategy.search(query, config)

    def search_with_params(
        self,
        query: Union[str, List[str]],
        mode: SearchMode,
        max_results: int = 10,
        filter_ids: Optional[List[str]] = None,
        similarity_threshold: float = 0.0,
        boost_exact_matches: bool = True,
        enable_physics_enhancement: bool = True,
    ) -> Dict[str, Any]:
        """Convenience method for search with individual parameters."""
        config = SearchConfig(
            mode=mode,
            max_results=max_results,
            filter_ids=filter_ids,
            similarity_threshold=similarity_threshold,
            boost_exact_matches=boost_exact_matches,
            enable_physics_enhancement=enable_physics_enhancement,
        )

        results = self.search(query, config)

        # Convert to server format
        return {
            "results": [result.to_dict() for result in results],
            "results_count": len(results),
            "search_strategy": config.mode.value,
            "max_results": config.max_results,
            "filter_ids": config.filter_ids,
        }

    def get_available_modes(self) -> List[SearchMode]:
        """Get list of available search modes."""
        return list(self.strategies.keys())
