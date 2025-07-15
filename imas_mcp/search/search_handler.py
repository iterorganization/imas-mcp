"""
Search handler implementations for different IMAS search strategies.

This module provides specialized search handlers that implement different
search strategies for the IMAS data dictionary.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

from ..json_data_accessor import JsonDataDictionaryAccessor
from ..lexicographic_search import LexicographicSearch
from ..search_result import SearchResult
from .search_router import QueryFeatures


@dataclass
class SearchRequest:
    """Request object for search operations."""

    query: Union[str, List[str]]
    ids_name: Optional[str] = None
    max_results: int = 10


@dataclass
class SearchResponse:
    """Response object containing search results and metadata."""

    results: List[SearchResult]
    total_results: int
    search_strategy: str
    query_features: Optional["QueryFeatures"] = None
    suggestions: Optional[List[str]] = None


class SearchHandler:
    """Handler for executing different search strategies."""

    def __init__(
        self, json_accessor: JsonDataDictionaryAccessor, lex_search: LexicographicSearch
    ):
        """
        Initialize search handler with data accessors.

        Args:
            json_accessor: JSON data dictionary accessor
            lex_search: Lexicographic search implementation
        """
        self.json_accessor = json_accessor
        self.lex_search = lex_search

    def handle_bulk_search(self, request: SearchRequest) -> SearchResponse:
        """
        Handle bulk search for multiple queries using enhanced ranking.

        Args:
            request: Search request with list of queries

        Returns:
            Combined search response with keyword-based ranking
        """
        if isinstance(request.query, str):
            raise ValueError("Bulk search requires list of queries")

        # Use the enhanced search method for list queries
        try:
            results = self.lex_search.search_by_keywords(
                keywords=request.query,
                page_size=request.max_results,
                page=1,
                fuzzy=False,
            )

            return SearchResponse(
                results=results,
                total_results=len(results),
                search_strategy="enhanced_bulk_ranking",
                suggestions=self._generate_bulk_suggestions(request.query),
            )

        except Exception:
            # Fallback to original implementation
            all_results = []
            strategies_used = set()

            for query in request.query:
                single_request = SearchRequest(
                    query=query,
                    ids_name=request.ids_name,
                    max_results=request.max_results,
                )
                response = self.handle_search(single_request)
                all_results.extend(response.results)
                strategies_used.add(response.search_strategy)

            # Remove duplicates while preserving order
            seen_paths = set()
            unique_results = []
            for result in all_results:
                if result.path not in seen_paths:
                    seen_paths.add(result.path)
                    unique_results.append(result)

            # Sort by relevance score
            unique_results.sort(key=lambda x: x.relevance, reverse=True)

            # Limit results
            limited_results = unique_results[: request.max_results]

            return SearchResponse(
                results=limited_results,
                total_results=len(unique_results),
                search_strategy=f"bulk_fallback ({', '.join(strategies_used)})",
                suggestions=self._generate_bulk_suggestions(request.query),
            )

    def handle_search(self, request: SearchRequest) -> SearchResponse:
        """
        Handle single search query using appropriate strategy.

        Args:
            request: Search request with single query

        Returns:
            Search response with results
        """
        if isinstance(request.query, list):
            return self.handle_bulk_search(request)

        query = request.query.strip()
        if not query:
            return self._empty_response("empty_query")

        # Use lexicographic search as primary strategy with exploratory mode
        try:
            results = self.lex_search.search_by_keywords(
                query_str=query,
                page_size=request.max_results,
                exploratory_mode=True,  # Use OR logic for broader exploration
            )

            return SearchResponse(
                results=results,
                total_results=len(results),
                search_strategy="lexicographic",
                suggestions=self._generate_suggestions(query, results),
            )

        except Exception:
            # Fallback to JSON accessor
            ids_filter = [request.ids_name] if request.ids_name else None
            raw_results = self.json_accessor.search_paths_by_pattern(
                pattern=query, ids_filter=ids_filter
            )

            # Convert to SearchResult objects (simplified)
            results = [
                SearchResult(
                    path=item.get("path", ""),
                    score=0.5,
                    documentation=item.get("documentation", ""),
                    units=item.get("units", ""),
                    ids_name=item.get("ids_name", ""),
                )
                for item in raw_results[: request.max_results]
            ]

            return SearchResponse(
                results=results,
                total_results=len(results),
                search_strategy="json_fallback",
                suggestions=self._generate_suggestions(query, results),
            )

    def handle_exact_path_search(self, request: SearchRequest) -> SearchResponse:
        """
        Handle exact path search.

        Args:
            request: Search request with exact path

        Returns:
            Search response with exact match
        """
        query = request.query if isinstance(request.query, str) else request.query[0]

        # Try to get exact path match
        try:
            result = self.lex_search.search_by_exact_path(query)
            if result:
                return SearchResponse(
                    results=[result], total_results=1, search_strategy="exact_path"
                )
        except Exception:
            pass

        # Fallback to regular search
        return self.handle_search(request)

    def handle_boolean_search(self, request: SearchRequest) -> SearchResponse:
        """
        Handle boolean search using natural query parsing.

        Args:
            request: Search request with boolean operators

        Returns:
            Search response with boolean results
        """
        query = request.query if isinstance(request.query, str) else request.query[0]

        # Use the regular search_by_keywords which handles boolean operators
        try:
            results = self.lex_search.search_by_keywords(
                query_str=query, page_size=request.max_results
            )

            return SearchResponse(
                results=results,
                total_results=len(results),
                search_strategy="boolean",
                suggestions=self._generate_boolean_suggestions(query),
            )

        except Exception:
            # Fallback to regular search without boolean operators
            clean_query = (
                query.replace(" AND ", " ").replace(" OR ", " ").replace(" NOT ", " ")
            )
            return self.handle_search(
                SearchRequest(
                    query=clean_query,
                    ids_name=request.ids_name,
                    max_results=request.max_results,
                )
            )

    def handle_wildcard_search(self, request: SearchRequest) -> SearchResponse:
        """
        Handle wildcard search using natural query parsing.

        Args:
            request: Search request with wildcards

        Returns:
            Search response with wildcard results
        """
        query = request.query if isinstance(request.query, str) else request.query[0]

        # Use the regular search_by_keywords which handles wildcards
        try:
            results = self.lex_search.search_by_keywords(
                query_str=query, page_size=request.max_results
            )

            return SearchResponse(
                results=results,
                total_results=len(results),
                search_strategy="wildcard",
                suggestions=self._generate_wildcard_suggestions(query),
            )

        except Exception:
            # Fallback to regular search without wildcards
            clean_query = query.replace("*", "").replace("?", "")
            return self.handle_search(
                SearchRequest(
                    query=clean_query,
                    ids_name=request.ids_name,
                    max_results=request.max_results,
                )
            )

    def handle_field_search(self, request: SearchRequest) -> SearchResponse:
        """
        Handle field-specific search using search_fields parameter.

        Args:
            request: Search request with field specifiers

        Returns:
            Search response with field-specific results
        """
        query = request.query if isinstance(request.query, str) else request.query[0]

        # Extract field specifications (e.g., "name:temperature" -> fields=["name"])
        search_fields = None
        if ":" in query:
            parts = query.split(":")
            if len(parts) == 2:
                field_name = parts[0].strip()
                query = parts[1].strip()
                # Map common field names to actual index fields
                field_mapping = {
                    "name": "path",
                    "description": "documentation",
                    "units": "units",
                }
                if field_name in field_mapping:
                    search_fields = [field_mapping[field_name]]

        try:
            results = self.lex_search.search_by_keywords(
                query_str=query,
                page_size=request.max_results,
                search_fields=search_fields,
            )

            return SearchResponse(
                results=results,
                total_results=len(results),
                search_strategy="field_specific",
                suggestions=self._generate_field_suggestions(query),
            )

        except Exception:
            # Fallback to regular search
            return self.handle_search(
                SearchRequest(
                    query=query,
                    ids_name=request.ids_name,
                    max_results=request.max_results,
                )
            )

    def _empty_response(self, strategy: str) -> SearchResponse:
        """Create empty search response."""
        return SearchResponse(
            results=[],
            total_results=0,
            search_strategy=strategy,
            suggestions=[
                "Try a more specific search term",
                "Check spelling",
                "Use wildcards like * or ?",
            ],
        )

    def _generate_suggestions(
        self, query: str, results: List[SearchResult]
    ) -> List[str]:
        """Generate search suggestions based on query and results."""
        suggestions = []

        if not results:
            suggestions.extend(
                [
                    f"Try searching for broader terms related to '{query}'",
                    f"Use wildcards: '{query}*' or '*{query}*'",
                    "Check spelling and try alternative terms",
                ]
            )
        elif len(results) < 3:
            suggestions.extend(
                [
                    f"Try broader search: '*{query}*'",
                    f"Related searches might include: {query}_*",
                ]
            )

        return suggestions[:3]

    def _generate_bulk_suggestions(self, queries: List[str]) -> List[str]:
        """Generate suggestions for bulk searches."""
        return [
            f"Consider combining related terms: {' OR '.join(queries[:2])}",
            "Use more specific terms to reduce overlap",
            "Try searching individual terms for better precision",
        ]

    def _generate_boolean_suggestions(self, query: str) -> List[str]:
        """Generate suggestions for boolean searches."""
        return [
            "Use AND for terms that must both appear",
            "Use OR for alternative terms",
            "Use NOT to exclude unwanted terms",
            "Group terms with parentheses: (term1 OR term2) AND term3",
        ]

    def _generate_wildcard_suggestions(self, pattern: str) -> List[str]:
        """Generate suggestions for wildcard searches."""
        return [
            "Use * to match any number of characters",
            "Use ? to match a single character",
            f"Try more specific patterns based on '{pattern}'",
        ]

    def _generate_field_suggestions(self, query: str) -> List[str]:
        """Generate suggestions for field searches."""
        return [
            "Available fields: name, description, units",
            "Example: name:temperature",
            "Example: units:m description:position",
        ]
