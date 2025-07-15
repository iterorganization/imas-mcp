"""
Search router for auto-detecting and routing search queries.

This module handles the detection of query features and routes to the
appropriate search strategy.
"""

import logging
from dataclasses import dataclass
from typing import List, Union

from ..json_data_accessor import JsonDataDictionaryAccessor
from ..lexicographic_search import LexicographicSearch

logger = logging.getLogger(__name__)


@dataclass
class QueryFeatures:
    """Detected features of a search query."""

    has_boolean: bool = False
    has_wildcards: bool = False
    has_field_prefix: bool = False
    has_regex_patterns: bool = False
    needs_prefix_search: bool = False
    is_exact_path: bool = False
    has_quotes: bool = False
    is_complex: bool = False


@dataclass
class SearchRouter:
    """Routes search queries to appropriate search strategies."""

    data_accessor: JsonDataDictionaryAccessor
    lexicographic_search: LexicographicSearch

    def detect_query_features(self, query: str) -> QueryFeatures:
        """Auto-detect what features the query needs."""
        query_upper = query.upper()

        return QueryFeatures(
            has_boolean=any(op in query_upper for op in [" AND ", " OR ", " NOT "]),
            has_wildcards="*" in query or "?" in query,
            has_field_prefix=":" in query and not query.startswith("http"),
            has_regex_patterns=any(
                char in query for char in [".*", "^", "$", "[", "]"]
            ),
            needs_prefix_search="/" in query
            and " " not in query
            and not any(op in query_upper for op in [" AND ", " OR ", " NOT "]),
            is_exact_path="/" in query
            and " " not in query
            and "*" not in query
            and "?" not in query
            and not any(op in query_upper for op in [" AND ", " OR ", " NOT "]),
            has_quotes='"' in query,
            is_complex=len(query.split()) > 3
            or any(op in query_upper for op in [" AND ", " OR ", " NOT "]),
        )

    def needs_lexicographic_search(self, features: QueryFeatures) -> bool:
        """Determine if lexicographic search capabilities are needed."""
        return any(
            [
                features.has_boolean,
                features.has_wildcards,
                features.has_field_prefix,
                features.has_regex_patterns,
                features.has_quotes,
                features.is_complex,
            ]
        )

    def determine_search_strategy(self, query: Union[str, List[str]]) -> str:
        """Determine the best search strategy for the given query."""
        if isinstance(query, list):
            return "bulk"

        features = self.detect_query_features(query)

        if self.needs_lexicographic_search(features):
            return "lexicographic"
        elif features.is_exact_path:
            return "exact_path"
        elif features.needs_prefix_search:
            return "prefix"
        else:
            return "basic"
