"""
PathIndex class refactored to use Whoosh for text indexing and search
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import string
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict

# Third-party imports
import whoosh
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser
from whoosh.query import Term


@dataclass
class PathIndex:
    """Optimized index of IDS paths with Whoosh for fast search and filtering"""

    version: str  # IMAS DD version
    index_dir: Optional[Path] = None  # Directory for Whoosh index
    
    # Keep these for backward compatibility and storing data
    ids: Set[str] = field(default_factory=set)
    paths: Set[str] = field(default_factory=set)
    docs: Dict[str, str] = field(default_factory=dict)
    segments: Dict[str, Set[str]] = field(default_factory=dict)
    
    # Internal Whoosh index
    _index: Any = field(default=None, repr=False)  # Whoosh index
            "for",
            "with",
            "about",
            "to",
            "in",
            "on",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "of",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "not",
            "as",
            "what",
            "who",
            "how",
            "why",
            "where",
            "when",
        }
    )

    def add_path(self, path: str, documentation: str = "") -> None:
        """Add a path to the index"""
        self.paths.add(path)

        # Split the path into segments
        segments = path.split("/")

        # Add root ids
        self.ids.add(segments[0])

        if documentation:
            self.docs[path] = documentation
            # Index keywords from documentation
            self._index_documentation_keywords(path, documentation)

        # Remove parents from paths and docs
        for i in range(1, len(segments)):
            parent_path = "/".join(segments[:i])
            if parent_path in self.paths:
                self.paths.remove(parent_path)
                if parent_path in self.docs:
                    del self.docs[parent_path]

        # Add to segments index
        for segment in segments:
            if segment not in self.segments:
                self.segments[segment] = set()
            self.segments[segment].add(path)

        # Add to keywords index (exclude common words)
        for segment in segments:
            for i in range(len(segment)):
                for j in range(i + 1, len(segment) + 1):
                    keyword = segment[i:j].lower()
                    if len(keyword) > 2:  # Skip very short keywords
                        if keyword not in self.keywords:
                            self.keywords[keyword] = set()
                        self.keywords[keyword].add(path)

        # Add to prefixes index
        for i in range(1, len(path)):
            prefix = path[:i]
            if prefix not in self.prefixes:
                self.prefixes[prefix] = set()
            self.prefixes[prefix].add(path)

    def _index_documentation_keywords(self, path: str, documentation: str) -> None:
        """Extract keywords from documentation and add to keyword index."""
        # Normalize text: lowercase and remove punctuation
        text = documentation.lower()
        # Replace punctuation with spaces
        for char in string.punctuation:
            text = text.replace(char, " ")

        # Split into words and filter out stop words and short words
        words = text.split()
        keywords = {
            word for word in words if len(word) > 2 and word not in self.stop_words
        }

        # Store keywords for this path
        if path not in self.doc_keywords:
            self.doc_keywords[path] = set()
        self.doc_keywords[path].update(keywords)

        # Add path to index for each keyword
        for keyword in keywords:
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = set()
            self.keyword_index[keyword].add(path)

    def _extract_keywords_from_query(self, query: str) -> Set[str]:
        """Extract keywords from a natural language query."""
        # Normalize text: lowercase and remove punctuation
        query = query.lower()
        for char in string.punctuation:
            query = query.replace(char, " ")

        # Split into words and filter out stop words and short words
        words = query.split()
        keywords = {
            word for word in words if len(word) > 2 and word not in self.stop_words
        }
        return keywords

    def search_by_keywords(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search the index for paths matching the given keywords.

        Args:
            query: Natural language query
            limit: Maximum number of results to return

        Returns:
            List of dictionaries with path, relevance score, and documentation
        """
        keywords = self._extract_keywords_from_query(query)

        if not keywords:
            return []

        # Count matches for each path
        path_scores = defaultdict(int)

        for keyword in keywords:
            # Look for exact keyword matches
            if keyword in self.keyword_index:
                for path in self.keyword_index[keyword]:
                    path_scores[path] += 2  # Higher weight for exact matches

            # Look for partial keyword matches
            for indexed_keyword in self.keyword_index:
                if keyword in indexed_keyword or indexed_keyword in keyword:
                    for path in self.keyword_index[indexed_keyword]:
                        path_scores[path] += 1

        # Sort paths by score (descending)
        results = [
            {"path": path, "score": score, "doc": self.docs.get(path, "")}
            for path, score in path_scores.items()
            if score > 0  # Only include paths with a positive score
        ]

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]

    def search_by_semantic_similarity(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for paths using semantic similarity with the query.

        Note: This is a placeholder for future implementation of more
        advanced semantic search capabilities using embeddings.

        Args:
            query: Natural language query
            limit: Maximum number of results to return

        Returns:
            List of dictionaries with path and documentation
        """
        # Currently falls back to keyword search
        # In the future, this could use word embeddings or language models
        # to find semantically similar documentation
        return self.search_by_keywords(query, limit)
