"""Pydantic models for search functionality.

This module contains the core data models used in search operations,
extracted from search_modes.py for better organization.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from .enums import SearchMode
from ..search.document_store import Document


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
