"""
Relationship extraction and analysis for IMAS data paths.

This module provides functionality to extract semantic relationships between
IMAS data paths using advanced clustering and embedding techniques.
"""

from .config import RelationshipExtractionConfig
from .extractor import RelationshipExtractor
from .models import RelationshipResult, RelationshipSet

__all__ = [
    "RelationshipExtractionConfig",
    "RelationshipExtractor",
    "RelationshipResult",
    "RelationshipSet",
]
