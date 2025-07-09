"""Composable extractors for IMAS data dictionary transformation."""

from .base import BaseExtractor, ExtractorContext
from .coordinate_extractor import CoordinateExtractor
from .metadata_extractor import MetadataExtractor
from .physics_extractor import PhysicsExtractor, LifecycleExtractor
from .semantic_extractor import SemanticExtractor, PathExtractor
from .relationship_extractor import RelationshipExtractor
from .validation_extractor import ValidationExtractor

__all__ = [
    "BaseExtractor",
    "ExtractorContext",
    "CoordinateExtractor",
    "LifecycleExtractor",
    "MetadataExtractor",
    "PathExtractor",
    "PhysicsExtractor",
    "RelationshipExtractor",
    "SemanticExtractor",
    "ValidationExtractor",
]
