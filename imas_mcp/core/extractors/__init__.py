"""Composable extractors for IMAS data dictionary transformation."""

from .base import BaseExtractor, ExtractorContext
from .coordinate_extractor import CoordinateExtractor
from .identifier_extractor import IdentifierExtractor
from .metadata_extractor import MetadataExtractor
from .physics_extractor import LifecycleExtractor, PhysicsExtractor
from .semantic_extractor import PathExtractor, SemanticExtractor
from .validation_extractor import ValidationExtractor

# Conditionally import RelationshipExtractor to avoid heavy dependencies during build
try:
    from .relationship_extractor import RelationshipExtractor

    _relationship_available = True
except ImportError:
    RelationshipExtractor = None
    _relationship_available = False

__all__ = [
    "BaseExtractor",
    "ExtractorContext",
    "CoordinateExtractor",
    "IdentifierExtractor",
    "LifecycleExtractor",
    "MetadataExtractor",
    "PathExtractor",
    "PhysicsExtractor",
    "SemanticExtractor",
    "ValidationExtractor",
]

if _relationship_available:
    __all__.append("RelationshipExtractor")
