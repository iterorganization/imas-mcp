"""
Physics Extraction System for IMAS Data Dictionary

AI-assisted, Pydantic-validated extraction of physics quantities from IMAS DD JSON data.
Supports batch processing, incremental updates, and conflict resolution.
"""

from .models import (
    PhysicsQuantity,
    ExtractionResult,
    PhysicsDatabase,
    ExtractionProgress,
    ConflictResolution,
    ProcessingPriority,
)

from .extractors import AIPhysicsExtractor, BatchProcessor

from .storage import PhysicsStorage, ProgressTracker, ConflictManager

from .coordination import ExtractionCoordinator, LockManager

__all__ = [
    "PhysicsQuantity",
    "ExtractionResult",
    "PhysicsDatabase",
    "ExtractionProgress",
    "ConflictResolution",
    "ProcessingPriority",
    "AIPhysicsExtractor",
    "BatchProcessor",
    "PhysicsStorage",
    "ProgressTracker",
    "ConflictManager",
    "ExtractionCoordinator",
    "LockManager",
]
