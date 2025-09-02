"""
Physics Extraction System for IMAS Data Dictionary

AI-assisted, Pydantic-validated extraction of physics quantities from IMAS DD JSON data.
Supports batch processing, incremental updates, and conflict resolution.
"""

from .coordination import ExtractionCoordinator, LockManager
from .extractors import AIPhysicsExtractor, BatchProcessor
from .models import (
    ConflictResolution,
    ExtractionProgress,
    ExtractionResult,
    PhysicsDatabase,
    PhysicsQuantity,
    ProcessingPriority,
)
from .storage import ConflictManager, PhysicsStorage, ProgressTracker

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
