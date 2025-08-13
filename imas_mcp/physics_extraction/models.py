"""
Pydantic models for physics extraction system.

Defines data structures for physics quantities, extraction results,
and system state management.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProcessingPriority(str, Enum):
    """Priority levels for physics extraction."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts between quantities."""

    MERGE = "merge"  # Combine information
    REPLACE = "replace"  # Use newer version
    MANUAL = "manual"  # Require manual review
    SKIP = "skip"  # Skip conflicting entries


class ExtractionStatus(str, Enum):
    """Status of extraction operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhysicsQuantity(BaseModel):
    """
    A physics quantity with comprehensive metadata.

    Enhanced version of the original PhysicsQuantity with additional
    fields for AI-assisted extraction and conflict resolution.
    """

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Human-readable name of the quantity")
    symbol: str | None = Field(None, description="Mathematical symbol")

    # Physics properties
    unit: str | None = Field(None, description="Physical unit")
    dimensions: str | None = Field(None, description="Physical dimensions")
    typical_range: str | None = Field(None, description="Typical value range")

    # Documentation
    description: str = Field(..., description="Detailed description")
    physics_context: str | None = Field(None, description="Physics domain/context")
    related_quantities: list[str] = Field(
        default_factory=list, description="Related quantity IDs"
    )

    # IMAS-specific
    imas_paths: list[str] = Field(default_factory=list, description="IMAS data paths")
    ids_sources: set[str] = Field(default_factory=set, description="Source IDS names")

    # Extraction metadata
    extraction_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="AI confidence score"
    )
    ai_source: str | None = Field(None, description="AI model used for extraction")
    human_verified: bool = Field(False, description="Human verification status")
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # Version control
    version: int = Field(1, description="Version number")
    created_by: str = Field("ai_extraction", description="Creation source")

    @field_validator("imas_paths")
    @classmethod
    def validate_imas_paths(cls, v):
        """Ensure IMAS paths are valid."""
        for path in v:
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"Invalid IMAS path: {path}")
        return v

    @field_validator("extraction_confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is reasonable."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    def merge_with(
        self,
        other: "PhysicsQuantity",
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MERGE,
    ) -> "PhysicsQuantity":
        """
        Merge this quantity with another, handling conflicts.

        Args:
            other: Another PhysicsQuantity to merge with
            strategy: How to resolve conflicts

        Returns:
            Merged PhysicsQuantity
        """
        if strategy == ConflictResolutionStrategy.REPLACE:
            return other
        elif strategy == ConflictResolutionStrategy.SKIP:
            return self
        elif strategy == ConflictResolutionStrategy.MERGE:
            # Combine information intelligently
            merged = self.copy()

            # Combine paths and sources
            merged.imas_paths = list(set(self.imas_paths + other.imas_paths))
            merged.ids_sources = self.ids_sources.union(other.ids_sources)
            merged.related_quantities = list(
                set(self.related_quantities + other.related_quantities)
            )

            # Use higher confidence values
            if other.extraction_confidence > self.extraction_confidence:
                merged.extraction_confidence = other.extraction_confidence
                merged.ai_source = other.ai_source

            # Prefer verified descriptions
            if other.human_verified and not self.human_verified:
                merged.description = other.description
                merged.human_verified = other.human_verified

            # Update metadata
            merged.version = max(self.version, other.version) + 1
            merged.last_updated = datetime.utcnow()

            return merged
        else:
            # Manual resolution - return conflict info
            raise ValueError(
                f"Manual resolution required for {self.name} vs {other.name}"
            )


class ExtractionResult(BaseModel):
    """Result of extracting physics quantities from IMAS data."""

    # Processing info
    ids_name: str = Field(..., description="IDS being processed")
    paths_processed: list[str] = Field(default_factory=list)
    processing_time: float = Field(
        default=0.0, description="Processing time in seconds"
    )

    # Results
    quantities_found: list[PhysicsQuantity] = Field(default_factory=list)
    quantities_updated: list[str] = Field(
        default_factory=list, description="IDs of updated quantities"
    )

    # Status and errors
    status: ExtractionStatus = Field(default=ExtractionStatus.PENDING)
    error_message: str | None = Field(default=None)
    warnings: list[str] = Field(default_factory=list)

    # AI metadata
    ai_model: str | None = Field(default=None)
    confidence_threshold: float = Field(default=0.5)

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = Field(default=None)

    # Catalog metadata (for enhanced tracking)
    catalog_metadata: dict[str, Any] | None = Field(
        default=None, description="Catalog-based metadata"
    )

    def mark_completed(self):
        """Mark extraction as completed."""
        self.status = ExtractionStatus.COMPLETED
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error: str):
        """Mark extraction as failed."""
        self.status = ExtractionStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.utcnow()


class ConflictResolution(BaseModel):
    """Information about a conflict between physics quantities."""

    conflict_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    quantity_name: str = Field(..., description="Name of conflicting quantity")

    # Conflicting versions
    existing_quantity: PhysicsQuantity
    new_quantity: PhysicsQuantity

    # Conflict details
    conflict_fields: list[str] = Field(
        default_factory=list, description="Fields that differ"
    )
    severity: ProcessingPriority = Field(default=ProcessingPriority.MEDIUM)

    # Resolution
    resolution_strategy: ConflictResolutionStrategy | None = Field(default=None)
    resolved_quantity: PhysicsQuantity | None = Field(default=None)
    resolved_at: datetime | None = Field(default=None)
    resolved_by: str | None = Field(default=None)

    # Status
    is_resolved: bool = Field(default=False)
    requires_human_review: bool = Field(default=False)

    def resolve(
        self, strategy: ConflictResolutionStrategy, resolved_by: str = "system"
    ) -> PhysicsQuantity:
        """
        Resolve the conflict using the specified strategy.

        Args:
            strategy: How to resolve the conflict
            resolved_by: Who/what resolved it

        Returns:
            The resolved PhysicsQuantity
        """
        self.resolution_strategy = strategy
        self.resolved_by = resolved_by
        self.resolved_at = datetime.utcnow()

        try:
            self.resolved_quantity = self.existing_quantity.merge_with(
                self.new_quantity, strategy
            )
            self.is_resolved = True
            return self.resolved_quantity
        except ValueError:
            # Manual resolution required
            self.requires_human_review = True
            self.is_resolved = False
            raise


class ExtractionProgress(BaseModel):
    """Track progress of physics extraction across all IDS."""

    # Session info
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # Overall progress
    total_ids: int = Field(default=0, description="Total number of IDS to process")
    completed_ids: int = Field(default=0, description="Number of completed IDS")
    failed_ids: int = Field(default=0, description="Number of failed IDS")

    # Detailed tracking
    ids_status: dict[str, ExtractionStatus] = Field(default_factory=dict)
    ids_progress: dict[str, float] = Field(
        default_factory=dict, description="Per-IDS progress (0-1)"
    )

    # Results summary
    total_quantities_found: int = Field(default=0)
    total_quantities_updated: int = Field(default=0)
    total_conflicts: int = Field(default=0)

    # Processing stats
    total_paths_processed: int = Field(default=0)
    total_processing_time: float = Field(default=0.0)

    # Configuration
    paths_per_batch: int = Field(default=10, description="Paths processed per batch")
    confidence_threshold: float = Field(default=0.5)

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_ids == 0:
            return 0.0
        return (self.completed_ids / self.total_ids) * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if extraction is complete."""
        return self.completed_ids + self.failed_ids == self.total_ids

    def update_ids_progress(
        self, ids_name: str, status: ExtractionStatus, progress: float = 0.0
    ):
        """Update progress for a specific IDS."""
        self.ids_status[ids_name] = status
        self.ids_progress[ids_name] = progress
        self.last_updated = datetime.utcnow()

        # Update counters
        completed = sum(
            1 for s in self.ids_status.values() if s == ExtractionStatus.COMPLETED
        )
        failed = sum(
            1 for s in self.ids_status.values() if s == ExtractionStatus.FAILED
        )

        self.completed_ids = completed
        self.failed_ids = failed


class PhysicsDatabase(BaseModel):
    """
    Complete database of physics quantities with metadata.

    This represents the full extracted knowledge base.
    """

    # Database metadata
    version: str = Field("1.0.0", description="Database version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # Content
    quantities: dict[str, PhysicsQuantity] = Field(
        default_factory=dict, description="Quantity ID -> Quantity"
    )

    # Organization
    quantities_by_name: dict[str, str] = Field(
        default_factory=dict, description="Name -> Quantity ID"
    )
    quantities_by_ids: dict[str, list[str]] = Field(
        default_factory=dict, description="IDS -> Quantity IDs"
    )
    quantities_by_context: dict[str, list[str]] = Field(
        default_factory=dict, description="Context -> Quantity IDs"
    )

    # Statistics
    total_quantities: int = Field(default=0)
    verified_quantities: int = Field(default=0)
    coverage_by_ids: dict[str, int] = Field(
        default_factory=dict, description="IDS -> quantity count"
    )

    # Extraction history
    extraction_sessions: list[str] = Field(
        default_factory=list, description="Session IDs"
    )
    conflicts_resolved: int = Field(default=0)

    def add_quantity(self, quantity: PhysicsQuantity) -> bool:
        """
        Add a physics quantity to the database.

        Args:
            quantity: The quantity to add

        Returns:
            True if added, False if it already exists
        """
        if quantity.id in self.quantities:
            return False

        self.quantities[quantity.id] = quantity
        self.quantities_by_name[quantity.name] = quantity.id

        # Update indices
        for ids_name in quantity.ids_sources:
            if ids_name not in self.quantities_by_ids:
                self.quantities_by_ids[ids_name] = []
            self.quantities_by_ids[ids_name].append(quantity.id)

        if quantity.physics_context:
            if quantity.physics_context not in self.quantities_by_context:
                self.quantities_by_context[quantity.physics_context] = []
            self.quantities_by_context[quantity.physics_context].append(quantity.id)

        # Update stats
        self.total_quantities += 1
        if quantity.human_verified:
            self.verified_quantities += 1

        for ids_name in quantity.ids_sources:
            self.coverage_by_ids[ids_name] = self.coverage_by_ids.get(ids_name, 0) + 1

        self.last_updated = datetime.utcnow()
        return True

    def get_quantity_by_name(self, name: str) -> PhysicsQuantity | None:
        """Get quantity by name."""
        quantity_id = self.quantities_by_name.get(name)
        if quantity_id:
            return self.quantities.get(quantity_id)
        return None

    def get_quantities_for_ids(self, ids_name: str) -> list[PhysicsQuantity]:
        """Get all quantities associated with an IDS."""
        quantity_ids = self.quantities_by_ids.get(ids_name, [])
        return [self.quantities[qid] for qid in quantity_ids if qid in self.quantities]

    def update_quantity(self, quantity: PhysicsQuantity):
        """Update an existing quantity."""
        if quantity.id in self.quantities:
            old_quantity = self.quantities[quantity.id]
            self.quantities[quantity.id] = quantity

            # Update verification count
            if quantity.human_verified and not old_quantity.human_verified:
                self.verified_quantities += 1
            elif not quantity.human_verified and old_quantity.human_verified:
                self.verified_quantities -= 1

            self.last_updated = datetime.utcnow()

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        return {
            "total_quantities": self.total_quantities,
            "verified_quantities": self.verified_quantities,
            "verification_rate": self.verified_quantities
            / max(1, self.total_quantities),
            "ids_coverage": dict(self.coverage_by_ids),
            "contexts": list(self.quantities_by_context.keys()),
            "last_updated": self.last_updated.isoformat(),
        }
