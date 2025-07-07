"""Enhanced data models for IMAS Data Dictionary using Pydantic."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PhysicsDomain(str, Enum):
    """Physics domains in IMAS."""

    TRANSPORT = "transport"
    EQUILIBRIUM = "equilibrium"
    MHD = "mhd"
    HEATING = "heating"
    DIAGNOSTICS = "diagnostics"
    WALL = "wall"
    GENERAL = "general"


class DataLifecycle(str, Enum):
    """Data lifecycle status."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    OBSOLETE = "obsolete"


class CoordinateSystem(BaseModel):
    """Coordinate system definition."""

    description: str
    units: str
    range: Optional[List[float]] = None
    usage: str


class PhysicsContext(BaseModel):
    """Physics context for a data field."""

    domain: str
    phenomena: List[str] = Field(default_factory=list)
    typical_values: Dict[str, str] = Field(default_factory=dict)


class ValidationRules(BaseModel):
    """Validation rules for data fields."""

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    units_required: bool = True
    coordinate_check: Optional[str] = None


class UsageExample(BaseModel):
    """Code usage example."""

    scenario: str
    code: str
    notes: str


class DataPath(BaseModel):
    """Complete data path information extracted from XML."""

    path: str  # The full path (added for consistency)
    documentation: str
    units: Optional[str] = None  # Make units optional
    coordinates: List[str] = Field(default_factory=list)
    lifecycle: str = "active"
    data_type: Optional[str] = None
    introduced_after: Optional[str] = None
    physics_context: Optional[PhysicsContext] = None
    related_paths: List[str] = Field(default_factory=list)
    usage_examples: List[UsageExample] = Field(default_factory=list)
    validation_rules: Optional[ValidationRules] = None

    # Additional XML attributes
    element_type: Optional[str] = None
    coordinate1: Optional[str] = None
    coordinate2: Optional[str] = None
    timebase: Optional[str] = None
    type: Optional[str] = None

    model_config = ConfigDict(extra="allow")  # Allow additional fields from XML


class IdsInfo(BaseModel):
    """Basic IDS information."""

    name: str
    description: str
    version: Optional[str] = None
    max_depth: int = 0
    leaf_count: int = 0
    physics_domain: PhysicsDomain = PhysicsDomain.GENERAL
    documentation_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    related_ids: List[str] = Field(default_factory=list)
    common_use_cases: List[str] = Field(default_factory=list)

    @field_validator("documentation_coverage")
    @classmethod
    def validate_coverage(cls, v):
        """Ensure coverage is between 0 and 1."""
        return max(0.0, min(1.0, v))


class IdsDetailed(BaseModel):
    """Detailed IDS information."""

    ids_info: IdsInfo
    coordinate_systems: Dict[str, CoordinateSystem] = Field(default_factory=dict)
    paths: Dict[str, DataPath] = Field(default_factory=dict)
    semantic_groups: Dict[str, List[str]] = Field(default_factory=dict)


class CatalogMetadata(BaseModel):
    """Catalog metadata structure."""

    version: str
    generation_date: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat() + "Z"
    )
    total_ids: int
    total_leaf_nodes: int
    total_relationships: int = 0  # Added for relationships metadata


class IdsCatalog(BaseModel):
    """High-level IDS catalog structure."""

    metadata: CatalogMetadata
    ids_catalog: Dict[str, IdsInfo]


class RelationshipInfo(BaseModel):
    """Information about relationships between IDS paths."""

    type: str
    description: str
    paths: List[str] = Field(default_factory=list)


class CrossIdsRelationship(BaseModel):
    """Cross-IDS relationship information."""

    type: str
    relationships: List[Dict[str, Any]] = Field(default_factory=list)


class PhysicsConcept(BaseModel):
    """Physics concept with related paths."""

    description: str
    relevant_paths: List[str] = Field(default_factory=list)
    key_relationships: List[str] = Field(default_factory=list)


class UnitFamily(BaseModel):
    """Unit family definition."""

    base_unit: str
    paths_using: List[str] = Field(default_factory=list)
    conversion_factors: Dict[str, float] = Field(default_factory=dict)


class Relationships(BaseModel):
    """Complete relationship graph structure."""

    metadata: CatalogMetadata = Field(
        default_factory=lambda: CatalogMetadata(
            version="unknown", total_ids=0, total_leaf_nodes=0
        )
    )
    cross_references: Dict[str, CrossIdsRelationship] = Field(default_factory=dict)
    physics_concepts: Dict[str, PhysicsConcept] = Field(default_factory=dict)
    unit_families: Dict[str, UnitFamily] = Field(default_factory=dict)


class TransformationOutputs(BaseModel):
    """Output paths from data dictionary transformation."""

    catalog: Path
    detailed: List[Path] = Field(default_factory=list)
    relationships: Path
