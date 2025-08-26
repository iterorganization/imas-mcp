"""Data models for IMAS Data Dictionary using Pydantic."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PhysicsDomain(str, Enum):
    """Physics domains in IMAS with comprehensive categorization."""

    # Core plasma physics domains
    EQUILIBRIUM = "equilibrium"
    TRANSPORT = "transport"
    MHD = "mhd"
    TURBULENCE = "turbulence"

    # Heating and current drive
    HEATING = "heating"
    CURRENT_DRIVE = "current_drive"

    # Plasma-material interactions
    WALL = "wall"
    DIVERTOR = "divertor"
    EDGE_PHYSICS = "edge_physics"

    # Diagnostics and measurements
    PARTICLE_DIAGNOSTICS = "particle_diagnostics"
    ELECTROMAGNETIC_DIAGNOSTICS = "electromagnetic_diagnostics"
    RADIATION_DIAGNOSTICS = "radiation_diagnostics"
    MAGNETIC_DIAGNOSTICS = "magnetic_diagnostics"
    MECHANICAL_DIAGNOSTICS = "mechanical_diagnostics"

    # Control and operation
    CONTROL = "control"
    OPERATIONAL = "operational"

    # System components
    COILS = "coils"
    STRUCTURE = "structure"
    SYSTEMS = "systems"

    # Data and workflow
    DATA_MANAGEMENT = "data_management"
    WORKFLOW = "workflow"

    # Fallback
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
    range: list[float] | None = None
    usage: str


class PhysicsContext(BaseModel):
    """Physics context for a data field."""

    domain: str
    phenomena: list[str] = Field(default_factory=list)
    typical_values: dict[str, str] = Field(default_factory=dict)


class ValidationRules(BaseModel):
    """Validation rules for data fields."""

    min_value: float | None = None
    max_value: float | None = None
    units_required: bool = True
    coordinate_check: str | None = None


class UsageExample(BaseModel):
    """Code usage example."""

    scenario: str
    code: str
    notes: str


class IdentifierOption(BaseModel):
    """Single identifier enumeration option."""

    name: str
    index: int
    description: str


class IdentifierSchema(BaseModel):
    """Complete identifier schema from XML file."""

    schema_path: str  # The path used to access the schema (e.g., 'equilibrium/equilibrium_profiles_2d_identifier.xml')
    documentation: str | None = None  # Documentation from the schema file
    options: list[IdentifierOption] = Field(
        default_factory=list
    )  # Available enumeration values
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )  # Additional metadata from schema


class IdsNode(BaseModel):
    """Complete data path information extracted from XML."""

    path: str
    documentation: str
    units: str | None = None  # Make units optional
    coordinates: list[str] = Field(default_factory=list)
    lifecycle: str = "active"
    data_type: str | None = None
    introduced_after_version: str | None = None  # Renamed from introduced_after
    lifecycle_status: str | None = None  # Added lifecycle status field
    lifecycle_version: str | None = None  # Added lifecycle version field
    physics_context: PhysicsContext | None = None
    related_paths: list[str] = Field(default_factory=list)
    usage_examples: list[UsageExample] = Field(default_factory=list)
    validation_rules: ValidationRules | None = None
    identifier_schema: IdentifierSchema | None = (
        None  # Schema information for identifier fields
    )

    # Additional XML attributes
    coordinate1: str | None = None
    coordinate2: str | None = None
    timebase: str | None = None
    type: str | None = None
    structure_reference: str | None = None  # Reference to structure definition

    model_config = ConfigDict(extra="allow")  # Allow additional fields from XML


class IdsInfo(BaseModel):
    """Basic IDS information."""

    name: str
    description: str
    version: str | None = None
    max_depth: int = 0
    leaf_count: int = 0
    physics_domain: PhysicsDomain = PhysicsDomain.GENERAL
    documentation_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    related_ids: list[str] = Field(default_factory=list)
    common_use_cases: list[str] = Field(default_factory=list)

    @field_validator("documentation_coverage")
    @classmethod
    def validate_coverage(cls, v):
        """Ensure coverage is between 0 and 1."""
        return max(0.0, min(1.0, v))


class IdsDetailed(BaseModel):
    """Detailed IDS information."""

    ids_info: IdsInfo
    coordinate_systems: dict[str, CoordinateSystem] = Field(default_factory=dict)
    paths: dict[str, IdsNode] = Field(default_factory=dict)
    semantic_groups: dict[str, list[str]] = Field(default_factory=dict)


class CatalogMetadata(BaseModel):
    """Catalog metadata structure."""

    version: str
    generation_date: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat() + "Z"
    )
    total_ids: int
    total_leaf_nodes: int
    total_paths: int = 0  # Total number of paths across all IDS
    total_relationships: int = 0  # Added for relationships metadata


class IdsCatalog(BaseModel):
    """High-level IDS catalog structure."""

    metadata: CatalogMetadata
    ids_catalog: dict[str, IdsInfo]


class RelationshipInfo(BaseModel):
    """Information about relationships between IDS paths."""

    type: str
    description: str
    paths: list[str] = Field(default_factory=list)


class CrossIdsRelationship(BaseModel):
    """Cross-IDS relationship information."""

    type: str
    relationships: list[dict[str, Any]] = Field(default_factory=list)


class PhysicsConcept(BaseModel):
    """Physics concept with related paths."""

    description: str
    relevant_paths: list[str] = Field(default_factory=list)
    key_relationships: list[str] = Field(default_factory=list)


class UnitFamily(BaseModel):
    """Unit family definition."""

    base_unit: str
    paths_using: list[str] = Field(default_factory=list)
    conversion_factors: dict[str, float] = Field(default_factory=dict)


class Relationships(BaseModel):
    """Complete relationship graph structure."""

    metadata: CatalogMetadata = Field(
        default_factory=lambda: CatalogMetadata(
            version="unknown", total_ids=0, total_leaf_nodes=0
        )
    )
    cross_references: dict[str, CrossIdsRelationship] = Field(default_factory=dict)
    physics_concepts: dict[str, PhysicsConcept] = Field(default_factory=dict)
    unit_families: dict[str, UnitFamily] = Field(default_factory=dict)


class TransformationOutputs(BaseModel):
    """Output paths from data dictionary transformation."""

    catalog: Path
    detailed: list[Path] = Field(default_factory=list)
    identifier_catalog: Path


class IdentifierPath(BaseModel):
    """Represents a single path with identifier schema."""

    path: str
    ids_name: str
    schema_name: str
    description: str
    option_count: int
    physics_domain: str | None = None
    usage_frequency: int = 1


class IdentifierCatalogSchema(BaseModel):
    """Complete identifier schema information for catalog."""

    schema_name: str
    schema_path: str
    description: str
    total_options: int
    options: list[IdentifierOption]
    usage_count: int
    usage_paths: list[str]
    physics_domains: list[str]
    branching_complexity: float  # Entropy measure


class IdentifierCatalog(BaseModel):
    """Complete identifier catalog structure."""

    metadata: CatalogMetadata
    schemas: dict[str, IdentifierCatalogSchema]
    paths_by_ids: dict[str, list[IdentifierPath]]
    cross_references: dict[str, list[str]]
    physics_mapping: dict[str, list[str]]
    branching_analytics: dict[str, Any]
