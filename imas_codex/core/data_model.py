"""Data models for IMAS Data Dictionary using Pydantic."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Import generated enum from LinkML schema
from imas_codex.core.physics_domain import PhysicsDomain


class DataLifecycle(StrEnum):
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


class ValidationRules(BaseModel):
    """Validation rules for data fields."""

    min_value: float | None = None
    max_value: float | None = None
    units_required: bool = True
    coordinate_check: str | None = None


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
    ids_name: str | None = None  # IDS name this path belongs to
    name: str | None = None  # Leaf name (last path component)
    units: str | None = None  # Make units optional
    coordinates: list[str] = Field(default_factory=list)
    lifecycle: str = "active"
    data_type: str | None = None
    physics_domain: str | None = None  # Physics domain from DD or inferred from IDS
    node_type: str | None = None  # Time-variation type (dynamic/constant/static)
    introduced_after_version: str | None = None  # Renamed from introduced_after
    lifecycle_status: str | None = None  # Added lifecycle status field
    lifecycle_version: str | None = None  # Added lifecycle version field
    cluster_labels: list[str] | None = None  # Cluster labels associated with this path
    validation_rules: ValidationRules | None = None
    identifier_schema: IdentifierSchema | None = (
        None  # Schema information for identifier fields
    )
    version_changes: list[dict[str, Any]] | None = (
        None  # Notable version changes (sign_convention, units, etc.)
    )

    # COCOS transformation fields
    cocos_label_transformation: str | None = None  # e.g., "psi_like", "ip_like"
    cocos_transformation_expression: str | None = None  # e.g., "- {psi_like}"

    # LLM enrichment fields
    description: str | None = None  # LLM-generated physics-aware description
    keywords: list[str] | None = None  # LLM-generated searchable keywords
    enrichment_source: str | None = None  # 'llm' or 'template'

    # Error fields (associated via HAS_ERROR relationships)
    error_fields: list[str] | None = None  # Error field paths linked to this data path

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
    total_ids: int = 0  # For IDS catalog: total IDS count; for identifier catalog: IDS using identifiers
    total_leaf_nodes: int = 0
    total_paths: int = 0  # Total number of paths across all IDS
    total_relationships: int = 0  # Added for relationships metadata
    total_schemas: int = 0  # Total number of identifier schemas


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
    branching_complexity: float  # Entropy measure


class IdentifierCatalog(BaseModel):
    """Complete identifier catalog structure."""

    metadata: CatalogMetadata
    schemas: dict[str, IdentifierCatalogSchema]
    paths_by_ids: dict[str, list[IdentifierPath]]
    branching_analytics: dict[str, Any]
