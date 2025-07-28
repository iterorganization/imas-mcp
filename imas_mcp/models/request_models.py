"""
Request/Input validation models for all tools.

This module consolidates all input validation schemas that were previously
scattered across search/schemas/ directory.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from .constants import (
    SearchMode,
    DetailLevel,
    RelationshipType,
    IdentifierScope,
    OutputFormat,
    AnalysisDepth,
)


class SearchInputSchema(BaseModel):
    """Input validation schema for search_imas tool."""

    query: str = Field(
        min_length=1,
        max_length=500,
        description="Search query for IMAS data",
    )
    search_mode: SearchMode = Field(
        default=SearchMode.AUTO,
        description="Search mode to use",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    ids_filter: Optional[List[str]] = Field(
        default=None,
        description="Optional list of IDS names to filter search results",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate search query."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v


class ExplainInputSchema(BaseModel):
    """Input validation schema for explain_concept tool."""

    concept: str = Field(
        min_length=1,
        max_length=200,
        description="Physics concept to explain",
    )
    detail_level: DetailLevel = Field(
        default=DetailLevel.INTERMEDIATE,
        description="Level of detail for explanation",
    )
    context: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional context for more targeted explanation",
    )

    @field_validator("concept")
    @classmethod
    def validate_concept(cls, v):
        """Validate concept format."""
        v = v.strip()
        if not v:
            raise ValueError("Concept cannot be empty")
        return v


class AnalysisInputSchema(BaseModel):
    """Input validation schema for analyze_ids_structure tool."""

    ids_name: str = Field(
        min_length=1,
        max_length=100,
        description="Name of the IDS to analyze structurally",
    )

    @field_validator("ids_name")
    @classmethod
    def validate_ids_name(cls, v):
        """Validate IDS name format."""
        v = v.strip()
        if not v:
            raise ValueError("IDS name cannot be empty")

        # Check for valid IDS name format
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "IDS name must contain only alphanumeric characters, underscores, and hyphens"
            )

        return v


class RelationshipsInputSchema(BaseModel):
    """Input validation schema for explore_relationships tool."""

    path: str = Field(
        min_length=1,
        max_length=500,
        description="Data path to explore relationships for",
    )
    relationship_type: RelationshipType = Field(
        default=RelationshipType.ALL,
        description="Type of relationships to explore",
    )
    max_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum relationship depth to explore",
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v):
        """Validate path format."""
        v = v.strip()
        if not v:
            raise ValueError("Path cannot be empty")

        # Check for hierarchical path format
        if "/" not in v and "." not in v:
            raise ValueError("Path should contain hierarchical separators (/ or .)")

        return v


class IdentifiersInputSchema(BaseModel):
    """Input validation schema for explore_identifiers tool."""

    scope: IdentifierScope = Field(
        default=IdentifierScope.ALL,
        description="Scope of identifier exploration",
    )
    query: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional query to filter identifiers",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate query if provided."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v


class OverviewInputSchema(BaseModel):
    """Input validation schema for get_overview tool."""

    query: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional query for focused overview",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate query if provided."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v


class ExportIdsInputSchema(BaseModel):
    """Input validation schema for export_ids tool."""

    ids_list: List[str] = Field(
        min_length=1,
        description="List of IDS names to export",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.STRUCTURED,
        description="Output format for export",
    )
    include_relationships: bool = Field(
        default=True,
        description="Whether to include relationship analysis",
    )
    include_physics: bool = Field(
        default=True,
        description="Whether to include physics context",
    )

    @field_validator("ids_list")
    @classmethod
    def validate_ids_list(cls, v):
        """Validate IDS list."""
        if not v:
            raise ValueError("No IDS specified for export")

        # Validate each IDS name
        for ids_name in v:
            if not isinstance(ids_name, str):
                raise ValueError("All IDS names must be strings")

            ids_name = ids_name.strip()
            if not ids_name:
                raise ValueError("IDS names cannot be empty")

            if not ids_name.replace("_", "").replace("-", "").isalnum():
                raise ValueError(
                    f"Invalid IDS name '{ids_name}': must contain only alphanumeric characters, underscores, and hyphens"
                )

        return [ids.strip() for ids in v]


class ExportPhysicsDomainInputSchema(BaseModel):
    """Input validation schema for export_physics_domain tool."""

    domain: str = Field(
        min_length=1,
        max_length=100,
        description="Physics domain to export",
    )
    analysis_depth: AnalysisDepth = Field(
        default=AnalysisDepth.OVERVIEW,
        description="Depth of analysis for domain export",
    )
    max_paths: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of paths to include",
    )
    include_cross_domain: bool = Field(
        default=False,
        description="Whether to include cross-domain relationships",
    )

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v):
        """Validate domain format."""
        v = v.strip()
        if not v:
            raise ValueError("Domain cannot be empty")

        # Allow alphanumeric, underscores, hyphens, and spaces
        if not v.replace("_", "").replace("-", "").replace(" ", "").isalnum():
            raise ValueError(
                "Domain must contain only alphanumeric characters, underscores, hyphens, and spaces"
            )

        return v
