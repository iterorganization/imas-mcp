"""
Request/Input validation models for all tools.

This module consolidates all input validation schemas that were previously
scattered across search/schemas/ directory.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import (
    AnalysisDepth,
    DetailLevel,
    IdentifierScope,
    OutputFormat,
    RelationshipType,
    ResponseProfile,
    SearchMode,
)


class OutputMode(str, Enum):
    """Output mode for search results."""

    FULL = "full"
    COMPACT = "compact"


class BaseInputSchema(BaseModel):
    """Base schema for all input validation with strict parameter checking."""

    model_config = ConfigDict(extra="forbid")


class SearchInput(BaseInputSchema):
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
        default=50,
        ge=1,
        description="Maximum number of hits to return (summary contains all matches)",
    )
    response_profile: ResponseProfile = Field(
        default=ResponseProfile.STANDARD,
        description="Response preset: minimal | standard | detailed",
    )
    ids_filter: list[str] | None = Field(
        default=None,
        description="Optional list of IDS names to filter search results",
    )
    # Hints merged into response_profile

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate search query."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v

    @field_validator("ids_filter", mode="before")
    @classmethod
    def validate_ids_filter(cls, v):
        """Convert string ids_filter to list."""
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("search_mode", mode="before")
    @classmethod
    def validate_search_mode(cls, v):
        """Convert string search_mode to enum, handle auto specially."""
        if v is None:
            return SearchMode.AUTO
        if isinstance(v, str):
            return SearchMode(v)
        return v

    @field_validator("response_profile", mode="before")
    @classmethod
    def validate_response_profile(cls, v):
        """Convert string response_profile to enum."""
        if v is None:
            return ResponseProfile.STANDARD
        if isinstance(v, str):
            return ResponseProfile(v)
        return v


class ExplainInput(BaseInputSchema):
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

    @field_validator("concept")
    @classmethod
    def validate_concept(cls, v):
        """Validate concept format."""
        v = v.strip()
        if not v:
            raise ValueError("Concept cannot be empty")
        return v


class AnalysisInput(BaseInputSchema):
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


class RelationshipsInput(BaseInputSchema):
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


class IdentifiersInput(BaseInputSchema):
    """Input validation schema for explore_identifiers tool."""

    scope: IdentifierScope = Field(
        default=IdentifierScope.ALL,
        description="Scope of identifier exploration",
    )
    query: str | None = Field(
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


class OverviewInput(BaseInputSchema):
    """Input validation schema for get_overview tool."""

    query: str | None = Field(
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


class ExportIdsInput(BaseInputSchema):
    """Input validation schema for export_ids tool."""

    ids_list: list[str] = Field(
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


class ExportPhysicsDomainInput(BaseInputSchema):
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


class ListFormat(str, Enum):
    """Output format for list_imas_paths tool."""

    YAML = "yaml"
    LIST = "list"
    JSON = "json"
    DICT = "dict"


class ListPathsInput(BaseInputSchema):
    """Input validation schema for list_imas_paths tool."""

    paths: str = Field(
        min_length=1,
        max_length=500,
        description="Space-delimited IDS names or path prefixes",
    )
    format: ListFormat = Field(
        default=ListFormat.YAML,
        description="Output format: yaml (indented tree, default, most token-efficient), list (array of path strings), json (JSON string), or dict (Python dictionary)",
    )
    leaf_only: bool = Field(
        default=False,
        description="Return only leaf nodes (actual data paths)",
    )
    include_ids_prefix: bool = Field(
        default=True,
        description="Include IDS name in paths",
    )
    max_paths: int | None = Field(
        default=None,
        ge=1,
        description="Maximum paths per IDS/prefix (defaults based on format)",
    )

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, v):
        """Validate paths format."""
        v = v.strip()
        if not v:
            raise ValueError("Paths cannot be empty")
        return v

    @field_validator("max_paths")
    @classmethod
    def validate_max_paths(cls, v, info):
        """Validate max_paths based on format with helpful error messages."""
        format_val = info.data.get("format", ListFormat.JSON)

        # Format-specific limits and defaults based on token analysis
        # YAML is ~2.7x more efficient than list due to shared prefixes
        limits = {
            ListFormat.DICT: 5000,  # Native Python dict, most efficient
            ListFormat.JSON: 5000,  # 30-40% token reduction, robust parsing
            ListFormat.YAML: 5000,  # ~2.7x token reduction (172% savings), clean indentation
            ListFormat.LIST: 3000,  # Baseline efficiency (full path strings)
        }

        # Set default based on format if not provided
        if v is None:
            v = limits.get(format_val, 3000)

        max_limit = limits.get(format_val, 3000)

        if v > max_limit:
            # Build helpful error message
            suggestions = []

            # Suggest lower limit
            suggestions.append(f"reduce max_paths to {max_limit} or less")

            # Suggest more efficient format
            if format_val == ListFormat.LIST:
                suggestions.append(
                    "use format='yaml' for better efficiency (~2.7x fewer tokens, limit: 5000)"
                )

            # Suggest filtering by prefix
            suggestions.append(
                "filter by path prefix (e.g., 'equilibrium/time_slice' instead of 'equilibrium')"
            )

            raise ValueError(
                f"max_paths={v} exceeds {format_val.value} format limit of {max_limit}. "
                f"Options: {' OR '.join(suggestions)}"
            )

        return v


# Documentation search tool input schemas


class SearchDocsInput(BaseInputSchema):
    """Input validation schema for search_docs tool."""

    query: str = Field(
        min_length=1,
        max_length=500,
        description="Documentation search query",
    )
    library: str | None = Field(
        default=None,
        max_length=100,
        description="Library name to search (required)",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return (1-20)",
    )
    version: str | None = Field(
        default=None,
        max_length=50,
        description="Specific library version to search",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate search query."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v

    @field_validator("library")
    @classmethod
    def validate_library(cls, v):
        """Validate library name."""
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Library name cannot be empty")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v):
        """Validate version string."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v


class ListDocsInput(BaseInputSchema):
    """Input validation schema for list_docs tool."""

    library: str | None = Field(
        default=None,
        max_length=100,
        description="Specific library name to get versions for (optional)",
    )

    @field_validator("library")
    @classmethod
    def validate_library(cls, v):
        """Validate library name."""
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Library name cannot be empty")
        return v
