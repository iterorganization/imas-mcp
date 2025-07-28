"""
Input validation schemas for search tools.

Provides Pydantic schemas for validating tool input parameters.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class SearchInputSchema(BaseModel):
    """Schema for search_imas input validation."""

    query: Union[str, List[str]] = Field(
        ..., description="Search query string or list of queries", min_length=1
    )
    ids_name: Optional[str] = Field(
        None, description="Optional IDS name to filter results"
    )
    max_results: int = Field(
        default=10, ge=1, le=100, description="Maximum number of results to return"
    )
    search_mode: str = Field(
        default="auto", description="Search mode: auto, semantic, lexical, or hybrid"
    )

    @field_validator("search_mode")
    @classmethod
    def validate_search_mode(cls, v: str) -> str:
        """Validate search mode parameter."""
        valid_modes = ["auto", "semantic", "lexical", "hybrid"]
        if v.lower() not in valid_modes:
            raise ValueError(f"Invalid search_mode. Must be one of: {valid_modes}")
        return v.lower()

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        """Validate query parameter."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Query string cannot be empty")
            return v.strip()
        elif isinstance(v, list):
            if not v:
                raise ValueError("Query list cannot be empty")
            # Filter out empty strings and validate
            filtered = [q.strip() for q in v if isinstance(q, str) and q.strip()]
            if not filtered:
                raise ValueError(
                    "Query list must contain at least one non-empty string"
                )
            return filtered
        else:
            raise ValueError("Query must be a string or list of strings")

    @field_validator("ids_name")
    @classmethod
    def validate_ids_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate IDS name parameter."""
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError("ids_name must be a string")
        return v.strip() if v.strip() else None


class ExplainInputSchema(BaseModel):
    """Schema for explain_concept input validation."""

    concept: str = Field(..., min_length=1, description="Concept to explain")
    detail_level: str = Field(
        default="intermediate",
        description="Level of detail: basic, intermediate, or advanced",
    )

    @field_validator("detail_level")
    @classmethod
    def validate_detail_level(cls, v: str) -> str:
        """Validate detail level parameter."""
        valid_levels = ["basic", "intermediate", "advanced"]
        if v.lower() not in valid_levels:
            raise ValueError(f"Invalid detail_level. Must be one of: {valid_levels}")
        return v.lower()

    @field_validator("concept")
    @classmethod
    def validate_concept(cls, v: str) -> str:
        """Validate concept parameter."""
        if not v.strip():
            raise ValueError("Concept cannot be empty")
        return v.strip()


class AnalysisInputSchema(BaseModel):
    """Schema for analyze_ids_structure input validation."""

    ids_name: str = Field(..., min_length=1, description="Name of the IDS to analyze")

    @field_validator("ids_name")
    @classmethod
    def validate_ids_name(cls, v: str) -> str:
        """Validate IDS name parameter."""
        if not v.strip():
            raise ValueError("IDS name cannot be empty")
        return v.strip()


class RelationshipInputSchema(BaseModel):
    """Schema for explore_relationships input validation."""

    path: str = Field(
        ..., min_length=1, description="Data path to explore relationships for"
    )
    max_depth: int = Field(
        default=2, ge=1, le=5, description="Maximum depth for relationship exploration"
    )
    relationship_type: str = Field(
        default="all", description="Type of relationships to explore"
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path parameter."""
        if not v.strip():
            raise ValueError("Path cannot be empty")
        return v.strip()

    @field_validator("relationship_type")
    @classmethod
    def validate_relationship_type(cls, v: str) -> str:
        """Validate relationship type parameter."""
        valid_types = ["all", "structural", "semantic", "physics"]
        if v.lower() not in valid_types:
            raise ValueError(
                f"Invalid relationship_type. Must be one of: {valid_types}"
            )
        return v.lower()


class IdentifierInputSchema(BaseModel):
    """Schema for explore_identifiers input validation."""

    query: Optional[str] = Field(
        None, description="Optional query to filter identifiers"
    )
    scope: str = Field(default="all", description="Scope of identifier exploration")

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, v: str) -> str:
        """Validate scope parameter."""
        valid_scopes = ["all", "ids", "physics", "units"]
        if v.lower() not in valid_scopes:
            raise ValueError(f"Invalid scope. Must be one of: {valid_scopes}")
        return v.lower()

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: Optional[str]) -> Optional[str]:
        """Validate query parameter."""
        if v is None:
            return v
        return v.strip() if v.strip() else None


class ExportInputSchema(BaseModel):
    """Schema for export tools input validation."""

    ids_list: List[str] = Field(
        ..., min_length=1, description="List of IDS names to export"
    )
    include_physics_context: bool = Field(
        default=True, description="Whether to include physics context"
    )
    include_relationships: bool = Field(
        default=True, description="Whether to include relationship information"
    )
    output_format: str = Field(
        default="structured", description="Output format for export"
    )

    @field_validator("ids_list")
    @classmethod
    def validate_ids_list(cls, v: List[str]) -> List[str]:
        """Validate IDS list parameter."""
        if not v:
            raise ValueError("IDS list cannot be empty")

        # Filter and validate each IDS name
        filtered = []
        for ids_name in v:
            if isinstance(ids_name, str) and ids_name.strip():
                filtered.append(ids_name.strip())

        if not filtered:
            raise ValueError("IDS list must contain at least one valid IDS name")

        return filtered

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format parameter."""
        valid_formats = ["structured", "json", "yaml", "markdown"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Invalid output_format. Must be one of: {valid_formats}")
        return v.lower()


class PhysicsDomainInputSchema(BaseModel):
    """Schema for export_physics_domain input validation."""

    domain: str = Field(..., min_length=1, description="Physics domain to export")
    analysis_depth: str = Field(
        default="focused", description="Analysis depth for export"
    )
    include_cross_domain: bool = Field(
        default=False, description="Whether to include cross-domain relationships"
    )
    max_paths: int = Field(
        default=10, ge=1, le=100, description="Maximum number of paths to export"
    )

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        """Validate domain parameter."""
        if not v.strip():
            raise ValueError("Domain cannot be empty")
        return v.strip()

    @field_validator("analysis_depth")
    @classmethod
    def validate_analysis_depth(cls, v: str) -> str:
        """Validate analysis depth parameter."""
        valid_depths = ["focused", "comprehensive", "detailed"]
        if v.lower() not in valid_depths:
            raise ValueError(f"Invalid analysis_depth. Must be one of: {valid_depths}")
        return v.lower()
