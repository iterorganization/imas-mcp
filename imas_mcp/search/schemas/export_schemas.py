"""
Input validation schemas for export tool.

This module defines Pydantic schemas for validating inputs to export tools.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List


class ExportIdsInputSchema(BaseModel):
    """Input validation schema for export_ids tool."""

    ids_list: List[str] = Field(min_length=1, description="List of IDS names to export")
    include_physics_context: bool = Field(
        default=True, description="Include physics context in export"
    )
    include_relationships: bool = Field(
        default=True, description="Include relationship information in export"
    )
    output_format: str = Field(
        default="structured", description="Output format for export"
    )

    @field_validator("ids_list")
    @classmethod
    def validate_ids_list(cls, v):
        """Validate IDS list."""
        if not v:
            raise ValueError("IDS list cannot be empty")

        # Validate each IDS name
        validated_list = []
        for ids_name in v:
            ids_name = ids_name.strip()
            if not ids_name:
                continue
            if not ids_name.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Invalid IDS name: {ids_name}")
            validated_list.append(ids_name)

        if not validated_list:
            raise ValueError("No valid IDS names provided")

        return validated_list

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v):
        """Validate output format."""
        valid_formats = ["structured", "detailed", "summary", "json", "markdown"]
        if v not in valid_formats:
            raise ValueError(f"Invalid output_format. Must be one of: {valid_formats}")
        return v


class ExportPhysicsDomainInputSchema(BaseModel):
    """Input validation schema for export_physics_domain tool."""

    domain: str = Field(
        min_length=1, max_length=100, description="Physics domain to export"
    )
    max_paths: int = Field(
        default=10, ge=1, le=100, description="Maximum number of paths to include"
    )
    analysis_depth: str = Field(
        default="focused", description="Depth of analysis for domain export"
    )
    include_cross_domain: bool = Field(
        default=False, description="Include cross-domain relationships"
    )

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v):
        """Validate physics domain name."""
        # Remove extra whitespace
        v = v.strip()
        if not v:
            raise ValueError("Domain cannot be empty")
        return v

    @field_validator("analysis_depth")
    @classmethod
    def validate_analysis_depth(cls, v):
        """Validate analysis depth."""
        valid_depths = ["shallow", "focused", "comprehensive", "detailed"]
        if v not in valid_depths:
            raise ValueError(f"Invalid analysis_depth. Must be one of: {valid_depths}")
        return v
