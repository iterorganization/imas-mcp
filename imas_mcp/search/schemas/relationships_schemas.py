"""
Input validation schemas for relationships tool.

This module defines Pydantic schemas for validating inputs to the explore_relationships tool.
"""

from pydantic import BaseModel, Field, field_validator


class RelationshipsInputSchema(BaseModel):
    """Input validation schema for explore_relationships tool."""

    path: str = Field(
        min_length=1,
        max_length=300,
        description="IMAS data path to explore relationships for",
    )
    max_depth: int = Field(
        default=2, ge=1, le=5, description="Maximum depth for relationship exploration"
    )
    relationship_type: str = Field(
        default="all", description="Type of relationships to explore"
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v):
        """Validate IMAS path format."""
        # Remove extra whitespace
        v = v.strip()
        if not v:
            raise ValueError("Path cannot be empty")

        # Basic path validation - should contain some structure
        if "/" not in v and "." not in v:
            raise ValueError("Path should contain hierarchical separators (/ or .)")

        return v

    @field_validator("relationship_type")
    @classmethod
    def validate_relationship_type(cls, v):
        """Validate relationship type."""
        valid_types = ["all", "semantic", "structural", "physics", "measurement"]
        if v not in valid_types:
            raise ValueError(
                f"Invalid relationship_type. Must be one of: {valid_types}"
            )
        return v
