"""
Input validation schemas for analysis tool.

This module defines Pydantic schemas for validating inputs to the analyze_ids_structure tool.
"""

from pydantic import BaseModel, Field, field_validator


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
        # Remove extra whitespace
        v = v.strip()
        if not v:
            raise ValueError("IDS name cannot be empty")

        # Check for valid IDS name format
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "IDS name must contain only alphanumeric characters, underscores, and hyphens"
            )

        return v
