"""
Input validation schemas for explain tool.

This module defines Pydantic schemas for validating inputs to the explain_concept tool.
"""

from pydantic import BaseModel, Field, field_validator


class ExplainInputSchema(BaseModel):
    """Input validation schema for explain_concept tool."""

    concept: str = Field(
        min_length=1,
        max_length=200,
        description="Concept to explain within IMAS context",
    )
    detail_level: str = Field(
        default="intermediate",
        description="Level of detail for explanation (basic, intermediate, advanced)",
    )

    @field_validator("detail_level")
    @classmethod
    def validate_detail_level(cls, v):
        """Validate detail level is one of allowed values."""
        valid_levels = ["basic", "intermediate", "advanced"]
        if v not in valid_levels:
            raise ValueError(f"Invalid detail_level. Must be one of: {valid_levels}")
        return v

    @field_validator("concept")
    @classmethod
    def validate_concept(cls, v):
        """Validate concept is meaningful."""
        # Remove extra whitespace
        v = v.strip()
        if not v:
            raise ValueError("Concept cannot be empty")

        # Check for minimum meaningful length
        if len(v) < 2:
            raise ValueError("Concept must be at least 2 characters long")

        return v
