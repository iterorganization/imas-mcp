"""
Input validation schemas for overview tool.

This module defines Pydantic schemas for validating inputs to the get_overview tool.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class OverviewInputSchema(BaseModel):
    """Input validation schema for get_overview tool."""

    query: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional specific query about the IMAS data dictionary",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate query if provided."""
        if v is not None:
            # Remove extra whitespace
            v = v.strip()
            if v == "":
                return None  # Empty string becomes None
        return v
