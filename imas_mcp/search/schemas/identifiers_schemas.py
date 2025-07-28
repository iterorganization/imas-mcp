"""
Input validation schemas for identifiers tool.

This module defines Pydantic schemas for validating inputs to the explore_identifiers tool.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class IdentifiersInputSchema(BaseModel):
    """Input validation schema for explore_identifiers tool."""

    query: Optional[str] = Field(
        default=None, max_length=200, description="Optional query to filter identifiers"
    )
    scope: str = Field(default="all", description="Scope of identifier exploration")

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

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, v):
        """Validate scope parameter."""
        valid_scopes = ["all", "enums", "identifiers", "coordinates", "constants"]
        if v not in valid_scopes:
            raise ValueError(f"Invalid scope. Must be one of: {valid_scopes}")
        return v
