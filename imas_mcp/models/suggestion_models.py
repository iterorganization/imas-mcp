"""Suggestion model types used across services and responses."""

from typing import Optional
from pydantic import BaseModel, Field


class ToolSuggestion(BaseModel):
    """A tool suggestion with context."""

    tool_name: str = Field(description="Name of the suggested tool")
    description: str = Field(description="Description of what the tool does")
    relevance: Optional[str] = Field(
        default=None, description="Why this tool is relevant"
    )


class SearchSuggestion(BaseModel):
    """A search suggestion with context."""

    suggestion: str = Field(description="The suggested search term or phrase")
    reason: Optional[str] = Field(
        default=None, description="Why this suggestion is relevant"
    )
    confidence: Optional[float] = Field(
        default=None, description="Confidence score 0-1"
    )
