"""
Context models for IMAS MCP tool operations.

These models represent shared context components that can be composed
into tool result models using multiple inheritance.
"""

from typing import Any

from pydantic import BaseModel, Field

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.physics_models import PhysicsSearchResult
from imas_mcp.models.suggestion_models import SearchSuggestion, ToolSuggestion

# ============================================================================
# BASE RESULT MODELS
# ============================================================================


class BaseToolResult(BaseModel):
    """
    Minimal base for tool results.

    Contains only query tracking fields.
    """

    query: str | list[str] | None = Field(
        default=None, description="Original user query"
    )
    search_mode: SearchMode | None = Field(default=None, description="Search mode used")
    ids_filter: list[str] | str | None = Field(
        default=None, description="IDS filter applied"
    )
    max_results: int | None = Field(
        default=None, description="Maximum results requested"
    )


# ============================================================================
# FEATURE MIXINS (use with multiple inheritance)
# ============================================================================


class WithHints(BaseModel):
    """
    Adds query and tool hints to results.

    Use for tools that provide follow-up suggestions.
    """

    query_hints: list[SearchSuggestion] = Field(
        default_factory=list, description="Query suggestions for follow-up searches"
    )
    tool_hints: list[ToolSuggestion] = Field(
        default_factory=list, description="Tool suggestions for follow-up analysis"
    )


class WithPhysics(BaseModel):
    """
    Adds physics domain aggregation to results.

    Use for tools that return multiple paths/nodes where physics categorization is useful.
    """

    physics_domains: list[str] = Field(
        default_factory=list, description="Physics domains covered by results"
    )
    physics_context: PhysicsSearchResult | None = Field(
        default=None, description="Detailed physics search context"
    )


class WithAIEnhancement(BaseModel):
    """
    Adds AI prompt and response tracking.

    Use only for tools that use LLM enhancement (e.g., explain_concept).
    """

    ai_prompt: dict[str, Any] = Field(
        default_factory=dict, description="AI prompts that were used"
    )
    ai_response: dict[str, Any] = Field(
        default_factory=dict, description="AI-generated responses"
    )


# ============================================================================
# LEGACY COMPATIBILITY (for gradual migration)
# ============================================================================


class SearchParameters(BaseModel):
    """Base model for search configuration parameters."""

    search_mode: SearchMode | None = Field(
        default=SearchMode.AUTO, description="Search mode to use"
    )
    ids_filter: list[str] | None = Field(
        default=None, description="IDS filter to apply"
    )
    max_results: int | None = Field(
        default=None, description="Maximum results to return"
    )


class ToolMetadata(BaseModel):
    """Standard tool metadata."""

    tool_name: str
    processing_timestamp: str = Field(default="")
    version: str = Field(default="1.0.0")
    operation_type: str | None = Field(default=None)


class ExportContext(BaseModel):
    """Export operation context."""

    include_relationships: bool = True
    output_format: str = "structured"
    analysis_depth: str = "focused"
    include_cross_domain: bool = False
    max_paths: int = 10


class AnalysisContext(BaseModel):
    """Analysis operation context."""

    ids_name: str | None = None
    analysis_type: str = "structure"
    max_depth: int = 0
    include_identifiers: bool = True


class RelationshipContext(BaseModel):
    """Relationship exploration context."""

    path: str
    relationship_type: str = "all"
    max_depth: int = 2
    include_cross_ids: bool = True
