"""
Context models for IMAS MCP tool operations.

These models represent shared context components that can be composed
into both service contexts and response models.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from imas_mcp.models.constants import SearchMode
from imas_mcp.models.physics_models import PhysicsSearchResult


# ============================================================================
# CONTEXT MODELS (shared components)
# ============================================================================


class QueryContext(BaseModel):
    """Query metadata and parameters."""

    query: Optional[Union[str, List[str]]] = Field(
        default=None, description="Original user query"
    )
    search_mode: Optional[SearchMode] = Field(
        default=SearchMode.AUTO, description="Search mode used"
    )
    ids_filter: Optional[List[str]] = Field(
        default=None, description="IDS filter applied"
    )
    max_results: Optional[int] = Field(
        default=None, description="Maximum results requested"
    )


class AIContext(BaseModel):
    """AI enhancement context."""

    ai_prompt: Dict[str, str] = Field(
        default_factory=dict,
        description="AI prompts that were used",
    )
    ai_response: Dict[str, Any] = Field(
        default_factory=dict,
        description="AI-generated responses",
    )


class PhysicsContext(BaseModel):
    """Physics enhancement context."""

    physics_domains: List[str] = Field(default_factory=list)
    physics_context: Optional[PhysicsSearchResult] = None


class ToolMetadata(BaseModel):
    """Standard tool metadata."""

    tool_name: str
    processing_timestamp: str = Field(default="")
    version: str = Field(default="1.0.0")
    operation_type: Optional[str] = Field(default=None)


class ExportContext(BaseModel):
    """Export operation context."""

    include_relationships: bool = True
    output_format: str = "structured"
    analysis_depth: str = "focused"
    include_cross_domain: bool = False
    max_paths: int = 10


class AnalysisContext(BaseModel):
    """Analysis operation context."""

    ids_name: Optional[str] = None
    analysis_type: str = "structure"
    max_depth: int = 0
    include_identifiers: bool = True


class RelationshipContext(BaseModel):
    """Relationship exploration context."""

    path: str
    relationship_type: str = "all"
    max_depth: int = 2
    include_cross_ids: bool = True
