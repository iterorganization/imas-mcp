"""Clean, focused Pydantic models for IMAS MCP tool responses."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from imas_mcp.core.data_model import IdsNode
from imas_mcp.search.search_strategy import SearchHit
from imas_mcp.models.physics_models import PhysicsSearchResult, ConceptExplanation
from imas_mcp.models.suggestion_models import ToolSuggestion, SearchSuggestion
from imas_mcp.models.constants import (
    SearchMode,
    DetailLevel,
    RelationshipType,
    IdentifierScope,
)


# ============================================================================
# BASE MODELS
# ============================================================================


class QueryContext(BaseModel):
    """Provides original query context and search metadata."""

    query: Optional[Union[str, List[str]]] = Field(
        default=None, description="Original user query that generated this response"
    )
    search_mode: Optional[SearchMode] = Field(
        default=SearchMode.AUTO, description="Search mode used for this query"
    )
    ids_filter: Optional[List[str]] = Field(
        default=None, description="IDS filter applied to search"
    )
    max_results: Optional[int] = Field(
        default=None, description="Maximum results requested"
    )


class AIResponse(BaseModel):
    """Provides AI enhancement information with separate prompts and responses."""

    ai_prompt: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="AI prompts that were used to generate responses",
    )
    ai_response: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="AI-generated responses and content",
    )


class IdsResponse(BaseModel):
    """Response containing IMAS data nodes."""

    nodes: List[IdsNode] = Field(default_factory=list)

    @property
    def node_count(self) -> int:
        """Number of nodes in the response."""
        return len(self.nodes)


class PhysicsResponse(BaseModel):
    """Response with physics context."""

    physics_domains: List[str] = Field(default_factory=list)
    physics_context: Optional[PhysicsSearchResult] = None


# ============================================================================
# EXPORT DATA STRUCTURES
# ============================================================================


class IdsPath(BaseModel):
    """Information about an exported path."""

    path: str = Field(description="Full IMAS path")
    documentation: str = Field(description="Path documentation")
    physics_domain: Optional[str] = Field(default=None, description="Physics domain")
    data_type: Optional[str] = Field(default=None, description="Data type")
    units: Optional[str] = Field(default=None, description="Physical units")


class IdsInfo(BaseModel):
    """Information about an exported IDS."""

    ids_name: str = Field(description="IDS name")
    description: Optional[str] = Field(default=None, description="IDS description")
    paths: List[IdsPath] = Field(default_factory=list, description="Paths in this IDS")
    physics_domains: List[str] = Field(
        default_factory=list, description="Physics domains"
    )
    measurement_types: List[str] = Field(
        default_factory=list, description="Measurement types"
    )


class ExportSummary(BaseModel):
    """Summary of export operation."""

    total_requested: int = Field(description="Total items requested")
    successfully_exported: int = Field(description="Successfully exported items")
    failed_exports: int = Field(description="Failed export count")
    total_paths_exported: int = Field(description="Total paths exported")
    export_completeness: str = Field(description="Export completeness status")


class ExportData(BaseModel):
    """Structured export data instead of free-form dict."""

    # For IDS exports
    ids_data: Optional[Dict[str, IdsInfo]] = Field(
        default=None, description="IDS export data"
    )
    cross_relationships: Optional[Dict[str, Any]] = Field(
        default=None, description="Cross-IDS relationships"
    )
    export_summary: Optional[ExportSummary] = Field(
        default=None, description="Export summary"
    )

    # For domain exports
    analysis_depth: Optional[str] = Field(
        default=None, description="Analysis depth used"
    )
    paths: Optional[List[IdsPath]] = Field(default=None, description="Domain paths")
    related_ids: Optional[List[str]] = Field(
        default=None, description="Related IDS names"
    )

    # For errors
    error: Optional[str] = Field(
        default=None, description="Error message if export failed"
    )
    explanation: Optional[str] = Field(default=None, description="Error explanation")
    suggestions: Optional[List[str]] = Field(
        default=None, description="Suggested actions"
    )


class ExportResponse(BaseModel):
    """Response from export operations with structured data."""

    data: ExportData = Field(
        default_factory=ExportData, description="Structured export data"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Export metadata"
    )


# ============================================================================
# SEARCH & DISCOVERY
# ============================================================================


class SearchHits(BaseModel):
    """Base class for responses that contain search hits."""

    hits: List["SearchHit"] = Field(default_factory=list, description="Search hits")

    @property
    def hit_count(self) -> int:
        """Number of search hits returned."""
        return len(self.hits)


class SearchResponse(
    IdsResponse, PhysicsResponse, QueryContext, AIResponse, SearchHits
):
    """Search tool response."""

    # Search-specific fields
    search_mode: SearchMode = Field(
        default=SearchMode.AUTO, description="Search mode used"
    )
    query_hints: List[SearchSuggestion] = Field(
        default_factory=list, description="Query suggestions for follow-up searches"
    )
    tool_hints: List[ToolSuggestion] = Field(
        default_factory=list, description="Tool suggestions for follow-up analysis"
    )


class OverviewResult(PhysicsResponse, QueryContext, AIResponse, SearchHits):
    """Overview tool response."""

    content: str
    available_ids: List[str] = Field(default_factory=list)

    # System analytics and statistics
    ids_statistics: Dict[str, Any] = Field(
        default_factory=dict, description="IDS usage and availability statistics"
    )
    usage_guidance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Usage guidance and getting started information",
    )


# ============================================================================
# ANALYSIS
# ============================================================================


class ConceptResult(IdsResponse, PhysicsResponse, QueryContext, AIResponse):
    """Concept explanation response."""

    concept: str
    explanation: str
    detail_level: DetailLevel = DetailLevel.INTERMEDIATE
    related_topics: List[str] = Field(default_factory=list)
    concept_explanation: Optional[ConceptExplanation] = None


class StructureResult(PhysicsResponse, AIResponse):
    """IDS structure analysis response."""

    ids_name: str
    description: str
    structure: Dict[str, int] = Field(default_factory=dict)
    sample_paths: List[str] = Field(default_factory=list)
    max_depth: int = 0


class IdentifierResult(AIResponse, BaseModel):
    """Identifier exploration response."""

    scope: IdentifierScope = IdentifierScope.ALL
    schemas: List[Dict[str, Any]] = Field(default_factory=list)
    paths: List[Dict[str, Any]] = Field(default_factory=list)
    analytics: Dict[str, Any] = Field(default_factory=dict)


class RelationshipResult(IdsResponse, PhysicsResponse, QueryContext, AIResponse):
    """Relationship exploration response."""

    path: str
    relationship_type: RelationshipType = RelationshipType.ALL
    max_depth: int = 2
    connections: Dict[str, List[str]] = Field(default_factory=dict)


# ============================================================================
# ERROR HANDLING
# ============================================================================


class ErrorResponse(AIResponse):
    """Error response with suggestions, context, and fallback data."""

    error: str = Field(description="Error message")
    suggestions: List[str] = Field(
        default_factory=list, description="Suggested actions"
    )
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    fallback_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional fallback data when primary operation fails"
    )


# ============================================================================
# EXPORT
# ============================================================================


class IDSExport(ExportResponse, AIResponse):
    """IDS export response."""

    ids_names: List[str]
    include_physics: bool = True
    include_relationships: bool = True


class DomainExport(ExportResponse, AIResponse):
    """Physics domain export response."""

    domain: str
    domain_info: Optional[Dict[str, Any]] = None
    include_cross_domain: bool = False
    max_paths: int = 10
