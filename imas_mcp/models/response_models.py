"""Clean, focused Pydantic models for IMAS MCP tool responses."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from imas_mcp.core.data_model import DataPath
from imas_mcp.search.search_strategy import SearchResult
from imas_mcp.models.physics_models import PhysicsSearchResult, ConceptExplanation
from imas_mcp.models.constants import (
    SearchMode,
    DetailLevel,
    RelationshipType,
    IdentifierScope,
)


# ============================================================================
# SUGGESTION MODELS
# ============================================================================


class SearchSuggestion(BaseModel):
    """A search suggestion with context."""

    suggestion: str = Field(description="The suggested search term or phrase")
    reason: Optional[str] = Field(
        default=None, description="Why this suggestion is relevant"
    )
    confidence: Optional[float] = Field(
        default=None, description="Confidence score 0-1"
    )


class ToolSuggestion(BaseModel):
    """A tool suggestion with context."""

    tool_name: str = Field(description="Name of the suggested tool")
    description: str = Field(description="Description of what the tool does")
    relevance: Optional[str] = Field(
        default=None, description="Why this tool is relevant"
    )


# ============================================================================
# ERROR HANDLING
# ============================================================================


class ErrorResponse(BaseModel):
    """Response for when tools encounter errors."""

    error: str
    suggestions: List[str] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None


# ============================================================================
# BASE MODELS
# ============================================================================


class QueryContext(BaseModel):
    """Provides original query context for LLM processing."""

    query: Optional[Union[str, List[str]]] = Field(
        default=None, description="Original user query that generated this response"
    )


class AIResponse(BaseModel):
    """Provides AI enhancement information."""

    ai_insights: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="AI enhancement insights and status information",
    )


class DataResponse(BaseModel):
    """Response containing IMAS data paths."""

    paths: List[DataPath] = Field(default_factory=list)
    count: int = 0


class PhysicsResponse(BaseModel):
    """Response with physics context."""

    physics_domains: List[str] = Field(default_factory=list)
    physics_context: Optional[PhysicsSearchResult] = None


class ExportResponse(BaseModel):
    """Response from export operations."""

    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# SEARCH & DISCOVERY
# ============================================================================


class SearchHit(SearchResult):
    """A single search hit that extends SearchResult with API-specific fields."""

    # API-specific fields (inherited from SearchResult: score->relevance_score, rank, search_mode, highlights)
    path: str = Field(description="Full IMAS path")
    documentation: str = Field(description="Path documentation")
    units: Optional[str] = Field(default=None, description="Physical units")
    data_type: Optional[str] = Field(default=None, description="Data type")
    ids_name: str = Field(description="IDS name this path belongs to")
    physics_domain: Optional[str] = Field(
        default=None, description="Physics domain classification"
    )

    # Make document field optional for API responses since we flatten its contents
    document: Optional[Any] = Field(
        default=None, exclude=True, description="Internal document reference"
    )

    @property
    def relevance_score(self) -> float:
        """Alias for score to maintain API compatibility."""
        return self.score


class SearchResponse(DataResponse, PhysicsResponse, QueryContext, AIResponse):
    """Search tool response."""

    # Core search results
    hits: List[SearchHit] = Field(default_factory=list, description="Search hits")
    search_mode: SearchMode = Field(
        default=SearchMode.AUTO, description="Search mode used"
    )
    query_hints: List[SearchSuggestion] = Field(
        default_factory=list, description="Query suggestions for follow-up searches"
    )
    tool_hints: List[ToolSuggestion] = Field(
        default_factory=list, description="Tool suggestions for follow-up analysis"
    )

    @property
    def hit_count(self) -> int:
        """Number of search hits returned."""
        return len(self.hits)


class OverviewResult(PhysicsResponse, QueryContext, AIResponse):
    """Overview tool response."""

    content: str
    available_ids: List[str] = Field(default_factory=list)


# ============================================================================
# ANALYSIS
# ============================================================================


class ConceptResult(DataResponse, PhysicsResponse, QueryContext, AIResponse):
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


class RelationshipResult(DataResponse, PhysicsResponse, QueryContext, AIResponse):
    """Relationship exploration response."""

    path: str
    relationship_type: RelationshipType = RelationshipType.ALL
    max_depth: int = 2
    connections: Dict[str, List[str]] = Field(default_factory=dict)


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
