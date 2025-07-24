"""Unified Pydantic models for IMAS MCP tool responses."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

# Import physics models
from imas_mcp.models.physics_models import PhysicsSearchResult


# ============================================================================
# COMMON BASE MODELS - Used across multiple tools
# ============================================================================


class ToolResponse(BaseModel):
    """Base response model for all IMAS MCP tools."""

    # Core fields that every tool response should have
    ai_prompt: Optional[str] = Field(None, description="AI enhancement prompt")

    # Error handling
    error: Optional[str] = Field(None, description="Error message if tool failed")
    suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for user"
    )


class DataPath(BaseModel):
    """Represents an IMAS data path with metadata."""

    path: str
    ids_name: str
    documentation: str
    relevance_score: float
    physics_domain: Optional[str] = None
    units: Optional[str] = None
    data_type: Optional[str] = None
    identifier: Dict[str, Any] = Field(
        default_factory=dict, description="Identifier schema information"
    )


class IdentifierInfo(BaseModel):
    """Information about identifier schemas and branching logic."""

    has_identifier: bool = False
    schema_path: Optional[str] = None
    option_count: Optional[int] = None
    branching_significance: Optional[str] = None
    sample_options: List[Dict[str, Any]] = Field(default_factory=list)


# ============================================================================
# TOOL-SPECIFIC RESPONSE MODELS - One per tool, inheriting from ToolResponse
# ============================================================================


class SearchResponse(ToolResponse):
    """Response from search_imas tool."""

    query: Union[str, List[str]]
    search_mode: str
    search_strategy: str
    ids_filter: Optional[str] = None
    results_count: int
    results: List[DataPath] = Field(default_factory=list)
    physics_enhancement: Optional[PhysicsSearchResult] = None


class ConceptResponse(ToolResponse):
    """Response from explain_concept tool."""

    concept: str
    detail_level: str
    sources_analyzed: int
    explanation: Dict[str, str]  # definition, physics_context, etc.
    related_paths: List[DataPath] = Field(default_factory=list)
    physics_domains: List[str] = Field(default_factory=list)
    physics_context: Optional[PhysicsSearchResult] = None
    measurement_contexts: List[Dict[str, Any]] = Field(default_factory=list)
    identifier_analysis: Dict[str, Any] = Field(default_factory=dict)
    physics_enhancement: Optional[Dict[str, Any]] = Field(
        None, description="Physics context enhancement from definitions"
    )


class OverviewResponse(ToolResponse):
    """Response from get_overview tool."""

    total_ids: int
    available_ids: List[str] = Field(default_factory=list)
    physics_domains: List[str] = Field(default_factory=list)
    data_types: List[str] = Field(default_factory=list)
    common_units: List[str] = Field(default_factory=list)
    sample_analysis: Dict[str, int] = Field(default_factory=dict)
    identifier_summary: Dict[str, Any] = Field(default_factory=dict)
    ids_statistics: Dict[str, Any] = Field(
        default_factory=dict, description="Per-IDS statistics"
    )
    question: Optional[str] = None
    question_analysis: Optional[Dict[str, Any]] = None
    question_results: Optional[List[Dict[str, Any]]] = None
    usage_guidance: Dict[str, Any] = Field(default_factory=dict)


class StructureResponse(ToolResponse):
    """Response from analyze_ids_structure tool."""

    ids_name: str
    total_paths: int
    description: str
    structure: Dict[str, int]  # root_level_paths, max_depth, document_count
    identifier_analysis: Dict[str, Any] = Field(default_factory=dict)
    sample_paths: List[str] = Field(default_factory=list)
    path_patterns: Dict[str, int] = Field(default_factory=dict)


class RelationshipResponse(ToolResponse):
    """Response from explore_relationships tool."""

    path: str
    relationship_type: str
    max_depth: int
    ids_name: str
    related_paths: List[DataPath] = Field(default_factory=list)
    relationship_count: int
    analysis: Dict[str, int] = Field(default_factory=dict)
    identifier_context: Dict[str, Any] = Field(default_factory=dict)
    physics_relationships: Optional[Dict[str, Any]] = None


class IdentifierResponse(ToolResponse):
    """Response from explore_identifiers tool."""

    query: Optional[str] = None
    scope: str
    total_schemas: int
    schemas: List[Dict[str, Any]] = Field(default_factory=list)
    identifier_paths: List[Dict[str, Any]] = Field(
        default_factory=list, description="Identifier paths for scope='paths'"
    )
    branching_analytics: Dict[str, Any] = Field(default_factory=dict)


class BulkExportResponse(ToolResponse):
    """Response from export_ids tool."""

    ids_list: List[str]
    include_relationships: bool
    include_physics_context: bool
    output_format: str
    export_data: Dict[str, Any] = Field(default_factory=dict)


class DomainExportResponse(ToolResponse):
    """Response from export_physics_domain tool."""

    domain: str
    include_cross_domain: bool
    analysis_depth: str
    max_paths: int
    export_data: Dict[str, Any] = Field(default_factory=dict)
