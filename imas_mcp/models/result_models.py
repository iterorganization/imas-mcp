"""Clean, focused Pydantic models for IMAS MCP tool responses."""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from functools import cached_property
from typing import Any

from pydantic import BaseModel, Field

from imas_mcp import __version__
from imas_mcp.core.data_model import IdsNode
from imas_mcp.models.constants import (
    DetailLevel,
    IdentifierScope,
    RelationshipType,
    SearchMode,
)
from imas_mcp.models.context_models import (
    AIContext,
    PhysicsContext,
    QueryContext,
)
from imas_mcp.models.physics_models import ConceptExplanation
from imas_mcp.models.structure_models import StructureAnalysis
from imas_mcp.models.suggestion_models import SearchSuggestion, ToolSuggestion
from imas_mcp.search.search_strategy import SearchHit

# ============================================================================
# BASE MODELS
# ============================================================================


class ToolResult(AIContext, QueryContext, BaseModel, ABC):
    """
    Abstract base class for all IMAS MCP tool results.

    Provides standard context that all tools can benefit from:
    - Query tracking (query, search_mode, ids_filter, max_results)
    - AI enhancement capabilities (ai_response, ai_prompt)
    - Standard metadata fields (tool_name must be implemented by subclasses)
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Name of the tool that generated this result - must be implemented by subclasses."""
        pass

    @cached_property
    def processing_timestamp(self) -> str:
        """When this result was generated."""
        return datetime.now(UTC).isoformat()

    @property
    def version(self) -> str:
        """Result format version."""
        return __version__

    # Hint fields for follow-up suggestions (populated by decorators)
    query_hints: list[SearchSuggestion] = Field(
        default_factory=list, description="Query suggestions for follow-up searches"
    )
    tool_hints: list[ToolSuggestion] = Field(
        default_factory=list, description="Tool suggestions for follow-up analysis"
    )


class IdsResult(BaseModel):
    """Result containing IMAS data nodes."""

    nodes: list[IdsNode] = Field(default_factory=list)

    @property
    def node_count(self) -> int:
        """Number of nodes in the result."""
        return len(self.nodes)


# ============================================================================
# EXPORT DATA STRUCTURES
# ============================================================================


class IdsPath(BaseModel):
    """Information about an exported path."""

    path: str = Field(description="Full IMAS path")
    documentation: str = Field(description="Path documentation")
    physics_domain: str | None = Field(default=None, description="Physics domain")
    data_type: str | None = Field(default=None, description="Data type")
    units: str | None = Field(default=None, description="Physical units")


class IdsInfo(BaseModel):
    """Information about an exported IDS."""

    ids_name: str = Field(description="IDS name")
    description: str | None = Field(default=None, description="IDS description")
    paths: list[IdsPath] = Field(default_factory=list, description="Paths in this IDS")
    physics_domains: list[str] = Field(
        default_factory=list, description="Physics domains"
    )
    measurement_types: list[str] = Field(
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
    ids_data: dict[str, IdsInfo] | None = Field(
        default=None, description="IDS export data"
    )
    cross_relationships: dict[str, Any] | None = Field(
        default=None, description="Cross-IDS relationships"
    )
    export_summary: ExportSummary | None = Field(
        default=None, description="Export summary"
    )

    # For domain exports
    analysis_depth: str | None = Field(default=None, description="Analysis depth used")
    paths: list[IdsPath] | None = Field(default=None, description="Domain paths")
    related_ids: list[str] | None = Field(default=None, description="Related IDS names")

    # For errors
    error: str | None = Field(
        default=None, description="Error message if export failed"
    )
    explanation: str | None = Field(default=None, description="Error explanation")
    suggestions: list[str] | None = Field(default=None, description="Suggested actions")


class ExportResult(BaseModel):
    """Result from export operations with structured data."""

    data: ExportData = Field(
        default_factory=ExportData, description="Structured export data"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Export metadata"
    )


# ============================================================================
# SEARCH & DISCOVERY
# ============================================================================


class SearchHits(BaseModel):
    """Base class for responses that contain search hits."""

    # Summary at the very top for immediate visibility
    summary: dict[str, Any] = Field(
        default_factory=dict, description="Summary of all matching paths"
    )

    hits: list["SearchHit"] = Field(default_factory=list, description="Search hits")

    @property
    def hit_count(self) -> int:
        """Number of search hits returned."""
        return len(self.hits)


class SearchResult(ToolResult, IdsResult, PhysicsContext, SearchHits):
    """Search tool result."""

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "search_imas"

    # Search-specific fields
    search_mode: SearchMode = Field(
        default=SearchMode.AUTO, description="Search mode used"
    )


class OverviewResult(ToolResult, PhysicsContext, SearchHits):
    """Overview tool response."""

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "get_overview"

    content: str
    available_ids: list[str] = Field(default_factory=list)

    mcp_version: str | None = Field(default=None, description="MCP Server version")
    dd_version: str | None = Field(
        default=None, description="IMAS Data Dictionary version"
    )
    generation_date: str | None = Field(
        default=None, description="When the data dictionary was generated"
    )

    # Additional metadata
    total_leaf_nodes: int | None = Field(
        default=None, description="Total number of individual data elements"
    )
    identifier_schemas_count: int | None = Field(
        default=None, description="Number of available identifier schemas"
    )
    mcp_tools: list[str] = Field(
        default_factory=list, description="Available MCP tools on this server"
    )

    # System analytics and statistics
    ids_statistics: dict[str, Any] = Field(
        default_factory=dict, description="IDS usage and availability statistics"
    )
    usage_guidance: dict[str, Any] = Field(
        default_factory=dict,
        description="Usage guidance and getting started information",
    )


# ============================================================================
# ANALYSIS
# ============================================================================


class ConceptResult(ToolResult, IdsResult, PhysicsContext):
    """Concept explanation result."""

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "explain_concept"

    concept: str
    explanation: str
    detail_level: DetailLevel = DetailLevel.INTERMEDIATE
    related_topics: list[str] = Field(default_factory=list)
    concept_explanation: ConceptExplanation | None = None


class StructureResult(ToolResult, PhysicsContext):
    """IDS structure analysis result."""

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "analyze_ids_structure"

    ids_name: str
    description: str
    structure: dict[str, Any] = Field(
        default_factory=dict
    )  # Structure metrics (mixed types)
    sample_paths: list[str] = Field(default_factory=list)
    max_depth: int = 0
    analysis: StructureAnalysis | None = Field(
        default=None, description="Enhanced structure analysis"
    )


class IdentifierResult(ToolResult, PhysicsContext):
    """Identifier exploration result."""

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "explore_identifiers"

    scope: IdentifierScope = IdentifierScope.ALL
    schemas: list[dict[str, Any]] = Field(default_factory=list)
    paths: list[dict[str, Any]] = Field(default_factory=list)
    analytics: dict[str, Any] = Field(default_factory=dict)


class RelationshipResult(ToolResult, IdsResult, PhysicsContext):
    """Relationship exploration result."""

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "explore_relationships"

    path: str
    relationship_type: RelationshipType = RelationshipType.ALL
    max_depth: int = 2
    connections: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Categorized relationship paths with intra-IDS and cross-IDS separation",
    )

    # Remove nodes field entirely - connections contain all the path information

    # Relationship-specific analysis fields
    relationship_insights: dict[str, Any] = Field(
        default_factory=dict,
        description="Discovery summary, strength analysis, and semantic insights",
    )
    physics_analysis: dict[str, Any] = Field(
        default_factory=dict,
        description="Physics domain analysis, domain connections, and phenomena",
    )


# ============================================================================
# EXPORT
# ============================================================================


class IDSExport(ToolResult, ExportResult):
    """IDS export result."""

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "export_ids"

    ids_names: list[str]
    include_physics: bool = True
    include_relationships: bool = True


class DomainExport(ToolResult, ExportResult):
    """Physics domain export result."""

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "export_physics_domain"

    domain: str
    domain_info: dict[str, Any] | None = None
    include_cross_domain: bool = False
    max_paths: int = 10
