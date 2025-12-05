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
    BaseToolResult,
    WithHints,
    WithPhysics,
)
from imas_mcp.models.physics_models import ConceptExplanation
from imas_mcp.models.structure_models import StructureAnalysis
from imas_mcp.search.search_strategy import SearchHit

# ============================================================================
# BASE MODELS WITH METADATA
# ============================================================================


class ToolResult(BaseToolResult, ABC):
    """
    Base class for all tool results.

    Includes query tracking and standard metadata (version, timestamps).
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


class DeprecatedPathInfo(BaseModel):
    """Migration info for a deprecated path.

    Returned when fetch_imas_paths encounters a path that doesn't exist
    in the current DD version but has a known migration.
    """

    path: str = Field(description="The deprecated path that was requested")
    new_path: str | None = Field(
        default=None,
        description="The current path to use (None if path was removed, not renamed)",
    )
    deprecated_in: str = Field(description="DD version where path was deprecated")
    last_valid_version: str = Field(description="Last DD version where path was valid")
    new_path_excluded: bool = Field(
        default=False,
        description="True if new_path exists but is excluded from search index",
    )
    exclusion_reason: str | None = Field(
        default=None,
        description="Human-readable explanation of why new_path is excluded",
    )


class ExcludedPathInfo(BaseModel):
    """Information about a path that exists but is excluded from search index.

    Returned when check_imas_paths or fetch_imas_paths encounters a path
    that is valid in the DD but excluded from indexing.
    """

    path: str = Field(description="The path that was requested")
    reason_key: str = Field(
        description="Exclusion reason key (error_field, ggd, metadata)"
    )
    reason_description: str = Field(
        description="Human-readable explanation of why the path is excluded"
    )


class IdsResult(BaseModel):
    """Result containing IMAS data nodes.

    Used by tools that return complete, authoritative IMAS path data.
    """

    nodes: list[IdsNode] = Field(default_factory=list)

    @property
    def node_count(self) -> int:
        """Number of nodes in the result."""
        return len(self.nodes)


class IdsPathResult(WithPhysics, ToolResult, IdsResult):
    """Path retrieval result with physics aggregation.

    Includes migration info for deprecated paths that weren't found,
    and exclusion info for paths that are valid but not indexed.
    """

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "fetch_imas_paths"

    # Summary information
    summary: dict[str, Any] = Field(
        default_factory=dict, description="Summary of path retrieval operation"
    )

    # Migration info for deprecated paths
    deprecated_paths: list[DeprecatedPathInfo] = Field(
        default_factory=list,
        description="Migration info for paths not found (deprecated in current DD version)",
    )

    # Info for paths that exist but are excluded from search index
    excluded_paths: list[ExcludedPathInfo] = Field(
        default_factory=list,
        description="Info for paths that are valid but excluded from search index",
    )


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
    # Enhanced format fields
    raw_data: dict[str, Any] | None = Field(
        default=None, description="Raw data for enhanced format"
    )
    identifier_info: dict[str, Any] | None = Field(
        default=None, description="Identifier info for enhanced format"
    )
    # Domain export field
    measurement_type: str | None = Field(
        default=None, description="Classified measurement type"
    )


class IdsInfo(BaseModel):
    """Information about an exported IDS."""

    ids_name: str = Field(description="IDS name")
    description: str | None = Field(default=None, description="IDS description")
    total_paths: int = Field(default=0, description="Total number of paths in this IDS")
    paths: list[IdsPath] = Field(default_factory=list, description="Paths in this IDS")
    physics_domains: list[str] = Field(
        default_factory=list, description="Physics domains"
    )
    identifier_paths: list[IdsPath] = Field(
        default_factory=list, description="Identifier paths in this IDS"
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

    # For IDS exports - use Any since export can return error dicts
    ids_data: dict[str, Any] | None = Field(
        default=None, description="IDS export data (IdsInfo or error dict)"
    )
    cross_relationships: dict[str, Any] | None = Field(
        default=None, description="Cross-IDS relationships"
    )
    # Accept dict or ExportSummary for flexibility
    export_summary: dict[str, Any] | None = Field(
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


class SearchResult(WithHints, WithPhysics, ToolResult, SearchHits):
    """Search tool result with hints and physics aggregation."""

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "search_imas"

    # Search-specific fields
    search_mode: SearchMode = Field(
        default=SearchMode.AUTO, description="Search mode used"
    )


class OverviewResult(WithHints, WithPhysics, ToolResult, SearchHits):
    """Overview tool response with hints and physics aggregation."""

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


class ConceptResult(WithHints, WithPhysics, ToolResult, SearchHits):
    """Concept explanation result.

    Returns ranked search results related to the concept.
    """

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "explain_concept"

    concept: str
    explanation: str
    detail_level: DetailLevel = DetailLevel.INTERMEDIATE
    related_topics: list[str] = Field(default_factory=list)
    concept_explanation: ConceptExplanation | None = None


class StructureResult(WithHints, WithPhysics, ToolResult):
    """IDS structure analysis result with hints and physics aggregation."""

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


class IdentifierResult(WithHints, WithPhysics, ToolResult):
    """Identifier exploration result with hints and physics aggregation."""

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "explore_identifiers"

    scope: IdentifierScope = IdentifierScope.ALL
    schemas: list[dict[str, Any]] = Field(default_factory=list)
    paths: list[dict[str, Any]] = Field(default_factory=list)
    analytics: dict[str, Any] = Field(default_factory=dict)


class RelationshipResult(WithHints, WithPhysics, ToolResult):
    """Relationship exploration result with hints and physics aggregation.

    Does not inherit from IdsResult as it returns relationship metadata
    in the connections dict, not complete IdsNode objects.
    """

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


class IDSExport(WithHints, ToolResult, ExportResult):
    """IDS export result with hints."""

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "export_ids"

    ids_names: list[str]
    include_physics: bool = True
    include_relationships: bool = True


class DomainExport(WithHints, ToolResult, ExportResult):
    """Physics domain export result with hints."""

    @property
    def tool_name(self) -> str:
        """Name of the tool that generated this result."""
        return "export_physics_domain"

    domain: str
    domain_info: dict[str, Any] | None = None
    include_cross_domain: bool = False
    max_paths: int = 10


# ============================================================================
# LIST PATHS
# ============================================================================


class PathListQueryResult(BaseModel):
    """Result for a single IDS/prefix query in list_imas_paths."""

    query: str = Field(description="The IDS name or prefix queried")
    path_count: int = Field(description="Total number of paths found")
    truncated_to: int | None = Field(
        default=None, description="Number of paths shown (only present when truncated)"
    )
    paths: str | dict[str, Any] | list[str] | None = Field(
        default=None,
        description="Formatted path listing: str (yaml/json), list[str] (list), dict (dict), or None (count)",
    )
    error: str | None = Field(default=None, description="Error message if query failed")


class PathListResult(BaseModel):
    """Result from list_imas_paths tool with minimal path enumeration.

    Uses minimal BaseModel instead of ToolResult to avoid unnecessary search-related fields.
    """

    format: str = Field(
        description="Output format used (yaml, list, count, json, dict)"
    )
    results: list[PathListQueryResult] = Field(
        default_factory=list, description="Results for each queried IDS/prefix"
    )
    summary: dict[str, Any] = Field(
        default_factory=dict, description="Overall statistics across all queries"
    )
