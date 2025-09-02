"""Response building service for consistent Pydantic model construction."""

import importlib.metadata
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel

from imas_mcp.models.constants import (
    DetailLevel,
    IdentifierScope,
    RelationshipType,
    SearchMode,
)
from imas_mcp.models.result_models import (
    ConceptResult,
    DomainExport,
    ExportData,
    IdentifierResult,
    IDSExport,
    OverviewResult,
    RelationshipResult,
    SearchResult,
    StructureResult,
)
from imas_mcp.search.search_strategy import SearchMatch

from .base import BaseService

try:
    VERSION = importlib.metadata.version("imas-mcp")
except importlib.metadata.PackageNotFoundError:
    VERSION = "development"

T = TypeVar("T", bound=BaseModel)


class ResponseService(BaseService):
    """Service for building standardized responses across all tool types."""

    def build_search_response(
        self,
        results: list[SearchMatch],
        query: str,
        search_mode: SearchMode,
        ids_filter: list[str] | None = None,
        max_results: int | None = None,
        ai_response: dict[str, Any] | None = None,
        ai_prompt: dict[str, str] | None = None,
        physics_context: Any | None = None,
        physics_domains: list[str] | None = None,
    ) -> SearchResult:
        """Build SearchResult from search results with complete context."""
        # Convert SearchMatch objects to SearchHit for API response
        hits = [result.to_hit() for result in results]

        return SearchResult(
            hits=hits,
            search_mode=search_mode,
            query=query,
            ids_filter=ids_filter,
            max_results=max_results,
            ai_response=ai_response or {},
            ai_prompt=ai_prompt or {},
            physics_context=physics_context,
            physics_domains=physics_domains or [],
        )

    def build_concept_response(
        self,
        concept: str,
        explanation: str,
        detail_level: DetailLevel,
        related_topics: list[str],
        nodes: list[Any],
        physics_domains: list[str],
        query: str,
        ai_prompt: dict[str, str] | None = None,
        ai_response: dict[str, Any] | None = None,
        physics_context: Any | None = None,
        concept_explanation: Any | None = None,
    ) -> ConceptResult:
        """Build ConceptResult for concept explanations."""
        return ConceptResult(
            concept=concept,
            explanation=explanation,
            detail_level=detail_level,
            related_topics=related_topics,
            nodes=nodes,
            physics_domains=physics_domains,
            query=query,
            search_mode=SearchMode.SEMANTIC,
            max_results=15,
            ids_filter=None,
            ai_prompt=ai_prompt or {},
            ai_response=ai_response or {},
            physics_context=physics_context,
            concept_explanation=concept_explanation,
        )

    def build_overview_response(
        self,
        content: str,
        available_ids: list[str],
        hits: list[Any],
        query: str | None = None,
        ai_prompt: dict[str, str] | None = None,
        ai_response: dict[str, Any] | None = None,
        physics_context: Any | None = None,
        physics_domains: list[str] | None = None,
        ids_statistics: dict[str, Any] | None = None,
        usage_guidance: dict[str, Any] | None = None,
    ) -> OverviewResult:
        """Build OverviewResult for system overviews."""
        return OverviewResult(
            content=content,
            available_ids=available_ids,
            hits=hits,
            query=query or "",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
            ai_prompt=ai_prompt or {},
            ai_response=ai_response or {},
            physics_context=physics_context,
            physics_domains=physics_domains or [],
            ids_statistics=ids_statistics or {},
            usage_guidance=usage_guidance or {},
        )

    def build_structure_response(
        self,
        ids_name: str,
        description: str,
        structure: dict[str, int],
        sample_paths: list[str],
        max_depth: int,
        tool_name: str,
        ai_prompt: dict[str, str] | None = None,
        ai_response: dict[str, Any] | None = None,
        physics_context: Any | None = None,
    ) -> StructureResult:
        """Build StructureResult for IDS structure analysis."""
        return StructureResult(
            ids_name=ids_name,
            description=description,
            structure=structure,
            sample_paths=sample_paths,
            max_depth=max_depth,
            tool_name=tool_name,
            processing_timestamp=datetime.now(UTC).isoformat(),
            version=VERSION,
            ai_prompt=ai_prompt or {},
            ai_response=ai_response or {},
            physics_context=physics_context,
        )

    def build_identifier_response(
        self,
        scope: IdentifierScope,
        schemas: list[dict[str, Any]],
        paths: list[dict[str, Any]],
        analytics: dict[str, Any],
        tool_name: str,
        query: str | None = None,
        ai_prompt: dict[str, str] | None = None,
        ai_response: dict[str, Any] | None = None,
    ) -> IdentifierResult:
        """Build IdentifierResult for identifier exploration."""
        return IdentifierResult(
            scope=scope,
            schemas=schemas,
            paths=paths,
            analytics=analytics,
            tool_name=tool_name,
            processing_timestamp=datetime.now(UTC).isoformat(),
            version=VERSION,
            query=query or "",
            search_mode=SearchMode.AUTO,
            max_results=None,
            ids_filter=None,
            ai_prompt=ai_prompt or {},
            ai_response=ai_response or {},
        )

    def build_relationship_response(
        self,
        path: str,
        relationship_type: RelationshipType,
        max_depth: int,
        connections: dict[str, list[str]],
        nodes: list[Any],
        physics_domains: list[str],
        query: str,
        ai_prompt: dict[str, str] | None = None,
        ai_response: dict[str, Any] | None = None,
        physics_context: Any | None = None,
    ) -> RelationshipResult:
        """Build RelationshipResult for relationship exploration."""
        return RelationshipResult(
            path=path,
            relationship_type=relationship_type,
            max_depth=max_depth,
            connections=connections,
            nodes=nodes,
            physics_domains=physics_domains,
            query=query,
            search_mode=SearchMode.SEMANTIC,
            max_results=None,
            ids_filter=None,
            ai_prompt=ai_prompt or {},
            ai_response=ai_response or {},
            physics_context=physics_context,
        )

    def build_export_response(
        self,
        ids_names: list[str],
        include_physics: bool,
        include_relationships: bool,
        export_data: ExportData,
        metadata: dict[str, Any],
        tool_name: str,
        ai_response: dict[str, Any] | None = None,
        ai_prompt: dict[str, str] | None = None,
    ) -> IDSExport:
        """Build IDSExport for IDS exports."""
        return IDSExport(
            ids_names=ids_names,
            include_physics=include_physics,
            include_relationships=include_relationships,
            data=export_data,
            metadata=metadata,
            tool_name=tool_name,
            processing_timestamp=datetime.now(UTC).isoformat(),
            version=VERSION,
            ai_response=ai_response or {},
            ai_prompt=ai_prompt or {},
        )

    def build_domain_export_response(
        self,
        domain: str,
        domain_info: dict[str, Any],
        include_cross_domain: bool,
        max_paths: int,
        export_data: ExportData,
        metadata: dict[str, Any],
        tool_name: str,
        ai_response: dict[str, Any] | None = None,
        ai_prompt: dict[str, str] | None = None,
    ) -> DomainExport:
        """Build DomainExport for physics domain exports."""
        return DomainExport(
            domain=domain,
            domain_info=domain_info,
            include_cross_domain=include_cross_domain,
            max_paths=max_paths,
            data=export_data,
            metadata=metadata,
            tool_name=tool_name,
            processing_timestamp=datetime.now(UTC).isoformat(),
            version=VERSION,
            ai_response=ai_response or {},
            ai_prompt=ai_prompt or {},
        )

    def add_standard_metadata(self, response: T, tool_name: str) -> T:
        """Add standard metadata to any response."""
        if hasattr(response, "metadata"):
            metadata = getattr(response, "metadata", {})
            if metadata is None:
                metadata = {}
            metadata.update(
                {
                    "tool": tool_name,
                    "processing_timestamp": datetime.now(UTC).isoformat(),
                    "version": VERSION,
                }
            )
            # Use setattr to update the metadata field
            response.metadata = metadata
        return response
