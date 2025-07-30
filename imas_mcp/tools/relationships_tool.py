"""
Relationships tool implementation.

This module contains the explore_relationships tool logic with decorators
for caching, validation, AI enhancement, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Optional, Set, Union
from fastmcp import Context

from imas_mcp.search.search_strategy import SearchConfig
from imas_mcp.models.request_models import RelationshipsInput
from imas_mcp.models.constants import SearchMode, RelationshipType
from imas_mcp.models.response_models import RelationshipResult, ErrorResponse
from imas_mcp.core.data_model import IdsNode, PhysicsContext

# Import all decorators
from imas_mcp.search.decorators import (
    cache_results,
    validate_input,
    sample,
    recommend_tools,
    measure_performance,
    handle_errors,
)

from .base import BaseTool

logger = logging.getLogger(__name__)


def mcp_tool(description: str):
    """Decorator to mark methods as MCP tools with descriptions."""

    def decorator(func):
        func._mcp_tool = True
        func._mcp_description = description
        return func

    return decorator


class RelationshipsTool(BaseTool):
    """Tool for exploring relationships."""

    def get_tool_name(self) -> str:
        return "explore_relationships"

    def _build_relationships_sample_prompt(
        self, path: str, relationship_type: str = "all"
    ) -> str:
        """Build sampling prompt for relationship exploration."""
        return f"""IMAS Relationship Exploration Request: "{path}"

Please provide comprehensive relationship analysis that includes:

1. **Connected Data Paths**: Related measurements and calculated quantities
2. **Physics Relationships**: How this data connects to physics phenomena
3. **Cross-IDS Connections**: Links between different data structures
4. **Measurement Dependencies**: What this data depends on or influences
5. **Data Flow Context**: How this fits into typical fusion data workflows
6. **Usage Patterns**: Common analysis patterns involving this data
7. **Recommendations**: Suggested follow-up analyses or related data to explore

Relationship type focus: {relationship_type}

Provide actionable insights for researchers exploring data relationships.
"""

    @cache_results(ttl=600, key_strategy="path_based")  # Cache relationships
    @validate_input(schema=RelationshipsInput)
    @sample(temperature=0.3, max_tokens=800)  # Balanced creativity for relationships
    @recommend_tools(strategy="relationships_based", max_tools=4)
    @measure_performance(include_metrics=True, slow_threshold=2.5)
    @handle_errors(fallback="relationships_suggestions")
    @mcp_tool("Explore relationships between IMAS data paths")
    async def explore_relationships(
        self,
        path: str,
        relationship_type: RelationshipType = RelationshipType.ALL,
        max_depth: int = 2,
        ctx: Optional[Context] = None,
    ) -> Union[RelationshipResult, ErrorResponse]:
        """
        Explore relationships between IMAS data paths using the rich relationship data.

        Advanced tool that discovers connections, physics concepts, and measurement
        relationships between different parts of the IMAS data dictionary.

        Args:
            path: Starting path (format: "ids_name/path" or just "ids_name")
            relationship_type: Type of relationships to explore
            max_depth: Maximum depth of relationship traversal (1-3, limited for performance)
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with relationship network and AI insights
        """
        try:
            # Validate and limit max_depth for performance
            max_depth = min(max_depth, 3)  # Hard limit to prevent excessive traversal
            if max_depth < 1:
                max_depth = 1

            # Parse the path to extract IDS name
            if "/" in path:
                ids_name = path.split("/")[0]
                specific_path = path
            else:
                ids_name = path
                specific_path = None

            # Validate IDS exists
            available_ids = self.document_store.get_available_ids()
            if ids_name not in available_ids:
                return ErrorResponse(
                    error=f"IDS '{ids_name}' not found",
                    suggestions=[f"Try: {ids}" for ids in available_ids[:5]],
                    context={
                        "available_ids": available_ids[:10],
                        "path": path,
                        "tool": "explore_relationships",
                    },
                )

            # Get relationship data through semantic search
            try:
                # Use semantic search to find related concepts
                if specific_path:
                    search_query = f"{ids_name} {specific_path} relationships"
                else:
                    search_query = f"{ids_name} relationships physics concepts"

                search_results_dict = await self._search_service.search(
                    query=search_query,
                    config=SearchConfig(
                        search_mode=SearchMode.SEMANTIC,
                        max_results=min(10, max_depth * 8),  # Scale with depth
                    ),
                )
            except Exception as e:
                return ErrorResponse(
                    error=f"Failed to search relationships: {e}",
                    suggestions=[
                        "Check path format and accessibility",
                        "Verify search service availability",
                    ],
                    context={
                        "path": path,
                        "tool": "explore_relationships",
                        "operation": "relationship_search",
                    },
                )

            # Process search results for relationships
            related_paths = []
            seen_paths: Set[str] = set()

            for search_result in search_results_dict:
                result_path = search_result.document.metadata.path_name
                if result_path not in seen_paths and result_path != path:
                    seen_paths.add(result_path)

                    # Build physics context if available
                    local_physics_context = None
                    if search_result.document.metadata.physics_domain:
                        local_physics_context = PhysicsContext(
                            domain=search_result.document.metadata.physics_domain,
                            phenomena=[],
                            typical_values={},
                        )

                    related_paths.append(
                        IdsNode(
                            path=result_path,
                            documentation=search_result.document.documentation[:200]
                            + "..."
                            if len(search_result.document.documentation) > 200
                            else search_result.document.documentation,
                            units=search_result.document.units.unit_str
                            if search_result.document.units
                            else "",
                            data_type=search_result.document.metadata.data_type or "",
                            physics_context=local_physics_context,
                        )
                    )

                    if len(related_paths) >= max_depth * 3:  # Limit results
                        break

            # Build final response using Pydantic
            response = RelationshipResult(
                path=path,
                relationship_type=relationship_type,
                max_depth=max_depth,
                connections={
                    "total_relationships": [p.path for p in related_paths],
                    "physics_connections": [
                        p.path
                        for p in related_paths
                        if p.physics_context and p.physics_context.domain
                    ],
                    "cross_ids_connections": list(
                        set(
                            p.path.split("/")[0] for p in related_paths if "/" in p.path
                        )
                    ),
                },
                nodes=related_paths[:5],
                physics_domains=[
                    p.physics_context.domain
                    for p in related_paths
                    if p.physics_context and p.physics_context.domain
                ],
                physics_context=None,
            )

            return response

        except Exception as e:
            logger.error(f"Relationship exploration failed: {e}")
            return ErrorResponse(
                error=str(e),
                suggestions=[
                    "Check path format (ids_name/path or just ids_name)",
                    "Verify IDS exists in data dictionary",
                    "Try search_imas() first to find valid paths",
                ],
                context={
                    "path": path,
                    "tool": "explore_relationships",
                    "operation": "relationship_exploration",
                },
            )
