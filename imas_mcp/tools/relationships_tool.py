"""
Relationships tool implementation.

This module contains the explore_relationships tool logic with decorators
for caching, validation, AI enhancement, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Optional, Set, Union
from fastmcp import Context

from imas_mcp.models.request_models import RelationshipsInput
from imas_mcp.models.constants import SearchMode, RelationshipType
from imas_mcp.models.result_models import RelationshipResult
from imas_mcp.models.error_models import ToolError
from imas_mcp.core.data_model import IdsNode, PhysicsContext

# Import only essential decorators
from imas_mcp.search.decorators import (
    cache_results,
    validate_input,
    measure_performance,
    handle_errors,
)

from .base import BaseTool
from imas_mcp.services.sampling import SamplingStrategy
from imas_mcp.services.tool_recommendations import RecommendationStrategy

logger = logging.getLogger(__name__)


def mcp_tool(description: str):
    """Decorator to mark methods as MCP tools with descriptions."""

    def decorator(func):
        func._mcp_tool = True
        func._mcp_description = description
        return func

    return decorator


class RelationshipsTool(BaseTool):
    """Tool for exploring relationships using service composition."""

    # Enable both services for comprehensive relationship analysis
    enable_sampling: bool = True
    enable_recommendations: bool = True

    # Use relationship-appropriate strategies
    sampling_strategy = SamplingStrategy.SMART
    recommendation_strategy = RecommendationStrategy.RELATIONSHIPS_BASED
    max_recommended_tools: int = 5

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
    @measure_performance(include_metrics=True, slow_threshold=2.5)
    @handle_errors(fallback="relationships_suggestions")
    @mcp_tool("Explore relationships between IMAS data paths")
    async def explore_relationships(
        self,
        path: str,
        relationship_type: RelationshipType = RelationshipType.ALL,
        max_depth: int = 2,
        ctx: Optional[Context] = None,
    ) -> Union[RelationshipResult, ToolError]:
        """
        Explore relationships between IMAS data paths using service composition.

        Uses service composition for business logic:
        - DocumentService: Validates paths and retrieves related documents
        - PhysicsService: Enhances relationships with physics context
        - ResponseService: Builds standardized Pydantic responses

        Advanced tool that discovers connections, physics concepts, and measurement
        relationships between different parts of the IMAS data dictionary.

        Args:
            path: Starting path (format: "ids_name/path" or just "ids_name")
            relationship_type: Type of relationships to explore
            max_depth: Maximum depth of relationship traversal (1-3, limited for performance)
            ctx: MCP context for AI enhancement

        Returns:
            RelationshipResult with relationship network and AI insights
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

            # Validate IDS exists using document service
            valid_ids, invalid_ids = await self.documents.validate_ids([ids_name])
            if not valid_ids:
                return self.documents.create_ids_not_found_error(
                    path, self.get_tool_name()
                )

            # Enhance query with physics context using service
            if specific_path:
                search_query = f"{ids_name} {specific_path} relationships"
            else:
                search_query = f"{ids_name} relationships physics concepts"

            physics_context = await self.physics.enhance_query(search_query)

            # Get relationship data through semantic search with improved config
            search_config = self.search_config.create_config(
                search_mode=SearchMode.SEMANTIC,
                max_results=min(15, max_depth * 8),  # Scale with depth
            )

            try:
                search_results = await self._search_service.search(
                    query=search_query, config=search_config
                )
            except Exception as e:
                return self._create_error_response(
                    f"Failed to search relationships: {e}", path
                )

            # Process search results for relationships
            related_paths = []
            seen_paths: Set[str] = set()

            for search_result in search_results:
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

                    if (
                        len(related_paths) >= max_depth * 5
                    ):  # Increase limit for better results
                        break

            # Prepare AI prompts and responses separately
            ai_prompt = {
                "relationship_analysis": self._build_relationship_guidance(
                    path, relationship_type, related_paths
                ),
            }

            ai_response = {
                "cross_ids_analysis": list(
                    set(p.path.split("/")[0] for p in related_paths if "/" in p.path)
                ),
                "physics_connections": [
                    p.path
                    for p in related_paths
                    if p.physics_context and p.physics_context.domain
                ],
            }

            # Add physics context if available
            if physics_context:
                ai_response["physics_context"] = physics_context
                logger.debug(f"Physics context added for relationship: {path}")

            # Build response using Pydantic
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
                nodes=related_paths[:8],  # Increase visible nodes
                physics_domains=[
                    p.physics_context.domain
                    for p in related_paths
                    if p.physics_context and p.physics_context.domain
                ],
                physics_context=physics_context,
                ai_response=ai_response,
                ai_prompt=ai_prompt,
            )

            # Apply post-processing services (sampling and recommendations)
            enhanced_response = await self.apply_services(
                result=response,
                query=search_query,
                path=path,
                relationship_type=relationship_type.value,
                tool_name=self.get_tool_name(),
                ctx=ctx,
            )

            # Add standard metadata and return properly typed response
            final_response = self.response.add_standard_metadata(
                enhanced_response, self.get_tool_name()
            )

            logger.info(f"Relationship exploration completed for path: {path}")
            # Ensure we return the correct type
            return (
                final_response
                if isinstance(final_response, RelationshipResult)
                else response
            )

        except Exception as e:
            logger.error(f"Relationship exploration failed: {e}")
            return self._create_error_response(
                f"Relationship exploration failed: {e}", path
            )

    def _build_relationship_guidance(
        self, path: str, relationship_type: RelationshipType, related_paths: list
    ) -> str:
        """Build comprehensive relationship guidance for AI enhancement."""
        return f"""IMAS Relationship Exploration: "{path}"

Found {len(related_paths)} related data paths with relationship type: {relationship_type.value}

Key relationship insights:
1. **Connected Data Paths**: Direct measurements and calculated quantities related to this path
2. **Physics Relationships**: How this data connects to underlying physics phenomena  
3. **Cross-IDS Connections**: Links between different IMAS data structures
4. **Measurement Dependencies**: What this data depends on or influences
5. **Data Flow Context**: How this fits into typical fusion data workflows
6. **Usage Patterns**: Common analysis patterns involving this data

Provide detailed analysis including:
- Physics significance of the identified relationships
- Recommended follow-up analyses or related data to explore
- Cross-IDS workflow patterns and data dependencies
- Validation considerations for related measurements"""
