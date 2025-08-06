"""
Explain tool implementation with service composition.

This module contains the explain_concept tool logic using service-based architecture
for physics integration, response building, and standardized metadata.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from fastmcp import Context

from imas_mcp.models.request_models import ExplainInput
from imas_mcp.models.constants import DetailLevel, SearchMode
from imas_mcp.models.result_models import ConceptResult
from imas_mcp.models.error_models import ToolError
from imas_mcp.models.context_models import QueryContext
from imas_mcp.core.data_model import IdsNode, PhysicsContext
from imas_mcp.services.sampling import SamplingStrategy
from imas_mcp.services.tool_recommendations import RecommendationStrategy

from imas_mcp.search.decorators import (
    cache_results,
    validate_input,
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


class ExplainTool(BaseTool):
    """Tool for explaining IMAS concepts with service composition."""

    # Enable services for concept-based tool
    enable_sampling: bool = True
    enable_recommendations: bool = True
    sampling_strategy = SamplingStrategy.SMART
    recommendation_strategy = RecommendationStrategy.CONCEPT_BASED
    max_recommended_tools: int = 4

    def get_tool_name(self) -> str:
        return "explain_concept"

    def _extract_identifier_info(self, document) -> Dict[str, Any]:
        """Extract identifier information from document."""
        # Simplified implementation - can be enhanced based on document structure
        has_identifier = document.raw_data.get("identifier_schema") is not None
        return {
            "has_identifier": has_identifier,
            "schema_type": document.raw_data.get("identifier_schema", {}).get("type")
            if has_identifier
            else None,
        }

    def _build_concept_analysis_prompt(
        self, concept: str, detail_level: str, search_results: List[Any]
    ) -> str:
        """Build AI analysis prompt for concept explanation."""
        prompt = f"""IMAS Concept Explanation: "{concept}"

Detail Level: {detail_level}

Found {len(search_results)} related IMAS paths. Please provide a comprehensive explanation covering:

1. **Physics Context**: What this concept means in fusion physics and tokamak operations
2. **IMAS Integration**: How this concept is represented in IMAS data structures  
3. **Measurement Context**: How this quantity is typically measured or calculated
4. **Data Access**: Practical guidance for accessing related data in IMAS
5. **Related Concepts**: Cross-references to related physics quantities and IMAS paths

Top related paths found:
"""

        for i, result in enumerate(search_results[:5], 1):
            prompt += f"{i}. {result.document.metadata.path_name} - {result.document.documentation[:100]}\n"

        return prompt

    @cache_results(ttl=600, key_strategy="semantic")  # Longer cache for explanations
    @validate_input(schema=ExplainInput)
    @measure_performance(include_metrics=True, slow_threshold=2.0)
    @handle_errors(fallback="concept_suggestions")
    @mcp_tool("Explain IMAS concepts with physics context")
    async def explain_concept(
        self,
        concept: str,
        detail_level: DetailLevel = DetailLevel.INTERMEDIATE,
        ctx: Optional[Context] = None,
    ) -> Union[ConceptResult, ToolError]:
        """
        Explain IMAS concepts with physics context using clean service orchestration.

        Args:
            concept: The concept to explain (physics concept, IMAS path, or general term)
            detail_level: Level of detail (basic, intermediate, advanced)
            ctx: MCP context for AI enhancement

        Returns:
            ConceptResult with explanation, physics context, and related information
        """
        try:
            # Create clean query context for concept explanation
            query_context = QueryContext(
                query=concept,
                search_mode=SearchMode.SEMANTIC,
                max_results=15,
            )

            # Use clean operation context
            async with self.operation_context(
                "explain", query_context
            ) as operation_ctx:
                # Execute search using orchestrator
                search_results = await self._orchestrator.search(operation_ctx)

                if not search_results:
                    return ConceptResult(
                        concept=concept,
                        detail_level=detail_level,
                        explanation="No information found for concept",
                        related_topics=[
                            "Try alternative terminology",
                            "Check concept spelling",
                            "Use search_imas() to explore related terms",
                        ],
                        query=concept,
                        search_mode=SearchMode.SEMANTIC,
                        max_results=15,
                        ids_filter=None,
                        ai_prompt={},
                        ai_response={
                            "sources_analyzed": 0,
                            "identifier_analysis": [],
                            "error": "No matching data found",
                        },
                        physics_context=None,
                        nodes=[],
                        physics_domains=[],
                        concept_explanation=None,
                    )

                # Process search results for concept explanation
                related_paths_data = []
                physics_domains = set()
                identifier_schemas = []

                for search_result in search_results[:8]:
                    # Collect physics domains
                    if search_result.document.metadata.physics_domain:
                        physics_domains.add(
                            search_result.document.metadata.physics_domain
                        )

                    # Check for identifier schemas
                    identifier_info = self._extract_identifier_info(
                        search_result.document
                    )
                    if identifier_info["has_identifier"]:
                        identifier_schemas.append(
                            {
                                "path": search_result.document.metadata.path_name,
                                "schema_info": identifier_info,
                            }
                        )

                    # Build PhysicsContext if available
                    local_physics_context = None
                    if search_result.document.metadata.physics_domain:
                        local_physics_context = PhysicsContext(
                            domain=search_result.document.metadata.physics_domain,
                            phenomena=[],
                            typical_values={},
                        )

                    related_paths_data.append(
                        IdsNode(
                            path=search_result.document.metadata.path_name,
                            documentation=search_result.document.documentation[:150],
                            units=search_result.document.units.unit_str
                            if search_result.document.units
                            else None,
                            data_type=search_result.document.metadata.data_type,
                            physics_context=local_physics_context,
                        )
                    )

                # Generate AI prompts for concept explanation
                ai_prompts = self._orchestrator.generate_ai_prompts(operation_ctx)

                # Build AI insights for sampling service
                ai_insights = {
                    "sources_analyzed": len(search_results),
                    "identifier_analysis": identifier_schemas,
                    "physics_enhancement": bool(
                        operation_ctx.ai_prompts.get("physics_context")
                    ),
                    "analysis_prompt": self._build_concept_analysis_prompt(
                        concept, detail_level.value, search_results
                    ),
                }

                # Build concept result
                concept_result = ConceptResult(
                    concept=concept,
                    explanation=f"Analysis of '{concept}' within IMAS data dictionary context. Found in {len(physics_domains)} physics domain(s): {', '.join(list(physics_domains)[:3])}. Found {len(related_paths_data)} related data paths.",
                    detail_level=detail_level,
                    related_topics=[
                        f"Explore {search_results[0].document.metadata.ids_name} for detailed data"
                        if search_results
                        else "Use search_imas() for more specific queries",
                        f"Investigate {list(physics_domains)[0]} domain connections"
                        if physics_domains
                        else "Consider broader physics concepts",
                        "Use analyze_ids_structure() for structural details"
                        if search_results
                        else "Try related terminology",
                    ],
                    concept_explanation=None,  # Will be populated by AI sampling service
                    # QueryContext fields
                    query=concept,
                    search_mode=SearchMode.SEMANTIC,
                    max_results=15,
                    ids_filter=None,
                    # AIContext fields
                    ai_prompt=ai_prompts,
                    ai_response=ai_insights,
                    # PhysicsContext fields
                    physics_context=None,  # Will be enhanced by service processing
                    # IdsResult fields
                    nodes=related_paths_data,
                    physics_domains=list(physics_domains),
                )

                # Apply metadata and return
                return self._orchestrator.add_metadata(concept_result)

        except Exception as e:
            self.logger.error(f"Concept explanation failed: {e}")
            return (
                self.documents.create_ids_not_found_error(concept, self.get_tool_name())
                if "not found" in str(e).lower()
                else ToolError(
                    error=str(e),
                    suggestions=[
                        "Try simpler concept terms",
                        "Check concept spelling",
                        "Use search_imas() to explore available data",
                    ],
                    context={
                        "concept": concept,
                        "detail_level": detail_level.value,
                        "tool": "explain_concept",
                        "operation": "concept_explanation",
                    },
                )
            )
