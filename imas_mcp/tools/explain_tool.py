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
from imas_mcp.core.data_model import IdsNode, PhysicsContext

from imas_mcp.search.decorators import (
    cache_results,
    validate_input,
    measure_performance,
    handle_errors,
    sample,
    tool_hints,
    physics_hints,
    mcp_tool,
)

from .base import BaseTool

logger = logging.getLogger(__name__)


class ExplainTool(BaseTool):
    """Tool for explaining IMAS concepts with service composition."""

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
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
            prompt += f"{i}. {result.path} - {result.documentation[:100]}\n"

        return prompt

    @cache_results(ttl=600, key_strategy="semantic")
    @validate_input(schema=ExplainInput)
    @measure_performance(include_metrics=True, slow_threshold=2.0)
    @handle_errors(fallback="concept_suggestions")
    @tool_hints(max_hints=3)
    @physics_hints()
    @sample(temperature=0.1, max_tokens=700)
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
            # Execute search using base tool method
            search_result = await self.execute_search(
                query=concept,
                search_mode=SearchMode.SEMANTIC,
                max_results=15,
            )
            search_results = search_result.hits

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
                )

            # Process search results for concept explanation
            related_paths_data = []
            physics_domains = set()
            identifier_schemas = []

            for search_result in search_results[:8]:
                # Collect physics domains
                if search_result.physics_domain:
                    physics_domains.add(search_result.physics_domain)

                # Check for identifier schemas (simplified without document access)
                if "identifier" in search_result.path.lower():
                    identifier_schemas.append(
                        {
                            "path": search_result.path,
                            "schema_info": {"has_identifier": True},
                        }
                    )

                # Build PhysicsContext if available
                local_physics_context = None
                if search_result.physics_domain:
                    local_physics_context = PhysicsContext(
                        domain=search_result.physics_domain,
                        phenomena=[],
                        typical_values={},
                    )

                related_paths_data.append(
                    IdsNode(
                        path=search_result.path,
                        documentation=search_result.documentation[:150],
                        units=search_result.units,
                        data_type=search_result.data_type,
                        physics_context=local_physics_context,
                    )
                )

            # Build concept result
            concept_result = ConceptResult(
                concept=concept,
                explanation=f"Analysis of '{concept}' within IMAS data dictionary context. Found in {len(physics_domains)} physics domain(s): {', '.join(list(physics_domains)[:3])}. Found {len(related_paths_data)} related data paths.",
                detail_level=detail_level,
                related_topics=[
                    f"Explore {search_results[0].ids_name} for detailed data"
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
                nodes=related_paths_data,
                physics_domains=list(physics_domains),
            )

            logger.info(f"Concept explanation completed for {concept}")
            return concept_result

        except Exception as e:
            self.logger.error(f"Concept explanation failed: {e}")
            return (
                self.documents.create_ids_not_found_error(concept, self.tool_name)
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
