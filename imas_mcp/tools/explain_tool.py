"""
Explain tool implementation with service composition.

This module contains the explain_concept tool logic using service-based architecture
for physics integration, response building, and standardized metadata.
"""

import logging

from fastmcp import Context

from imas_mcp.models.constants import DetailLevel, SearchMode
from imas_mcp.models.error_models import ToolError
from imas_mcp.models.physics_models import ConceptExplanation
from imas_mcp.models.request_models import ExplainInput
from imas_mcp.models.result_models import ConceptResult
from imas_mcp.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    measure_performance,
    validate_input,
)
from imas_mcp.search.decorators.physics_hints import physics_hints
from imas_mcp.search.decorators.tool_hints import tool_hints

from .base import BaseTool

logger = logging.getLogger(__name__)


class ExplainTool(BaseTool):
    """Tool for explaining IMAS concepts with service composition."""

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "explain_concept"

    @cache_results(ttl=600, key_strategy="semantic")
    @validate_input(schema=ExplainInput)
    @measure_performance(include_metrics=True, slow_threshold=2.0)
    @handle_errors(fallback="concept_suggestions")
    @tool_hints(max_hints=3)
    @physics_hints()
    @mcp_tool(
        "Provide detailed explanations of fusion physics concepts and IMAS terminology"
    )
    async def explain_concept(
        self,
        concept: str,
        detail_level: DetailLevel = DetailLevel.INTERMEDIATE,
        ctx: Context | None = None,
    ) -> ConceptResult | ToolError:
        """
        Provide detailed explanations of fusion physics concepts and IMAS terminology.

        Educational tool that explains physics concepts, measurement techniques, and
        IMAS-specific terminology with appropriate technical depth. Includes related
        data paths and practical guidance for data access.

        Args:
            concept: Physics concept, measurement name, or IMAS term to explain
            detail_level: Explanation depth - basic, intermediate, or advanced
            ctx: FastMCP context

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

            # Get physics context from physics service
            physics_search_result = None
            concept_explanation = None
            try:
                physics_context_data = await self.physics.get_concept_context(
                    concept, detail_level.value
                )
                if physics_context_data:
                    # Create a ConceptExplanation object from physics service data
                    concept_explanation = ConceptExplanation(
                        concept=concept,
                        domain=physics_context_data["domain"],
                        description=physics_context_data["description"],
                        phenomena=physics_context_data["phenomena"],
                        typical_units=physics_context_data["typical_units"],
                        complexity_level=physics_context_data["complexity_level"],
                    )

                    # Also try to get physics search result
                    physics_enhancement = await self.physics.enhance_query(concept)
                    if physics_enhancement:
                        physics_search_result = physics_enhancement

                    logger.info(f"Physics context enhanced concept '{concept}'")
            except Exception as e:
                logger.warning(f"Physics enhancement failed for '{concept}': {e}")

            # Process search results for concept explanation
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

            # Enhanced explanation with physics integration
            base_explanation = (
                f"Analysis of '{concept}' within IMAS data dictionary context. "
                f"Found in {len(physics_domains)} physics domain(s): {', '.join(list(physics_domains)[:3])}. "
                f"Found {len(search_results[:8])} related data paths."
            )

            # Combine with physics explanation if available
            final_explanation = (
                f"{concept_explanation.description} {base_explanation}"
                if concept_explanation
                else base_explanation
            )

            # Build concept result with physics integration
            concept_result = ConceptResult(
                concept=concept,
                explanation=final_explanation,
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
                concept_explanation=concept_explanation,
                # QueryContext fields
                query=concept,
                search_mode=SearchMode.SEMANTIC,
                max_results=15,
                hits=search_results[:8],
                physics_domains=list(physics_domains),
                physics_context=physics_search_result,
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
