"""
Explain tool implementation with service composition.

This module contains the explain_concept tool logic using service-based architecture
for physics integration, response building, and standardized metadata.
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp.core.data_model import IdsNode, PhysicsContext
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
from imas_mcp.search.decorators.sample import sample
from imas_mcp.search.decorators.tool_hints import tool_hints

from .base import BaseTool

logger = logging.getLogger(__name__)


class ExplainTool(BaseTool):
    """Tool for explaining IMAS concepts with service composition."""

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "explain_concept"

    def build_prompt(self, prompt_type: str, tool_context: dict[str, Any]) -> str:
        """Build concept explanation-specific AI prompts."""
        if prompt_type == "concept_analysis":
            # Standard context from sample decorator
            concept = tool_context.get("query", "")
            search_results = tool_context.get("results", [])
            tool_context.get("hit_count", 0)

            # Try to determine detail level from the query or use default
            detail_level = "intermediate"  # Default level

            return self._build_concept_analysis_prompt(
                concept, detail_level, search_results
            )
        elif prompt_type == "no_results":
            return self._build_no_results_prompt(tool_context)
        elif prompt_type == "concept_explanation_sampling":
            return self._build_explanation_sampling_prompt(tool_context)
        elif prompt_type == "related_topics_sampling":
            return self._build_topics_sampling_prompt(tool_context)
        elif prompt_type == "sample_documentation":
            return self._build_documentation_sampling_prompt(tool_context)
        return ""

    def system_prompt(self) -> str:
        """Get explain tool-specific system prompt."""
        return """You are an expert fusion physics educator and IMAS data specialist with deep knowledge in:

- Tokamak physics fundamentals and advanced plasma phenomena
- Plasma diagnostics, measurement techniques, and instrumentation principles
- IMAS data structures, conventions, and integration patterns
- Physics relationships between different measurement systems
- Educational communication across basic, intermediate, and advanced levels

Your role is to provide clear, comprehensive explanations that:
1. Start with fundamental physics principles and build to specific IMAS contexts
2. Explain measurement techniques and their underlying physics
3. Connect concepts to practical data access and interpretation
4. Provide appropriate technical depth for the requested detail level
5. Suggest related concepts and cross-references for deeper understanding
6. Include practical guidance for working with the data in research contexts

Adapt your explanations to the specified detail level:
- Basic: Focus on core concepts and practical understanding
- Intermediate: Include technical details and measurement contexts
- Advanced: Provide comprehensive physics derivations and implementation details

Always connect abstract concepts to concrete IMAS data paths and real-world measurements."""

    def build_sample_tasks(self, tool_result) -> list[dict[str, Any]]:
        """Build sampling tasks specific to ConceptResult."""
        from imas_mcp.models.result_models import ConceptResult

        tasks = super().build_sample_tasks(tool_result)  # Get base tasks

        if isinstance(tool_result, ConceptResult):
            # Sample concept-specific fields
            if tool_result.explanation:
                tasks.append(
                    {
                        "field": "explanation",
                        "prompt_type": "concept_explanation_sampling",
                        "context": {
                            "concept": tool_result.concept,
                            "current_explanation": tool_result.explanation,
                            "detail_level": tool_result.detail_level.value,
                            "nodes": tool_result.nodes[:3] if tool_result.nodes else [],
                        },
                    }
                )

            tasks.append(
                {
                    "field": "related_topics",
                    "prompt_type": "related_topics_sampling",
                    "context": {
                        "concept": tool_result.concept,
                        "current_topics": tool_result.related_topics,
                        "physics_domains": getattr(tool_result, "physics_domains", []),
                        "explanation_summary": tool_result.explanation[:200]
                        if tool_result.explanation
                        else "",
                    },
                }
            )

        return tasks

    def _apply_explanation_sampling(self, tool_result, sampled_content: str) -> None:
        """Apply custom sampling for explanation field."""
        from imas_mcp.models.result_models import ConceptResult

        if isinstance(tool_result, ConceptResult):
            tool_result.explanation = sampled_content

    def _apply_related_topics_sampling(self, tool_result, sampled_content: str) -> None:
        """Apply custom sampling for related_topics field."""
        from imas_mcp.models.result_models import ConceptResult

        if isinstance(tool_result, ConceptResult):
            # Parse response into topics list
            topics = [t.strip() for t in sampled_content.split("\n") if t.strip()]
            tool_result.related_topics = topics[:10]  # Limit to reasonable number

    def _build_no_results_prompt(self, tool_context: dict[str, Any]) -> str:
        """Build prompt for when no concept results are found."""
        concept = tool_context.get("query", "")

        return f"""Concept Explanation Request: "{concept}"

No results were found for this concept in the IMAS data dictionary.

Please provide:
1. Suggestions for alternative concept terms or physics terminology
2. Possible related IMAS concepts or fusion physics domains
3. Common physics contexts where this concept might appear
4. Educational resources for understanding this concept
5. Recommended follow-up queries for exploration"""

    def _extract_identifier_info(self, document) -> dict[str, Any]:
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
        self, concept: str, detail_level: str, search_results: list[Any]
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
            # Handle both dict and object results
            if hasattr(result, "path"):
                path = result.path
                doc = (
                    result.documentation[:100]
                    if hasattr(result, "documentation")
                    else ""
                )
            elif isinstance(result, dict):
                path = result.get("path", str(result))
                doc = result.get("documentation", "")[:100]
            else:
                path = str(result)[:100]
                doc = ""

            prompt += f"{i}. {path}"
            if doc:
                prompt += f" - {doc}"
            prompt += "\n"

        return prompt

    @cache_results(ttl=600, key_strategy="semantic")
    @validate_input(schema=ExplainInput)
    @measure_performance(include_metrics=True, slow_threshold=2.0)
    @handle_errors(fallback="concept_suggestions")
    @tool_hints(max_hints=3)
    @physics_hints()
    @sample(temperature=0.3, max_tokens=800)
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

                # Use physics domain if available
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

            # Enhanced explanation with physics integration
            base_explanation = (
                f"Analysis of '{concept}' within IMAS data dictionary context. "
                f"Found in {len(physics_domains)} physics domain(s): {', '.join(list(physics_domains)[:3])}. "
                f"Found {len(related_paths_data)} related data paths."
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
                concept_explanation=concept_explanation,  # Now properly populated
                # QueryContext fields
                query=concept,
                search_mode=SearchMode.SEMANTIC,
                max_results=15,
                nodes=related_paths_data,
                physics_domains=list(physics_domains),
                physics_context=physics_search_result,  # Use PhysicsSearchResult
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

    def _build_explanation_sampling_prompt(self, context: dict[str, Any]) -> str:
        """Build prompt for sampling concept explanation."""
        concept = context.get("concept", "")
        current_explanation = context.get("current_explanation", "")
        detail_level = context.get("detail_level", "intermediate")
        nodes = context.get("nodes", [])

        prompt = f"""Current explanation for "{concept}" ({detail_level} level):
{current_explanation}

Sample and improve this explanation by:
1. Adding more physics depth appropriate for {detail_level} level
2. Including relevant IMAS data paths and measurement contexts
3. Connecting to broader fusion physics concepts
4. Adding practical guidance for data access and interpretation
5. Ensuring accuracy and educational value

Focus on clarity, accuracy, and practical utility for researchers working with IMAS data."""

        if nodes:
            prompt += f"\n\nRelevant IMAS paths found: {[node.get('path', '') for node in nodes[:3]]}"

        return prompt

    def _build_topics_sampling_prompt(self, context: dict[str, Any]) -> str:
        """Build prompt for sampling related topics."""
        concept = context.get("concept", "")
        current_topics = context.get("current_topics", [])
        physics_domains = context.get("physics_domains", [])

        prompt = f"""Current related topics for "{concept}":
{chr(10).join([f"- {topic}" for topic in current_topics])}

Sample and improve this list of related topics by:
1. Adding relevant fusion physics concepts and phenomena
2. Including related IMAS measurement techniques and diagnostics
3. Suggesting complementary analysis tools and methods
4. Connecting to broader tokamak physics domains
5. Providing educational learning pathways

Return a newline-separated list of topics (one per line)."""

        if physics_domains:
            prompt += f"\n\nPhysics domains context: {physics_domains}"

        return prompt

    def _build_documentation_sampling_prompt(self, context: dict[str, Any]) -> str:
        """Build prompt for sampling node documentation."""
        nodes = context.get("nodes", [])
        query = context.get("query", "")

        if not nodes:
            return ""

        prompt = f"""Sample and improve documentation for IMAS data paths related to "{query}":

Current paths and documentation:"""

        for node in nodes[:3]:
            path = node.get("path", "")
            doc = node.get("documentation", "")[:200]
            prompt += f"\n\nPath: {path}\nCurrent doc: {doc}"

        prompt += """

Provide enhanced documentation that:
1. Clarifies the physics meaning and measurement context
2. Explains practical usage in fusion research
3. Connects to related IMAS data structures
4. Includes typical units and value ranges
5. Suggests analysis workflows and tools

Focus on practical utility for researchers accessing this data."""

        return prompt
