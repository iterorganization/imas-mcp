"""
Explain tool implementation.

This module contains the explain_concept tool logic with decorators
for caching, validation, AI sampling, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Dict, Any, Optional, List, Set, Union
from fastmcp import Context

from imas_mcp.search.search_strategy import SearchConfig, SearchResult
from imas_mcp.search.services.search_service import SearchService
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.engines.semantic_engine import SemanticSearchEngine
from imas_mcp.search.engines.lexical_engine import LexicalSearchEngine
from imas_mcp.search.engines.hybrid_engine import HybridSearchEngine
from imas_mcp.models.request_models import ExplainInputSchema
from imas_mcp.models.constants import SearchMode, DetailLevel
from imas_mcp.models.response_models import ConceptResult, ErrorResponse
from imas_mcp.core.data_model import DataPath, PhysicsContext

# Import physics integration for enhanced search
from imas_mcp.physics_integration import physics_search

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


class ExplainTool(BaseTool):
    """Tool for explaining IMAS concepts."""

    def __init__(self, document_store: DocumentStore):
        """Initialize the explain tool with document store."""
        super().__init__()
        self.document_store = document_store
        self._search_service = self._create_search_service()

    def _create_search_service(self) -> SearchService:
        """Create search service with appropriate engines."""
        # Create engines for each mode
        engines = {}
        for mode in [SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID]:
            config = SearchConfig(
                mode=mode, max_results=100
            )  # Service will limit based on request
            engine = self._create_engine(mode.value, config)
            engines[mode] = engine

        return SearchService(engines)

    def _create_engine(self, engine_type: str, config: SearchConfig):
        """Create a search engine of the specified type."""
        engine_map = {
            "semantic": SemanticSearchEngine,
            "lexical": LexicalSearchEngine,
            "hybrid": HybridSearchEngine,
        }

        if engine_type not in engine_map:
            raise ValueError(f"Unknown engine type: {engine_type}")

        engine_class = engine_map[engine_type]
        return engine_class(config)

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

    def _build_concept_sample_prompt(
        self, concept: str, detail_level: str, search_results: List[SearchResult]
    ) -> str:
        """Build sampling prompt for concept explanation."""
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
    @validate_input(schema=ExplainInputSchema)
    @sample(
        temperature=0.2, max_tokens=1000
    )  # Lower temperature for factual explanations
    @recommend_tools(strategy="concept_based", max_tools=4)
    @measure_performance(include_metrics=True, slow_threshold=2.0)
    @handle_errors(fallback="concept_suggestions")
    @mcp_tool("Explain IMAS concepts with physics context")
    async def explain_concept(
        self,
        concept: str,
        detail_level: Union[str, DetailLevel] = "intermediate",
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Explain IMAS concepts with physics context.

        Args:
            concept: The concept to explain (physics concept, IMAS path, or general term)
            detail_level: Level of detail (basic, intermediate, advanced)
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with explanation, physics context, IMAS mappings, and related information
        """
        try:
            # Use SearchService with standardized SearchResult format
            search_config = SearchConfig(
                mode=SearchMode.SEMANTIC,
                max_results=15,
                enable_physics_enhancement=True,
            )
            search_results = await self._search_service.search(concept, search_config)

            # Try physics search enhancement
            physics_context = None
            try:
                physics_context = physics_search(concept)
            except Exception as e:
                logger.warning(f"Physics enhancement failed: {e}")
                # physics_context remains None for optional enhancement

            if not search_results:
                return {
                    "concept": concept,
                    "detail_level": detail_level,
                    "error": "No information found for concept",
                    "suggestions": [
                        "Try alternative terminology",
                        "Check concept spelling",
                        "Use search_imas() to explore related terms",
                    ],
                    "related_paths": [],
                    "physics_context": None,
                    "sources_analyzed": 0,
                    "identifier_analysis": [],
                }

            # Build comprehensive explanation from standardized SearchResult objects
            related_paths = []
            physics_domains: Set[str] = set()
            measurement_contexts = []
            identifier_schemas = []

            for search_result in search_results[:10]:
                # Use direct field access instead of to_dict()
                path_info = {
                    "path": search_result.document.metadata.path_name,
                    "ids_name": search_result.document.metadata.ids_name,
                    "description": search_result.document.documentation[:150],
                    "relevance_score": search_result.score,
                    "physics_domain": search_result.document.metadata.physics_domain,
                    "units": search_result.document.units.unit_str
                    if search_result.document.units
                    else None,
                }
                related_paths.append(path_info)

                # Use direct field access for physics domain
                if search_result.document.metadata.physics_domain:
                    physics_domains.add(search_result.document.metadata.physics_domain)

                # Skip measurement context extraction for now - method doesn't exist
                # measurement_context = search_result.extract_measurement_context()
                # if measurement_context:
                #     measurement_contexts.append(measurement_context)

                # Check for identifier schemas using SearchResult
                identifier_info = self._extract_identifier_info(search_result.document)
                if identifier_info["has_identifier"]:
                    identifier_schemas.append(
                        {
                            "path": search_result.document.metadata.path_name,
                            "schema_info": identifier_info,
                        }
                    )

            # Build explanation response using Pydantic models
            related_paths_data = []
            for search_result in search_results[:8]:
                # Build DataPath with correct fields
                local_physics_context = None
                if search_result.document.metadata.physics_domain:
                    local_physics_context = PhysicsContext(
                        domain=search_result.document.metadata.physics_domain,
                        phenomena=[],
                        typical_values={},
                    )

                related_paths_data.append(
                    DataPath(
                        path=search_result.document.metadata.path_name,
                        documentation=search_result.document.documentation[:150],
                        units=search_result.document.units.unit_str
                        if search_result.document.units
                        else None,
                        data_type=search_result.document.metadata.data_type,
                        physics_context=local_physics_context,
                    )
                )

            # Convert detail_level to enum (handle both string and enum inputs)
            detail_level_enum = self._convert_to_enum(detail_level, DetailLevel)

            # Build sampling prompt for AI enhancement - handled by decorator

            response = ConceptResult(
                concept=concept,
                explanation=f"Analysis of '{concept}' within IMAS data dictionary context. Found in {len(physics_domains)} physics domain(s): {', '.join(list(physics_domains)[:3])}. Related to {len(measurement_contexts)} measurement contexts. Found {len(related_paths)} related data paths.",
                detail_level=detail_level_enum,
                related_topics=[
                    f"Explore {related_paths[0]['ids_name']} for detailed data"
                    if related_paths
                    else "Use search_imas() for more specific queries",
                    f"Investigate {list(physics_domains)[0]} domain connections"
                    if physics_domains
                    else "Consider broader physics concepts",
                    "Use analyze_ids_structure() for structural details"
                    if related_paths
                    else "Try related terminology",
                ],
                concept_explanation=None,  # Will be populated by AI enhancement
                paths=related_paths_data,
                count=len(related_paths_data),
                physics_domains=list(physics_domains),
                physics_context=physics_context,
            )

            return response.model_dump()

        except Exception as e:
            logger.error(f"Concept explanation failed: {e}")
            return ErrorResponse(
                error=f"Error explaining concept '{concept}': {str(e)}",
                suggestions=[
                    "Try simpler concept terms",
                    "Check concept spelling",
                    "Use search_imas() to explore available data",
                ],
                context={"concept": concept, "detail_level": detail_level},
            ).model_dump()
