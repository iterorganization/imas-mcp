"""
Analysis tool implementation with service composition.

This module contains the analyze_ids_structure tool logic using service-based architecture
for physics integration, response building, and standardized metadata.
"""

import logging
from typing import Optional, Union
from fastmcp import Context

from imas_mcp.models.request_models import AnalysisInput
from imas_mcp.models.result_models import StructureResult
from imas_mcp.models.error_models import ToolError

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


class AnalysisTool(BaseTool):
    """Tool for analyzing IDS structure with service composition."""

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "analyze_ids_structure"

    def _build_analysis_sample_prompt(self, ids_name: str) -> str:
        """Build sampling prompt for structure analysis."""
        return f"""IMAS IDS Structure Analysis Request: "{ids_name}"

Please provide a comprehensive structural analysis that includes:

1. **Architecture Overview**: High-level organization of the IDS
2. **Data Hierarchy**: How data is structured and nested
3. **Key Components**: Major data groups and their purposes
4. **Identifier Schemas**: Important branching points and enumerations
5. **Physics Context**: What physics phenomena this IDS represents
6. **Usage Patterns**: Common ways this IDS is used in fusion research
7. **Relationships**: How this IDS connects to other data structures

Focus on providing actionable insights for researchers working with this specific IDS.
"""

    @cache_results(ttl=900, key_strategy="path_based")
    @validate_input(schema=AnalysisInput)
    @measure_performance(include_metrics=True, slow_threshold=2.0)
    @handle_errors(fallback="analysis_suggestions")
    @tool_hints(max_hints=3)
    @physics_hints()
    @sample(temperature=0.2, max_tokens=600)
    @mcp_tool("Get detailed structural analysis of a specific IDS")
    async def analyze_ids_structure(
        self, ids_name: str, ctx: Optional[Context] = None
    ) -> Union[StructureResult, ToolError]:
        """
        Get detailed structural analysis of a specific IDS using service composition.

        Uses service composition and context manager for consistent orchestration:
        - Service context manager handles pre/post processing
        - DocumentService: Validates IDS and retrieves documents
        - PhysicsService: Enhances analysis with physics context

        Args:
            ids_name: Name of the IDS to analyze
            ctx: MCP context for AI enhancement

        Returns:
            StructureResult with detailed analysis and AI insights
        """
        try:
            # Validate IDS exists using document service
            valid_ids, invalid_ids = await self.documents.validate_ids([ids_name])
            if not valid_ids:
                return self.documents.create_ids_not_found_error(
                    ids_name, self.tool_name
                )

            # Get detailed IDS data from document store
            ids_documents = await self.documents.get_documents_safe(ids_name)

            if not ids_documents:
                result = StructureResult(
                    ids_name=ids_name,
                    description=f"No detailed structure data available for {ids_name}",
                    structure={"total_paths": 0},
                    sample_paths=[],
                    max_depth=0,
                    ai_response={
                        "analysis": "No structure data available",
                        "note": "IDS exists but has no accessible structure information",
                    },
                )
                logger.info(f"Structure analysis completed for {ids_name}")
                return result

            # Analyze structure
            structure_analysis = self._analyze_structure(ids_documents)
            sample_paths = [doc.metadata.path_name for doc in ids_documents[:10]]

            # Get physics context
            physics_context = await self.physics.enhance_query(ids_name)

            # Build response
            result = StructureResult(
                ids_name=ids_name,
                description=f"Structural analysis of {ids_name} IDS containing {len(ids_documents)} data paths",
                structure=structure_analysis,
                sample_paths=sample_paths,
                max_depth=structure_analysis.get("max_depth", 0),
                ai_response={},  # Reserved for LLM sampling only
                physics_context=physics_context,
            )

            logger.info(f"Structure analysis completed for {ids_name}")
            return result

        except Exception as e:
            logger.error(f"Structure analysis failed for {ids_name}: {e}")
            return self._create_error_response(f"Analysis failed: {e}", ids_name)

    def _analyze_structure(self, ids_documents):
        """Analyze structure of IDS documents."""
        paths = [doc.metadata.path_name for doc in ids_documents]

        # Analyze identifier schemas
        identifier_nodes = []
        for doc in ids_documents:
            identifier_schema = doc.raw_data.get("identifier_schema")
            if identifier_schema and isinstance(identifier_schema, dict):
                options = identifier_schema.get("options", [])
                identifier_nodes.append(
                    {
                        "path": doc.metadata.path_name,
                        "option_count": len(options),
                    }
                )

        # Build structure analysis
        structure_data = {
            "root_level_paths": len([p for p in paths if "/" not in p.strip("/")]),
            "max_depth": max(len(p.split("/")) for p in paths) if paths else 0,
            "document_count": len(ids_documents),
            "identifier_nodes": len(identifier_nodes),
            "branching_complexity": sum(
                node["option_count"] for node in identifier_nodes
            ),
        }

        return structure_data
