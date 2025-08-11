"""
Analysis tool implementation with service composition.

This module contains the analyze_ids_structure tool logic using service-based architecture
for physics integration, response building, and standardized metadata.
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp.models.error_models import ToolError
from imas_mcp.models.request_models import AnalysisInput
from imas_mcp.models.result_models import StructureResult
from imas_mcp.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    measure_performance,
    physics_hints,
    sample,
    tool_hints,
    validate_input,
)

from .base import BaseTool

logger = logging.getLogger(__name__)


class AnalysisTool(BaseTool):
    """Tool for analyzing IDS structure with service composition."""

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "analyze_ids_structure"

    def build_prompt(self, prompt_type: str, tool_context: dict[str, Any]) -> str:
        """Build analysis-specific AI prompts."""
        if prompt_type == "structure_analysis":
            # For structure analysis, we expect query to be the IDS name
            ids_name = tool_context.get("query", "")
            return self._build_structure_analysis_prompt_simple(ids_name)
        elif prompt_type == "no_data":
            # For no data case, we expect query to be the IDS name
            ids_name = tool_context.get("query", "")
            return self._build_no_data_prompt_simple(ids_name)
        return ""

    def system_prompt(self) -> str:
        """Get analysis tool-specific system prompt."""
        return """You are an expert IMAS data architect and fusion physics analyst specializing in:

- IMAS data dictionary structure, organization principles, and design patterns
- Hierarchical data relationships and identifier schema architectures
- Physics-based data organization and measurement categorization
- Data access optimization and workflow design patterns
- Cross-IDS relationships and data integration strategies

Your expertise enables you to:
1. Analyze complex data hierarchies and identify key organizational patterns
2. Explain the physics rationale behind data structure decisions
3. Recommend optimal data access strategies for different research workflows
4. Identify important branching points, enumerations, and identifier schemas
5. Suggest related data structures and cross-references
6. Provide actionable guidance for navigating large datasets efficiently

When analyzing IDS structures, focus on:
- High-level architectural insights and design principles
- Key access patterns and common usage workflows
- Physics-motivated organization and measurement groupings
- Practical guidance for researchers working with the data
- Relationships to other IDS and broader data integration strategies
- Performance considerations and optimization opportunities

Provide analysis that helps researchers understand not just what data is available, but how to work with it effectively in their specific research contexts."""

    def _build_structure_analysis_prompt(self, tool_context: dict[str, Any]) -> str:
        """Build prompt for IDS structure analysis."""
        ids_name = tool_context.get("ids_name", "")
        structure_analysis = tool_context.get("structure_analysis", {})
        sample_paths = tool_context.get("sample_paths", [])
        document_count = tool_context.get("document_count", 0)
        physics_context = tool_context.get("physics_context")

        prompt = f"""IMAS IDS Structure Analysis: "{ids_name}"

Structural Data:
- Total data paths: {document_count}
- Root level paths: {structure_analysis.get("root_level_paths", 0)}
- Maximum depth: {structure_analysis.get("max_depth", 0)}
- Identifier nodes: {structure_analysis.get("identifier_nodes", 0)}
- Branching complexity: {structure_analysis.get("branching_complexity", 0)}

Sample data paths:
"""
        for i, path in enumerate(sample_paths[:8], 1):
            prompt += f"{i}. {path}\n"

        if physics_context:
            # Handle physics_context properly - it might not be a dict
            context_desc = ""
            if hasattr(physics_context, "description"):
                context_desc = physics_context.description
            elif hasattr(physics_context, "get"):
                context_desc = physics_context.get("description", "")
            elif isinstance(physics_context, str):
                context_desc = physics_context
            else:
                context_desc = str(physics_context)

            if context_desc:
                prompt += f"\nPhysics Context: {context_desc}\n"

        prompt += """
Please provide a comprehensive structural analysis that includes:

1. **Architecture Overview**: High-level organization of the IDS and its design patterns
2. **Data Hierarchy**: How data is structured, nested, and organized
3. **Key Components**: Major data groups, their purposes, and relationships
4. **Identifier Schemas**: Important branching points, enumerations, and access patterns
5. **Physics Context**: What physics phenomena this IDS represents and measures
6. **Usage Patterns**: Common ways this IDS is used in fusion research workflows
7. **Data Access Guidance**: Best practices for accessing and interpreting this data
8. **Relationships**: How this IDS connects to other IMAS data structures

Focus on providing actionable insights for researchers working with this specific IDS.
"""
        return prompt

    def _build_no_data_prompt(self, tool_context: dict[str, Any]) -> str:
        """Build prompt for when no structure data is available."""
        ids_name = tool_context.get("ids_name", "")

        return f"""IDS Structure Analysis Request: "{ids_name}"

No structure data is available for this IDS.

Please provide:
1. General information about this IDS type if known
2. Suggestions for alternative IDS names or spellings
3. Common IMAS IDS that might be related
4. Guidance on how to explore available IDS structures
5. Recommended follow-up actions for data discovery"""

    def _build_no_data_prompt_simple(self, ids_name: str) -> str:
        """Build simplified prompt for when no structure data is available."""
        return f"""IDS Structure Analysis Request: "{ids_name}"

No structure data is available for this IDS.

Please provide:
1. General information about this IDS type if known
2. Suggestions for alternative IDS names or spellings
3. Common IMAS IDS that might be related
4. Guidance on how to explore available IDS structures
5. Recommended follow-up actions for data discovery"""

    def _build_structure_analysis_prompt_simple(self, ids_name: str) -> str:
        """Build simplified prompt for structure analysis."""
        return f"""IMAS IDS Structure Analysis: "{ids_name}"

Please provide a comprehensive structural analysis that includes:

1. **Architecture Overview**: High-level organization of the {ids_name} IDS and its design patterns
2. **Data Hierarchy**: How data is structured, nested, and organized within this IDS
3. **Key Components**: Major data groups, their purposes, and relationships
4. **Identifier Schemas**: Important branching points, enumerations, and access patterns
5. **Physics Context**: What physics phenomena this IDS represents and measures
6. **Usage Patterns**: Common ways this IDS is used in fusion research workflows
7. **Data Access Guidance**: Best practices for accessing and interpreting this data
8. **Relationships**: How this IDS connects to other IMAS data structures

Focus on providing actionable insights for researchers working with the {ids_name} IDS specifically."""

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
    @mcp_tool("Analyze the internal structure and organization of a specific IMAS IDS")
    async def analyze_ids_structure(
        self, ids_name: str, ctx: Context | None = None
    ) -> StructureResult | ToolError:
        """
        Analyze the internal structure and organization of a specific IMAS IDS.

        Provides comprehensive structural analysis including data hierarchy, branching
        points, identifier schemas, and physics context. Use this tool to understand
        how data is organized within an IDS before accessing specific measurements.

        Args:
            ids_name: Name of the IDS to analyze (e.g., 'equilibrium', 'thomson_scattering')
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

            # AI prompt will be built automatically by the @sample decorator
            # The decorator uses this tool instance (via PromptBuilder protocol)
            # to call build_prompt() and system_prompt() methods when needed

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
