"""
Analysis tool implementation.

This module contains the analyze_ids_structure tool logic with decorators
for caching, validation, AI enhancement, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Optional, Union
from fastmcp import Context

from imas_mcp.search.document_store import DocumentStore
from imas_mcp.models.request_models import AnalysisInput
from imas_mcp.models.response_models import StructureResult, ErrorResponse

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


class AnalysisTool(BaseTool):
    """Tool for analyzing IDS structure."""

    def __init__(self, document_store: Optional[DocumentStore] = None):
        """Initialize the analysis tool with document store."""
        super().__init__()
        self.document_store = document_store or DocumentStore()

    def get_tool_name(self) -> str:
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

    @cache_results(ttl=900, key_strategy="path_based")  # Cache structure analysis
    @validate_input(schema=AnalysisInput)
    @sample(temperature=0.2, max_tokens=800)  # Conservative temperature for analysis
    @recommend_tools(strategy="analysis_based", max_tools=3)
    @measure_performance(include_metrics=True, slow_threshold=2.0)
    @handle_errors(fallback="analysis_suggestions")
    @mcp_tool("Get detailed structural analysis of a specific IDS")
    async def analyze_ids_structure(
        self, ids_name: str, ctx: Optional[Context] = None
    ) -> Union[StructureResult, ErrorResponse]:
        """
        Get detailed structural analysis of a specific IDS using graph metrics.

        Args:
            ids_name: Name of the IDS to analyze
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with detailed graph analysis and AI insights
        """
        try:
            # Validate IDS exists
            available_ids = self.document_store.get_available_ids()
            if ids_name not in available_ids:
                return ErrorResponse(
                    error=f"IDS '{ids_name}' not found",
                    suggestions=[f"Try: {ids}" for ids in available_ids[:5]],
                    context={
                        "available_ids": available_ids[:10],
                        "ids_name": ids_name,
                        "tool": "analyze_ids_structure",
                    },
                )

            # Get detailed IDS data from document store
            ids_documents = self.document_store.get_documents_by_ids(ids_name)

            # Build structural analysis from documents with identifier awareness
            paths = [doc.metadata.path_name for doc in ids_documents]

            # Analyze identifier schemas in this IDS
            identifier_nodes = []
            for doc in ids_documents:
                identifier_schema = doc.raw_data.get("identifier_schema")
                if identifier_schema and isinstance(identifier_schema, dict):
                    options = identifier_schema.get("options", [])
                    identifier_nodes.append(
                        {
                            "path": doc.metadata.path_name,
                            "schema_path": identifier_schema.get(
                                "schema_path", "unknown"
                            ),
                            "option_count": len(options),
                            "branching_significance": "CRITICAL"
                            if len(options) > 5
                            else "MODERATE"
                            if len(options) > 1
                            else "MINIMAL",
                            "sample_options": [
                                {
                                    "name": opt.get("name", ""),
                                    "index": opt.get("index", 0),
                                }
                                for opt in options[:3]
                            ]
                            if options
                            else [],
                        }
                    )

            # Build structure analysis using dict format
            structure_data = {
                "root_level_paths": len([p for p in paths if "/" not in p.strip("/")]),
                "max_depth": max(len(p.split("/")) for p in paths) if paths else 0,
                "document_count": len(ids_documents),
            }

            # Analyze path patterns
            path_patterns = {}
            for path in paths:
                segments = path.split("/")
                if len(segments) > 1:
                    root = segments[0]
                    path_patterns[root] = path_patterns.get(root, 0) + 1

            # Build final response using Pydantic
            response = StructureResult(
                ids_name=ids_name,
                description=f"IDS '{ids_name}' containing {len(paths)} data paths",
                structure=structure_data,
                sample_paths=paths[:10],
                max_depth=max(len(path.split("/")) for path in paths) if paths else 0,
                physics_domains=[
                    domain
                    for domain in [
                        "equilibrium",
                        "core_profiles",
                        "disruptions",
                        "transport",
                        "heating",
                        "current_drive",
                        "mhd",
                    ]
                    if domain in ids_name.lower()
                ],
                physics_context=None,
            )

            # Add identifier analysis as ai_insights
            response.ai_insights = {
                "total_identifier_nodes": len(identifier_nodes),
                "branching_paths": identifier_nodes,
                "coverage": f"{len(identifier_nodes) / len(paths) * 100:.1f}%"
                if paths
                else "0%",
            }

            return response

        except Exception as e:
            logger.error(f"IDS structure analysis failed: {e}")
            return ErrorResponse(
                error=str(e),
                suggestions=[
                    "Check IDS name spelling",
                    "Verify IDS exists in data dictionary",
                    "Try get_overview() to see available IDS",
                ],
                context={
                    "ids_name": ids_name,
                    "tool": "analyze_ids_structure",
                    "operation": "structure_analysis",
                },
            )
