"""
Overview tool implementation.

This module contains the get_overview tool logic with decorators
for caching, validation, AI enhancement, tool recommendations, performance
monitoring, and error handling.
"""

import logging
from typing import Dict, Any, Optional
from fastmcp import Context

from imas_mcp.search.search_strategy import SearchConfig
from imas_mcp.search.services.search_service import SearchService
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.engines.semantic_engine import SemanticSearchEngine
from imas_mcp.search.engines.lexical_engine import LexicalSearchEngine
from imas_mcp.search.engines.hybrid_engine import HybridSearchEngine
from imas_mcp.models.request_models import OverviewInputSchema
from imas_mcp.models.constants import SearchMode
from imas_mcp.models.response_models import OverviewResult

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


class OverviewTool(BaseTool):
    """Tool for getting IMAS overview."""

    def __init__(self, document_store: DocumentStore):
        """Initialize the overview tool with document store."""
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
        return "get_overview"

    def _build_overview_sample_prompt(self, query: Optional[str] = None) -> str:
        """Build sampling prompt for overview generation."""
        if query:
            return f"""IMAS Data Dictionary Overview Request: "{query}"

Please provide a comprehensive overview that addresses the specific query while covering:

1. **Relevant IMAS Context**: How the query relates to IMAS data structures
2. **Data Availability**: What data is available related to the query
3. **Physics Context**: Relevant physics domains and phenomena
4. **Usage Guidance**: Practical recommendations for data access and analysis
5. **Related Tools**: Which IMAS MCP tools would be most helpful for this query

Focus on being helpful and informative while maintaining accuracy about IMAS capabilities.
"""
        else:
            return """IMAS Data Dictionary General Overview

Please provide a comprehensive overview of the IMAS (Integrated Modelling & Analysis Suite) data dictionary covering:

1. **Structure Overview**: General organization of IMAS data
2. **Key Physics Domains**: Main areas covered (core plasma, transport, equilibrium, etc.)
3. **Data Types Available**: Common measurement types and calculated quantities
4. **Getting Started**: How new users should approach IMAS data access
5. **Tool Ecosystem**: Overview of available MCP tools for data exploration

Provide practical guidance for fusion researchers and IMAS users.
"""

    @cache_results(ttl=1800, key_strategy="semantic")  # Longer cache for overviews
    @validate_input(schema=OverviewInputSchema)
    @sample(temperature=0.3, max_tokens=1000)  # Balanced creativity for overviews
    @recommend_tools(strategy="overview_based", max_tools=4)
    @measure_performance(include_metrics=True, slow_threshold=3.0)
    @handle_errors(fallback="overview_suggestions")
    @mcp_tool("Get IMAS overview or answer analytical queries")
    async def get_overview(
        self,
        query: Optional[str] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Get IMAS overview or answer analytical queries with insights.

        Args:
            query: Optional specific query about the data dictionary
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with overview information, analytics, and optional domain-specific data
        """
        try:
            # Get basic statistics - mock implementation for now
            available_ids = [
                "core_profiles",
                "equilibrium",
                "transport",
                "heating",
                "wall",
            ]

            # Get sample analysis data
            physics_domains = {"core_plasma", "transport", "equilibrium", "heating"}
            data_types = {"float", "integer", "string", "array"}
            units_found = {"eV", "T", "m", "s", "A", "Pa"}

            # Handle specific queries
            query_results = []
            if query:
                # Search for query-related content
                try:
                    search_config = SearchConfig(
                        mode=SearchMode.SEMANTIC,
                        max_results=10,
                    )
                    search_results = await self._search_service.search(
                        query, search_config
                    )
                    query_results = [
                        {
                            "path": result.document.metadata.path_name,
                            "documentation": result.document.documentation,
                            "score": result.score,
                            "physics_domain": result.document.metadata.physics_domain,
                        }
                        for result in search_results[:5]
                    ]
                except Exception as e:
                    logger.warning(f"Query search failed: {e}")

            # Usage guidance for users
            usage_guidance = {
                "tools_available": [
                    "search_imas - Find specific data paths",
                    "explain_concept - Get physics explanations",
                    "analyze_ids_structure - Detailed IDS analysis",
                    "explore_relationships - Find data connections",
                    "explore_identifiers - Identifier schema analysis",
                    "export_ids - Multi-IDS data export",
                    "export_physics_domain - Domain-specific export",
                ],
                "getting_started": [
                    "Use search_imas('temperature') to find temperature-related data",
                    "Try explain_concept('plasma') for physics context",
                    "Use get_overview('magnetic field') for domain-specific queries",
                ],
            }

            # Generate per-IDS statistics
            ids_statistics = {}
            for ids_name in available_ids:
                # Mock statistics - in real implementation would query document store
                ids_statistics[ids_name] = {
                    "path_count": 50,  # Would be actual count
                    "identifier_count": 10,  # Would be actual count
                    "description": f"{ids_name.replace('_', ' ').title()} IDS",
                }

            # Build sampling prompt for AI enhancement - handled by decorator

            # Build overview response using Pydantic
            content_parts = [
                "IMAS Data Dictionary Overview:",
                f"Total IDS: {len(available_ids)}",
                f"Physics domains: {', '.join(list(physics_domains))}",
                f"Data types available: {', '.join(list(data_types))}",
                f"Common units: {', '.join(list(units_found)[:10])}",
            ]

            if query and query_results:
                content_parts.append(
                    f"Found {len(query_results)} paths related to your query."
                )

            content_parts.append("\nGetting Started:")
            content_parts.extend(
                [f"  • {item}" for item in usage_guidance["getting_started"]]
            )

            overview_response = OverviewResult(
                content="\n".join(content_parts),
                available_ids=available_ids,
                query=query,
                physics_domains=list(physics_domains),
                physics_context=None,
            )

            result = overview_response.model_dump()

            # Add additional context for queries
            if query and query_results:
                result["query_results"] = query_results
                result["query_results_count"] = len(query_results)

            result["ids_statistics"] = ids_statistics
            result["usage_guidance"] = usage_guidance

            return result

        except Exception as e:
            logger.error(f"Overview generation failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "overview": "Failed to generate overview",
                "available_ids": [],
                "physics_domains": [],
                "suggestions": [
                    "Try simpler queries",
                    "Use search_imas() to explore specific topics",
                    "Check system status",
                ],
            }
