"""
Overview tool implementation with service composition.

This module contains the get_overview tool logic using service-based architecture
for physics integration, response building, and standardized metadata.
"""

import logging
from typing import Optional, Union
from fastmcp import Context

from imas_mcp.models.request_models import OverviewInput
from imas_mcp.models.response_models import OverviewResult, ErrorResponse
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


class OverviewTool(BaseTool):
    """Tool for getting IMAS overview."""

    # Enable services for overview-based tool
    enable_sampling: bool = True
    enable_recommendations: bool = True
    sampling_strategy = SamplingStrategy.SMART
    recommendation_strategy = RecommendationStrategy.OVERVIEW_BASED
    max_recommended_tools: int = 4

    def get_tool_name(self) -> str:
        return "get_overview"

    def _build_overview_analysis_prompt(self, query: Optional[str] = None) -> str:
        """Build AI analysis prompt for overview generation."""
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
    @validate_input(schema=OverviewInput)
    @measure_performance(include_metrics=True, slow_threshold=3.0)
    @handle_errors(fallback="overview_suggestions")
    @mcp_tool("Get IMAS overview or answer analytical queries")
    async def get_overview(
        self,
        query: Optional[str] = None,
        ctx: Optional[Context] = None,
    ) -> Union[OverviewResult, ErrorResponse]:
        """
        Get IMAS overview or answer analytical queries using service composition.

        Args:
            query: Optional specific query about the data dictionary
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with overview information, analytics, and optional domain-specific data
        """
        try:
            # Get all available IDS first
            all_available_ids = self.document_store.get_available_ids()

            # Determine relevant IDS based on query
            if query:
                # Use search to find most relevant IDS for the query
                try:
                    config = self.search_config.create_config(
                        search_mode="semantic",
                        max_results=50,  # Cast wider net for overview
                    )
                    config = self.search_config.optimize_for_query(query, config)
                    search_results = await self._search_service.search(query, config)

                    # Extract unique IDS names from search results
                    relevant_ids_set = set()
                    for result in search_results:
                        if hasattr(result.document.metadata, "ids_name"):
                            relevant_ids_set.add(result.document.metadata.ids_name)

                    # Use relevant IDS if found, otherwise fall back to all
                    relevant_ids = (
                        list(relevant_ids_set)
                        if relevant_ids_set
                        else all_available_ids
                    )

                except Exception as e:
                    logger.warning(f"Query-based IDS filtering failed: {e}")
                    relevant_ids = all_available_ids
            else:
                relevant_ids = all_available_ids

            # Gather dynamic statistics from document store
            physics_domains = set()
            data_types = set()
            units_found = set()
            query_results = []

            for ids_name in relevant_ids[:10]:  # Limit analysis for performance
                try:
                    # Get documents for this IDS
                    ids_documents = await self.documents.get_documents_safe(ids_name)

                    for doc in ids_documents[:20]:  # Sample documents per IDS
                        # Collect physics domains
                        if (
                            hasattr(doc.metadata, "physics_domain")
                            and doc.metadata.physics_domain
                        ):
                            physics_domains.add(doc.metadata.physics_domain)

                        # Collect data types
                        if (
                            hasattr(doc.metadata, "data_type")
                            and doc.metadata.data_type
                        ):
                            data_types.add(doc.metadata.data_type)

                        # Collect units
                        if (
                            hasattr(doc, "units")
                            and doc.units
                            and hasattr(doc.units, "unit_str")
                        ):
                            if doc.units.unit_str:
                                units_found.add(doc.units.unit_str)

                except Exception as e:
                    logger.warning(f"Failed to analyze IDS {ids_name}: {e}")

            # If we have a specific query, get search results for display
            if query:
                try:
                    config = self.search_config.create_config(
                        search_mode="semantic",
                        max_results=10,
                    )
                    config = self.search_config.optimize_for_query(query, config)
                    search_results = await self._search_service.search(query, config)
                    query_results = [result.to_hit() for result in search_results[:5]]
                except Exception as e:
                    logger.warning(f"Query search failed: {e}")

            # Validate the relevant IDS exist
            valid_ids, invalid_ids = await self.documents.validate_ids(relevant_ids)

            # Generate dynamic usage guidance using services and context
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
                "getting_started": [],
            }

            # Use physics and context to generate intelligent suggestions
            suggestions_added = 0
            max_suggestions = 4

            # Physics domain-based suggestions
            if physics_domains and suggestions_added < max_suggestions:
                first_domain = next(iter(physics_domains))
                # Use PhysicsService to get better context if available
                try:
                    physics_context = await self.physics.get_concept_context(
                        first_domain
                    )
                    if physics_context:
                        domain_desc = physics_context.get("description", first_domain)
                        usage_guidance["getting_started"].append(
                            f"Try explain_concept('{first_domain}') - {domain_desc[:50]}..."
                        )
                    else:
                        usage_guidance["getting_started"].append(
                            f"Try explain_concept('{first_domain}') to understand this physics domain"
                        )
                    suggestions_added += 1
                except Exception as e:
                    logger.warning(f"Physics context lookup failed: {e}")
                    usage_guidance["getting_started"].append(
                        f"Try explain_concept('{first_domain}') to understand this physics domain"
                    )
                    suggestions_added += 1

            # Data type-based suggestions
            if data_types and suggestions_added < max_suggestions:
                usage_guidance["getting_started"].append(
                    f"Search for specific data types like: {', '.join(list(data_types)[:3])}"
                )
                suggestions_added += 1

            # IDS structure suggestions
            if valid_ids and suggestions_added < max_suggestions:
                first_ids = valid_ids[0]
                usage_guidance["getting_started"].append(
                    f"Use analyze_ids_structure('{first_ids}') for detailed structural analysis"
                )
                suggestions_added += 1

            # Query-specific suggestions
            if query and suggestions_added < max_suggestions:
                usage_guidance["getting_started"].append(
                    f"Use search_imas('{query}') to find more specific data paths"
                )
                suggestions_added += 1

            # Fallback suggestions if no dynamic content
            if not usage_guidance["getting_started"]:
                usage_guidance["getting_started"] = [
                    "Use search_imas('temperature') to find temperature-related data",
                    "Try explain_concept('plasma') for physics context",
                    "Use get_overview('magnetic field') for domain-specific queries",
                ]

            # Generate dynamic per-IDS statistics
            ids_statistics = {}
            for ids_name in valid_ids:
                try:
                    # Get actual document count for this IDS
                    ids_documents = await self.documents.get_documents_safe(ids_name)
                    path_count = len(ids_documents)

                    # Count identifiers (simplified - could be enhanced)
                    identifier_count = sum(
                        1
                        for doc in ids_documents
                        if hasattr(doc, "raw_data")
                        and doc.raw_data.get("identifier_schema") is not None
                    )

                    # Generate description based on physics domains found
                    doc_domains = {
                        doc.metadata.physics_domain
                        for doc in ids_documents[:5]
                        if hasattr(doc.metadata, "physics_domain")
                        and doc.metadata.physics_domain
                    }

                    if doc_domains:
                        domain_desc = f" ({', '.join(list(doc_domains)[:2])})"
                    else:
                        domain_desc = ""

                    ids_statistics[ids_name] = {
                        "path_count": path_count,
                        "identifier_count": identifier_count,
                        "description": f"{ids_name.replace('_', ' ').title()} IDS{domain_desc}",
                        "sample_domains": list(doc_domains)[:3] if doc_domains else [],
                    }

                except Exception as e:
                    logger.warning(f"Failed to generate statistics for {ids_name}: {e}")
                    # Fallback to basic info
                    ids_statistics[ids_name] = {
                        "path_count": 0,
                        "identifier_count": 0,
                        "description": f"{ids_name.replace('_', ' ').title()} IDS",
                        "sample_domains": [],
                    }

            # Prepare AI insights for sampling service using PhysicsService
            analysis_prompt = self._build_overview_analysis_prompt(query)

            # Enhance AI insights with physics context if available
            physics_insights = {}
            if physics_domains:
                try:
                    first_domain = next(iter(physics_domains))
                    physics_context = await self.physics.get_concept_context(
                        first_domain
                    )
                    if physics_context:
                        physics_insights = {
                            "primary_domain": first_domain,
                            "domain_context": physics_context.get("description", ""),
                            "complexity": physics_context.get(
                                "complexity_level", "intermediate"
                            ),
                            "typical_units": physics_context.get("typical_units", []),
                        }
                except Exception as e:
                    logger.warning(f"Physics insights generation failed: {e}")

            ai_insights = {
                "analysis_prompt": analysis_prompt,
                "stats_analyzed": len(valid_ids),
                "query_results_found": len(query_results) if query else 0,
                "dynamic_analysis": {
                    "physics_domains_found": len(physics_domains),
                    "data_types_found": len(data_types),
                    "units_found": len(units_found),
                    "query_focused": bool(query),
                },
                "physics_insights": physics_insights,
            }

            # Build dynamic overview response content
            content_parts = []

            if query:
                content_parts.append(f"IMAS Data Dictionary Overview for: '{query}'")
                content_parts.append(
                    f"Analysis focused on {len(valid_ids)} relevant IDS"
                )
            else:
                content_parts.append("IMAS Data Dictionary Overview")
                content_parts.append(f"Total available IDS: {len(all_available_ids)}")

            # Add dynamic statistics
            if physics_domains:
                domain_list = ", ".join(list(physics_domains)[:5])
                if len(physics_domains) > 5:
                    domain_list += f" and {len(physics_domains) - 5} more"
                content_parts.append(f"Physics domains found: {domain_list}")

            if data_types:
                type_list = ", ".join(list(data_types)[:6])
                if len(data_types) > 6:
                    type_list += f" and {len(data_types) - 6} more"
                content_parts.append(f"Data types available: {type_list}")

            if units_found:
                unit_list = ", ".join(list(units_found)[:8])
                if len(units_found) > 8:
                    unit_list += f" and {len(units_found) - 8} more"
                content_parts.append(f"Common units: {unit_list}")

            if query and query_results:
                content_parts.append(
                    f"Found {len(query_results)} specific paths matching your query"
                )
            elif query and not query_results:
                content_parts.append(
                    "No specific paths found for your query - consider broader terms"
                )

            content_parts.append("\nRecommended next steps:")
            content_parts.extend(
                [f"  • {item}" for item in usage_guidance["getting_started"]]
            )

            overview_response = OverviewResult(
                content="\n".join(content_parts),
                available_ids=valid_ids,
                query=query,
                physics_domains=list(physics_domains),
                physics_context=None,
                hits=query_results if query else [],
                ids_statistics=ids_statistics,
                usage_guidance=usage_guidance,
                ai_insights=ai_insights,
            )

            # Add standard metadata using service
            overview_response = self.response.add_standard_metadata(
                overview_response, self.get_tool_name()
            )

            # Apply post-processing services (sampling and recommendations)
            processed_response = await self.apply_services(
                result=overview_response,
                query=query,
                tool_name=self.get_tool_name(),
                ctx=ctx,
            )

            # Type check and return proper response
            if isinstance(processed_response, OverviewResult):
                return processed_response
            elif isinstance(processed_response, ErrorResponse):
                return processed_response
            else:
                # If apply_services changed the type unexpectedly, return original
                logger.warning(
                    f"Service processing changed response type: {type(processed_response)}"
                )
                return overview_response

        except Exception as e:
            logger.error(f"Overview generation failed: {e}")
            return ErrorResponse(
                error=str(e),
                suggestions=[
                    "Try simpler queries",
                    "Use search_imas() to explore specific topics",
                    "Check for system availability",
                ],
                context={
                    "query": query,
                    "tool": "get_overview",
                    "operation": "overview_generation",
                },
            )
