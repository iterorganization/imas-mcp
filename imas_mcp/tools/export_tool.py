"""
Export tool implementation.

This module contains the export_ids and export_physics_domain tool logic
with decorators for caching, validation, AI enhancement, tool recommendations,
performance monitoring, and error handling.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Union
from fastmcp import Context

from imas_mcp.search.search_strategy import SearchConfig
from imas_mcp.models.request_models import (
    ExportIdsInput,
    ExportPhysicsDomainInput,
)
from imas_mcp.models.constants import SearchMode, OutputFormat
from imas_mcp.models.response_models import IDSExport, DomainExport, ErrorResponse

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


class ExportTool(BaseTool):
    """Tool for exporting IDS and physics domain data."""

    def get_tool_name(self) -> str:
        return "export_tools"

    def _extract_identifier_info(self, doc) -> Dict[str, Any]:
        """Extract identifier information from document."""
        identifier_schema = doc.raw_data.get("identifier_schema", {})
        if identifier_schema:
            return {
                "schema_path": identifier_schema.get("schema_path", ""),
                "options_count": len(identifier_schema.get("options", [])),
                "sample_options": [
                    opt.get("name", "")
                    for opt in identifier_schema.get("options", [])[:3]
                ],
            }
        return {}

    def _build_export_sample_prompt(
        self, ids_list: List[str], output_format: str = "structured"
    ) -> str:
        """Build sampling prompt for bulk export."""
        return f"""IMAS Bulk Export Analysis Request:
IDS Requested: {ids_list}
Export Format: {output_format}

Please provide comprehensive export guidance that includes:

1. **Data Usage Recommendations**: Best practices for this specific IDS combination
2. **Physics Insights**: Relationships between exported IDS and physics phenomena
3. **Analysis Workflows**: Suggested workflows utilizing the exported data
4. **Integration Patterns**: How these IDS work together in fusion research
5. **Quality Considerations**: Data validation and consistency checks
6. **Measurement Dependencies**: How different measurements relate
7. **Research Applications**: Common use cases for this data combination

Focus on providing actionable guidance for researchers working with this specific data combination.
"""

    def _build_domain_export_sample_prompt(
        self, domain: str, analysis_depth: str = "focused"
    ) -> str:
        """Build sampling prompt for domain export."""
        return f"""IMAS Physics Domain Export Request: "{domain}"
Analysis Depth: {analysis_depth}

Please provide comprehensive domain export analysis that includes:

1. **Domain Overview**: Key characteristics of this physics domain
2. **Data Hierarchy**: How domain data is organized in IMAS
3. **Measurement Types**: Key measurements and calculated quantities
4. **Cross-Domain Connections**: Links to other physics domains
5. **Analysis Patterns**: Common analysis workflows for this domain
6. **Research Context**: How this domain fits into fusion research
7. **Data Quality**: Important validation considerations
8. **Recommendations**: Best practices for domain-specific analysis

Focus on providing actionable insights for researchers working in this physics domain.
"""

    @cache_results(ttl=600, key_strategy="content_based")  # Cache export results
    @validate_input(schema=ExportIdsInput)
    @sample(temperature=0.3, max_tokens=1000)  # Balanced creativity for export
    @recommend_tools(strategy="export_based", max_tools=3)
    @measure_performance(include_metrics=True, slow_threshold=5.0)
    @handle_errors(fallback="export_suggestions")
    @mcp_tool("Export bulk IMAS data for multiple IDS")
    async def export_ids(
        self,
        ids_list: List[str],
        include_relationships: bool = True,
        include_physics: bool = True,
        output_format: str = "structured",
        ctx: Optional[Context] = None,
    ) -> Union[IDSExport, ErrorResponse]:
        """
        Export bulk IMAS data for multiple IDS with sophisticated relationship analysis.

        Advanced bulk export tool that extracts comprehensive data for multiple IDS,
        including cross-IDS relationships, physics context, and structural analysis.

        Args:
            ids_list: List of IDS names to export
            include_relationships: Whether to include cross-IDS relationship analysis
            include_physics: Whether to include physics domain context
            output_format: Export format (structured, json, yaml, markdown)
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with bulk export data, relationships, and AI insights
        """
        try:
            if not ids_list:
                return IDSExport(
                    ids_names=[],
                    include_physics=include_physics,
                    include_relationships=include_relationships,
                    output_format=output_format,
                    data={
                        "error": "No IDS specified for bulk export",
                        "suggestions": [
                            "Provide at least one IDS name",
                            "Use get_overview to see available IDS",
                        ],
                    },
                )

            # Validate format
            valid_formats = [format.value for format in OutputFormat]
            if output_format not in valid_formats:
                return IDSExport(
                    ids_names=ids_list,
                    include_physics=include_physics,
                    include_relationships=include_relationships,
                    output_format=output_format,
                    data={
                        "error": f"Invalid format: {output_format}. Use: {', '.join(valid_formats)}",
                        "suggestions": [
                            "Use 'structured' for organized data with relationships",
                            "Use 'json' for JSON format export",
                            "Use 'yaml' for YAML format export",
                            "Use 'markdown' for documentation-style export",
                        ],
                    },
                )

            # Validate IDS names
            available_ids = self.document_store.get_available_ids()
            invalid_ids = [ids for ids in ids_list if ids not in available_ids]
            valid_ids = [ids for ids in ids_list if ids in available_ids]

            if not valid_ids:
                return IDSExport(
                    ids_names=ids_list,
                    include_physics=include_physics,
                    include_relationships=include_relationships,
                    data={
                        "error": "No valid IDS names provided",
                        "invalid_ids": invalid_ids,
                        "available_ids": available_ids[:10],
                        "suggestions": [
                            "Check IDS name spelling",
                            "Use get_overview to see all available IDS",
                        ],
                    },
                )

            export_data = {
                "requested_ids": ids_list,
                "valid_ids": valid_ids,
                "invalid_ids": invalid_ids,
                "export_format": output_format,
                "timestamp": "bulk_export",
                "ids_data": {},
                "cross_relationships": {},
                "physics_domains": {},
                "export_summary": {},
            }

            # Export data for each valid IDS
            for ids_name in valid_ids:
                try:
                    # Get all documents for this IDS
                    ids_documents = self.document_store.get_documents_by_ids(ids_name)

                    ids_info: Dict[str, Any] = {
                        "ids_name": ids_name,
                        "total_paths": len(ids_documents),
                        "paths": [],
                        "physics_domains": set(),
                        "identifier_paths": [],
                        "measurement_types": set(),
                    }

                    # Process documents based on output format
                    for doc in ids_documents:
                        path_data: Dict[str, Any] = {
                            "path": doc.metadata.path_name,
                            "documentation": doc.documentation
                            if output_format == "enhanced"
                            else doc.documentation[:200],
                            "data_type": doc.metadata.data_type,
                            "physics_domain": doc.metadata.physics_domain,
                            "units": doc.metadata.units,
                        }

                        # Add detailed information for enhanced format
                        if output_format == "enhanced":
                            path_data["raw_data"] = doc.raw_data
                            path_data["identifier_info"] = (
                                self._extract_identifier_info(doc)
                            )

                        ids_info["paths"].append(path_data)

                        if doc.metadata.physics_domain:
                            ids_info["physics_domains"].add(doc.metadata.physics_domain)

                        if doc.metadata.data_type == "identifier_path":
                            ids_info["identifier_paths"].append(path_data)

                        # Extract measurement types from documentation
                        if any(
                            term in doc.documentation.lower()
                            for term in [
                                "temperature",
                                "density",
                                "pressure",
                                "magnetic",
                                "electric",
                            ]
                        ):
                            if "temperature" in doc.documentation.lower():
                                ids_info["measurement_types"].add("temperature")
                            if "density" in doc.documentation.lower():
                                ids_info["measurement_types"].add("density")
                            if "pressure" in doc.documentation.lower():
                                ids_info["measurement_types"].add("pressure")
                            if "magnetic" in doc.documentation.lower():
                                ids_info["measurement_types"].add("magnetic_field")
                            if "electric" in doc.documentation.lower():
                                ids_info["measurement_types"].add("electric_field")

                    # Convert sets to lists for JSON serialization
                    ids_info["physics_domains"] = list(ids_info["physics_domains"])
                    ids_info["measurement_types"] = list(ids_info["measurement_types"])

                    export_data["ids_data"][ids_name] = ids_info

                except Exception as e:
                    logger.warning(f"Failed to export IDS {ids_name}: {e}")
                    export_data["ids_data"][ids_name] = {"error": str(e)}

            # Add cross-IDS relationship analysis if requested
            if include_relationships and len(valid_ids) > 1:
                try:
                    relationship_analysis = {}

                    for i, ids1 in enumerate(valid_ids):
                        for ids2 in valid_ids[i + 1 :]:
                            try:
                                # Find relationships between IDS pairs
                                search_results = await self._search_service.search(
                                    query=f"{ids1} {ids2} relationships",
                                    config=SearchConfig(
                                        search_mode=SearchMode.SEMANTIC,
                                        max_results=5,
                                    ),
                                )

                                if search_results:
                                    relationship_analysis[f"{ids1}_{ids2}"] = {
                                        "shared_concepts": len(search_results),
                                        "top_connections": [
                                            {
                                                "path": r.document.metadata.path_name,
                                                "relevance_score": r.score,
                                                "context": r.document.documentation[
                                                    :100
                                                ],
                                            }
                                            for r in search_results[:3]
                                        ],
                                    }
                            except Exception as e:
                                relationship_analysis[f"{ids1}_{ids2}"] = {
                                    "error": str(e)
                                }

                    export_data["cross_relationships"] = relationship_analysis

                except Exception as e:
                    logger.warning(f"Cross-relationship analysis failed: {e}")
                    export_data["cross_relationships"] = {"error": str(e)}

            # Generate export summary
            export_summary = {
                "total_requested": len(ids_list),
                "successfully_exported": len(valid_ids),
                "failed_exports": len(invalid_ids),
                "total_paths_exported": sum(
                    len(ids_data.get("paths", []))
                    for ids_data in export_data["ids_data"].values()
                    if isinstance(ids_data, dict)
                ),
                "export_completeness": "complete" if not invalid_ids else "partial",
            }

            export_data["export_summary"] = export_summary

            # Build final response using Pydantic
            response = IDSExport(
                ids_names=ids_list,
                include_physics=include_physics,
                include_relationships=include_relationships,
                output_format=output_format,
                data=export_data,
                metadata={
                    "export_timestamp": "2024-01-01T00:00:00Z",
                },
            )

            return response

        except Exception as e:
            logger.error(f"Bulk export failed: {e}")
            return ErrorResponse(
                error=str(e),
                suggestions=[
                    "Check IDS names are valid",
                    "Try with fewer IDS names",
                    "Use get_overview to see available IDS",
                    "Try with output_format='structured' for organized data",
                ],
                context={
                    "ids_list": ids_list,
                    "tool": "export_ids",
                    "operation": "bulk_export",
                },
            )

    @cache_results(ttl=900, key_strategy="content_based")  # Cache domain exports
    @validate_input(schema=ExportPhysicsDomainInput)
    @sample(temperature=0.3, max_tokens=1000)  # Balanced creativity for domain export
    @recommend_tools(strategy="domain_export_based", max_tools=3)
    @measure_performance(include_metrics=True, slow_threshold=3.0)
    @handle_errors(fallback="domain_export_suggestions")
    @mcp_tool("Export physics domain-specific data")
    async def export_physics_domain(
        self,
        domain: str,
        include_cross_domain: bool = False,
        analysis_depth: str = "focused",
        max_paths: int = 10,
        ctx: Optional[Context] = None,
    ) -> Union[DomainExport, ErrorResponse]:
        """
        Export physics domain-specific data with sophisticated relationship analysis.

        Advanced domain export tool that extracts comprehensive data for a specific
        physics domain, including cross-domain relationships, measurement dependencies,
        and structural analysis.

        Args:
            domain: Physics domain name to export
            include_cross_domain: Whether to include cross-domain relationship analysis
            analysis_depth: Analysis depth (comprehensive, focused, overview)
            max_paths: Maximum number of paths to include in export (limit: 50)
            ctx: MCP context for AI enhancement

        Returns:
            Dictionary with domain export data, relationships, and AI insights
        """
        try:
            if not domain:
                return DomainExport(
                    domain="",
                    include_cross_domain=include_cross_domain,
                    max_paths=max_paths,
                    output_format="structured",
                    data={
                        "error": "No domain specified for export",
                        "suggestions": [
                            "Provide a physics domain name",
                            "Try: 'core_profiles', 'equilibrium', 'transport'",
                            "Use get_overview() to see available domains",
                        ],
                    },
                )

            # Limit max_paths for performance
            max_paths = min(max_paths, 50)

            # Search for domain-related paths
            search_results = await self._search_service.search(
                query=domain,
                config=SearchConfig(
                    search_mode=SearchMode.SEMANTIC,
                    max_results=max_paths,
                ),
            )

            if not search_results:
                return DomainExport(
                    domain=domain,
                    include_cross_domain=include_cross_domain,
                    max_paths=max_paths,
                    output_format="structured",
                    data={
                        "error": f"No data found for domain '{domain}'",
                        "suggestions": [
                            "Check domain name spelling",
                            "Try broader physics terms",
                            "Use search_imas() to explore available data",
                        ],
                    },
                )

            # Process results based on analysis depth
            domain_paths = []
            related_ids: Set[str] = set()

            for result in search_results:
                path_info = {
                    "path": result.document.metadata.path_name,
                    "documentation": result.document.documentation[:300]
                    if analysis_depth == "comprehensive"
                    else result.document.documentation[:150],
                    "physics_domain": result.document.metadata.physics_domain or "",
                    "data_type": result.document.metadata.data_type or "",
                    "units": result.document.units.unit_str
                    if result.document.units
                    else "",
                }

                # Extract IDS name from path
                if "/" in result.document.metadata.path_name:
                    ids_name = result.document.metadata.path_name.split("/")[0]
                    related_ids.add(ids_name)

                domain_paths.append(path_info)

            # Build final response
            response = DomainExport(
                domain=domain,
                domain_info={
                    "analysis_depth": analysis_depth,
                    "paths": domain_paths,
                    "related_ids": list(related_ids),
                },
                include_cross_domain=include_cross_domain,
                max_paths=max_paths,
                output_format="structured",  # Default format for domain export
                metadata={
                    "total_found": len(domain_paths),
                },
            )

            return response

        except Exception as e:
            logger.error(f"Domain export failed: {e}")
            return ErrorResponse(
                error=str(e),
                suggestions=[
                    "Check domain name spelling",
                    "Try broader physics terms",
                    "Use get_overview() to see available domains",
                ],
                context={
                    "domain": domain,
                    "tool": "export_physics_domain",
                    "operation": "domain_export",
                },
            )
