"""
Export tool implementation.

This module contains the export_ids and export_physics_domain tool logic
with decorators for caching, validation, AI enhancement, tool recommendations,
performance monitoring, and error handling.
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp.models.constants import OutputFormat, SearchMode
from imas_mcp.models.error_models import ToolError
from imas_mcp.models.request_models import (
    ExportIdsInput,
    ExportPhysicsDomainInput,
)
from imas_mcp.models.result_models import (
    DomainExport,
    ExportData,
    IDSExport,
)

# Import export-appropriate decorators
from imas_mcp.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    measure_performance,
    tool_hints,
    validate_input,
)

from .base import BaseTool

logger = logging.getLogger(__name__)


class ExportTool(BaseTool):
    """Tool for exporting IDS and physics domain data using service composition."""

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "export_tools"

    def _extract_identifier_info(self, doc) -> dict[str, Any]:
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

    @cache_results(ttl=600, key_strategy="content_based")
    @validate_input(schema=ExportIdsInput)
    @measure_performance(include_metrics=True, slow_threshold=5.0)
    @handle_errors(fallback="export_suggestions")
    @tool_hints(max_hints=2)
    @mcp_tool(
        "Extract comprehensive data from multiple IDS for analysis and comparison"
    )
    async def export_ids(
        self,
        ids_list: list[str],
        include_relationships: bool = True,
        include_physics: bool = True,
        output_format: str = "structured",
        ctx: Context | None = None,
    ) -> IDSExport | ToolError:
        """
        Extract comprehensive data from multiple IDS for analysis and comparison.

        Bulk data extraction tool that retrieves all data paths, documentation,
        and metadata from specified IDS. Useful for comparative analysis,
        cross-IDS workflows, and comprehensive data exploration.

        Args:
            ids_list: List of IDS names to export (e.g., ['equilibrium', 'transport'])
            include_relationships: Include cross-IDS relationship analysis
            include_physics: Include physics domain context and insights
            output_format: Export format - structured, json, yaml, or markdown
            ctx: MCP context for AI enhancement

        Returns:
            IDSExport with comprehensive data, relationships, and analysis guidance
        """
        try:
            if not ids_list:
                return self._create_error_response(
                    "No IDS specified for bulk export", "export_ids"
                )

            # Validate format
            valid_formats = [format.value for format in OutputFormat]
            if output_format not in valid_formats:
                return self._create_error_response(
                    f"Invalid format: {output_format}. Use: {', '.join(valid_formats)}",
                    "export_ids",
                )

            # Validate IDS names using document service
            valid_ids, invalid_ids = await self.documents.validate_ids(ids_list)
            if not valid_ids:
                return self.documents.create_ids_not_found_error(
                    str(ids_list), self.tool_name
                )

            # Process export data
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
                    ids_documents = await self.documents.get_documents_safe(ids_name)
                    ids_info = {
                        "ids_name": ids_name,
                        "total_paths": len(ids_documents),
                        "paths": [],
                        "physics_domains": set(),
                        "identifier_paths": [],
                        "measurement_types": set(),
                    }

                    for doc in ids_documents:
                        path_data = {
                            "path": doc.metadata.path_name,
                            "documentation": doc.documentation
                            if output_format == "enhanced"
                            else doc.documentation[:200],
                            "data_type": doc.metadata.data_type,
                            "physics_domain": doc.metadata.physics_domain,
                            "units": doc.metadata.units,
                        }

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

                    # Convert sets to lists
                    ids_info["physics_domains"] = list(ids_info["physics_domains"])
                    ids_info["measurement_types"] = list(ids_info["measurement_types"])
                    export_data["ids_data"][ids_name] = ids_info

                except Exception as e:
                    logger.warning(f"Failed to export IDS {ids_name}: {e}")
                    export_data["ids_data"][ids_name] = {"error": str(e)}

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

            # Build response with proper structure
            export_data_obj = ExportData(
                ids_data=export_data.get("ids_data", {}),
                cross_relationships=export_data.get("cross_relationships", {}),
                export_summary=export_data.get("export_summary", {}),
            )

            # Build final response
            export_result = IDSExport(
                ids_names=ids_list,
                include_physics=include_physics,
                include_relationships=include_relationships,
                data=export_data_obj,
                metadata={
                    "export_timestamp": "2024-01-01T00:00:00Z",
                },
            )

            logger.info(f"Bulk export completed for {len(valid_ids)} IDS")
            return export_result

        except Exception as e:
            logger.error(f"Bulk export failed: {e}")
            return self._create_error_response(
                f"Bulk export failed: {e}", str(ids_list)
            )

    @cache_results(ttl=900, key_strategy="content_based")
    @validate_input(schema=ExportPhysicsDomainInput)
    @measure_performance(include_metrics=True, slow_threshold=3.0)
    @handle_errors(fallback="domain_export_suggestions")
    @tool_hints(max_hints=2)
    @mcp_tool(
        "Extract all data related to a specific physics domain across multiple IDS"
    )
    async def export_physics_domain(
        self,
        domain: str,
        include_cross_domain: bool = False,
        analysis_depth: str = "focused",
        max_paths: int = 10,
        ctx: Context | None = None,
    ) -> DomainExport | ToolError:
        """
        Extract all data related to a specific physics domain across multiple IDS.

        Domain-focused extraction tool that gathers measurements, calculations,
        and diagnostics from a particular physics area (e.g., 'equilibrium',
        'transport', 'heating'). Ideal for domain-specific research workflows.

        Args:
            domain: Physics domain name (e.g., 'transport', 'equilibrium', 'heating')
            include_cross_domain: Include connections to related physics domains
            analysis_depth: Detail level - comprehensive, focused, or overview
            max_paths: Maximum data paths to include (limit: 50)
            ctx: MCP context for AI enhancement

        Returns:
            DomainExport with domain-specific data, relationships, and research guidance
        """
        try:
            if not domain:
                return self._create_error_response(
                    "No domain specified for export", domain
                )

            # Limit max_paths for performance
            max_paths = min(max_paths, 50)

            # Execute search using the base search method
            search_result = await self.execute_search(
                query=domain, search_mode=SearchMode.SEMANTIC, max_results=max_paths
            )

            search_results = search_result.hits

            if not search_results:
                return self._create_error_response(
                    f"No data found for domain '{domain}'", domain
                )

            # Process results based on analysis depth
            domain_paths = []
            related_ids: set[str] = set()

            for result in search_results:
                path_info = {
                    "path": result.path,
                    "documentation": result.documentation[:300]
                    if analysis_depth == "comprehensive"
                    else result.documentation[:150],
                    "physics_domain": result.physics_domain or "",
                    "data_type": result.data_type or "",
                    "units": result.units or "",
                }

                # Extract IDS name from path or use the ids_name field
                if "/" in result.path:
                    ids_name = result.path.split("/")[0]
                    related_ids.add(ids_name)
                elif result.ids_name:
                    related_ids.add(result.ids_name)

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
                metadata={
                    "total_found": len(domain_paths),
                },
            )

            logger.info(f"Domain export completed for: {domain}")
            return response

        except Exception as e:
            logger.error(f"Domain export failed: {e}")
            return self._create_error_response(f"Domain export failed: {e}", domain)
