"""
Export tool implementation with clustering-enhanced analysis.

This module contains the export_ids and export_physics_domain tool logic
with decorators for caching, validation, AI enhancement, tool recommendations,
performance monitoring, and error handling. Enhanced with clustering analysis
to provide better domain insights and cross-IDS relationship discovery.
"""

import logging
from datetime import UTC, datetime
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
from imas_mcp.physics.domain_analyzer import PhysicsDomainAnalyzer

# Import export-appropriate decorators
from imas_mcp.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    measure_performance,
    validate_input,
)
from imas_mcp.search.decorators.tool_hints import tool_hints

from .base import BaseTool

logger = logging.getLogger(__name__)


class ExportTool(BaseTool):
    """Tool for exporting IDS and physics domain data using service composition."""

    def __init__(self, *args, **kwargs):
        """Initialize export tool with domain analyzer and relationships data."""
        super().__init__(*args, **kwargs)
        self._domain_analyzer = PhysicsDomainAnalyzer()
        self._relationships_data = {}
        self._load_relationships_data()

    def _load_relationships_data(self):
        """Load relationships data for clustering-enhanced analysis."""
        try:
            import importlib.resources
            import json

            relationships_file = (
                importlib.resources.files("imas_mcp.resources.schemas")
                / "relationships.json"
            )
            with relationships_file.open("r", encoding="utf-8") as f:
                self._relationships_data = json.load(f)
                logger.info("Loaded relationships data for enhanced export analysis")
        except Exception as e:
            logger.warning(f"Failed to load relationships data for export tool: {e}")
            self._relationships_data = {}

    def _enhance_domain_with_clustering(
        self, domain: str, domain_paths: list[dict]
    ) -> dict:
        """Enhance domain analysis with clustering information."""
        if not self._relationships_data:
            return {}

        clusters = self._relationships_data.get("clusters", [])
        unit_families = self._relationships_data.get("unit_families", {})

        enhancement = {
            "clustering_analysis": {
                "relevant_clusters": [],
                "cross_ids_connections": [],
                "intra_ids_connections": [],
                "unit_family_coverage": {},
            }
        }

        # Find paths that appear in clusters
        domain_path_strings = [p["path"] for p in domain_paths]

        for cluster in clusters:
            cluster_paths = cluster.get("paths", [])
            matching_paths = [p for p in domain_path_strings if p in cluster_paths]

            if matching_paths:
                cluster_info = {
                    "cluster_id": cluster["id"],
                    "similarity_score": cluster.get("similarity_score", 0.0),
                    "is_cross_ids": cluster.get("is_cross_ids", False),
                    "cluster_size": cluster.get("size", 0),
                    "ids_names": cluster.get("ids_names", []),
                    "matching_paths": matching_paths,
                    "coverage_ratio": len(matching_paths) / len(cluster_paths)
                    if cluster_paths
                    else 0,
                }

                enhancement["clustering_analysis"]["relevant_clusters"].append(
                    cluster_info
                )

                # Categorize connections
                if cluster.get("is_cross_ids", False):
                    enhancement["clustering_analysis"]["cross_ids_connections"].extend(
                        matching_paths
                    )
                else:
                    enhancement["clustering_analysis"]["intra_ids_connections"].extend(
                        matching_paths
                    )

        # Check unit family coverage
        for unit_name, unit_data in unit_families.items():
            unit_paths = unit_data.get("paths_using", [])
            matching_unit_paths = [p for p in domain_path_strings if p in unit_paths]

            if matching_unit_paths:
                enhancement["clustering_analysis"]["unit_family_coverage"][
                    unit_name
                ] = {
                    "matching_paths": matching_unit_paths,
                    "total_family_size": len(unit_paths),
                    "coverage_ratio": len(matching_unit_paths) / len(unit_paths)
                    if unit_paths
                    else 0,
                }

        # Add summary statistics
        enhancement["clustering_analysis"]["summary"] = {
            "total_relevant_clusters": len(
                enhancement["clustering_analysis"]["relevant_clusters"]
            ),
            "cross_ids_cluster_count": len(
                [
                    c
                    for c in enhancement["clustering_analysis"]["relevant_clusters"]
                    if c["is_cross_ids"]
                ]
            ),
            "intra_ids_cluster_count": len(
                [
                    c
                    for c in enhancement["clustering_analysis"]["relevant_clusters"]
                    if not c["is_cross_ids"]
                ]
            ),
            "unit_families_covered": len(
                enhancement["clustering_analysis"]["unit_family_coverage"]
            ),
            "avg_cluster_similarity": sum(
                c["similarity_score"]
                for c in enhancement["clustering_analysis"]["relevant_clusters"]
            )
            / len(enhancement["clustering_analysis"]["relevant_clusters"])
            if enhancement["clustering_analysis"]["relevant_clusters"]
            else 0.0,
        }

        return enhancement

    def _enhance_export_with_clustering(
        self, export_data: dict, valid_ids: list[str]
    ) -> dict:
        """Enhance IDS export with clustering-based cross-relationships."""
        if not self._relationships_data:
            return {}

        clusters = self._relationships_data.get("clusters", [])
        unit_families = self._relationships_data.get("unit_families", {})

        cross_relationships = {
            "clustering_analysis": {
                "shared_clusters": [],
                "cross_ids_clusters": [],
                "unit_family_connections": {},
                "relationship_matrix": {},
            }
        }

        # Collect all paths from exported IDS
        all_exported_paths = []
        ids_paths_map = {}
        for ids_name, ids_data in export_data.get("ids_data", {}).items():
            if isinstance(ids_data, dict) and "paths" in ids_data:
                ids_paths = [p["path"] for p in ids_data["paths"]]
                ids_paths_map[ids_name] = ids_paths
                all_exported_paths.extend(ids_paths)

        # Find clusters that connect the exported IDS
        for cluster in clusters:
            cluster_paths = cluster.get("paths", [])
            matching_paths = [p for p in all_exported_paths if p in cluster_paths]

            if len(matching_paths) > 1:  # Only include clusters with multiple matches
                # Check which IDS are connected by this cluster
                connected_ids = []
                for ids_name, ids_paths in ids_paths_map.items():
                    if any(p in cluster_paths for p in ids_paths):
                        connected_ids.append(ids_name)

                if len(connected_ids) > 1:  # Cross-IDS cluster
                    cluster_info = {
                        "cluster_id": cluster["id"],
                        "similarity_score": cluster.get("similarity_score", 0.0),
                        "is_cross_ids": cluster.get("is_cross_ids", False),
                        "connected_ids": connected_ids,
                        "connecting_paths": matching_paths,
                        "cluster_size": cluster.get("size", 0),
                    }

                    if cluster.get("is_cross_ids", False):
                        cross_relationships["clustering_analysis"][
                            "cross_ids_clusters"
                        ].append(cluster_info)
                    else:
                        cross_relationships["clustering_analysis"][
                            "shared_clusters"
                        ].append(cluster_info)

        # Analyze unit family connections between IDS
        for unit_name, unit_data in unit_families.items():
            unit_paths = unit_data.get("paths_using", [])
            connected_ids_for_unit = []

            for ids_name, ids_paths in ids_paths_map.items():
                matching_unit_paths = [p for p in ids_paths if p in unit_paths]
                if matching_unit_paths:
                    connected_ids_for_unit.append(
                        {"ids_name": ids_name, "matching_paths": matching_unit_paths}
                    )

            if len(connected_ids_for_unit) > 1:
                cross_relationships["clustering_analysis"]["unit_family_connections"][
                    unit_name
                ] = connected_ids_for_unit

        # Build relationship matrix
        for i, ids1 in enumerate(valid_ids):
            for ids2 in valid_ids[i + 1 :]:
                relationship_strength = 0.0
                connection_types = []

                # Check clustering connections
                for cluster_info in (
                    cross_relationships["clustering_analysis"]["cross_ids_clusters"]
                    + cross_relationships["clustering_analysis"]["shared_clusters"]
                ):
                    if (
                        ids1 in cluster_info["connected_ids"]
                        and ids2 in cluster_info["connected_ids"]
                    ):
                        relationship_strength = max(
                            relationship_strength, cluster_info["similarity_score"]
                        )
                        connection_types.append(f"cluster_{cluster_info['cluster_id']}")

                # Check unit family connections
                for unit_name, unit_connections in cross_relationships[
                    "clustering_analysis"
                ]["unit_family_connections"].items():
                    connected_ids = [conn["ids_name"] for conn in unit_connections]
                    if ids1 in connected_ids and ids2 in connected_ids:
                        relationship_strength = max(
                            relationship_strength, 0.4
                        )  # Moderate strength for unit connections
                        connection_types.append(f"unit_{unit_name}")

                if relationship_strength > 0:
                    cross_relationships["clustering_analysis"]["relationship_matrix"][
                        f"{ids1}-{ids2}"
                    ] = {
                        "strength": relationship_strength,
                        "connection_types": connection_types,
                    }

        # Add summary statistics
        cross_relationships["clustering_analysis"]["summary"] = {
            "total_shared_clusters": len(
                cross_relationships["clustering_analysis"]["shared_clusters"]
            ),
            "total_cross_ids_clusters": len(
                cross_relationships["clustering_analysis"]["cross_ids_clusters"]
            ),
            "unit_families_connecting_ids": len(
                cross_relationships["clustering_analysis"]["unit_family_connections"]
            ),
            "ids_pairs_with_relationships": len(
                cross_relationships["clustering_analysis"]["relationship_matrix"]
            ),
        }

        return cross_relationships

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
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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

            # Add clustering-based cross-relationships if requested
            if include_relationships:
                clustering_relationships = self._enhance_export_with_clustering(
                    export_data, valid_ids
                )
                export_data["cross_relationships"].update(clustering_relationships)

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
                    "export_timestamp": datetime.now(UTC)
                    .isoformat()
                    .replace("+00:00", "Z"),
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
        'transport', 'heating'). Provides comprehensive analysis including
        theoretical foundations, experimental methods, and cross-domain relationships.

        Args:
            domain: Physics domain name (e.g., 'transport', 'equilibrium', 'heating')
            include_cross_domain: Include connections to related physics domains
            analysis_depth: Detail level - comprehensive, focused, or overview
            max_paths: Maximum data paths to include (limit: 50)
            ctx: MCP context for AI enhancement

        Returns:
            DomainExport with comprehensive domain analysis, measurements, workflows, and guidance
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

            search_results_list = search_result.hits

            if not search_results_list:
                return self._create_error_response(
                    f"No data found for domain '{domain}'", domain
                )

            # Use enhanced domain analyzer for comprehensive analysis
            domain_analysis = self._domain_analyzer.analyze_domain(
                domain=domain, search_results=search_results_list, depth=analysis_depth
            )

            # Process results with enhanced path extraction
            domain_paths = []
            related_ids: set[str] = set()

            for result in search_results_list:
                path_info = {
                    "path": result.path,
                    "documentation": result.documentation[:300]
                    if analysis_depth == "comprehensive"
                    else result.documentation[:150],
                    "physics_domain": result.physics_domain or "",
                    "data_type": result.data_type or "",
                    "units": result.units or "",
                    "measurement_type": self._classify_measurement_type(result),
                }

                # Extract IDS name from path or use the ids_name field
                if "/" in result.path:
                    ids_name = result.path.split("/")[0]
                    related_ids.add(ids_name)
                elif result.ids_name:
                    related_ids.add(result.ids_name)

                domain_paths.append(path_info)

            # Build enhanced domain information
            domain_info = {
                "analysis_depth": analysis_depth,
                "paths": domain_paths,
                "related_ids": list(related_ids),
                # Enhanced analysis components
                "key_measurements": domain_analysis.get("key_measurements", []),
                "theoretical_foundations": domain_analysis.get(
                    "theoretical_foundations", {}
                ),
                "experimental_methods": domain_analysis.get("experimental_methods", []),
                "typical_workflows": domain_analysis.get("typical_workflows", []),
                "data_characteristics": domain_analysis.get("data_characteristics", {}),
                "complexity_assessment": domain_analysis.get(
                    "complexity_assessment", {}
                ),
            }

            # Add clustering enhancement to domain analysis
            clustering_enhancement = self._enhance_domain_with_clustering(
                domain, domain_paths
            )
            if clustering_enhancement:
                domain_info.update(clustering_enhancement)

            # Add cross-domain analysis if requested
            if include_cross_domain:
                domain_info["cross_domain_links"] = domain_analysis.get(
                    "cross_domain_links", []
                )
                domain_info["measurement_integration"] = domain_analysis.get(
                    "measurement_integration", {}
                )

            # Add comprehensive features for detailed analysis
            if analysis_depth == "comprehensive":
                domain_info.update(
                    {
                        "detailed_physics_context": domain_analysis.get(
                            "detailed_physics_context", {}
                        ),
                        "research_applications": domain_analysis.get(
                            "research_applications", []
                        ),
                        "data_quality_assessment": domain_analysis.get(
                            "data_quality_assessment", {}
                        ),
                    }
                )

            # Build final response with enhanced metadata
            response = DomainExport(
                domain=domain,
                domain_info=domain_info,
                include_cross_domain=include_cross_domain,
                max_paths=max_paths,
                metadata={
                    "total_found": len(domain_paths),
                    "analysis_timestamp": datetime.now(UTC)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "analysis_engine": "PhysicsDomainAnalyzer",
                    "enhancement_features": [
                        "theoretical_foundations",
                        "experimental_methods",
                        "measurement_classification",
                        "workflow_analysis",
                        "complexity_assessment",
                        "clustering_analysis",
                        "unit_family_analysis",
                    ]
                    + (["cross_domain_analysis"] if include_cross_domain else []),
                },
            )

            logger.info(
                f"Enhanced domain export completed for: {domain} ({analysis_depth} depth)"
            )
            return response

        except Exception as e:
            logger.error(f"Enhanced domain export failed: {e}")
            return self._create_error_response(f"Domain export failed: {e}", domain)

    def _classify_measurement_type(self, result: Any) -> str:
        """Classify measurement type for enhanced path information."""
        path = result.path.lower()

        # Physics-based measurement classification
        if "density" in path:
            return "density_measurement"
        elif "temperature" in path:
            return "temperature_measurement"
        elif "pressure" in path:
            return "pressure_measurement"
        elif "magnetic" in path or "field" in path:
            return "magnetic_field_measurement"
        elif "current" in path:
            return "current_measurement"
        elif "velocity" in path or "flow" in path:
            return "velocity_measurement"
        elif "radiation" in path or "emission" in path:
            return "radiation_measurement"
        elif "position" in path or "geometry" in path:
            return "geometric_measurement"
        else:
            return (
                "general_measurement"  # Always return measurement type, not data_type
            )
