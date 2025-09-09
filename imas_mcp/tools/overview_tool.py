"""
Overview tool implementation with catalog-based architecture.

This module provides an intelligent interface to the IMAS catalog files,
serving as the primary entry point for users to discover and navigate
the IMAS data dictionary structure.
"""

import importlib.metadata
import importlib.resources
import json
import logging

from fastmcp import Context

from imas_mcp.models.error_models import ToolError
from imas_mcp.models.request_models import OverviewInput
from imas_mcp.models.result_models import OverviewResult
from imas_mcp.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    validate_input,
)

from .base import BaseTool

logger = logging.getLogger(__name__)


class OverviewTool(BaseTool):
    """
    IDS catalog-based overview tool for IMAS data discovery.

    Provides intelligent access to the IDS catalog (ids_catalog.json),
    serving as the primary interface for users to discover relevant
    IDS structures and physics domains.

    Other specialized tools handle:
    - explore_relationships() -> relationships.json
    - explore_identifiers() -> identifier_catalog.json
    """

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "overview_tool"

    def __init__(self, *args, **kwargs):
        """Initialize with IDS catalog data loading."""
        super().__init__(*args, **kwargs)
        self._ids_catalog = {}
        self._identifier_catalog = {}
        self._load_ids_catalog()
        self._load_identifier_catalog()

    def _load_ids_catalog(self):
        """Load the IDS catalog file specifically."""
        try:
            try:
                catalog_file = (
                    importlib.resources.files("imas_mcp.resources.schemas")
                    / "ids_catalog.json"
                )
                with catalog_file.open("r", encoding="utf-8") as f:
                    self._ids_catalog = json.load(f)
                    logger.info("Loaded IDS catalog for overview tool")
            except FileNotFoundError:
                logger.warning(
                    "IDS catalog (ids_catalog.json) not found in resources/schemas/"
                )

        except Exception as e:
            logger.error(f"Failed to load IDS catalog: {e}")
            self._ids_catalog = {}

    def _load_identifier_catalog(self):
        """Load the identifier catalog file."""
        try:
            catalog_file = (
                importlib.resources.files("imas_mcp.resources.schemas")
                / "identifier_catalog.json"
            )
            with catalog_file.open("r", encoding="utf-8") as f:
                self._identifier_catalog = json.load(f)
                logger.info("Loaded identifier catalog for overview tool")
        except Exception as e:
            logger.warning(f"Failed to load identifier catalog: {e}")
            self._identifier_catalog = {}

    def _get_mcp_tools(self) -> list[str]:
        """Get list of available MCP tools on this server."""
        return [
            "search_imas",
            "explain_concept",
            "get_overview",
            "analyze_ids_structure",
            "explore_relationships",
            "explore_identifiers",
            "export_ids",
            "export_physics_domain",
        ]

    def _get_physics_domains(self) -> dict[str, list[str]]:
        """Get all physics domains and their associated IDS."""
        if not self._ids_catalog:
            return {}

        domains = {}
        ids_catalog = self._ids_catalog.get("ids_catalog", {})

        for ids_name, ids_info in ids_catalog.items():
            physics_domain = ids_info.get("physics_domain", "unclassified")
            if physics_domain not in domains:
                domains[physics_domain] = []
            domains[physics_domain].append(ids_name)

        return domains

    def _filter_ids_by_query(self, query: str) -> list[str]:
        """Filter IDS based on query terms."""
        if not self._ids_catalog:
            return []

        query_lower = query.lower()
        relevant_ids = []
        ids_catalog = self._ids_catalog.get("ids_catalog", {})

        for ids_name, ids_info in ids_catalog.items():
            # Check name match
            if query_lower in ids_name.lower():
                relevant_ids.append(ids_name)
                continue

            # Check description match
            description = ids_info.get("description", "").lower()
            if query_lower in description:
                relevant_ids.append(ids_name)
                continue

            # Check physics domain match
            physics_domain = ids_info.get("physics_domain", "").lower()
            if query_lower in physics_domain:
                relevant_ids.append(ids_name)
                continue

        return relevant_ids

    def _get_complexity_rankings(self) -> list[tuple]:
        """Get IDS ranked by complexity (path count)."""
        if not self._ids_catalog:
            return []

        ids_catalog = self._ids_catalog.get("ids_catalog", {})
        rankings = []

        for ids_name, ids_info in ids_catalog.items():
            path_count = ids_info.get("path_count", 0)
            rankings.append((ids_name, path_count))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def _generate_recommendations(
        self, query: str | None, relevant_ids: list[str]
    ) -> list[str]:
        """Generate tool usage recommendations based on context."""
        recommendations = []

        if query:
            recommendations.append(
                f"🔍 Use search_imas('{query}') to find specific data paths"
            )

            # Domain-specific recommendations
            if any(
                term in query.lower() for term in ["magnetic", "field", "equilibrium"]
            ):
                recommendations.append(
                    "⚡ Try explore_relationships('equilibrium/time_slice/profiles_2d') for magnetic field data"
                )

            if any(
                term in query.lower() for term in ["temperature", "density", "pressure"]
            ):
                recommendations.append(
                    "🌡️ Use export_physics_domain('transport') for temperature and density profiles"
                )

            if any(term in query.lower() for term in ["diagnostic", "measurement"]):
                recommendations.append(
                    "📊 Try analyze_ids_structure() on diagnostic IDS like 'thomson_scattering'"
                )

            if any(
                term in query.lower() for term in ["identifier", "enum", "enumeration"]
            ):
                recommendations.append(
                    "🔢 Use explore_identifiers() to browse identifier schemas and enumerations"
                )

        if relevant_ids:
            # Recommend tools based on number of relevant IDS
            match len(relevant_ids):
                case 1:
                    recommendations.append(
                        f"� Use export_ids(['{relevant_ids[0]}']) to extract this IDS data"
                    )
                case 2:
                    recommendations.append(
                        f"� Use analyze_ids_structure('{relevant_ids[0]}') for detailed structure analysis"
                    )
                case 3:
                    recommendations.append(
                        "🌐 Use explore_relationships() to find connections between these IDS"
                    )

        # Always include general recommendations
        recommendations.extend(
            [
                "💡 Use explain_concept() to understand physics concepts",
                "🔗 Use explore_identifiers() to browse available enumerations",
                "🌐 Use explore_relationships() to find data connections",
                "📈 Use export_physics_domain() for domain-specific data exports",
            ]
        )

        return recommendations[:6]  # Limit to 6 recommendations

    @cache_results(ttl=3600, key_strategy="semantic")
    @validate_input(schema=OverviewInput)
    @handle_errors(fallback="overview_suggestions")
    @mcp_tool("Get high-level overview of IMAS data dictionary structure and contents")
    async def get_overview(
        self,
        query: str | None = None,
        ctx: Context | None = None,
    ) -> OverviewResult | ToolError:
        """
        Get high-level overview of IMAS data dictionary structure and contents.

        Primary orientation tool for understanding available IDS, physics domains,
        and data organization. Use this tool first to explore what data is available
        and get recommendations for specific analysis workflows.

        Args:
            query: Optional focus area (physics domain, measurement type, or IDS name)
            ctx: MCP context for potential AI enhancement

        Returns:
            OverviewResult with available IDS, physics domains, and usage guidance
        """
        try:
            # Check if IDS catalog is loaded
            if not self._ids_catalog:
                return ToolError(
                    error="IDS catalog data not available",
                    suggestions=[
                        "Check if ids_catalog.json exists in resources/schemas/",
                        "Try restarting the MCP server",
                        "Use search_imas() for direct data access",
                    ],
                    context={"tool": "get_overview", "operation": "catalog_access"},
                )

            # Get basic metadata
            metadata = self._ids_catalog.get("metadata", {})
            ids_info = self._ids_catalog.get("ids_catalog", {})
            identifier_metadata = self._identifier_catalog.get("metadata", {})

            # Use catalog metadata as defaults
            total_ids = metadata.get("total_ids", len(ids_info))
            total_paths = metadata.get("total_paths", 0)
            total_leaf_nodes = metadata.get("total_leaf_nodes")
            dd_version = metadata.get("version")
            generation_date = metadata.get("generation_date")
            identifier_schemas_count = identifier_metadata.get("total_ids", 0)
            mcp_tools = self._get_mcp_tools()

            # Get MCP server version
            try:
                version = importlib.metadata.version("imas-mcp")
            except importlib.metadata.PackageNotFoundError:
                version = "0.0.0"

            # Get filtered IDS list from DocumentStore (respects ids_set filter)
            available_ids = self.document_store._get_available_ids()

            # If the server is running with an IDS filter, adjust the stats to reflect the filtered view
            if available_ids and len(available_ids) < len(ids_info):
                # Recompute totals for the filtered set
                total_ids = len(available_ids)
                try:
                    total_paths = sum(
                        ids_info.get(ids_name, {}).get("path_count", 0)
                        for ids_name in available_ids
                    )
                except Exception:
                    # Fall back to catalog total if anything goes wrong
                    total_paths = metadata.get("total_paths", 0)

            # Determine relevant IDS based on query
            if query:
                relevant_ids = self._filter_ids_by_query(query)
                # Filter query results to only include available IDS
                relevant_ids = [ids for ids in relevant_ids if ids in available_ids]
                if not relevant_ids:
                    # Fallback to showing available IDS if no matches
                    relevant_ids = available_ids
                    query_feedback = (
                        f"No direct matches for '{query}' - showing available overview"
                    )
                else:
                    query_feedback = f"Found {len(relevant_ids)} IDS matching '{query}'"
            else:
                relevant_ids = available_ids
                if len(available_ids) < len(ids_info):
                    query_feedback = (
                        f"Filtered IMAS overview ({len(available_ids)} IDS)"
                    )
                else:
                    query_feedback = "General IMAS overview"

            # Sort relevant_ids by complexity (path count) in descending order
            # This helps with LLM primacy bias - larger/more important IDS appear first
            relevant_ids.sort(
                key=lambda ids_name: ids_info.get(ids_name, {}).get("path_count", 0),
                reverse=True,
            )

            # Get physics domain breakdown
            physics_domains = self._get_physics_domains()
            domain_summary = {
                domain: len(ids_list) for domain, ids_list in physics_domains.items()
            }

            # Get complexity rankings for context
            complexity_rankings = self._get_complexity_rankings()

            # Build IDS statistics for relevant IDS
            ids_statistics = {}
            physics_domains_found = set()

            for ids_name in relevant_ids:
                if ids_name in ids_info:
                    ids_data = ids_info[ids_name]
                    ids_statistics[ids_name] = {
                        "path_count": ids_data.get("path_count", 0),
                        "description": ids_data.get("description", f"{ids_name} IDS"),
                        "physics_domain": ids_data.get(
                            "physics_domain", "unclassified"
                        ),
                    }
                    physics_domains_found.add(
                        ids_data.get("physics_domain", "unclassified")
                    )

            # Generate usage recommendations
            recommendations = self._generate_recommendations(query, relevant_ids)

            # Build content summary
            content_parts = [
                f"🔬 **{query_feedback}**",
                f"📊 **Dataset Statistics**: {total_ids} IDS with {total_paths:,} total data paths",
            ]

            # Add version information if available
            if version:
                content_parts.append(f"🚀 **MCP Server Version**: {version}")
            if dd_version:
                content_parts.append(f"🏷️ **DD Version**: {dd_version}")

            # Add additional metadata
            if total_leaf_nodes:
                content_parts.append(
                    f"🔢 **Data Elements**: {total_leaf_nodes:,} individual data elements"
                )

            if identifier_schemas_count and identifier_schemas_count > 0:
                content_parts.append(
                    f"🔣 **Identifier Schemas**: {identifier_schemas_count} available types"
                )

            if mcp_tools:
                content_parts.append(
                    f"🛠️ **MCP Tools**: {len(mcp_tools)} tools available"
                )

            if query:
                content_parts.append(
                    f"🎯 **Relevant IDS**: {len(relevant_ids)} structures match your query"
                )

            if physics_domains_found:
                domain_list = ", ".join(sorted(physics_domains_found))
                content_parts.append(f"🧪 **Physics Domains**: {domain_list}")

            # Add top domains summary for general overview
            if not query and domain_summary:
                top_domains = sorted(
                    domain_summary.items(), key=lambda x: x[1], reverse=True
                )[:5]
                domain_text = ", ".join(
                    [f"{domain} ({count})" for domain, count in top_domains]
                )
                content_parts.append(f"📈 **Top Physics Domains**: {domain_text}")

            # Add complexity insights
            if complexity_rankings:
                most_complex = complexity_rankings[0]
                least_complex = complexity_rankings[-1]
                content_parts.append(
                    f"🏗️ **Complexity Range**: {least_complex[0]} ({least_complex[1]} paths) → "
                    f"{most_complex[0]} ({most_complex[1]} paths)"
                )

            content_parts.extend(["", "**🚀 Recommended Next Steps:**"])
            content_parts.extend([f"  {rec}" for rec in recommendations])

            # Build usage guidance
            usage_guidance = {
                "tools_available": [
                    "search_imas - Find specific data paths with semantic search",
                    "analyze_ids_structure - Get detailed structural analysis of any IDS",
                    "explain_concept - Understand physics concepts and terminology",
                    "explore_relationships - Discover data connections and cross-references (uses relationships.json)",
                    "explore_identifiers - Browse identifier schemas and enumerations (uses identifier_catalog.json)",
                    "export_ids - Extract data from multiple IDS simultaneously",
                    "export_physics_domain - Get domain-specific data exports",
                ],
                "getting_started": recommendations,
                "catalog_focus": "This tool serves the IDS catalog - use explore_relationships() and explore_identifiers() for other catalog data",
            }

            return OverviewResult(
                content="\n".join(content_parts),
                available_ids=relevant_ids,
                query=query,
                physics_domains=list(physics_domains_found),
                ids_statistics=ids_statistics,
                usage_guidance=usage_guidance,
                mcp_version=version,
                dd_version=dd_version,
                generation_date=generation_date,
                total_leaf_nodes=total_leaf_nodes,
                identifier_schemas_count=identifier_schemas_count,
                mcp_tools=mcp_tools,
            )

        except Exception as e:
            logger.error(f"Catalog-based overview generation failed: {e}")
            return ToolError(
                error=str(e),
                suggestions=[
                    "Try a simpler query or general overview",
                    "Use search_imas() for direct data exploration",
                    "Check catalog file availability",
                ],
                context={
                    "query": query,
                    "tool": "get_overview",
                    "operation": "catalog_overview",
                    "ids_catalog_loaded": bool(self._ids_catalog),
                },
            )
