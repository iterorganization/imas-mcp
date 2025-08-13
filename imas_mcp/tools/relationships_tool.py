"""
Relationships tool implementation with enhanced semantic analysis.

This module provides an intelligent interface to the relationships catalog,
serving as the primary entry point for users to discover and navigate
data relationships and cross-references in the IMAS data dictionary.
Enhanced with semantic analysis, physics domain mapping, and strength scoring.
"""

import importlib.resources
import json
import logging
from typing import Any

from fastmcp import Context

from imas_mcp.core.data_model import IdsNode, PhysicsContext
from imas_mcp.models.constants import RelationshipType
from imas_mcp.models.error_models import ToolError
from imas_mcp.models.request_models import RelationshipsInput
from imas_mcp.models.result_models import RelationshipResult
from imas_mcp.physics_extraction.relationship_engine import (
    EnhancedRelationshipEngine,
    create_enhanced_relationship_nodes,
)
from imas_mcp.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    validate_input,
)

from .base import BaseTool

logger = logging.getLogger(__name__)


class RelationshipsTool(BaseTool):
    """
    Enhanced relationships catalog-based tool for IMAS data relationship discovery.

    Provides intelligent access to the relationships catalog (relationships.json)
    with advanced semantic analysis, physics domain mapping, and strength-based
    scoring for enhanced relationship discovery.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with relationships catalog data loading and enhanced engine."""
        super().__init__(*args, **kwargs)
        self._relationships_catalog = {}
        self._enhanced_engine = None
        self._load_relationships_catalog()

    def _load_relationships_catalog(self):
        """Load the relationships catalog file and initialize enhanced engine."""
        try:
            try:
                catalog_file = (
                    importlib.resources.files("imas_mcp.resources.schemas")
                    / "relationships.json"
                )
                with catalog_file.open("r", encoding="utf-8") as f:
                    self._relationships_catalog = json.load(f)

                # Initialize enhanced relationship engine
                if self._relationships_catalog:
                    self._enhanced_engine = EnhancedRelationshipEngine(
                        self._relationships_catalog
                    )
                    logger.info(
                        "Loaded relationships catalog with enhanced engine for relationships tool"
                    )
                else:
                    logger.warning("Empty relationships catalog loaded")

            except FileNotFoundError:
                logger.warning(
                    "Relationships catalog (relationships.json) not found in resources/schemas/"
                )

        except Exception as e:
            logger.error(f"Failed to load relationships catalog: {e}")
            self._relationships_catalog = {}
            self._enhanced_engine = None

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "explore_relationships"

    def _find_related_paths(
        self, path: str, relationship_type: RelationshipType, max_depth: int
    ) -> list[dict]:
        """Find related paths from the catalog based on the input path."""
        if not self._relationships_catalog:
            return []

        cross_references = self._relationships_catalog.get("cross_references", {})
        physics_concepts = self._relationships_catalog.get("physics_concepts", {})
        unit_families = self._relationships_catalog.get("unit_families", {})
        related_paths = []

        # Direct match in cross_references
        if path in cross_references:
            relationships = cross_references[path].get("relationships", [])
            for rel in relationships[: max_depth * 5]:  # Limit results
                related_paths.append(
                    {
                        "path": rel.get("path", ""),
                        "type": rel.get("type", ""),
                        "distance": 1,
                    }
                )

        # Direct match in physics_concepts
        if path in physics_concepts:
            physics_data = physics_concepts[path]
            # Extract relationships from relevant_paths and key_relationships
            relevant_paths = physics_data.get("relevant_paths", [])
            key_relationships = physics_data.get("key_relationships", [])

            for rel_path in relevant_paths[: max_depth * 3]:  # Limit results
                related_paths.append(
                    {
                        "path": rel_path,
                        "type": "physics_concept",
                        "distance": 1,
                    }
                )

            for rel_path in key_relationships[: max_depth * 2]:  # Limit results
                if rel_path not in [r["path"] for r in related_paths]:
                    related_paths.append(
                        {
                            "path": rel_path,
                            "type": "key_relationship",
                            "distance": 1,
                        }
                    )

        # Check unit_families for paths that share the same units
        for unit_name, unit_data in unit_families.items():
            paths_with_unit = unit_data.get("paths_using", [])
            if path in paths_with_unit:
                # Add other paths that use the same unit
                for related_path in paths_with_unit[: max_depth * 2]:  # Limit results
                    if related_path != path and related_path not in [
                        r["path"] for r in related_paths
                    ]:
                        related_paths.append(
                            {
                                "path": related_path,
                                "type": "unit_relationship",
                                "distance": 1,
                                "unit": unit_name,
                            }
                        )

        # Partial path matching for broader search
        if len(related_paths) < 3:  # If we don't have many direct matches
            path_lower = path.lower()

            # Search in cross_references
            for ref_path, ref_data in cross_references.items():
                if path_lower in ref_path.lower() or any(
                    path_lower in rel.get("path", "").lower()
                    for rel in ref_data.get("relationships", [])
                ):
                    for rel in ref_data.get("relationships", [])[
                        :3
                    ]:  # Limit per reference
                        if rel.get("path") not in [r["path"] for r in related_paths]:
                            related_paths.append(
                                {
                                    "path": rel.get("path", ""),
                                    "type": rel.get("type", ""),
                                    "distance": 2,
                                }
                            )

                if len(related_paths) >= max_depth * 8:  # Overall limit
                    break

            # Search in physics_concepts if still need more results
            if len(related_paths) < 5:
                for ref_path, ref_data in physics_concepts.items():
                    if path_lower in ref_path.lower():
                        relevant_paths = ref_data.get("relevant_paths", [])
                        for rel_path in relevant_paths[:3]:  # Limit per reference
                            if rel_path not in [r["path"] for r in related_paths]:
                                related_paths.append(
                                    {
                                        "path": rel_path,
                                        "type": "physics_partial",
                                        "distance": 2,
                                    }
                                )

                    if len(related_paths) >= max_depth * 8:  # Overall limit
                        break

            # Search in unit_families for partial matches if still need more results
            if len(related_paths) < 7:
                for unit_name, unit_data in unit_families.items():
                    paths_with_unit = unit_data.get("paths_using", [])
                    for unit_path in paths_with_unit:
                        if path_lower in unit_path.lower() and unit_path not in [
                            r["path"] for r in related_paths
                        ]:
                            related_paths.append(
                                {
                                    "path": unit_path,
                                    "type": "unit_partial",
                                    "distance": 2,
                                    "unit": unit_name,
                                }
                            )

                    if len(related_paths) >= max_depth * 8:  # Overall limit
                        break

        # Filter by relationship type if not ALL
        if relationship_type != RelationshipType.ALL:
            type_filter = relationship_type.value.lower()
            if type_filter == "cross_ids":
                related_paths = [r for r in related_paths if "IDS:" in r["path"]]
            elif type_filter == "structural":
                related_paths = [
                    r
                    for r in related_paths
                    if r["type"]
                    in ["cross_reference", "structure", "unit_relationship"]
                ]
            elif type_filter == "physics":
                # For physics relationships, we'd need additional metadata
                # For now, include all as physics connections are implicit
                pass

        return related_paths[: max_depth * 6]  # Final limit

    def _build_nodes_from_relationships(
        self, related_paths: list[dict]
    ) -> list[IdsNode]:
        """Build IdsNode objects from relationship data."""
        nodes = []

        for rel_info in related_paths:
            path = rel_info["path"]

            # Extract IDS name and create basic documentation
            if path.startswith("IDS:"):
                path = path[4:]  # Remove 'IDS:' prefix

            documentation = f"Related to input path via {rel_info['type']} relationship (distance: {rel_info['distance']})"

            # Create basic physics context if available
            physics_context = None
            if "equilibrium" in path.lower():
                physics_context = PhysicsContext(
                    domain="equilibrium",
                    phenomena=[],
                    typical_values={},
                )
            elif "transport" in path.lower():
                physics_context = PhysicsContext(
                    domain="transport",
                    phenomena=[],
                    typical_values={},
                )
            elif any(term in path.lower() for term in ["diagnostic", "measurement"]):
                physics_context = PhysicsContext(
                    domain="diagnostics",
                    phenomena=[],
                    typical_values={},
                )

            node = IdsNode(
                path=path,
                documentation=documentation,
                units="",  # Not available in relationships catalog
                data_type="",  # Not available in relationships catalog
                physics_context=physics_context,
            )
            nodes.append(node)

        return nodes

    def _generate_relationship_recommendations(
        self, path: str, related_paths: list[dict]
    ) -> list[str]:
        """Generate usage recommendations based on relationship context."""
        recommendations = []

        recommendations.append(
            f"🔍 Use search_imas('{path}') to find specific data paths"
        )

        # Path-specific recommendations
        if "equilibrium" in path.lower():
            recommendations.append(
                "⚡ Use analyze_ids_structure('equilibrium') for detailed equilibrium data structure"
            )

        if any("diagnostic" in rel["path"].lower() for rel in related_paths):
            recommendations.append(
                "📊 Use export_physics_domain('diagnostics') for measurement data"
            )

        if len(related_paths) > 5:
            cross_ids = {
                rel["path"].split("/")[0] for rel in related_paths if "/" in rel["path"]
            }
            recommendations.append(
                f"🔗 Use export_ids({list(cross_ids)[:3]}) to compare related IDS"
            )

        # Always include general recommendations
        recommendations.extend(
            [
                "💡 Use get_overview() to understand overall IMAS structure",
                "🌐 Use explore_identifiers() to browse available enumerations",
                "📈 Use analyze_ids_structure() for detailed structural analysis",
            ]
        )

        return recommendations[:6]  # Limit to 6 recommendations

    @cache_results(ttl=600, key_strategy="path_based")
    @validate_input(schema=RelationshipsInput)
    @handle_errors(fallback="relationships_suggestions")
    @mcp_tool("Discover connections and cross-references between IMAS data paths")
    async def explore_relationships(
        self,
        path: str,
        relationship_type: RelationshipType = RelationshipType.ALL,
        max_depth: int = 2,
        ctx: Context | None = None,
    ) -> RelationshipResult | ToolError:
        """
        Discover connections and cross-references between IMAS data paths.

        Network analysis tool that reveals how different measurements and calculations
        relate to each other across IDS. Use to understand data dependencies,
        find related measurements, and plan multi-IDS analysis workflows.

        Args:
            path: Starting data path or IDS name (e.g., 'equilibrium/time_slice/profiles_2d')
            relationship_type: Connection type - all, semantic, structural, physics, or measurement
            max_depth: Relationship traversal depth (1-3, limited for performance)
            ctx: MCP context for potential AI enhancement

        Returns:
            RelationshipResult with connected data paths and relationship insights
        """
        try:
            # Check if enhanced engine is available
            if not self._enhanced_engine:
                return ToolError(
                    error="Enhanced relationship engine not available",
                    suggestions=[
                        "Check if relationships.json exists in resources/schemas/",
                        "Try restarting the MCP server",
                        "Use search_imas() for direct data access",
                    ],
                    context={
                        "tool": "explore_relationships",
                        "operation": "enhanced_engine_access",
                    },
                )

            # Validate and limit max_depth for performance
            max_depth = min(max_depth, 3)  # Hard limit to prevent excessive traversal
            if max_depth < 1:
                max_depth = 1

            # Remove old path validation - enhanced engine handles various path formats
            # Use enhanced relationship discovery
            try:
                relationship_data = self._enhanced_engine.discover_relationships(
                    path, relationship_type, max_depth
                )
            except Exception as e:
                logger.error(f"Enhanced relationship discovery failed: {e}")
                return ToolError(
                    error=f"Relationship discovery failed: {str(e)}",
                    suggestions=[
                        f"Try search_imas('{path}') for direct path exploration",
                        "Use get_overview() to explore available IDS",
                        "Check if the path format is valid",
                    ],
                    context={
                        "tool": "explore_relationships",
                        "path": path,
                        "error": str(e),
                    },
                )

            if not relationship_data or not any(relationship_data.values()):
                return ToolError(
                    error=f"No relationships found for path: {path}",
                    suggestions=[
                        f"Try search_imas('{path}') for direct path exploration",
                        "Use get_overview() to explore available IDS",
                        "Try a broader path or different relationship type",
                    ],
                    context={"tool": "explore_relationships", "path": path},
                )

            # Generate enhanced physics context
            physics_context = self._enhanced_engine.generate_physics_context(
                path, relationship_data
            )

            # Create enhanced nodes with relationship metadata
            nodes = create_enhanced_relationship_nodes(
                relationship_data, physics_context
            )

            # Extract enhanced connection information
            all_paths = []
            physics_connections = []
            cross_ids_connections = set()

            for _rel_type, rel_list in relationship_data.items():
                for rel in rel_list:
                    rel_path = rel["path"]
                    all_paths.append(rel_path)

                    # Enhanced physics connection detection
                    if (
                        rel.get("type") in ["physics_domain", "semantic"]
                        and rel.get("strength", 0) > 0.3
                    ):
                        physics_connections.append(rel_path)

                    # Cross-IDS detection
                    if "/" in rel_path and not rel_path.startswith("IDS:"):
                        cross_ids_connections.add(rel_path.split("/")[0])

            # Extract physics domains from enhanced analysis
            physics_domains = []
            if physics_context:
                physics_domains.append(physics_context.domain)

            # Add domains from semantic analysis
            for rel_list in relationship_data.values():
                for rel in rel_list:
                    if rel.get("semantic_details", {}).get("domain_relationship"):
                        if "physics_domain_source" in rel:
                            physics_domains.extend(
                                [
                                    rel["physics_domain_source"],
                                    rel.get("physics_domain_target"),
                                ]
                            )

            # Build enhanced response
            response = RelationshipResult(
                path=path,
                relationship_type=relationship_type,
                max_depth=max_depth,
                connections={
                    "total_relationships": all_paths[:20],  # Limit for response size
                    "physics_connections": physics_connections[:10],
                    "cross_ids_connections": list(cross_ids_connections)[:10],
                },
                nodes=nodes[:12],  # Increased limit for enhanced nodes
                physics_domains=list(set(filter(None, physics_domains))),
                ai_response={
                    "relationship_insights": self._generate_relationship_insights(
                        relationship_data
                    ),
                    "physics_analysis": self._generate_physics_analysis(
                        path, relationship_data, physics_context
                    ),
                },
            )

            logger.info(f"Enhanced relationship exploration completed for path: {path}")
            return response
            return response

        except Exception as e:
            logger.error(f"Catalog-based relationship exploration failed: {e}")
            return ToolError(
                error=str(e),
                suggestions=[
                    "Try a simpler path or different relationship type",
                    "Use get_overview() for general IMAS exploration",
                    "Check relationships catalog file availability",
                ],
                context={
                    "path": path,
                    "relationship_type": relationship_type.value,
                    "tool": "explore_relationships",
                    "operation": "catalog_relationships",
                    "relationships_catalog_loaded": bool(self._relationships_catalog),
                },
            )

    def _generate_relationship_insights(
        self, relationship_data: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Generate insights from enhanced relationship analysis."""
        insights = {
            "discovery_summary": {},
            "strength_analysis": {},
            "semantic_insights": [],
        }

        total_relationships = sum(
            len(rel_list) for rel_list in relationship_data.values()
        )
        insights["discovery_summary"] = {
            "total_relationships": total_relationships,
            "relationship_types": list(relationship_data.keys()),
            "avg_strength": 0.0,
        }

        # Calculate average strength
        all_strengths = []
        semantic_insights = []

        for rel_type, rel_list in relationship_data.items():
            for rel in rel_list:
                if "strength" in rel:
                    all_strengths.append(rel["strength"])

                # Collect semantic insights
                if rel.get("semantic_details", {}).get("semantic_description"):
                    semantic_insights.append(
                        {
                            "path": rel["path"],
                            "description": rel["semantic_details"][
                                "semantic_description"
                            ],
                            "type": rel_type,
                        }
                    )

        if all_strengths:
            insights["discovery_summary"]["avg_strength"] = sum(all_strengths) / len(
                all_strengths
            )
            insights["strength_analysis"] = {
                "strongest_connections": [s for s in all_strengths if s > 0.7],
                "moderate_connections": [s for s in all_strengths if 0.3 <= s <= 0.7],
                "weak_connections": [s for s in all_strengths if s < 0.3],
            }

        insights["semantic_insights"] = semantic_insights[:5]  # Limit for response size

        return insights

    def _generate_physics_analysis(
        self,
        path: str,
        relationship_data: dict[str, list[dict[str, Any]]],
        physics_context: PhysicsContext | None,
    ) -> dict[str, Any]:
        """Generate physics-focused analysis from relationships."""
        analysis = {
            "primary_domain": None,
            "domain_connections": [],
            "physics_phenomena": [],
            "measurement_chains": [],
        }

        if physics_context:
            analysis["primary_domain"] = physics_context.domain
            analysis["physics_phenomena"] = physics_context.phenomena

        # Analyze domain connections
        domain_connections = []
        measurement_chains = []

        for _rel_type, rel_list in relationship_data.items():
            for rel in rel_list:
                # Physics domain analysis
                if rel.get("type") == "physics_domain":
                    domain_connections.append(
                        {
                            "source_domain": rel.get("physics_domain_source"),
                            "target_domain": rel.get("physics_domain_target"),
                            "connection_strength": rel.get("strength", 0),
                            "path": rel["path"],
                        }
                    )

                # Measurement chain analysis
                if rel.get("type") == "measurement_chain":
                    measurement_chains.append(
                        {
                            "path": rel["path"],
                            "connection_type": rel.get("measurement_connection"),
                            "strength": rel.get("strength", 0),
                        }
                    )

        analysis["domain_connections"] = domain_connections[:5]
        analysis["measurement_chains"] = measurement_chains[:5]

        return analysis
