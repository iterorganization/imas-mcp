"""
Relationships tool implementation with catalog-based architecture.

This module provides an intelligent interface to the relationships catalog,
serving as the primary entry point for users to discover and navigate
data relationships and cross-references in the IMAS data dictionary.
"""

import importlib.resources
import json
import logging

from fastmcp import Context

from imas_mcp.core.data_model import IdsNode, PhysicsContext
from imas_mcp.models.constants import RelationshipType
from imas_mcp.models.error_models import ToolError
from imas_mcp.models.request_models import RelationshipsInput
from imas_mcp.models.result_models import RelationshipResult
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
    Relationships catalog-based tool for IMAS data relationship discovery.

    Provides intelligent access to the relationships catalog (relationships.json),
    serving as the primary interface for users to discover data connections
    and cross-references in the IMAS data dictionary.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with relationships catalog data loading."""
        super().__init__(*args, **kwargs)
        self._relationships_catalog = {}
        self._load_relationships_catalog()

    def _load_relationships_catalog(self):
        """Load the relationships catalog file specifically."""
        try:
            try:
                catalog_file = (
                    importlib.resources.files("imas_mcp.resources.schemas")
                    / "relationships.json"
                )
                with catalog_file.open("r", encoding="utf-8") as f:
                    self._relationships_catalog = json.load(f)
                    logger.info("Loaded relationships catalog for relationships tool")
            except FileNotFoundError:
                logger.warning(
                    "Relationships catalog (relationships.json) not found in resources/schemas/"
                )

        except Exception as e:
            logger.error(f"Failed to load relationships catalog: {e}")
            self._relationships_catalog = {}

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
            # Check if relationships catalog is loaded
            if not self._relationships_catalog:
                return ToolError(
                    error="Relationships catalog data not available",
                    suggestions=[
                        "Check if relationships.json exists in resources/schemas/",
                        "Try restarting the MCP server",
                        "Use search_imas() for direct data access",
                    ],
                    context={
                        "tool": "explore_relationships",
                        "operation": "catalog_access",
                    },
                )

            # Validate and limit max_depth for performance
            max_depth = min(max_depth, 3)  # Hard limit to prevent excessive traversal
            if max_depth < 1:
                max_depth = 1

            # Parse the path to extract IDS name (for validation)
            if "/" in path:
                specific_path = path
            else:
                specific_path = path

            # Find related paths from catalog
            related_paths = self._find_related_paths(
                specific_path, relationship_type, max_depth
            )

            if not related_paths:
                return ToolError(
                    error=f"No relationships found for path: {path}",
                    suggestions=[
                        f"Try search_imas('{path}') for direct path exploration",
                        "Use get_overview() to explore available IDS",
                        "Check if the path exists using analyze_ids_structure()",
                    ],
                    context={"tool": "explore_relationships", "path": path},
                )

            # Build nodes from relationships
            nodes = self._build_nodes_from_relationships(related_paths)

            # Extract connection information
            total_relationships = [rel["path"] for rel in related_paths]
            physics_connections = [
                rel["path"]
                for rel in related_paths
                if any(
                    domain in rel["path"].lower()
                    for domain in ["equilibrium", "transport", "heating"]
                )
            ]
            cross_ids_connections = list(
                {
                    rel["path"].split("/")[0]
                    for rel in related_paths
                    if "/" in rel["path"] and not rel["path"].startswith("IDS:")
                }
            )

            # Extract physics domains from nodes
            physics_domains = [
                node.physics_context.domain
                for node in nodes
                if node.physics_context and node.physics_context.domain
            ]

            # Generate recommendations - used in future enhancements
            # recommendations = self._generate_relationship_recommendations(path, related_paths)

            # Build response using Pydantic
            response = RelationshipResult(
                path=path,
                relationship_type=relationship_type,
                max_depth=max_depth,
                connections={
                    "total_relationships": total_relationships,
                    "physics_connections": physics_connections,
                    "cross_ids_connections": cross_ids_connections,
                },
                nodes=nodes[:8],  # Limit nodes for response size
                physics_domains=list(set(physics_domains)),
                physics_context=None,  # Simplified for catalog-based data
                ai_response={},  # No AI processing needed for catalog data
            )

            logger.info(f"Relationship exploration completed for path: {path}")
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
