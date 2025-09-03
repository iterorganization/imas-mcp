"""
Relationships tool implementation with enhanced clustering analysis.

This module provides an intelligent interface to the relationships catalog,
serving as the primary entry point for users to discover and navigate
data relationships and cross-references in the IMAS data dictionary.
Enhanced with advanced clustering analysis, physics domain mapping, and
strength scoring based on machine learning clustering of IMAS paths.

Key Features:
- Cluster-based relationship discovery using similarity scores
- Cross-IDS and intra-IDS clustering analysis
- Unit family relationships with clustering validation
- Enhanced semantic analysis with physics domain mapping
- Multi-layered strength classification with quantitative scoring
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp.core.data_model import IdsNode, PhysicsContext
from imas_mcp.core.relationships import Relationships
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
    Enhanced relationships catalog-based tool for IMAS data relationship discovery.

    Provides intelligent access to the relationships catalog (relationships.json)
    with advanced semantic analysis, physics domain mapping, and strength-based
    scoring for enhanced relationship discovery.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with relationships manager."""
        super().__init__(*args, **kwargs)
        # Pass ids_set from document store to Relationships manager
        ids_set = (
            self.document_store.ids_set
            if hasattr(self.document_store, "ids_set")
            else None
        )
        self._relationships = Relationships(ids_set=ids_set)
        self._load_relationships_catalog()

    def _load_relationships_catalog(self):
        """Load the relationships catalog using the unified manager."""
        try:
            # The unified manager handles auto-rebuild internally
            # when get_data() is called if dependencies are outdated
            self._relationships_catalog = self._relationships.get_data()

            # Extract clustering information for efficient access
            self._clusters = self._relationships_catalog.get("clusters", [])
            self._cluster_metadata = self._relationships_catalog.get("metadata", {})
            self._unit_families = self._relationships_catalog.get("unit_families", {})

            # Create path-to-cluster lookup for fast access
            self._path_to_clusters = {}
            for cluster in self._clusters:
                cluster_id = cluster["id"]
                for path in cluster.get("paths", []):
                    if path not in self._path_to_clusters:
                        self._path_to_clusters[path] = []
                    self._path_to_clusters[path].append(cluster_id)

            # Get enhanced engine through unified manager
            self._enhanced_engine = self._relationships.get_enhanced_engine()

            logger.info(
                f"Loaded relationships catalog with {len(self._clusters)} clusters "
                f"({self._cluster_metadata.get('statistics', {}).get('cross_ids_clustering', {}).get('total_clusters', 0)} cross-IDS, "
                f"{self._cluster_metadata.get('statistics', {}).get('intra_ids_clustering', {}).get('total_clusters', 0)} intra-IDS)"
            )

        except Exception as e:
            logger.error(f"Failed to load relationships catalog: {e}")
            self._relationships_catalog = {}
            self._clusters = []
            self._cluster_metadata = {}
            self._unit_families = {}
            self._path_to_clusters = {}
            self._enhanced_engine = None

    def _find_cluster_relationships(
        self, path: str, relationship_type: RelationshipType, max_depth: int
    ) -> list[dict]:
        """Find related paths using the new clustering information."""
        if not self._clusters or not self._path_to_clusters:
            return []

        related_paths = []
        cluster_ids = self._path_to_clusters.get(path, [])

        # Direct cluster membership
        for cluster_id in cluster_ids:
            cluster = next((c for c in self._clusters if c["id"] == cluster_id), None)
            if not cluster:
                continue

            # Include all clusters regardless of relationship type
            # (removed filtering to show complete cluster contents)

            similarity_score = cluster.get("similarity_score", 0.0)
            cluster_size = cluster.get("size", 0)
            is_cross_ids = cluster.get("is_cross_ids", False)
            ids_names = cluster.get("ids_names", [])

            # Add related paths from the same cluster
            for related_path in cluster.get("paths", []):
                if related_path != path:  # Don't include the input path
                    related_paths.append(
                        {
                            "path": related_path,
                            "type": "cluster_cross_ids"
                            if is_cross_ids
                            else "cluster_intra_ids",
                            "distance": 1,
                            "similarity_score": similarity_score,
                            "cluster_id": cluster_id,
                            "cluster_size": cluster_size,
                            "ids_names": ids_names,
                            "strength": min(
                                similarity_score, 0.9
                            ),  # Cap at 0.9 for very strong
                        }
                    )

        # Find related clusters (clusters that share IDS or have high similarity)
        if max_depth > 1 and cluster_ids:
            source_cluster = next(
                (c for c in self._clusters if c["id"] in cluster_ids), None
            )
            if source_cluster:
                source_ids = set(source_cluster.get("ids_names", []))

                for cluster in self._clusters:
                    if cluster["id"] in cluster_ids:
                        continue  # Skip already processed clusters

                    cluster_ids_set = set(cluster.get("ids_names", []))

                    # Check for IDS overlap
                    ids_overlap = len(source_ids.intersection(cluster_ids_set))
                    if ids_overlap > 0:
                        similarity_boost = min(
                            0.2 * ids_overlap, 0.4
                        )  # Boost for shared IDS
                        effective_similarity = (
                            cluster.get("similarity_score", 0.0) + similarity_boost
                        )

                        # Add paths from related clusters
                        for related_path in cluster.get(
                            "paths", []
                        ):  # Show all cluster paths
                            related_paths.append(
                                {
                                    "path": related_path,
                                    "type": "related_cluster",
                                    "distance": 2,
                                    "similarity_score": cluster.get(
                                        "similarity_score", 0.0
                                    ),
                                    "effective_similarity": effective_similarity,
                                    "cluster_id": cluster["id"],
                                    "cluster_size": cluster.get("size", 0),
                                    "ids_overlap": ids_overlap,
                                    "strength": max(
                                        0.3, min(effective_similarity * 0.8, 0.7)
                                    ),  # Moderate to strong
                                }
                            )

        # Unit family relationships from clustering
        for unit_name, unit_data in self._unit_families.items():
            paths_with_unit = unit_data.get("paths_using", [])
            if path in paths_with_unit:
                for related_path in paths_with_unit[:8]:  # Limit unit relationships
                    if related_path != path and related_path not in [
                        r["path"] for r in related_paths
                    ]:
                        related_paths.append(
                            {
                                "path": related_path,
                                "type": "unit_cluster",
                                "distance": 1,
                                "unit": unit_name,
                                "strength": 0.4,  # Moderate strength for unit relationships
                            }
                        )

        # Sort by strength and similarity, return all results
        related_paths.sort(
            key=lambda x: (-x.get("strength", 0), -x.get("similarity_score", 0))
        )
        return related_paths  # Return complete cluster data

    def _merge_relationship_sources(
        self,
        cluster_relationships: list[dict],
        enhanced_relationships: dict[str, list[dict]],
    ) -> dict[str, list[dict]]:
        """Merge cluster-based relationships with enhanced relationship discovery."""
        merged = {"cluster_based": cluster_relationships}

        # Add enhanced relationships
        for rel_type, rel_list in enhanced_relationships.items():
            if rel_type not in merged:
                merged[rel_type] = []
            merged[rel_type].extend(rel_list)

        # Remove duplicates based on path
        for rel_type in merged:
            seen_paths = set()
            unique_rels = []
            for rel in merged[rel_type]:
                path = rel.get("path", "")
                if path not in seen_paths:
                    seen_paths.add(path)
                    unique_rels.append(rel)
            merged[rel_type] = unique_rels

        return merged

    def _generate_cluster_insights(
        self, relationship_data: dict[str, list[dict[str, Any]]], cluster_info: dict
    ) -> dict[str, Any]:
        """Generate focused cluster insights with reduced noise."""
        if not cluster_info:
            return {
                "total_relationships": sum(
                    len(rel_list) for rel_list in relationship_data.values()
                )
            }

        # Calculate key statistics
        total_relationships = sum(
            len(rel_list) for rel_list in relationship_data.values()
        )
        cross_ids_count = sum(
            1 for c in cluster_info.values() if c.get("is_cross_ids", False)
        )

        # Get strength distribution
        all_strengths = []
        for rel_list in relationship_data.values():
            for rel in rel_list:
                if "strength" in rel:
                    all_strengths.append(rel["strength"])

        avg_strength = sum(all_strengths) / len(all_strengths) if all_strengths else 0

        # Extract unique IDS coverage
        all_ids = set()
        for cluster_data in cluster_info.values():
            all_ids.update(cluster_data.get("ids_names", []))

        # Get cluster size distribution
        cluster_sizes = [c.get("size", 0) for c in cluster_info.values()]
        max_cluster_size = max(cluster_sizes, default=0)

        return {
            "total_relationships": total_relationships,
            "avg_strength": round(avg_strength, 3),
            "clusters": {
                "total": len(cluster_info),
                "cross_ids": cross_ids_count,
                "intra_ids": len(cluster_info) - cross_ids_count,
                "max_size": max_cluster_size,
                "ids_coverage": sorted(all_ids),
            },
            "strength_distribution": {
                "strong": len([s for s in all_strengths if s > 0.7]),
                "moderate": len([s for s in all_strengths if 0.3 <= s <= 0.7]),
                "weak": len([s for s in all_strengths if s < 0.3]),
            },
        }

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
        self, related_paths: list[dict], limit: int = 10
    ) -> list[IdsNode]:
        """Build simplified IdsNode objects focusing on essential information."""
        nodes = []

        # Sort by strength and take top results
        sorted_paths = sorted(
            related_paths,
            key=lambda x: (-x.get("strength", 0), -x.get("similarity_score", 0)),
        )

        for rel_info in sorted_paths[:limit]:
            path = rel_info["path"]

            # Extract IDS name and clean path
            if path.startswith("IDS:"):
                path = path[4:]

            # Create concise, meaningful documentation explaining the relationship
            rel_type = rel_info.get("type", "related")
            strength = rel_info.get("strength", 0)

            if strength >= 0.7:
                strength_label = "Strong"
            elif strength >= 0.4:
                strength_label = "Moderate"
            else:
                strength_label = "Weak"

            # Create physics-aware documentation
            path_parts = path.split("/")
            if len(path_parts) >= 3:
                quantity_name = path_parts[-1].replace("_", " ").title()
                if len(path_parts) >= 4:
                    context = path_parts[-2].replace("_", " ")
                    if rel_type == "cluster_cross_ids":
                        documentation = f"{strength_label} cross-IDS relationship: {quantity_name} in {context} context"
                    elif rel_type == "cluster_intra_ids":
                        documentation = f"{strength_label} structural relationship: {quantity_name} in {context}"
                    else:
                        documentation = (
                            f"{strength_label} {rel_type} relationship: {quantity_name}"
                        )
                else:
                    documentation = f"{strength_label} relationship: {quantity_name}"
            else:
                documentation = f"{strength_label} {rel_type} relationship"

            # Add clustering context if available
            if rel_info.get("cluster_size"):
                cluster_size = rel_info["cluster_size"]
                similarity = rel_info.get("similarity_score", 0)
                if cluster_size > 50:  # Large cluster
                    documentation += f" (part of {cluster_size}-path cluster, {similarity:.2f} similarity)"
                elif cluster_size > 10:  # Medium cluster
                    documentation += f" (clustered with {cluster_size} similar paths)"

            # Only create physics context if we have meaningful data
            physics_context = None
            if rel_info.get("physics_domain"):
                physics_context = PhysicsContext(
                    domain=rel_info["physics_domain"],
                    phenomena=rel_info.get("phenomena", []),
                    typical_values={},
                )

            # Create minimal node with only meaningful fields
            node_data = {
                "path": path,
                "documentation": documentation,
            }

            # Only include non-empty, meaningful metadata
            if rel_info.get("units") and rel_info["units"].strip():
                node_data["units"] = rel_info["units"]
            if rel_info.get("data_type") and rel_info["data_type"].strip():
                node_data["data_type"] = rel_info["data_type"]

            # Add relationship metadata for LLM context
            node_data["relationship_strength"] = round(strength, 2)
            node_data["relationship_type"] = rel_type

            # Add physics context only if meaningful
            if physics_context and physics_context.domain:
                node_data["physics_domain"] = physics_context.domain
            elif rel_info.get("physics_domain"):
                node_data["physics_domain"] = rel_info["physics_domain"]

            # Store as dictionary for LLM consumption
            node_dict = {
                "path": node_data["path"],
                "documentation": node_data["documentation"],
                "relationship_strength": node_data["relationship_strength"],
                "relationship_type": node_data["relationship_type"],
            }

            # Add optional fields
            if "units" in node_data:
                node_dict["units"] = node_data["units"]
            if "data_type" in node_data:
                node_dict["data_type"] = node_data["data_type"]
            if "physics_domain" in node_data:
                node_dict["physics_domain"] = node_data["physics_domain"]

            nodes.append(node_dict)

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

        **CRITICAL: This tool ONLY accepts IMAS data paths, not queries or descriptions.**
        Use search_imas() first to find valid paths if you don't have exact path strings.

        This tool provides sophisticated relationship discovery with multi-layered analysis,
        semantic understanding, and physics domain integration. It reveals how different
        measurements and calculations relate across IDS structures using advanced algorithms.

        **Core Capabilities:**
        - Multi-layered relationship discovery (semantic, structural, physics, measurement)
        - 5-tier strength classification with quantitative scoring
        - Physics domain integration with cross-domain bridging analysis
        - Comprehensive metadata and contextual insights

        **Strength Classification System:**
        - very_strong (0.9): Direct physics coupling (e.g., density ↔ density_fit)
        - strong (0.7): Same measurement type (e.g., electron_density ↔ ion_density)
        - moderate (0.5): Related physics domain (e.g., transport ↔ heating)
        - weak (0.3): Structural similarity (e.g., same coordinate system)
        - very_weak (0.1): Unit similarity only

        **INPUT REQUIREMENTS - CRITICAL:**
        ✅ VALID: IMAS data paths only: "core_profiles/profiles_1d/electrons/density"
        ❌ INVALID: Queries like "electron density" or "find temperature data"
        ❌ INVALID: Natural language descriptions or partial paths

        **Usage Examples:**

        1. Comprehensive relationship analysis (recommended):
        ```python
        result = await explore_relationships(
            path="core_profiles/profiles_1d/electrons/density",
            relationship_type="all",  # Gets all 4 relationship types
            max_depth=2               # Standard depth for comprehensive results
        )
        # Returns: ~15-20 relationships across semantic, structural, physics, measurement types
        # Includes: strength scores, physics domains, cross-IDS connections
        ```

        2. Focus on physics relationships only:
        ```python
        result = await explore_relationships(
            path="equilibrium/time_slice/profiles_2d/b_field_r",
            relationship_type="semantic",  # Physics concepts and domain relationships
            max_depth=1                    # Immediate relationships only
        )
        # Returns: ~3-8 semantically related paths with physics context
        # Includes: domain bridging (e.g., mhd ↔ transport connections)
        ```

        3. Structural analysis for data organization:
        ```python
        result = await explore_relationships(
            path="transport/model/profiles_1d/conductivity_parallel",
            relationship_type="structural", # Hierarchical and organizational
            max_depth=1                     # Close structural relatives only
        )
        # Returns: ~2-6 structurally similar paths within same IDS
        ```

        4. Cross-domain physics analysis:
        ```python
        result = await explore_relationships(
            path="heating/nbi/unit/power_launched",
            relationship_type="physics",    # Physics domain relationships
            max_depth=2                     # Extended physics connections
        )
        # Returns: ~5-12 physics-related paths with domain mapping
        ```

        **Typical Results by Physics Domain:**
        - Transport paths: 15-20 relationships (high connectivity)
        - Equilibrium paths: 8-15 relationships (magnetic field coupling)
        - Heating paths: 6-12 relationships (power and energy flow)
        - Diagnostic paths: 5-10 relationships (measurement chains)

        Args:
            path: **IMAS data path ONLY** - exact path string from IMAS data dictionary
                  Examples: "core_profiles/profiles_1d/electrons/density"
                           "equilibrium/time_slice/global_quantities/psi_boundary"
                           "thomson_scattering/channel/position/r"
                  Must be valid IMAS path - use search_imas() to find valid paths
            relationship_type: Filter for specific relationship types:
                - "all": All relationship types (semantic + structural + physics + measurement)
                - "semantic": Physics concepts, domain relationships, phenomena connections
                - "structural": Hierarchical organization, IDS structure, coordinate sharing
                - "physics": Physics domain coupling, cross-domain analysis, measurement chains
                - "measurement": Diagnostic chains, measurement dependencies, error propagation
            max_depth: Relationship traversal depth (1-3):
                - 1: Immediate relationships only (fast, focused)
                - 2: Standard depth (recommended, balanced performance/coverage)
                - 3: Extended analysis (comprehensive but slower)
            ctx: MCP context for potential future AI enhancement

        Returns:
            RelationshipResult containing:
            - **nodes**: List of related IdsNode objects with documentation and metadata
            - **connections**: Categorized relationship lists (total, physics, cross-IDS)
            - **physics_domains**: Identified physics domains (transport, mhd, thermal, etc.)
            - **relationship_insights**: Discovery summary, strength analysis, semantic insights
            - **physics_analysis**: Domain connections, phenomena, measurement chains
            - **Standard metadata**: Query context, tool hints, processing timestamps

        Raises:
            ToolError: When path not found, invalid format, or no relationships discovered
                      Includes helpful suggestions for alternative approaches

        **Integration Patterns:**
        1. **Discovery → Relationship → Analysis**: search_imas() → explore_relationships() → analyze_ids_structure()
        2. **Relationship → Export**: explore_relationships() → export_physics_domain()
        3. **Cross-domain mapping**: Use relationship_type="physics" for domain bridging

        **Performance Notes:**
        - Typical execution: 0.5-2.0 seconds depending on path complexity
        - Results limited to prevent overwhelming responses (nodes<15, connections<20)
        - Caching enabled for repeated queries (TTL: 600 seconds)

        **Path Discovery Workflow:**
        If you don't have exact IMAS paths:
        1. Use search_imas("your concept") to find relevant paths
        2. Use get_overview() to browse available IDS structures
        3. Use explore_identifiers() to understand enumeration options
        4. Then use explore_relationships() with discovered paths
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
            # Use cluster-based relationship discovery first
            cluster_relationships = self._find_cluster_relationships(
                path, relationship_type, max_depth
            )

            # Enhance with traditional relationship discovery
            try:
                enhanced_relationships = self._enhanced_engine.discover_relationships(
                    path, relationship_type, max_depth
                )

                # Merge cluster-based and enhanced relationships
                relationship_data = self._merge_relationship_sources(
                    cluster_relationships, enhanced_relationships
                )
            except Exception as e:
                logger.warning(
                    f"Enhanced relationship discovery failed, using cluster-only: {e}"
                )
                # Fall back to cluster-based relationships only
                relationship_data = {"cluster_based": cluster_relationships}

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

            # Extract categorized connection information
            intra_ids_paths = set()
            cross_ids_paths = set()
            involved_ids = set()
            cluster_info = {}

            # Collect relationships with strength threshold
            strength_threshold = 0.3
            for _rel_type, rel_list in relationship_data.items():
                for rel in rel_list:
                    rel_path = rel["path"]
                    strength = rel.get("strength", 0)
                    rel_type = rel.get("type", "")

                    # Only include relationships above threshold
                    if strength >= strength_threshold:
                        # Always categorize based on actual IDS analysis, not cluster labels
                        query_ids = path.split("/")[0] if "/" in path else path
                        rel_ids = (
                            rel_path.split("/")[0] if "/" in rel_path else rel_path
                        )

                        if query_ids == rel_ids:
                            # Same IDS = intra-IDS relationship
                            intra_ids_paths.add(rel_path)
                        else:
                            # Different IDS = cross-IDS relationship
                            cross_ids_paths.add(rel_path)

                        # Track involved IDS names
                        if "/" in rel_path and not rel_path.startswith("IDS:"):
                            involved_ids.add(rel_path.split("/")[0])

                        # Collect cluster metadata (avoid duplicates)
                        if "cluster_id" in rel:
                            cluster_id = rel["cluster_id"]
                            if cluster_id not in cluster_info:
                                cluster_info[cluster_id] = {
                                    "similarity_score": rel.get(
                                        "similarity_score", 0.0
                                    ),
                                    "size": rel.get("cluster_size", 0),
                                    "is_cross_ids": rel.get("type")
                                    == "cluster_cross_ids",
                                    "ids_names": rel.get("ids_names", []),
                                    "path_count": 0,
                                }
                            cluster_info[cluster_id]["path_count"] += 1

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

            # After building connections, create summary using the actual connection data
            strong_relationships = []
            for _rel_type, rel_list in relationship_data.items():
                for rel in rel_list:
                    rel_path = rel["path"]
                    strength = rel.get("strength", 0)
                    rel_type = rel.get("type", "")

                    # Track strong relationships for summary
                    if strength >= 0.7:
                        strong_relationships.append(
                            {
                                "path": rel_path,
                                "strength": round(strength, 2),
                                "type": rel_type,
                            }
                        )

            # Create summary using the actual connection sets
            relationship_summary = {
                "query_path": path,
                "total_found": len(intra_ids_paths) + len(cross_ids_paths),
                "intra_ids_similar": sorted(
                    intra_ids_paths
                ),  # Show all intra-IDS paths
                "cross_ids_similar": sorted(
                    cross_ids_paths
                ),  # Show all cross-IDS paths
                "strongest_relationships": sorted(
                    strong_relationships, key=lambda x: x["strength"], reverse=True
                ),
                "ids_involved": sorted(involved_ids),
                "primary_physics_domain": physics_domains[0]
                if physics_domains
                else None,
            }

            # Generate enhanced insights with summary at top
            if cluster_info:
                cluster_insights = self._generate_cluster_insights(
                    relationship_data, cluster_info
                )
                relationship_insights = cluster_insights
                # Add summary to insights
                relationship_insights["summary"] = relationship_summary
            else:
                relationship_insights = {"summary": relationship_summary}

            # Build enhanced response with improved connections structure
            response = RelationshipResult(
                path=path,
                relationship_type=relationship_type,
                max_depth=max_depth,
                connections={
                    "intra_ids_paths": sorted(intra_ids_paths),
                    "cross_ids_paths": sorted(cross_ids_paths),
                    "involved_ids": sorted(involved_ids),
                },
                physics_domains=list(set(filter(None, physics_domains))),
                relationship_insights=relationship_insights,
                physics_analysis=self._generate_physics_analysis(
                    path, relationship_data, physics_context
                ),
            )

            logger.info(f"Enhanced relationship exploration completed for path: {path}")
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
