"""
Enhanced relationship discovery engine for IMAS data paths.

This module implements advanced relationship discovery algorithms including
semantic analysis, physics domain mapping, and strength-based scoring.
"""

import logging
from typing import Any

from imas_mcp.core.data_model import IdsNode, PhysicsContext
from imas_mcp.models.constants import RelationshipType

logger = logging.getLogger(__name__)


class RelationshipStrength:
    """Relationship strength metrics and classification."""

    # Strength categories
    VERY_STRONG = 0.9  # Direct physics coupling (e.g., temperature -> pressure)
    STRONG = 0.7  # Same measurement type (e.g., density -> ion density)
    MODERATE = 0.5  # Related physics domain (e.g., transport -> heating)
    WEAK = 0.3  # Structural similarity (e.g., same coordinate system)
    VERY_WEAK = 0.1  # Unit similarity only

    @classmethod
    def get_category(cls, strength: float) -> str:
        """Get human-readable strength category."""
        if strength >= cls.VERY_STRONG:
            return "very_strong"
        elif strength >= cls.STRONG:
            return "strong"
        elif strength >= cls.MODERATE:
            return "moderate"
        elif strength >= cls.WEAK:
            return "weak"
        else:
            return "very_weak"


class SemanticRelationshipAnalyzer:
    """Semantic analysis for physics relationships."""

    # Physics concept mappings for semantic analysis
    PHYSICS_CONCEPTS = {
        "density": {
            "domain": "transport",
            "related_concepts": ["pressure", "temperature", "velocity", "flux"],
            "measurement_types": ["particle", "mass", "charge"],
            "physics_laws": ["continuity_equation", "ideal_gas_law"],
        },
        "temperature": {
            "domain": "thermal",
            "related_concepts": ["pressure", "density", "energy", "heat_flux"],
            "measurement_types": ["electron", "ion", "kinetic"],
            "physics_laws": ["ideal_gas_law", "heat_equation"],
        },
        "pressure": {
            "domain": "thermal",
            "related_concepts": ["temperature", "density", "velocity"],
            "measurement_types": ["kinetic", "magnetic", "radiation"],
            "physics_laws": ["ideal_gas_law", "momentum_equation"],
        },
        "current": {
            "domain": "electromagnetic",
            "related_concepts": ["magnetic_field", "voltage", "resistance"],
            "measurement_types": ["plasma", "bootstrap", "ohmic"],
            "physics_laws": ["ohms_law", "amperes_law"],
        },
        "magnetic_field": {
            "domain": "electromagnetic",
            "related_concepts": ["current", "flux", "equilibrium"],
            "measurement_types": ["poloidal", "toroidal", "radial"],
            "physics_laws": ["amperes_law", "faradays_law"],
        },
        "flux": {
            "domain": "transport",
            "related_concepts": ["density", "velocity", "diffusion"],
            "measurement_types": ["particle", "heat", "momentum"],
            "physics_laws": ["fick_law", "fourier_law"],
        },
        "equilibrium": {
            "domain": "mhd",
            "related_concepts": ["magnetic_field", "pressure", "current"],
            "measurement_types": ["force_balance", "grad_shafranov"],
            "physics_laws": ["force_balance", "grad_shafranov_equation"],
        },
    }

    # Physics domain relationships
    DOMAIN_RELATIONSHIPS = {
        "transport": ["thermal", "mhd", "heating"],
        "thermal": ["transport", "heating"],
        "electromagnetic": ["mhd", "heating"],
        "mhd": ["electromagnetic", "transport", "equilibrium"],
        "heating": ["thermal", "transport", "electromagnetic"],
        "diagnostics": ["transport", "thermal", "electromagnetic"],
        "equilibrium": ["mhd", "electromagnetic"],
    }

    def __init__(self):
        """Initialize semantic analyzer."""
        self._concept_cache = {}

    def analyze_concept(self, path: str) -> dict[str, Any]:
        """Extract physics concepts from a path."""
        if path in self._concept_cache:
            return self._concept_cache[path]

        path_lower = path.lower()
        concepts = []
        primary_domain = None

        # Extract concepts from path
        for concept, data in self.PHYSICS_CONCEPTS.items():
            if concept in path_lower:
                concepts.append(concept)
                if not primary_domain:
                    primary_domain = data["domain"]

        # Detect measurement types
        measurement_types = []
        for concept_data in self.PHYSICS_CONCEPTS.values():
            for mtype in concept_data["measurement_types"]:
                if mtype in path_lower:
                    measurement_types.append(mtype)

        result = {
            "concepts": concepts,
            "primary_domain": primary_domain,
            "measurement_types": measurement_types,
            "path_components": path.split("/"),
        }

        self._concept_cache[path] = result
        return result

    def calculate_semantic_similarity(
        self, path1: str, path2: str
    ) -> tuple[float, dict[str, Any]]:
        """Calculate semantic similarity between two paths."""
        concept1 = self.analyze_concept(path1)
        concept2 = self.analyze_concept(path2)

        similarity_score = 0.0
        details = {
            "shared_concepts": [],
            "shared_measurement_types": [],
            "domain_relationship": None,
            "semantic_description": "",
        }

        # Shared concepts (high weight)
        shared_concepts = set(concept1["concepts"]) & set(concept2["concepts"])
        if shared_concepts:
            similarity_score += len(shared_concepts) * 0.4
            details["shared_concepts"] = list(shared_concepts)

        # Shared measurement types (medium weight)
        shared_measurements = set(concept1["measurement_types"]) & set(
            concept2["measurement_types"]
        )
        if shared_measurements:
            similarity_score += len(shared_measurements) * 0.2
            details["shared_measurement_types"] = list(shared_measurements)

        # Domain relationship (medium weight)
        domain1 = concept1["primary_domain"]
        domain2 = concept2["primary_domain"]
        if domain1 and domain2:
            if domain1 == domain2:
                similarity_score += 0.3
                details["domain_relationship"] = "same_domain"
            elif domain2 in self.DOMAIN_RELATIONSHIPS.get(domain1, []):
                similarity_score += 0.2
                details["domain_relationship"] = "related_domains"

        # Generate semantic description
        if shared_concepts:
            details["semantic_description"] = (
                f"Related through {', '.join(shared_concepts)} physics"
            )
        elif shared_measurements:
            details["semantic_description"] = (
                f"Share {', '.join(shared_measurements)} measurement types"
            )
        elif details["domain_relationship"]:
            details["semantic_description"] = (
                f"Connected via {domain1}-{domain2} physics domains"
            )

        return min(similarity_score, 1.0), details


class EnhancedRelationshipEngine:
    """Enhanced multi-layered relationship discovery engine."""

    def __init__(self, relationships_catalog: dict[str, Any]):
        """Initialize with relationships catalog."""
        self.relationships_catalog = relationships_catalog
        self.semantic_analyzer = SemanticRelationshipAnalyzer()
        self._path_cache = {}

    def discover_relationships(
        self,
        path: str,
        relationship_type: RelationshipType = RelationshipType.ALL,
        max_depth: int = 2,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Multi-layered relationship discovery with semantic analysis and strength scoring.

        Returns relationships organized by type with strength metrics and semantic descriptions.
        """
        # Get base relationships from catalog
        catalog_relationships = self._get_catalog_relationships(path, max_depth)

        # Enhance with semantic analysis
        semantic_relationships = self._analyze_semantic_relationships(
            path, catalog_relationships
        )

        # Analyze physics domain connections
        physics_relationships = self._analyze_physics_domain_relationships(
            path, catalog_relationships
        )

        # Analyze measurement chain relationships
        measurement_relationships = self._analyze_measurement_chains(
            path, catalog_relationships
        )

        # Combine and rank all relationships
        all_relationships = {
            "semantic": semantic_relationships,
            "structural": catalog_relationships,
            "physics": physics_relationships,
            "measurement": measurement_relationships,
        }

        # Filter by relationship type if specified
        if relationship_type != RelationshipType.ALL:
            type_filter = relationship_type.value.lower()
            if type_filter in all_relationships:
                all_relationships = {type_filter: all_relationships[type_filter]}
            else:
                # Map to closest category
                if type_filter == "cross_ids":
                    all_relationships = {"structural": catalog_relationships}
                elif type_filter in ["semantic", "physics", "measurement"]:
                    all_relationships = {
                        type_filter: all_relationships.get(type_filter, [])
                    }

        return self._rank_and_filter_relationships(all_relationships, max_depth)

    def _get_catalog_relationships(
        self, path: str, max_depth: int
    ) -> list[dict[str, Any]]:
        """Get enhanced relationships from catalog including clustering data."""
        if not self.relationships_catalog:
            return []

        cross_references = self.relationships_catalog.get("cross_references", {})
        physics_concepts = self.relationships_catalog.get("physics_concepts", {})
        unit_families = self.relationships_catalog.get("unit_families", {})
        clusters = self.relationships_catalog.get("clusters", [])

        relationships = []

        # Enhanced cluster-based relationships
        path_clusters = []
        for cluster in clusters:
            if path in cluster.get("paths", []):
                path_clusters.append(cluster)

        for cluster in path_clusters:
            cluster_id = cluster["id"]
            similarity_score = cluster.get("similarity_score", 0.0)
            is_cross_ids = cluster.get("is_cross_ids", False)
            cluster_size = cluster.get("size", 0)
            ids_names = cluster.get("ids_names", [])

            # Add other paths from the same cluster
            for cluster_path in cluster.get("paths", []):
                if cluster_path != path:  # Don't include the source path
                    # Enhanced strength calculation based on cluster properties
                    base_strength = min(similarity_score, 0.9)  # Cap at very strong
                    if is_cross_ids:
                        base_strength *= 1.1  # Boost cross-IDS relationships
                    if cluster_size < 5:
                        base_strength *= 1.05  # Boost smaller, more focused clusters

                    relationships.append(
                        {
                            "path": cluster_path,
                            "type": "cluster_cross_ids"
                            if is_cross_ids
                            else "cluster_intra_ids",
                            "distance": 1,
                            "strength": min(base_strength, 0.95),
                            "source": "clustering",
                            "cluster_id": cluster_id,
                            "cluster_size": cluster_size,
                            "similarity_score": similarity_score,
                            "ids_names": ids_names,
                            "cluster_type": "cross-IDS"
                            if is_cross_ids
                            else "intra-IDS",
                        }
                    )

        # Traditional direct cross-references (enhanced with cluster context)
        if path in cross_references:
            for rel in cross_references[path].get("relationships", []):
                rel_path = rel.get("path", "")
                # Check if this relationship is also in a cluster for enhanced strength
                enhanced_strength = RelationshipStrength.STRONG
                for cluster in path_clusters:
                    if rel_path in cluster.get("paths", []):
                        enhanced_strength = min(
                            RelationshipStrength.VERY_STRONG,
                            cluster.get("similarity_score", 0.7) + 0.1,
                        )
                        break

                relationships.append(
                    {
                        "path": rel_path,
                        "type": "cross_reference",
                        "distance": 1,
                        "strength": enhanced_strength,
                        "source": "catalog_direct",
                    }
                )

        # Enhanced physics concepts with cluster validation
        if path in physics_concepts:
            physics_data = physics_concepts[path]
            relevant_paths = physics_data.get("relevant_paths", [])
            key_relationships = physics_data.get("key_relationships", [])

            # Prioritize key relationships
            for rel_path in key_relationships:
                if rel_path not in [r["path"] for r in relationships]:
                    # Check cluster membership for enhanced strength
                    cluster_validated = any(
                        rel_path in cluster.get("paths", [])
                        for cluster in path_clusters
                    )
                    strength = (
                        RelationshipStrength.STRONG
                        if cluster_validated
                        else RelationshipStrength.MODERATE
                    )

                    relationships.append(
                        {
                            "path": rel_path,
                            "type": "physics_key_concept",
                            "distance": 1,
                            "strength": strength,
                            "source": "catalog_physics_key",
                            "cluster_validated": cluster_validated,
                        }
                    )

            # Add relevant paths with lower priority
            for rel_path in relevant_paths:
                if rel_path not in [r["path"] for r in relationships]:
                    relationships.append(
                        {
                            "path": rel_path,
                            "type": "physics_concept",
                            "distance": 1,
                            "strength": RelationshipStrength.MODERATE,
                            "source": "catalog_physics",
                        }
                    )

        # Enhanced unit relationships with clustering context
        for unit_name, unit_data in unit_families.items():
            paths_with_unit = unit_data.get("paths_using", [])
            if path in paths_with_unit:
                for related_path in paths_with_unit:
                    if related_path != path and related_path not in [
                        r["path"] for r in relationships
                    ]:
                        # Check if unit relationship is validated by clustering
                        cluster_validated = any(
                            related_path in cluster.get("paths", [])
                            for cluster in path_clusters
                        )
                        strength = (
                            RelationshipStrength.MODERATE
                            if cluster_validated
                            else RelationshipStrength.WEAK
                        )

                        relationships.append(
                            {
                                "path": related_path,
                                "type": "unit_relationship",
                                "distance": 1,
                                "strength": strength,
                                "source": "catalog_units",
                                "unit": unit_name,
                                "cluster_validated": cluster_validated,
                            }
                        )

        # Sort by strength and limit results
        relationships.sort(key=lambda x: (-x.get("strength", 0), x.get("distance", 1)))
        return relationships[
            : max_depth * 12
        ]  # Increased limit for enhanced clustering data

    def _analyze_semantic_relationships(
        self, path: str, catalog_relationships: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Enhance relationships with semantic analysis."""
        semantic_relationships = []

        for rel in catalog_relationships:
            rel_path = rel["path"]

            # Calculate semantic similarity
            similarity, details = self.semantic_analyzer.calculate_semantic_similarity(
                path, rel_path
            )

            if similarity > 0.1:  # Only include meaningful semantic relationships
                semantic_rel = rel.copy()
                semantic_rel.update(
                    {
                        "type": "semantic",
                        "semantic_similarity": similarity,
                        "semantic_details": details,
                        "strength": max(
                            rel["strength"], similarity
                        ),  # Boost strength with semantic similarity
                        "description": details.get(
                            "semantic_description", "Semantically related"
                        ),
                    }
                )
                semantic_relationships.append(semantic_rel)

        return semantic_relationships

    def _analyze_physics_domain_relationships(
        self, path: str, catalog_relationships: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze physics domain connections."""
        physics_relationships = []
        path_concepts = self.semantic_analyzer.analyze_concept(path)

        for rel in catalog_relationships:
            rel_path = rel["path"]
            rel_concepts = self.semantic_analyzer.analyze_concept(rel_path)

            # Check for physics domain connections
            if path_concepts["primary_domain"] and rel_concepts["primary_domain"]:
                domain1 = path_concepts["primary_domain"]
                domain2 = rel_concepts["primary_domain"]

                strength = RelationshipStrength.WEAK
                description = "Different physics domains"

                if domain1 == domain2:
                    strength = RelationshipStrength.STRONG
                    description = f"Same physics domain: {domain1}"
                elif domain2 in self.semantic_analyzer.DOMAIN_RELATIONSHIPS.get(
                    domain1, []
                ):
                    strength = RelationshipStrength.MODERATE
                    description = f"Related physics domains: {domain1} ↔ {domain2}"

                physics_rel = rel.copy()
                physics_rel.update(
                    {
                        "type": "physics_domain",
                        "physics_domain_source": domain1,
                        "physics_domain_target": domain2,
                        "strength": strength,
                        "description": description,
                    }
                )
                physics_relationships.append(physics_rel)

        return physics_relationships

    def _analyze_measurement_chains(
        self, path: str, catalog_relationships: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze measurement chain relationships."""
        measurement_relationships = []

        # Identify measurement types in the input path
        path_lower = path.lower()
        is_diagnostic = any(
            term in path_lower
            for term in ["diagnostic", "measurement", "sensor", "detector"]
        )

        for rel in catalog_relationships:
            rel_path = rel["path"]
            rel_lower = rel_path.lower()

            # Check for measurement chain connections
            rel_is_diagnostic = any(
                term in rel_lower
                for term in ["diagnostic", "measurement", "sensor", "detector"]
            )

            if is_diagnostic or rel_is_diagnostic:
                measurement_rel = rel.copy()
                measurement_rel.update(
                    {
                        "type": "measurement_chain",
                        "measurement_connection": "diagnostic_chain"
                        if (is_diagnostic and rel_is_diagnostic)
                        else "measurement_data",
                        "strength": RelationshipStrength.MODERATE
                        if (is_diagnostic and rel_is_diagnostic)
                        else RelationshipStrength.WEAK,
                        "description": "Connected through measurement/diagnostic chain",
                    }
                )
                measurement_relationships.append(measurement_rel)

        return measurement_relationships

    def _rank_and_filter_relationships(
        self, all_relationships: dict[str, list[dict[str, Any]]], max_depth: int
    ) -> dict[str, list[dict[str, Any]]]:
        """Rank relationships by strength and filter by max_depth."""
        filtered_relationships = {}

        for rel_type, relationships in all_relationships.items():
            # Sort by strength (descending) and distance (ascending)
            sorted_rels = sorted(
                relationships,
                key=lambda x: (-x.get("strength", 0), x.get("distance", 1)),
            )

            # Limit results per type
            limit = max_depth * 3  # Allow more results for enhanced discovery
            filtered_relationships[rel_type] = sorted_rels[:limit]

        return filtered_relationships

    def generate_physics_context(
        self, path: str, relationships: dict[str, list[dict[str, Any]]]
    ) -> PhysicsContext | None:
        """Generate rich physics context from relationships."""
        path_concepts = self.semantic_analyzer.analyze_concept(path)

        if not path_concepts["primary_domain"]:
            return None

        # Extract phenomena from relationships
        phenomena = []
        for rel_list in relationships.values():
            for rel in rel_list:
                if rel.get("semantic_details", {}).get("shared_concepts"):
                    phenomena.extend(rel["semantic_details"]["shared_concepts"])

        # Extract typical values (simplified implementation)
        typical_values = {}
        domain = path_concepts["primary_domain"]
        if "density" in path.lower():
            typical_values = {"range": "1e19 - 1e21 m^-3", "typical": "5e19 m^-3"}
        elif "temperature" in path.lower():
            typical_values = {"range": "0.1 - 50 keV", "typical": "10 keV"}

        return PhysicsContext(
            domain=domain, phenomena=list(set(phenomena)), typical_values=typical_values
        )


def create_enhanced_relationship_nodes(
    relationships: dict[str, list[dict[str, Any]]],
    physics_context: PhysicsContext | None = None,
) -> list[IdsNode]:
    """Create enhanced IdsNode objects with relationship metadata."""
    nodes = []
    seen_paths = set()

    for _rel_type, rel_list in relationships.items():
        for rel in rel_list:
            rel_path = rel["path"]
            if rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)

            # Generate rich documentation
            strength_category = RelationshipStrength.get_category(
                rel.get("strength", 0)
            )
            description = rel.get(
                "description", f"Related via {rel.get('type', 'unknown')} relationship"
            )

            documentation = f"{description} (strength: {strength_category}, distance: {rel.get('distance', 'unknown')})"

            # Set physics context for relevant nodes
            node_physics_context = None
            if any(
                term in rel_path.lower()
                for term in ["equilibrium", "transport", "heating", "magnetic"]
            ):
                domain = (
                    "equilibrium" if "equilibrium" in rel_path.lower() else "transport"
                )
                node_physics_context = PhysicsContext(
                    domain=domain, phenomena=[], typical_values={}
                )

            node = IdsNode(
                path=rel_path,
                documentation=documentation,
                units="",  # Would need additional data source
                data_type="",  # Would need additional data source
                physics_context=node_physics_context,
            )
            nodes.append(node)

    return nodes
