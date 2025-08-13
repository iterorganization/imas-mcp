"""
Relationship analysis engine for IMAS physics data.

This module provides relationship discovery and analysis between IMAS data paths
using external physics definitions and semantic analysis.
"""

import logging
from enum import Enum
from typing import Any

from imas_mcp.core.data_model import IdsNode, PhysicsContext
from imas_mcp.physics_extraction.physics_data_loader import PhysicsDataLoader

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of relationships that can be discovered."""

    ALL = "all"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    PHYSICS = "physics"
    MEASUREMENT = "measurement"


class RelationshipStrength:
    """Relationship strength categories and classification."""

    VERY_STRONG = 0.9
    STRONG = 0.7
    MODERATE = 0.5
    WEAK = 0.3
    VERY_WEAK = 0.1

    @classmethod
    def get_category(cls, strength: float) -> str:
        """Get the category name for a strength value."""
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


class SemanticAnalyzer:
    """Semantic analysis for physics relationships using external definitions."""

    def __init__(self, physics_loader: PhysicsDataLoader | None = None):
        """Initialize semantic analyzer.

        Args:
            physics_loader: Physics data loader. If None, creates default.
        """
        self.physics_loader = physics_loader or PhysicsDataLoader()
        self._concept_cache = {}
        self._physics_concepts = None
        self._domain_relationships = None

    @property
    def physics_concepts(self) -> dict[str, dict[str, Any]]:
        """Get physics concepts from external definitions."""
        if self._physics_concepts is None:
            self._physics_concepts = self.physics_loader.generate_physics_concepts()
        return self._physics_concepts

    @property
    def domain_relationships(self) -> dict[str, list[str]]:
        """Get domain relationships from external definitions."""
        if self._domain_relationships is None:
            self._domain_relationships = self.physics_loader.load_domain_relationships()
        return self._domain_relationships

    def analyze_concept(self, path: str) -> dict[str, Any]:
        """Extract physics concepts from a path."""
        if path in self._concept_cache:
            return self._concept_cache[path]

        path_lower = path.lower()
        concepts = []
        primary_domain = None

        # Extract concepts from path using external definitions
        for concept, data in self.physics_concepts.items():
            if concept in path_lower:
                concepts.append(concept)
                if not primary_domain:
                    primary_domain = data["domain"]

        # Detect measurement types
        measurement_types = []
        for concept_data in self.physics_concepts.values():
            for mtype in concept_data.get("measurement_types", []):
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
            elif domain2 in self.domain_relationships.get(domain1, []):
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


class RelationshipEngine:
    """Main relationship discovery engine for IMAS physics data."""

    def __init__(
        self,
        relationships_catalog: dict[str, Any],
        physics_loader: PhysicsDataLoader | None = None,
    ):
        """Initialize relationship engine.

        Args:
            relationships_catalog: Catalog of structural relationships
            physics_loader: Physics data loader for external definitions
        """
        self.relationships_catalog = relationships_catalog
        self.physics_loader = physics_loader or PhysicsDataLoader()
        self.semantic_analyzer = SemanticAnalyzer(self.physics_loader)

    def discover_relationships(
        self,
        path: str,
        relationship_type: RelationshipType = RelationshipType.ALL,
        max_depth: int = 2,
    ) -> dict[str, list[dict[str, Any]]]:
        """Discover relationships for a given path.

        Args:
            path: IMAS path to analyze
            relationship_type: Type of relationships to find
            max_depth: Maximum relationship depth

        Returns:
            Dictionary with relationship types as keys and lists of relationships as values
        """
        results = {"semantic": [], "structural": [], "physics": [], "measurement": []}

        try:
            # Get candidate paths from catalog
            candidate_paths = self._get_candidate_paths(path)

            if relationship_type in (RelationshipType.ALL, RelationshipType.STRUCTURAL):
                results["structural"] = self._get_catalog_relationships(path, max_depth)

            if relationship_type in (RelationshipType.ALL, RelationshipType.SEMANTIC):
                results["semantic"] = self._analyze_semantic_relationships(
                    path, candidate_paths, max_depth
                )

            if relationship_type in (RelationshipType.ALL, RelationshipType.PHYSICS):
                results["physics"] = self._analyze_physics_domain_relationships(
                    path, candidate_paths, max_depth
                )

            if relationship_type in (
                RelationshipType.ALL,
                RelationshipType.MEASUREMENT,
            ):
                results["measurement"] = self._analyze_measurement_chains(
                    path, candidate_paths, max_depth
                )

            # Rank and filter results
            for rel_type in results:
                results[rel_type] = self._rank_and_filter_relationships(
                    results[rel_type]
                )

        except Exception as e:
            logger.error(f"Error discovering relationships for {path}: {e}")

        return results

    def _get_candidate_paths(self, path: str) -> list[str]:
        """Get candidate paths for relationship analysis."""
        candidates = set()

        # Add paths from catalog cross-references
        cross_refs = self.relationships_catalog.get("cross_references", {})
        if path in cross_refs:
            for rel in cross_refs[path].get("relationships", []):
                candidates.add(rel["path"])

        # Add paths from physics concepts
        physics_concepts = self.relationships_catalog.get("physics_concepts", {})
        if path in physics_concepts:
            candidates.update(physics_concepts[path].get("relevant_paths", []))

        # Add paths from unit families
        unit_families = self.relationships_catalog.get("unit_families", {})
        for unit_data in unit_families.values():
            paths_using = unit_data.get("paths_using", [])
            if path in paths_using:
                candidates.update(paths_using)

        # Remove the original path
        candidates.discard(path)
        return list(candidates)

    def _get_catalog_relationships(
        self, path: str, max_depth: int
    ) -> list[dict[str, Any]]:
        """Get structural relationships from catalog."""
        relationships = []

        # Direct cross-references
        cross_refs = self.relationships_catalog.get("cross_references", {})
        if path in cross_refs:
            for rel in cross_refs[path].get("relationships", []):
                relationships.append(
                    {
                        "path": rel["path"],
                        "type": rel.get("type", "cross_reference"),
                        "source": "catalog_direct",
                        "strength": RelationshipStrength.STRONG,
                        "distance": 1,
                    }
                )

        # Physics concept relationships
        physics_concepts = self.relationships_catalog.get("physics_concepts", {})
        if path in physics_concepts:
            for related_path in physics_concepts[path].get("relevant_paths", []):
                relationships.append(
                    {
                        "path": related_path,
                        "type": "physics_concept",
                        "source": "catalog_physics",
                        "strength": RelationshipStrength.MODERATE,
                        "distance": 1,
                    }
                )

        # Unit-based relationships
        unit_families = self.relationships_catalog.get("unit_families", {})
        for unit, unit_data in unit_families.items():
            paths_using = unit_data.get("paths_using", [])
            if path in paths_using:
                for related_path in paths_using:
                    if related_path != path:
                        relationships.append(
                            {
                                "path": related_path,
                                "type": "unit_family",
                                "source": "catalog_units",
                                "strength": RelationshipStrength.WEAK,
                                "distance": 1,
                                "unit": unit,
                            }
                        )

        return relationships

    def _analyze_semantic_relationships(
        self, path: str, candidate_paths: list[str], max_depth: int
    ) -> list[dict[str, Any]]:
        """Analyze semantic relationships using physics concepts."""
        relationships = []

        for candidate in candidate_paths:
            similarity, details = self.semantic_analyzer.calculate_semantic_similarity(
                path, candidate
            )

            if similarity > 0.1:  # Threshold for semantic relationships
                relationships.append(
                    {
                        "path": candidate,
                        "type": "semantic",
                        "strength": similarity,
                        "distance": 1,
                        "semantic_details": details,
                        "description": details.get(
                            "semantic_description", "Semantically related"
                        ),
                    }
                )

        return relationships

    def _analyze_physics_domain_relationships(
        self, path: str, candidate_paths: list[str], max_depth: int
    ) -> list[dict[str, Any]]:
        """Analyze physics domain relationships."""
        relationships = []
        path_concept = self.semantic_analyzer.analyze_concept(path)
        path_domain = path_concept.get("primary_domain")

        if not path_domain:
            return relationships

        for candidate in candidate_paths:
            candidate_concept = self.semantic_analyzer.analyze_concept(candidate)
            candidate_domain = candidate_concept.get("primary_domain")

            if candidate_domain:
                if candidate_domain == path_domain:
                    relationships.append(
                        {
                            "path": candidate,
                            "type": "physics_domain",
                            "strength": RelationshipStrength.STRONG,
                            "distance": 1,
                            "physics_domain_source": path_domain,
                            "physics_domain_target": candidate_domain,
                            "description": f"Same physics domain: {path_domain}",
                        }
                    )
                elif (
                    candidate_domain
                    in self.semantic_analyzer.domain_relationships.get(path_domain, [])
                ):
                    relationships.append(
                        {
                            "path": candidate,
                            "type": "physics_domain",
                            "strength": RelationshipStrength.MODERATE,
                            "distance": 1,
                            "physics_domain_source": path_domain,
                            "physics_domain_target": candidate_domain,
                            "description": f"Related physics domains: {path_domain} -> {candidate_domain}",
                        }
                    )

        return relationships

    def _analyze_measurement_chains(
        self, path: str, candidate_paths: list[str], max_depth: int
    ) -> list[dict[str, Any]]:
        """Analyze measurement chain relationships."""
        relationships = []
        path_concept = self.semantic_analyzer.analyze_concept(path)
        path_measurements = set(path_concept.get("measurement_types", []))

        for candidate in candidate_paths:
            candidate_concept = self.semantic_analyzer.analyze_concept(candidate)
            candidate_measurements = set(candidate_concept.get("measurement_types", []))

            shared_measurements = path_measurements & candidate_measurements
            if shared_measurements:
                relationships.append(
                    {
                        "path": candidate,
                        "type": "measurement_chain",
                        "strength": RelationshipStrength.MODERATE,
                        "distance": 1,
                        "shared_measurements": list(shared_measurements),
                        "description": f"Shared measurement types: {', '.join(shared_measurements)}",
                    }
                )

        return relationships

    def _rank_and_filter_relationships(
        self, relationships: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Rank and filter relationships by strength and relevance."""
        # Sort by strength (descending)
        relationships.sort(key=lambda x: x.get("strength", 0), reverse=True)

        # Remove duplicates (same path)
        seen_paths = set()
        filtered = []
        for rel in relationships:
            path = rel["path"]
            if path not in seen_paths:
                seen_paths.add(path)
                filtered.append(rel)

        return filtered

    def generate_physics_context(
        self, path: str, relationships: dict[str, list[dict[str, Any]]]
    ) -> PhysicsContext | None:
        """Generate physics context for a path and its relationships."""
        try:
            path_concept = self.semantic_analyzer.analyze_concept(path)
            primary_domain = path_concept.get("primary_domain")

            if not primary_domain:
                return None

            domain_characteristics = self.physics_loader.get_domain_characteristics(
                primary_domain
            )

            return PhysicsContext(
                domain=primary_domain,
                phenomena=domain_characteristics.get("primary_phenomena", []),
                measurement_methods=domain_characteristics.get(
                    "measurement_methods", []
                ),
                related_domains=domain_characteristics.get("related_domains", []),
                description=domain_characteristics.get("description", ""),
                typical_units=domain_characteristics.get("typical_units", []),
            )
        except Exception as e:
            logger.error(f"Error generating physics context for {path}: {e}")
            return None


def create_relationship_nodes(
    path: str,
    relationships: dict[str, list[dict[str, Any]]],
    physics_context: PhysicsContext | None = None,
) -> list[IdsNode]:
    """Create IdsNode objects from relationship analysis results.

    Args:
        path: Source path
        relationships: Relationship analysis results
        physics_context: Physics context for the path

    Returns:
        List of IdsNode objects representing related paths
    """
    nodes = []

    for rel_type, rel_list in relationships.items():
        for rel in rel_list:
            try:
                node = IdsNode(
                    path=rel["path"],
                    name=rel["path"].split("/")[-1],
                    ids_name=rel["path"].split("/")[0]
                    if "/" in rel["path"]
                    else rel["path"],
                    level=rel.get("distance", 1),
                    parent_path="/".join(rel["path"].split("/")[:-1])
                    if "/" in rel["path"]
                    else "",
                    data_type="relationship",
                    description=rel.get(
                        "description", f"{rel_type.title()} relationship"
                    ),
                    units="",
                    dimensions="",
                    lifecycle_status="",
                    documentation="",
                )

                # Add relationship-specific metadata
                node.relationship_type = rel_type
                node.relationship_strength = rel.get("strength", 0)
                node.relationship_source = rel.get("source", "analysis")

                if physics_context:
                    node.physics_context = physics_context

                nodes.append(node)
            except Exception as e:
                logger.error(f"Error creating node for relationship {rel}: {e}")

    return nodes
