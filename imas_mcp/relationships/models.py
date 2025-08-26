"""
Data models for relationship extraction results.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RelationshipResult:
    """A single relationship between two paths."""

    path: str
    type: str
    similarity_score: float
    cluster_id: int | None = None
    source_path: str | None = None
    relationship_type: str = "semantic_cluster"


@dataclass
class ClusterInfo:
    """Information about a cluster of related paths."""

    cluster_id: int
    paths: list[str]
    centroid: Any  # numpy array
    size: int

    def __post_init__(self):
        """Validate cluster info."""
        self.size = len(self.paths)


@dataclass
class CrossReference:
    """Cross-reference relationship between IDS."""

    type: str
    relationships: list[RelationshipResult]


@dataclass
class UnitFamily:
    """Group of paths sharing the same units."""

    base_unit: str
    paths_using: list[str]
    conversion_factors: dict[str, float]


@dataclass
class RelationshipMetadata:
    """Metadata about the relationship extraction process."""

    total_clusters: int
    total_relationships: int
    clustering_method: str = "DBSCAN"
    similarity_threshold: float = 0.7
    total_paths_processed: int = 0
    noise_points: int = 0


@dataclass
class RelationshipSet:
    """Complete set of relationships extracted from IMAS data."""

    metadata: RelationshipMetadata
    cross_references: dict[str, CrossReference]
    physics_concepts: dict[str, Any]  # Can be expanded later
    unit_families: dict[str, UnitFamily]

    def get_relationships_for_path(self, path: str) -> list[RelationshipResult]:
        """Get all relationships for a specific path."""
        if path in self.cross_references:
            return self.cross_references[path].relationships
        return []

    def get_related_paths(self, path: str, max_results: int = 5) -> list[str]:
        """Get list of related paths for a given path."""
        relationships = self.get_relationships_for_path(path)
        return [rel.path for rel in relationships[:max_results]]
