"""
Data models for cluster-based relationship extraction results.

This module defines the core data structures for cluster-based
semantic groupings of IMAS paths, including:
- ClusterInfo: Individual cluster with scope and enrichment metadata
- ClusterScope: Hierarchical scope (global, domain, ids)
- PathMembership: Multi-cluster membership tracking
- RelationshipSet: Complete clustering results
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Type aliases for clarity
ClusterScope = Literal["global", "domain", "ids"]
MappingRelevanceLevel = Literal["high", "medium", "low"]


class ClusterInfo(BaseModel):
    """Information about a cluster of semantically related paths.

    Clusters can exist at three hierarchical levels:
    - global: Discovered from full embedding space (all IDS)
    - domain: Discovered within a physics domain (e.g., transport)
    - ids: Discovered within a single IDS (e.g., core_profiles)

    Enrichment fields are populated by LLM labeling using controlled
    vocabularies from definitions/clusters/*.yaml.
    """

    # Core identification (UUID string)
    id: str  # UUID for unique identification across all scopes
    similarity_score: float = Field(ge=0, le=1)
    size: int = Field(ge=1)
    is_cross_ids: bool
    ids_names: list[str]
    paths: list[str]
    centroid: list[float] | None = None  # Centroid embedding for semantic search

    # Hierarchical scope (NEW)
    scope: ClusterScope = "global"
    scope_detail: str | None = None  # Domain name or IDS name for domain/ids scopes

    # LLM-derived labels (existing, kept for compatibility)
    label: str | None = None
    description: str | None = None

    # LLM-derived enrichment (NEW)
    physics_concepts: list[str] = Field(default_factory=list)
    data_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    mapping_relevance: MappingRelevanceLevel = "medium"

    # Composite ranking score (computed, NEW)
    relevance_score: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context):
        """Validate and compute derived fields."""
        self.size = len(self.paths)

        # Compute IDS names from paths - handle both dot and slash notation
        ids_set = set()
        for path in self.paths:
            # Handle both "core_profiles.profiles_1d.0.psi" and "core_profiles/profiles_1d/psi" formats
            if "." in path:
                ids_name = path.split(".")[0]
            else:
                ids_name = path.split("/")[0]
            ids_set.add(ids_name)
        self.ids_names = sorted(ids_set)

        # Check if cross-IDS
        self.is_cross_ids = len(self.ids_names) > 1

    def compute_relevance_score(self, query_similarity: float) -> float:
        """Compute composite relevance score for ranking.

        Blends semantic similarity with mapping relevance, cluster type,
        and scope to produce a unified ranking score.

        Args:
            query_similarity: Cosine similarity to query (0-1)

        Returns:
            Composite relevance score (0-1)
        """
        # Base: semantic similarity (50% weight)
        score = query_similarity * 0.5

        # Mapping relevance boost (up to 25%)
        relevance_weights = {"high": 0.25, "medium": 0.15, "low": 0.0}
        score += relevance_weights.get(self.mapping_relevance, 0.1)

        # Cross-IDS boost (10%) - broader applicability
        if self.is_cross_ids:
            score += 0.1

        # Scope boost - higher abstraction levels
        scope_weights = {"global": 0.1, "domain": 0.05, "ids": 0.0}
        score += scope_weights.get(self.scope, 0.0)

        # Internal coherence boost (5%)
        score += self.similarity_score * 0.05

        self.relevance_score = min(1.0, score)
        return self.relevance_score


class PathMembership(BaseModel):
    """Tracks which clusters a path belongs to."""

    cross_ids_cluster: str | None = None  # UUID
    intra_ids_cluster: str | None = None  # UUID


class CrossIDSSummary(BaseModel):
    """Summary of cross-IDS clustering results."""

    cluster_count: int
    cluster_index: list[str]  # UUIDs
    avg_similarity: float
    total_paths: int


class IntraIDSSummary(BaseModel):
    """Summary of intra-IDS clustering results."""

    cluster_count: int
    cluster_index: list[str]  # UUIDs
    by_ids: dict[str, dict[str, Any]]
    avg_similarity: float
    total_paths: int


class ClusteringStatistics(BaseModel):
    """Statistics about the clustering process."""

    cross_ids_clustering: dict[str, Any]
    intra_ids_clustering: dict[str, Any]
    multi_membership_paths: int
    isolated_paths: int


class ClusteringParameters(BaseModel):
    """Parameters used for clustering."""

    eps: float
    min_samples: int
    metric: str = "cosine"


class UnitFamily(BaseModel):
    """Group of paths sharing the same units."""

    base_unit: str
    paths_using: list[str]


class RelationshipMetadata(BaseModel):
    """Metadata about the relationship extraction process."""

    generation_timestamp: str
    total_paths_processed: int
    clustering_parameters: dict[str, ClusteringParameters]
    statistics: ClusteringStatistics


class PathToClusterIndex(BaseModel):
    """Index mapping each path to its cluster for fast lookup."""

    path_to_cluster: dict[str, int]  # path -> cluster_id
    cluster_to_paths: dict[int, list[str]]  # cluster_id -> list of paths

    def get_cluster_for_path(self, path: str) -> int | None:
        """Get cluster ID for a given path."""
        return self.path_to_cluster.get(path)

    def get_related_paths(self, path: str, exclude_same_ids: bool = False) -> list[str]:
        """Get all paths in the same cluster as the given path."""
        cluster_id = self.get_cluster_for_path(path)
        if cluster_id is None:
            return []

        related_paths = self.cluster_to_paths.get(cluster_id, [])

        if exclude_same_ids:
            # This would require access to path metadata - implement in RelationshipSet
            pass

        return [p for p in related_paths if p != path]


class RelationshipSet(BaseModel):
    """Complete set of cluster-based relationships extracted from IMAS data."""

    metadata: RelationshipMetadata
    clusters: list[ClusterInfo]
    path_index: dict[str, PathMembership]
    cross_ids_summary: CrossIDSSummary
    intra_ids_summary: IntraIDSSummary

    # Optional additional groupings for tool compatibility
    _unit_families: dict[str, dict[str, Any]] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_cluster_for_path(self, path: str) -> ClusterInfo | None:
        """Get the cluster containing the given path."""
        membership = self.path_index.get(path)
        if not membership:
            return None

        # Try cross-IDS first, then intra-IDS
        cluster_id = membership.cross_ids_cluster or membership.intra_ids_cluster
        if cluster_id is not None:
            for cluster in self.clusters:
                if cluster.id == cluster_id:
                    return cluster
        return None

    def get_related_paths(
        self, path: str, exclude_same_ids: bool = False, max_results: int | None = None
    ) -> list[str]:
        """Get all paths related to the given path (in same cluster)."""
        cluster = self.get_cluster_for_path(path)
        if cluster is None:
            return []

        related_paths = []
        path_ids = None

        # Find the IDS of the query path
        if path in cluster.paths:
            path_ids = path.split(".")[0]

        for cluster_path in cluster.paths:
            if cluster_path != path:
                if exclude_same_ids and path_ids:
                    cluster_path_ids = cluster_path.split(".")[0]
                    if cluster_path_ids == path_ids:
                        continue
                related_paths.append(cluster_path)

        if max_results:
            related_paths = related_paths[:max_results]

        return related_paths

    def get_cross_ids_clusters(self) -> list[ClusterInfo]:
        """Get all clusters that contain paths from multiple IDS."""
        return [cluster for cluster in self.clusters if cluster.is_cross_ids]

    def get_clusters_for_ids(self, ids_name: str) -> list[ClusterInfo]:
        """Get all clusters containing paths from the specified IDS."""
        return [cluster for cluster in self.clusters if ids_name in cluster.ids_names]
