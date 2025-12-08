"""
Semantic search over clusters using centroid embeddings.

This module provides functionality to search clusters by comparing
a query embedding to cluster centroids.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClusterSearchResult:
    """Result from semantic cluster search."""

    cluster_id: int
    similarity_score: float
    is_cross_ids: bool
    ids_names: list[str]
    paths: list[str]
    cluster_similarity: float  # Internal cluster similarity


@dataclass
class ClusterSearcher:
    """Searches clusters using centroid embeddings for semantic similarity."""

    clusters: list[dict[str, Any]]
    centroids: np.ndarray | None = field(default=None, init=False)
    cluster_ids: list[int] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Build centroids matrix from clusters."""
        self._build_centroids_matrix()

    def _build_centroids_matrix(self) -> None:
        """Build a matrix of centroid embeddings for efficient search."""
        centroids_list = []
        cluster_ids = []

        for cluster in self.clusters:
            centroid = cluster.get("centroid")
            if centroid is not None:
                centroids_list.append(centroid)
                cluster_ids.append(cluster["id"])

        if centroids_list:
            self.centroids = np.array(centroids_list, dtype=np.float32)
            self.cluster_ids = cluster_ids
            logger.debug(
                f"Built centroids matrix: {self.centroids.shape[0]} clusters, "
                f"dim={self.centroids.shape[1]}"
            )
        else:
            self.centroids = None
            self.cluster_ids = []
            logger.warning("No centroid embeddings found in clusters")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        cross_ids_only: bool = False,
    ) -> list[ClusterSearchResult]:
        """Search for clusters most similar to the query embedding.

        Args:
            query_embedding: Query embedding vector (normalized)
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score to include
            cross_ids_only: If True, only return cross-IDS clusters

        Returns:
            List of ClusterSearchResult sorted by similarity
        """
        if self.centroids is None or len(self.centroids) == 0:
            logger.warning("No centroids available for search")
            return []

        # Ensure query is normalized
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Compute similarities via dot product (centroids are normalized)
        similarities = np.dot(self.centroids, query_embedding)

        # Build results
        results = []
        for i, sim in enumerate(similarities):
            if sim < similarity_threshold:
                continue

            cluster_id = self.cluster_ids[i]
            cluster = self._get_cluster_by_id(cluster_id)
            if cluster is None:
                continue

            if cross_ids_only and not cluster.get("is_cross_ids", False):
                continue

            results.append(
                ClusterSearchResult(
                    cluster_id=cluster_id,
                    similarity_score=float(sim),
                    is_cross_ids=cluster.get("is_cross_ids", False),
                    ids_names=cluster.get("ids_names", []),
                    paths=cluster.get("paths", []),
                    cluster_similarity=cluster.get("similarity_score", 0.0),
                )
            )

        # Sort by similarity and limit
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    def search_by_text(
        self,
        query: str,
        encoder,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        cross_ids_only: bool = False,
    ) -> list[ClusterSearchResult]:
        """Search clusters using a text query.

        Args:
            query: Text query to search for
            encoder: Encoder instance to generate query embedding
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity
            cross_ids_only: Only return cross-IDS clusters

        Returns:
            List of ClusterSearchResult
        """
        query_embedding = encoder.embed_texts([query])[0]
        return self.search(
            query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            cross_ids_only=cross_ids_only,
        )

    def _get_cluster_by_id(self, cluster_id: int) -> dict[str, Any] | None:
        """Get cluster by ID."""
        for cluster in self.clusters:
            if cluster["id"] == cluster_id:
                return cluster
        return None

    def get_similar_clusters(
        self,
        cluster_id: int,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> list[ClusterSearchResult]:
        """Find clusters similar to a given cluster.

        Args:
            cluster_id: ID of the source cluster
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity

        Returns:
            List of similar clusters (excluding the source)
        """
        cluster = self._get_cluster_by_id(cluster_id)
        if cluster is None or cluster.get("centroid") is None:
            return []

        centroid = np.array(cluster["centroid"], dtype=np.float32)
        results = self.search(
            centroid,
            top_k=top_k + 1,  # +1 to account for self
            similarity_threshold=similarity_threshold,
        )

        # Remove the source cluster from results
        return [r for r in results if r.cluster_id != cluster_id][:top_k]
