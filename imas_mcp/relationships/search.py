"""
Semantic search over clusters using centroid and label embeddings.

This module provides functionality to search clusters by comparing
a query embedding to cluster centroids or label embeddings.
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
    label: str = ""
    description: str = ""


@dataclass
class ClusterSearcher:
    """Searches clusters using centroid and label embeddings for semantic similarity."""

    clusters: list[dict[str, Any]]
    centroids: np.ndarray | None = field(default=None, init=False)
    label_embeddings: np.ndarray | None = field(default=None, init=False)
    cluster_ids: list[int] = field(default_factory=list, init=False)
    label_cluster_ids: list[int] = field(default_factory=list, init=False)
    _indexes: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Build embedding matrices and indexes from clusters."""
        self._build_centroids_matrix()
        self._build_label_embeddings_matrix()
        self._build_indexes()

    def _build_centroids_matrix(self) -> None:
        """Build a matrix of centroid embeddings for efficient search."""
        centroids_list = []
        cluster_ids = []

        for cluster in self.clusters:
            centroid = cluster.get("centroid")
            if centroid is not None and len(centroid) > 0:
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

    def _build_label_embeddings_matrix(self) -> None:
        """Build a matrix of label embeddings for natural language search."""
        embeddings_list = []
        cluster_ids = []

        for cluster in self.clusters:
            label_embedding = cluster.get("label_embedding")
            if label_embedding is not None and len(label_embedding) > 0:
                embeddings_list.append(label_embedding)
                cluster_ids.append(cluster["id"])

        if embeddings_list:
            self.label_embeddings = np.array(embeddings_list, dtype=np.float32)
            self.label_cluster_ids = cluster_ids
            logger.debug(
                f"Built label embeddings matrix: {self.label_embeddings.shape[0]} clusters"
            )
        else:
            self.label_embeddings = None
            self.label_cluster_ids = []
            logger.debug("No label embeddings found in clusters")

    def _build_indexes(self) -> None:
        """Build path and IDS indexes for fast lookup."""
        path_index: dict[str, list[int]] = {}
        ids_index: dict[str, list[int]] = {}

        for cluster in self.clusters:
            cluster_id = cluster["id"]

            # Path index
            for path in cluster.get("paths", []):
                if path not in path_index:
                    path_index[path] = []
                path_index[path].append(cluster_id)

            # IDS index
            ids_names = cluster.get("ids_names", cluster.get("ids", []))
            for ids_name in ids_names:
                if ids_name not in ids_index:
                    ids_index[ids_name] = []
                if cluster_id not in ids_index[ids_name]:
                    ids_index[ids_name].append(cluster_id)

        self._indexes = {
            "path": path_index,
            "ids": ids_index,
        }

    def search_by_path(self, path: str) -> list[ClusterSearchResult]:
        """Find clusters containing a specific path.

        Args:
            path: IMAS path to search for

        Returns:
            List of clusters containing this path
        """
        cluster_ids = self._indexes.get("path", {}).get(path, [])
        results = []
        for cluster_id in cluster_ids:
            cluster = self._get_cluster_by_id(cluster_id)
            if cluster:
                results.append(self._cluster_to_result(cluster, similarity_score=1.0))
        return results

    def search_by_ids(
        self, ids_name: str, cross_ids_only: bool = False
    ) -> list[ClusterSearchResult]:
        """Find clusters containing paths from a specific IDS.

        Args:
            ids_name: IDS name to search for
            cross_ids_only: Only return cross-IDS clusters

        Returns:
            List of clusters containing this IDS
        """
        cluster_ids = self._indexes.get("ids", {}).get(ids_name, [])
        results = []
        for cluster_id in cluster_ids:
            cluster = self._get_cluster_by_id(cluster_id)
            if cluster:
                is_cross = cluster.get(
                    "is_cross_ids", cluster.get("type") == "cross_ids"
                )
                if cross_ids_only and not is_cross:
                    continue
                results.append(self._cluster_to_result(cluster, similarity_score=1.0))
        return results

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        cross_ids_only: bool = False,
        use_labels: bool = False,
    ) -> list[ClusterSearchResult]:
        """Search for clusters most similar to the query embedding.

        Args:
            query_embedding: Query embedding vector (normalized)
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score to include
            cross_ids_only: If True, only return cross-IDS clusters
            use_labels: If True, search label embeddings instead of centroids

        Returns:
            List of ClusterSearchResult sorted by similarity
        """
        if use_labels:
            embeddings = self.label_embeddings
            cluster_ids = self.label_cluster_ids
        else:
            embeddings = self.centroids
            cluster_ids = self.cluster_ids

        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings available for search")
            return []

        # Ensure query is normalized
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Compute similarities via dot product (embeddings are normalized)
        similarities = np.dot(embeddings, query_embedding)

        # Build results
        results = []
        for i, sim in enumerate(similarities):
            if sim < similarity_threshold:
                continue

            cluster_id = cluster_ids[i]
            cluster = self._get_cluster_by_id(cluster_id)
            if cluster is None:
                continue

            is_cross = cluster.get("is_cross_ids", cluster.get("type") == "cross_ids")
            if cross_ids_only and not is_cross:
                continue

            results.append(self._cluster_to_result(cluster, float(sim)))

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

        Automatically detects if query is a path (contains '/') or natural language.
        For paths, uses exact lookup. For natural language, searches both
        centroids and label embeddings.

        Args:
            query: Text query to search for (path or natural language)
            encoder: Encoder instance to generate query embedding
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity
            cross_ids_only: Only return cross-IDS clusters

        Returns:
            List of ClusterSearchResult
        """
        # Detect if query is a path
        if "/" in query and " " not in query:
            # Path lookup
            results = self.search_by_path(query)
            if cross_ids_only:
                results = [r for r in results if r.is_cross_ids]
            return results[:top_k]

        # Natural language query - search both centroids and labels
        query_embedding = encoder.embed_texts([query])[0]

        # Search centroids (physics concept similarity)
        centroid_results = self.search(
            query_embedding,
            top_k=top_k * 2,
            similarity_threshold=similarity_threshold,
            cross_ids_only=cross_ids_only,
            use_labels=False,
        )

        # Search label embeddings if available
        label_results = []
        if self.label_embeddings is not None:
            label_results = self.search(
                query_embedding,
                top_k=top_k * 2,
                similarity_threshold=similarity_threshold,
                cross_ids_only=cross_ids_only,
                use_labels=True,
            )

        # Merge and deduplicate results
        seen_ids = set()
        merged = []

        # Interleave results, preferring higher similarity
        all_results = centroid_results + label_results
        all_results.sort(key=lambda r: r.similarity_score, reverse=True)

        for result in all_results:
            if result.cluster_id not in seen_ids:
                seen_ids.add(result.cluster_id)
                merged.append(result)
                if len(merged) >= top_k:
                    break

        return merged

    def _get_cluster_by_id(self, cluster_id: int) -> dict[str, Any] | None:
        """Get cluster by ID."""
        for cluster in self.clusters:
            if cluster["id"] == cluster_id:
                return cluster
        return None

    def _cluster_to_result(
        self, cluster: dict[str, Any], similarity_score: float
    ) -> ClusterSearchResult:
        """Convert cluster dict to ClusterSearchResult."""
        is_cross = cluster.get("is_cross_ids", cluster.get("type") == "cross_ids")
        ids_names = cluster.get("ids_names", cluster.get("ids", []))
        cluster_sim = cluster.get("similarity_score", cluster.get("similarity", 0.0))

        return ClusterSearchResult(
            cluster_id=cluster["id"],
            similarity_score=similarity_score,
            is_cross_ids=is_cross,
            ids_names=ids_names,
            paths=cluster.get("paths", []),
            cluster_similarity=cluster_sim,
            label=cluster.get("label", ""),
            description=cluster.get("description", ""),
        )

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
