"""
Clustering functionality for relationship extraction.

Uses HDBSCAN for semantic-first clustering that adapts to varying density
in the embedding space without requiring epsilon tuning.
"""

import logging
from typing import Any, Literal

import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .models import ClusterInfo, PathMembership


def _compute_cluster_similarity(
    cluster_indices: list[int], embeddings: np.ndarray
) -> float:
    """Compute average intra-cluster cosine similarity."""
    if len(cluster_indices) < 2:
        return 1.0  # Single item clusters have perfect similarity

    cluster_embeddings = embeddings[cluster_indices]
    similarity_matrix = cosine_similarity(cluster_embeddings)

    # Get upper triangular part (excluding diagonal) to avoid double counting
    upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[upper_tri_indices]

    # Clamp to [0, 1] to handle floating point precision issues
    avg_similarity = float(np.mean(similarities))
    return min(1.0, max(0.0, avg_similarity))


def _compute_cluster_centroid(
    cluster_indices: list[int], embeddings: np.ndarray, normalize: bool = True
) -> list[float]:
    """Compute the centroid embedding for a cluster.

    Args:
        cluster_indices: Indices of paths in the cluster
        embeddings: Full embeddings matrix
        normalize: Whether to L2-normalize the centroid

    Returns:
        Centroid embedding as a list of floats
    """
    cluster_embeddings = embeddings[cluster_indices]
    centroid = np.mean(cluster_embeddings, axis=0)

    if normalize:
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

    return centroid.tolist()


class EmbeddingClusterer:
    """Performs semantic clustering using HDBSCAN.

    HDBSCAN automatically adapts to varying density in the embedding space,
    eliminating the need for epsilon tuning. Uses a single-pass approach
    that lets semantic similarity naturally determine cluster membership,
    deriving cross-IDS vs intra-IDS properties post-hoc from results.
    """

    def __init__(
        self,
        config,
        logger: logging.Logger | None = None,
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
    ):
        """Initialize the clusterer with configuration.

        Args:
            config: Clustering configuration
            logger: Optional logger instance
            cluster_selection_method: HDBSCAN method - 'eom' for broader clusters,
                                      'leaf' for finer-grained clusters
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.cluster_selection_method = cluster_selection_method

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        path_list: list[str],
        filtered_paths: dict[str, dict[str, Any]],
    ) -> tuple[list[ClusterInfo], dict[str, PathMembership], dict[str, Any]]:
        """
        Perform semantic clustering using HDBSCAN.

        Uses a single-pass approach that clusters all paths based on embedding
        similarity, then derives cross-IDS/intra-IDS properties post-hoc.

        Returns:
            - List of all clusters with is_cross_ids derived from membership
            - Path index mapping each path to its cluster membership
            - Statistics about the clustering process
        """
        if len(path_list) < 2:
            self.logger.warning("Not enough paths for clustering")
            return [], {}, {"error": "insufficient_paths"}

        # Convert embeddings to distance matrix for HDBSCAN
        # HDBSCAN works with distances, use 1 - cosine_similarity
        self.logger.info(
            f"Clustering {len(path_list)} paths with HDBSCAN "
            f"(min_cluster_size=2, method={self.cluster_selection_method})"
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=2,
            metric="euclidean",  # Use euclidean on normalized embeddings
            cluster_selection_method=self.cluster_selection_method,
        )

        # Normalize embeddings for cosine-like behavior with euclidean metric
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / np.where(norms > 0, norms, 1)

        cluster_labels = clusterer.fit_predict(normalized_embeddings)
        probabilities = clusterer.probabilities_

        # Build clusters from labels
        clusters = []
        path_memberships: dict[str, PathMembership] = {}

        unique_labels = set(cluster_labels)
        noise_count = list(cluster_labels).count(-1)
        cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)

        self.logger.info(
            f"HDBSCAN found {cluster_count} clusters, {noise_count} noise points"
        )

        for label in sorted(unique_labels):
            if label == -1:  # Skip noise
                continue

            cluster_indices = [
                i for i, lbl in enumerate(cluster_labels) if lbl == label
            ]
            cluster_paths = [path_list[i] for i in cluster_indices]
            cluster_probs = [probabilities[i] for i in cluster_indices]

            # Compute cluster properties
            similarity_score = _compute_cluster_similarity(cluster_indices, embeddings)
            centroid = _compute_cluster_centroid(cluster_indices, embeddings)

            # Derive IDS membership from paths (post-hoc)
            ids_names = sorted({self._extract_ids_name(p) for p in cluster_paths})
            is_cross_ids = len(ids_names) > 1

            cluster = ClusterInfo(
                id=label,
                similarity_score=similarity_score,
                size=len(cluster_paths),
                is_cross_ids=is_cross_ids,
                ids_names=ids_names,
                paths=cluster_paths,
                centroid=centroid,
            )
            clusters.append(cluster)

            # Record memberships (probability available for future confidence scoring)
            for path, _prob in zip(cluster_paths, cluster_probs, strict=True):
                if is_cross_ids:
                    path_memberships[path] = PathMembership(
                        cross_ids_cluster=label, intra_ids_cluster=None
                    )
                else:
                    path_memberships[path] = PathMembership(
                        cross_ids_cluster=None, intra_ids_cluster=label
                    )

        # Add noise paths with no membership
        for i, label in enumerate(cluster_labels):
            if label == -1:
                path_memberships[path_list[i]] = PathMembership()

        # Calculate statistics
        statistics = self._calculate_statistics(clusters, path_memberships)

        cross_count = sum(1 for c in clusters if c.is_cross_ids)
        intra_count = len(clusters) - cross_count
        self.logger.info(
            f"Clustering complete: {cross_count} cross-IDS, {intra_count} intra-IDS clusters"
        )

        return clusters, path_memberships, statistics

    def _extract_ids_name(self, path: str) -> str:
        """Extract IDS name from path."""
        return path.split("/")[0]

    def _calculate_statistics(
        self,
        clusters: list[ClusterInfo],
        path_index: dict[str, PathMembership],
    ) -> dict[str, Any]:
        """Calculate clustering statistics."""
        cross_clusters = [c for c in clusters if c.is_cross_ids]
        intra_clusters = [c for c in clusters if not c.is_cross_ids]

        multi_membership = sum(
            1
            for m in path_index.values()
            if m.cross_ids_cluster is not None and m.intra_ids_cluster is not None
        )
        isolated = sum(
            1
            for m in path_index.values()
            if m.cross_ids_cluster is None and m.intra_ids_cluster is None
        )

        cross_avg_sim = (
            sum(c.similarity_score for c in cross_clusters) / len(cross_clusters)
            if cross_clusters
            else 0.0
        )
        intra_avg_sim = (
            sum(c.similarity_score for c in intra_clusters) / len(intra_clusters)
            if intra_clusters
            else 0.0
        )

        return {
            "cross_ids_clustering": {
                "total_clusters": len(cross_clusters),
                "paths_in_clusters": sum(c.size for c in cross_clusters),
                "noise_points": isolated,
                "avg_similarity": cross_avg_sim,
            },
            "intra_ids_clustering": {
                "total_clusters": len(intra_clusters),
                "paths_in_clusters": sum(c.size for c in intra_clusters),
                "noise_points": isolated,
                "avg_similarity": intra_avg_sim,
            },
            "multi_membership_paths": multi_membership,
            "isolated_paths": isolated,
        }


class RelationshipBuilder:
    """Builds relationship indices from clustering results."""

    def __init__(self, config, logger: logging.Logger | None = None):
        """Initialize the relationship builder."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def build_path_index(self, cluster_infos: dict[int, ClusterInfo]) -> dict[str, Any]:
        """Build path-to-cluster index for fast lookups (backward compatibility)."""
        path_to_cluster = {}
        cluster_to_paths = {}

        for cluster_id, cluster_info in cluster_infos.items():
            cluster_paths = cluster_info.paths
            cluster_to_paths[cluster_id] = cluster_paths

            for path in cluster_info.paths:
                path_to_cluster[path] = cluster_id

        return {
            "path_to_cluster": path_to_cluster,
            "cluster_to_paths": cluster_to_paths,
        }
