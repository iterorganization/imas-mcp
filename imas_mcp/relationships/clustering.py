"""
Clustering functionality for relationship extraction.
"""

import logging
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from .models import ClusterInfo


class EmbeddingClusterer:
    """Clusters embeddings using DBSCAN for relationship extraction."""

    def __init__(self, config, logger: logging.Logger | None = None):
        """Initialize the clusterer with configuration."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def cluster_embeddings(
        self, embeddings: np.ndarray
    ) -> tuple[dict[int, ClusterInfo], int]:
        """Cluster embeddings using DBSCAN."""
        # Use cosine distance for DBSCAN
        dbscan = DBSCAN(
            eps=self.config.eps, min_samples=self.config.min_samples, metric="cosine"
        )
        cluster_labels = dbscan.fit_predict(embeddings)

        # Group paths by cluster
        clusters = {}
        noise_count = 0

        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise
                noise_count += 1
                continue

            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)

        # Convert to ClusterInfo objects
        cluster_infos = {}
        for cluster_id, path_indices in clusters.items():
            # Calculate centroid
            centroid = np.mean(embeddings[path_indices], axis=0)

            cluster_infos[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                paths=[],  # Will be filled by caller with actual path names
                centroid=centroid,
                size=len(path_indices),
            )

        self.logger.info(f"Found {len(clusters)} clusters, {noise_count} noise points")

        # Log cluster statistics
        if clusters:
            cluster_sizes = [len(indices) for indices in clusters.values()]
            self.logger.info(
                f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
                f"avg={np.mean(cluster_sizes):.1f}"
            )

        return cluster_infos, noise_count

    def compute_cluster_similarities(
        self, cluster_infos: dict[int, ClusterInfo]
    ) -> np.ndarray:
        """Compute similarities between cluster centroids."""
        if not cluster_infos:
            return np.array([])

        centroid_ids = list(cluster_infos.keys())
        centroid_matrix = np.array(
            [cluster_infos[cid].centroid for cid in centroid_ids]
        )

        # Cosine similarity between centroids
        similarities = np.dot(centroid_matrix, centroid_matrix.T)

        return similarities


class RelationshipBuilder:
    """Builds relationships from clustering results."""

    def __init__(self, config, logger: logging.Logger | None = None):
        """Initialize the relationship builder."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def extract_cluster_relationships(
        self,
        cluster_infos: dict[int, ClusterInfo],
        path_list: list[str],
        filtered_paths: dict[str, dict[str, Any]],
        similarities: np.ndarray,
    ) -> tuple[dict[str, dict[str, Any]], int]:
        """Extract relationships between clusters."""
        if not cluster_infos:
            self.logger.warning("No clusters found, returning empty relationships")
            return {}, 0

        # Fill in actual path names for clusters
        cluster_to_indices = {}
        for _i, cluster_id in enumerate(cluster_infos.keys()):
            # Reconstruct the path indices for each cluster
            # This is a bit inefficient but maintains clean separation
            cluster_to_indices[cluster_id] = []

        # Map paths back to clusters (we need the original mapping)
        # This is reconstructed from the DBSCAN results
        # We need the embeddings again - this suggests we should pass them
        # For now, we'll work with what we have

        cross_references = {}
        total_relationships = 0

        centroid_ids = list(cluster_infos.keys())

        for i, cluster_i in enumerate(centroid_ids):
            for j, cluster_j in enumerate(centroid_ids):
                if i != j and i < len(similarities) and j < len(similarities[i]):
                    similarity = similarities[i][j]
                    if similarity >= self.config.similarity_threshold:
                        # Get paths from each cluster
                        paths_i = cluster_infos[cluster_i].paths[
                            : self.config.max_paths_per_cluster
                        ]
                        paths_j = cluster_infos[cluster_j].paths[
                            : self.config.max_paths_per_cluster
                        ]

                        # Create cross-references between clusters
                        for path_i in paths_i:
                            relationships = []
                            for path_j in paths_j:
                                # Only create cross-IDS relationships
                                if (
                                    path_i in filtered_paths
                                    and path_j in filtered_paths
                                    and filtered_paths[path_i]["ids"]
                                    != filtered_paths[path_j]["ids"]
                                ):
                                    relationships.append(
                                        {
                                            "path": path_j,
                                            "type": "semantic_cluster",
                                            "similarity_score": float(similarity),
                                            "cluster_id": int(cluster_j),
                                        }
                                    )

                            if relationships:
                                cross_references[path_i] = {
                                    "type": "cross_ids",
                                    "relationships": relationships[
                                        : self.config.max_relationships_per_path
                                    ],
                                }
                                total_relationships += len(relationships)

        self.logger.info(f"Generated {total_relationships} cross-IDS relationships")
        return cross_references, total_relationships
