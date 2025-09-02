"""
Clustering functionality for relationship extraction.
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
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


class EmbeddingClusterer:
    """Performs multi-membership clustering with separate cross-IDS and intra-IDS stages."""

    def __init__(self, config, logger: logging.Logger | None = None):
        """Initialize the clusterer with configuration."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        path_list: list[str],
        filtered_paths: dict[str, dict[str, Any]],
    ) -> tuple[list[ClusterInfo], dict[str, PathMembership], dict[str, Any]]:
        """
        Perform multi-membership clustering with separate cross-IDS and intra-IDS stages.

        Returns:
            - List of all clusters (both cross-IDS and intra-IDS)
            - Path index mapping each path to its cluster memberships
            - Statistics about the clustering process
        """
        # Stage 1: Cross-IDS clustering
        cross_clusters, cross_memberships = self._cluster_cross_ids(
            embeddings, path_list, filtered_paths
        )

        # Stage 2: Intra-IDS clustering
        intra_clusters, intra_memberships = self._cluster_intra_ids(
            embeddings, path_list, filtered_paths
        )

        # Combine clusters
        all_clusters = cross_clusters + intra_clusters

        # Build unified path index
        path_index = self._build_unified_path_index(
            path_list, cross_memberships, intra_memberships
        )

        # Calculate statistics
        statistics = self._calculate_statistics(
            cross_clusters, intra_clusters, path_index
        )

        self.logger.info(
            f"Multi-membership clustering complete: "
            f"{len(cross_clusters)} cross-IDS clusters, "
            f"{len(intra_clusters)} intra-IDS clusters"
        )

        return all_clusters, path_index, statistics

    def _cluster_cross_ids(
        self,
        embeddings: np.ndarray,
        path_list: list[str],
        filtered_paths: dict[str, dict[str, Any]],
    ) -> tuple[list[ClusterInfo], dict[str, int]]:
        """Cluster paths that span multiple IDS."""
        # Filter for cross-IDS candidates - only paths that have potential cross-IDS relationships
        cross_candidates = self._get_cross_ids_candidates(path_list, filtered_paths)
        if len(cross_candidates) < self.config.cross_ids_min_samples:
            self.logger.info("Not enough cross-IDS candidates for clustering")
            return [], {}

        cross_indices = [
            i for i, path in enumerate(path_list) if path in cross_candidates
        ]
        cross_embeddings = embeddings[cross_indices]
        cross_paths = [path_list[i] for i in cross_indices]

        self.logger.info(f"Clustering {len(cross_candidates)} cross-IDS candidates")

        # DEBUG: Log the exact parameters being used
        self.logger.info(
            f"DEBUG: Cross-IDS DBSCAN parameters - eps={self.config.cross_ids_eps}, min_samples={self.config.cross_ids_min_samples}, metric=cosine"
        )

        # Cluster cross-IDS paths with stricter parameters
        clustering = DBSCAN(
            eps=self.config.cross_ids_eps,
            min_samples=self.config.cross_ids_min_samples,
            metric="cosine",
        )
        cluster_labels = clustering.fit_predict(cross_embeddings)

        # DEBUG: Log clustering results before validation
        unique_labels = set(cluster_labels)
        noise_count = list(cluster_labels).count(-1)
        cluster_count_before_validation = len(unique_labels) - (
            1 if -1 in unique_labels else 0
        )
        self.logger.info(
            f"DEBUG: Cross-IDS raw clustering results - {cluster_count_before_validation} clusters, {noise_count} noise points"
        )

        # Build clusters - ONLY accept clusters that actually span multiple IDS
        clusters = []
        path_memberships = {}
        cluster_id = 0
        rejected_single_ids = 0

        for label in set(cluster_labels):
            if label == -1:  # Skip noise
                continue

            cluster_indices = [
                i for i, label_val in enumerate(cluster_labels) if label_val == label
            ]
            cluster_paths = [cross_paths[i] for i in cluster_indices]

            # Validate that this cluster actually spans multiple IDS
            ids_in_cluster = {
                path.split("/")[0] for path in cluster_paths
            }  # Use "/" not "." for separator

            if len(ids_in_cluster) < 2:
                rejected_single_ids += 1
                continue

            similarity_score = _compute_cluster_similarity(
                [cross_indices[i] for i in cluster_indices], embeddings
            )

            cluster = ClusterInfo(
                id=cluster_id,
                similarity_score=similarity_score,
                size=len(cluster_paths),
                is_cross_ids=True,
                ids_names=sorted(ids_in_cluster),
                paths=cluster_paths,
            )
            clusters.append(cluster)

            # Record memberships
            for path in cluster_paths:
                path_memberships[path] = cluster_id

            cluster_id += 1

        self.logger.info(
            f"Cross-IDS clustering: {len(clusters)} valid cross-IDS clusters, {rejected_single_ids} rejected single-IDS clusters"
        )
        return clusters, path_memberships

    def _cluster_intra_ids(
        self,
        embeddings: np.ndarray,
        path_list: list[str],
        filtered_paths: dict[str, dict[str, Any]],
    ) -> tuple[list[ClusterInfo], dict[str, int]]:
        """Cluster paths within each IDS separately."""
        clusters = []
        path_memberships = {}
        cluster_id = 1000  # Start from high number to distinguish from cross-IDS

        # Group paths by IDS
        ids_groups = defaultdict(list)
        for i, path in enumerate(path_list):
            ids_name = path.split("/")[0]  # Use "/" not "." for separator
            ids_groups[ids_name].append((i, path))

        # Cluster each IDS separately
        for ids_name, path_data in ids_groups.items():
            if len(path_data) < self.config.intra_ids_min_samples:
                continue

            indices = [idx for idx, _ in path_data]
            ids_paths = [path for _, path in path_data]
            ids_embeddings = embeddings[indices]

            # DEBUG: Log the exact parameters being used for this IDS (only for first few)
            if len(clusters) < 5:  # Only log first few IDS to avoid spam
                self.logger.info(
                    f"DEBUG: Intra-IDS DBSCAN for {ids_name} - eps={self.config.intra_ids_eps}, min_samples={self.config.intra_ids_min_samples}, metric=cosine, paths={len(ids_paths)}"
                )

            # Cluster this IDS
            clustering = DBSCAN(
                eps=self.config.intra_ids_eps,
                min_samples=self.config.intra_ids_min_samples,
                metric="cosine",
            )
            cluster_labels = clustering.fit_predict(ids_embeddings)

            # DEBUG: Log clustering results for this IDS (only for first few)
            if len(clusters) < 5:
                unique_labels = set(cluster_labels)
                noise_count = list(cluster_labels).count(-1)
                cluster_count_for_ids = len(unique_labels) - (
                    1 if -1 in unique_labels else 0
                )
                self.logger.info(
                    f"DEBUG: Intra-IDS {ids_name} clustering results - {cluster_count_for_ids} clusters, {noise_count} noise points"
                )

            # Create cluster objects
            for label in set(cluster_labels):
                if label == -1:  # Skip noise
                    continue

                cluster_indices = [
                    i
                    for i, label_val in enumerate(cluster_labels)
                    if label_val == label
                ]
                cluster_paths = [ids_paths[i] for i in cluster_indices]

                similarity_score = _compute_cluster_similarity(
                    [indices[i] for i in cluster_indices], embeddings
                )

                cluster = ClusterInfo(
                    id=cluster_id,
                    similarity_score=similarity_score,
                    size=len(cluster_paths),
                    is_cross_ids=False,
                    ids_names=[ids_name],
                    paths=cluster_paths,
                )
                clusters.append(cluster)

                # Record memberships
                for path in cluster_paths:
                    path_memberships[path] = cluster_id

                cluster_id += 1

        self.logger.info(f"Intra-IDS clustering: {len(clusters)} clusters")
        return clusters, path_memberships

    def _get_cross_ids_candidates(
        self, path_list: list[str], filtered_paths: dict[str, dict[str, Any]]
    ) -> set[str]:
        """Identify paths that could potentially form cross-IDS relationships."""
        candidates = set()

        # Group paths by semantic concept (more sophisticated approach)
        concept_to_paths = defaultdict(list)
        for path in path_list:
            concept = self._extract_concept(path)
            if len(concept) > 3:  # Skip very short concepts
                concept_to_paths[concept].append(path)

        # Find concepts that appear in multiple IDS
        for _concept, paths in concept_to_paths.items():
            if len(paths) < 2:
                continue

            ids_in_concept = {
                path.split("/")[0] for path in paths
            }  # Use "/" not "." for separator
            if len(ids_in_concept) > 1:  # Appears in multiple IDS
                candidates.update(paths)

        # Additional filtering: look for physics-related terms that are likely to be cross-IDS
        physics_terms = {
            "temperature",
            "density",
            "pressure",
            "current",
            "magnetic",
            "field",
            "flux",
            "psi",
            "phi",
            "radius",
            "rho",
            "profile",
            "boundary",
            "outline",
        }

        for path in path_list:
            path_lower = path.lower()
            if any(term in path_lower for term in physics_terms):
                # Check if this type of path exists in other IDS
                path_concept = self._extract_concept(path)
                other_ids_paths = [
                    p for p in path_list if p.split("/")[0] != path.split("/")[0]
                ]  # Use "/" not "." for separator
                if any(
                    self._extract_concept(other_path) == path_concept
                    for other_path in other_ids_paths
                ):
                    candidates.add(path)

        self.logger.info(
            f"Found {len(candidates)} cross-IDS candidates from {len(path_list)} total paths"
        )
        return candidates

    def _extract_concept(self, path: str) -> str:
        """Extract the core concept from a path."""
        parts = path.split("/")  # Use "/" not "." for separator
        # Remove IDS name and array indices, focus on semantic components
        filtered_parts = []
        for part in parts[1:]:  # Skip IDS name
            if not part.isdigit() and "time_slice" not in part:
                filtered_parts.append(part)
        return ".".join(filtered_parts[-2:])  # Last 2 components for concept

    def _build_unified_path_index(
        self,
        path_list: list[str],
        cross_memberships: dict[str, int],
        intra_memberships: dict[str, int],
    ) -> dict[str, PathMembership]:
        """Build unified path index combining both clustering results."""
        path_index = {}

        for path in path_list:
            cross_cluster = cross_memberships.get(path)
            intra_cluster = intra_memberships.get(path)

            path_index[path] = PathMembership(
                cross_ids_cluster=cross_cluster,
                intra_ids_cluster=intra_cluster,
            )

        return path_index

    def _calculate_statistics(
        self,
        cross_clusters: list[ClusterInfo],
        intra_clusters: list[ClusterInfo],
        path_index: dict[str, PathMembership],
    ) -> dict[str, Any]:
        """Calculate clustering statistics based on final validated clusters."""
        # Count membership types from final results
        multi_membership = 0
        isolated = 0
        paths_in_cross_clusters = 0
        paths_in_intra_clusters = 0

        for membership in path_index.values():
            if (
                membership.cross_ids_cluster is not None
                and membership.intra_ids_cluster is not None
            ):
                multi_membership += 1
            elif (
                membership.cross_ids_cluster is None
                and membership.intra_ids_cluster is None
            ):
                isolated += 1

        # Calculate paths in clusters from validated clusters
        # Re-validate cross-IDS clusters to ensure statistics match reality
        actual_cross_clusters = [c for c in cross_clusters if len(c.ids_names) > 1]
        actual_intra_clusters = [
            c for c in cross_clusters if len(c.ids_names) == 1
        ] + intra_clusters

        paths_in_cross_clusters = sum(c.size for c in actual_cross_clusters)
        paths_in_intra_clusters = sum(c.size for c in actual_intra_clusters)

        # Calculate noise points
        total_paths = len(path_index)
        cross_noise = total_paths - paths_in_cross_clusters
        intra_noise = total_paths - paths_in_intra_clusters

        # Calculate averages from validated clusters
        cross_similarities = [c.similarity_score for c in actual_cross_clusters]
        intra_similarities = [c.similarity_score for c in actual_intra_clusters]

        cross_avg_sim = (
            sum(cross_similarities) / len(cross_similarities)
            if cross_similarities
            else 0.0
        )
        intra_avg_sim = (
            sum(intra_similarities) / len(intra_similarities)
            if intra_similarities
            else 0.0
        )

        return {
            "cross_ids_clustering": {
                "total_clusters": len(actual_cross_clusters),
                "paths_in_clusters": paths_in_cross_clusters,
                "noise_points": cross_noise,
                "avg_similarity": cross_avg_sim,
            },
            "intra_ids_clustering": {
                "total_clusters": len(actual_intra_clusters),
                "paths_in_clusters": paths_in_intra_clusters,
                "noise_points": intra_noise,
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
