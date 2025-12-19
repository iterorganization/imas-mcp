"""
Hierarchical clustering for IMAS data paths.

Runs HDBSCAN at three levels to discover clusters that may be hidden
by density competition in the global embedding space:

1. Global: Full embedding space (all IDS)
2. Domain: Per physics domain (e.g., transport, equilibrium)
3. IDS: Per individual IDS (e.g., core_profiles, magnetics)

Each level may discover unique clusters not visible at other levels.
All clusters are stored in a flat index with scope metadata and UUIDs.
"""

import json
import logging
import uuid
from collections import defaultdict
from typing import Literal

import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from imas_codex.definitions.physics import IDS_DOMAINS_FILE

from .models import ClusterInfo, ClusterScope

logger = logging.getLogger(__name__)


def _load_ids_domain_mappings() -> dict[str, str]:
    """Load IDS to physics domain mappings."""
    if not IDS_DOMAINS_FILE.exists():
        logger.warning(
            "IDS domain mappings not found, domain-level clustering disabled"
        )
        return {}

    with open(IDS_DOMAINS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("ids_domain_mappings", {})


def _compute_cluster_similarity(
    cluster_indices: list[int], embeddings: np.ndarray
) -> float:
    """Compute average intra-cluster cosine similarity."""
    if len(cluster_indices) < 2:
        return 1.0

    cluster_embeddings = embeddings[cluster_indices]
    similarity_matrix = cosine_similarity(cluster_embeddings)
    upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[upper_tri_indices]
    avg_similarity = float(np.mean(similarities))
    return min(1.0, max(0.0, avg_similarity))


def _compute_cluster_centroid(
    cluster_indices: list[int], embeddings: np.ndarray, normalize: bool = True
) -> list[float]:
    """Compute the centroid embedding for a cluster."""
    cluster_embeddings = embeddings[cluster_indices]
    centroid = np.mean(cluster_embeddings, axis=0)

    if normalize:
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

    return centroid.tolist()


class HierarchicalClusterer:
    """Run HDBSCAN at global, domain, and IDS levels.

    Produces a flat list of ClusterInfo objects with scope metadata.
    Each cluster has:
    - scope: "global", "domain", or "ids"
    - scope_detail: None for global, domain/IDS name otherwise
    """

    def __init__(
        self,
        min_cluster_size: int = 2,
        min_samples: int = 2,
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
    ):
        """Initialize the hierarchical clusterer.

        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for core points
            cluster_selection_method: 'eom' for broader, 'leaf' for finer clusters
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.ids_domain_mappings = _load_ids_domain_mappings()

    def cluster_all_levels(
        self,
        embeddings: np.ndarray,
        paths: list[str],
        include_global: bool = True,
        include_domain: bool = True,
        include_ids: bool = True,
    ) -> list[ClusterInfo]:
        """Run HDBSCAN at all requested levels.

        Args:
            embeddings: Embedding matrix (n_paths x embedding_dim)
            paths: List of path strings corresponding to embedding rows
            include_global: Include global-level clustering
            include_domain: Include domain-level clustering
            include_ids: Include IDS-level clustering

        Returns:
            Flat list of ClusterInfo objects with scope metadata and UUIDs
        """
        all_clusters: list[ClusterInfo] = []
        path_to_idx = {p: i for i, p in enumerate(paths)}

        if include_global:
            logger.info("Running global-level clustering...")
            global_clusters = self._cluster_subset(
                embeddings=embeddings,
                paths=paths,
                path_to_idx=path_to_idx,
                scope="global",
                scope_detail=None,
            )
            all_clusters.extend(global_clusters)
            logger.info(f"Global level: {len(global_clusters)} clusters")

        if include_domain and self.ids_domain_mappings:
            logger.info("Running domain-level clustering...")
            domain_count = 0

            # Group paths by domain
            domain_paths: dict[str, list[str]] = defaultdict(list)
            for path in paths:
                ids_name = path.split("/")[0]
                domain = self.ids_domain_mappings.get(ids_name, "general")
                domain_paths[domain].append(path)

            for domain, dpaths in domain_paths.items():
                if len(dpaths) < self.min_cluster_size:
                    continue

                domain_clusters = self._cluster_subset(
                    embeddings=embeddings,
                    paths=dpaths,
                    path_to_idx=path_to_idx,
                    scope="domain",
                    scope_detail=domain,
                )
                all_clusters.extend(domain_clusters)
                domain_count += len(domain_clusters)

            logger.info(
                f"Domain level: {domain_count} clusters across {len(domain_paths)} domains"
            )

        if include_ids:
            logger.info("Running IDS-level clustering...")
            ids_count = 0

            # Group paths by IDS
            ids_paths: dict[str, list[str]] = defaultdict(list)
            for path in paths:
                ids_name = path.split("/")[0]
                ids_paths[ids_name].append(path)

            for ids_name, ipaths in ids_paths.items():
                if len(ipaths) < self.min_cluster_size:
                    continue

                ids_clusters = self._cluster_subset(
                    embeddings=embeddings,
                    paths=ipaths,
                    path_to_idx=path_to_idx,
                    scope="ids",
                    scope_detail=ids_name,
                )
                all_clusters.extend(ids_clusters)
                ids_count += len(ids_clusters)

            logger.info(f"IDS level: {ids_count} clusters across {len(ids_paths)} IDS")

        logger.info(f"Total: {len(all_clusters)} clusters across all levels")
        return all_clusters

    def _cluster_subset(
        self,
        embeddings: np.ndarray,
        paths: list[str],
        path_to_idx: dict[str, int],
        scope: ClusterScope,
        scope_detail: str | None,
    ) -> list[ClusterInfo]:
        """Run HDBSCAN on a subset of paths.

        Args:
            embeddings: Full embedding matrix
            paths: Subset of paths to cluster
            path_to_idx: Mapping from path to index in full embeddings
            scope: Cluster scope level
            scope_detail: Scope detail (domain/IDS name)

        Returns:
            List of clusters with UUIDs
        """
        if len(paths) < self.min_cluster_size:
            return []

        # Extract embeddings for this subset
        indices = [path_to_idx[p] for p in paths]
        subset_embeddings = embeddings[indices]

        # Normalize for cosine-like behavior with euclidean metric
        norms = np.linalg.norm(subset_embeddings, axis=1, keepdims=True)
        normalized_embeddings = subset_embeddings / np.where(norms > 0, norms, 1)

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method=self.cluster_selection_method,
        )
        cluster_labels = clusterer.fit_predict(normalized_embeddings)

        # Build clusters from labels
        clusters: list[ClusterInfo] = []
        unique_labels = set(cluster_labels)

        for label in sorted(unique_labels):
            if label == -1:  # Skip noise
                continue

            # Get paths in this cluster
            cluster_mask = cluster_labels == label
            cluster_paths = [paths[i] for i, is_in in enumerate(cluster_mask) if is_in]
            cluster_indices = [
                indices[i] for i, is_in in enumerate(cluster_mask) if is_in
            ]

            # Compute cluster properties using full embeddings
            similarity_score = _compute_cluster_similarity(cluster_indices, embeddings)
            centroid = _compute_cluster_centroid(cluster_indices, embeddings)

            # Derive IDS membership from paths
            ids_set = {p.split("/")[0] for p in cluster_paths}
            is_cross_ids = len(ids_set) > 1

            # Generate UUID for this cluster
            cluster_uuid = str(uuid.uuid4())

            cluster = ClusterInfo(
                id=cluster_uuid,
                similarity_score=similarity_score,
                size=len(cluster_paths),
                is_cross_ids=is_cross_ids,
                ids_names=sorted(ids_set),
                paths=cluster_paths,
                centroid=centroid,
                scope=scope,
                scope_detail=scope_detail,
            )
            clusters.append(cluster)

        return clusters

    def get_domain_for_ids(self, ids_name: str) -> str:
        """Get the physics domain for an IDS."""
        return self.ids_domain_mappings.get(ids_name, "general")

    def get_ids_in_domain(self, domain: str) -> list[str]:
        """Get all IDS names in a physics domain."""
        return [
            ids_name for ids_name, d in self.ids_domain_mappings.items() if d == domain
        ]
