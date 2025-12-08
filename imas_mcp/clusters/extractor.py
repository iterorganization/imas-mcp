"""
Main relationship extractor class.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from imas_mcp import dd_version
from imas_mcp.embeddings import EmbeddingCache
from imas_mcp.embeddings.encoder import Encoder
from imas_mcp.resource_path_accessor import ResourcePathAccessor
from imas_mcp.search.document_store import DocumentStore

from .clustering import EmbeddingClusterer, RelationshipBuilder
from .config import RelationshipExtractionConfig
from .models import (
    ClusteringParameters,
    ClusteringStatistics,
    CrossIDSSummary,
    IntraIDSSummary,
    RelationshipMetadata,
    RelationshipSet,
)
from .preprocessing import PathFilter, UnitFamilyBuilder


class RelationshipExtractor:
    """
    Main class for extracting relationships between IMAS data paths.

    Uses semantic embeddings and multi-membership clustering to identify
    related paths both within and across different IDS structures.
    """

    def __init__(self, config: RelationshipExtractionConfig | None = None):
        """Initialize the relationship extractor.

        The rich output configuration is now taken directly from the provided
        RelationshipExtractionConfig (config.use_rich). A separate use_rich
        argument is no longer supported.
        """
        self.config = config or RelationshipExtractionConfig()
        self._use_rich = getattr(self.config, "use_rich", True)
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.path_filter = PathFilter(self.config)
        self.unit_builder = UnitFamilyBuilder()
        self.clusterer = EmbeddingClusterer(self.config, self.logger)
        self.relationship_builder = RelationshipBuilder(self.config, self.logger)

        # Use shared embedding infrastructure
        self._embedding_manager: Any | None = None

    def extract_relationships(
        self, input_dir: Path | None = None, force_rebuild: bool = False
    ) -> RelationshipSet:
        """
        Extract cluster-based relationships from IDS data.

        Args:
            input_dir: Directory containing detailed IDS JSON files
            force_rebuild: Force rebuilding even if cache exists

        Returns:
            RelationshipSet containing all extracted cluster relationships
        """
        input_dir = input_dir or self.config.input_dir

        self.logger.info("Starting relationship extraction process...")

        # Load IDS data
        self.logger.info("Loading IDS data from %s", input_dir)
        ids_data = self._load_ids_data(input_dir)

        # Filter meaningful paths
        self.logger.info("Filtering meaningful paths...")
        filtered_paths = self.path_filter.filter_meaningful_paths(ids_data)

        # Generate embeddings
        self.logger.info("Generating embeddings for %d paths...", len(filtered_paths))
        embeddings, path_list = self._generate_embeddings(filtered_paths)

        # Cluster embeddings using multi-membership approach
        self.logger.info("Clustering embeddings...")
        all_clusters, path_index, statistics = self.clusterer.cluster_embeddings(
            embeddings, path_list, filtered_paths
        )

        # Build additional groupings for tool compatibility
        self.logger.info("Building unit families...")
        unit_families = self.unit_builder.build_unit_families(filtered_paths)

        # Build summaries
        cross_ids_summary = self._build_cross_ids_summary(all_clusters)
        intra_ids_summary = self._build_intra_ids_summary(all_clusters)

        # Create metadata
        generation_timestamp = datetime.now().isoformat()

        clustering_stats = ClusteringStatistics(
            cross_ids_clustering=statistics["cross_ids_clustering"],
            intra_ids_clustering=statistics["intra_ids_clustering"],
            multi_membership_paths=statistics["multi_membership_paths"],
            isolated_paths=statistics["isolated_paths"],
        )

        metadata = RelationshipMetadata(
            generation_timestamp=generation_timestamp,
            total_paths_processed=len(filtered_paths),
            clustering_parameters={
                "cross_ids": ClusteringParameters(
                    eps=self.config.cross_ids_eps,
                    min_samples=self.config.cross_ids_min_samples,
                    metric="cosine",
                ),
                "intra_ids": ClusteringParameters(
                    eps=self.config.intra_ids_eps,
                    min_samples=self.config.intra_ids_min_samples,
                    metric="cosine",
                ),
            },
            statistics=clustering_stats,
        )

        # Build final relationship set
        relationships = RelationshipSet(
            metadata=metadata,
            clusters=all_clusters,
            path_index=path_index,
            cross_ids_summary=cross_ids_summary,
            intra_ids_summary=intra_ids_summary,
        )

        # Store additional groupings for saving
        relationships._unit_families = unit_families

        self.logger.info("Relationship extraction completed successfully")
        self.logger.info(
            f"Generated {len(all_clusters)} total clusters: "
            f"{len([c for c in all_clusters if c.is_cross_ids])} cross-IDS, "
            f"{len([c for c in all_clusters if not c.is_cross_ids])} intra-IDS"
        )
        return relationships

    def _build_cross_ids_summary(self, all_clusters: list) -> CrossIDSSummary:
        """Build summary for cross-IDS clusters."""
        cross_clusters = [c for c in all_clusters if c.is_cross_ids]
        if not cross_clusters:
            return CrossIDSSummary(
                cluster_count=0,
                cluster_index=[],
                avg_similarity=0.0,
                total_paths=0,
            )

        cluster_indices = [c.id for c in cross_clusters]
        avg_similarity = sum(c.similarity_score for c in cross_clusters) / len(
            cross_clusters
        )
        total_paths = sum(c.size for c in cross_clusters)

        return CrossIDSSummary(
            cluster_count=len(cross_clusters),
            cluster_index=cluster_indices,
            avg_similarity=avg_similarity,
            total_paths=total_paths,
        )

    def _build_intra_ids_summary(self, all_clusters: list) -> IntraIDSSummary:
        """Build summary for intra-IDS clusters."""
        intra_clusters = [c for c in all_clusters if not c.is_cross_ids]
        if not intra_clusters:
            return IntraIDSSummary(
                cluster_count=0,
                cluster_index=[],
                by_ids={},
                avg_similarity=0.0,
                total_paths=0,
            )

        cluster_indices = [c.id for c in intra_clusters]
        avg_similarity = sum(c.similarity_score for c in intra_clusters) / len(
            intra_clusters
        )
        total_paths = sum(c.size for c in intra_clusters)

        # Group by IDS
        by_ids = {}
        for cluster in intra_clusters:
            ids_name = cluster.ids_names[0]  # Intra-IDS clusters have exactly one IDS
            if ids_name not in by_ids:
                by_ids[ids_name] = {
                    "cluster_index": [],
                    "path_count": 0,
                }
            by_ids[ids_name]["cluster_index"].append(cluster.id)
            by_ids[ids_name]["path_count"] += cluster.size

        return IntraIDSSummary(
            cluster_count=len(intra_clusters),
            cluster_index=cluster_indices,
            by_ids=by_ids,
            avg_similarity=avg_similarity,
            total_paths=total_paths,
        )

    def save_relationships(
        self, relationships: RelationshipSet, output_file: Path | None = None
    ) -> None:
        """Save relationships to JSON file with additional groupings."""
        output_file = output_file or self.config.output_file

        # Determine clusters.json path (same directory, new filename)
        clusters_file = output_file.parent / "clusters.json"

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization using Pydantic
        data = relationships.model_dump()

        # Add additional groupings for tool compatibility if they exist
        if hasattr(relationships, "_unit_families"):
            data["unit_families"] = relationships._unit_families

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info("Saved relationships to %s", output_file)

        # Also save in new clusters.json format with labels
        self._save_clusters_json(relationships, clusters_file)

    def _save_clusters_json(
        self,
        relationships: RelationshipSet,
        output_file: Path,
    ) -> None:
        """Save clusters in optimized format with LLM-generated labels.

        Args:
            relationships: The extracted relationships
            output_file: Path to clusters.json
        """
        from imas_mcp import dd_version

        clusters_list = relationships.clusters

        # Generate labels using LLM
        labels_map = self._generate_cluster_labels(clusters_list)

        # Generate label embeddings for semantic search
        label_embeddings = self._generate_label_embeddings(labels_map)

        # Build clusters array with labels
        clusters_data = []
        for cluster in clusters_list:
            cluster_id = cluster.id
            label_info = labels_map.get(cluster_id, {})

            cluster_entry = {
                "id": cluster_id,
                "label": label_info.get("label", f"Cluster {cluster_id}"),
                "description": label_info.get("description", ""),
                "type": "cross_ids" if cluster.is_cross_ids else "intra_ids",
                "similarity": round(cluster.similarity_score, 4),
                "ids": cluster.ids_names,
                "paths": cluster.paths,
                "centroid": cluster.centroid,
            }

            # Add label embedding if available
            if cluster_id in label_embeddings:
                cluster_entry["label_embedding"] = label_embeddings[cluster_id]

            clusters_data.append(cluster_entry)

        # Build indexes
        path_to_cluster: dict[str, list[int]] = {}
        ids_to_clusters: dict[str, list[int]] = {}
        cross_ids_list: list[int] = []
        intra_ids_list: list[int] = []

        for cluster in clusters_list:
            cluster_id = cluster.id

            # Path index (supports multi-membership)
            for path in cluster.paths:
                if path not in path_to_cluster:
                    path_to_cluster[path] = []
                path_to_cluster[path].append(cluster_id)

            # IDS index
            for ids_name in cluster.ids_names:
                if ids_name not in ids_to_clusters:
                    ids_to_clusters[ids_name] = []
                if cluster_id not in ids_to_clusters[ids_name]:
                    ids_to_clusters[ids_name].append(cluster_id)

            # Type lists
            if cluster.is_cross_ids:
                cross_ids_list.append(cluster_id)
            else:
                intra_ids_list.append(cluster_id)

        # Build final structure
        output_data = {
            "version": "1.0",
            "dd_version": dd_version,
            "generated": datetime.now().isoformat(),
            "labeling_model": self._get_labeling_model(),
            "clusters": clusters_data,
            "indexes": {
                "path": path_to_cluster,
                "ids": ids_to_clusters,
                "cross_ids": cross_ids_list,
                "intra_ids": intra_ids_list,
            },
            "statistics": {
                "total_clusters": len(clusters_list),
                "cross_ids_count": len(cross_ids_list),
                "intra_ids_count": len(intra_ids_list),
                "total_paths": len(path_to_cluster),
                "clustering_params": {
                    "cross_ids": {
                        "eps": self.config.cross_ids_eps,
                        "min_samples": self.config.cross_ids_min_samples,
                    },
                    "intra_ids": {
                        "eps": self.config.intra_ids_eps,
                        "min_samples": self.config.intra_ids_min_samples,
                    },
                },
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved clusters.json to {output_file}")

    def _get_labeling_model(self) -> str:
        """Get the labeling model name."""
        from imas_mcp.settings import get_language_model

        return get_language_model()

    def _generate_cluster_labels(self, clusters: list) -> dict[int, dict[str, str]]:
        """Generate labels for clusters using LLM.

        Args:
            clusters: List of ClusterInfo objects

        Returns:
            Dict mapping cluster_id to {"label": str, "description": str}
        """
        import os

        # Check if API key is available for LLM labeling
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            self.logger.warning(
                "No API key available for LLM labeling. Using fallback labels."
            )
            return self._generate_fallback_labels(clusters)

        try:
            from .labeler import ClusterLabeler

            labeler = ClusterLabeler()

            # Convert ClusterInfo to dicts for labeler
            cluster_dicts = []
            for cluster in clusters:
                cluster_dicts.append(
                    {
                        "id": cluster.id,
                        "is_cross_ids": cluster.is_cross_ids,
                        "ids_names": cluster.ids_names,
                        "paths": cluster.paths,
                    }
                )

            labels = labeler.label_clusters(cluster_dicts)

            return {
                label.cluster_id: {
                    "label": label.label,
                    "description": label.description,
                }
                for label in labels
            }

        except Exception as e:
            self.logger.error(f"LLM labeling failed: {e}. Using fallback labels.")
            return self._generate_fallback_labels(clusters)

    def _generate_fallback_labels(self, clusters: list) -> dict[int, dict[str, str]]:
        """Generate fallback labels without LLM."""
        labels = {}
        for cluster in clusters:
            # Extract common terms from paths
            paths = cluster.paths[:5]
            if paths:
                segments = [p.split("/")[-1] for p in paths]
                common = " ".join(segments[:2]).replace("_", " ").title()
                label = common[:50]  # Truncate
            else:
                label = f"Cluster {cluster.id}"

            type_str = "cross-IDS" if cluster.is_cross_ids else "intra-IDS"
            ids_str = ", ".join(cluster.ids_names[:3])
            description = f"A {type_str} cluster of related paths from {ids_str}."

            labels[cluster.id] = {"label": label, "description": description}

        return labels

    def _generate_label_embeddings(
        self, labels_map: dict[int, dict[str, str]]
    ) -> dict[int, list[float]]:
        """Generate embeddings for cluster labels+descriptions.

        Args:
            labels_map: Dict mapping cluster_id to {"label": str, "description": str}

        Returns:
            Dict mapping cluster_id to embedding vector
        """
        if not labels_map:
            return {}

        try:
            encoder_config = self.config.get_encoder_config()
            encoder = Encoder(encoder_config)

            # Combine label and description for embedding
            texts = []
            cluster_ids = []
            for cluster_id, info in labels_map.items():
                text = f"{info['label']}: {info['description']}"
                texts.append(text)
                cluster_ids.append(cluster_id)

            # Generate embeddings
            embeddings = encoder.embed_texts(texts)

            return {
                cluster_id: embedding.tolist()
                for cluster_id, embedding in zip(cluster_ids, embeddings, strict=True)
            }

        except Exception as e:
            self.logger.warning(f"Failed to generate label embeddings: {e}")
            return {}

    def _load_ids_data(self, input_dir: Path) -> dict[str, Any]:
        """Load detailed IDS JSON files, optionally filtered by ids_set."""
        ids_data = {}
        json_files = list(input_dir.glob("*.json"))

        # Filter files based on ids_set if provided
        if self.config.ids_set:
            filtered_files = []
            for json_file in json_files:
                if json_file.stem in self.config.ids_set:
                    filtered_files.append(json_file)
            json_files = filtered_files
            self.logger.info(
                f"Filtered to {len(json_files)} IDS files based on ids_set: {sorted(self.config.ids_set)}"
            )
        else:
            self.logger.info(f"Found {len(json_files)} IDS files")

        for json_file in json_files:
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                ids_data[json_file.stem] = data
            except Exception as e:
                self.logger.warning(f"Failed to load {json_file.name}: {e}")

        return ids_data

    def _generate_embeddings(
        self, filtered_paths: dict[str, dict[str, Any]]
    ) -> tuple[np.ndarray, list[str]]:
        """Generate embeddings for filtered paths using shared encoder cache.

        Reuses the same cache filename logic as the build_embeddings script so we
        don't regenerate embeddings unnecessarily.
        """

        # Create DocumentStore with same configuration as build_embeddings.py
        if self.config.ids_set:
            self.logger.info(
                f"Creating document store with IDS filter: {sorted(self.config.ids_set)}"
            )
            document_store = DocumentStore(ids_set=self.config.ids_set)
        else:
            self.logger.info("Creating document store with all available IDS")
            document_store = DocumentStore()

        # Get all documents (same as build_embeddings.py does)
        all_documents = document_store.get_all_documents()
        all_texts = [doc.embedding_text for doc in all_documents]
        all_identifiers = [doc.metadata.path_id for doc in all_documents]

        self.logger.info(
            f"Found {len(all_documents)} total documents in document store"
        )

        # Use encoder config from relationship config (single source of truth)
        encoder_config = self.config.get_encoder_config()
        encoder = Encoder(encoder_config)

        # Generate cache key using same method as build_embeddings.py and SemanticSearch
        cache_key = encoder_config.generate_cache_key()

        # Get source data directory for validation (same as build_embeddings.py)
        source_data_dir = None
        try:
            path_accessor = ResourcePathAccessor(dd_version=dd_version)
            source_data_dir = path_accessor.schemas_dir
        except Exception:
            pass

        # Get embeddings for ALL documents (same approach as build_embeddings.py)
        # This will reuse the cache if it exists
        try:
            all_embeddings, all_result_identifiers, was_cached = (
                encoder.build_document_embeddings(
                    texts=all_texts,
                    identifiers=all_identifiers,
                    cache_key=cache_key,
                    force_rebuild=False,
                    source_data_dir=source_data_dir,
                )
            )

            cache_status = "loaded from cache" if was_cached else "generated fresh"
            self.logger.info(f"Embeddings {cache_status}: {all_embeddings.shape}")

            # Now filter to only the paths we need for relationships
            # Create mapping from identifier to index
            id_to_idx = {
                identifier: idx for idx, identifier in enumerate(all_result_identifiers)
            }

            # Extract embeddings for our filtered paths
            filtered_embeddings = []
            filtered_identifiers = []

            for path in filtered_paths.keys():
                if path in id_to_idx:
                    idx = id_to_idx[path]
                    filtered_embeddings.append(all_embeddings[idx])
                    filtered_identifiers.append(path)
                else:
                    self.logger.warning(
                        f"Path {path} not found in embeddings, skipping"
                    )

            if not filtered_embeddings:
                raise ValueError("No embeddings found for filtered paths")

            embeddings = np.vstack(filtered_embeddings)

            self.logger.info(
                f"Extracted {len(filtered_embeddings)} embeddings for relationship analysis"
            )

            # Store cache for compatibility
            self._embeddings_cache = EmbeddingCache(
                embeddings=embeddings,
                path_ids=filtered_identifiers,
                model_name=self.config.encoder_config.model_name,
                document_count=len(filtered_paths),
            )

            return embeddings, filtered_identifiers

        except Exception as e:
            self.logger.error(f"Failed to get embeddings using shared approach: {e}")
            raise
