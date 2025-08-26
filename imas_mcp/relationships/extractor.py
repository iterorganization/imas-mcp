"""
Main relationship extractor class.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from imas_mcp.embeddings import EmbeddingCache, EmbeddingConfig, EmbeddingManager
from imas_mcp.embeddings.manager import get_embedding_manager

from .clustering import EmbeddingClusterer, RelationshipBuilder
from .config import RelationshipExtractionConfig
from .models import (
    CrossReference,
    RelationshipMetadata,
    RelationshipResult,
    RelationshipSet,
    UnitFamily,
)
from .preprocessing import PathFilter, UnitFamilyBuilder


class RelationshipExtractor:
    """
    Main class for extracting relationships between IMAS data paths.

    Uses semantic embeddings and clustering to identify related paths
    across different IDS structures.
    """

    def __init__(self, config: RelationshipExtractionConfig | None = None):
        """Initialize the relationship extractor."""
        self.config = config or RelationshipExtractionConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.path_filter = PathFilter(self.config)
        self.unit_builder = UnitFamilyBuilder()
        self.clusterer = EmbeddingClusterer(self.config, self.logger)
        self.relationship_builder = RelationshipBuilder(self.config, self.logger)

        # Use shared embedding infrastructure
        self._embedding_manager: EmbeddingManager | None = None

    def extract_relationships(
        self, input_dir: Path | None = None, force_rebuild: bool = False
    ) -> RelationshipSet:
        """
        Extract relationships from IDS data.

        Args:
            input_dir: Directory containing detailed IDS JSON files
            force_rebuild: Force rebuilding even if cache exists

        Returns:
            RelationshipSet containing all extracted relationships
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

        # Cluster embeddings
        self.logger.info("Clustering embeddings...")
        cluster_infos, noise_count = self.clusterer.cluster_embeddings(embeddings)

        # Update cluster infos with actual path names
        self._update_cluster_paths(cluster_infos, path_list, embeddings)

        # Extract relationships between clusters
        self.logger.info("Extracting relationships between clusters...")
        similarities = self.clusterer.compute_cluster_similarities(cluster_infos)
        cross_references, total_relationships = (
            self.relationship_builder.extract_cluster_relationships(
                cluster_infos, path_list, filtered_paths, similarities
            )
        )

        # Build unit families
        self.logger.info("Building unit families...")
        unit_families = self.unit_builder.build_unit_families(filtered_paths)

        # Create metadata
        metadata = RelationshipMetadata(
            total_clusters=len(cluster_infos),
            total_relationships=total_relationships,
            total_paths_processed=len(filtered_paths),
            noise_points=noise_count,
            similarity_threshold=self.config.similarity_threshold,
        )

        # Build final relationship set
        # Convert cross_references to proper format
        cross_refs = {}
        for path, rels in cross_references.items():
            relationships = [
                RelationshipResult(
                    path=rel["path"],
                    type=rel["type"],
                    similarity_score=rel["similarity_score"],
                    cluster_id=rel.get("cluster_id"),
                )
                for rel in rels["relationships"]
            ]
            cross_refs[path] = CrossReference(
                type=rels["type"], relationships=relationships
            )

        # Convert unit_families to proper format
        unit_fams = {}
        for unit, data in unit_families.items():
            unit_fams[unit] = UnitFamily(
                base_unit=data["base_unit"],
                paths_using=data["paths_using"],
                conversion_factors=data["conversion_factors"],
            )

        relationships = RelationshipSet(
            metadata=metadata,
            cross_references=cross_refs,
            physics_concepts={},  # Can be expanded later
            unit_families=unit_fams,
        )

        self.logger.info("Relationship extraction completed successfully")
        return relationships

    def save_relationships(
        self, relationships: RelationshipSet, output_file: Path | None = None
    ) -> None:
        """Save relationships to JSON file."""
        output_file = output_file or self.config.output_file

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        data = {
            "metadata": {
                "total_clusters": relationships.metadata.total_clusters,
                "total_relationships": relationships.metadata.total_relationships,
                "clustering_method": relationships.metadata.clustering_method,
                "similarity_threshold": relationships.metadata.similarity_threshold,
                "total_paths_processed": relationships.metadata.total_paths_processed,
                "noise_points": relationships.metadata.noise_points,
            },
            "cross_references": relationships.cross_references,
            "physics_concepts": relationships.physics_concepts,
            "unit_families": relationships.unit_families,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info("Saved relationships to %s", output_file)

    def _load_ids_data(self, input_dir: Path) -> dict[str, Any]:
        """Load all detailed IDS JSON files."""
        ids_data = {}
        json_files = list(input_dir.glob("*.json"))

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
        """Generate embeddings for filtered paths using EmbeddingManager."""
        # Initialize embedding manager if needed
        if self._embedding_manager is None:
            embedding_config = EmbeddingConfig(
                model_name=self.config.model_name,
                device=None,  # Match semantic search default device
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                enable_cache=True,
                cache_dir="embeddings",
                use_rich=True,
            )

            # Use same manager ID strategy as semantic search for sharing
            manager_id = f"{embedding_config.model_name}_{embedding_config.device}"
            self._embedding_manager = get_embedding_manager(
                config=embedding_config, manager_id=manager_id
            )

        # Extract texts and identifiers
        path_list = list(filtered_paths.keys())
        descriptions = [filtered_paths[path]["description"] for path in path_list]

        # Use embedding manager to get embeddings
        embeddings, identifiers = self._embedding_manager.get_embeddings(
            texts=descriptions,
            identifiers=path_list,
            cache_key="relationships",
            force_rebuild=False,
        )

        # Store cache for compatibility
        self._embeddings_cache = EmbeddingCache(
            embeddings=embeddings,
            path_ids=identifiers,
            model_name=self.config.model_name,
            document_count=len(filtered_paths),
        )

        self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings, identifiers

    def _update_cluster_paths(
        self,
        cluster_infos: dict[int, Any],
        path_list: list[str],
        embeddings: np.ndarray,
    ) -> None:
        """Update cluster infos with actual path names."""
        # Re-run clustering to get the mapping
        dbscan = DBSCAN(
            eps=self.config.eps, min_samples=self.config.min_samples, metric="cosine"
        )
        cluster_labels = dbscan.fit_predict(embeddings)

        # Map indices to cluster IDs
        for i, label in enumerate(cluster_labels):
            if label != -1 and label in cluster_infos:
                cluster_infos[label].paths.append(path_list[i])
