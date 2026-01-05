"""
Unified clusters management for IMAS semantic cluster data.

This module provides a single, coherent dataclass for managing clusters.json with
intelligent cache management, dependency tracking, and automatic rebuilding.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from imas_codex import dd_version
from imas_codex.clusters import (
    ClusterSearcher,
    RelationshipExtractionConfig,
    RelationshipExtractor,
)
from imas_codex.embeddings.cache import EmbeddingCache
from imas_codex.embeddings.config import EncoderConfig
from imas_codex.embeddings.encoder import Encoder
from imas_codex.resource_path_accessor import ResourcePathAccessor
from imas_codex.search.document_store import DocumentStore

logger = logging.getLogger(__name__)

# Optimal clustering parameters from Latin Hypercube optimization
OPTIMAL_CROSS_IDS_EPS = 0.0751
OPTIMAL_CROSS_IDS_MIN_SAMPLES = 2
OPTIMAL_INTRA_IDS_EPS = 0.0319
OPTIMAL_INTRA_IDS_MIN_SAMPLES = 2


@dataclass
class Clusters:
    """
    Unified clusters manager with intelligent cache management and auto-rebuild.

    Features:
    - Automatic dependency tracking (embeddings)
    - Cache busting when dependencies are newer
    - Lazy loading with error handling
    - Automatic rebuilding with optimal parameters
    - Consistent interface for both building and accessing clusters
    """

    encoder_config: EncoderConfig  # Required: encoder configuration for embeddings
    clusters_file: Path | None = None

    # Cache state
    _cached_data: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _cached_mtime: float | None = field(default=None, init=False, repr=False)
    _cluster_searcher: ClusterSearcher | None = field(
        default=None, init=False, repr=False
    )
    _encoder: Encoder | None = field(default=None, init=False, repr=False)

    # Dependency tracking
    _last_dependency_check: float | None = field(default=None, init=False, repr=False)
    _dependency_check_interval: float = field(default=1.0, init=False, repr=False)

    def __post_init__(self):
        """Initialize computed fields after dataclass initialization."""
        if self.clusters_file is None:
            import hashlib

            path_accessor = ResourcePathAccessor(dd_version=dd_version)

            # Generate filename with hash suffix if IDS set is filtered
            # This prevents test builds from overwriting production builds
            if self.encoder_config.ids_set:
                # Create hash based on sorted IDS names
                ids_str = "_".join(sorted(self.encoder_config.ids_set))
                ids_hash = hashlib.md5(ids_str.encode()).hexdigest()[:8]
                filename = f"clusters_{ids_hash}.json"
                logger.debug(
                    f"Using filtered clusters file: {filename} for IDS set: {sorted(self.encoder_config.ids_set)}"
                )
            else:
                # Full dataset uses simple name (production default)
                filename = "clusters.json"
                logger.debug("Using full dataset clusters file: clusters.json")

            self.clusters_file = path_accessor.clusters_dir / filename

    @property
    def file_path(self) -> Path:
        """Get the clusters file path, guaranteed to be non-None after __post_init__."""
        assert self.clusters_file is not None, (
            "clusters_file should be set in __post_init__"
        )
        return self.clusters_file

    def _get_file_mtime(self, file_path: Path) -> float:
        """Get modification time of a file, returning 0 if file doesn't exist."""
        try:
            return file_path.stat().st_mtime
        except (FileNotFoundError, OSError):
            return 0.0

    def _get_embedding_cache_file(self) -> Path | None:
        """
        Get the specific embedding cache file that clusters will load from.

        Uses the encoder_config to determine the cache file path.
        """
        try:
            encoder = Encoder(self.encoder_config)
            cache_key = self.encoder_config.generate_cache_key()
            encoder._set_cache_path(cache_key)

            cache_path = encoder._cache_path
            return cache_path if cache_path and cache_path.exists() else None

        except Exception as e:
            logger.debug(f"Could not determine embedding cache file: {e}")
            return None

    def _get_cached_embedding_doc_count(self, cache_file: Path) -> int | None:
        """
        Get the document count from a cached embedding file.

        Returns:
            The document count from the cache, or None if it can't be read.
        """
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            if isinstance(cache, EmbeddingCache):
                return cache.document_count
        except Exception as e:
            logger.debug(f"Could not read embedding cache document count: {e}")
        return None

    def _get_clusters_document_count(self) -> int | None:
        """
        Get the document count from clusters.json metadata (fast file read).

        This avoids loading the entire DocumentStore just to count documents.

        Returns:
            The total_paths count from clusters metadata, or None if unavailable.
        """
        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            stats = data.get("statistics", {})
            return stats.get("total_paths")
        except Exception as e:
            logger.debug(f"Could not read clusters document count: {e}")
            return None

    def _get_expected_document_count(self) -> int:
        """
        Get the expected document count from the document store.

        Returns:
            The number of documents the document store would produce.
        """
        try:
            store = DocumentStore(ids_set=self.encoder_config.ids_set)
            store.load_all_documents()
            return len(store)
        except Exception as e:
            logger.debug(f"Could not get expected document count: {e}")
            return 0

    def _check_dependency_freshness(self) -> bool:
        """
        Check if any dependency files are newer than the clusters file.

        Uses fast file-based checks only. Does NOT load DocumentStore to avoid
        expensive initialization (~30s for embeddings). Compares:
        1. File modification times (clusters vs embeddings)
        2. Document counts from cached metadata (embeddings cache vs clusters.json)

        Returns:
            True if clusters file should be regenerated, False otherwise.
        """
        if not self.file_path.exists():
            logger.debug("Clusters file does not exist")
            return True

        clusters_mtime = self._get_file_mtime(self.file_path)
        clusters_time = datetime.fromtimestamp(clusters_mtime)

        # Check the specific embedding cache file that will be loaded
        embedding_cache = self._get_embedding_cache_file()
        if embedding_cache:
            # Check mtime first
            embedding_mtime = self._get_file_mtime(embedding_cache)
            if embedding_mtime > clusters_mtime:
                embedding_time = datetime.fromtimestamp(embedding_mtime)
                time_diff = embedding_mtime - clusters_mtime
                logger.info(
                    f"Embedding cache newer than clusters: {embedding_cache.name} "
                    f"(embedding: {embedding_time.isoformat()}, clusters: {clusters_time.isoformat()}, diff: {time_diff:.1f}s)"
                )
                return True

            # Fast document count check: compare embedding cache count with clusters metadata
            # This avoids loading the entire DocumentStore just to count documents
            cached_count = self._get_cached_embedding_doc_count(embedding_cache)
            if cached_count is not None:
                # Get expected count from clusters.json metadata (fast file read)
                clusters_doc_count = self._get_clusters_document_count()
                if (
                    clusters_doc_count is not None
                    and cached_count != clusters_doc_count
                ):
                    logger.info(
                        f"Embedding cache document count mismatch with clusters: "
                        f"embedding={cached_count}, clusters={clusters_doc_count}. Clusters rebuild required."
                    )
                    return True
                logger.debug(
                    f"Embedding cache document count valid: {cached_count} documents"
                )

            logger.debug(
                f"Embedding cache is up-to-date: {embedding_cache.name} "
                f"(embedding: {datetime.fromtimestamp(embedding_mtime).isoformat()}, clusters: {clusters_time.isoformat()})"
            )
        else:
            # If no embedding cache exists, we need to rebuild
            logger.info("No embedding cache file found - clusters rebuild required")
            return True

        return False

    def _should_check_dependencies(self) -> bool:
        """Determine if we should check dependencies based on interval."""
        if self._last_dependency_check is None:
            return True

        current_time = datetime.now().timestamp()
        return (
            current_time - self._last_dependency_check
        ) > self._dependency_check_interval

    def _invalidate_cache(self) -> None:
        """Invalidate cached data and engine."""
        logger.debug("Invalidating clusters cache")
        self._cached_data = None
        self._cached_mtime = None
        self._cluster_engine = None
        self._cluster_searcher = None

    def _load_clusters_data(self) -> dict[str, Any]:
        """
        Load clusters data from file with dependency checking and auto-rebuild.

        Returns:
            Dictionary containing clusters data.

        Raises:
            FileNotFoundError: If clusters file doesn't exist and can't be built.
            json.JSONDecodeError: If file contains invalid JSON.
        """
        # Check if we should verify dependencies
        if self._should_check_dependencies():
            self._last_dependency_check = datetime.now().timestamp()

            if self._check_dependency_freshness():
                logger.info("Clusters file is outdated. Auto-rebuilding...")
                try:
                    self.build(force=True)
                except Exception as e:
                    logger.error(f"Auto-rebuild failed: {e}")
                    # Continue to try loading existing file if available

        # Check if cached data is still valid
        current_mtime = self._get_file_mtime(self.file_path)
        if (
            self._cached_data is not None
            and self._cached_mtime is not None
            and current_mtime == self._cached_mtime
        ):
            return self._cached_data

        # Load fresh data
        logger.debug(f"Loading clusters data from {self.file_path}")

        if not self.file_path.exists():
            logger.info("Clusters file not found. Building...")
            self.build(force=True)

        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Cache the loaded data
            self._cached_data = data
            self._cached_mtime = current_mtime
            self._cluster_engine = None  # Reset engine on data reload

            logger.debug(
                f"Loaded {len(data.get('clusters', []))} clusters from clusters file"
            )
            return data

        except UnicodeDecodeError:
            # Fallback to latin-1 encoding
            logger.warning("UTF-8 decode failed, trying latin-1 encoding")
            with self.file_path.open("r", encoding="latin-1") as f:
                data = json.load(f)

            self._cached_data = data
            self._cached_mtime = current_mtime
            self._cluster_engine = None

            return data

    def build(self, force: bool = False, **config_overrides) -> bool:
        """
        Build clusters file using optimal parameters.

        Args:
            force: Force rebuild even if dependencies aren't newer
            **config_overrides: Override default optimal parameters

        Returns:
            True if rebuild was performed, False if skipped
        """
        # Check if rebuild is actually needed
        if not force and not self.needs_rebuild():
            logger.debug("Clusters rebuild not needed")
            return False

        logger.info("Building clusters with optimal parameters...")

        try:
            # Get version-specific paths using ResourcePathAccessor
            path_accessor = ResourcePathAccessor(dd_version=dd_version)
            input_dir = path_accessor.version_dir / "schemas" / "detailed"
            # Use the file path determined in __post_init__ (includes hash suffix if needed)
            output_file = self.file_path

            # Create configuration with optimal parameters
            default_config = {
                "encoder_config": self.encoder_config,
                "cross_ids_eps": OPTIMAL_CROSS_IDS_EPS,
                "cross_ids_min_samples": OPTIMAL_CROSS_IDS_MIN_SAMPLES,
                "intra_ids_eps": OPTIMAL_INTRA_IDS_EPS,
                "intra_ids_min_samples": OPTIMAL_INTRA_IDS_MIN_SAMPLES,
                "use_rich": True,
                "ids_set": self.encoder_config.ids_set,
                "input_dir": input_dir,
                "output_file": output_file,
            }
            default_config.update(config_overrides)

            config = RelationshipExtractionConfig(**default_config)

            # Build clusters using the extractor
            extractor = RelationshipExtractor(config)
            clusters = extractor.extract_relationships(force_rebuild=True)

            # Save clusters
            extractor.save_relationships(clusters)

            # Invalidate cache to pick up new data
            self._invalidate_cache()

            logger.info("Clusters build completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to build clusters: {e}")
            raise

    def get_data(self) -> dict[str, Any]:
        """
        Get clusters data with caching and auto-rebuild.

        Returns:
            Dictionary containing clusters data.
        """
        return self._load_clusters_data()

    def get_clusters(self) -> list[dict[str, Any]]:
        """Get relationship clusters."""
        data = self.get_data()
        return data.get("clusters", [])

    def get_metadata(self) -> dict[str, Any]:
        """Get clusters metadata."""
        data = self.get_data()
        return data.get("metadata", {})

    def get_unit_families(self) -> dict[str, Any]:
        """Get unit families information."""
        data = self.get_data()
        return data.get("unit_families", {})

    def get_cross_references(self) -> dict[str, Any]:
        """Get cross-references information."""
        data = self.get_data()
        return data.get("cross_references", {})

    def is_available(self) -> bool:
        """Check if clusters data is available."""
        try:
            self.get_data()
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def needs_rebuild(self) -> bool:
        """
        Check if clusters file needs rebuilding based on dependencies.

        Returns:
            True if rebuild is recommended, False otherwise.
        """
        return self._check_dependency_freshness()

    def get_cache_info(self) -> dict[str, Any]:
        """
        Get information about the current cache state.

        Returns:
            Dictionary with cache statistics and status.
        """
        info = {
            "file_path": str(self.file_path),
            "file_exists": self.file_path.exists(),
            "cached": self._cached_data is not None,
            "engine_cached": self._cluster_engine is not None,
        }

        if self.file_path.exists():
            info["file_size_mb"] = self.file_path.stat().st_size / (1024 * 1024)
            info["file_mtime"] = datetime.fromtimestamp(
                self._get_file_mtime(self.file_path)
            ).isoformat()

        if self._cached_data:
            info["clusters_count"] = len(self._cached_data.get("clusters", []))
            info["has_metadata"] = "metadata" in self._cached_data
            info["has_unit_families"] = "unit_families" in self._cached_data

        info["needs_rebuild"] = self.needs_rebuild()

        return info

    def force_reload(self) -> None:
        """Force reload of clusters data, bypassing cache."""
        logger.info("Forcing reload of clusters data")
        self._invalidate_cache()
        # Next call to get_data() will reload from disk

    def get_cluster_searcher(self) -> ClusterSearcher:
        """Get cluster searcher for semantic search over clusters.

        Returns:
            ClusterSearcher instance with centroids loaded.
        """
        if self._cluster_searcher is None:
            clusters = self.get_clusters()
            data = self.get_data()

            # Get embeddings file path and hash from clusters.json metadata
            embeddings_filename = data.get("embeddings_file", "cluster_embeddings.npz")
            embeddings_file = self.file_path.parent / embeddings_filename
            embeddings_hash = data.get("embeddings_hash")

            self._cluster_searcher = ClusterSearcher(
                clusters=clusters,
                embeddings_file=embeddings_file,
                expected_embeddings_hash=embeddings_hash,
            )
            logger.debug(f"Created cluster searcher with {len(clusters)} clusters")

        return self._cluster_searcher

    def get_encoder(self) -> Encoder:
        """Get encoder for embedding queries.

        Returns:
            Encoder instance.
        """
        if self._encoder is None:
            self._encoder = Encoder(self.encoder_config)
            logger.debug("Created encoder for cluster search")

        return self._encoder

    def search_clusters(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        cross_ids_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Search for clusters matching a text query.

        Uses centroid embeddings to find semantically similar clusters.

        Args:
            query: Text query to search for
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            cross_ids_only: If True, only return cross-IDS clusters

        Returns:
            List of matching clusters with similarity scores
        """
        searcher = self.get_cluster_searcher()
        encoder = self.get_encoder()

        results = searcher.search_by_text(
            query=query,
            encoder=encoder,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            cross_ids_only=cross_ids_only,
        )

        return [
            {
                "cluster_id": r.cluster_id,
                "similarity_score": r.similarity_score,
                "is_cross_ids": r.is_cross_ids,
                "ids_names": r.ids_names,
                "paths": r.paths,
                "cluster_similarity": r.cluster_similarity,
            }
            for r in results
        ]
