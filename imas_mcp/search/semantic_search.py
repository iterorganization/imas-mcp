"""
Semantic search using sentence transformers with IMAS DocumentStore.

This module provides high-performance semantic search capabilities optimized for
LLM usage, using state-of-the-art sentence transformer models with efficient
vector storage and retrieval.
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from imas_mcp.embeddings import EmbeddingCache, EmbeddingConfig, EmbeddingManager
from imas_mcp.search.document_store import Document, DocumentStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticSearchResult:
    """Result from semantic search with similarity score."""

    document: Document
    similarity_score: float
    rank: int

    @property
    def path_id(self) -> str:
        """Get the document path ID."""
        return self.document.metadata.path_id

    @property
    def ids_name(self) -> str:
        """Get the IDS name."""
        return self.document.metadata.ids_name


@dataclass
class SemanticSearchConfig:
    """Configuration for semantic search."""

    # Model configuration
    model_name: str = "all-MiniLM-L6-v2"  # Fast, good quality model
    device: str | None = None  # Auto-detect GPU/CPU

    # Search configuration
    default_top_k: int = 10
    similarity_threshold: float = 0.0  # Minimum similarity to return
    batch_size: int = 250  # For embedding generation
    ids_set: set | None = None  # Limit to specific IDS for testing/performance

    # Cache configuration
    enable_cache: bool = True

    # Performance optimization
    normalize_embeddings: bool = True  # Faster cosine similarity
    use_half_precision: bool = False  # Reduce memory usage
    auto_initialize: bool = True  # Auto-initialize embeddings on construction
    use_rich: bool = True  # Use rich progress display when available


@dataclass
class SemanticSearch:
    """
    High-performance semantic search using sentence transformers.

    Optimized for LLM usage with intelligent caching, batch processing,
    and efficient similarity computation. Uses state-of-the-art sentence
    transformer models for semantic understanding.

    Features:
    - Automatic embedding caching with validation
    - GPU acceleration when available
    - Batch processing for efficiency
    - Multiple similarity metrics
    - Integration with DocumentStore full-text search
    """

    config: SemanticSearchConfig = field(default_factory=SemanticSearchConfig)
    document_store: DocumentStore = field(default_factory=DocumentStore)

    # Internal state
    _embedding_manager: EmbeddingManager | None = field(default=None, init=False)
    _embeddings_cache: EmbeddingCache | None = field(default=None, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Initialize the semantic search system metadata only."""
        # Create DocumentStore with ids_set if none provided
        if self.document_store is None:
            self.document_store = DocumentStore(ids_set=self.config.ids_set)
        else:
            # Validate that provided DocumentStore has matching ids_set
            if self.document_store.ids_set != self.config.ids_set:
                raise ValueError(
                    f"DocumentStore ids_set {self.document_store.ids_set} "
                    f"does not match SemanticSearchConfig ids_set {self.config.ids_set}"
                )

        # Create embedding manager
        embedding_config = EmbeddingConfig(
            model_name=self.config.model_name,
            device=self.config.device,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            use_half_precision=self.config.use_half_precision,
            enable_cache=self.config.enable_cache,
            ids_set=self.config.ids_set,
            use_rich=self.config.use_rich,
        )

        from imas_mcp.embeddings.manager import get_embedding_manager

        # Create manager_id based on model configuration for sharing
        manager_id = f"{embedding_config.model_name}_{embedding_config.device}"
        self._embedding_manager = get_embedding_manager(
            config=embedding_config, manager_id=manager_id
        )

        # Initialize the embeddings only if auto_initialize is True
        if self.config.auto_initialize:
            self._initialize()

    def _get_embeddings_dir(self) -> Path:
        """Get the embeddings directory within resources using modern importlib."""
        # Get the resources directory for the imas_mcp package
        resources_dir = Path(str(files("imas_mcp") / "resources"))

        # Create embeddings subdirectory
        embeddings_dir = resources_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        return embeddings_dir

    def _generate_cache_filename(self) -> str:
        """Generate a unique cache filename based on configuration."""
        # Extract clean model name (remove path and normalize)
        model_name = self.config.model_name.split("/")[-1].replace("-", "_")

        # Build configuration parts for hashing (excluding model name,
        # batch_size, and threshold)
        config_parts = [
            f"norm_{self.config.normalize_embeddings}",
            f"half_{self.config.use_half_precision}",
        ]

        # Add IDS set to hash computation only if using a subset
        if self.config.ids_set:
            # Sort IDS names for consistent hashing
            ids_list = sorted(self.config.ids_set)
            config_parts.append(f"ids_{'_'.join(ids_list)}")

        # Compute short hash from config parts
        config_str = "_".join(config_parts)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Generate clean filename: .{model_name}_{hash}.pkl for ids_set,
        # .{model_name}.pkl for full
        if self.config.ids_set:
            filename = f".{model_name}_{config_hash}.pkl"
        else:
            filename = f".{model_name}.pkl"

        logger.debug(
            f"Generated cache filename: {filename} (from config: {config_str})"
        )
        return filename

    def _initialize(self, force_rebuild: bool = False) -> None:
        """Initialize the sentence transformer model and embeddings.

        Args:
            force_rebuild: If True, regenerate embeddings even if valid cache exists
        """
        with self._lock:
            if self._initialized and not force_rebuild:
                return

            logger.info(
                f"⚡ IMAS-MCP: Initializing semantic search with model: "
                f"{self.config.model_name}"
            )
            if self.config.ids_set:
                logger.info(
                    f"⚡ IMAS-MCP: Limited to IDS: {sorted(self.config.ids_set)}"
                )
            else:
                logger.info("⚡ IMAS-MCP: Processing all available IDS")

            # Load or generate embeddings using embedding manager
            logger.info("⚡ IMAS-MCP: Preparing document embeddings...")
            self._load_or_generate_embeddings(force_rebuild=force_rebuild)

            self._initialized = True
            logger.info("⚡ IMAS-MCP: Semantic search initialization complete! 🚀")

    def _load_or_generate_embeddings(self, force_rebuild: bool = False) -> None:
        """Load cached embeddings or generate new ones using EmbeddingManager.

        Args:
            force_rebuild: If True, regenerate embeddings even if valid cache exists
        """
        # Get all documents and their embedding texts
        documents = self.document_store.get_all_documents()

        if not documents:
            logger.info("No documents found for embedding generation")
            self._embeddings_cache = EmbeddingCache()
            return

        texts = [doc.embedding_text for doc in documents]
        identifiers = [doc.metadata.path_id for doc in documents]

        # Use embedding manager to get embeddings
        if self._embedding_manager is None:
            raise RuntimeError("Embedding manager not initialized")

        # Generate cache key using centralized config method
        cache_key = self._embedding_manager.config.generate_cache_key()

        embeddings, path_ids, was_cached = self._embedding_manager.get_embeddings(
            texts=texts,
            identifiers=identifiers,
            cache_key=cache_key,
            force_rebuild=force_rebuild,
            source_data_dir=self.document_store._data_dir,
        )

        # Store in our format for compatibility
        self._embeddings_cache = EmbeddingCache(
            embeddings=embeddings,
            path_ids=path_ids,
            model_name=self.config.model_name,
            document_count=len(documents),
            ids_set=self.config.ids_set,
        )

        # Update source metadata
        source_data_dir = self.document_store._data_dir
        self._embeddings_cache.update_source_metadata(source_data_dir)

        logger.info("IMAS-MCP: Loaded embeddings using EmbeddingManager")

    def get_document_count(self) -> int:
        """Get the count of documents in the document store."""
        return self.document_store.get_document_count()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        ids_filter: list[str] | None = None,
        hybrid_search: bool = True,
    ) -> list[SemanticSearchResult]:
        """
        Perform semantic search with optional hybrid full-text search.

        Args:
            query: Search query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            ids_filter: Optional list of IDS names to filter by
            hybrid_search: Combine with full-text search for better results

        Returns:
            List of search results ordered by similarity
        """
        # Ensure initialization before search
        if not self._initialized:
            self._initialize()

        if not self._embedding_manager or not self._embeddings_cache:
            raise RuntimeError("Search not properly initialized")

        top_k = top_k or self.config.default_top_k
        similarity_threshold = similarity_threshold or self.config.similarity_threshold

        # Generate query embedding using direct encoding (no caching)
        query_embedding = self._embedding_manager.encode_texts([query])[0]

        # Compute similarities
        similarities = self._compute_similarities(query_embedding)

        # Get candidate indices
        candidate_indices = self._get_candidate_indices(
            similarities, top_k * 2, similarity_threshold, ids_filter
        )

        # Create results
        results = []
        for rank, idx in enumerate(candidate_indices):
            path_id = self._embeddings_cache.path_ids[idx]
            document = self.document_store.get_document(path_id)

            if document:
                result = SemanticSearchResult(
                    document=document,
                    similarity_score=float(similarities[idx]),
                    rank=rank,
                )
                results.append(result)

        # Optional hybrid search - boost results that also match full-text search
        if hybrid_search and len(results) > 0:
            results = self._apply_hybrid_boost(query, results)

        # Final filtering and sorting
        results = [r for r in results if r.similarity_score >= similarity_threshold]
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        return results[:top_k]

    def _compute_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between query and all document embeddings."""
        if not self._embeddings_cache:
            raise RuntimeError("Embeddings cache not initialized")

        # Handle empty embeddings case
        if self._embeddings_cache.embeddings.shape[0] == 0:
            return np.array([])

        if self.config.normalize_embeddings:
            # Fast cosine similarity for normalized embeddings
            similarities = np.dot(self._embeddings_cache.embeddings, query_embedding)
        else:
            # Standard cosine similarity
            doc_norms = np.linalg.norm(self._embeddings_cache.embeddings, axis=1)
            query_norm = np.linalg.norm(query_embedding)
            similarities = np.dot(
                self._embeddings_cache.embeddings, query_embedding
            ) / (doc_norms * query_norm)

        return similarities

    def _get_candidate_indices(
        self,
        similarities: np.ndarray,
        max_candidates: int,
        similarity_threshold: float,
        ids_filter: list[str] | None,
    ) -> list[int]:
        """Get candidate document indices based on similarity and filters."""
        if not self._embeddings_cache:
            raise RuntimeError("Embeddings cache not initialized")

        # Apply similarity threshold
        valid_mask = similarities >= similarity_threshold

        # Apply IDS filter if specified
        if ids_filter:
            ids_mask = []
            for path_id in self._embeddings_cache.path_ids:
                doc = self.document_store.get_document(path_id)
                if doc and doc.metadata.ids_name in ids_filter:
                    ids_mask.append(True)
                else:
                    ids_mask.append(False)
            ids_mask = np.array(ids_mask)
            valid_mask = valid_mask & ids_mask

        # Get top candidates
        valid_indices = np.where(valid_mask)[0]
        valid_similarities = similarities[valid_indices]

        # Sort by similarity
        sorted_order = np.argsort(valid_similarities)[::-1]
        top_indices = valid_indices[sorted_order[:max_candidates]]

        return top_indices.tolist()

    def _apply_hybrid_boost(
        self, query: str, results: list[SemanticSearchResult]
    ) -> list[SemanticSearchResult]:
        """Apply hybrid boost by combining with full-text search."""
        try:
            # Get full-text search results
            fts_results = self.document_store.search_full_text(query, max_results=50)
            fts_path_ids = {doc.metadata.path_id for doc in fts_results}

            # Boost semantic results that also appear in full-text search
            boosted_results = []
            for result in results:
                boost_factor = 1.1 if result.path_id in fts_path_ids else 1.0

                boosted_result = SemanticSearchResult(
                    document=result.document,
                    similarity_score=result.similarity_score * boost_factor,
                    rank=result.rank,
                )
                boosted_results.append(boosted_result)

            return boosted_results

        except Exception as e:
            logger.warning(f"Hybrid search boost failed: {e}")
            return results

    def search_similar_documents(
        self, path_id: str, top_k: int = 5
    ) -> list[SemanticSearchResult]:
        """Find documents similar to a given document."""
        document = self.document_store.get_document(path_id)
        if not document:
            return []

        return self.search(
            document.embedding_text,
            top_k=top_k + 1,  # +1 to exclude the source document
            hybrid_search=False,
        )[1:]  # Skip the first result (the document itself)

    def get_embeddings_info(self) -> dict[str, Any]:
        """Get information about the embeddings cache."""
        if not self._embeddings_cache:
            return {"status": "not_initialized"}

        if self._embedding_manager:
            return self._embedding_manager.get_cache_info()

        # Fallback to basic info if no embedding manager
        cache_info = {
            "model_name": self._embeddings_cache.model_name,
            "document_count": self._embeddings_cache.document_count,
            "embedding_dimension": self._embeddings_cache.embeddings.shape[1],
            "dtype": str(self._embeddings_cache.embeddings.dtype),
            "created_at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._embeddings_cache.created_at)
            ),
            "memory_usage_mb": self._embeddings_cache.embeddings.nbytes / (1024 * 1024),
        }

        return cache_info

    def cache_status(self) -> dict[str, Any]:
        """Get cache status without initializing embeddings.

        Returns information about cache file existence and validity
        without loading model or generating embeddings.
        """
        if not self.config.enable_cache:
            return {"status": "cache_disabled"}

        if self._embedding_manager:
            return self._embedding_manager.get_cache_info()

        return {"status": "not_initialized"}

    def list_cache_files(self) -> list[dict[str, Any]]:
        """List all cache files in the embeddings directory.

        Returns a list of cache file information including size and modification time.
        Useful for cache management and cleanup.
        """
        if self._embedding_manager:
            return self._embedding_manager.list_cache_files()

        return []

    def cleanup_old_caches(self, keep_count: int = 3) -> int:
        """Remove old cache files, keeping only the most recent ones.

        Args:
            keep_count: Number of most recent cache files to keep

        Returns:
            Number of files removed
        """
        if self._embedding_manager:
            return self._embedding_manager.cleanup_old_caches(keep_count)

        return 0

    @staticmethod
    def list_all_cache_files() -> list[dict[str, Any]]:
        """List all cache files in the embeddings directory (static method).

        Returns a list of cache file information including size and modification time.
        Useful for cache management without needing a SemanticSearch instance.
        """
        try:
            # Get embeddings directory
            from importlib.resources import files

            resources_dir = Path(str(files("imas_mcp") / "resources"))
            embeddings_dir = resources_dir / "embeddings"

            if not embeddings_dir.exists():
                return []

            cache_files = []
            for cache_file in embeddings_dir.glob("*.pkl"):
                if cache_file.name.startswith("."):  # Our cache files start with .
                    stat = cache_file.stat()
                    cache_files.append(
                        {
                            "filename": cache_file.name,
                            "path": str(cache_file),
                            "size_mb": stat.st_size / (1024 * 1024),
                            "modified": time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
                            ),
                            "current": False,  # Can't determine current without config
                        }
                    )

            # Sort by modification time (newest first)
            cache_files.sort(key=lambda x: x["modified"], reverse=True)
            return cache_files

        except Exception as e:
            logger.error(f"Failed to list cache files: {e}")
            return []

    @staticmethod
    def cleanup_all_old_caches(keep_count: int = 3) -> int:
        """Remove old cache files, keeping only the most recent ones (static method).

        Args:
            keep_count: Number of most recent cache files to keep

        Returns:
            Number of files removed
        """
        cache_files = SemanticSearch.list_all_cache_files()
        removed_count = 0

        try:
            # Keep most recent ones
            files_to_remove = cache_files[keep_count:]

            for cache_info in files_to_remove:
                cache_path = Path(cache_info["path"])
                cache_path.unlink()
                logger.info(f"Removed old cache: {cache_info['filename']}")
                removed_count += 1

            return removed_count

        except Exception as e:
            logger.error(f"Failed to cleanup old caches: {e}")
            return removed_count

    def rebuild_embeddings(self) -> None:
        """Force rebuild of embeddings by overwriting existing cache.

        This method safely overwrites the existing cache file, so if the rebuild
        is cancelled, the original cache remains intact until completion.
        """
        with self._lock:
            # Clear in-memory cache but keep file until new one is written
            self._embeddings_cache = None
            self._initialized = False

            # Force rebuild - _save_cache will overwrite existing file
            logger.info("Rebuilding embeddings...")
            self._initialize(force_rebuild=True)

    def batch_search(
        self, queries: list[str], top_k: int = 10
    ) -> list[list[SemanticSearchResult]]:
        """Perform batch semantic search for multiple queries."""
        if not self._initialized:
            self._initialize()

        if not self._embedding_manager or not self._embeddings_cache:
            raise RuntimeError("Search not properly initialized")

        # Generate query embeddings in batch using direct encoding (no caching)
        query_embeddings = self._embedding_manager.encode_texts(queries)

        # Search for each query
        results = []
        for _i, query_embedding in enumerate(query_embeddings):
            similarities = self._compute_similarities(query_embedding)
            candidate_indices = self._get_candidate_indices(
                similarities, top_k, self.config.similarity_threshold, None
            )

            query_results = []
            for rank, idx in enumerate(candidate_indices):
                path_id = self._embeddings_cache.path_ids[idx]
                document = self.document_store.get_document(path_id)

                if document:
                    result = SemanticSearchResult(
                        document=document,
                        similarity_score=float(similarities[idx]),
                        rank=rank,
                    )
                    query_results.append(result)

            results.append(query_results[:top_k])

        return results

    @staticmethod
    def build_embeddings_on_install(
        ids_set: set | None = None,
        config: SemanticSearchConfig | None = None,
        force_rebuild: bool = False,
    ) -> bool:
        """
        Build embeddings during installation using default parameters.

        This method is designed to be called from build hooks to pre-generate
        embeddings during package installation, improving first-run performance.

        Args:
            ids_set: Optional set of IDS names to limit embedding generation
            config: Optional custom configuration (uses defaults if not provided)
            force_rebuild: Force rebuild even if cache exists

        Returns:
            True if embeddings were built successfully, False otherwise
        """
        try:
            # Use default configuration if not provided
            if config is None:
                config = SemanticSearchConfig()

            # Apply IDS set filter if provided
            if ids_set is not None:
                config.ids_set = ids_set

            logger.info(
                f"Building embeddings during installation with model: "
                f"{config.model_name}"
            )
            if ids_set:
                logger.info(f"Limited to IDS set: {sorted(ids_set)}")

            # Create document store with appropriate IDS set
            document_store = DocumentStore(ids_set=ids_set)

            # Create semantic search instance (this will trigger embedding generation)
            semantic_search = SemanticSearch(
                config=config, document_store=document_store
            )

            # Check if cache already exists and skip if not forcing rebuild
            if not force_rebuild and config.enable_cache:
                cache_filename = semantic_search._generate_cache_filename()
                cache_path = semantic_search._get_embeddings_dir() / cache_filename

                if cache_path.exists():
                    logger.info(f"Embeddings cache already exists: {cache_path}")
                    # Verify cache is valid by checking status
                    try:
                        cache_status = semantic_search.cache_status()
                        if cache_status.get("status") == "valid_cache":
                            logger.info("Existing cache is valid, skipping rebuild")
                            return True
                    except Exception as e:
                        logger.warning(f"Cache validation failed: {e}, rebuilding...")

            # Force initialization which will generate embeddings
            semantic_search._initialize()

            # Get info about generated embeddings
            info = semantic_search.get_embeddings_info()
            logger.info(
                f"Successfully built embeddings: {info['document_count']} documents, "
                f"{info.get('memory_usage_mb', 0):.1f} MB"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to build embeddings during installation: {e}")
            return False
            return False
