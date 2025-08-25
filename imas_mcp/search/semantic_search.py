"""
Semantic search using sentence transformers with IMAS DocumentStore.

This module provides high-performance semantic search capabilities optimized for
LLM usage, using state-of-the-art sentence transformer models with efficient
vector storage and retrieval.
"""

import hashlib
import logging
import pickle
import threading
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from imas_mcp.core.progress_monitor import create_progress_monitor
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
class EmbeddingCache:
    """Cache for document embeddings with metadata."""

    embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    path_ids: list[str] = field(default_factory=list)
    model_name: str = ""
    created_at: float = field(default_factory=time.time)
    document_count: int = 0
    ids_set: set | None = None  # IDS set used for this cache
    source_content_hash: str = ""  # Hash of source data directory content
    source_max_mtime: float = 0.0  # Maximum modification time of source files

    def is_valid(
        self,
        current_doc_count: int,
        current_model: str,
        current_ids_set: set | None = None,
        source_data_dir: Path | None = None,
    ) -> bool:
        """Check if cache is valid for current state."""
        # Basic validation
        basic_valid = (
            self.document_count == current_doc_count
            and self.model_name == current_model
            and len(self.embeddings) > 0
            and len(self.path_ids) > 0
            and self.ids_set == current_ids_set
        )

        if not basic_valid:
            return False

        # Enhanced validation with source file checking
        if source_data_dir is not None:
            # Check if any source files are newer than cache creation
            if self._has_newer_source_files(source_data_dir):
                return False

            # Check source content hash if available
            if self.source_content_hash:
                current_hash = self._compute_source_content_hash(source_data_dir)
                if current_hash != self.source_content_hash:
                    return False

        return True

    def _has_newer_source_files(self, source_data_dir: Path) -> bool:
        """Check if any source JSON files are newer than cache creation."""
        try:
            # Check catalog file
            catalog_path = source_data_dir / "ids_catalog.json"
            if catalog_path.exists() and catalog_path.stat().st_mtime > self.created_at:
                return True

            # Check detailed files
            detailed_dir = source_data_dir / "detailed"
            if detailed_dir.exists():
                for json_file in detailed_dir.glob("*.json"):
                    if json_file.stat().st_mtime > self.created_at:
                        return True

            # Check if maximum mtime has changed (additional validation)
            if self.source_max_mtime > 0:
                current_max_mtime = self._get_max_source_mtime(source_data_dir)
                if current_max_mtime > self.source_max_mtime:
                    return True

            return False
        except Exception:
            # If we can't check, assume files are newer (safer to rebuild)
            return True

    def _compute_source_content_hash(self, source_data_dir: Path) -> str:
        """Compute hash of source data directory content."""
        import hashlib

        hash_data = str(source_data_dir.resolve())

        # Include IDS set in hash for proper cache isolation
        if self.ids_set:
            ids_str = "|".join(sorted(self.ids_set))
            hash_data += f"|ids:{ids_str}"

        return hashlib.md5(hash_data.encode()).hexdigest()

    def _get_max_source_mtime(self, source_data_dir: Path) -> float:
        """Get the maximum modification time of all source files."""
        max_mtime = 0.0

        try:
            # Check catalog file
            catalog_path = source_data_dir / "ids_catalog.json"
            if catalog_path.exists():
                max_mtime = max(max_mtime, catalog_path.stat().st_mtime)

            # Check detailed files
            detailed_dir = source_data_dir / "detailed"
            if detailed_dir.exists():
                for json_file in detailed_dir.glob("*.json"):
                    max_mtime = max(max_mtime, json_file.stat().st_mtime)
        except Exception:
            pass

        return max_mtime

    def update_source_metadata(self, source_data_dir: Path) -> None:
        """Update source file metadata for cache validation."""
        self.source_content_hash = self._compute_source_content_hash(source_data_dir)
        self.source_max_mtime = self._get_max_source_mtime(source_data_dir)


@dataclass
class SemanticSearchConfig:
    """Configuration for semantic search."""

    # Model configuration
    model_name: str = "all-MiniLM-L6-v2"  # Fast, good quality model
    device: str | None = None  # Auto-detect GPU/CPU

    # Search configuration
    default_top_k: int = 10
    similarity_threshold: float = 0.0  # Minimum similarity to return
    batch_size: int = 50  # For embedding generation
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
    _model: SentenceTransformer | None = field(default=None, init=False)
    _embeddings_cache: EmbeddingCache | None = field(default=None, init=False)
    _cache_path: Path | None = field(default=None, init=False)
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

        # Set cache path in embeddings directory within resources
        if self.config.enable_cache:
            cache_filename = self._generate_cache_filename()
            self._cache_path = self._get_embeddings_dir() / cache_filename

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

            # Load sentence transformer model
            logger.info("⚡ IMAS-MCP: Loading sentence transformer model...")
            self._load_model()

            # Load or generate embeddings
            logger.info("⚡ IMAS-MCP: Preparing document embeddings...")
            self._load_or_generate_embeddings(force_rebuild=force_rebuild)

            self._initialized = True
            logger.info("⚡ IMAS-MCP: Semantic search initialization complete! 🚀")

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            # Set cache folder to embeddings directory
            cache_folder = str(self._get_embeddings_dir() / "models")

            # Try to load with local_files_only first for speed
            try:
                logger.info("⚡ IMAS-MCP: Loading cached sentence transformer model...")
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                    cache_folder=cache_folder,
                    local_files_only=True,  # Prevent internet downloads
                )
                logger.info(
                    f"⚡ IMAS-MCP: Model {self.config.model_name} loaded from "
                    f"cache on device: {self._model.device}"
                )
            except Exception:
                # If local loading fails, try downloading
                logger.info(
                    f"⚡ IMAS-MCP: Model not in cache, downloading "
                    f"{self.config.model_name}..."
                )
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                    cache_folder=cache_folder,
                    local_files_only=False,  # Allow downloads
                )
                logger.info(
                    f"⚡ IMAS-MCP: Downloaded and loaded model "
                    f"{self.config.model_name} on device: {self._model.device}"
                )

        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            # Fallback to a known working model
            fallback_model = "all-MiniLM-L6-v2"
            logger.info(f"⚡ IMAS-MCP: Trying fallback model: {fallback_model}")
            self._model = SentenceTransformer(fallback_model, device=self.config.device)
            self.config.model_name = fallback_model

    def _load_or_generate_embeddings(self, force_rebuild: bool = False) -> None:
        """Load cached embeddings or generate new ones.

        Args:
            force_rebuild: If True, regenerate embeddings even if valid cache exists
        """
        if not force_rebuild and self.config.enable_cache and self._try_load_cache():
            logger.info("IMAS-MCP: Loaded embeddings from cache")
            return

        if force_rebuild:
            logger.info("IMAS-MCP: Force rebuild requested, regenerating embeddings...")

        self._generate_embeddings()

        if self.config.enable_cache:
            self._save_cache()

    def get_document_count(self) -> int:
        """Get the count of documents in the document store."""
        return self.document_store.get_document_count()

    def _try_load_cache(self) -> bool:
        """Try to load embeddings from cache."""
        if not self._cache_path or not self._cache_path.exists():
            return False

        try:
            with open(self._cache_path, "rb") as f:
                cache = pickle.load(f)

            if not isinstance(cache, EmbeddingCache):
                logger.warning("Invalid cache format")
                return False

            current_doc_count = self.get_document_count()
            # Get source data directory for enhanced validation
            source_data_dir = self.document_store._data_dir

            if not cache.is_valid(
                current_doc_count,
                self.config.model_name,
                self.config.ids_set,
                source_data_dir,
            ):
                logger.info(
                    f"Cache invalid - cached: {cache.document_count} docs, "
                    f"current: {current_doc_count} docs, "
                    f"cached model: {cache.model_name}, "
                    f"current model: {self.config.model_name}, "
                    f"cached IDS set: {cache.ids_set}, "
                    f"current IDS set: {self.config.ids_set}"
                )
                return False

            self._embeddings_cache = cache
            return True

        except Exception as e:
            logger.error(f"Failed to load embeddings cache: {e}")
            return False

    def _generate_embeddings(self) -> None:
        """Generate embeddings for all documents with memory-efficient processing."""
        if not self._model:
            raise RuntimeError("Model not loaded")

        # Get document count first to avoid loading all documents into memory
        doc_count = self.document_store.get_document_count()

        if doc_count == 0:
            logger.info("No documents found for embedding generation")
            self._embeddings_cache = EmbeddingCache()
            return

        logger.info(f"IMAS-MCP: Generating embeddings for {doc_count} documents...")

        # Calculate batch information for progress monitoring
        total_batches = (
            doc_count + self.config.batch_size - 1
        ) // self.config.batch_size
        batch_names = [
            f"{min((i + 1) * self.config.batch_size, doc_count)}/{doc_count}"
            for i in range(total_batches)
        ]

        # Create progress monitor for batch processing
        progress = create_progress_monitor(
            use_rich=self.config.use_rich,  # Use config setting
            logger=logger,
            item_names=batch_names,
            description_template="Embedding documents: {item}",
        )

        # Start progress monitoring
        progress.start_processing(batch_names, "IMAS-MCP: Embedding documents")

        try:
            embeddings_list = []
            path_ids = []

            # Get all documents once for iteration
            documents = self.document_store.get_all_documents()

            # Process documents in batches to reduce memory usage
            for i in range(0, len(documents), self.config.batch_size):
                docs_processed = min(
                    (i // self.config.batch_size + 1) * self.config.batch_size,
                    len(documents),
                )
                batch_name = f"{docs_processed}/{len(documents)}"
                progress.set_current_item(batch_name)

                # Extract batch documents
                batch_docs = documents[i : i + self.config.batch_size]

                # Extract texts and path_ids for this batch only
                batch_texts = [doc.embedding_text for doc in batch_docs]
                batch_path_ids = [doc.metadata.path_id for doc in batch_docs]

                # Generate embeddings for this batch
                batch_embeddings = self._model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False,  # Disable individual progress, use our own
                )

                embeddings_list.append(batch_embeddings)
                path_ids.extend(batch_path_ids)
                progress.update_progress(batch_name)

            # Combine all batch embeddings
            embeddings = np.vstack(embeddings_list)

        except Exception as e:
            progress.finish_processing()
            logger.error(f"Error during embedding generation: {e}")
            raise
        finally:
            progress.finish_processing()

        # Convert to half precision if requested
        if self.config.use_half_precision:
            embeddings = embeddings.astype(np.float16)

        # Create cache with enhanced metadata
        self._embeddings_cache = EmbeddingCache(
            embeddings=embeddings,
            path_ids=path_ids,
            model_name=self.config.model_name,
            document_count=len(documents),
            ids_set=self.config.ids_set,
        )

        # Update source metadata for validation
        source_data_dir = self.document_store._data_dir
        self._embeddings_cache.update_source_metadata(source_data_dir)

        logger.info(
            f"IMAS-MCP: Generated embeddings: shape={embeddings.shape}, "
            f"dtype={embeddings.dtype}"
        )

    def _save_cache(self) -> None:
        """Save embeddings to cache."""
        if not self._cache_path or not self._embeddings_cache:
            return

        try:
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._embeddings_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

            cache_size_mb = self._cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"Saved embeddings cache: {cache_size_mb:.1f} MB")

        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {e}")

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

        if not self._model or not self._embeddings_cache:
            raise RuntimeError("Search not properly initialized")

        top_k = top_k or self.config.default_top_k
        similarity_threshold = similarity_threshold or self.config.similarity_threshold

        # Generate query embedding
        query_embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )[0]

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

        if self._cache_path and self._cache_path.exists():
            cache_info["cache_file_size_mb"] = self._cache_path.stat().st_size / (
                1024 * 1024
            )
            cache_info["cache_file_path"] = str(self._cache_path)

        return cache_info

    def cache_status(self) -> dict[str, Any]:
        """Get cache status without initializing embeddings.

        Returns information about cache file existence and validity
        without loading model or generating embeddings.
        """
        if not self.config.enable_cache or not self._cache_path:
            return {"status": "cache_disabled"}

        if not self._cache_path.exists():
            return {"status": "no_cache_file"}

        try:
            # Try to load cache metadata without full initialization
            with open(self._cache_path, "rb") as f:
                cache = pickle.load(f)

            if not isinstance(cache, EmbeddingCache):
                return {"status": "invalid_cache_file"}

            # Check basic validity
            document_count = len(self.document_store.get_all_documents())
            is_valid = cache.is_valid(
                current_doc_count=document_count,
                current_model=self.config.model_name,
                current_ids_set=self.config.ids_set,
                source_data_dir=self.document_store._data_dir,
            )

            if is_valid:
                return {
                    "status": "valid_cache",
                    "model_name": cache.model_name,
                    "document_count": cache.document_count,
                    "cache_file_size_mb": self._cache_path.stat().st_size
                    / (1024 * 1024),
                    "cache_file_path": str(self._cache_path),
                    "created_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(cache.created_at)
                    ),
                }
            else:
                return {"status": "invalid_cache"}

        except Exception as e:
            return {"status": "cache_error", "error": str(e)}

    def list_cache_files(self) -> list[dict[str, Any]]:
        """List all cache files in the embeddings directory.

        Returns a list of cache file information including size and modification time.
        Useful for cache management and cleanup.
        """
        embeddings_dir = self._get_embeddings_dir()
        cache_files = []

        try:
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
                            "current": cache_file == self._cache_path,
                        }
                    )

            # Sort by modification time (newest first)
            cache_files.sort(key=lambda x: x["modified"], reverse=True)
            return cache_files

        except Exception as e:
            logger.error(f"Failed to list cache files: {e}")
            return []

    def cleanup_old_caches(self, keep_count: int = 3) -> int:
        """Remove old cache files, keeping only the most recent ones.

        Args:
            keep_count: Number of most recent cache files to keep

        Returns:
            Number of files removed
        """
        cache_files = self.list_cache_files()
        removed_count = 0

        try:
            # Keep current cache file and most recent ones
            files_to_remove = []
            current_cache = str(self._cache_path) if self._cache_path else None

            for cache_info in cache_files[keep_count:]:
                # Never remove the current cache file
                if cache_info["path"] != current_cache:
                    files_to_remove.append(cache_info)

            for cache_info in files_to_remove:
                cache_path = Path(cache_info["path"])
                cache_path.unlink()
                logger.info(f"Removed old cache: {cache_info['filename']}")
                removed_count += 1

            return removed_count

        except Exception as e:
            logger.error(f"Failed to cleanup old caches: {e}")
            return removed_count

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

        if not self._model or not self._embeddings_cache:
            raise RuntimeError("Search not properly initialized")

        # Generate query embeddings in batch
        query_embeddings = self._model.encode(
            queries,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,  # Disable individual progress, use our own
        )

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
                    # Verify cache is valid
                    try:
                        if semantic_search._try_load_cache():
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
