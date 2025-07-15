"""
Semantic search using sentence transformers with IMAS DocumentStore.

This module provides high-performance semantic search capabilities optimized for
LLM usage, using state-of-the-art sentence transformer models with efficient
vector storage and retrieval.
"""

import logging
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .document_store import Document, DocumentStore

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
    path_ids: List[str] = field(default_factory=list)
    model_name: str = ""
    created_at: float = field(default_factory=time.time)
    document_count: int = 0
    ids_set: Optional[set] = None  # IDS set used for this cache

    def is_valid(
        self,
        current_doc_count: int,
        current_model: str,
        current_ids_set: Optional[set] = None,
    ) -> bool:
        """Check if cache is valid for current state."""
        return (
            self.document_count == current_doc_count
            and self.model_name == current_model
            and len(self.path_ids) == current_doc_count
            and self.embeddings.shape[0] == current_doc_count
            and self.ids_set == current_ids_set
        )


@dataclass
class SemanticSearchConfig:
    """Configuration for semantic search."""

    # Model configuration
    model_name: str = "all-MiniLM-L6-v2"  # Fast, good quality model
    device: Optional[str] = None  # Auto-detect GPU/CPU

    # Search configuration
    default_top_k: int = 10
    similarity_threshold: float = 0.0  # Minimum similarity to return
    batch_size: int = 32  # For embedding generation
    ids_set: Optional[set] = None  # Limit to specific IDS for testing/performance

    # Cache configuration
    enable_cache: bool = True
    cache_file: Optional[str] = None  # Auto-generated based on config if None

    # Performance optimization
    normalize_embeddings: bool = True  # Faster cosine similarity
    use_half_precision: bool = False  # Reduce memory usage


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
    document_store: Optional[DocumentStore] = field(default=None)

    # Internal state
    _model: Optional[SentenceTransformer] = field(default=None, init=False)
    _embeddings_cache: Optional[EmbeddingCache] = field(default=None, init=False)
    _cache_path: Optional[Path] = field(default=None, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Initialize the semantic search system metadata only."""
        if self.document_store is None:
            self.document_store = DocumentStore()

        # Set cache path in same directory as document store
        if self.config.enable_cache:
            cache_filename = self._generate_cache_filename()
            self._cache_path = self.document_store._data_dir / cache_filename

        # NOTE: Removed automatic _initialize() call for lazy initialization
        # Model and embeddings are now loaded only when needed

    def _generate_cache_filename(self) -> str:
        """Generate a unique cache filename based on configuration."""
        import hashlib

        if self.config.cache_file:
            # Use provided cache file name
            return self.config.cache_file

        # Extract clean model name (remove path and normalize)
        model_name = self.config.model_name.split("/")[-1].replace("-", "_")

        # Build configuration parts for hashing (excluding model name)
        config_parts = [
            f"batch_{self.config.batch_size}",
            f"norm_{self.config.normalize_embeddings}",
            f"half_{self.config.use_half_precision}",
            f"threshold_{self.config.similarity_threshold}",
        ]

        # Add IDS set to hash computation
        if self.config.ids_set:
            # Sort IDS names for consistent hashing
            ids_list = sorted(list(self.config.ids_set))
            config_parts.append(f"ids_{'_'.join(ids_list)}")
        else:
            config_parts.append("ids_all")

        # Compute short hash from config parts
        config_str = "_".join(config_parts)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Generate clean filename: .{model_name}_{hash}.pkl
        filename = f".{model_name}_{config_hash}.pkl"

        logger.debug(
            f"Generated cache filename: {filename} (from config: {config_str})"
        )
        return filename

    def _initialize(self) -> None:
        """Initialize the sentence transformer model and embeddings."""
        with self._lock:
            if self._initialized:
                return

            logger.info(
                f"Initializing semantic search with model: {self.config.model_name}"
            )

            # Load sentence transformer model
            self._load_model()

            # Load or generate embeddings
            self._load_or_generate_embeddings()

            self._initialized = True
            logger.info("Semantic search initialization complete")

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            self._model = SentenceTransformer(
                self.config.model_name, device=self.config.device
            )
            logger.info(
                f"Loaded model {self.config.model_name} on device: {self._model.device}"
            )

        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            # Fallback to a known working model
            fallback_model = "all-MiniLM-L6-v2"
            logger.info(f"Trying fallback model: {fallback_model}")
            self._model = SentenceTransformer(fallback_model, device=self.config.device)
            self.config.model_name = fallback_model

    def _load_or_generate_embeddings(self) -> None:
        """Load cached embeddings or generate new ones."""
        if self.config.enable_cache and self._try_load_cache():
            logger.info("Loaded embeddings from cache")
            return

        logger.info("Generating embeddings for all documents...")
        self._generate_embeddings()

        if self.config.enable_cache:
            self._save_cache()

    def _get_filtered_document_count(self) -> int:
        """Get the count of documents after applying IDS filtering."""
        if self.document_store is None:
            return 0

        if not self.config.ids_set:
            return len(self.document_store._index.by_path_id)

        # Count documents that match the IDS set
        documents = self.document_store.get_all_documents()
        filtered_count = sum(
            1 for doc in documents if doc.metadata.ids_name in self.config.ids_set
        )
        return filtered_count

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

            current_doc_count = self._get_filtered_document_count()
            if not cache.is_valid(
                current_doc_count, self.config.model_name, self.config.ids_set
            ):
                logger.info("Cache invalid - document count or model changed")
                return False

            self._embeddings_cache = cache
            return True

        except Exception as e:
            logger.warning(f"Failed to load embeddings cache: {e}")
            return False

    def _generate_embeddings(self) -> None:
        """Generate embeddings for all documents."""
        documents = self.document_store.get_all_documents()

        # Filter documents by IDS set if specified
        if self.config.ids_set:
            documents = [
                doc for doc in documents if doc.metadata.ids_name in self.config.ids_set
            ]
            logger.info(
                f"Filtered to {len(documents)} documents from IDS set: {self.config.ids_set}"
            )

        if not documents:
            logger.warning("No documents found for embedding generation")
            self._embeddings_cache = EmbeddingCache()
            return

        logger.info(f"Generating embeddings for {len(documents)} documents...")

        # Extract embedding texts
        texts = [doc.embedding_text for doc in documents]
        path_ids = [doc.metadata.path_id for doc in documents]

        # Generate embeddings in batches
        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )

        # Convert to half precision if requested
        if self.config.use_half_precision:
            embeddings = embeddings.astype(np.float16)

        # Create cache
        self._embeddings_cache = EmbeddingCache(
            embeddings=embeddings,
            path_ids=path_ids,
            model_name=self.config.model_name,
            document_count=len(documents),
            ids_set=self.config.ids_set,
        )

        logger.info(
            f"Generated embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}"
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
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_ids: Optional[List[str]] = None,
        hybrid_search: bool = True,
    ) -> List[SemanticSearchResult]:
        """
        Perform semantic search with optional hybrid full-text search.

        Args:
            query: Search query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            filter_ids: Optional list of IDS names to filter by
            hybrid_search: Combine with full-text search for better results

        Returns:
            List of search results ordered by similarity
        """
        # Ensure initialization before search
        if not self._initialized:
            self._initialize()

        top_k = top_k or self.config.default_top_k
        similarity_threshold = similarity_threshold or self.config.similarity_threshold

        # Generate query embedding
        query_embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )[0]

        # Compute similarities
        similarities = self._compute_similarities(query_embedding)

        # Get candidate indices
        candidate_indices = self._get_candidate_indices(
            similarities, top_k * 2, similarity_threshold, filter_ids
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
        filter_ids: Optional[List[str]],
    ) -> List[int]:
        """Get candidate document indices based on similarity and filters."""
        # Apply similarity threshold
        valid_mask = similarities >= similarity_threshold

        # Apply IDS filter if specified
        if filter_ids:
            ids_mask = np.array(
                [
                    self.document_store.get_document(path_id).metadata.ids_name
                    in filter_ids
                    for path_id in self._embeddings_cache.path_ids
                ]
            )
            valid_mask = valid_mask & ids_mask

        # Get top candidates
        valid_indices = np.where(valid_mask)[0]
        valid_similarities = similarities[valid_indices]

        # Sort by similarity
        sorted_order = np.argsort(valid_similarities)[::-1]
        top_indices = valid_indices[sorted_order[:max_candidates]]

        return top_indices.tolist()

    def _apply_hybrid_boost(
        self, query: str, results: List[SemanticSearchResult]
    ) -> List[SemanticSearchResult]:
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
    ) -> List[SemanticSearchResult]:
        """Find documents similar to a given document."""
        document = self.document_store.get_document(path_id)
        if not document:
            return []

        return self.search(
            document.embedding_text,
            top_k=top_k + 1,  # +1 to exclude the source document
            hybrid_search=False,
        )[1:]  # Skip the first result (the document itself)

    def get_embeddings_info(self) -> Dict[str, Any]:
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

    def clear_cache(self) -> None:
        """Clear the embeddings cache."""
        with self._lock:
            if self._cache_path and self._cache_path.exists():
                try:
                    self._cache_path.unlink()
                    logger.info("Embeddings cache cleared")
                except Exception as e:
                    logger.error(f"Failed to clear cache: {e}")

            self._embeddings_cache = None
            self._initialized = False

    def rebuild_embeddings(self) -> None:
        """Force rebuild of embeddings."""
        with self._lock:
            logger.info("Rebuilding embeddings...")
            self.clear_cache()
            self._initialize()

    def batch_search(
        self, queries: List[str], top_k: int = 10
    ) -> List[List[SemanticSearchResult]]:
        """Perform batch semantic search for multiple queries."""
        if not self._initialized:
            self._initialize()

        # Generate query embeddings in batch
        query_embeddings = self._model.encode(
            queries,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )

        # Search for each query
        results = []
        for i, query_embedding in enumerate(query_embeddings):
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
