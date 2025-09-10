"""Semantic search implementation using injected Embeddings instance.

This module assumes the caller supplies an initialized (or initializing) Embeddings
object; no global registry or implicit construction is performed here.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imas_mcp.embeddings import Embeddings
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
    """Configuration for semantic search.

    Args:
        default_top_k: Fallback top_k if none provided at query time
        similarity_threshold: Minimum similarity score to keep a result
        ids_set: Optional subset of IDS names to restrict embedding/document space
    """

    default_top_k: int = 10
    similarity_threshold: float = 0.0
    ids_set: set[str] | None = None


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
    embeddings: Embeddings = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.document_store is None:
            self.document_store = DocumentStore(ids_set=self.config.ids_set)
        if self.embeddings is None:  # explicit dependency required
            raise ValueError(
                "Embeddings instance must be provided explicitly (registry removed)"
            )

    def _get_embeddings_and_ids(self) -> tuple[np.ndarray, list[str]]:
        if not self.embeddings:
            raise RuntimeError("Embeddings component not set")
        matrix = self.embeddings.get_embeddings_matrix()
        path_ids = self.embeddings.get_path_ids()
        return matrix, path_ids

    def get_document_count(self) -> int:
        return self.document_store.get_document_count()

    def _compute_similarities(
        self, query_embedding: np.ndarray, matrix: np.ndarray
    ) -> np.ndarray:
        if matrix.size == 0:
            return np.array([])
        return np.dot(matrix, query_embedding)

    def _get_candidate_indices(
        self,
        similarities: np.ndarray,
        max_candidates: int,
        similarity_threshold: float,
        ids_filter: list[str] | None,
        path_ids: list[str],
    ) -> list[int]:
        valid_mask = similarities >= similarity_threshold
        if ids_filter:
            ids_mask = []
            for path_id in path_ids:
                doc = self.document_store.get_document(path_id)
                ids_mask.append(bool(doc and doc.metadata.ids_name in ids_filter))
            ids_mask = np.array(ids_mask)
            valid_mask = valid_mask & ids_mask
        valid_indices = np.where(valid_mask)[0]
        valid_similarities = similarities[valid_indices]
        sorted_order = np.argsort(valid_similarities)[::-1]
        top_indices = valid_indices[sorted_order[:max_candidates]]
        return top_indices.tolist()

    def _apply_hybrid_boost(
        self, query: str, results: list[SemanticSearchResult]
    ) -> list[SemanticSearchResult]:
        try:
            fts_results = self.document_store.search_full_text(query, max_results=50)
            fts_path_ids = {doc.metadata.path_id for doc in fts_results}
            boosted = []
            for r in results:
                factor = 1.1 if r.path_id in fts_path_ids else 1.0
                boosted.append(
                    SemanticSearchResult(
                        document=r.document,
                        similarity_score=r.similarity_score * factor,
                        rank=r.rank,
                    )
                )
            return boosted
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Hybrid search boost failed: {e}")
            return results

    def search(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        ids_filter: list[str] | None = None,
        hybrid_search: bool = True,
    ) -> list[SemanticSearchResult]:
        if not self.embeddings:
            raise RuntimeError("Embeddings not configured")
        top_k = top_k or self.config.default_top_k
        similarity_threshold = similarity_threshold or self.config.similarity_threshold
        query_embedding = self.embeddings.encode_texts([query])[0]
        matrix, path_ids = self._get_embeddings_and_ids()
        similarities = self._compute_similarities(query_embedding, matrix)
        candidate_indices = self._get_candidate_indices(
            similarities, top_k * 2, similarity_threshold, ids_filter, path_ids
        )
        results: list[SemanticSearchResult] = []
        for rank, idx in enumerate(candidate_indices):
            path_id = path_ids[idx]
            document = self.document_store.get_document(path_id)
            if document:
                results.append(
                    SemanticSearchResult(
                        document=document,
                        similarity_score=float(similarities[idx]),
                        rank=rank,
                    )
                )
        if hybrid_search and results:
            results = self._apply_hybrid_boost(query, results)
        results = [r for r in results if r.similarity_score >= similarity_threshold]
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    def search_similar_documents(
        self, path_id: str, top_k: int = 5
    ) -> list[SemanticSearchResult]:
        document = self.document_store.get_document(path_id)
        if not document:
            return []
        return self.search(
            document.embedding_text, top_k=top_k + 1, hybrid_search=False
        )[1:]

    def batch_search(
        self, queries: list[str], top_k: int = 10
    ) -> list[list[SemanticSearchResult]]:
        if not self.embeddings:
            raise RuntimeError("Embeddings not configured")
        matrix, path_ids = self._get_embeddings_and_ids()
        if matrix.size == 0:
            return [[] for _ in queries]
        query_embeddings = self.embeddings.encode_texts(queries)
        all_results: list[list[SemanticSearchResult]] = []
        for query_embedding in query_embeddings:
            similarities = self._compute_similarities(query_embedding, matrix)
            candidate_indices = self._get_candidate_indices(
                similarities, top_k, self.config.similarity_threshold, None, path_ids
            )
            query_results: list[SemanticSearchResult] = []
            for rank, idx in enumerate(candidate_indices):
                doc_id = path_ids[idx]
                doc = self.document_store.get_document(doc_id)
                if doc:
                    query_results.append(
                        SemanticSearchResult(
                            document=doc,
                            similarity_score=float(similarities[idx]),
                            rank=rank,
                        )
                    )
            all_results.append(query_results[:top_k])
        return all_results

    # Cache / info proxy methods to maintain external API
    def get_embeddings_info(self) -> dict[str, Any]:
        if not self.embeddings or not self.embeddings._embedding_manager:  # type: ignore[attr-defined]
            return {"status": "not_initialized"}
        return self.embeddings._embedding_manager.get_cache_info()  # type: ignore[attr-defined]

    def cache_status(self) -> dict[str, Any]:
        if self.embeddings and self.embeddings._embedding_manager:  # type: ignore[attr-defined]
            return self.embeddings._embedding_manager.get_cache_info()  # type: ignore[attr-defined]
        return {"status": "not_initialized"}

    def list_cache_files(self) -> list[dict[str, Any]]:
        if self.embeddings and self.embeddings._embedding_manager:  # type: ignore[attr-defined]
            return self.embeddings._embedding_manager.list_cache_files()  # type: ignore[attr-defined]
        return []

    def cleanup_old_caches(self, keep_count: int = 3) -> int:
        if self.embeddings and self.embeddings._embedding_manager:  # type: ignore[attr-defined]
            return self.embeddings._embedding_manager.cleanup_old_caches(keep_count)  # type: ignore[attr-defined]
        return 0

    @staticmethod
    def list_all_cache_files() -> list[dict[str, Any]]:
        try:
            from importlib.resources import files
            from pathlib import Path

            resources_dir = Path(str(files("imas_mcp") / "resources"))
            embeddings_dir = resources_dir / "embeddings"
            if not embeddings_dir.exists():
                return []
            cache_files = []
            for cache_file in embeddings_dir.glob("*.pkl"):
                if cache_file.name.startswith("."):
                    stat = cache_file.stat()
                    cache_files.append(
                        {
                            "filename": cache_file.name,
                            "path": str(cache_file),
                            "size_mb": stat.st_size / (1024 * 1024),
                            "modified": time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
                            ),
                            "current": False,
                        }
                    )
            cache_files.sort(key=lambda x: x["modified"], reverse=True)
            return cache_files
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to list cache files: {e}")
            return []

    @staticmethod
    def cleanup_all_old_caches(keep_count: int = 3) -> int:
        cache_files = SemanticSearch.list_all_cache_files()
        removed = 0
        try:
            files_to_remove = cache_files[keep_count:]
            from pathlib import Path

            for info in files_to_remove:
                Path(info["path"]).unlink()
                logger.info(f"Removed old cache: {info['filename']}")
                removed += 1
            return removed
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to cleanup old caches: {e}")
            return removed

    def rebuild_embeddings(self) -> None:  # pragma: no cover - explicit unsupported
        raise NotImplementedError(
            "Rebuild now handled by Embeddings component (not implemented here)."
        )
