"""Centralized embedding management for IMAS MCP."""

import hashlib
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from imas_mcp.core.progress_monitor import create_progress_monitor

from .cache import EmbeddingCache
from .config import EmbeddingConfig


class EmbeddingManager:
    """
    Centralized manager for embedding generation, caching, and retrieval.

    This class provides a clean interface for multiple systems to access
    embeddings with automatic caching and validation.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """Initialize the embedding manager."""
        self.config = config or EmbeddingConfig()
        self.logger = logging.getLogger(__name__)

        # Internal state
        self._model: SentenceTransformer | None = None
        self._cache: EmbeddingCache | None = None
        self._cache_path: Path | None = None
        self._lock = threading.RLock()
        self._initialized = False

    def get_embeddings(
        self,
        texts: list[str],
        identifiers: list[str] | None = None,
        cache_key: str | None = None,
        force_rebuild: bool = False,
        source_data_dir: Path | None = None,
        enable_caching: bool = True,
    ) -> tuple[np.ndarray, list[str], bool]:
        """
        Get embeddings for the given texts with automatic caching.

        Args:
            texts: List of texts to embed
            identifiers: Optional list of identifiers for the texts
            cache_key: Optional cache key for this specific embedding set
            force_rebuild: Force regeneration even if cache exists
            source_data_dir: Source data directory for validation
            enable_caching: Whether to enable caching for these embeddings (default: True)

        Returns:
            Tuple of (embeddings, identifiers, was_loaded_from_cache)
        """
        with self._lock:
            identifiers = identifiers or [f"text_{i}" for i in range(len(texts))]

            if len(texts) != len(identifiers):
                raise ValueError("Texts and identifiers must have the same length")

            # Set cache path only if caching is enabled
            if enable_caching:
                self._set_cache_path(cache_key)

            # Try to load from cache
            if (
                enable_caching
                and not force_rebuild
                and self._try_load_cache(texts, identifiers, source_data_dir)
            ):
                self.logger.info("Loaded embeddings from cache")
                return self._cache.embeddings, self._cache.path_ids, True

            # Generate fresh embeddings
            self.logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self._generate_embeddings(texts)

            # Create and save cache only if caching is enabled
            if enable_caching:
                self._create_cache(embeddings, identifiers, source_data_dir)

            return embeddings, identifiers, False

    def get_model(self) -> SentenceTransformer:
        """Get the sentence transformer model."""
        if self._model is None:
            self._load_model()
        return self._model

    def encode_texts(self, texts: list[str], **kwargs) -> np.ndarray:
        """Encode texts using the configured model."""
        model = self.get_model()

        # Apply default settings from config
        encode_kwargs = {
            "convert_to_numpy": True,
            "normalize_embeddings": self.config.normalize_embeddings,
            "batch_size": self.config.batch_size,
            "show_progress_bar": False,
            **kwargs,
        }

        return model.encode(texts, **encode_kwargs)

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the current cache."""
        if not self._cache:
            return {"status": "no_cache"}

        cache_info = {
            "model_name": self._cache.model_name,
            "document_count": self._cache.document_count,
            "embedding_dimension": self._cache.embeddings.shape[1]
            if len(self._cache.embeddings.shape) > 1
            else 0,
            "dtype": str(self._cache.embeddings.dtype),
            "created_at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._cache.created_at)
            ),
            "memory_usage_mb": self._cache.embeddings.nbytes / (1024 * 1024),
        }

        if self._cache_path and self._cache_path.exists():
            cache_info["cache_file_size_mb"] = self._cache_path.stat().st_size / (
                1024 * 1024
            )
            cache_info["cache_file_path"] = str(self._cache_path)

        return cache_info

    def list_cache_files(self) -> list[dict[str, Any]]:
        """List all cache files in the embeddings directory."""
        cache_dir = self._get_cache_directory()
        cache_files = []

        try:
            for cache_file in cache_dir.glob("*.pkl"):
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
            self.logger.error(f"Failed to list cache files: {e}")
            return []

    def cleanup_old_caches(self, keep_count: int = 3) -> int:
        """Remove old cache files, keeping only the most recent ones."""
        cache_files = self.list_cache_files()
        removed_count = 0

        try:
            current_cache = str(self._cache_path) if self._cache_path else None
            files_to_remove = []

            for cache_info in cache_files[keep_count:]:
                if cache_info["path"] != current_cache:
                    files_to_remove.append(cache_info)

            for cache_info in files_to_remove:
                cache_path = Path(cache_info["path"])
                cache_path.unlink()
                self.logger.info(f"Removed old cache: {cache_info['filename']}")
                removed_count += 1

            return removed_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup old caches: {e}")
            return removed_count

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            cache_folder = str(self._get_cache_directory() / "models")

            # Try local first
            try:
                self.logger.info("Loading cached sentence transformer model...")
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                    cache_folder=cache_folder,
                    local_files_only=True,
                )
                self.logger.info(
                    f"Model {self.config.model_name} loaded from cache on device: {self._model.device}"
                )
            except Exception:
                self.logger.info(
                    f"Model not in cache, downloading {self.config.model_name}..."
                )
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                    cache_folder=cache_folder,
                    local_files_only=False,
                )
                self.logger.info(
                    f"Downloaded and loaded model {self.config.model_name} on device: {self._model.device}"
                )

        except Exception as e:
            self.logger.error(f"Failed to load model {self.config.model_name}: {e}")
            # Fallback to known working model
            fallback_model = "all-MiniLM-L6-v2"
            self.logger.info(f"Trying fallback model: {fallback_model}")
            self._model = SentenceTransformer(fallback_model, device=self.config.device)
            self.config.model_name = fallback_model

    def _generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts with progress monitoring."""
        if not self._model:
            self._load_model()

        # Calculate batch information
        total_batches = (
            len(texts) + self.config.batch_size - 1
        ) // self.config.batch_size

        if self.config.use_rich:
            batch_names = [
                f"{min((i + 1) * self.config.batch_size, len(texts))}/{len(texts)} ({i + 1}/{total_batches})"
                for i in range(total_batches)
            ]
            description_template = "Embedding texts: {item}"
            start_description = "Embedding texts"
        else:
            batch_names = [
                f"{min((i + 1) * self.config.batch_size, len(texts))}/{len(texts)}"
                for i in range(total_batches)
            ]
            description_template = "Embedding texts: {item}"
            start_description = f"Embedding {len(texts)} texts"

        progress = create_progress_monitor(
            use_rich=self.config.use_rich,
            logger=self.logger,
            item_names=batch_names,
            description_template=description_template,
        )

        progress.start_processing(batch_names, start_description)

        try:
            embeddings_list = []

            for i in range(0, len(texts), self.config.batch_size):
                texts_processed = min(
                    (i // self.config.batch_size + 1) * self.config.batch_size,
                    len(texts),
                )
                batch_name = f"{texts_processed}/{len(texts)}"
                progress.set_current_item(batch_name)

                batch_texts = texts[i : i + self.config.batch_size]

                batch_embeddings = self._model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False,
                )

                embeddings_list.append(batch_embeddings)
                progress.update_progress(batch_name)

            embeddings = np.vstack(embeddings_list)

        except Exception as e:
            progress.finish_processing()
            self.logger.error(f"Error during embedding generation: {e}")
            raise
        finally:
            progress.finish_processing()

        # Convert to half precision if requested
        if self.config.use_half_precision:
            embeddings = embeddings.astype(np.float16)

        self.logger.info(
            f"Generated embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}"
        )
        return embeddings

    def _get_cache_directory(self) -> Path:
        """Get the cache directory."""
        from importlib.resources import files

        resources_dir = Path(str(files("imas_mcp") / "resources"))
        cache_dir = resources_dir / self.config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir

    def _set_cache_path(self, cache_key: str | None = None) -> None:
        """Set the cache file path with informative logging."""
        if self._cache_path is None:
            cache_filename = self._generate_cache_filename(cache_key)
            self._cache_path = self._get_cache_directory() / cache_filename

            # Log cache information
            if cache_key:
                self.logger.info(f"Using cache key: '{cache_key}'")
            else:
                self.logger.info("Using full dataset cache (no cache key)")

            self.logger.info(f"Cache filename: {cache_filename}")

            if self._cache_path.exists():
                cache_size_mb = self._cache_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"Cache file found: {cache_size_mb:.1f} MB")
            else:
                self.logger.info("Cache file not found - rebuild is required")

    def _generate_cache_filename(self, cache_key: str | None = None) -> str:
        """Generate cache filename based on configuration."""
        model_name = self.config.model_name.split("/")[-1].replace("-", "_")

        config_parts = [
            f"norm_{self.config.normalize_embeddings}",
            f"half_{self.config.use_half_precision}",
        ]

        if self.config.ids_set:
            ids_list = sorted(self.config.ids_set)
            config_parts.append(f"ids_{'_'.join(ids_list)}")

        if cache_key:
            config_parts.append(f"key_{cache_key}")

        config_str = "_".join(config_parts)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        if cache_key or self.config.ids_set:
            filename = f".{model_name}_{config_hash}.pkl"
        else:
            filename = f".{model_name}.pkl"

        return filename

    def _try_load_cache(
        self,
        texts: list[str],
        identifiers: list[str],
        source_data_dir: Path | None = None,
    ) -> bool:
        """Try to load embeddings from cache with detailed validation logging."""
        if not self.config.enable_cache:
            self.logger.info("Cache disabled in configuration")
            return False

        if not self._cache_path:
            self.logger.warning("No cache path set")
            return False

        if not self._cache_path.exists():
            self.logger.info(f"Cache file does not exist: {self._cache_path.name}")
            self.logger.info("Rebuild required: Cache file not found")
            return False

        try:
            self.logger.info(f"Attempting to load cache: {self._cache_path.name}")
            with open(self._cache_path, "rb") as f:
                cache = pickle.load(f)

            if not isinstance(cache, EmbeddingCache):
                self.logger.warning("Rebuild required: Invalid cache format")
                return False

            # Validate cache with detailed reason
            is_valid, reason = cache.validate_with_reason(
                len(texts),
                self.config.model_name,
                self.config.ids_set,
                source_data_dir,
            )

            if not is_valid:
                self.logger.info(f"Rebuild required: {reason}")
                return False

            # Check if identifiers match
            cached_set = set(cache.path_ids)
            current_set = set(identifiers)
            if cached_set != current_set:
                self.logger.info("Rebuild required: Path identifiers have changed")
                return False

            self._cache = cache
            self.logger.info("Cache validation successful - using existing embeddings")
            return True

        except Exception as e:
            self.logger.error(f"Rebuild required: Failed to load cache - {e}")
            return False

    def _create_cache(
        self,
        embeddings: np.ndarray,
        identifiers: list[str],
        source_data_dir: Path | None = None,
    ) -> None:
        """Create and save embedding cache."""
        if not self.config.enable_cache:
            return

        self._cache = EmbeddingCache(
            embeddings=embeddings,
            path_ids=identifiers,
            model_name=self.config.model_name,
            document_count=len(identifiers),
            ids_set=self.config.ids_set,
            created_at=time.time(),
        )

        # Update source metadata if provided
        if source_data_dir:
            self._cache.update_source_metadata(source_data_dir)

        # Save to file
        if self._cache_path:
            try:
                with open(self._cache_path, "wb") as f:
                    pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)

                cache_size_mb = self._cache_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"Saved embeddings cache: {cache_size_mb:.1f} MB")

            except Exception as e:
                self.logger.error(f"Failed to save embeddings cache: {e}")


# Global registry for shared embedding managers
_embedding_managers: dict[str, EmbeddingManager] = {}
_manager_lock = threading.RLock()


def get_embedding_manager(
    config: EmbeddingConfig | None = None, manager_id: str = "default"
) -> EmbeddingManager:
    """Get or create a shared embedding manager."""
    with _manager_lock:
        if manager_id not in _embedding_managers:
            _embedding_managers[manager_id] = EmbeddingManager(config)
        return _embedding_managers[manager_id]


def clear_embedding_managers() -> None:
    """Clear all embedding managers (useful for testing)."""
    with _manager_lock:
        _embedding_managers.clear()
