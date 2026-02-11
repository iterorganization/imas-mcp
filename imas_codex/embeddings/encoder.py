"""Embedding encoder providing model loading, cached corpus build, and ad-hoc embedding.

Supports two embedding backends:
- local: Uses SentenceTransformer with optional GPU acceleration
- remote: Connects to GPU server via HTTP (ITER login node or SSH tunnel)

No silent fallback: if the configured backend is unavailable, an error is raised.
All embeddings must use the same model (Qwen3-Embedding-0.6B) to ensure
vector compatibility across the knowledge graph.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_codex import dd_version

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
from imas_codex.core.progress_monitor import create_progress_monitor
from imas_codex.resource_path_accessor import ResourcePathAccessor
from imas_codex.settings import get_embedding_dimension

from .cache import EmbeddingCache
from .client import RemoteEmbeddingClient
from .config import EmbeddingBackend, EncoderConfig
from .openrouter_embed import EmbeddingResult


class EmbeddingBackendError(Exception):
    """Error raised when embedding backend is unavailable or misconfigured."""

    pass


class Encoder:
    """Load a sentence transformer model and produce embeddings with optional caching.

    No silent fallback: if the configured backend is unavailable, an error is
    raised immediately. This prevents mixing embeddings from different models
    which would corrupt vector indexes.
    """

    # Class-level: track which remote URLs have already logged validation
    _validated_urls: set[str] = set()
    _validated_urls_lock = threading.Lock()

    def __init__(
        self,
        config: EncoderConfig | None = None,
    ):
        self.config = config or EncoderConfig()
        self.logger = logging.getLogger(__name__)
        self._model: SentenceTransformer | None = None  # Lazy import at runtime
        self._cache: EmbeddingCache | None = None
        self._cache_path: Path | None = None
        self._lock = threading.RLock()
        self._remote_client: RemoteEmbeddingClient | None = None
        self._backend_validated: bool = False
        self._remote_hostname: str | None = None

        # Initialize based on backend selection
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the selected embedding backend."""
        backend = self.config.backend or EmbeddingBackend.LOCAL

        if backend == EmbeddingBackend.LOCAL:
            self._load_model()
        elif backend == EmbeddingBackend.REMOTE:
            if not self.config.remote_url:
                raise EmbeddingBackendError(
                    "Remote backend selected but embed-remote-url not configured. "
                    "Set IMAS_CODEX_EMBED_REMOTE_URL or embed-remote-url in pyproject.toml."
                )
            self._remote_client = RemoteEmbeddingClient(self.config.remote_url)
            # Eagerly fetch hostname for progress display source resolution.
            self._fetch_remote_hostname()
        else:
            raise EmbeddingBackendError(f"Unknown backend: {backend}")

    def _fetch_remote_hostname(self) -> None:
        """Eagerly fetch remote server hostname for source display.

        Tries multiple sources in order:
        1. /info endpoint (server.hostname) - requires updated server
        2. /health endpoint (hostname field) - simpler response

        Non-blocking best-effort: if nothing works, hostname
        will remain None and display will show 'remote'.
        """
        if not self._remote_client:
            return

        # Try /info endpoint first (has structured server.hostname)
        try:
            detailed = self._remote_client.get_detailed_info()
            if detailed:
                hn = detailed.get("server", {}).get("hostname")
                if hn:
                    self._remote_hostname = hn
                    self.logger.debug("Resolved hostname from /info: %s", hn)
                    return
        except Exception:
            pass

        # Try /health endpoint (may have hostname on newer servers)
        try:
            info = self._remote_client.get_info()
            if info and hasattr(info, "hostname") and info.hostname:
                self._remote_hostname = info.hostname
                self.logger.debug("Resolved hostname from /health: %s", info.hostname)
                return
        except Exception:
            pass

        # Fallback: if URL is localhost/127.0.0.1, server runs on this machine
        try:
            url = self.config.remote_url or ""
            if any(h in url for h in ("localhost", "127.0.0.1", "[::1]")):
                import socket

                hn = socket.gethostname()
                if hn:
                    self._remote_hostname = hn
                    self.logger.debug("Resolved hostname from local socket: %s", hn)
                    return
        except Exception:
            pass

    def _validate_remote_backend(self) -> None:
        """Validate remote backend is available and model matches.

        Checks server health with retries. Raises EmbeddingBackendError
        if the server is unavailable — no silent fallback.
        """
        if self._backend_validated:
            return

        if not self._remote_client:
            raise EmbeddingBackendError("Remote client not initialized")

        # Verify server health with retries
        last_error = None
        for attempt in range(3):
            timeout = 5.0 + attempt * 5.0  # 5s, 10s, 15s
            if self._remote_client.is_available(timeout=timeout):
                break
            last_error = (
                f"Health check timed out (attempt {attempt + 1}/3, timeout={timeout}s)"
            )
            if attempt < 2:
                time.sleep(1.0 + attempt)
        else:
            raise EmbeddingBackendError(
                f"Remote embedding server not available at {self.config.remote_url}. "
                f"{last_error}.\n"
                "Ensure SSH tunnel is active: ssh -f -N -L 18765:127.0.0.1:18765 iter\n"
                "Start server on ITER: imas-codex serve embed start --gpu 1\n"
                "Check server health: curl http://localhost:18765/health"
            )

        info = self._remote_client.get_info()
        if not info:
            raise EmbeddingBackendError(
                "Failed to get remote server info. Server may be misconfigured."
            )

        # Strict model validation - must match exactly
        expected_model = self.config.model_name
        if expected_model and info.model != expected_model:
            raise EmbeddingBackendError(
                f"Remote embedder model mismatch: expected '{expected_model}', "
                f"got '{info.model}'. Update embedding-backend config or restart server."
            )

        # Resolve hostname if not already set by eager fetch
        if not self._remote_hostname:
            # Try /health hostname first (cheapest, already fetched)
            if info.hostname:
                self._remote_hostname = info.hostname
            else:
                # Try /info fallback
                self._fetch_remote_hostname()

        # Log once per URL across all Encoder instances
        url = self.config.remote_url or ""
        with Encoder._validated_urls_lock:
            first_validation = url not in Encoder._validated_urls
            Encoder._validated_urls.add(url)

        if first_validation:
            self.logger.info(
                "Remote embedder: %s on %s (source=%s)",
                info.model,
                info.device,
                self._resolve_remote_source(),
            )
        else:
            self.logger.debug(
                "Remote embedder validated (source=%s)",
                self._resolve_remote_source(),
            )
        self._backend_validated = True

    def _truncate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Truncate embeddings from a remote server to configured dimension.

        Used only for the *remote* backend where the server may return
        native-dimension vectors.  Local and server-side encode paths
        use SentenceTransformer's native ``truncate_dim`` parameter instead,
        which truncates **before** normalization (correct Matryoshka ordering).

        Args:
            embeddings: Raw embeddings array (N x native_dim)

        Returns:
            Truncated and re-normalized embeddings (N x target_dim)
        """
        target_dim = get_embedding_dimension()
        if embeddings.ndim < 2 or embeddings.shape[1] <= target_dim:
            return embeddings

        truncated = embeddings[:, :target_dim]
        # Re-normalize after truncation to maintain unit vectors
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return truncated / norms

    @property
    def current_source(self) -> str:
        """Get current embedding source identifier.

        Returns:
            'local', 'remote', or host-specific like 'iter-login' or 'iter-titan'
        """
        backend = self.config.backend or EmbeddingBackend.LOCAL
        if backend == EmbeddingBackend.REMOTE:
            return self._resolve_remote_source()
        return "local"

    def _resolve_remote_source(self) -> str:
        """Resolve remote source to a descriptive name based on hostname."""
        hn = self._remote_hostname
        if not hn:
            return "remote"
        # ITER login nodes: 98dci4-srv-XXXX
        if hn.startswith("98dci4-srv"):
            return "iter-login"
        # ITER GPU compute nodes: 98dci4-gpu-XXXX (titan partition)
        if hn.startswith("98dci4-gpu"):
            return "iter-titan"
        # Generic: strip domain, use short hostname
        short = hn.split(".")[0]
        return f"iter-{short}" if "iter" in hn.lower() or "98dci4" in hn else short

    def build_document_embeddings(
        self,
        texts: list[str],
        identifiers: list[str] | None = None,
        cache_key: str | None = None,
        force_rebuild: bool = False,
        source_data_dir: Path | None = None,
        enable_caching: bool = True,
    ) -> tuple[np.ndarray, list[str], bool]:
        """Build or load embeddings for a corpus of documents.

        Returns (embeddings, identifiers, loaded_from_cache).
        """
        with self._lock:
            identifiers = identifiers or [f"text_{i}" for i in range(len(texts))]
            if len(texts) != len(identifiers):
                raise ValueError("Texts and identifiers must have the same length")
            if enable_caching:
                self._set_cache_path(cache_key)
            if (
                enable_caching
                and not force_rebuild
                and self._try_load_cache(texts, identifiers, source_data_dir)
            ):
                self.logger.debug("Loaded embeddings from cache")
                return self._cache.embeddings, self._cache.path_ids, True  # type: ignore[union-attr]
            self.logger.debug(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self._generate_embeddings(texts)
            if enable_caching:
                self._create_cache(embeddings, identifiers, source_data_dir)
            return embeddings, identifiers, False

    def embed_texts(self, texts: list[str], **kwargs) -> np.ndarray:
        """Embed ad-hoc texts (no caching).

        Uses the configured backend (local or remote).

        Raises:
            EmbeddingBackendError: If the backend is unavailable.
        """
        backend = self.config.backend or EmbeddingBackend.LOCAL

        if backend == EmbeddingBackend.REMOTE:
            with self._lock:
                self._validate_remote_backend()

            embeddings = self._remote_client.embed(  # type: ignore[union-attr]
                texts, normalize=self.config.normalize_embeddings
            )
            return self._truncate_embeddings(embeddings)

        # Local backend — truncate_dim set on model at init
        model = self.get_model()
        encode_kwargs = {
            "convert_to_numpy": True,
            "normalize_embeddings": self.config.normalize_embeddings,
            "batch_size": self.config.batch_size,
            "show_progress_bar": False,
            **kwargs,
        }
        return model.encode(texts, **encode_kwargs)

    def embed_texts_with_result(self, texts: list[str], **kwargs) -> EmbeddingResult:
        """Embed ad-hoc texts and return result with source tracking.

        Returns EmbeddingResult with:
        - source: "local" or "remote"
        - cost_usd: always 0.0 (local/remote are free)
        - total_tokens: always 0

        Raises:
            EmbeddingBackendError: If the backend is unavailable.
        """
        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                total_tokens=0,
                cost_usd=0.0,
                model=self.config.model_name,
                elapsed_seconds=0.0,
                source="local",
            )

        backend = self.config.backend or EmbeddingBackend.LOCAL
        start = time.time()

        if backend == EmbeddingBackend.REMOTE:
            with self._lock:
                self._validate_remote_backend()

            embeddings = self._remote_client.embed(  # type: ignore[union-attr]
                texts, normalize=self.config.normalize_embeddings
            )
            embeddings = self._truncate_embeddings(embeddings)
            elapsed = time.time() - start
            return EmbeddingResult(
                embeddings=embeddings,
                total_tokens=0,
                cost_usd=0.0,
                model=self.config.model_name,
                elapsed_seconds=elapsed,
                source=self._resolve_remote_source(),
            )

        # Local backend — truncate_dim set on model at init
        model = self.get_model()
        encode_kwargs = {
            "convert_to_numpy": True,
            "normalize_embeddings": self.config.normalize_embeddings,
            "batch_size": self.config.batch_size,
            "show_progress_bar": False,
            **kwargs,
        }
        embeddings = model.encode(texts, **encode_kwargs)
        elapsed = time.time() - start
        return EmbeddingResult(
            embeddings=embeddings,
            total_tokens=0,
            cost_usd=0.0,
            model=self.config.model_name,
            elapsed_seconds=elapsed,
            source="local",
        )

    @property
    def is_using_remote(self) -> bool:
        """Check if currently configured for remote embedding."""
        return self.config.backend == EmbeddingBackend.REMOTE

    def get_cache_info(self) -> dict[str, Any]:
        if not self._cache:
            return {"status": "no_cache"}
        info: dict[str, Any] = {
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
            info["cache_file_size_mb"] = self._cache_path.stat().st_size / (1024 * 1024)
            info["cache_file_path"] = str(self._cache_path)
        return info

    def list_cache_files(self) -> list[dict[str, Any]]:
        cache_dir = self._get_cache_directory()
        out: list[dict[str, Any]] = []
        try:
            for cache_file in cache_dir.glob("*.pkl"):
                stat = cache_file.stat()
                out.append(
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
            out.sort(key=lambda x: x["modified"], reverse=True)
            return out
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to list cache files: {e}")
            return []

    def cleanup_old_caches(self, keep_count: int = 3) -> int:
        files = self.list_cache_files()
        removed = 0
        current = str(self._cache_path) if self._cache_path else None
        try:
            for cache_info in files[keep_count:]:
                if cache_info["path"] != current:
                    Path(cache_info["path"]).unlink()
                    self.logger.debug(f"Removed old cache: {cache_info['filename']}")
                    removed += 1
            return removed
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to cleanup old caches: {e}")
            return removed

    def get_model(self) -> SentenceTransformer:
        """Get the local SentenceTransformer model, loading if needed."""
        if self._model is None:
            self._load_model()
        return self._model  # type: ignore[return-value]

    @staticmethod
    def _import_sentence_transformers() -> type:
        """Lazy import of SentenceTransformer to avoid torch/CUDA at import time."""
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer

    @staticmethod
    def _resolve_model_path(model_name: str, cache_folder: str | None = None) -> str:
        """Resolve a HuggingFace model name to a local directory path.

        If the model is cached locally (either in ``cache_folder`` or the
        default HuggingFace hub cache), returns the absolute path to the
        snapshot directory.  This lets ``SentenceTransformer`` load directly
        from disk without any HuggingFace API calls — critical for
        air-gapped GPU nodes on HPC clusters.

        Returns the original model name if not found locally.
        """
        try:
            from huggingface_hub import try_to_load_from_cache
            from huggingface_hub.errors import EntryNotFoundError

            # Check custom cache first, then default HF cache
            cache_dirs = [cache_folder] if cache_folder else []
            cache_dirs.append(None)  # None = default HF cache

            for cache_dir in cache_dirs:
                try:
                    path = try_to_load_from_cache(
                        model_name,
                        "config.json",
                        cache_dir=cache_dir,
                    )
                    if path and isinstance(path, (str, Path)):
                        # Return the snapshot directory (parent of config.json)
                        return str(Path(path).parent)
                except (EntryNotFoundError, Exception):
                    continue
        except ImportError:
            pass
        return model_name

    def _load_model(self) -> None:
        """Load local SentenceTransformer model.

        Searches for model files in two stages:
        1. Resolve model to local snapshot path (project cache or HF hub cache)
        2. Fall back to downloading from HuggingFace (requires internet)

        For CUDA devices, uses float16 for efficiency.

        Sets ``truncate_dim`` on the model so all encode calls
        (including ``encode_multi_process``) automatically truncate
        to the configured Matryoshka dimension and normalize **after**
        truncation — the correct ordering for Matryoshka embeddings.
        """
        ST = self._import_sentence_transformers()
        model_kwargs: dict = {}
        device = self.config.device
        target_dim = get_embedding_dimension()

        if device and "cuda" in device:
            import torch

            # float16 for GPU — P100 doesn't support bfloat16 natively.
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["low_cpu_mem_usage"] = True
        else:
            model_kwargs["torch_dtype"] = "auto"

        try:
            cache_folder = str(self._get_cache_directory() / "models")

            # Try to resolve to a local directory path first.
            # This bypasses all HuggingFace API calls and cache resolution.
            resolved = self._resolve_model_path(
                self.config.model_name, cache_folder=cache_folder
            )
            if resolved != self.config.model_name:
                self.logger.debug(
                    "Resolved %s to local path: %s", self.config.model_name, resolved
                )
                self._model = ST(
                    resolved,
                    device=device,
                    truncate_dim=target_dim,
                    model_kwargs=model_kwargs,
                )
            else:
                # No local cache found — download from HuggingFace
                self.logger.debug(
                    "No local cache for %s, downloading...", self.config.model_name
                )
                self._model = ST(
                    self.config.model_name,
                    device=device,
                    truncate_dim=target_dim,
                    cache_folder=cache_folder,
                    local_files_only=False,
                    model_kwargs=model_kwargs,
                )

            self.logger.debug(
                "Model %s loaded (device=%s)",
                self.config.model_name,
                self._model.device,
            )
        except Exception as e:  # pragma: no cover
            self.logger.error("Failed to load model %s: %s", self.config.model_name, e)
            raise EmbeddingBackendError(
                f"Failed to load embedding model '{self.config.model_name}': {e}"
            ) from e

    @property
    def device_info(self) -> str:
        """Human-readable device description for health reporting.

        Returns something like ``"Tesla P100-PCIE-16GB"`` on GPU
        or ``str(model.device)`` otherwise.
        """
        if not self._model:
            return "not loaded"
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return name
        except Exception:
            pass
        return str(self._model.device)

    def _generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts using configured backend.

        Raises:
            EmbeddingBackendError: If the configured backend is unavailable.
        """
        backend = self.config.backend or EmbeddingBackend.LOCAL

        if backend == EmbeddingBackend.REMOTE:
            self._validate_remote_backend()

            self.logger.debug(
                f"Generating embeddings remotely for {len(texts)} texts..."
            )
            embeddings = self._remote_client.embed(  # type: ignore[union-attr]
                texts, normalize=self.config.normalize_embeddings
            )
            embeddings = self._truncate_embeddings(embeddings)
            if self.config.use_half_precision:
                embeddings = embeddings.astype(np.float16)
            self.logger.debug(
                f"Generated embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}"
            )
            return embeddings

        # Local embedding with progress tracking
        if not self._model:
            self._load_model()
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
                batch_embeddings = self._model.encode(  # type: ignore[union-attr]
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False,
                )
                embeddings_list.append(batch_embeddings)
                progress.update_progress(batch_name)
            embeddings = np.vstack(embeddings_list)
        except Exception as e:  # pragma: no cover
            progress.finish_processing()
            self.logger.error(f"Error during embedding generation: {e}")
            raise
        finally:
            progress.finish_processing()
        if self.config.use_half_precision:
            embeddings = embeddings.astype(np.float16)
        self.logger.debug(
            f"Generated embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}"
        )
        return embeddings

    def _get_cache_directory(self) -> Path:
        path_accessor = ResourcePathAccessor(dd_version=dd_version)
        return path_accessor.embeddings_dir

    def _set_cache_path(self, cache_key: str | None = None) -> None:
        if self._cache_path is None:
            cache_filename = self._generate_cache_filename(cache_key)
            self._cache_path = self._get_cache_directory() / cache_filename
            if cache_key:
                self.logger.debug(f"Using cache key: '{cache_key}'")
            else:
                self.logger.debug("Using full dataset cache (no cache key)")
            self.logger.debug(f"Cache filename: {cache_filename}")
            if self._cache_path.exists():
                size_mb = self._cache_path.stat().st_size / (1024 * 1024)
                self.logger.debug(f"Cache file found: {size_mb:.1f} MB")
            else:
                self.logger.debug("Cache file not found - rebuild is required")

    def _generate_cache_filename(self, cache_key: str | None = None) -> str:
        model_name = self.config.model_name.split("/")[-1].replace("-", "_")
        parts = [
            f"norm_{self.config.normalize_embeddings}",
            f"half_{self.config.use_half_precision}",
        ]
        if self.config.ids_set:
            ids_list = sorted(self.config.ids_set)
            parts.append(f"ids_{'_'.join(ids_list)}")
        if cache_key:
            parts.append(f"key_{cache_key}")
        config_str = "_".join(parts)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        if cache_key or self.config.ids_set:
            return f".{model_name}_{config_hash}.pkl"
        return f".{model_name}.pkl"

    def _try_load_cache(
        self,
        texts: list[str],
        identifiers: list[str],
        source_data_dir: Path | None = None,
    ) -> bool:
        if not self.config.enable_cache:
            self.logger.debug("Cache disabled in configuration")
            return False
        if not self._cache_path:
            self.logger.warning("No cache path set")
            return False
        if not self._cache_path.exists():
            self.logger.debug(f"Cache file does not exist: {self._cache_path.name}")
            self.logger.debug("Rebuild required: Cache file not found")
            return False
        try:
            self.logger.debug(f"Attempting to load cache: {self._cache_path.name}")
            with open(self._cache_path, "rb") as f:
                cache = pickle.load(f)
            if not isinstance(cache, EmbeddingCache):
                self.logger.warning("Rebuild required: Invalid cache format")
                return False
            is_valid, reason = cache.validate_with_reason(
                len(texts), self.config.model_name, self.config.ids_set, source_data_dir
            )
            if not is_valid:
                self.logger.debug(f"Rebuild required: {reason}")
                return False
            if set(cache.path_ids) != set(identifiers):
                self.logger.debug("Rebuild required: Path identifiers have changed")
                return False
            self._cache = cache
            self.logger.debug("Cache validation successful - using existing embeddings")
            return True
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Rebuild required: Failed to load cache - {e}")
            return False

    def _create_cache(
        self,
        embeddings: np.ndarray,
        identifiers: list[str],
        source_data_dir: Path | None = None,
    ) -> None:
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
        if source_data_dir:
            self._cache.update_source_metadata(source_data_dir)
        if self._cache_path:
            try:
                with open(self._cache_path, "wb") as f:
                    pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                size_mb = self._cache_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"Saved embeddings cache: {size_mb:.1f} MB")
            except Exception as e:  # pragma: no cover
                self.logger.error(f"Failed to save embeddings cache: {e}")


__all__ = ["Encoder", "EmbeddingBackendError"]
