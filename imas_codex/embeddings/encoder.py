"""Embedding encoder providing model loading, cached corpus build, and ad-hoc embedding.

Supports multiple embedding backends:
- local: Uses SentenceTransformer with optional GPU acceleration
- remote: Connects to GPU server via HTTP (typically through SSH tunnel)
- openrouter: Uses OpenRouter API for cloud embeddings

No fallback between backends - if the selected backend fails, errors are raised.
The remote client has built-in retry logic for transient connection failures.
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

from .cache import EmbeddingCache
from .client import RemoteEmbeddingClient
from .config import EmbeddingBackend, EncoderConfig
from .openrouter_embed import (
    EmbeddingCostTracker,
    EmbeddingResult,
    OpenRouterEmbeddingClient,
)


class EmbeddingBackendError(Exception):
    """Error raised when embedding backend is unavailable or misconfigured."""

    pass


class Encoder:
    """Load a sentence transformer model and produce embeddings with optional caching.

    No fallback between backends - if the selected backend is unavailable,
    errors are raised. The remote client has built-in retry logic for
    transient connection failures.
    """

    def __init__(
        self,
        config: EncoderConfig | None = None,
        cost_tracker: EmbeddingCostTracker | None = None,
    ):
        self.config = config or EncoderConfig()
        self.logger = logging.getLogger(__name__)
        self._model: SentenceTransformer | None = None  # Lazy import at runtime
        self._cache: EmbeddingCache | None = None
        self._cache_path: Path | None = None
        self._lock = threading.RLock()
        self._remote_client: RemoteEmbeddingClient | None = None
        self._openrouter_client: OpenRouterEmbeddingClient | None = None
        self._backend_validated: bool = False
        self._fallback_warned: bool = False
        self._using_fallback: bool = False
        self._remote_hostname: str | None = None

        # Cost tracking for OpenRouter backend
        self.cost_tracker = cost_tracker or EmbeddingCostTracker()

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
            # No OpenRouter fallback - remote backend should retry remote only.
            # Validate remote on first use (lazy), but eagerly fetch hostname
            # for progress display source resolution.
            self._fetch_remote_hostname()
        elif backend == EmbeddingBackend.OPENROUTER:
            # Direct OpenRouter mode (no remote fallback)
            self._openrouter_client = OpenRouterEmbeddingClient(
                model_name=self.config.model_name or "Qwen/Qwen3-Embedding-0.6B"
            )
            if not self._openrouter_client.is_available():
                raise EmbeddingBackendError(
                    "OpenRouter backend selected but API key not configured. "
                    "Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable."
                )
        else:
            raise EmbeddingBackendError(f"Unknown backend: {backend}")

    def _fetch_remote_hostname(self) -> None:
        """Eagerly fetch remote server hostname for source display.

        Non-blocking best-effort: if the server isn't up yet, hostname
        will be resolved lazily during _validate_remote_backend().
        """
        try:
            detailed = self._remote_client.get_detailed_info()
            if detailed:
                self._remote_hostname = detailed.get("server", {}).get("hostname")
        except Exception:
            pass  # Will be resolved on first embed call

    def _validate_remote_backend(self) -> None:
        """Validate remote backend is available and model matches.

        No fallback to OpenRouter - if remote is unavailable, raises error.
        The remote client has its own retry logic for transient failures.

        Retries the health check up to 3 times with backoff to handle
        transient timeouts when the server is under heavy load.

        If the server is unreachable, attempts SLURM auto-launch (submits
        a GPU job on the titan partition and waits for it to start).
        """
        if self._backend_validated:
            return

        if not self._remote_client:
            raise EmbeddingBackendError("Remote client not initialized")

        # Retry health check with increasing timeout under load
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
            # Server unreachable - try SLURM auto-launch
            if self._try_slurm_auto_launch():
                # Re-check health after SLURM launch
                if self._remote_client.is_available(timeout=15.0):
                    self.logger.info("Server available after SLURM auto-launch")
                else:
                    raise EmbeddingBackendError(
                        "SLURM job started but server not reachable. "
                        "Check: imas-codex serve embed slurm status"
                    )
            else:
                raise EmbeddingBackendError(
                    f"Remote embedding server not available at {self.config.remote_url}. "
                    f"{last_error}. "
                    "Ensure SSH tunnel is active: ssh -f -N -L 18765:127.0.0.1:18765 iter\n"
                    "Or submit a SLURM job: imas-codex serve embed slurm submit"
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

        # Get hostname from /info for detailed source tracking
        detailed = self._remote_client.get_detailed_info()
        if detailed:
            self._remote_hostname = detailed.get("server", {}).get("hostname")

        self.logger.info(
            "Using remote embedder: %s on %s (host=%s)",
            info.model,
            info.device,
            self._remote_hostname or "unknown",
        )
        self._backend_validated = True

    def _try_slurm_auto_launch(self) -> bool:
        """Try to auto-launch the embedding server via SLURM.

        Checks if SLURM (sbatch) is available — either locally (on ITER)
        or via SSH to the ITER login node.

        Returns:
            True if SLURM launch succeeded and server is available
        """
        import shutil
        import subprocess

        # Check 1: sbatch available locally (we're on ITER)?
        if not shutil.which("sbatch"):
            # Check 2: sbatch available via SSH to ITER?
            try:
                result = subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "ConnectTimeout=3",
                        "-o",
                        "BatchMode=yes",
                        "iter",
                        "which sbatch",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    self.logger.debug("SLURM not available locally or via SSH")
                    return False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.logger.debug("Cannot reach ITER via SSH for SLURM auto-launch")
                return False

        self.logger.info(
            "Remote embed server unreachable, attempting SLURM auto-launch..."
        )

        try:
            from imas_codex.embeddings.slurm import ensure_server

            return ensure_server()
        except Exception as e:
            self.logger.warning("SLURM auto-launch failed: %s", e)
            return False

    def _try_openrouter_fallback(self) -> bool:
        """Attempt to use OpenRouter as fallback for remote backend.

        Deprecated: No longer used for REMOTE backend.
        Only returns True for explicitly configured OPENROUTER backend.

        Returns:
            True if fallback is available and configured
        """
        if not self._openrouter_client:
            return False

        if not self._openrouter_client.is_available():
            return False

        # Warn once on first fallback
        if not self._fallback_warned:
            self.logger.warning(
                "Remote embedding server unavailable. "
                f"Falling back to OpenRouter API ({self._openrouter_client.model_name}). "
                "Costs will be tracked. Will retry remote every 60s."
            )
            self._fallback_warned = True

        self._using_fallback = True
        self._backend_validated = True
        return True

    def _embed_via_openrouter(self, texts: list[str]) -> np.ndarray:
        """Embed texts using OpenRouter fallback with cost tracking.

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of embeddings

        Raises:
            EmbeddingBackendError: If OpenRouter unavailable or fails
        """
        if not self._openrouter_client:
            raise EmbeddingBackendError("OpenRouter client not initialized")

        result = self._openrouter_client.embed_with_cost(
            texts,
            normalize=self.config.normalize_embeddings,
            cost_tracker=self.cost_tracker,
        )
        return result.embeddings

    @property
    def is_using_fallback(self) -> bool:
        """Check if currently using OpenRouter fallback."""
        return self._using_fallback

    @property
    def current_source(self) -> str:
        """Get current embedding source identifier.

        Returns:
            'local', 'remote', 'openrouter', or host-specific
            like 'iter-login' or 'iter-titan'
        """
        if self._using_fallback:
            return "openrouter"
        backend = self.config.backend or EmbeddingBackend.LOCAL
        if backend == EmbeddingBackend.REMOTE:
            return self._resolve_remote_source()
        elif backend == EmbeddingBackend.OPENROUTER:
            return "openrouter"
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

    @property
    def cost_summary(self) -> str:
        """Get human-readable cost summary for OpenRouter usage."""
        return self.cost_tracker.summary()

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

        Uses the configured backend with transparent fallback for remote.

        Raises:
            EmbeddingBackendError: If all backends are unavailable.
        """
        backend = self.config.backend or EmbeddingBackend.LOCAL

        if backend == EmbeddingBackend.REMOTE:
            self._validate_remote_backend()

            try:
                return self._remote_client.embed(  # type: ignore[union-attr]
                    texts, normalize=self.config.normalize_embeddings
                )
            except (ConnectionError, RuntimeError) as e:
                # No fallback to OpenRouter - raise directly.
                # Remote client already has retry logic for transient failures.
                raise EmbeddingBackendError(
                    f"Remote embedding failed: {e}. Check SSH tunnel: "
                    "ssh -f -N -L 18765:127.0.0.1:18765 iter"
                ) from e

        elif backend == EmbeddingBackend.OPENROUTER:
            return self._embed_via_openrouter(texts)

        # Local backend
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
        """Embed ad-hoc texts and return result with source and cost tracking.

        Returns EmbeddingResult with:
        - source: "local", "remote", or "openrouter"
        - cost_usd: 0.0 for local/remote, actual cost for openrouter
        - total_tokens: 0 for local/remote, actual tokens for openrouter

        Uses the configured backend with transparent fallback for remote.

        Raises:
            EmbeddingBackendError: If all backends are unavailable.
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
            self._validate_remote_backend()

            try:
                embeddings = self._remote_client.embed(  # type: ignore[union-attr]
                    texts, normalize=self.config.normalize_embeddings
                )
                elapsed = time.time() - start
                return EmbeddingResult(
                    embeddings=embeddings,
                    total_tokens=0,
                    cost_usd=0.0,
                    model=self.config.model_name,
                    elapsed_seconds=elapsed,
                    source="remote",
                )
            except (ConnectionError, RuntimeError) as e:
                # No fallback to OpenRouter - raise directly.
                # Remote client already has retry logic for transient failures.
                raise EmbeddingBackendError(
                    f"Remote embedding failed: {e}. Check SSH tunnel: "
                    "ssh -f -N -L 18765:127.0.0.1:18765 iter"
                ) from e

        elif backend == EmbeddingBackend.OPENROUTER:
            return self._embed_via_openrouter_with_result(texts)

        # Local backend
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

    def _embed_via_openrouter_with_result(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts using OpenRouter with full result and cost tracking.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult with embeddings, actual cost, and source="openrouter"

        Raises:
            EmbeddingBackendError: If OpenRouter unavailable or fails
        """
        if not self._openrouter_client:
            raise EmbeddingBackendError("OpenRouter client not initialized")

        result = self._openrouter_client.embed_with_cost(
            texts,
            normalize=self.config.normalize_embeddings,
            cost_tracker=self.cost_tracker,
        )
        return result

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

    def _load_model(self) -> None:
        """Load local SentenceTransformer model."""
        ST = self._import_sentence_transformers()
        try:
            cache_folder = str(self._get_cache_directory() / "models")
            try:
                self.logger.debug("Loading cached sentence transformer model...")
                self._model = ST(
                    self.config.model_name,
                    device=self.config.device,
                    cache_folder=cache_folder,
                    local_files_only=True,
                )
                self.logger.debug(
                    f"Model {self.config.model_name} loaded from cache on device: {self._model.device}"
                )
            except Exception:
                self.logger.debug(
                    f"Model not in cache, downloading {self.config.model_name}..."
                )
                self._model = ST(
                    self.config.model_name,
                    device=self.config.device,
                    cache_folder=cache_folder,
                    local_files_only=False,
                )
                self.logger.debug(
                    f"Downloaded and loaded model {self.config.model_name} on device: {self._model.device}"
                )
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to load model {self.config.model_name}: {e}")
            fallback = "all-MiniLM-L6-v2"
            self.logger.debug(f"Trying fallback model: {fallback}")
            self._model = ST(fallback, device=self.config.device)
            self.config.model_name = fallback

    def _generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts using configured backend."""
        backend = self.config.backend or EmbeddingBackend.LOCAL

        if backend == EmbeddingBackend.REMOTE:
            self._validate_remote_backend()

            try:
                self.logger.debug(
                    f"Generating embeddings remotely for {len(texts)} texts..."
                )
                embeddings = self._remote_client.embed(  # type: ignore[union-attr]
                    texts, normalize=self.config.normalize_embeddings
                )
                if self.config.use_half_precision:
                    embeddings = embeddings.astype(np.float16)
                self.logger.debug(
                    f"Generated embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}"
                )
                return embeddings
            except (ConnectionError, RuntimeError) as e:
                # No fallback to OpenRouter - raise directly.
                # Remote client already has retry logic for transient failures.
                raise EmbeddingBackendError(
                    f"Remote embedding failed: {e}. Check SSH tunnel: "
                    "ssh -f -N -L 18765:127.0.0.1:18765 iter"
                ) from e

        elif backend == EmbeddingBackend.OPENROUTER:
            self.logger.debug(
                f"Generating embeddings via OpenRouter for {len(texts)} texts..."
            )
            embeddings = self._embed_via_openrouter(texts)
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
