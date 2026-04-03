"""FastAPI embedding server for GPU-accelerated remote embedding.

Provides an HTTP API for embedding text using GPU acceleration.
Clients connect via localhost, direct network, or SSH tunnel.

Multi-worker mode::

    # 4 workers on GPUs 0-3 (Titan: 8x P100, use half)
    imas-codex embed start --gpu 0,1,2,3 --workers 4

    Each uvicorn worker process claims a unique GPU from the pool
    via an atomic file-based counter during startup.

Single-worker mode::

    imas-codex embed start --gpu 1

GPU Memory Protection:
    On shared nodes (login T4), the server caps its memory fraction
    to 0.6 of total VRAM, uses small batch sizes, and clears CUDA
    cache after each batch to coexist with other GPU consumers.
"""

import asyncio
import fcntl
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global encoder instance (loaded at startup)
_encoder = None
_startup_time: float = 0
_last_request_time: float = 0
_idle_timeout: int = 0  # 0 = disabled
_shutdown_event: asyncio.Event | None = None
_gpu_name: str | None = None  # Cached at startup
_gpu_memory_mb: int | None = None  # Cached at startup
_cached_device_info: str = "not loaded"  # Cached at startup
_cached_embedding_dim: int = 0  # Cached at startup (native model dimension)
_default_output_dim: int = 0  # Configured output dimension (Matryoshka truncation)
_encode_timeout: float = 300.0  # 5 minutes max per embed request
_location: str | None = None  # Deployment location label (e.g. "titan")
_worker_gpu: int | None = None  # GPU index claimed by this worker
_gpu_pool: list[int] = []  # Available GPU indices for multi-worker
_gpu_slot_fds: list[int] = []  # Open lock fds — prevent GC/release
_request_count: int = 0  # Total embed requests served
_total_texts: int = 0  # Total texts embedded
_total_elapsed_ms: float = 0.0  # Total encoding time in ms


def _worker_state_dir() -> str:
    """Return temp directory for worker state files, keyed to master PID."""
    master_pid = os.getppid()
    d = os.path.join(tempfile.gettempdir(), f"codex-embed-workers-{master_pid}")
    os.makedirs(d, exist_ok=True)
    return d


def _write_worker_state() -> None:
    """Write this worker's current state to a shared temp file."""
    try:
        import torch

        gpu_info: dict[str, Any] = {
            "name": _gpu_name,
            "memory_mb": _gpu_memory_mb,
        }
        if torch.cuda.is_available():
            gpu_info["memory_used_mb"] = int(
                torch.cuda.memory_allocated() / (1024 * 1024)
            )
            gpu_info["memory_reserved_mb"] = int(
                torch.cuda.memory_reserved() / (1024 * 1024)
            )
            free, total = torch.cuda.mem_get_info()
            gpu_info["memory_free_mb"] = int(free / (1024 * 1024))
            gpu_info["memory_total_mb"] = int(total / (1024 * 1024))
    except Exception:
        gpu_info = {"name": _gpu_name, "memory_mb": _gpu_memory_mb}

    state = {
        "worker_gpu": _worker_gpu,
        "worker_pid": os.getpid(),
        "gpu": gpu_info,
        "stats": {
            "request_count": _request_count,
            "total_texts": _total_texts,
            "total_elapsed_ms": _total_elapsed_ms,
            "avg_ms_per_request": (
                _total_elapsed_ms / _request_count if _request_count > 0 else 0
            ),
        },
        "uptime_seconds": time.time() - _startup_time if _startup_time else 0,
        "idle_seconds": time.time() - _last_request_time if _last_request_time else 0,
        "timestamp": time.time(),
    }
    path = os.path.join(_worker_state_dir(), f"gpu-{_worker_gpu}.json")
    tmp = path + f".{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _read_all_worker_states() -> list[dict[str, Any]]:
    """Read all worker state files and return as a sorted list."""
    state_dir = _worker_state_dir()
    workers = []
    try:
        for fname in os.listdir(state_dir):
            if fname.endswith(".json") and ".tmp" not in fname:
                path = os.path.join(state_dir, fname)
                try:
                    with open(path) as f:
                        state = json.load(f)
                    # Skip stale entries (>60s old = worker likely dead)
                    age = time.time() - state.get("timestamp", 0)
                    if age < 120:
                        state["state_age_seconds"] = round(age, 1)
                        workers.append(state)
                except (json.JSONDecodeError, OSError):
                    continue
    except FileNotFoundError:
        pass
    workers.sort(key=lambda w: w.get("worker_gpu", 999))
    return workers


# Server-side text length ceiling (characters).
# At ~4 chars/token, 40K chars ≈ 10K tokens.  Self-attention memory
# scales O(n²): 10K tokens needs ~11 GiB peak on Qwen3-0.6B, fitting
# comfortably in the 15.5 GiB available (95% of P100-16GB).  At 16K
# tokens it jumps to ~21 GiB, exceeding the GPU.
# The pipeline already chunks to 10K chars; this only fires for direct
# API callers bypassing the chunking pipeline.
_MAX_TEXT_CHARS = 40_000


class EmbedRequest(BaseModel):
    """Request body for embedding texts."""

    texts: list[str] = Field(..., description="List of texts to embed", min_length=1)
    normalize: bool = Field(True, description="Normalize embeddings to unit length")
    dimension: int | None = Field(
        None,
        description="Output dimension (Matryoshka truncation). "
        "If omitted, uses the server's configured default. "
        "Must be <= native model dimension.",
    )
    prompt_name: str | None = Field(
        None,
        description="Named prompt for instruction-aware models (e.g., 'query'). "
        "Applied only for local encoding.",
    )


class EmbedResponse(BaseModel):
    """Response containing embeddings."""

    embeddings: list[list[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="Model name used for embedding")
    dimension: int = Field(..., description="Embedding dimension")
    count: int = Field(..., description="Number of texts embedded")
    elapsed_ms: float = Field(..., description="Time taken in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Server status")
    model: str = Field(..., description="Loaded model name")
    device: str = Field(..., description="Compute device (cuda/cpu)")
    gpu_name: str | None = Field(None, description="GPU name if available")
    gpu_memory_mb: int | None = Field(None, description="GPU memory in MB")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    idle_seconds: float = Field(0, description="Seconds since last request")
    idle_timeout: int = Field(0, description="Auto-shutdown timeout (0=disabled)")
    hostname: str | None = Field(None, description="Server hostname")
    location: str | None = Field(None, description="Deployment location (e.g. titan)")
    worker_gpu: int | None = Field(None, description="GPU index used by this worker")


def _claim_gpu() -> int | None:
    """Claim a unique GPU index from the pool for this worker process.

    Uses per-slot lock files so that each GPU index can only be held by
    one live worker at a time.  When a worker dies its lock is released
    automatically by the OS (flock is process-scoped), making the slot
    available for a respawned replacement.

    Previous approach used a monotonic counter (idx++ mod pool_size)
    which broke when uvicorn respawned dead workers — the counter never
    reset, causing wrapping collisions where two workers loaded models
    on the same physical GPU → CUDA OOM → death spiral.

    Returns the claimed GPU index, or None if no pool is configured.
    """
    if not _gpu_pool:
        return None

    master_pid = os.getppid()
    my_pid = os.getpid()

    # Try each slot in the pool — first unlocked slot wins.
    for slot_idx, gpu_id in enumerate(_gpu_pool):
        lock_path = os.path.join(
            tempfile.gettempdir(),
            f"codex-embed-gpu-{master_pid}-slot-{slot_idx}.lock",
        )
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            # Non-blocking exclusive lock — skip if another worker holds it.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Write our PID for debugging (lock itself is the guard).
            os.ftruncate(fd, 0)
            os.write(fd, str(my_pid).encode())
            # Keep fd open — lock is released when this process exits.
            # Store on module global so GC doesn't close it.
            _gpu_slot_fds.append(fd)
            logger.info(
                "Worker PID %d claimed GPU %d (slot %d of %d)",
                my_pid,
                gpu_id,
                slot_idx,
                len(_gpu_pool),
            )
            return gpu_id
        except OSError:
            # LOCK_NB raises OSError if another process holds the lock.
            try:
                os.close(fd)
            except OSError:
                pass
            continue

    # All slots taken — should not happen with workers <= pool size.
    logger.error(
        "Worker PID %d: all %d GPU slots occupied, defaulting to pool[0]",
        my_pid,
        len(_gpu_pool),
    )
    return _gpu_pool[0]


def _get_gpu_info() -> tuple[str | None, int | None]:
    """Get GPU name and memory if available."""
    try:
        import torch

        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            name = torch.cuda.get_device_name(device_id)
            memory = torch.cuda.get_device_properties(device_id).total_memory // (
                1024 * 1024
            )
            return name, memory
    except Exception:
        pass
    return None, None


def _cuda_available() -> bool:
    """Check if CUDA is available.

    Catches both ImportError (torch not installed) and AttributeError
    (corrupted torch installation missing __init__.py / cuda submodule).

    Respects IMAS_CODEX_FORCE_CPU=1 to skip the CUDA driver check
    entirely — critical when the NVIDIA kernel module is deadlocked
    (D-state processes) and torch.cuda.is_available() would hang.
    """
    if os.environ.get("IMAS_CODEX_FORCE_CPU"):
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except (ImportError, AttributeError):
        return False


def _init_worker_gpu() -> None:
    """Claim a GPU and load the model with serialized CUDA initialization.

    Uses an exclusive file lock so only ONE worker at a time performs
    CUDA initialization.  This prevents the CUDA driver deadlock that
    occurs when multiple workers call torch.cuda.init() simultaneously,
    which leaves processes in uninterruptible D state and causes the
    node to enter SLURM "draining" with "Kill task failed".

    The lock is held during: GPU claim → torch import → CUDA context
    creation → model download/load → first inference warmup.  Released
    once the model is fully on the GPU and ready to serve.

    Runs in a background thread (via asyncio.to_thread) so the async
    event loop isn't blocked — uvicorn can still respond to the master
    process's health probes while waiting for the lock.
    """
    global _encoder, _worker_gpu
    global _gpu_name, _gpu_memory_mb, _cached_device_info, _cached_embedding_dim
    global _default_output_dim

    # Ensure Python logging works in worker subprocesses.
    # Uvicorn configures its own loggers but not the root logger,
    # so our logger.info() etc. would go nowhere without this.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:     %(message)s",
        force=True,
    )

    master_pid = os.getppid()
    lock_path = os.path.join(
        tempfile.gettempdir(),
        f"codex-embed-startup-{master_pid}.lock",
    )
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)

    try:
        logger.info("Worker PID %d: waiting for CUDA init lock...", os.getpid())
        # Blocking exclusive lock — only one worker initializes at a time.
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        logger.info("Worker PID %d: acquired CUDA init lock", os.getpid())

        # Claim a GPU slot
        if _gpu_pool:
            gpu_id = _claim_gpu()
            if gpu_id is not None:
                _worker_gpu = gpu_id
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Set BEFORE importing torch
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        logger.info("Loading embedding model...")
        start = time.time()

        from imas_codex.embeddings.config import EmbeddingBackend, EncoderConfig
        from imas_codex.embeddings.encoder import Encoder
        from imas_codex.settings import get_embedding_model

        model_name = get_embedding_model()
        device = "cuda" if _cuda_available() else "cpu"

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible:
            logger.info("CUDA_VISIBLE_DEVICES=%s", cuda_visible)

        # GPU memory protection.
        is_dedicated = bool(os.environ.get("SLURM_JOB_ID"))
        default_fraction = "0.95" if is_dedicated else "0.6"

        if device == "cuda":
            try:
                import torch

                mem_fraction = float(
                    os.environ.get("IMAS_CODEX_GPU_MEMORY_FRACTION", default_fraction)
                )
                torch.cuda.set_per_process_memory_fraction(mem_fraction)
                free_mb, total_mb = torch.cuda.mem_get_info(0)
                logger.info(
                    "GPU memory cap: %.0f%% of %.0f MiB = %.0f MiB "
                    "(free: %.0f MiB, others using: %.0f MiB, dedicated=%s)",
                    mem_fraction * 100,
                    total_mb / 1024 / 1024,
                    mem_fraction * total_mb / 1024 / 1024,
                    free_mb / 1024 / 1024,
                    (total_mb - free_mb) / 1024 / 1024,
                    is_dedicated,
                )
            except Exception as e:
                logger.warning("Failed to set GPU memory cap: %s", e)

        config = EncoderConfig(
            model_name=model_name,
            device=device,
            backend=EmbeddingBackend.LOCAL,
            normalize_embeddings=True,
            use_rich=False,
            batch_size=32,
        )

        _encoder = Encoder(config=config)

        # Cache GPU info
        _gpu_name, _gpu_memory_mb = _get_gpu_info()
        _cached_device_info = _encoder.device_info
        try:
            model = _encoder.get_model()
            # Store native dimension before removing truncation.
            # Matryoshka models always produce native-dim vectors internally;
            # truncate_dim only slices the output.  By removing it here and
            # truncating in the /embed handler, we support per-request
            # dimension selection (e.g. benchmarking 256 vs 512 vs 1024).
            _cached_embedding_dim = model.get_sentence_embedding_dimension() or 0
            from imas_codex.settings import get_embedding_dimension

            _default_output_dim = get_embedding_dimension()
            model.truncate_dim = None
            native_dim = model.get_sentence_embedding_dimension() or 0
            logger.info(
                "Matryoshka: native_dim=%d, default_output_dim=%d",
                native_dim,
                _default_output_dim,
            )
            _cached_embedding_dim = native_dim
        except Exception:
            _cached_embedding_dim = 0
            _default_output_dim = 0

        load_time = time.time() - start
        logger.info(
            "Model %s loaded in %.1fs (device=%s, gpu=%s)",
            model_name,
            load_time,
            device,
            _gpu_name or "none",
        )

    finally:
        # Release the lock so the next worker can initialize.
        # Small grace period for the CUDA driver to settle.
        time.sleep(2)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        logger.info(
            "Worker PID %d GPU %s: released CUDA init lock",
            os.getpid(),
            _worker_gpu,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    global _startup_time, _last_request_time, _shutdown_event

    # Multi-worker GPU claim: each worker process picks a unique GPU
    # from the pool configured via IMAS_CODEX_GPU_POOL env var.
    pool_env = os.environ.get("IMAS_CODEX_GPU_POOL", "")
    if pool_env and not _gpu_pool:
        _gpu_pool.extend(int(g) for g in pool_env.split(",") if g.strip())

    # Run CUDA initialization in a thread so the event loop isn't blocked.
    # This lets uvicorn's master process communicate with the worker
    # (health probes, signal handling) while waiting for the startup lock.
    await asyncio.to_thread(_init_worker_gpu)

    _startup_time = time.time()
    _last_request_time = time.time()

    # Write initial worker state for /workers endpoint
    _write_worker_state()

    # Start idle watchdog if timeout is configured
    _shutdown_event = asyncio.Event()
    watchdog_task = None
    if _idle_timeout > 0:
        logger.info("Idle timeout enabled: %ds", _idle_timeout)
        watchdog_task = asyncio.create_task(_idle_watchdog())

    yield

    # Cancel watchdog on shutdown
    if watchdog_task and not watchdog_task.done():
        watchdog_task.cancel()

    logger.info("Shutting down embedding server")
    _encoder = None


async def _idle_watchdog() -> None:
    """Background task that shuts down the server after idle timeout."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        idle_seconds = time.time() - _last_request_time
        if idle_seconds >= _idle_timeout:
            logger.info(
                "Server idle for %.0fs (timeout=%ds), shutting down...",
                idle_seconds,
                _idle_timeout,
            )
            # Signal uvicorn to shut down
            os.kill(os.getpid(), 15)  # SIGTERM
            return


def _encode_batch_safe(
    model: Any,
    texts: list[str],
    normalize: bool,
    batch_size: int,
    prompt_name: str | None = None,
) -> Any:
    """Encode a batch with CUDA OOM recovery.

    On OOM, clears GPU cache and retries with halved batch size.
    If batch_size reaches 1 and still OOMs, encodes texts one at a
    time, skipping any individual text that triggers OOM (returns a
    zero vector for that text).

    This prevents worker crashloops where uvicorn respawns a worker
    that immediately hits the same OOM-causing request pattern.
    """
    import numpy as np
    import torch

    extra_kwargs: dict = {}
    if prompt_name is not None:
        extra_kwargs["prompt_name"] = prompt_name

    while batch_size >= 1:
        try:
            if len(texts) <= batch_size:
                result = model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    **extra_kwargs,
                )
                return result

            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings = model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    **extra_kwargs,
                )
                results.append(embeddings)
                _release_gpu_cache()

            return np.vstack(results)

        except torch.cuda.OutOfMemoryError:
            _release_gpu_cache()
            new_batch_size = max(1, batch_size // 2)
            if new_batch_size == batch_size:
                # Already at batch_size=1, fall through to one-at-a-time
                break
            logger.warning(
                "CUDA OOM at batch_size=%d for %d texts (max_len=%d chars), "
                "retrying with batch_size=%d",
                batch_size,
                len(texts),
                max(len(t) for t in texts) if texts else 0,
                new_batch_size,
            )
            batch_size = new_batch_size

    # Last resort: encode one at a time, substituting zero vectors for
    # texts that still OOM (e.g. extremely long individual texts).
    logger.warning("Falling back to one-at-a-time encoding for %d texts", len(texts))
    dim = _cached_embedding_dim or 1024
    results = []
    for _j, text in enumerate(texts):
        try:
            emb = model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                batch_size=1,
                show_progress_bar=False,
                **extra_kwargs,
            )
            results.append(emb)
        except torch.cuda.OutOfMemoryError:
            _release_gpu_cache()
            logger.error(
                "CUDA OOM on single text (%d chars), returning zero vector",
                len(text),
            )
            results.append(np.zeros((1, dim), dtype=np.float32))
        _release_gpu_cache()

    return np.vstack(results)


def _encode_texts_sync(
    texts: list[str],
    normalize: bool,
    batch_size: int,
    prompt_name: str | None = None,
) -> Any:
    """Synchronous encoding helper (runs in thread pool via asyncio.to_thread).

    Encodes in sub-batches and releases CUDA cache between batches so
    freed GPU memory is returned to the driver for other processes
    (especially NX desktop sessions on the shared login node T4).
    """
    # Truncate oversized texts to prevent O(n²) attention OOM.
    # The pipeline already chunks to 10K chars; this catches direct API calls.
    texts = [t[:_MAX_TEXT_CHARS] if len(t) > _MAX_TEXT_CHARS else t for t in texts]

    # Adaptive batch sizing: self-attention memory scales O(batch × seq²).
    # With 10K-token texts (40K chars), batch=32 needs ~33 GiB — well over
    # P100-16GB.  Scale batch_size down for long texts to stay within VRAM.
    max_len = max(len(t) for t in texts) if texts else 0
    if max_len > 20_000:
        # Very long texts (>5K tokens): process one at a time
        batch_size = 1
    elif max_len > 8_000:
        # Long texts (>2K tokens): small batches
        batch_size = min(batch_size, 4)
    elif max_len > 2_000:
        # Medium texts: moderate batches
        batch_size = min(batch_size, 16)

    model = _encoder.get_model()

    try:
        return _encode_batch_safe(model, texts, normalize, batch_size, prompt_name)
    finally:
        # Always release GPU cache — critical on error paths
        # where partial allocations remain in PyTorch's caching allocator.
        _release_gpu_cache()


def _release_gpu_cache() -> None:
    """Release unused CUDA cache back to the driver.

    On a shared GPU, this is critical — PyTorch's caching allocator
    holds freed blocks in a pool by default.  Without explicit release,
    other processes (NX sessions) see no free memory even though our
    model isn't actively using it.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="IMAS Codex Embedding Server",
        description="GPU-accelerated embedding service for IMAS Codex",
        version="2.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint.

        Uses GPU info cached at startup to avoid blocking CUDA calls
        in the async event loop.
        """
        if _encoder is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return HealthResponse(
            status="healthy",
            model=_encoder.config.model_name,
            device=_cached_device_info,
            gpu_name=_gpu_name,
            gpu_memory_mb=_gpu_memory_mb,
            uptime_seconds=time.time() - _startup_time,
            idle_seconds=time.time() - _last_request_time,
            idle_timeout=_idle_timeout,
            hostname=os.uname().nodename,
            location=_location,
            worker_gpu=_worker_gpu,
        )

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: EmbedRequest) -> EmbedResponse:
        """Embed texts and return vectors.

        Runs encoding in a thread pool to avoid blocking the async event
        loop (which would make /health unresponsive during encoding).

        Supports per-request Matryoshka dimension via ``dimension`` field.
        The model always encodes at native dimension; truncation and
        re-normalization happen after encoding.
        """
        global _last_request_time, _request_count, _total_texts, _total_elapsed_ms

        if _encoder is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        target_dim = request.dimension or _default_output_dim or _cached_embedding_dim
        if target_dim and _cached_embedding_dim and target_dim > _cached_embedding_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Requested dimension {target_dim} exceeds native "
                f"model dimension {_cached_embedding_dim}",
            )

        _last_request_time = time.time()
        start = time.time()

        try:
            embeddings = await asyncio.wait_for(
                asyncio.to_thread(
                    _encode_texts_sync,
                    request.texts,
                    request.normalize,
                    _encoder.config.batch_size,
                    request.prompt_name,
                ),
                timeout=_encode_timeout,
            )

            # Matryoshka truncation: slice to requested dimension and
            # re-normalize so the truncated vectors remain unit-length.
            if target_dim and embeddings.ndim == 2 and embeddings.shape[1] > target_dim:
                import numpy as np

                embeddings = embeddings[:, :target_dim]
                if request.normalize:
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    embeddings = embeddings / norms

            elapsed_ms = (time.time() - start) * 1000
            _request_count += 1
            _total_texts += len(request.texts)
            _total_elapsed_ms += elapsed_ms

            # Update worker state file for /workers endpoint
            _write_worker_state()

            return EmbedResponse(
                embeddings=embeddings.tolist(),
                model=_encoder.config.model_name,
                dimension=embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
                count=len(request.texts),
                elapsed_ms=elapsed_ms,
            )

        except TimeoutError:
            logger.error(
                "Encoding timed out after %.0fs for %d texts",
                _encode_timeout,
                len(request.texts),
            )
            raise HTTPException(
                status_code=504,
                detail=f"Encoding timed out after {_encode_timeout:.0f}s",
            ) from None
        except Exception as e:
            logger.error("Embedding error: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/info")
    async def info() -> dict[str, Any]:
        """Detailed server information (uses cached GPU data)."""
        if _encoder is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        gpu_info: dict[str, Any] = {
            "name": _gpu_name,
            "memory_mb": _gpu_memory_mb,
            "cuda_available": _cuda_available(),
        }
        # Add live GPU memory usage
        try:
            import torch

            if torch.cuda.is_available():
                gpu_info["memory_used_mb"] = int(
                    torch.cuda.memory_allocated() / (1024 * 1024)
                )
                gpu_info["memory_reserved_mb"] = int(
                    torch.cuda.memory_reserved() / (1024 * 1024)
                )
                free, total = torch.cuda.mem_get_info()
                gpu_info["memory_free_mb"] = int(free / (1024 * 1024))
                gpu_info["memory_total_mb"] = int(total / (1024 * 1024))
        except Exception:
            pass

        return {
            "model": {
                "name": _encoder.config.model_name,
                "device": _cached_device_info,
                "native_dimension": _cached_embedding_dim,
                "default_output_dimension": _default_output_dim,
                "embedding_dimension": _default_output_dim or _cached_embedding_dim,
            },
            "gpu": gpu_info,
            "config": {
                "normalize_embeddings": _encoder.config.normalize_embeddings,
                "batch_size": _encoder.config.batch_size,
            },
            "server": {
                "uptime_seconds": time.time() - _startup_time,
                "idle_seconds": time.time() - _last_request_time,
                "idle_timeout": _idle_timeout,
                "hostname": os.uname().nodename,
                "location": _location,
                "worker_gpu": _worker_gpu,
                "worker_pid": os.getpid(),
                "version": "2.0.0",
            },
            "stats": {
                "request_count": _request_count,
                "total_texts": _total_texts,
                "total_elapsed_ms": _total_elapsed_ms,
                "avg_ms_per_request": (
                    _total_elapsed_ms / _request_count if _request_count > 0 else 0
                ),
            },
        }

    @app.get("/workers")
    async def workers() -> dict[str, Any]:
        """Aggregated status of all worker processes.

        Each worker writes its GPU state to a temp file after startup
        and after each /embed request.  This endpoint reads all state
        files and returns them sorted by GPU index.
        """
        all_workers = _read_all_worker_states()
        return {
            "worker_count": len(all_workers),
            "gpu_pool": _gpu_pool,
            "workers": all_workers,
        }

    return app


# Create app instance for uvicorn
app = create_app()
