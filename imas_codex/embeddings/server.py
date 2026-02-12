"""FastAPI embedding server for GPU-accelerated remote embedding.

Runs on the ITER login node using a T4 GPU and provides an HTTP API
for embedding text.  Clients connect via localhost (on ITER) or the
network (from SLURM compute nodes or workstation SSH tunnel).

GPU Memory Protection:
    The login node T4 GPUs are shared with NoMachine desktop sessions
    (gnome-shell, firefox, paraview, etc.) which consume unpredictable
    amounts of VRAM.  The server caps its memory fraction to 0.6 of
    total VRAM, uses small batch sizes, and clears CUDA cache after
    each batch to coexist safely with other GPU consumers.

Start server::

    imas-codex serve embed start --gpu 1

Client access (compute nodes or workstation)::

    # From SLURM compute node (server bound to 0.0.0.0)
    curl http://98dci4-srv-1001:18765/health
    # From workstation via SSH tunnel
    ssh -L 18765:127.0.0.1:18765 iter
    curl http://localhost:18765/health
"""

import asyncio
import logging
import os
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
_cached_embedding_dim: int = 0  # Cached at startup
_encode_timeout: float = 300.0  # 5 minutes max per embed request


class EmbedRequest(BaseModel):
    """Request body for embedding texts."""

    texts: list[str] = Field(..., description="List of texts to embed", min_length=1)
    normalize: bool = Field(True, description="Normalize embeddings to unit length")


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
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    global _encoder
    global _startup_time, _last_request_time, _shutdown_event
    global _gpu_name, _gpu_memory_mb, _cached_device_info, _cached_embedding_dim

    logger.info("Loading embedding model...")
    start = time.time()

    # Import here to avoid circular imports
    from imas_codex.embeddings.config import EmbeddingBackend, EncoderConfig
    from imas_codex.embeddings.encoder import Encoder
    from imas_codex.settings import get_embedding_model

    model_name = get_embedding_model()
    device = "cuda" if _cuda_available() else "cpu"

    # Log CUDA device selection
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible:
        logger.info("CUDA_VISIBLE_DEVICES=%s", cuda_visible)

    # GPU memory protection: cap our usage to coexist with NX desktops.
    # The T4 GPUs on the login node are shared with NoMachine sessions
    # that consume 1-11 GB of VRAM unpredictably.  Without capping,
    # PyTorch grabs all available memory and then OOMs when NX allocates
    # more, causing a cascade fallback to CPU that destroys the login node.
    if device == "cuda":
        try:
            import torch

            # Enable expandable segments to reduce fragmentation
            # (recommended by PyTorch's own OOM error message)
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            # Cap our process to 60% of GPU memory.  On a 15 GB T4 this
            # gives us ~9 GB which is plenty for Qwen3-0.6B (~1.2 GB model
            # + batch workspace), while leaving ~6 GB for NX sessions.
            mem_fraction = float(
                os.environ.get("IMAS_CODEX_GPU_MEMORY_FRACTION", "0.6")
            )
            torch.cuda.set_per_process_memory_fraction(mem_fraction)
            free_mb, total_mb = torch.cuda.mem_get_info(0)
            logger.info(
                "GPU memory cap: %.0f%% of %.0f MiB = %.0f MiB "
                "(free: %.0f MiB, others using: %.0f MiB)",
                mem_fraction * 100,
                total_mb / 1024 / 1024,
                mem_fraction * total_mb / 1024 / 1024,
                free_mb / 1024 / 1024,
                (total_mb - free_mb) / 1024 / 1024,
            )
        except Exception as e:
            logger.warning("Failed to set GPU memory cap: %s", e)

    config = EncoderConfig(
        model_name=model_name,
        device=device,
        # CRITICAL: Server must always use LOCAL backend (its own GPU).
        # Without this, it reads pyproject.toml which says "remote"
        # causing the server to try calling itself via HTTP.
        backend=EmbeddingBackend.LOCAL,
        normalize_embeddings=True,
        use_rich=False,
        # Conservative batch size to limit peak VRAM.  Smaller batches
        # reduce the memory high-water mark and allow empty_cache() to
        # release blocks between batches for NX sessions.
        batch_size=32,
    )

    _encoder = Encoder(config=config)
    _startup_time = time.time()
    _last_request_time = time.time()

    # Cache GPU info once at startup to avoid blocking CUDA calls
    # in request handlers (which would block the async event loop)
    _gpu_name, _gpu_memory_mb = _get_gpu_info()
    _cached_device_info = _encoder.device_info
    try:
        _cached_embedding_dim = (
            _encoder.get_model().get_sentence_embedding_dimension() or 0
        )
    except Exception:
        _cached_embedding_dim = 0

    load_time = time.time() - start
    logger.info(
        "Model %s loaded in %.1fs (device=%s, gpu=%s)",
        model_name,
        load_time,
        device,
        _gpu_name or "none",
    )

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


def _encode_texts_sync(
    texts: list[str],
    normalize: bool,
    batch_size: int,
) -> Any:
    """Synchronous encoding helper (runs in thread pool via asyncio.to_thread).

    Encodes in sub-batches and releases CUDA cache between batches so
    freed GPU memory is returned to the driver for other processes
    (especially NX desktop sessions on the shared login node T4).
    """
    import numpy as np

    model = _encoder.get_model()

    # For small requests, encode directly
    if len(texts) <= batch_size:
        result = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        _release_gpu_cache()
        return result

    # For larger requests, encode in sub-batches with cache release
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        results.append(embeddings)
        _release_gpu_cache()

    return np.vstack(results)


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
        )

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: EmbedRequest) -> EmbedResponse:
        """Embed texts and return vectors.

        Runs encoding in a thread pool to avoid blocking the async event
        loop (which would make /health unresponsive during encoding).
        """
        global _last_request_time

        if _encoder is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        _last_request_time = time.time()
        start = time.time()

        try:
            embeddings = await asyncio.wait_for(
                asyncio.to_thread(
                    _encode_texts_sync,
                    request.texts,
                    request.normalize,
                    _encoder.config.batch_size,
                ),
                timeout=_encode_timeout,
            )

            elapsed_ms = (time.time() - start) * 1000

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

        return {
            "model": {
                "name": _encoder.config.model_name,
                "device": _cached_device_info,
                "embedding_dimension": _cached_embedding_dim,
            },
            "gpu": {
                "name": _gpu_name,
                "memory_mb": _gpu_memory_mb,
                "cuda_available": _cuda_available(),
            },
            "config": {
                "normalize_embeddings": _encoder.config.normalize_embeddings,
                "batch_size": _encoder.config.batch_size,
            },
            "server": {
                "uptime_seconds": time.time() - _startup_time,
                "idle_seconds": time.time() - _last_request_time,
                "idle_timeout": _idle_timeout,
                "hostname": os.uname().nodename,
                "version": "2.0.0",
            },
        }

    return app


# Create app instance for uvicorn
app = create_app()
