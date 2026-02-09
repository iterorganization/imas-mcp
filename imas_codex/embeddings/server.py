"""FastAPI embedding server for GPU-accelerated remote embedding.

This server runs on a GPU-equipped machine (e.g., ITER cluster) and provides
an HTTP API for embedding text. Clients connect via SSH tunnel.

Start server:
    imas-codex serve embed start --host 127.0.0.1 --port 18765

With idle timeout (auto-shutdown after 30 minutes of inactivity):
    imas-codex serve embed start --idle-timeout 1800

Multi-GPU (automatic when multiple GPUs visible):
    CUDA_VISIBLE_DEVICES=0,1,2,3 imas-codex serve embed start

Client access (via SSH tunnel):
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
_multi_process_pool = None  # SentenceTransformer multi-GPU pool
_gpu_count: int = 0  # Number of GPUs in use
_startup_time: float = 0
_last_request_time: float = 0
_idle_timeout: int = 0  # 0 = disabled
_shutdown_event: asyncio.Event | None = None


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
    gpu_count: int = Field(0, description="Number of GPUs in use")
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


def _get_cuda_device_count() -> int:
    """Get number of visible CUDA devices."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    global _encoder, _multi_process_pool, _gpu_count
    global _startup_time, _last_request_time, _shutdown_event

    logger.info("Loading embedding model...")
    start = time.time()

    # Import here to avoid circular imports
    from imas_codex.embeddings.config import EmbeddingBackend, EncoderConfig
    from imas_codex.embeddings.encoder import Encoder
    from imas_codex.settings import get_imas_embedding_model

    model_name = get_imas_embedding_model()
    device = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if device:
        logger.info(f"Using CUDA device(s): {device}")

    config = EncoderConfig(
        model_name=model_name,
        device="cuda" if _cuda_available() else "cpu",
        # CRITICAL: Server must always use LOCAL backend (its own GPU).
        # Without this, it reads pyproject.toml which may say "remote"
        # causing the server to try calling itself via HTTP.
        backend=EmbeddingBackend.LOCAL,
        normalize_embeddings=True,
        use_rich=False,
    )

    _encoder = Encoder(config=config)
    _startup_time = time.time()
    _last_request_time = time.time()
    _gpu_count = _get_cuda_device_count()

    load_time = time.time() - start
    model_device = _encoder.get_model().device
    logger.info(f"Model {model_name} loaded on {model_device} in {load_time:.1f}s")

    # Multi-GPU handling: device_map="auto" distributes the model across
    # GPUs automatically (for large models like 8B). In that case, the
    # model.device reports 'cpu' or the first device, but inference uses
    # all mapped GPUs. Don't try encode_multi_process which duplicates
    # the model (4 copies × 16GB = 64GB, won't fit on 4×16GB P100s).
    # Only use multi-process pool if the model fits on a single GPU.
    if _gpu_count > 1 and str(model_device).startswith("cuda"):
        try:
            model = _encoder.get_model()
            devices = [f"cuda:{i}" for i in range(_gpu_count)]
            _multi_process_pool = model.start_multi_process_pool(devices)
            logger.info("Multi-GPU pool started: %d GPUs %s", _gpu_count, devices)
        except Exception as e:
            logger.warning("Failed to start multi-GPU pool, using single GPU: %s", e)
            _multi_process_pool = None
            _gpu_count = 1
    else:
        if _gpu_count > 1:
            logger.info(
                "Model distributed across GPUs via device_map (gpu_count=%d)",
                _gpu_count,
            )
        else:
            logger.info("Single GPU mode (gpu_count=%d)", _gpu_count)

    # Start idle watchdog if timeout is configured
    _shutdown_event = asyncio.Event()
    watchdog_task = None
    if _idle_timeout > 0:
        logger.info(f"Idle timeout enabled: {_idle_timeout}s")
        watchdog_task = asyncio.create_task(_idle_watchdog())

    yield

    # Cancel watchdog on shutdown
    if watchdog_task and not watchdog_task.done():
        watchdog_task.cancel()

    # Stop multi-process pool
    if _multi_process_pool is not None:
        try:
            _encoder.get_model().stop_multi_process_pool(_multi_process_pool)
            logger.info("Multi-GPU pool stopped")
        except Exception as e:
            logger.warning("Error stopping multi-GPU pool: %s", e)

    logger.info("Shutting down embedding server")
    _encoder = None
    _multi_process_pool = None


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


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="IMAS Codex Embedding Server",
        description="GPU-accelerated embedding service for IMAS Codex",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        if _encoder is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        model = _encoder.get_model()
        gpu_name, gpu_memory = _get_gpu_info()

        return HealthResponse(
            status="healthy",
            model=_encoder.config.model_name,
            device=str(model.device),
            gpu_name=gpu_name,
            gpu_memory_mb=gpu_memory,
            gpu_count=_gpu_count,
            uptime_seconds=time.time() - _startup_time,
            idle_seconds=time.time() - _last_request_time,
            idle_timeout=_idle_timeout,
            hostname=os.uname().nodename,
        )

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: EmbedRequest) -> EmbedResponse:
        """Embed texts and return vectors."""
        global _last_request_time

        if _encoder is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        _last_request_time = time.time()
        start = time.time()

        try:
            # Update normalize setting if different from default
            original_normalize = _encoder.config.normalize_embeddings
            if request.normalize != original_normalize:
                _encoder.config.normalize_embeddings = request.normalize

            # Use multi-GPU pool when available for parallel encoding
            if _multi_process_pool is not None:
                import numpy as np

                model = _encoder.get_model()
                embeddings = model.encode_multi_process(
                    request.texts,
                    _multi_process_pool,
                    normalize_embeddings=request.normalize,
                    batch_size=_encoder.config.batch_size,
                )
                if not isinstance(embeddings, np.ndarray):
                    embeddings = np.array(embeddings)
            else:
                embeddings = _encoder.embed_texts(request.texts)

            # Restore original setting
            if request.normalize != original_normalize:
                _encoder.config.normalize_embeddings = original_normalize

            elapsed_ms = (time.time() - start) * 1000

            return EmbedResponse(
                embeddings=embeddings.tolist(),
                model=_encoder.config.model_name,
                dimension=embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
                count=len(request.texts),
                elapsed_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/info")
    async def info() -> dict[str, Any]:
        """Detailed server information."""
        if _encoder is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        gpu_name, gpu_memory = _get_gpu_info()
        model = _encoder.get_model()

        return {
            "model": {
                "name": _encoder.config.model_name,
                "device": str(model.device),
                "embedding_dimension": model.get_sentence_embedding_dimension(),
            },
            "gpu": {
                "name": gpu_name,
                "memory_mb": gpu_memory,
                "count": _gpu_count,
                "multi_process": _multi_process_pool is not None,
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
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "hostname": os.uname().nodename,
                "version": "1.0.0",
            },
        }

    return app


# Create app instance for uvicorn
app = create_app()
