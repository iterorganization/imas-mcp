"""Embedding management module for IMAS Codex.

Supports multiple embedding backends:
- local: SentenceTransformer with optional GPU
- remote: GPU server via HTTP (iter cluster through SSH tunnel)
- openrouter: OpenRouter API for cloud embeddings

Fallback chain for remote backend:
  remote (login GPU) → local (CPU) → openrouter (cloud API)

Explicit local or openrouter backends have no fallback.
Cost tracking via EmbeddingCostTracker.
Source tracking via get_embedding_source() for progress display.
"""

from .cache import EmbeddingCache
from .client import RemoteEmbeddingClient, RemoteServerInfo, get_remote_client
from .config import EmbeddingBackend, EncoderConfig
from .embed import (
    EncoderEmbedding,
    get_embed_model,
    get_embedding_source,
)
from .embeddings import Embeddings
from .encoder import EmbeddingBackendError, Encoder
from .openrouter_embed import (
    EmbeddingBudgetExhaustedError,
    EmbeddingCostTracker,
    EmbeddingResult,
    OpenRouterEmbeddingClient,
    OpenRouterEmbeddingError,
    OpenRouterServerInfo,
    calculate_embedding_cost,
    estimate_embedding_cost,
    get_openrouter_client,
)
from .readiness import ensure_embedding_ready

__all__ = [
    "EmbeddingBackend",
    "EmbeddingBackendError",
    "EmbeddingBudgetExhaustedError",
    "EmbeddingCache",
    "EmbeddingCostTracker",
    "EmbeddingResult",
    "EncoderConfig",
    "EncoderEmbedding",
    "Embeddings",
    "Encoder",
    "OpenRouterEmbeddingClient",
    "OpenRouterEmbeddingError",
    "OpenRouterServerInfo",
    "RemoteEmbeddingClient",
    "RemoteServerInfo",
    "calculate_embedding_cost",
    "ensure_embedding_ready",
    "estimate_embedding_cost",
    "get_embed_model",
    "get_embedding_source",
    "get_openrouter_client",
    "get_remote_client",
]
