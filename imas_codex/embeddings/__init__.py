"""Embedding management module for IMAS Codex.

Supports multiple embedding backends:
- local: SentenceTransformer with optional GPU
- remote: GPU server via HTTP (iter cluster through SSH tunnel)
- openrouter: OpenRouter API for cloud embeddings

Transparent fallback for remote backend:
- If remote is unavailable, automatically falls back to OpenRouter
- Single warning on first fallback, then silent operation
- Cost tracking via EmbeddingCostTracker
- Source tracking via get_embedding_source() for progress display
"""

from .cache import EmbeddingCache
from .client import RemoteEmbeddingClient, RemoteServerInfo, get_remote_client
from .config import EmbeddingBackend, EncoderConfig
from .embeddings import Embeddings
from .encoder import EmbeddingBackendError, Encoder
from .llama_index import (
    EncoderLlamaEmbedding,
    RemoteLlamaEmbedding,
    get_embedding_source,
    get_llama_embed_model,
)
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

__all__ = [
    "EmbeddingBackend",
    "EmbeddingBackendError",
    "EmbeddingBudgetExhaustedError",
    "EmbeddingCache",
    "EmbeddingCostTracker",
    "EmbeddingResult",
    "EncoderConfig",
    "EncoderLlamaEmbedding",
    "Embeddings",
    "Encoder",
    "OpenRouterEmbeddingClient",
    "OpenRouterEmbeddingError",
    "OpenRouterServerInfo",
    "RemoteEmbeddingClient",
    "RemoteLlamaEmbedding",
    "RemoteServerInfo",
    "calculate_embedding_cost",
    "estimate_embedding_cost",
    "get_embedding_source",
    "get_llama_embed_model",
    "get_openrouter_client",
    "get_remote_client",
]
