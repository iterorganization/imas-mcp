"""Embedding management module for IMAS Codex.

Supports embedding backends:
- local: SentenceTransformer with optional GPU
- remote: GPU server via HTTP (iter cluster through SSH tunnel)

No silent fallback: if the configured backend is unavailable, an error
is raised immediately.  Source tracking via get_embedding_source() for
progress display.
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
    EmbeddingResult,
)
from .readiness import ensure_embedding_ready

__all__ = [
    "EmbeddingBackend",
    "EmbeddingBackendError",
    "EmbeddingCache",
    "EmbeddingResult",
    "EncoderConfig",
    "EncoderEmbedding",
    "Embeddings",
    "Encoder",
    "RemoteEmbeddingClient",
    "RemoteServerInfo",
    "ensure_embedding_ready",
    "get_embed_model",
    "get_embedding_source",
    "get_remote_client",
]
