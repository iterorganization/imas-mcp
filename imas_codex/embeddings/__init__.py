"""Embedding management module for IMAS Codex.

Supports multiple embedding backends:
- local: SentenceTransformer with optional GPU
- remote: GPU server via HTTP (iter cluster through SSH tunnel)
- openrouter: OpenRouter API for cloud embeddings
"""

from .cache import EmbeddingCache
from .client import RemoteEmbeddingClient, RemoteServerInfo, get_remote_client
from .config import EmbeddingBackend, EncoderConfig
from .embeddings import Embeddings
from .encoder import EmbeddingBackendError, Encoder
from .llama_index import RemoteLlamaEmbedding, get_llama_embed_model

__all__ = [
    "EmbeddingBackend",
    "EmbeddingBackendError",
    "EmbeddingCache",
    "EncoderConfig",
    "Embeddings",
    "Encoder",
    "RemoteEmbeddingClient",
    "RemoteLlamaEmbedding",
    "RemoteServerInfo",
    "get_llama_embed_model",
    "get_remote_client",
]
