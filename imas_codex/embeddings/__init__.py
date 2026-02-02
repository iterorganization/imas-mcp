"""Embedding management module for IMAS Codex.

Supports both local and remote embedding:
- Local: SentenceTransformer with optional GPU
- Remote: GPU server via HTTP (typically through SSH tunnel)
"""

from .cache import EmbeddingCache
from .client import RemoteEmbeddingClient, RemoteServerInfo, get_remote_client
from .config import EncoderConfig
from .embeddings import Embeddings
from .encoder import Encoder

__all__ = [
    "EmbeddingCache",
    "EncoderConfig",
    "Embeddings",
    "Encoder",
    "RemoteEmbeddingClient",
    "RemoteServerInfo",
    "get_remote_client",
]
