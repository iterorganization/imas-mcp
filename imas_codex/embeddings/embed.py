"""Embedding model access via Encoder singleton.

Provides a cached Encoder instance that handles:
- Backend selection (local or remote)
- No silent fallback: if the configured backend is unavailable, an error is raised
- Source tracking via get_embedding_source() for progress display

Usage:
    from imas_codex.embeddings import get_encoder, get_embedding_source

    encoder = get_encoder()  # Respects embedding-backend config
    embeddings = encoder.embed_texts(["some text"])
    source = get_embedding_source()  # Returns "local" or "remote"
"""

import logging
import threading

from imas_codex.settings import get_embedding_location, get_embedding_model

from .config import EmbeddingBackend, EncoderConfig
from .encoder import Encoder

logger = logging.getLogger(__name__)


# Global embedding source tracking for live indicator
_embedding_source: str = "unknown"
_embedding_source_lock = threading.Lock()


def get_embedding_source() -> str:
    """Get the current embedding source for progress display.

    Returns:
        Source identifier: "local" or "remote"
    """
    with _embedding_source_lock:
        return _embedding_source


def _set_embedding_source(source: str) -> None:
    """Set the current embedding source."""
    global _embedding_source
    with _embedding_source_lock:
        _embedding_source = source


# Module-level cache for encoder singleton
_cached_encoder: Encoder | None = None


def get_encoder(*, cached: bool = True) -> Encoder:
    """Get Encoder instance respecting embedding-backend config.

    Args:
        cached: If True (default), return a cached singleton to avoid
            reloading the model for each call.

    Returns:
        Encoder instance configured for the active backend.
        No silent fallback — if the backend is unavailable, an error is raised.
    """
    global _cached_encoder
    if cached and _cached_encoder is not None:
        return _cached_encoder

    backend_str = get_embedding_location()
    model_name = get_embedding_model()

    if backend_str == "local":
        backend = EmbeddingBackend.LOCAL
    else:
        backend = EmbeddingBackend.REMOTE

    logger.debug("Creating encoder: %s (backend=%s)", model_name, backend)

    config = EncoderConfig(
        model_name=model_name,
        backend=backend,
        normalize_embeddings=True,
        use_rich=False,
    )

    encoder = Encoder(config=config)
    _set_embedding_source(encoder.current_source)

    if cached:
        _cached_encoder = encoder

    return encoder


# Backward-compatible aliases
def get_embed_model(*, cached: bool = True) -> Encoder:
    """Backward-compatible alias for get_encoder()."""
    return get_encoder(cached=cached)


EncoderEmbedding = Encoder  # Backward-compatible alias


__all__ = [
    "Encoder",
    "EncoderEmbedding",
    "get_embed_model",
    "get_embedding_source",
    "get_encoder",
]
