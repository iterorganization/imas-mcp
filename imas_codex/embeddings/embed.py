"""LlamaIndex-compatible embedding wrapper that respects embedding-backend config.

This provides a LlamaIndex BaseEmbedding implementation that uses the Encoder
class internally, which handles:
- Backend selection (local or remote)
- No silent fallback: if the configured backend is unavailable, an error is raised
- Source tracking via get_embedding_source() for progress display

Usage:
    from imas_codex.embeddings import get_embed_model, get_embedding_source

    embed_model = get_embed_model()  # Respects embedding-backend config
    source = get_embedding_source()  # Returns "local" or "remote"
"""

import logging
import threading
from typing import TYPE_CHECKING, Any

from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

from imas_codex.settings import get_embedding_backend, get_imas_embedding_model

from .config import EmbeddingBackend, EncoderConfig
from .encoder import Encoder

if TYPE_CHECKING:
    pass

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


class EncoderEmbedding(BaseEmbedding):
    """LlamaIndex embedding that uses Encoder internally.

    No silent fallback: if the configured backend (local or remote) is
    unavailable, an error is raised immediately.

    The embedding source is tracked globally and can be queried via
    get_embedding_source() for progress display indicators.
    """

    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    _encoder: Encoder = PrivateAttr()

    def __init__(
        self,
        model_name: str | None = None,
        backend: EmbeddingBackend | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize embedding with Encoder backend.

        Args:
            model_name: Model name for embeddings
            backend: Explicit backend selection (None uses config)
            **kwargs: Additional BaseEmbedding arguments
        """
        model_name = model_name or get_imas_embedding_model()
        super().__init__(model_name=model_name, **kwargs)

        # Create encoder config
        config = EncoderConfig(
            model_name=model_name,
            backend=backend,
            normalize_embeddings=True,
            use_rich=False,  # No progress display in embedding wrapper
        )

        self._encoder = Encoder(config=config)

        # Update global source tracking
        self._update_source()

        logger.debug(
            f"Initialized EncoderEmbedding: {model_name} "
            f"(backend={config.backend}, source={get_embedding_source()})"
        )

    def _update_source(self) -> None:
        """Update global embedding source based on encoder state."""
        _set_embedding_source(self._encoder.current_source)

    @classmethod
    def class_name(cls) -> str:
        return "EncoderEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query string."""
        result = self._encoder.embed_texts_with_result([query])
        _set_embedding_source(self._encoder.current_source)
        return result.embeddings[0].tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a text string."""
        result = self._encoder.embed_texts_with_result([text])
        _set_embedding_source(self._encoder.current_source)
        return result.embeddings[0].tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async version - falls back to sync for simplicity."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async version - falls back to sync for simplicity."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        result = self._encoder.embed_texts_with_result(texts)
        _set_embedding_source(self._encoder.current_source)
        return result.embeddings.tolist()


# Module-level cache for embed model singleton
_cached_embed_model: BaseEmbedding | None = None


def get_embed_model(*, cached: bool = True) -> BaseEmbedding:
    """Get LlamaIndex embedding model respecting embedding-backend config.

    Args:
        cached: If True (default), return a cached singleton to avoid
            reloading the model for each ingestion call.

    Returns:
        BaseEmbedding: EncoderEmbedding that handles local and remote backends.
        No silent fallback — if the backend is unavailable, an error is raised.
    """
    global _cached_embed_model
    if cached and _cached_embed_model is not None:
        return _cached_embed_model

    backend_str = get_embedding_backend()
    model_name = get_imas_embedding_model()

    try:
        backend = EmbeddingBackend(backend_str)
    except ValueError:
        backend = EmbeddingBackend.LOCAL

    logger.debug(f"Creating embed model: {model_name} (backend={backend})")

    model = EncoderEmbedding(
        model_name=model_name,
        backend=backend,
    )

    if cached:
        _cached_embed_model = model

    return model


__all__ = [
    "EncoderEmbedding",
    "get_embed_model",
    "get_embedding_source",
]
