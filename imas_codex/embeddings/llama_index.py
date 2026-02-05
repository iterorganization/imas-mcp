"""LlamaIndex-compatible embedding wrapper that respects embedding-backend config.

This provides a LlamaIndex BaseEmbedding implementation that uses the Encoder
class internally, which handles:
- Backend selection (local, remote, openrouter)
- Transparent fallback from remote to OpenRouter when unavailable
- Single warning on first fallback, then silent operation
- Cost tracking for OpenRouter usage
- Source tracking via get_embedding_source() for progress display

Usage:
    from imas_codex.embeddings.llama_index import get_llama_embed_model, get_embedding_source

    embed_model = get_llama_embed_model()  # Respects embedding-backend config
    source = get_embedding_source()  # Returns "local", "remote", or "openrouter"
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
        Source identifier: "local", "remote", or "openrouter"
    """
    with _embedding_source_lock:
        return _embedding_source


def _set_embedding_source(source: str) -> None:
    """Set the current embedding source."""
    global _embedding_source
    with _embedding_source_lock:
        _embedding_source = source


class EncoderLlamaEmbedding(BaseEmbedding):
    """LlamaIndex embedding that uses Encoder internally.

    This wraps the Encoder class which has all the fallback logic:
    - For remote backend: tries remote server, falls back to OpenRouter
    - For local backend: uses SentenceTransformer
    - For openrouter backend: uses OpenRouter API directly

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

        logger.info(
            f"Initialized EncoderLlamaEmbedding: {model_name} "
            f"(backend={config.backend}, source={get_embedding_source()})"
        )

    def _update_source(self) -> None:
        """Update global embedding source based on encoder state."""
        _set_embedding_source(self._encoder.current_source)

    @classmethod
    def class_name(cls) -> str:
        return "EncoderLlamaEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query string."""
        result = self._encoder.embed_texts_with_result([query])
        _set_embedding_source(result.source)
        return result.embeddings[0].tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a text string."""
        result = self._encoder.embed_texts_with_result([text])
        _set_embedding_source(result.source)
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
        _set_embedding_source(result.source)
        return result.embeddings.tolist()

    @property
    def is_using_fallback(self) -> bool:
        """Check if currently using OpenRouter fallback."""
        return self._encoder.is_using_fallback

    @property
    def cost_summary(self) -> str:
        """Get cost summary for OpenRouter usage."""
        return self._encoder.cost_summary


# Backwards compatibility alias
RemoteLlamaEmbedding = EncoderLlamaEmbedding


def get_llama_embed_model() -> BaseEmbedding:
    """Get LlamaIndex embedding model respecting embedding-backend config.

    Returns:
        BaseEmbedding: EncoderLlamaEmbedding that handles all backend types
        with automatic fallback for remote backend.

    The returned model uses the Encoder class internally which provides:
    - Transparent fallback from remote to OpenRouter when unavailable
    - Cost tracking for OpenRouter usage
    - Source tracking for progress display
    """
    backend_str = get_embedding_backend()
    model_name = get_imas_embedding_model()

    try:
        backend = EmbeddingBackend(backend_str)
    except ValueError:
        backend = EmbeddingBackend.LOCAL

    logger.info(f"Creating LlamaIndex embed model: {model_name} (backend={backend})")

    return EncoderLlamaEmbedding(
        model_name=model_name,
        backend=backend,
    )


__all__ = [
    "EncoderLlamaEmbedding",
    "RemoteLlamaEmbedding",  # Backwards compatibility
    "get_embedding_source",
    "get_llama_embed_model",
]
