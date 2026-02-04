"""LlamaIndex-compatible embedding wrapper that respects embedding-backend config.

This provides a LlamaIndex BaseEmbedding implementation that routes to either
the local HuggingFaceEmbedding or the remote GPU embedding server based on the
embedding-backend configuration.

Usage:
    from imas_codex.embeddings.llama_index import get_llama_embed_model

    embed_model = get_llama_embed_model()  # Respects embedding-backend config
"""

import logging
from typing import TYPE_CHECKING, Any

from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

from imas_codex.settings import get_embedding_backend, get_imas_embedding_model

from .client import RemoteEmbeddingClient
from .config import EmbeddingBackend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RemoteLlamaEmbedding(BaseEmbedding):
    """LlamaIndex embedding that uses remote GPU server via HTTP.

    This is a drop-in replacement for HuggingFaceEmbedding that sends
    embedding requests to the remote GPU server instead of loading
    the model locally.
    """

    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    _client: RemoteEmbeddingClient = PrivateAttr()

    def __init__(
        self,
        model_name: str | None = None,
        remote_url: str = "http://localhost:18765",
        **kwargs: Any,
    ) -> None:
        """Initialize remote embedding.

        Args:
            model_name: Model name (must match server's model)
            remote_url: URL of the remote embedding server
            **kwargs: Additional BaseEmbedding arguments
        """
        model_name = model_name or get_imas_embedding_model()
        super().__init__(model_name=model_name, **kwargs)
        self._client = RemoteEmbeddingClient(remote_url)

        # Validate server is available and model matches
        if not self._client.is_available():
            raise ConnectionError(
                f"Remote embedding server not available at {remote_url}. "
                "Ensure SSH tunnel is active: ssh -L 18765:127.0.0.1:18765 iter"
            )

        info = self._client.get_info()
        if info and info.model != model_name:
            logger.warning(
                f"Remote server model ({info.model}) differs from expected ({model_name})"
            )

        logger.info(f"Using remote embedding: {model_name} via {remote_url}")

    @classmethod
    def class_name(cls) -> str:
        return "RemoteLlamaEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query string."""
        embeddings = self._client.embed([query], normalize=True)
        return embeddings[0].tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a text string."""
        embeddings = self._client.embed([text], normalize=True)
        return embeddings[0].tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async version - falls back to sync for simplicity."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async version - falls back to sync for simplicity."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        embeddings = self._client.embed(texts, normalize=True)
        return embeddings.tolist()


def get_llama_embed_model() -> BaseEmbedding:
    """Get LlamaIndex embedding model respecting embedding-backend config.

    Returns:
        BaseEmbedding: Either RemoteLlamaEmbedding or HuggingFaceEmbedding
        based on the embedding-backend setting in pyproject.toml.

    Raises:
        ConnectionError: If remote backend selected but server unavailable
    """
    backend_str = get_embedding_backend()

    try:
        backend = EmbeddingBackend(backend_str)
    except ValueError:
        backend = EmbeddingBackend.LOCAL

    model_name = get_imas_embedding_model()

    if backend == EmbeddingBackend.REMOTE:
        from imas_codex.settings import get_embed_remote_url

        remote_url = get_embed_remote_url()
        logger.info(f"Using remote embedding backend: {remote_url}")
        return RemoteLlamaEmbedding(model_name=model_name, remote_url=remote_url)

    elif backend == EmbeddingBackend.LOCAL:
        # Import HuggingFaceEmbedding only when local backend is used
        # This avoids downloading models on import when using remote backend
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        logger.info(f"Using local embedding backend: {model_name}")
        return HuggingFaceEmbedding(
            model_name=model_name,
            trust_remote_code=False,
        )

    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")


__all__ = ["RemoteLlamaEmbedding", "get_llama_embed_model"]
