"""Remote embedding client for connecting to GPU embedding server.

This client connects to a remote embedding server (typically on ITER cluster)
via SSH tunnel and provides a transparent interface for embedding texts.

Usage:
    client = RemoteEmbeddingClient("http://localhost:18765")
    if client.is_available():
        embeddings = client.embed(["text1", "text2"])
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# Default timeout for embedding requests (seconds)
DEFAULT_TIMEOUT = 120.0
# Health check timeout (seconds)
HEALTH_TIMEOUT = 5.0
# Connection timeout (seconds)
CONNECT_TIMEOUT = 3.0


@dataclass
class RemoteServerInfo:
    """Information about the remote embedding server."""

    status: str
    model: str
    device: str
    gpu_name: str | None
    gpu_memory_mb: int | None
    uptime_seconds: float


class RemoteEmbeddingClient:
    """Client for remote embedding server.

    Connects via HTTP to an embedding server running on a GPU machine.
    Typically accessed through SSH tunnel.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:18765",
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize client.

        Args:
            base_url: Base URL of the embedding server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=CONNECT_TIMEOUT,
                    read=self.timeout,
                    write=self.timeout,
                    pool=self.timeout,
                ),
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "RemoteEmbeddingClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def is_available(self, timeout: float = HEALTH_TIMEOUT) -> bool:
        """Check if remote server is available.

        Args:
            timeout: Timeout for health check

        Returns:
            True if server is healthy and responding
        """
        try:
            with httpx.Client(
                timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT)
            ) as client:
                response = client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("status") == "healthy"
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
            logger.debug(f"Remote embedder not available: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error checking remote embedder: {e}")
        return False

    def get_info(self) -> RemoteServerInfo | None:
        """Get server information.

        Returns:
            Server info or None if unavailable
        """
        try:
            client = self._get_client()
            response = client.get("/health")
            if response.status_code == 200:
                data = response.json()
                return RemoteServerInfo(
                    status=data.get("status", "unknown"),
                    model=data.get("model", "unknown"),
                    device=data.get("device", "unknown"),
                    gpu_name=data.get("gpu_name"),
                    gpu_memory_mb=data.get("gpu_memory_mb"),
                    uptime_seconds=data.get("uptime_seconds", 0),
                )
        except Exception as e:
            logger.debug(f"Failed to get server info: {e}")
        return None

    def get_detailed_info(self) -> dict[str, Any] | None:
        """Get detailed server information.

        Returns:
            Detailed info dict or None if unavailable
        """
        try:
            client = self._get_client()
            response = client.get("/info")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug(f"Failed to get detailed info: {e}")
        return None

    def embed(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed texts using remote server.

        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings

        Raises:
            ConnectionError: If server is unavailable
            RuntimeError: If embedding fails
        """
        if not texts:
            return np.array([])

        client = self._get_client()
        start = time.time()

        try:
            response = client.post(
                "/embed",
                json={"texts": texts, "normalize": normalize},
            )

            if response.status_code != 200:
                error_detail = response.json().get("detail", response.text)
                raise RuntimeError(f"Embedding failed: {error_detail}")

            data = response.json()
            embeddings = np.array(data["embeddings"], dtype=np.float32)

            elapsed = time.time() - start
            logger.debug(
                f"Remote embedding: {len(texts)} texts in {elapsed:.2f}s "
                f"(server: {data.get('elapsed_ms', 0):.0f}ms)"
            )

            return embeddings

        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to embedding server: {e}") from e
        except httpx.TimeoutException as e:
            raise ConnectionError(f"Embedding request timed out: {e}") from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"HTTP error during embedding: {e}") from e


def get_remote_client(url: str | None = None) -> RemoteEmbeddingClient | None:
    """Get a remote embedding client if URL is configured.

    Args:
        url: Optional explicit URL, otherwise uses settings

    Returns:
        Client instance or None if not configured
    """
    if url is None:
        from imas_codex.settings import get_embed_remote_url

        url = get_embed_remote_url()

    if url:
        return RemoteEmbeddingClient(url)
    return None


__all__ = ["RemoteEmbeddingClient", "RemoteServerInfo", "get_remote_client"]
