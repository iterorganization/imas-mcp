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
DEFAULT_TIMEOUT = 300.0
# Health check timeout (seconds)
HEALTH_TIMEOUT = 5.0
# Connection timeout (seconds)
CONNECT_TIMEOUT = 3.0
# Maximum texts per HTTP request (server processes in GPU sub-batches)
MAX_TEXTS_PER_REQUEST = 500


@dataclass
class RemoteServerInfo:
    """Information about the remote embedding server."""

    status: str
    model: str
    device: str
    gpu_name: str | None
    gpu_memory_mb: int | None
    uptime_seconds: float
    hostname: str | None = None


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
                    hostname=data.get("hostname"),
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
        max_retries: int = 3,
    ) -> np.ndarray:
        """Embed texts using remote server with retry logic.

        Automatically chunks large requests into sub-batches of
        ``MAX_TEXTS_PER_REQUEST`` texts to avoid server-side and
        client-side timeouts.

        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings
            max_retries: Maximum retry attempts for transient errors

        Returns:
            Numpy array of embeddings

        Raises:
            ConnectionError: If server is unavailable after retries
            RuntimeError: If embedding fails after retries
        """
        if not texts:
            return np.array([])

        # Chunk large batches into sub-requests
        if len(texts) > MAX_TEXTS_PER_REQUEST:
            return self._embed_chunked(texts, normalize, max_retries)

        return self._embed_single(texts, normalize, max_retries)

    def _embed_chunked(
        self,
        texts: list[str],
        normalize: bool,
        max_retries: int,
    ) -> np.ndarray:
        """Embed a large batch by splitting into chunked HTTP requests."""
        from imas_codex.core.progress_monitor import create_progress_monitor

        chunks = [
            texts[i : i + MAX_TEXTS_PER_REQUEST]
            for i in range(0, len(texts), MAX_TEXTS_PER_REQUEST)
        ]
        chunk_names = [
            f"{min((i + 1) * MAX_TEXTS_PER_REQUEST, len(texts))}/{len(texts)}"
            for i in range(len(chunks))
        ]
        logger.debug(
            "Chunking %d texts into %d requests of ≤%d texts",
            len(texts),
            len(chunks),
            MAX_TEXTS_PER_REQUEST,
        )
        progress = create_progress_monitor(
            logger=logger,
            item_names=chunk_names,
            description_template="Embedding: {item}",
        )
        progress.start_processing(chunk_names, "Embedding remotely")
        start = time.time()
        results = []
        for i, chunk in enumerate(chunks):
            progress.set_current_item(chunk_names[i])
            result = self._embed_single(chunk, normalize, max_retries)
            results.append(result)
            progress.update_progress(chunk_names[i])
        elapsed = time.time() - start
        progress.finish_processing()
        logger.debug(
            "Remote embedding: %d texts in %.1fs (%d chunks)",
            len(texts),
            elapsed,
            len(chunks),
        )
        return np.vstack(results)

    def _embed_single(
        self,
        texts: list[str],
        normalize: bool,
        max_retries: int,
    ) -> np.ndarray:
        """Embed a single batch of texts with retry logic."""
        client = self._get_client()
        start = time.time()
        last_error: Exception | None = None

        for attempt in range(max_retries):
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
                last_error = ConnectionError(f"Cannot connect to embedding server: {e}")
                # Retry on connection errors
                if attempt < max_retries - 1:
                    delay = 2**attempt
                    logger.debug(
                        f"Connection error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                raise last_error from e

            except httpx.TimeoutException as e:
                last_error = ConnectionError(f"Embedding request timed out: {e}")
                # Retry on timeout
                if attempt < max_retries - 1:
                    delay = 2**attempt
                    logger.debug(
                        f"Timeout (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                raise last_error from e

            except httpx.HTTPError as e:
                # Retry on server disconnection and transient HTTP errors
                error_msg = str(e).lower()
                is_transient = any(
                    x in error_msg
                    for x in ["disconnected", "reset", "broken", "503", "502", "429"]
                )
                last_error = RuntimeError(f"HTTP error during embedding: {e}")

                if is_transient and attempt < max_retries - 1:
                    delay = 2**attempt
                    logger.debug(
                        f"Transient error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                raise last_error from e

        # Should not reach here, but handle edge case
        raise last_error or RuntimeError("Embedding failed after retries")


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
