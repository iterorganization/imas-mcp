"""OpenRouter API client for embeddings (OpenAI-compatible interface).

This module provides an OpenRouterClient class that implements the same
interface as SentenceTransformer for embedding generation using OpenRouter's
OpenAI-compatible API.
"""

import logging
import os
import time

import numpy as np
import requests

logger = logging.getLogger(__name__)


class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""

    pass


class OpenRouterClient:
    """OpenRouter API client for generating embeddings.

    This client provides a SentenceTransformer-compatible interface for
    generating embeddings using OpenRouter's OpenAI-compatible API.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 250,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize OpenRouter client.

        Args:
            model_name: Name of the embedding model to use
            api_key: OpenRouter API key (if None, will use OPENAI_API_KEY env var)
            base_url: Base URL for API (if None, will use OPENAI_BASE_URL env var)
            batch_size: Maximum number of texts to process in a single request
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            raise OpenRouterError(
                "OpenRouter API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        if not self.base_url:
            raise OpenRouterError(
                "OpenRouter base URL required. Set OPENAI_BASE_URL environment variable or pass base_url parameter."
            )

        # Validate API key is not a placeholder
        if self.api_key.startswith("your_") and self.api_key.endswith("_here"):
            raise OpenRouterError("Invalid API key - appears to be placeholder text.")

        self.embeddings_url = f"{self.base_url.rstrip('/')}/embeddings"

        # Skip connection test - will fail on first actual request if needed
        # This avoids unnecessary fallback to local models during initialization

        logger.info(f"OpenRouter client initialized for model: {self.model_name}")

    def _test_connection(self) -> None:
        """Test API connection with a minimal request."""
        try:
            # Send a minimal test request
            test_data = {"model": self.model_name, "input": ["connection test"]}
            headers = self._get_headers()

            response = requests.post(
                self.embeddings_url, headers=headers, json=test_data, timeout=10
            )

            if response.status_code != 200:
                raise OpenRouterError(
                    f"API connection test failed: {response.status_code} - {response.text}"
                )

        except Exception as e:
            raise OpenRouterError(f"Failed to connect to OpenRouter API: {e}") from e

    def _get_headers(self) -> dict:
        """Get request headers for API calls."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "imas-mcp",  # Optional: identify your application
            "X-Title": "IMAS MCP Server",  # Optional: application name
        }

    def _make_embedding_request(self, texts: list[str]) -> list[list[float]]:
        """Make a single embedding request to the API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            OpenRouterError: If the API request fails
        """
        data = {"model": self.model_name, "input": texts}

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.embeddings_url,
                    headers=self._get_headers(),
                    json=data,
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    embeddings = [item["embedding"] for item in result["data"]]
                    return embeddings

                elif response.status_code == 429:  # Rate limited
                    if attempt < self.max_retries:
                        wait_time = self.retry_delay * (
                            2**attempt
                        )  # Exponential backoff
                        logger.warning(
                            f"Rate limited, retrying in {wait_time:.1f} seconds..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        raise OpenRouterError(
                            f"Rate limit exceeded after {self.max_retries} retries"
                        )

                else:
                    error_msg = (
                        f"API request failed: {response.status_code} - {response.text}"
                    )
                    if attempt < self.max_retries:
                        logger.warning(f"{error_msg}, retrying...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise OpenRouterError(error_msg)

            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(f"Request failed: {e}, retrying...")
                    time.sleep(self.retry_delay)
                else:
                    break

        raise OpenRouterError(
            f"Failed to make embedding request after {self.max_retries} retries: {last_error}"
        )

    def encode(
        self,
        sentences: str | list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Generate embeddings for the given sentences.

        This method provides a SentenceTransformer-compatible interface.

        Args:
            sentences: String or list of strings to encode
            convert_to_numpy: If True, return numpy array
            normalize_embeddings: If True, normalize embeddings to unit length
            batch_size: Override default batch size
            show_progress_bar: Ignored (for interface compatibility)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Numpy array of embeddings
        """
        # Handle single string input
        if isinstance(sentences, str):
            sentences = [sentences]

        if not sentences:
            return np.array([]).reshape(0, 0)

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]

            if show_progress_bar:
                logger.info(
                    f"Processing batch {i // batch_size + 1}/{(len(sentences) + batch_size - 1) // batch_size}"
                )

            try:
                batch_embeddings = self._make_embedding_request(batch)
                all_embeddings.extend(batch_embeddings)

            except OpenRouterError as e:
                logger.error(f"Failed to process batch {i // batch_size + 1}: {e}")
                raise

        embeddings_array = np.array(all_embeddings)

        # Normalize embeddings if requested
        if normalize_embeddings and embeddings_array.size > 0:
            # Normalize to unit length
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1
            embeddings_array = embeddings_array / norms

        logger.info(
            f"Generated embeddings: {embeddings_array.shape[0]} texts, dim={embeddings_array.shape[1]}"
        )

        return embeddings_array

    @property
    def device(self) -> str:
        """Return device type for compatibility with SentenceTransformer."""
        return "api"

    def get_sentence_embedding_dimension(self) -> int:
        """Get the embedding dimension for this model.

        Note: This makes a test request to determine the dimension.
        For production, you might want to cache this value.
        """
        try:
            test_embedding = self._make_embedding_request(["test"])[0]
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Failed to determine embedding dimension: {e}")
            raise OpenRouterError(f"Cannot determine embedding dimension: {e}") from e

    def save(self, path: str) -> None:
        """Save model (not applicable for API models)."""
        raise NotImplementedError(
            "Cannot save API-based models. This is an OpenRouter API client."
        )

    def encode_multi_process(self, *args, **kwargs) -> np.ndarray:
        """Multi-process encoding (not applicable for API models)."""
        raise NotImplementedError(
            "Multi-process encoding not supported for API models."
        )


# Factory function for easy instantiation
def create_openrouter_client(
    model_name: str, api_key: str | None = None, base_url: str | None = None, **kwargs
) -> OpenRouterClient:
    """Create and return an OpenRouterClient instance.

    Args:
        model_name: Name of the embedding model
        api_key: Optional API key (uses env var if not provided)
        base_url: Optional base URL (uses env var if not provided)
        **kwargs: Additional arguments passed to OpenRouterClient constructor

    Returns:
        OpenRouterClient instance
    """
    return OpenRouterClient(
        model_name=model_name, api_key=api_key, base_url=base_url, **kwargs
    )
