"""OpenRouter embedding client for cloud-based embeddings.

Uses Qwen3-Embedding via OpenRouter API with Matryoshka dimension
projection to 256d.

Model name mapping:
- HuggingFace: Qwen/Qwen3-Embedding-0.6B (default), Qwen/Qwen3-Embedding-4B, 8B
- OpenRouter API: qwen/qwen3-embedding-4b, qwen/qwen3-embedding-8b

Cost Tracking:
- Uses EmbeddingCostTracker for session-level cost tracking
- Prices from OpenRouter (https://openrouter.ai/models)
- embed_with_cost() returns EmbeddingResult with cost breakdown

Usage:
    client = OpenRouterEmbeddingClient()
    if client.is_available():
        embeddings = client.embed(["text1", "text2"])

    # With cost tracking:
    tracker = EmbeddingCostTracker(limit_usd=1.0)
    result = client.embed_with_cost(["text1", "text2"], tracker)
    print(f"Cost: ${result.cost_usd:.6f}")
"""

import logging
import os
import time
from dataclasses import dataclass

import httpx
import numpy as np

from imas_codex.settings import get_embedding_dimension, get_imas_embedding_model

logger = logging.getLogger(__name__)

# OpenRouter embedding API endpoint
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model name mappings between HuggingFace and OpenRouter formats.
# Generic conversion: "Org/Model-Name" → "org/model-name"
# Explicit overrides only needed when the mapping is non-trivial.
MODEL_NAME_MAP: dict[str, str] = {}

# Reverse mapping for validation (populated dynamically)
OPENROUTER_TO_HF_MAP: dict[str, str] = {}

# Default timeout for embedding requests (seconds)
DEFAULT_TIMEOUT = 120.0
# Connection timeout (seconds)
CONNECT_TIMEOUT = 10.0

# OpenRouter embedding model pricing (USD per 1M tokens)
EMBEDDING_MODEL_COSTS: dict[str, float] = {
    "qwen/qwen3-embedding-8b": 0.01,
    "qwen/qwen3-embedding-4b": 0.02,
}

# Default fallback cost estimate for unknown models
DEFAULT_EMBEDDING_COST_PER_1M = 0.10


class OpenRouterEmbeddingError(Exception):
    """Error raised when OpenRouter embedding fails."""

    pass


class EmbeddingBudgetExhaustedError(Exception):
    """Raised when embedding cost limit is exceeded."""

    def __init__(self, summary: str):
        self.summary = summary
        super().__init__(f"Embedding budget exhausted: {summary}")


@dataclass
class EmbeddingResult:
    """Result of an embedding operation with cost tracking.

    Separates embeddings from cost metadata for clean API.

    Attributes:
        embeddings: The embedding vectors
        total_tokens: Actual token count from API response (0 for local/remote)
        cost_usd: Actual cost from API (0.0 for local/remote)
        model: Model identifier used
        elapsed_seconds: Time taken for the operation
        source: Embedding source - "local", "remote", or "openrouter"
    """

    embeddings: np.ndarray
    total_tokens: int
    cost_usd: float
    model: str
    elapsed_seconds: float
    source: str = "openrouter"  # "local" | "remote" | "openrouter"


@dataclass
class EmbeddingCostTracker:
    """Track embedding costs across operations with optional budget enforcement.

    Similar to CostTracker in agentic/session.py but for embeddings.
    Uses per-token pricing model (embeddings only have input tokens).

    Attributes:
        limit_usd: Maximum cost in USD (None = no limit)
        total_cost_usd: Running total of costs incurred
        total_tokens: Total tokens processed
        request_count: Number of embedding requests made
    """

    limit_usd: float | None = None
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    request_count: int = 0

    def record(
        self,
        tokens: int,
        model: str,
        cost_usd: float | None = None,
    ) -> float:
        """Record a request and return its cost.

        Args:
            tokens: Actual number of tokens processed
            model: Model identifier for pricing lookup
            cost_usd: Actual cost from API (if available). If None, calculates from tokens.

        Returns:
            Cost of this request in USD
        """
        if cost_usd is not None:
            cost = cost_usd
        else:
            cost = calculate_embedding_cost(tokens, model)
        self.total_cost_usd += cost
        self.total_tokens += tokens
        self.request_count += 1
        return cost

    def is_exhausted(self) -> bool:
        """Check if budget limit has been reached."""
        if self.limit_usd is None:
            return False
        return self.total_cost_usd >= self.limit_usd

    def remaining_usd(self) -> float | None:
        """Return remaining budget in USD, or None if unlimited."""
        if self.limit_usd is None:
            return None
        return max(0.0, self.limit_usd - self.total_cost_usd)

    def summary(self) -> str:
        """Return human-readable cost summary."""
        parts = [f"${self.total_cost_usd:.6f} spent"]
        if self.limit_usd is not None:
            parts.append(f"(limit: ${self.limit_usd:.2f})")
        parts.append(f"| {self.request_count} requests")
        parts.append(f"| {self.total_tokens:,} tokens")
        return " ".join(parts)


def calculate_embedding_cost(tokens: int, model: str) -> float:
    """Calculate cost for an embedding request using actual token count.

    Args:
        tokens: Actual number of tokens (from API response)
        model: Model identifier for pricing lookup

    Returns:
        Cost in USD based on actual token usage
    """
    cost_per_1m = EMBEDDING_MODEL_COSTS.get(
        model.lower(), DEFAULT_EMBEDDING_COST_PER_1M
    )
    return (tokens * cost_per_1m) / 1_000_000


# Keep estimate_embedding_cost as alias for backwards compatibility
estimate_embedding_cost = calculate_embedding_cost


@dataclass
class OpenRouterServerInfo:
    """Information about OpenRouter embedding service."""

    model: str
    dimension: int


def get_openrouter_model_name(hf_model_name: str) -> str:
    """Convert HuggingFace model name to OpenRouter format.

    Uses explicit overrides from MODEL_NAME_MAP first, then falls back
    to generic conversion: "Org/Model-Name" → "org/model-name".

    Availability is validated at request time by OpenRouter's API, not here.

    Args:
        hf_model_name: Model name in HuggingFace format (e.g., Qwen/Qwen3-Embedding-4B)

    Returns:
        Model name in OpenRouter format (e.g., qwen/qwen3-embedding-4b)

    Raises:
        ValueError: If model name format is unrecognizable
    """
    # Check explicit overrides first
    openrouter_name = MODEL_NAME_MAP.get(hf_model_name)
    if openrouter_name:
        return openrouter_name

    # Generic conversion: lowercase the HuggingFace name
    # HuggingFace: "Org/Model-Name" → OpenRouter: "org/model-name"
    name = hf_model_name.strip()
    if "/" in name:
        return name.lower()

    raise ValueError(
        f"Cannot convert '{hf_model_name}' to OpenRouter format. "
        "Expected 'org/model-name' format."
    )


class OpenRouterEmbeddingClient:
    """Client for OpenRouter embedding API.

    Provides cloud-based embeddings using the same Qwen3-Embedding model
    as the remote GPU server, enabling transparent fallback.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        dimensions: int | None = None,
    ):
        """Initialize client.

        Args:
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY
                    or OPENAI_API_KEY environment variable.
            model_name: Model name (OpenRouter or HuggingFace format accepted).
                       Defaults to configured model from settings.
            timeout: Request timeout in seconds
            dimensions: Matryoshka projection dimension. Defaults to configured
                       dimension from settings.
        """
        if model_name is None:
            model_name = get_imas_embedding_model()

        self.api_key = (
            api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        self.timeout = timeout
        self.dimensions = dimensions or get_embedding_dimension()
        self._client: httpx.Client | None = None
        self._warned_fallback: bool = False

        # Handle model name in either format
        if model_name in MODEL_NAME_MAP:
            self.model_name = MODEL_NAME_MAP[model_name]
        else:
            self.model_name = model_name.lower()

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=OPENROUTER_BASE_URL,
                timeout=httpx.Timeout(
                    connect=CONNECT_TIMEOUT,
                    read=self.timeout,
                    write=self.timeout,
                    pool=self.timeout,
                ),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/iterorganization/imas-codex",
                    "X-Title": "IMAS Codex",
                },
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OpenRouterEmbeddingClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def is_available(self) -> bool:
        """Check if OpenRouter embedding API is available.

        Returns:
            True if API key is configured and valid
        """
        if not self.api_key:
            logger.debug("OpenRouter API key not configured")
            return False

        # Validate API key is not a placeholder
        if self.api_key.startswith("your_") and self.api_key.endswith("_here"):
            logger.debug("OpenRouter API key appears to be placeholder")
            return False

        return True

    def get_info(self) -> OpenRouterServerInfo | None:
        """Get embedding service information.

        Returns:
            Service info or None if unavailable
        """
        if not self.is_available():
            return None

        return OpenRouterServerInfo(
            model=self.model_name,
            dimension=self.dimensions,
        )

    def embed(
        self,
        texts: list[str],
        normalize: bool = True,
        max_retries: int = 3,
        cost_tracker: EmbeddingCostTracker | None = None,
    ) -> np.ndarray:
        """Embed texts using OpenRouter API.

        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings (L2 normalization)
            max_retries: Maximum retry attempts for transient errors
            cost_tracker: Optional tracker for accumulating costs

        Returns:
            Numpy array of embeddings (shape: [len(texts), dimensions])

        Raises:
            OpenRouterEmbeddingError: If API key not configured
            EmbeddingBudgetExhaustedError: If cost tracker budget exceeded
            ConnectionError: If API is unavailable after retries
            RuntimeError: If embedding fails after retries
        """
        result = self.embed_with_cost(texts, normalize, max_retries, cost_tracker)
        return result.embeddings

    def embed_with_cost(
        self,
        texts: list[str],
        normalize: bool = True,
        max_retries: int = 3,
        cost_tracker: EmbeddingCostTracker | None = None,
    ) -> EmbeddingResult:
        """Embed texts and return result with cost information.

        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings (L2 normalization)
            max_retries: Maximum retry attempts for transient errors
            cost_tracker: Optional tracker for accumulating costs

        Returns:
            EmbeddingResult with embeddings and cost breakdown

        Raises:
            OpenRouterEmbeddingError: If API key not configured
            EmbeddingBudgetExhaustedError: If cost tracker budget exceeded
            ConnectionError: If API is unavailable after retries
            RuntimeError: If embedding fails after retries
        """
        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                total_tokens=0,
                cost_usd=0.0,
                model=self.model_name,
                elapsed_seconds=0.0,
                source="openrouter",
            )

        if not self.is_available():
            raise OpenRouterEmbeddingError(
                "OpenRouter API key not configured. "
                "Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable."
            )

        # Check budget before making request
        if cost_tracker and cost_tracker.is_exhausted():
            raise EmbeddingBudgetExhaustedError(cost_tracker.summary())

        client = self._get_client()
        start = time.time()
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                # OpenRouter uses OpenAI-compatible embeddings endpoint
                response = client.post(
                    "/embeddings",
                    json={
                        "model": self.model_name,
                        "input": texts,
                        "dimensions": self.dimensions,
                    },
                )

                if response.status_code == 401:
                    raise OpenRouterEmbeddingError(
                        "Invalid OpenRouter API key. Check OPENROUTER_API_KEY."
                    )

                if response.status_code == 429:
                    # Rate limited - retry with backoff
                    if attempt < max_retries - 1:
                        delay = 2 ** (attempt + 1)
                        logger.warning(
                            f"Rate limited (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        continue
                    raise RuntimeError("OpenRouter rate limit exceeded")

                if response.status_code != 200:
                    error_detail = response.text
                    try:
                        error_detail = (
                            response.json()
                            .get("error", {})
                            .get("message", response.text)
                        )
                    except Exception:
                        pass
                    raise RuntimeError(f"OpenRouter embedding failed: {error_detail}")

                data = response.json()

                # Extract embeddings from OpenAI-format response
                embedding_data = data.get("data", [])
                if not embedding_data:
                    # Some providers don't support the dimensions parameter.
                    # Retry without it and truncate client-side.
                    logger.debug(
                        "Empty embeddings with dimensions=%d, "
                        "retrying without dimensions parameter",
                        self.dimensions,
                    )
                    retry_response = client.post(
                        "/embeddings",
                        json={
                            "model": self.model_name,
                            "input": texts,
                        },
                    )
                    if retry_response.status_code == 200:
                        data = retry_response.json()
                        embedding_data = data.get("data", [])

                if not embedding_data:
                    raise RuntimeError(
                        f"No embeddings returned from OpenRouter "
                        f"(model={self.model_name}, texts={len(texts)})"
                    )

                # Sort by index to ensure correct order
                embedding_data.sort(key=lambda x: x.get("index", 0))
                embeddings = np.array(
                    [item["embedding"] for item in embedding_data],
                    dtype=np.float32,
                )

                # Truncate to configured dimension if server returned
                # native dimensions (e.g., when dimensions param was
                # not supported and we retried without it)
                if embeddings.ndim == 2 and embeddings.shape[1] > self.dimensions:
                    embeddings = embeddings[:, : self.dimensions]

                # Apply L2 normalization if requested
                if normalize:
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
                    embeddings = embeddings / norms

                elapsed = time.time() - start
                usage = data.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                cost_usd = calculate_embedding_cost(total_tokens, self.model_name)

                # Record cost if tracker provided
                if cost_tracker:
                    cost_tracker.record(total_tokens, self.model_name, cost_usd)

                logger.debug(
                    f"OpenRouter embedding: {len(texts)} texts in {elapsed:.2f}s "
                    f"(tokens: {total_tokens}, cost: ${cost_usd:.6f})"
                )

                return EmbeddingResult(
                    embeddings=embeddings,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                    model=self.model_name,
                    elapsed_seconds=elapsed,
                    source="openrouter",
                )

            except httpx.ConnectError as e:
                last_error = ConnectionError(f"Cannot connect to OpenRouter: {e}")
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
                last_error = ConnectionError(f"OpenRouter request timed out: {e}")
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
                error_msg = str(e).lower()
                is_transient = any(
                    x in error_msg
                    for x in ["disconnected", "reset", "broken", "503", "502"]
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

        raise last_error or RuntimeError("Embedding failed after retries")


def get_openrouter_client(
    api_key: str | None = None,
    model_name: str | None = None,
) -> OpenRouterEmbeddingClient | None:
    """Get an OpenRouter embedding client if configured.

    Args:
        api_key: Optional explicit API key
        model_name: Optional model name (defaults to configured model)

    Returns:
        Client instance or None if not configured
    """
    if model_name is None:
        model_name = get_imas_embedding_model()

    client = OpenRouterEmbeddingClient(api_key=api_key, model_name=model_name)
    if client.is_available():
        return client
    return None


__all__ = [
    "EmbeddingBudgetExhaustedError",
    "EmbeddingCostTracker",
    "EmbeddingResult",
    "EMBEDDING_MODEL_COSTS",
    "MODEL_NAME_MAP",
    "OpenRouterEmbeddingClient",
    "OpenRouterEmbeddingError",
    "OpenRouterServerInfo",
    "calculate_embedding_cost",
    "estimate_embedding_cost",  # backwards compatibility alias
    "get_openrouter_client",
    "get_openrouter_model_name",
]
