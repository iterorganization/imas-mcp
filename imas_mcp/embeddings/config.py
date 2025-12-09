"""Configuration for embedding generation and caching."""

import os
from dataclasses import dataclass
from typing import Any

# Load .env file for local development
from dotenv import load_dotenv

from imas_mcp.settings import get_imas_embedding_model

load_dotenv(override=True)  # Load .env file values, overriding existing env vars


# Define constants - uses pyproject.toml defaults with env var override
IMAS_MCP_EMBEDDING_MODEL = get_imas_embedding_model()


@dataclass
class EncoderConfig:
    """Configuration for embedding generation and management."""

    # Model configuration
    model_name: str | None = None
    device: str | None = None

    # Generation settings
    batch_size: int = 250
    normalize_embeddings: bool = True
    use_half_precision: bool = False

    # Cache configuration
    enable_cache: bool = True
    cache_dir: str = "embeddings"

    # Filtering
    ids_set: set[str] | None = None

    # Progress display
    use_rich: bool = True

    # OpenRouter API configuration
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    use_api_embeddings: bool = False  # Set to True when using API models

    def __post_init__(self) -> None:
        """Initialize configuration with environment variables if not explicitly set."""
        # Load model name: explicit param > env var > fallback
        if self.model_name is None:
            self.model_name = os.getenv(
                "IMAS_MCP_EMBEDDING_MODEL", IMAS_MCP_EMBEDDING_MODEL
            )

        # Load OpenRouter API configuration
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if self.openai_base_url is None:
            self.openai_base_url = os.getenv("OPENAI_BASE_URL")

        # Auto-detect if we should use API embeddings based on model name
        if self.use_api_embeddings is False:  # Check against default
            # Use API if model contains common API model indicators
            # The "/" indicator is key: OpenRouter models use "provider/model" format
            # while local SentenceTransformer models don't have slashes
            is_api_model = any(
                indicator in self.model_name
                for indicator in ["/", "embedding", "text-embedding", "qwen", "openai"]
            )
            self.use_api_embeddings = is_api_model

    def validate_api_config(self) -> None:
        """Validate API configuration when using API embeddings."""
        if self.use_api_embeddings:
            # Check if we have an API key
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable required for API embeddings"
                )

            # Validate API key is not placeholder
            if self.openai_api_key.startswith("your_") and self.openai_api_key.endswith(
                "_here"
            ):
                raise ValueError(
                    "Invalid API key - appears to be placeholder text. Please set a valid OPENAI_API_KEY."
                )

            # Check base URL
            if not self.openai_base_url:
                raise ValueError(
                    "OPENAI_BASE_URL environment variable required for API embeddings"
                )

    def generate_cache_key(self) -> str | None:
        """
        Generate consistent cache key for embeddings based on dataset characteristics.

        Document count validation is handled during cache loading, so we only
        need to identify the dataset subset (full vs filtered).

        Returns:
            Cache key string for filtered datasets, None for full dataset
            (None results in simpler cache filename without hash)
        """
        if self.ids_set:
            # For filtered datasets, use sorted IDS names
            ids_part = "_".join(sorted(self.ids_set))
            return f"filtered_{ids_part}"
        else:
            # For full dataset, return None to get simple filename
            return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "use_half_precision": self.use_half_precision,
            "enable_cache": self.enable_cache,
            "cache_dir": self.cache_dir,
            "ids_set": list(self.ids_set) if self.ids_set else None,
            "use_rich": self.use_rich,
            "use_api_embeddings": self.use_api_embeddings,
            "openai_base_url": self.openai_base_url,
            # Note: Don't include openai_api_key in serialization for security
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EncoderConfig":
        """Create from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                if key == "ids_set" and value is not None:
                    setattr(config, key, set(value))
                elif key == "openai_api_key":
                    # Don't restore API key from dict for security
                    continue
                else:
                    setattr(config, key, value)

        # Re-run post_init to set environment-based defaults if not explicitly set
        config.__post_init__()
        return config

    @classmethod
    def from_environment(cls) -> "EncoderConfig":
        """Create configuration entirely from environment variables.

        With the new None-based approach, simply creating an instance
        without parameters will automatically load from environment variables.
        """
        return cls()

    def get_api_info(self) -> dict[str, Any]:
        """Get API configuration info (without sensitive data)."""
        return {
            "use_api_embeddings": self.use_api_embeddings,
            "model_name": self.model_name,
            "has_api_key": bool(self.openai_api_key),
            "base_url": self.openai_base_url,
            "api_key_prefix": self.openai_api_key[:10] + "..."
            if self.openai_api_key
            else None,
        }
