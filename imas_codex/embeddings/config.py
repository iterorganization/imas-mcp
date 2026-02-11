"""Configuration for embedding generation and caching."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Load .env file for local development
from dotenv import load_dotenv

from imas_codex.settings import (
    get_embed_remote_url,
    get_embedding_backend,
    get_imas_embedding_model,
)

load_dotenv(override=True)  # Load .env file values, overriding existing env vars


# Define constants - uses pyproject.toml defaults with env var override
IMAS_CODEX_EMBEDDING_MODEL = get_imas_embedding_model()


class EmbeddingBackend(Enum):
    """Embedding backend selection.

    No silent fallback: if the configured backend is unavailable, an error
    is raised immediately.
    """

    LOCAL = "local"  # Local CPU/GPU via SentenceTransformer
    REMOTE = "remote"  # Remote GPU server (iter cluster via SSH tunnel)


@dataclass
class EncoderConfig:
    """Configuration for embedding generation and management."""

    # Model configuration
    model_name: str | None = None
    device: str | None = None

    # Backend selection (explicit - no fallback)
    backend: EmbeddingBackend | None = None

    # Remote configuration (for REMOTE backend)
    remote_url: str | None = None

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

    def __post_init__(self) -> None:
        """Initialize configuration with environment variables if not explicitly set."""
        # Load model name: explicit param > env var > fallback
        if self.model_name is None:
            self.model_name = os.getenv(
                "IMAS_CODEX_EMBEDDING_MODEL", IMAS_CODEX_EMBEDDING_MODEL
            )

        # Load backend from settings if not explicitly set
        if self.backend is None:
            backend_str = get_embedding_backend()
            try:
                self.backend = EmbeddingBackend(backend_str)
            except ValueError:
                # Default to local if invalid backend specified
                self.backend = EmbeddingBackend.LOCAL

        # Load remote URL for remote backend
        if self.backend == EmbeddingBackend.REMOTE and self.remote_url is None:
            self.remote_url = get_embed_remote_url()

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
            "backend": self.backend.value if self.backend else None,
            "remote_url": self.remote_url,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "use_half_precision": self.use_half_precision,
            "enable_cache": self.enable_cache,
            "cache_dir": self.cache_dir,
            "ids_set": list(self.ids_set) if self.ids_set else None,
            "use_rich": self.use_rich,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EncoderConfig":
        """Create from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                if key == "ids_set" and value is not None:
                    setattr(config, key, set(value))
                elif key == "backend" and value is not None:
                    setattr(config, key, EmbeddingBackend(value))
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
