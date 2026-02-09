"""Project settings loaded from pyproject.toml [tool.imas-codex] section.

This module provides centralized access to project configuration defaults,
with environment variable overrides for runtime flexibility.
"""

import importlib.resources
import os
from functools import cache

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]


@cache
def _load_pyproject_settings() -> dict[str, str]:
    """Load settings from pyproject.toml [tool.imas-codex] section.

    Returns:
        Dictionary of settings from pyproject.toml, empty dict if not found.
    """
    try:
        # Try package resources first (installed package)
        files = importlib.resources.files("imas_codex")
        pyproject_path = files.joinpath("..", "pyproject.toml")

        # If package resource doesn't exist, try filesystem
        if not pyproject_path.is_file():  # type: ignore[union-attr]
            from pathlib import Path

            # Walk up to find pyproject.toml (for development)
            current = Path(__file__).resolve().parent
            while current != current.parent:
                candidate = current / "pyproject.toml"
                if candidate.exists():
                    pyproject_path = candidate
                    break
                current = current.parent
            else:
                return {}

        # Read and parse the TOML file
        if hasattr(pyproject_path, "read_text"):
            content = pyproject_path.read_text()  # type: ignore[union-attr]
        else:
            from pathlib import Path

            content = Path(pyproject_path).read_text()  # type: ignore[arg-type]

        data = tomllib.loads(content)
        return data.get("tool", {}).get("imas-codex", {})
    except Exception:
        return {}


def get_imas_embedding_model() -> str:
    """Get the IMAS DD embedding model name.

    Priority:
        1. IMAS_CODEX_EMBEDDING_MODEL environment variable
        2. pyproject.toml [tool.imas-codex] imas-embedding-model
        3. Fallback default: Qwen/Qwen3-Embedding-8B

    Returns:
        Model name string.
    """
    if env_model := os.getenv("IMAS_CODEX_EMBEDDING_MODEL"):
        return env_model

    settings = _load_pyproject_settings()
    if model := settings.get("imas-embedding-model"):
        return model

    return "Qwen/Qwen3-Embedding-8B"


def get_imas_embedding_dimension() -> int:
    """Get the target embedding dimension (Matryoshka projection).

    Priority:
        1. IMAS_CODEX_EMBEDDING_DIMENSION environment variable
        2. pyproject.toml [tool.imas-codex] imas-embedding-dimension
        3. Fallback default: 256

    Returns:
        Target dimension for embedding output.
    """
    if env_dim := os.getenv("IMAS_CODEX_EMBEDDING_DIMENSION"):
        return int(env_dim)

    settings = _load_pyproject_settings()
    if (dim := settings.get("imas-embedding-dimension")) is not None:
        return int(dim)

    return 256


def get_language_model() -> str:
    """Get the LLM model for IMAS tasks (labeling, etc).

    Priority:
        1. IMAS_CODEX_LANGUAGE_MODEL environment variable
        2. pyproject.toml [tool.imas-codex] imas-language-model
        3. Fallback default: google/gemini-3-pro-preview

    Returns:
        Model name string.
    """
    if env_model := os.getenv("IMAS_CODEX_LANGUAGE_MODEL"):
        return env_model

    settings = _load_pyproject_settings()
    if model := settings.get("imas-language-model"):
        return model

    return "google/gemini-3-pro-preview"


def get_labeling_batch_size() -> int:
    """Get batch size for cluster labeling.

    Smaller batches provide more frequent progress updates and better
    resilience to failures. Cost difference is negligible (<2%).

    Priority:
        1. IMAS_CODEX_LABELING_BATCH_SIZE environment variable
        2. pyproject.toml [tool.imas-codex] labeling-batch-size
        3. Fallback default: 50

    Returns:
        Batch size as integer.
    """
    if env_val := os.getenv("IMAS_CODEX_LABELING_BATCH_SIZE"):
        return int(env_val)

    settings = _load_pyproject_settings()
    if (val := settings.get("labeling-batch-size")) is not None:
        return int(val)

    return 50


def _parse_bool(value: str | bool) -> bool:
    """Parse a boolean value from string or bool."""
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes")


def get_include_ggd() -> bool:
    """Get whether to include GGD (Grid Geometry Description) paths.

    Priority:
        1. IMAS_CODEX_INCLUDE_GGD environment variable
        2. pyproject.toml [tool.imas-codex] include-ggd
        3. Fallback default: True

    Returns:
        Boolean indicating whether to include GGD paths.
    """
    if env_val := os.getenv("IMAS_CODEX_INCLUDE_GGD"):
        return _parse_bool(env_val)

    settings = _load_pyproject_settings()
    if (val := settings.get("include-ggd")) is not None:
        return _parse_bool(val)

    return True


def get_include_error_fields() -> bool:
    """Get whether to include error fields (_error_upper, _error_lower, etc.).

    Priority:
        1. IMAS_CODEX_INCLUDE_ERROR_FIELDS environment variable
        2. pyproject.toml [tool.imas-codex] include-error-fields
        3. Fallback default: False

    Returns:
        Boolean indicating whether to include error fields.
    """
    if env_val := os.getenv("IMAS_CODEX_INCLUDE_ERROR_FIELDS"):
        return _parse_bool(env_val)

    settings = _load_pyproject_settings()
    if (val := settings.get("include-error-fields")) is not None:
        return _parse_bool(val)

    return False


def get_embedding_backend() -> str:
    """Get the embedding backend selection.

    Backend options:
        - "local": Local CPU/GPU via SentenceTransformer
        - "remote": Remote GPU server (iter cluster via SSH tunnel)
        - "openrouter": OpenRouter API for cloud embeddings

    No fallback between backends - if selected backend is unavailable,
    an error is raised rather than silently switching.

    Priority:
        1. IMAS_CODEX_EMBEDDING_BACKEND environment variable
        2. pyproject.toml [tool.imas-codex] embedding-backend
        3. Fallback default: "local"

    Returns:
        Backend name string.
    """
    if env_backend := os.getenv("IMAS_CODEX_EMBEDDING_BACKEND"):
        return env_backend.lower()

    settings = _load_pyproject_settings()
    if backend := settings.get("embedding-backend"):
        return str(backend).lower()

    return "local"


def get_embed_remote_url() -> str | None:
    """Get the remote embedding server URL.

    Priority:
        1. IMAS_CODEX_EMBED_REMOTE_URL environment variable
        2. pyproject.toml [tool.imas-codex] embed-remote-url
        3. Fallback default: None (local embedding only)

    Returns:
        URL string or None if not configured.
    """
    if env_url := os.getenv("IMAS_CODEX_EMBED_REMOTE_URL"):
        return env_url if env_url else None

    settings = _load_pyproject_settings()
    if url := settings.get("embed-remote-url"):
        return url if url else None

    return None


def get_embed_server_port() -> int:
    """Get the embedding server port.

    Priority:
        1. IMAS_CODEX_EMBED_PORT environment variable
        2. pyproject.toml [tool.imas-codex] embed-server-port
        3. Fallback default: 18765

    Returns:
        Port number.
    """
    if env_port := os.getenv("IMAS_CODEX_EMBED_PORT"):
        return int(env_port)

    settings = _load_pyproject_settings()
    if (port := settings.get("embed-server-port")) is not None:
        return int(port)

    return 18765


# Native model dimensions (before Matryoshka projection)
# Used only for validation — operational dimension comes from get_embedding_dimension()
MODEL_NATIVE_DIMENSIONS: dict[str, int] = {
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "qwen/qwen3-embedding-0.6b": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "qwen/qwen3-embedding-4b": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
    "qwen/qwen3-embedding-8b": 4096,
}

# Backwards-compatible alias
MODEL_DIMENSIONS = MODEL_NATIVE_DIMENSIONS


def get_embedding_dimension(model_name: str | None = None) -> int:
    """Get the operational embedding dimension.

    Returns the Matryoshka projection dimension from config, not the
    model's native dimension. All vectors in the graph, indexes, and
    caches use this dimension.

    Args:
        model_name: Ignored. Kept for API compatibility.

    Returns:
        Configured embedding dimension (default: 256).
    """
    return get_imas_embedding_dimension()


# Computed defaults (for use in module-level constants)
IMAS_CODEX_EMBEDDING_MODEL = get_imas_embedding_model()
IMAS_CODEX_LANGUAGE_MODEL = get_language_model()
LABELING_BATCH_SIZE = get_labeling_batch_size()
INCLUDE_GGD = get_include_ggd()
INCLUDE_ERROR_FIELDS = get_include_error_fields()
EMBEDDING_BACKEND = get_embedding_backend()
EMBED_REMOTE_URL = get_embed_remote_url()
EMBED_SERVER_PORT = get_embed_server_port()
EMBEDDING_DIMENSION = get_embedding_dimension()
IMAS_EMBEDDING_DIMENSION = get_imas_embedding_dimension()
