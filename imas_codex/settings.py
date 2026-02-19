"""Project settings loaded from pyproject.toml [tool.imas-codex] section.

Configuration is organized into subsections:
  [tool.imas-codex]                — general settings
  [tool.imas-codex.graph]          — Neo4j graph URI, username, password
  [tool.imas-codex.data-dictionary] — DD version, include-ggd, include-error-fields
  [tool.imas-codex.embedding]      — embedding model, dimension, backend
  [tool.imas-codex.language]       — language models, batch-size for structured output
  [tool.imas-codex.vision]         — vision models for image/document tasks
  [tool.imas-codex.agent]          — agent models for planning/exploration tasks
  [tool.imas-codex.compaction]     — compaction models for summarization tasks

All settings support environment variable overrides (IMAS_CODEX_* prefix / NEO4J_*).
"""

import importlib.resources
import os
from functools import cache

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]


@cache
def _load_pyproject_settings() -> dict:
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


def _get_section(section: str) -> dict:
    """Get a subsection from [tool.imas-codex.{section}]."""
    return _load_pyproject_settings().get(section, {})


# ─── Valid model sections ───────────────────────────────────────────────────

MODEL_SECTIONS = frozenset({"embedding", "language", "vision", "agent", "compaction"})

# Default model per section (fallback when not configured)
_MODEL_DEFAULTS: dict[str, str] = {
    "embedding": "Qwen/Qwen3-Embedding-0.6B",
    "language": "google/gemini-3-flash-preview",
    "vision": "google/gemini-3-flash-preview",
    "agent": "anthropic/claude-sonnet-4.5",
    "compaction": "anthropic/claude-haiku-4.5",
}

# Environment variable names per section
_MODEL_ENV_VARS: dict[str, str] = {
    "embedding": "IMAS_CODEX_EMBEDDING_MODEL",
    "language": "IMAS_CODEX_LANGUAGE_MODEL",
    "vision": "IMAS_CODEX_VISION_MODEL",
    "agent": "IMAS_CODEX_AGENT_MODEL",
    "compaction": "IMAS_CODEX_COMPACTION_MODEL",
}


def get_model(section: str) -> str:
    """Get the configured model for a pyproject.toml section.

    Accepted sections match [tool.imas-codex.*]:
        language, vision, agent, compaction, embedding

    Priority: env var → [tool.imas-codex.{section}].model → default.

    Args:
        section: One of the MODEL_SECTIONS keys.

    Returns:
        Model identifier string (e.g. 'google/gemini-3-flash-preview').

    Raises:
        ValueError: If section is not a valid model section.
    """
    if section not in MODEL_SECTIONS:
        raise ValueError(
            f"Unknown model section '{section}'. "
            f"Valid sections: {', '.join(sorted(MODEL_SECTIONS))}"
        )
    if env_var := _MODEL_ENV_VARS.get(section):
        if env := os.getenv(env_var):
            return env
    return _get_section(section).get("model", _MODEL_DEFAULTS[section])


# ─── Embedding settings ────────────────────────────────────────────────────


def get_embedding_model() -> str:
    """Get the embedding model name.

    Convenience wrapper around get_model("embedding").
    """
    return get_model("embedding")


def get_embedding_dimension() -> int:
    """Get the target embedding dimension (Matryoshka projection).

    All vectors in the graph, indexes, and caches use this dimension.

    Priority: IMAS_CODEX_EMBEDDING_DIMENSION env → [embedding].dimension → 256.
    """
    if env := os.getenv("IMAS_CODEX_EMBEDDING_DIMENSION"):
        return int(env)
    dim = _get_section("embedding").get("dimension")
    return int(dim) if dim is not None else 256


def get_embedding_backend() -> str:
    """Get the embedding backend ('local' or 'remote').

    Priority: IMAS_CODEX_EMBEDDING_BACKEND env → [embedding].backend → 'remote'.
    """
    if env := os.getenv("IMAS_CODEX_EMBEDDING_BACKEND"):
        return env.lower()
    backend = _get_section("embedding").get("backend")
    return str(backend).lower() if backend else "remote"


def get_embed_remote_url() -> str | None:
    """Get the remote embedding server URL.

    Priority: IMAS_CODEX_EMBED_REMOTE_URL env → [embedding].remote-url → None.
    """
    if env := os.getenv("IMAS_CODEX_EMBED_REMOTE_URL"):
        return env or None
    url = _get_section("embedding").get("remote-url")
    return url or None


def get_embed_server_port() -> int:
    """Get the embedding server port.

    Priority: IMAS_CODEX_EMBED_PORT env → [embedding].server-port → 18765.
    """
    if env := os.getenv("IMAS_CODEX_EMBED_PORT"):
        return int(env)
    port = _get_section("embedding").get("server-port")
    return int(port) if port is not None else 18765


def get_embed_host() -> str | None:
    """Get the hostname where the embedding server runs.

    When set, tunnel commands forward to this host instead of 127.0.0.1.
    Used for routing when the embed server runs on a compute node (e.g. Titan).

    Priority: IMAS_CODEX_EMBED_HOST env → [embedding].embed-host → None.
    """
    if env := os.getenv("IMAS_CODEX_EMBED_HOST"):
        return env or None
    host = _get_section("embedding").get("embed-host")
    return host or None


# ─── Model accessors ───────────────────────────────────────────────────────
# All callers should use get_model("language"), get_model("vision"), etc.
# The embedding model is accessed via get_embedding_model() for consistency
# with the other embedding accessors (dimension, backend, etc.).


# ─── Graph settings (Neo4j) ────────────────────────────────────────────────
# Resolved via named graph profiles.  See imas_codex.graph.profiles for
# the full resolution chain (profiles → convention).


def get_graph_profile():  # type: ignore[return]
    """Get the fully resolved :class:`Neo4jProfile` for the active graph.

    This is the canonical entry point — all other ``get_graph_*()``
    accessors delegate here.
    """
    from imas_codex.graph.profiles import resolve_neo4j

    return resolve_neo4j()


def get_graph_uri() -> str:
    """Get the Neo4j bolt URI for the active graph profile."""
    return get_graph_profile().uri


def get_graph_username() -> str:
    """Get the Neo4j username for the active graph profile."""
    return get_graph_profile().username


def get_graph_password() -> str:
    """Get the Neo4j password for the active graph profile."""
    return get_graph_profile().password


def get_graph_name() -> str:
    """Get the active graph identity name (e.g. ``"codex"``, ``"tcv"``)."""
    return get_graph_profile().name


def get_graph_location() -> str:
    """Get the active graph location (e.g. ``"iter"``, ``"local"``)."""
    return get_graph_profile().location


# ─── Data dictionary settings ──────────────────────────────────────────────


def get_dd_version() -> str:
    """Get the default data dictionary version.

    Priority: IMAS_DD_VERSION env → [data-dictionary].version → '4.1.0'.
    """
    if env := os.getenv("IMAS_DD_VERSION"):
        return env
    return _get_section("data-dictionary").get("version", "4.1.0")


def get_include_ggd() -> bool:
    """Get whether to include GGD (Grid Geometry Description) paths.

    Priority: IMAS_CODEX_INCLUDE_GGD env → [data-dictionary].include-ggd → True.
    """
    if env := os.getenv("IMAS_CODEX_INCLUDE_GGD"):
        return _parse_bool(env)
    val = _get_section("data-dictionary").get("include-ggd")
    if val is not None:
        return _parse_bool(val)
    return True


def get_include_error_fields() -> bool:
    """Get whether to include error fields (_error_upper, _error_lower, etc.).

    Priority: IMAS_CODEX_INCLUDE_ERROR_FIELDS env → [data-dictionary].include-error-fields → False.
    """
    if env := os.getenv("IMAS_CODEX_INCLUDE_ERROR_FIELDS"):
        return _parse_bool(env)
    val = _get_section("data-dictionary").get("include-error-fields")
    if val is not None:
        return _parse_bool(val)
    return False


# ─── General settings ──────────────────────────────────────────────────────


def get_labeling_batch_size() -> int:
    """Get batch size for cluster labeling.

    Priority: IMAS_CODEX_LABELING_BATCH_SIZE env → [language].batch-size → 50.
    """
    if env := os.getenv("IMAS_CODEX_LABELING_BATCH_SIZE"):
        return int(env)
    if (val := _get_section("language").get("batch-size")) is not None:
        return int(val)
    return 50


def _parse_bool(value: str | bool) -> bool:
    """Parse a boolean value from string or bool."""
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes")


# ─── Native model dimensions (for validation only) ─────────────────────────

MODEL_NATIVE_DIMENSIONS: dict[str, int] = {
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "qwen/qwen3-embedding-0.6b": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "qwen/qwen3-embedding-4b": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
    "qwen/qwen3-embedding-8b": 4096,
}

MODEL_DIMENSIONS = MODEL_NATIVE_DIMENSIONS


# ─── Module-level constants ────────────────────────────────────────────────

LABELING_BATCH_SIZE = get_labeling_batch_size()
INCLUDE_GGD = get_include_ggd()
INCLUDE_ERROR_FIELDS = get_include_error_fields()
EMBEDDING_BACKEND = get_embedding_backend()
EMBED_REMOTE_URL = get_embed_remote_url()
EMBED_SERVER_PORT = get_embed_server_port()
EMBED_HOST = get_embed_host()
EMBEDDING_DIMENSION = get_embedding_dimension()
