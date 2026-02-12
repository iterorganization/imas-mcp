"""Project settings loaded from pyproject.toml [tool.imas-codex] section.

Configuration is organized into subsections:
  [tool.imas-codex]           — general settings (dd-version, include-ggd, etc.)
  [tool.imas-codex.embedding] — embedding model, dimension, backend
  [tool.imas-codex.language]  — language models for structured output tasks
  [tool.imas-codex.agent]     — agent models for planning/exploration tasks

All settings support environment variable overrides (IMAS_CODEX_* prefix).
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


# ─── Task routing ───────────────────────────────────────────────────────────

# Tasks routed to [tool.imas-codex.language]
LANGUAGE_TASKS = frozenset(
    {
        "discovery",
        "score",
        "enrichment",
        "vlm",
        "captioning",
        "labeling",
    }
)

# Tasks routed to [tool.imas-codex.agent]
AGENT_TASKS = frozenset(
    {
        "exploration",
        "scout",
        "summarization",
        "default",
    }
)


# ─── Embedding settings ────────────────────────────────────────────────────


def get_embedding_model() -> str:
    """Get the embedding model name.

    Priority: IMAS_CODEX_EMBEDDING_MODEL env → [embedding].model → fallback.
    """
    if env := os.getenv("IMAS_CODEX_EMBEDDING_MODEL"):
        return env
    return _get_section("embedding").get("model", "Qwen/Qwen3-Embedding-0.6B")


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


# ─── Language model settings ───────────────────────────────────────────────


def get_language_model(task: str | None = None) -> str:
    """Get the language model for structured output tasks.

    Used for high-volume tasks: scoring, discovery, enrichment, labeling.

    Args:
        task: Optional task override (e.g. 'vlm'). Falls back to section default.

    Priority: IMAS_CODEX_LANGUAGE_MODEL env → [language].{task} → [language].model.
    """
    if env := os.getenv("IMAS_CODEX_LANGUAGE_MODEL"):
        return env
    section = _get_section("language")
    if task and (model := section.get(task)):
        return model
    return section.get("model", "google/gemini-3-flash-preview")


# ─── Agent model settings ─────────────────────────────────────────────────


def get_agent_model(task: str | None = None) -> str:
    """Get the agent model for planning and exploration tasks.

    Args:
        task: Optional task key (e.g. 'summarization', 'default').
              Falls back to section default (capable model).
    """
    section = _get_section("agent")
    if task and (model := section.get(task)):
        return model
    return section.get("model", "anthropic/claude-sonnet-4.5")


# ─── Unified task routing ──────────────────────────────────────────────────


def get_model_for_task(task: str) -> str:
    """Get the configured model for a task type.

    Routes to the appropriate config section:
      Language tasks (discovery, score, enrichment, vlm) → [language]
      Agent tasks (exploration, scout, summarization, default) → [agent]
    """
    if task in LANGUAGE_TASKS:
        return get_language_model(task)
    if task in AGENT_TASKS:
        return get_agent_model(task)
    # Unknown task → agent default
    return get_agent_model("default")


# ─── General settings ──────────────────────────────────────────────────────


def get_labeling_batch_size() -> int:
    """Get batch size for cluster labeling.

    Priority: IMAS_CODEX_LABELING_BATCH_SIZE env → [tool.imas-codex] → 50.
    """
    if env := os.getenv("IMAS_CODEX_LABELING_BATCH_SIZE"):
        return int(env)
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

    Priority: IMAS_CODEX_INCLUDE_GGD env → [tool.imas-codex] → True.
    """
    if env := os.getenv("IMAS_CODEX_INCLUDE_GGD"):
        return _parse_bool(env)
    settings = _load_pyproject_settings()
    if (val := settings.get("include-ggd")) is not None:
        return _parse_bool(val)
    return True


def get_include_error_fields() -> bool:
    """Get whether to include error fields (_error_upper, _error_lower, etc.).

    Priority: IMAS_CODEX_INCLUDE_ERROR_FIELDS env → [tool.imas-codex] → False.
    """
    if env := os.getenv("IMAS_CODEX_INCLUDE_ERROR_FIELDS"):
        return _parse_bool(env)
    settings = _load_pyproject_settings()
    if (val := settings.get("include-error-fields")) is not None:
        return _parse_bool(val)
    return False


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


# ─── Backward compatibility aliases ────────────────────────────────────────

get_imas_embedding_model = get_embedding_model
get_imas_embedding_dimension = get_embedding_dimension


# ─── Module-level constants ────────────────────────────────────────────────

IMAS_CODEX_EMBEDDING_MODEL = get_embedding_model()
IMAS_CODEX_LANGUAGE_MODEL = get_language_model()
LABELING_BATCH_SIZE = get_labeling_batch_size()
INCLUDE_GGD = get_include_ggd()
INCLUDE_ERROR_FIELDS = get_include_error_fields()
EMBEDDING_BACKEND = get_embedding_backend()
EMBED_REMOTE_URL = get_embed_remote_url()
EMBED_SERVER_PORT = get_embed_server_port()
EMBEDDING_DIMENSION = get_embedding_dimension()
IMAS_EMBEDDING_DIMENSION = get_embedding_dimension()
