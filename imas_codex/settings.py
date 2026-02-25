"""Project settings loaded from pyproject.toml [tool.imas-codex] section.

Configuration is organized into subsections:
  [tool.imas-codex]                — general settings
  [tool.imas-codex.graph]          — Neo4j graph URI, username, password
  [tool.imas-codex.data-dictionary] — DD version, include-ggd, include-error-fields
  [tool.imas-codex.embedding]      — embedding model, dimension, location
  [tool.imas-codex.language]       — language models, batch-size for structured output
  [tool.imas-codex.vision]         — vision models for image/document tasks
  [tool.imas-codex.agent]          — agent models for planning/exploration tasks
  [tool.imas-codex.compaction]     — compaction models for summarization tasks

All settings support environment variable overrides (IMAS_CODEX_* prefix / NEO4J_*).
"""

import importlib.resources
import os
from functools import cache
from pathlib import Path

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

EMBED_BASE_PORT = 18765


def _get_location_offset(location: str) -> int:
    """Get the port offset for a location from the ``locations`` list.

    Reads ``[tool.imas-codex].locations`` — the same list used by graph
    profiles.  Position in the list is the offset.
    """
    locations = _load_pyproject_settings().get("locations", [])
    if isinstance(locations, list):
        offsets = {name: i for i, name in enumerate(locations)}
    else:
        offsets = {k: int(v) for k, v in locations.items()}
    return offsets.get(location, 0)


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


def get_embedding_location() -> str:
    """Get the embedding location — a facility name or ``"local"``.

    When set to a facility name (e.g. ``"iter"``), embeddings are served
    via an HTTP server running at that facility (reached via SSH tunnel
    from a workstation, or directly when on-site).  ``"local"`` loads
    the model in-process.

    Priority: IMAS_CODEX_EMBEDDING_LOCATION env → IMAS_CODEX_EMBEDDING_BACKEND env
              → [embedding].location → [embedding].backend → 'local'.
    """
    if env := os.getenv("IMAS_CODEX_EMBEDDING_LOCATION"):
        return env.lower()
    # Legacy env var
    if env := os.getenv("IMAS_CODEX_EMBEDDING_BACKEND"):
        return env.lower()
    section = _get_section("embedding")
    location = section.get("location") or section.get("backend")
    return str(location).lower() if location else "local"


# Backward-compatible alias
get_embedding_backend = get_embedding_location


def is_embedding_remote() -> bool:
    """True when embedding location targets a remote facility (not in-process)."""
    return get_embedding_location() != "local"


def get_embed_remote_url() -> str | None:
    """Get the remote embedding server URL.

    Delegates to :func:`resolve_service_url` from the shared locations
    module for unified local/SLURM/remote resolution.

    Override: ``IMAS_CODEX_EMBED_REMOTE_URL`` env var.
    """
    if env := os.getenv("IMAS_CODEX_EMBED_REMOTE_URL"):
        return env or None

    location = get_embedding_location()
    if location == "local":
        return None

    from imas_codex.remote.locations import resolve_service_url

    port = get_embed_server_port()
    return resolve_service_url(location, port)


def get_embed_server_port() -> int:
    """Get the embedding server port.

    Derived from the embedding location's facility offset:
    ``embed_port = 18765 + facility_offset``.
    For compute locations (e.g. ``"titan"``), uses the parent facility's offset.

    Priority: IMAS_CODEX_EMBED_PORT env → convention offset → 18765.
    """
    if env := os.getenv("IMAS_CODEX_EMBED_PORT"):
        return int(env)
    location = get_embedding_location()
    if location == "local":
        return EMBED_BASE_PORT
    from imas_codex.remote.locations import get_port_offset

    return EMBED_BASE_PORT + get_port_offset(location)


def get_embed_scheduler() -> str:
    """Get the embed server job scheduler.

    Derived from the location — compute locations (e.g. ``"titan"``)
    automatically resolve to ``"slurm"``.  Direct facility locations
    and ``"local"`` resolve to ``"none"``.

    Priority: IMAS_CODEX_EMBED_SCHEDULER env → location resolution → "none".
    """
    if env := os.getenv("IMAS_CODEX_EMBED_SCHEDULER"):
        return env.lower()
    location = get_embedding_location()
    if location == "local":
        return "none"
    from imas_codex.remote.locations import resolve_location

    return resolve_location(location).scheduler


def get_embed_host() -> str | None:
    """Get the hostname where the embedding server runs.

    For compute locations (e.g. ``"titan"``), reads the GPU node hostname
    from the facility's compute config.  For direct facility locations,
    returns ``None`` (server on login node / localhost).

    Override: IMAS_CODEX_EMBED_HOST env var (escape hatch).
    """
    if env := os.getenv("IMAS_CODEX_EMBED_HOST"):
        return env or None
    if get_embed_scheduler() != "slurm":
        return None
    return _embed_host_from_facility()


def _embed_host_from_facility() -> str | None:
    """Discover the compute node running the embedding server via SLURM.

    Resolves the facility from the embedding location (e.g. ``"titan"`` →
    ``"iter"``), then uses ``squeue`` to find the active service job's
    compute node.
    """
    try:
        from imas_codex.remote.locations import resolve_location
        from imas_codex.remote.tunnel import discover_compute_node_local

        location = get_embedding_location()
        info = resolve_location(location)
        if info.facility == "local":
            return None
        return discover_compute_node_local(
            service_job_name=info.service_job_name,
        )
    except Exception:
        pass
    return None


# ─── LLM proxy settings ────────────────────────────────────────────────────

LLM_BASE_PORT = 18400


def get_llm_proxy_port() -> int:
    """Get the LLM proxy (LiteLLM) port.

    Follows the same convention as embed/graph ports:
    ``llm_port = 18400 + facility_offset``.

    Priority: IMAS_CODEX_LLM_PORT env → LITELLM_PORT env → convention offset → 18400.
    """
    if env := os.getenv("IMAS_CODEX_LLM_PORT"):
        return int(env)
    if env := os.getenv("LITELLM_PORT"):
        return int(env)
    location = get_llm_location()
    if location == "local":
        return LLM_BASE_PORT
    from imas_codex.remote.locations import get_port_offset

    return LLM_BASE_PORT + get_port_offset(location)


def get_llm_proxy_url() -> str:
    """Get the LLM proxy URL.

    When the proxy runs on a remote facility (``[llm].location`` is set),
    resolves the login node hostname from facility config so that compute
    nodes can reach the proxy directly.  Falls back to ``127.0.0.1`` when
    running locally or when the hostname cannot be resolved.

    Override: LITELLM_PROXY_URL env var.
    """
    if env := os.getenv("LITELLM_PROXY_URL"):
        return env
    port = get_llm_proxy_port()
    host = _get_llm_proxy_host()
    return f"http://{host}:{port}"


def _get_llm_proxy_host() -> str:
    """Resolve the host where the LLM proxy runs.

    For facility locations, the proxy runs on the login node.  If we are
    on that facility (via ``is_local_host``), use the current machine's
    hostname directly.  Otherwise return ``"127.0.0.1"`` for SSH tunnel
    access.
    """
    import socket

    location = get_llm_location()
    if location == "local":
        return "127.0.0.1"

    from imas_codex.remote.locations import resolve_location

    info = resolve_location(location)

    # Proxy on SLURM compute node — access via localhost (tunnel or direct)
    if info.scheduler == "slurm":
        return "127.0.0.1"

    # If we're NOT on the facility, use 127.0.0.1 (access via SSH tunnel)
    try:
        from imas_codex.remote.executor import is_local_host

        if not is_local_host(info.ssh_host):
            return "127.0.0.1"
    except Exception:
        return "127.0.0.1"

    # We're on the facility — use the current machine's hostname.
    # The proxy runs on the login node we're currently logged into.
    return socket.gethostname().split(".")[0]


def get_llm_location() -> str:
    """Get the LLM proxy location — a facility name or ``"local"``.

    When set to a facility name (e.g. ``"iter"``), the proxy runs on that
    facility's compute node (reached via SSH tunnel).  ``"local"`` runs
    the proxy on the current machine.

    Priority: IMAS_CODEX_LLM_LOCATION env → [llm].location → 'local'.
    """
    if env := os.getenv("IMAS_CODEX_LLM_LOCATION"):
        return env.lower()
    location = _get_section("llm").get("location")
    return str(location).lower() if location else "local"


def get_llm_scheduler() -> str:
    """Get the LLM proxy job scheduler.

    Derived from the location — compute locations automatically resolve
    to ``"slurm"``.

    Priority: IMAS_CODEX_LLM_SCHEDULER env → location resolution → "none".
    """
    if env := os.getenv("IMAS_CODEX_LLM_SCHEDULER"):
        return env.lower()
    location = get_llm_location()
    if location == "local":
        return "none"
    from imas_codex.remote.locations import resolve_location

    return resolve_location(location).scheduler


# ─── Graph scheduler settings ──────────────────────────────────────────────


def get_graph_scheduler() -> str:
    """Get the graph server job scheduler.

    Derived from the graph location — compute locations (e.g. ``"titan"``)
    automatically resolve to ``"slurm"``.

    Priority: IMAS_CODEX_GRAPH_SCHEDULER env → location resolution → "none".
    """
    if env := os.getenv("IMAS_CODEX_GRAPH_SCHEDULER"):
        return env.lower()
    from imas_codex.graph.profiles import get_graph_location
    from imas_codex.remote.locations import resolve_location

    location = get_graph_location()
    if location == "local":
        return "none"
    return resolve_location(location).scheduler


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


def get_neo4j_version() -> str:
    """Get the Neo4j Docker image tag (e.g. ``"2026.01.4-community"``).

    Priority: NEO4J_VERSION env → [graph].neo4j-version → default.
    """
    if env := os.getenv("NEO4J_VERSION"):
        return env
    return _get_section("graph").get("neo4j-version", "2026.01.4-community")


def get_neo4j_image_path() -> Path:
    """Get the local Apptainer SIF image path for Neo4j."""
    return Path.home() / "apptainer" / f"neo4j_{get_neo4j_version()}.sif"


def get_neo4j_image_shell() -> str:
    """Get the shell-expandable Apptainer SIF image path for Neo4j.

    Uses ``$HOME`` so the path resolves correctly on remote nodes.
    """
    return f'"$HOME/apptainer/neo4j_{get_neo4j_version()}.sif"'


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
EMBEDDING_DIMENSION = get_embedding_dimension()
