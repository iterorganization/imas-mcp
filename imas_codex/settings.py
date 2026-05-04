"""Project settings loaded from pyproject.toml [tool.imas-codex] section.

Configuration is organized into subsections:
  [tool.imas-codex]                — general settings
  [tool.imas-codex.graph]          — Neo4j graph URI, username, password
  [tool.imas-codex.data-dictionary] — DD version, include-ggd, include-error-fields
  [tool.imas-codex.embedding]      — embedding model, dimension, location
  [tool.imas-codex.language]       — language models, batch-size for structured output
  [tool.imas-codex.dd-enrichment]  — model for DD path enrichment/refinement
  [tool.imas-codex.vision]         — vision models for image/document tasks
  [tool.imas-codex.agent]          — agent models for planning/exploration tasks
  [tool.imas-codex.compaction]     — compaction models for summarization tasks
  [tool.imas-codex.sn.review]       — shared disagreement threshold and max-cycles
  [tool.imas-codex.sn.review.names] — name-axis reviewer model chain (primary/secondary/escalator)
  [tool.imas-codex.sn.review.docs]  — docs-axis reviewer model chain (primary/secondary/escalator)
  [tool.imas-codex.sn.benchmark]   — SN benchmark compose-models and reviewer-model

All settings support environment variable overrides (IMAS_CODEX_* prefix / NEO4J_*).
"""

import importlib.resources
import os
from functools import cache
from pathlib import Path
from typing import Any

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

MODEL_SECTIONS = frozenset(
    {
        "embedding",
        "language",
        "vision",
        "agent",
        "compaction",
        "reasoning",
        "dd-enrichment",
        "sn-run",
        "sn-enrich",
        "refine",
    }
)

# Default model per section (fallback when not configured)
_MODEL_DEFAULTS: dict[str, str] = {
    "embedding": "Qwen/Qwen3-Embedding-0.6B",
    "language": "google/gemini-3.1-flash-lite-preview",
    "vision": "google/gemini-3.1-flash-lite-preview",
    "agent": "openrouter/anthropic/claude-sonnet-4.6",
    "compaction": "openrouter/anthropic/claude-haiku-4.5",
    "reasoning": "openrouter/anthropic/claude-sonnet-4.6",
    "dd-enrichment": "openrouter/anthropic/claude-sonnet-4.6",
    "sn-run": "openrouter/anthropic/claude-sonnet-4.6",
    "sn-enrich": "openrouter/anthropic/claude-opus-4.6",
    # Refine pass for SN names + docs.  E3 telemetry showed flash-lite
    # could not lift critiqued names (cl=0 accepted at ~42%, cl=1+ at
    # ~5%).  Sonnet 4.6 matches compose tier so the refine pass is
    # capable enough to recover from reviewer feedback.
    "refine": "openrouter/anthropic/claude-sonnet-4.6",
}

# Environment variable names per section
_MODEL_ENV_VARS: dict[str, str] = {
    "embedding": "IMAS_CODEX_EMBEDDING_MODEL",
    "language": "IMAS_CODEX_LANGUAGE_MODEL",
    "vision": "IMAS_CODEX_VISION_MODEL",
    "agent": "IMAS_CODEX_AGENT_MODEL",
    "compaction": "IMAS_CODEX_COMPACTION_MODEL",
    "reasoning": "IMAS_CODEX_REASONING_MODEL",
    "dd-enrichment": "IMAS_CODEX_DD_ENRICHMENT_MODEL",
    "sn-run": "IMAS_CODEX_SN_RUN_MODEL",
    "sn-enrich": "IMAS_CODEX_SN_ENRICH_MODEL",
    "refine": "IMAS_CODEX_REFINE_MODEL",
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
    return resolve_service_url(location, port, service_job_name="codex-embed")


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
POSTGRES_BASE_PORT = 18450


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


def get_postgres_port() -> int:
    """Get the PostgreSQL port for LiteLLM's database.

    Follows the same convention as other ports:
    ``pg_port = 18450 + facility_offset``.

    Priority: IMAS_CODEX_POSTGRES_PORT env → convention offset → 18450.
    """
    if env := os.getenv("IMAS_CODEX_POSTGRES_PORT"):
        return int(env)
    location = get_llm_location()
    if location == "local":
        return POSTGRES_BASE_PORT
    from imas_codex.remote.locations import get_port_offset

    return POSTGRES_BASE_PORT + get_port_offset(location)


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

    The LLM proxy always runs on the default SSH target node (no
    separate alias).  Off-facility access goes through the SSH tunnel
    at ``127.0.0.1``.  On-facility resolves to the current machine's
    hostname.
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

    # On-facility: LLM runs on the same node we're on
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


# ─── Discovery settings ─────────────────────────────────────────────────────


def get_discovery_threshold() -> float:
    """Get the minimum score threshold for high-value path processing.

    Used by enrichment auto-threshold, refinement gate, and code CLI default.

    Priority: IMAS_CODEX_DISCOVERY_THRESHOLD env → [discovery].threshold → 0.90.
    """
    if env := os.getenv("IMAS_CODEX_DISCOVERY_THRESHOLD"):
        return float(env)
    return float(_get_section("discovery").get("threshold", 0.90))


def get_triage_threshold() -> float:
    """Get the minimum triage composite for enrichment/scoring.

    Derived from the discovery threshold minus a configurable offset.
    Always lower than the ingestion threshold — files must pass this
    cheap gate before full LLM scoring.

    ``triage_threshold = discovery_threshold - triage_offset``

    Priority: IMAS_CODEX_TRIAGE_THRESHOLD env → computed from offset → 0.75.
    """
    if env := os.getenv("IMAS_CODEX_TRIAGE_THRESHOLD"):
        return float(env)
    offset = float(_get_section("discovery").get("triage-offset", 0.15))
    return get_discovery_threshold() - offset


# ─── Log settings ──────────────────────────────────────────────────────────


def get_log_location() -> str:
    """Get the log location — where CLI commands run and write logs.

    When set to a facility name (e.g. ``"iter"``), MCP log tools fetch
    logs via SSH from that host's ``~/.local/share/imas-codex/logs/``.
    ``"local"`` reads from the local filesystem (default).

    Priority: IMAS_CODEX_LOG_LOCATION env → [logs].location → 'local'.
    """
    if env := os.getenv("IMAS_CODEX_LOG_LOCATION"):
        return env.lower()
    location = _get_section("logs").get("location")
    return str(location).lower() if location else "local"


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


# ─── SN example-injection and retry tunables ───────────────────────────────

_SN_DEFAULTS: dict[str, Any] = {
    "example-target-scores": [1.0, 0.8, 0.65, 0.4],
    "example-tolerance": 0.05,
    "example-per-bucket": 1,
    "retry-attempts": 1,
    "retry-k-expansion": 12,
}


def get_sn_example_target_scores() -> tuple[float, ...]:
    """Score thresholds for selecting exemplar StandardName nodes.

    Each target defines a bucket; the example loader picks the closest
    reviewed StandardName whose ``reviewer_score`` falls within
    ``target ± tolerance``.

    Priority: IMAS_CODEX_SN_EXAMPLE_TARGET_SCORES env (comma-separated)
              → [sn].example-target-scores → ``(1.0, 0.8, 0.65, 0.4)``.
    """
    if env := os.getenv("IMAS_CODEX_SN_EXAMPLE_TARGET_SCORES"):
        return tuple(float(v) for v in env.split(",") if v.strip())
    section = _get_section("sn")
    raw = section.get("example-target-scores", _SN_DEFAULTS["example-target-scores"])
    return tuple(float(v) for v in raw)


def get_sn_example_tolerance() -> float:
    """Tolerance band around each target score for example selection.

    Priority: IMAS_CODEX_SN_EXAMPLE_TOLERANCE env
              → [sn].example-tolerance → ``0.05``.
    """
    if env := os.getenv("IMAS_CODEX_SN_EXAMPLE_TOLERANCE"):
        return float(env)
    section = _get_section("sn")
    return float(section.get("example-tolerance", _SN_DEFAULTS["example-tolerance"]))


def get_sn_example_per_bucket() -> int:
    """Maximum number of examples per score bucket.

    Priority: IMAS_CODEX_SN_EXAMPLE_PER_BUCKET env
              → [sn].example-per-bucket → ``1``.
    """
    if env := os.getenv("IMAS_CODEX_SN_EXAMPLE_PER_BUCKET"):
        return int(env)
    section = _get_section("sn")
    return int(section.get("example-per-bucket", _SN_DEFAULTS["example-per-bucket"]))


def get_sn_retry_attempts() -> int:
    """Max retry attempts on grammar/validation failure during compose.

    Priority: IMAS_CODEX_SN_RETRY_ATTEMPTS env
              → [sn].retry-attempts → ``1``.
    """
    if env := os.getenv("IMAS_CODEX_SN_RETRY_ATTEMPTS"):
        return int(env)
    section = _get_section("sn")
    return int(section.get("retry-attempts", _SN_DEFAULTS["retry-attempts"]))


def get_sn_retry_k_expansion() -> int:
    """Hybrid-search k expansion factor used on compose retry.

    On retry, the compose worker re-enriches items with an expanded
    hybrid DD search using ``search_k=retry_k_expansion``.

    Priority: IMAS_CODEX_SN_RETRY_K_EXPANSION env
              → [sn].retry-k-expansion → ``12``.
    """
    if env := os.getenv("IMAS_CODEX_SN_RETRY_K_EXPANSION"):
        return int(env)
    section = _get_section("sn")
    return int(section.get("retry-k-expansion", _SN_DEFAULTS["retry-k-expansion"]))


def get_sn_staging_dir() -> Path:
    """Default staging directory for sn export/preview/publish.

    Resolution order: ``IMAS_CODEX_SN_STAGING`` env var →
    ``[tool.imas-codex.sn].staging-dir`` config → ``~/.cache/imas-codex/staging``.
    """
    env = os.environ.get("IMAS_CODEX_SN_STAGING")
    if env:
        return Path(env).expanduser()
    cfg = _get_section("sn").get("staging-dir", "~/.cache/imas-codex/staging")
    return Path(cfg).expanduser()


def get_sn_isnc_dir() -> Path | None:
    """Path to ISNC (imas-standard-names-catalog) git checkout.

    Resolution order: ``IMAS_CODEX_SN_ISNC`` env var →
    ``[tool.imas-codex.sn].isnc-dir`` config → sibling auto-discovery → None.

    Auto-discovery scans sibling directories of the project root for
    directories matching ``*standard-names-catalog*``. An exact match on
    ``imas-standard-names-catalog`` wins. Multiple ambiguous matches
    return None (with a logged warning).
    """
    import logging

    env = os.environ.get("IMAS_CODEX_SN_ISNC")
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None

    cfg = _get_section("sn").get("isnc-dir", "")
    if cfg:
        p = Path(cfg).expanduser()
        return p if p.is_dir() else None

    # Auto-discover from sibling directories of the project root
    project_root = Path(__file__).resolve().parent.parent
    parent = project_root.parent
    if not parent.is_dir():
        return None

    exact = parent / "imas-standard-names-catalog"
    if exact.is_dir():
        return exact

    candidates = [
        d
        for d in parent.iterdir()
        if d.is_dir() and "standard-names-catalog" in d.name and d != project_root
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Multiple ISNC candidates found: %s. Set IMAS_CODEX_SN_ISNC or "
            "[tool.imas-codex.sn].isnc-dir to resolve.",
            ", ".join(str(c) for c in candidates),
        )
    return None


def get_compose_lean() -> bool:
    """Whether to use the lean compose system prompt (≤8K tokens).

    When True, workers.compose_worker selects ``sn/generate_name_system_lean``
    instead of the full 39K-token ``sn/generate_name_system`` template, and caps
    ``existing_names`` at 50 (vs 200).  Reduces per-batch cost ~4× (Phase A
    of plan 43 — see plans/features/standard-names/43-pipeline-rd-fix.md).

    Default: False — the legacy 39K prompt is preserved for production until
    Phase B (budget rebalance) and Phase E (probe loop) confirm the lean
    prompt meets quality thresholds.

    Priority: IMAS_CODEX_SN_COMPOSE_LEAN env (``1``/``true`` → True)
              → [sn-compose].lean-prompt (bool) → False.
    """
    if env := os.getenv("IMAS_CODEX_SN_COMPOSE_LEAN"):
        return env.lower() in ("1", "true", "yes")
    return bool(_get_section("sn-compose").get("lean-prompt", False))


def get_compose_concurrency() -> int:
    """Maximum concurrent LLM requests for the SN compose worker.

    Sized from the 2026-04-24 OpenRouter rate-limit probe: Anthropic
    Opus/Sonnet/Haiku all handled N=32 concurrent requests with zero 429s.
    The default of 24 applies 75 % headroom against the measured floor
    (0.75 × 32 = 24).  See ``docs/ops/openrouter-rate-ceilings.md``.

    Priority: IMAS_CODEX_SN_COMPOSE_MAX_CONCURRENCY env
              → [sn-compose].max-concurrency → ``24``.
    """
    if env := os.getenv("IMAS_CODEX_SN_COMPOSE_MAX_CONCURRENCY"):
        return int(env)
    return int(_get_section("sn-compose").get("max-concurrency", 24))


# ─── SN review settings ────────────────────────────────────────────────────

_SN_REVIEW_DEFAULTS: dict[str, Any] = {
    "names-models": ["openrouter/anthropic/claude-opus-4.6"],
    "docs-models": ["openrouter/anthropic/claude-opus-4.6"],
    "disagreement-threshold": 0.15,
    "max-cycles": 3,
}


def _validate_review_models(models: list[str], axis: str) -> list[str]:
    """Validate a review model list for the given axis.

    Args:
        models: Raw list of model strings from config.
        axis: Axis name ("names" or "docs") used in error messages.

    Returns:
        Validated list of model strings.

    Raises:
        ValueError: If the list length is 0 or >3, or any entry is empty.
    """
    import logging as _logging

    if len(models) == 0:
        raise ValueError(
            f"[sn.review.{axis}].models must have at least 1 entry; got 0. "
            f"Set 1 model to disable quorum, 2 for blind pair, 3 for full RD-quorum."
        )
    if len(models) > 3:
        raise ValueError(
            f"[sn.review.{axis}].models accepts at most 3 entries "
            f"(primary, secondary, escalator); got {len(models)}."
        )
    validated: list[str] = []
    for m in models:
        if not isinstance(m, str) or not m.strip():
            raise ValueError(
                f"[sn.review.{axis}].models entries must be non-empty strings; "
                f"got {m!r}."
            )
        if not m.startswith("openrouter/"):
            _logging.getLogger(__name__).warning(
                "[sn.review.%s].models entry %r does not have the 'openrouter/' prefix; "
                "prompt caching will not be available for this model.",
                axis,
                m,
            )
        validated.append(m)
    return validated


_VALID_REVIEWER_PROFILES: frozenset[str] = frozenset(
    {"default", "quality-cost-balanced", "opus-only"}
)


def get_sn_review_active_profile() -> str:
    """Return the active reviewer profile name.

    Resolution order:
      1. ``IMAS_CODEX_SN_REVIEW_PROFILE`` environment variable.
      2. Hard-coded default: ``"default"``.

    Valid profile names: ``"default"``, ``"quality-cost-balanced"``,
    ``"opus-only"``. (Reviewer floor is Sonnet 4.6 — no Haiku profiles.)

    Returns:
        Profile name string (not validated here — validation happens in
        :func:`get_sn_review_profile_models`).
    """
    import os as _os

    return _os.environ.get("IMAS_CODEX_SN_REVIEW_PROFILE", "default")


def get_sn_review_profile_models(profile: str) -> list[str]:
    """Return the ordered reviewer-model chain for *profile*.

    Reads from ``[tool.imas-codex.sn.review.names.profiles.<profile>].models``.
    For ``"default"``, falls back to the top-level ``[sn.review.names].models``
    key when no ``profiles`` section is present (backward-compat).

    Length semantics (same as :func:`get_sn_review_names_models`):
      * 1 model  → quorum disabled
      * 2 models → blind pair, no escalator
      * 3 models → full RD-quorum: primary, secondary, escalator
      * 4+       → rejected (``ValueError``)

    Args:
        profile: Profile name — one of ``"default"``,
            ``"quality-cost-balanced"``, ``"opus-only"``.

    Raises:
        ValueError: If *profile* is not in :data:`_VALID_REVIEWER_PROFILES`
            or the model list fails validation.
    """
    names_section = _get_section("sn").get("review", {}).get("names", {})
    profiles = names_section.get("profiles", {})

    if profile in profiles:
        raw = profiles[profile].get("models", [])
        return _validate_review_models(
            [str(m) for m in raw if m], f"names.profiles.{profile}"
        )
    if profile == "default":
        # Backward-compat: no profiles section → read top-level models key.
        raw = names_section.get("models", _SN_REVIEW_DEFAULTS["names-models"])
        return _validate_review_models([str(m) for m in raw if m], "names")
    raise ValueError(
        f"Unknown reviewer profile {profile!r}. "
        f"Valid profiles: {sorted(_VALID_REVIEWER_PROFILES)}. "
        f"Configure under [tool.imas-codex.sn.review.names.profiles]."
    )


def get_sn_review_profile_threshold(profile: str) -> float:
    """Return the disagreement threshold for *profile*.

    Reads from
    ``[tool.imas-codex.sn.review.names.profiles.<profile>].disagreement-threshold``.
    For ``"default"``, falls back to ``[sn.review].disagreement-threshold``
    when no ``profiles`` section is present (backward-compat).

    Args:
        profile: Profile name.

    Raises:
        ValueError: If *profile* is not in :data:`_VALID_REVIEWER_PROFILES`.
    """
    names_section = _get_section("sn").get("review", {}).get("names", {})
    profiles = names_section.get("profiles", {})
    review_section = _get_section("sn").get("review", {})

    if profile in profiles:
        return float(
            profiles[profile].get(
                "disagreement-threshold",
                _SN_REVIEW_DEFAULTS["disagreement-threshold"],
            )
        )
    if profile == "default":
        # Backward-compat: no profiles section → read shared review setting.
        return float(
            review_section.get(
                "disagreement-threshold",
                _SN_REVIEW_DEFAULTS["disagreement-threshold"],
            )
        )
    raise ValueError(
        f"Unknown reviewer profile {profile!r}. "
        f"Valid profiles: {sorted(_VALID_REVIEWER_PROFILES)}."
    )


def get_sn_review_names_models() -> list[str]:
    """Return the ordered reviewer-model chain for the names review axis.

    Delegates to :func:`get_sn_review_profile_models` using the active
    profile (see :func:`get_sn_review_active_profile`).  When no profile
    is active and no ``profiles`` section exists in config, falls back to
    the top-level ``[sn.review.names].models`` key (backward-compat).

    Length semantics:
      * 1 model  → quorum disabled (single reviewer, mirrors legacy behaviour)
      * 2 models → blind primary + blind secondary, no escalator
      * 3 models → full RD-quorum: [0] primary (blind), [1] secondary (blind),
                   [2] escalator (sees both reviews, authoritative)
      * 4+       → rejected at config load time (``ValueError``)

    Raises:
        ValueError: If list is empty or has more than 3 entries.
    """
    return get_sn_review_profile_models(get_sn_review_active_profile())


def get_sn_review_docs_models() -> list[str]:
    """Return the ordered reviewer-model chain for the docs review axis.

    Same length semantics as :func:`get_sn_review_names_models`.
    Reads from ``[sn.review.docs].models`` (docs axis has no profile system;
    use ``--models`` CLI override for ad-hoc docs model changes).

    Priority: ``[sn.review.docs].models`` in pyproject.toml → default
    (a single canonical model).

    Raises:
        ValueError: If list is empty or has more than 3 entries.
    """
    section = _get_section("sn").get("review", {}).get("docs", {})
    raw = section.get("models", _SN_REVIEW_DEFAULTS["docs-models"])
    return _validate_review_models([str(m) for m in raw if m], "docs")


def get_sn_review_max_cycles() -> int:
    """Get the maximum number of RD-quorum review cycles.

    1 → primary only, 2 → blind pair (no escalator), 3 → full quorum.

    Priority: ``[sn.review].max-cycles`` → ``3``.
    """
    section = _get_section("sn").get("review", {})
    return int(section.get("max-cycles", _SN_REVIEW_DEFAULTS["max-cycles"]))


def get_sn_review_disagreement_threshold() -> float:
    """Get the spread threshold that flags review disagreement.

    Delegates to :func:`get_sn_review_profile_threshold` using the active
    profile (see :func:`get_sn_review_active_profile`).  Falls back to the
    top-level ``[sn.review].disagreement-threshold`` key when no profile is
    active (backward-compat).

    When N >= 2 reviewers are configured, ``review_disagreement`` is
    set ``true`` if ``max(scores) - min(scores) >= threshold``.
    """
    return get_sn_review_profile_threshold(get_sn_review_active_profile())


# ─── SN benchmark settings ─────────────────────────────────────────────────

_SN_BENCHMARK_DEFAULTS = {
    "compose-models": [
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-haiku-4.5",
        "openai/gpt-5.4",
        "openai/gpt-5.4-mini",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3-flash-preview",
        "google/gemini-3.1-flash-lite-preview",
    ],
    "reviewer-model": "anthropic/claude-opus-4.6",
}


def get_sn_benchmark_compose_models() -> list[str]:
    """Get list of models for SN benchmark composition.

    Priority: [sn.benchmark].compose-models in pyproject.toml → defaults.
    """
    section = _get_section("sn").get("benchmark", {})
    return section.get("compose-models", _SN_BENCHMARK_DEFAULTS["compose-models"])


def get_sn_benchmark_reviewer_model() -> str:
    """Reviewer model for SN benchmark scoring.

    Priority: ``[sn.benchmark].reviewer-model`` →
    ``[sn.review.names].models[0]`` → default.
    """
    section = _get_section("sn").get("benchmark", {})
    try:
        review_models = get_sn_review_names_models()
        fallback = (
            review_models[0]
            if review_models
            else _SN_BENCHMARK_DEFAULTS["reviewer-model"]
        )
    except (ValueError, IndexError):
        fallback = _SN_BENCHMARK_DEFAULTS["reviewer-model"]
    return section.get("reviewer-model", fallback)
