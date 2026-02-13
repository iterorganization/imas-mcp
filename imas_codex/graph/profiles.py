"""Named graph profiles for multi-instance Neo4j management.

Each profile maps a human-readable name to a Neo4j instance with a specific
bolt port, HTTP port, data directory, and **host** — where Neo4j physically
runs.  Facility names (iter, tcv, jt60sa, etc.) map to ports and hosts by
convention — no config required.

The ``host`` field is an *absolute* property: ``"iter"`` means the ITER
login node regardless of whether you are on ITER or WSL.  At connection
time, ``is_local_host(host)`` (from ``imas_codex.remote.executor``)
determines whether to connect directly or via a tunnel.

Switching is done via ``IMAS_CODEX_GRAPH`` env var or
``[tool.imas-codex.graph].default`` in pyproject.toml.

Port convention (both bolt and HTTP offset together):
    iter   → bolt 7687, http 7474  (Neo4j defaults)
    tcv    → bolt 7688, http 7475
    jt60sa → bolt 7689, http 7476
    jet    → bolt 7690, http 7477
    west   → bolt 7691, http 7478
    mast-u → bolt 7692, http 7479
    asdex-u→ bolt 7693, http 7480
    east   → bolt 7694, http 7481
    diii-d → bolt 7695, http 7482
    kstar  → bolt 7696, http 7483

Custom profiles override convention via ``[tool.imas-codex.graph.profiles.*]``
in pyproject.toml.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Port convention ────────────────────────────────────────────────────────
# Ordered list — position determines port offset.  New facilities are appended
# at the end so existing port assignments are stable.

BOLT_BASE_PORT = 7687
HTTP_BASE_PORT = 7474

FACILITY_PORT_OFFSETS: dict[str, int] = {
    "iter": 0,
    "tcv": 1,
    "jt60sa": 2,
    "jet": 3,
    "west": 4,
    "mast-u": 5,
    "asdex-u": 6,
    "east": 7,
    "diii-d": 8,
    "kstar": 9,
}

# Convention host defaults: known facilities that run on a remote host.
# Profiles without a host entry (or not in this dict) default to None (local).
FACILITY_HOST_DEFAULTS: dict[str, str] = {
    "iter": "iter",
    "tcv": "tcv",
    "jt60sa": "jt60sa",
}

DEFAULT_PROFILE = "iter"
DEFAULT_USERNAME = "neo4j"
DEFAULT_PASSWORD = "imas-codex"
DATA_BASE_DIR = Path.home() / ".local" / "share" / "imas-codex"
BACKUPS_DIR = DATA_BASE_DIR / "backups"


# ─── GraphProfile ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraphProfile:
    """Resolved connection parameters for a named graph instance.

    Attributes:
        name:      Profile name (e.g. ``"iter"``, ``"tcv"``).
        host:      Where Neo4j physically runs — SSH alias, hostname,
                   or ``None`` meaning ``"local"``.
        uri:       Bolt URI (e.g. ``"bolt://localhost:7687"``).
        username:  Neo4j username.
        password:  Neo4j password.
        bolt_port: Bolt protocol port *at the host*.
        http_port: HTTP API port (health checks, browser).
        data_dir:  On-disk data directory for this instance.
    """

    name: str
    host: str | None
    uri: str
    username: str
    password: str
    bolt_port: int
    http_port: int
    data_dir: Path


# ─── Tunnel conflict detection ──────────────────────────────────────────────


def is_port_bound_by_tunnel(port: int = 7687) -> bool:
    """Check if a port is bound by an SSH tunnel (not a local Neo4j).

    This detects the case where an SSH tunnel is forwarding a remote Neo4j
    to localhost, which would conflict with starting a local Neo4j instance.
    """
    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "ssh" in line.lower():
                return True
        return False
    except Exception:
        return False


def check_graph_conflict(bolt_port: int = 7687) -> str | None:
    """Check for conflicting graph configurations.

    Returns an error message if a conflict is detected, None otherwise.
    """
    if is_port_bound_by_tunnel(bolt_port):
        return (
            f"Port {bolt_port} is bound by an SSH tunnel.\n"
            "Starting a local Neo4j would create two separate graphs.\n"
            "\n"
            "To use the remote graph via tunnel:\n"
            "  - Don't start local Neo4j\n"
            "  - Ensure SSH tunnel is active: ssh -f -N iter\n"
            "\n"
            "To use a local graph instead:\n"
            "  - Kill the SSH tunnel: ssh -O exit iter\n"
            "  - Then start local Neo4j"
        )
    return None


# ─── Resolution ─────────────────────────────────────────────────────────────


def get_active_graph_name() -> str:
    """Return the name of the currently active graph profile.

    Priority:
        1. ``IMAS_CODEX_GRAPH`` environment variable
        2. ``[tool.imas-codex.graph].default`` in pyproject.toml
        3. ``"iter"`` built-in default
    """
    if env := os.getenv("IMAS_CODEX_GRAPH"):
        return env

    from imas_codex.settings import _get_section

    return _get_section("graph").get("default", DEFAULT_PROFILE)


def _resolve_uri(host: str | None, bolt_port: int) -> str:
    """Resolve the bolt URI based on host locality.

    1. If host is None / "local" / matches local machine → direct localhost
    2. If not local → check ``IMAS_CODEX_TUNNEL_BOLT_{HOST}`` env for override
    3. If no override → ``bolt://localhost:{bolt_port}`` (assume same-port tunnel)
    """
    from imas_codex.remote.executor import is_local_host

    if host is None or is_local_host(host):
        return f"bolt://localhost:{bolt_port}"

    # Non-local: check tunnel port override
    env_key = f"IMAS_CODEX_TUNNEL_BOLT_{host.upper().replace('-', '_')}"
    if tunnel_port := os.getenv(env_key):
        logger.debug(
            "Using tunnel port override %s=%s for remote host %s",
            env_key,
            tunnel_port,
            host,
        )
        return f"bolt://localhost:{tunnel_port}"

    # No override — assume same-port tunnel
    logger.debug(
        "Remote host %s: using bolt://localhost:%d (assume same-port tunnel)",
        host,
        bolt_port,
    )
    return f"bolt://localhost:{bolt_port}"


def resolve_graph(name: str | None = None) -> GraphProfile:
    """Resolve a graph profile by name.

    Resolution order for connection parameters:
        1. ``NEO4J_URI`` / ``NEO4J_USERNAME`` / ``NEO4J_PASSWORD`` env vars
           (escape-hatch overrides, applied last)
        2. Explicit ``[tool.imas-codex.graph.profiles.<name>]`` in pyproject.toml
        3. Convention-based port mapping for known facility names

    The ``host`` field is resolved from:
        - Explicit ``host`` in profile config
        - ``FACILITY_HOST_DEFAULTS`` for known facilities
        - ``None`` (local) for everything else

    The bolt URI is then derived via :func:`_resolve_uri`, which uses
    ``is_local_host()`` to determine direct vs tunnel access.

    Args:
        name: Profile name.  ``None`` resolves via :func:`get_active_graph_name`.

    Returns:
        Fully resolved :class:`GraphProfile`.

    Raises:
        ValueError: If *name* is not a known facility and has no explicit
            profile configuration.
    """
    if name is None:
        name = get_active_graph_name()

    from imas_codex.settings import _get_section

    graph_section = _get_section("graph")

    # Shared defaults (top-level [graph] section)
    shared_username = graph_section.get("username", DEFAULT_USERNAME)
    shared_password = graph_section.get("password", DEFAULT_PASSWORD)

    # Check for explicit profile override
    profiles = graph_section.get("profiles", {})
    explicit = profiles.get(name, {})

    if explicit:
        # Explicit profile — use its values, fall back to shared/convention
        bolt_port = explicit.get("bolt-port", _convention_bolt_port(name))
        http_port = explicit.get("http-port", _convention_http_port(name))
        host = explicit.get("host", FACILITY_HOST_DEFAULTS.get(name))
        username = explicit.get("username", shared_username)
        password = explicit.get("password", shared_password)
        data_dir = Path(explicit.get("data-dir", str(_convention_data_dir(name))))
    elif name in FACILITY_PORT_OFFSETS:
        # Convention-based profile
        bolt_port = _convention_bolt_port(name)
        http_port = _convention_http_port(name)
        host = FACILITY_HOST_DEFAULTS.get(name)
        username = shared_username
        password = shared_password
        data_dir = _convention_data_dir(name)
    else:
        msg = (
            f"Unknown graph profile '{name}'. "
            f"Known facilities: {', '.join(sorted(FACILITY_PORT_OFFSETS))}. "
            f"Define a custom profile in pyproject.toml: "
            f"[tool.imas-codex.graph.profiles.{name}]"
        )
        raise ValueError(msg)

    # Location-aware URI resolution
    uri = _resolve_uri(host, bolt_port)

    # Env var escape hatches (always win)
    if env_uri := os.getenv("NEO4J_URI"):
        uri = env_uri
    if env_user := os.getenv("NEO4J_USERNAME"):
        username = env_user
    if env_pass := os.getenv("NEO4J_PASSWORD"):
        password = env_pass

    return GraphProfile(
        name=name,
        host=host,
        uri=uri,
        username=username,
        password=password,
        bolt_port=bolt_port,
        http_port=http_port,
        data_dir=data_dir,
    )


# ─── Convention helpers ─────────────────────────────────────────────────────


def _convention_bolt_port(name: str) -> int:
    """Bolt port for a known facility name, or base port for unknown."""
    return BOLT_BASE_PORT + FACILITY_PORT_OFFSETS.get(name, 0)


def _convention_http_port(name: str) -> int:
    """HTTP port for a known facility name, or base port for unknown."""
    return HTTP_BASE_PORT + FACILITY_PORT_OFFSETS.get(name, 0)


def _convention_data_dir(name: str) -> Path:
    """Data directory for a named graph instance.

    Default (iter) uses ``neo4j/``, others use ``neo4j-{name}/``.
    """
    if name == DEFAULT_PROFILE:
        return DATA_BASE_DIR / "neo4j"
    return DATA_BASE_DIR / f"neo4j-{name}"


def list_profiles() -> list[GraphProfile]:
    """List all available graph profiles (convention + explicit).

    Returns profiles for all known facilities plus any explicit profiles
    defined in pyproject.toml.
    """
    from imas_codex.settings import _get_section

    graph_section = _get_section("graph")
    explicit_names = set(graph_section.get("profiles", {}).keys())
    all_names = set(FACILITY_PORT_OFFSETS.keys()) | explicit_names

    result = []
    for name in sorted(all_names):
        try:
            result.append(resolve_graph(name))
        except ValueError:
            continue
    return result
