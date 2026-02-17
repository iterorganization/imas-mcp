"""Neo4j connection profiles for server access and graph switching.

Two orthogonal concerns, each with its own authority:

- **location** (pyproject.toml) — where Neo4j physically runs.  Locations
  correspond to SSH aliases (``"iter"``, ``"tcv"``) or ``"local"``.
  Each location has a fixed port slot.  This is deploy-time config.

- **identity** (on-disk ``.neo4j/`` store) — which graph data directory
  to use.  Each graph lives in a hash-named directory in
  ``~/.local/share/imas-codex/.neo4j/<hash>/``.  The ``neo4j/`` symlink
  points to the active graph.  Switching graphs means repointing the
  symlink and restarting Neo4j.  Graph identity (name + facilities +
  hash) is stored both on disk (``.meta.json``) and in the database
  (``(:GraphMeta)`` node).

Port convention (both bolt and HTTP offset together)::

    iter   → bolt 7687, http 7474  (slot 0, Neo4j defaults)
    tcv    → bolt 7688, http 7475  (slot 1)
    jt60sa → bolt 7689, http 7476  (slot 2)
    jet    → bolt 7690, http 7477  (slot 3)
    ...

When connecting to a **remote** location, auto-tunnels are established with
a +10 000 offset to prevent port clashes::

    Direct (on the host):    bolt = 7687 + offset
    Tunneled (remote):       bolt = 17687 + offset

Server configuration lives in ``[tool.imas-codex.graph]`` in pyproject.toml::

    location = "iter"     # where Neo4j runs  (override: IMAS_CODEX_GRAPH_LOCATION)
    username = "neo4j"
    password = "imas-codex"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache for resolved URIs — avoids repeated tunnel checks and log spam.
# Key: (host, bolt_port), Value: resolved URI string.
_resolved_uri_cache: dict[tuple[str | None, int], str] = {}

# ─── Port convention ────────────────────────────────────────────────────────
# Port offsets and host defaults are managed exclusively in pyproject.toml:
#   [tool.imas-codex.graph].locations  — list of locations (index = port offset)
#   [tool.imas-codex.graph.hosts]      — SSH alias overrides (optional)

BOLT_BASE_PORT = 7687
HTTP_BASE_PORT = 7474

DEFAULT_LOCATION = "iter"
DEFAULT_USERNAME = "neo4j"
DEFAULT_PASSWORD = "imas-codex"

# Re-export from dirs for backward compatibility of import paths.
# Canonical source is imas_codex.graph.dirs.
from imas_codex.graph.dirs import ACTIVE_LINK, DATA_BASE_DIR  # noqa: E402, F401

BACKUPS_DIR = DATA_BASE_DIR / "backups"


def _get_all_offsets() -> dict[str, int]:
    """Return location→port-offset map from pyproject.toml ``[graph].locations``.

    Locations is a list where position encodes the port slot::

        locations = ["iter", "tcv", "jt60sa", ...]  # iter=0, tcv=1, ...
    """
    from imas_codex.settings import _get_section

    configured = _get_section("graph").get("locations", [])
    if isinstance(configured, list):
        return {name: i for i, name in enumerate(configured)}
    # Backward compat: dict form (name = offset)
    return {k: int(v) for k, v in configured.items()}


def _get_all_hosts() -> dict[str, str]:
    """Return location→SSH-alias map.

    Explicit entries from ``[graph.hosts]`` override, but any location
    in ``[graph.locations]`` implicitly uses its own name as SSH alias.
    """
    from imas_codex.settings import _get_section

    explicit = _get_section("graph").get("hosts", {})
    # Every location implicitly has host == name; explicit entries override
    hosts = {name: name for name in _get_all_offsets()}
    hosts.update(explicit)
    return hosts


# ─── Neo4jProfile ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Neo4jProfile:
    """Resolved connection parameters for a Neo4j instance.

    Separates two concerns:
    - **Server access** (location, host, uri, ports) — from pyproject.toml
    - **Data identity** (name, data_dir) — from the active ``.neo4j/`` symlink

    Attributes:
        name:      Graph data identity (e.g. ``"codex"``, ``"dev"``).
                   Read from ``.meta.json`` in the active graph directory.
        location:  Where Neo4j physically runs — location name from
                   the ``locations`` list, or ``"local"``.
        host:      SSH alias for the location (often same as location).
        uri:       Bolt URI (e.g. ``"bolt://localhost:7687"``).
        username:  Neo4j username.
        password:  Neo4j password.
        bolt_port: Bolt protocol port *at the host*.
        http_port: HTTP API port (health checks, browser).
        data_dir:  On-disk data directory (``neo4j/`` symlink).
    """

    name: str
    location: str
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
    from imas_codex.remote.tunnel import is_port_bound_by_ssh

    return is_port_bound_by_ssh(port)


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


# ─── Location helpers ───────────────────────────────────────────────────────


def get_graph_location() -> str:
    """Return where Neo4j runs (the location, not the graph name).

    Priority:
        1. ``IMAS_CODEX_GRAPH_LOCATION`` environment variable
        2. ``[tool.imas-codex.graph].location`` in pyproject.toml
        3. ``DEFAULT_LOCATION`` (``"iter"``)
    """
    if env := os.getenv("IMAS_CODEX_GRAPH_LOCATION"):
        return env

    from imas_codex.settings import _get_section

    graph_section = _get_section("graph")

    if location := graph_section.get("location"):
        return location

    return DEFAULT_LOCATION


def get_location_offset(location: str) -> int:
    """Get the port offset for a location (its index in the locations list)."""
    return _get_all_offsets().get(location, 0)


# ─── Resolution ─────────────────────────────────────────────────────────────


def get_active_graph_name() -> str:
    """Return the name of the currently active graph.

    Reads from the ``.meta.json`` file in the active graph directory
    (the ``neo4j/`` symlink target).  Falls back to ``"uninitialized"``
    if no active graph is configured.

    Returns:
        Graph name string.
    """
    from imas_codex.graph.dirs import get_active_graph

    active = get_active_graph()
    if active is not None:
        return active.name
    return "uninitialized"


def _resolve_uri(host: str | None, bolt_port: int) -> str:
    """Resolve the bolt URI based on host locality, with auto-tunneling.

    Results are cached per (host, bolt_port) pair so tunnel checks and
    log messages only fire once per process lifetime.

    1. If host is None / "local" / matches local machine → direct localhost
    2. If not local → check ``IMAS_CODEX_TUNNEL_BOLT_{HOST}`` env for override
    3. If no override → auto-tunnel with +10000 offset
    """
    cache_key = (host, bolt_port)
    if cache_key in _resolved_uri_cache:
        return _resolved_uri_cache[cache_key]

    uri = _resolve_uri_uncached(host, bolt_port)
    _resolved_uri_cache[cache_key] = uri
    return uri


def _resolve_uri_uncached(host: str | None, bolt_port: int) -> str:
    """Uncached URI resolution — called once per (host, bolt_port) pair."""
    from imas_codex.remote.executor import is_local_host

    if host is None or host == "local" or is_local_host(host):
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

    # Auto-tunnel: remote_port + TUNNEL_OFFSET
    from imas_codex.remote.tunnel import TUNNEL_OFFSET, ensure_tunnel

    tunnel_port_int = bolt_port + TUNNEL_OFFSET
    ok = ensure_tunnel(
        port=bolt_port,
        ssh_host=host,
        tunnel_port=tunnel_port_int,
    )
    if ok:
        logger.info(
            "Auto-tunnel: %s:%d → localhost:%d", host, bolt_port, tunnel_port_int
        )
        return f"bolt://localhost:{tunnel_port_int}"

    # Tunnel failed — fall back to same-port (user may have manual tunnel)
    logger.warning(
        "Auto-tunnel to %s:%d failed. Falling back to bolt://localhost:%d",
        host,
        bolt_port,
        bolt_port,
    )
    return f"bolt://localhost:{bolt_port}"


def resolve_neo4j(
    *,
    auto_tunnel: bool = True,
) -> Neo4jProfile:
    """Resolve Neo4j connection profile for the active graph.

    Server access (location → ports, host, URI) comes from pyproject.toml.
    Data identity (name, data_dir) comes from the active ``.neo4j/``
    symlink and its ``.meta.json``.

    Args:
        auto_tunnel: When True (default), automatically start an SSH tunnel
            if the location is remote.  Set to False when only port/host
            metadata is needed (e.g. tunnel CLI) to avoid side-effects.

    Returns:
        Fully resolved :class:`Neo4jProfile`.
    """
    name = get_active_graph_name()

    from imas_codex.settings import _get_section

    graph_section = _get_section("graph")

    # Server access — all from pyproject.toml [graph] section
    username = graph_section.get("username", DEFAULT_USERNAME)
    password = graph_section.get("password", DEFAULT_PASSWORD)
    location = get_graph_location()

    bolt_port = _convention_bolt_port(location)
    http_port = _convention_http_port(location)
    host = _get_all_hosts().get(location)

    # Location-aware URI resolution (with optional auto-tunneling)
    if auto_tunnel:
        uri = _resolve_uri(host, bolt_port)
    else:
        uri = f"bolt://localhost:{bolt_port}"

    # Env var escape hatches (always win)
    if env_uri := os.getenv("NEO4J_URI"):
        uri = env_uri
    if env_user := os.getenv("NEO4J_USERNAME"):
        username = env_user
    if env_pass := os.getenv("NEO4J_PASSWORD"):
        password = env_pass

    return Neo4jProfile(
        name=name,
        location=location,
        host=host,
        uri=uri,
        username=username,
        password=password,
        bolt_port=bolt_port,
        http_port=http_port,
        data_dir=ACTIVE_LINK,
    )


# ─── Convention helpers ─────────────────────────────────────────────────────


def _convention_bolt_port(location: str) -> int:
    """Bolt port for a known location, or base port for unknown."""
    return BOLT_BASE_PORT + _get_all_offsets().get(location, 0)


def _convention_http_port(location: str) -> int:
    """HTTP port for a known location, or base port for unknown."""
    return HTTP_BASE_PORT + _get_all_offsets().get(location, 0)


def list_profiles() -> list[Neo4jProfile]:
    """List Neo4j profiles for all known locations.

    Returns one profile per location defined in pyproject.toml.
    Constructs profiles directly (no auto-tunneling, no side-effects).
    """
    from imas_codex.settings import _get_section

    graph_section = _get_section("graph")
    username = graph_section.get("username", DEFAULT_USERNAME)
    password = graph_section.get("password", DEFAULT_PASSWORD)
    hosts = _get_all_hosts()
    name = get_active_graph_name()

    result = []
    for location in sorted(_get_all_offsets().keys()):
        bolt_port = _convention_bolt_port(location)
        http_port = _convention_http_port(location)
        host = hosts.get(location)
        result.append(
            Neo4jProfile(
                name=name,
                location=location,
                host=host,
                uri=f"bolt://localhost:{bolt_port}",
                username=username,
                password=password,
                bolt_port=bolt_port,
                http_port=http_port,
                data_dir=ACTIVE_LINK,
            )
        )
    return result
