"""Named graph profiles for multi-instance Neo4j management.

Two orthogonal concerns:

- **name** — what data lives in the graph (``"codex"``, ``"tcv-only"``,
  ``"sandbox"``).  The default graph ``"codex"`` contains all facilities
  plus the IMAS data dictionary.
- **location** — where Neo4j physically runs.  Locations correspond to
  SSH aliases (``"iter"``, ``"tcv"``) or ``"local"`` for a machine-local
  instance.  Each location has a fixed port slot.

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

Configuration lives in ``[tool.imas-codex.graph]`` in pyproject.toml::

    name = "codex"        # graph identity  (override: IMAS_CODEX_GRAPH)
    location = "iter"     # where it runs   (override: IMAS_CODEX_GRAPH_LOCATION)

Custom profiles override convention via ``[tool.imas-codex.graph.profiles.*]``.
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

DEFAULT_NAME = "codex"
DEFAULT_LOCATION = "iter"
DEFAULT_USERNAME = "neo4j"
DEFAULT_PASSWORD = "imas-codex"
DATA_BASE_DIR = Path.home() / ".local" / "share" / "imas-codex"
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


# ─── GraphProfile ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraphProfile:
    """Resolved connection parameters for a named graph instance.

    Attributes:
        name:      Graph identity (e.g. ``"codex"``, ``"tcv-only"``).
        location:  Where Neo4j physically runs — location name from
                   the ``locations`` list, or ``"local"``.
        host:      SSH alias for the location (often same as location).
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

    @property
    def location(self) -> str:
        """Where Neo4j runs — SSH alias or ``"local"``."""
        return self.host or "local"


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

    Priority:
        1. ``IMAS_CODEX_GRAPH`` environment variable
        2. ``[tool.imas-codex.graph].name`` in pyproject.toml
        3. ``DEFAULT_NAME`` (``"codex"``)
    """
    if env := os.getenv("IMAS_CODEX_GRAPH"):
        return env

    from imas_codex.settings import _get_section

    graph_section = _get_section("graph")
    return graph_section.get("name") or DEFAULT_NAME


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


def resolve_graph(
    name: str | None = None,
    *,
    auto_tunnel: bool = True,
) -> GraphProfile:
    """Resolve a graph profile by name.

    Resolution order for connection parameters:
        1. ``NEO4J_URI`` / ``NEO4J_USERNAME`` / ``NEO4J_PASSWORD`` env vars
           (escape-hatch overrides, applied last)
        2. Explicit ``[tool.imas-codex.graph.profiles.<name>]`` in pyproject.toml
        3. If name matches a known location → ports from that location slot
        4. Otherwise → ports from :func:`get_graph_location`

    **Name vs location**: The *name* is the graph identity (what data is
    inside — ``"codex"`` is all facilities + IMAS).  The *location*
    determines where Neo4j runs and therefore which ports to use.

    Args:
        name: Graph name.  ``None`` resolves via :func:`get_active_graph_name`.
        auto_tunnel: When True (default), automatically start an SSH tunnel
            if the location is remote.  Set to False when only port/host
            metadata is needed (e.g. tunnel CLI) to avoid side-effects.

    Returns:
        Fully resolved :class:`GraphProfile`.
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
        # Explicit profile — uses its own location/port settings
        location = explicit.get("location", get_graph_location())
        bolt_port = explicit.get("bolt-port", _convention_bolt_port(location))
        http_port = explicit.get("http-port", _convention_http_port(location))
        host = _get_all_hosts().get(location)
        username = explicit.get("username", shared_username)
        password = explicit.get("password", shared_password)
        data_dir = Path(explicit.get("data-dir", str(_convention_data_dir(name))))
    elif name in _get_all_offsets():
        # Name matches a known location — use that location's port slot
        location = name
        bolt_port = _convention_bolt_port(location)
        http_port = _convention_http_port(location)
        host = _get_all_hosts().get(location)
        username = shared_username
        password = shared_password
        data_dir = _convention_data_dir(name)
    else:
        # Name is not a location (e.g. "codex") — resolve via location config
        location = get_graph_location()
        bolt_port = _convention_bolt_port(location)
        http_port = _convention_http_port(location)
        host = _get_all_hosts().get(location)
        username = shared_username
        password = shared_password
        data_dir = _convention_data_dir(name)

    # Location override: IMAS_CODEX_GRAPH_LOCATION changes where we connect
    if loc_override := os.getenv("IMAS_CODEX_GRAPH_LOCATION"):
        if loc_override == "local":
            host = None
        else:
            location = loc_override
            host = _get_all_hosts().get(loc_override, loc_override)
        bolt_port = _convention_bolt_port(location)
        http_port = _convention_http_port(location)

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


def _convention_bolt_port(location: str) -> int:
    """Bolt port for a known location, or base port for unknown."""
    return BOLT_BASE_PORT + _get_all_offsets().get(location, 0)


def _convention_http_port(location: str) -> int:
    """HTTP port for a known location, or base port for unknown."""
    return HTTP_BASE_PORT + _get_all_offsets().get(location, 0)


def _convention_data_dir(name: str) -> Path:
    """Data directory for a named graph instance.

    The default graph (``"codex"``) uses ``neo4j/``, others use ``neo4j-{name}/``.
    """
    if name == DEFAULT_NAME:
        return DATA_BASE_DIR / "neo4j"
    return DATA_BASE_DIR / f"neo4j-{name}"


def list_profiles() -> list[GraphProfile]:
    """List all available graph profiles (convention + explicit).

    Returns profiles for all known facilities plus any explicit profiles
    defined in pyproject.toml.  Uses ``auto_tunnel=False`` to avoid
    side-effects (tunnel starts) when just listing metadata.
    """
    from imas_codex.settings import _get_section

    graph_section = _get_section("graph")
    explicit_names = set(graph_section.get("profiles", {}).keys())
    all_names = set(_get_all_offsets().keys()) | explicit_names

    result = []
    for name in sorted(all_names):
        try:
            result.append(resolve_graph(name, auto_tunnel=False))
        except ValueError:
            continue
    return result
