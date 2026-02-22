"""Unified service location resolution.

Maps pyproject.toml location strings (``"iter"``, ``"titan"``, ``"local"``)
to resolved service endpoints. This is the single source of truth for how
to reach any remote service (Neo4j, embedding, LLM proxy).

A location can be:

- **``"local"``** — service runs in-process or on localhost.
- **A facility name** (``"iter"``, ``"tcv"``) — service runs on that facility's
  login node.  Reached directly when on-site, via SSH tunnel when remote.
- **A compute location** (``"titan"``) — service runs on a SLURM compute node
  within a facility.  The mapping from compute location to facility is
  defined in the facility's public YAML under ``compute_locations``.

Resolution is cached and shared across all services (graph, embed, LLM).
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from functools import cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocationInfo:
    """Resolved service location metadata.

    Attributes:
        name: The location as specified in config (e.g. ``"titan"``).
        facility: The facility this location belongs to (e.g. ``"iter"``).
            Same as ``name`` for direct facility locations.
        ssh_host: SSH alias for reaching the facility (e.g. ``"iter"``).
        scheduler: Job scheduler type (``"slurm"`` or ``"none"``).
        partition: SLURM partition name, if scheduler is slurm.
        service_job_name: SLURM job name for service discovery.
        is_compute: True when this is a SLURM compute location.
    """

    name: str
    facility: str
    ssh_host: str
    scheduler: str = "none"
    partition: str | None = None
    service_job_name: str = "imas-codex-services"
    is_compute: bool = False

    @property
    def display_label(self) -> str:
        """Human-readable label for status displays."""
        return self.name


@cache
def resolve_location(location: str) -> LocationInfo:
    """Resolve a location string to full service metadata.

    Resolution order:

    1. ``"local"`` → local LocationInfo (no facility).
    2. Check all facility configs for ``compute_locations.{location}`` —
       if found, this is a compute sub-location within that facility.
    3. Otherwise treat as a direct facility location (login node).

    Args:
        location: Location string from pyproject.toml (e.g. ``"titan"``).

    Returns:
        Resolved :class:`LocationInfo`.
    """
    if location == "local":
        return LocationInfo(
            name="local",
            facility="local",
            ssh_host="local",
        )

    # Check if this is a compute location defined in any facility config
    compute_info = _find_compute_location(location)
    if compute_info is not None:
        return compute_info

    # Direct facility location (service on login node)
    ssh_host = _get_ssh_host(location)
    return LocationInfo(
        name=location,
        facility=location,
        ssh_host=ssh_host,
    )


def _find_compute_location(name: str) -> LocationInfo | None:
    """Search facility configs for a compute location definition.

    Looks for ``compute_locations.{name}`` in each facility's public YAML.
    """
    try:
        from imas_codex.discovery.base.facility import get_facility, list_facilities
    except Exception:
        return None

    for facility_id in list_facilities():
        try:
            cfg = get_facility(facility_id)
        except Exception:
            continue
        compute_locs = cfg.get("compute_locations", {})
        if name in compute_locs:
            loc_cfg = compute_locs[name]
            ssh_host = cfg.get("ssh_host", facility_id)
            return LocationInfo(
                name=name,
                facility=facility_id,
                ssh_host=ssh_host,
                scheduler=loc_cfg.get("scheduler", "slurm"),
                partition=loc_cfg.get("partition", name),
                service_job_name=loc_cfg.get("service_job_name", "imas-codex-services"),
                is_compute=True,
            )
    return None


def _get_ssh_host(location: str) -> str:
    """Get SSH host alias for a facility location.

    Checks ``[tool.imas-codex.hosts]`` for explicit overrides,
    otherwise uses the location name as the SSH alias.
    """
    try:
        from imas_codex.settings import _load_pyproject_settings

        hosts = _load_pyproject_settings().get("hosts", {})
        return hosts.get(location, location)
    except Exception:
        return location


def get_port_offset(location: str) -> int:
    """Get the port offset for a location.

    For compute locations, uses the parent facility's offset.
    For direct facility locations, uses their position in the locations list.
    """
    info = resolve_location(location)
    facility = info.facility
    if facility == "local":
        return 0
    try:
        from imas_codex.settings import _load_pyproject_settings

        locations_list = _load_pyproject_settings().get("locations", [])
        if isinstance(locations_list, list):
            offsets = {name: i for i, name in enumerate(locations_list)}
        else:
            offsets = {k: int(v) for k, v in locations_list.items()}
        return offsets.get(facility, 0)
    except Exception:
        return 0


def is_location_local(location: str) -> bool:
    """Check if a location resolves to the current machine.

    Returns True when:
    - location is ``"local"``
    - location names a facility/compute location and we're on that facility
    """
    if location == "local":
        return True
    info = resolve_location(location)
    try:
        from imas_codex.remote.executor import is_local_host

        return is_local_host(info.ssh_host)
    except Exception:
        return False


def resolve_service_url(
    location: str,
    port: int,
    *,
    protocol: str = "http",
) -> str | None:
    """Resolve the URL to reach a service at the given location.

    Handles three connectivity modes:

    1. **Local, no scheduler** — ``{protocol}://localhost:{port}``
    2. **Local + SLURM** — discover compute node via ``squeue``,
       connect directly or via localhost if we're on that node.
    3. **Remote** — ``{protocol}://localhost:{port}`` (via SSH tunnel)

    Args:
        location: Location string (e.g. ``"titan"``, ``"iter"``, ``"local"``).
        port: Service port number.
        protocol: URL protocol (default ``"http"``).

    Returns:
        URL string, or None for ``"local"`` locations.
    """
    if location == "local":
        return None

    info = resolve_location(location)
    local = is_location_local(location)

    # Mode 1: local, no scheduler → direct localhost
    if local and info.scheduler != "slurm":
        return f"{protocol}://localhost:{port}"

    # Mode 2: local + SLURM → discover compute node
    if local and info.scheduler == "slurm":
        from imas_codex.remote.tunnel import discover_compute_node_local

        compute_node = discover_compute_node_local(
            service_job_name=info.service_job_name
        )
        if compute_node:
            my_hostname = socket.gethostname().split(".")[0]
            if compute_node.split(".")[0] == my_hostname:
                return f"{protocol}://localhost:{port}"
            return f"{protocol}://{compute_node}:{port}"
        # No SLURM job found — fall back to localhost
        return f"{protocol}://localhost:{port}"

    # Mode 3: remote → access via SSH tunnel (localhost)
    return f"{protocol}://localhost:{port}"
