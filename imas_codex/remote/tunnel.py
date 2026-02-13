"""Shared SSH tunnel management.

Provides a single utility for establishing and managing SSH tunnels to
remote hosts. Used by both graph connections and embedding server access.

Port allocation uses a **tunnel offset** (default +10000) to separate
tunneled connections from direct ones, preventing port clashes between
local Neo4j and tunneled remote Neo4j on the same port.

Example::

    from imas_codex.remote.tunnel import ensure_tunnel, TUNNEL_OFFSET

    # For graph: remote bolt 7687 → local tunnel 17687
    ok = ensure_tunnel(port=7687, tunnel_port=17687, ssh_host="iter")

    # For embedding server: same-port forwarding
    ok = ensure_tunnel(port=18765, ssh_host="iter")
"""

from __future__ import annotations

import logging
import socket
import subprocess
import time

logger = logging.getLogger(__name__)

# Default offset added to remote port to derive the local tunnel port.
# e.g. remote 7687 → local 17687
TUNNEL_OFFSET = 10000


def is_tunnel_active(port: int) -> bool:
    """Check if a local port is already bound (tunnel or other service).

    Args:
        port: Local port to probe.

    Returns:
        True if something is listening on the port.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def is_port_bound_by_ssh(port: int) -> bool:
    """Check if a port is bound specifically by an SSH tunnel process.

    Uses ``ss -tlnp`` to distinguish SSH-bound ports from other services.
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


def ensure_tunnel(
    port: int,
    ssh_host: str,
    tunnel_port: int | None = None,
    timeout: float = 15.0,
) -> bool:
    """Ensure an SSH tunnel is active from localhost to a remote host.

    If ``tunnel_port`` is already bound locally, assumes the tunnel is
    active and returns immediately.  Otherwise launches::

        ssh -f -N -o ExitOnForwardFailure=yes \\
            -L {tunnel_port}:127.0.0.1:{port} {ssh_host}

    Args:
        port: Remote port to forward.
        ssh_host: SSH host alias or hostname.
        tunnel_port: Local port for the tunnel.  Defaults to ``port``
            (same-port forwarding, used for embedding server).
        timeout: Seconds to wait for SSH command and connection probe.

    Returns:
        True if tunnel is active (pre-existing or newly created).
    """
    local_port = tunnel_port if tunnel_port is not None else port

    # Already bound → assume tunnel active
    if is_tunnel_active(local_port):
        logger.debug("Port %d already bound locally, tunnel likely active", local_port)
        return True

    logger.info(
        "Starting SSH tunnel %s:%d → localhost:%d ...", ssh_host, port, local_port
    )

    try:
        subprocess.run(
            [
                "ssh",
                "-f",
                "-N",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "ConnectTimeout=10",
                "-L",
                f"{local_port}:127.0.0.1:{port}",
                ssh_host,
            ],
            timeout=timeout,
            check=True,
            capture_output=True,
            text=True,
        )
        # Give tunnel a moment to establish
        time.sleep(1.0)

        # Verify the tunnel came up
        if is_tunnel_active(local_port):
            logger.info(
                "SSH tunnel established: %s:%d → localhost:%d",
                ssh_host,
                port,
                local_port,
            )
            return True

        logger.warning("SSH tunnel started but port %d not reachable", local_port)
        return False

    except subprocess.TimeoutExpired:
        logger.warning("SSH tunnel start timed out (host: %s)", ssh_host)
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(
            "SSH tunnel start failed: %s", e.stderr.strip() if e.stderr else e
        )
        return False
    except FileNotFoundError:
        logger.warning("ssh command not found")
        return False


def stop_tunnel(ssh_host: str) -> bool:
    """Stop an SSH tunnel to the given host.

    Tries ``ssh -O exit`` first (clean ControlMaster shutdown), then
    falls back to ``pkill``.

    Returns:
        True if a tunnel was stopped.
    """
    result = subprocess.run(
        ["ssh", "-O", "exit", ssh_host],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("Tunnel to %s stopped via ControlMaster", ssh_host)
        return True

    result = subprocess.run(
        ["pkill", "-f", f"ssh.*-N.*{ssh_host}"],
        capture_output=True,
    )
    if result.returncode == 0:
        logger.info("Tunnel to %s killed via pkill", ssh_host)
        return True

    logger.debug("No active tunnel to %s found", ssh_host)
    return False


__all__ = [
    "TUNNEL_OFFSET",
    "ensure_tunnel",
    "is_port_bound_by_ssh",
    "is_tunnel_active",
    "stop_tunnel",
]
