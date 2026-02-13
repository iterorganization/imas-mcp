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
import signal
import socket
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Default offset added to remote port to derive the local tunnel port.
# e.g. remote 7687 → local 17687
TUNNEL_OFFSET = 10000

# PID file directory for tracking our tunnel processes
_PID_DIR = Path.home() / ".local" / "share" / "imas-codex" / "tunnels"


def _pid_file(ssh_host: str) -> Path:
    """Return the PID file path for a given host."""
    return _PID_DIR / f"{ssh_host}.pid"


def _write_pid(ssh_host: str, pid: int) -> None:
    """Record an autossh/ssh tunnel PID for later cleanup."""
    _PID_DIR.mkdir(parents=True, exist_ok=True)
    _pid_file(ssh_host).write_text(str(pid))


def _read_pid(ssh_host: str) -> int | None:
    """Read the recorded PID for a host's tunnel, or None."""
    path = _pid_file(ssh_host)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
        # Verify process still exists
        try:
            import os

            os.kill(pid, 0)
            return pid
        except ProcessLookupError:
            path.unlink(missing_ok=True)
            return None
    except (ValueError, OSError):
        path.unlink(missing_ok=True)
        return None


def _clear_pid(ssh_host: str) -> None:
    """Remove the PID file for a host."""
    _pid_file(ssh_host).unlink(missing_ok=True)


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

        ssh -f -N -o ControlMaster=no -o ControlPath=none \\
            -L {tunnel_port}:127.0.0.1:{port} {ssh_host}

    ``ExitOnForwardFailure`` is intentionally omitted — config-level
    ``RemoteForward`` directives (e.g. ``RemoteForward 2222``) would kill
    the entire connection.  Port liveness is verified after connect instead.

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
                "ControlMaster=no",
                "-o",
                "ControlPath=none",
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
    """Stop SSH tunnel processes to the given host.

    Uses PID files written at tunnel start to target only our processes.
    Falls back to pattern matching as a safety net, but **only** for
    ``imas-codex``-style tunnel patterns (``-L {offset_port}:``).

    Does **not** use ``ssh -O exit`` (would kill ControlMaster sessions)
    and does **not** blindly ``pkill autossh.*{host}`` (would kill the
    user's unrelated autossh processes).

    Returns:
        True if a tunnel process was killed.
    """
    stopped = False

    # Primary: kill by recorded PID
    pid = _read_pid(ssh_host)
    if pid is not None:
        try:
            import os

            # Kill the process group to catch autossh + its child ssh
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            logger.info("Killed tunnel process group for %s (pid %d)", ssh_host, pid)
            stopped = True
        except (ProcessLookupError, PermissionError):
            logger.debug("PID %d already gone", pid)
        _clear_pid(ssh_host)

    # Fallback: pattern match only for imas-codex-style tunnel ports
    # (offset ports like 17687, 17474 that only we create)
    offset_pattern = f"-L 1[0-9]{{4}}:.*{ssh_host}"
    result = subprocess.run(
        ["pgrep", "-f", offset_pattern],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        for pid_str in result.stdout.strip().splitlines():
            try:
                import os

                os.kill(int(pid_str), signal.SIGTERM)
                logger.info("Killed orphan tunnel pid %s for %s", pid_str, ssh_host)
                stopped = True
            except (ProcessLookupError, PermissionError, ValueError):
                pass

    if not stopped:
        logger.debug("No active tunnel to %s found", ssh_host)

    return stopped


__all__ = [
    "TUNNEL_OFFSET",
    "_write_pid",
    "ensure_tunnel",
    "is_port_bound_by_ssh",
    "is_tunnel_active",
    "stop_tunnel",
]
