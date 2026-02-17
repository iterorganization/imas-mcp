"""Shared SSH tunnel management.

Provides a single utility for establishing and managing SSH tunnels to
remote hosts. Used by both graph connections and embedding server access.

Prefers ``autossh`` when available for automatic reconnection after
network interruptions. Falls back to plain ``ssh -f -N`` otherwise.

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
import os
import shutil
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

# SSH keepalive settings — tuned for fast drop detection.
# ServerAliveInterval × ServerAliveCountMax = detection window.
# 15 × 3 = 45 seconds (vs default 90s).
_SSH_ALIVE_INTERVAL = 15
_SSH_ALIVE_COUNT_MAX = 3

# Common SSH options for tunnel connections.
# These are used by both ensure_tunnel() and the CLI.
SSH_TUNNEL_OPTS: list[str] = [
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-o", "TCPKeepAlive=yes",
    "-o", f"ServerAliveInterval={_SSH_ALIVE_INTERVAL}",
    "-o", f"ServerAliveCountMax={_SSH_ALIVE_COUNT_MAX}",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ConnectTimeout=10",
]  # fmt: skip


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


def _find_and_record_pid(ssh_host: str, local_port: int) -> None:
    """Find the SSH/autossh process for our tunnel and record its PID."""
    for prog in ("autossh", "ssh"):
        result = subprocess.run(
            ["pgrep", "-f", f"{prog}.*-L {local_port}:.*{ssh_host}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for pid_str in result.stdout.strip().splitlines():
                try:
                    _write_pid(ssh_host, int(pid_str))
                    return
                except ValueError:
                    continue


def _has_autossh() -> bool:
    """Check if autossh is available on PATH."""
    return shutil.which("autossh") is not None


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


def verify_tunnel(port: int, ssh_host: str) -> bool:
    """Actively verify a tunnel is alive by checking both port and process.

    Unlike ``is_tunnel_active`` (which only probes the port), this also
    verifies the SSH process is still running. A port can appear open
    briefly after the process dies due to TCP TIME_WAIT.

    Args:
        port: Local tunnel port.
        ssh_host: SSH host the tunnel connects to.

    Returns:
        True if tunnel port is open AND backed by an SSH/autossh process.
    """
    if not is_tunnel_active(port):
        return False
    # Confirm an SSH process is behind this port
    return is_port_bound_by_ssh(port)


def ensure_tunnel(
    port: int,
    ssh_host: str,
    tunnel_port: int | None = None,
    timeout: float = 15.0,
) -> bool:
    """Ensure an SSH tunnel is active from localhost to a remote host.

    Prefers ``autossh`` for automatic reconnection. Falls back to plain
    ``ssh -f -N`` if autossh is not installed.

    If ``tunnel_port`` is already bound locally, verifies the tunnel
    process is alive and returns immediately.

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

    # Already bound → verify it's still a valid tunnel
    if is_tunnel_active(local_port):
        logger.debug("Port %d already bound locally, tunnel likely active", local_port)
        return True

    # Port is dead — clean up stale PID if any
    _clear_pid(ssh_host)

    logger.info(
        "Starting SSH tunnel %s:%d → localhost:%d ...", ssh_host, port, local_port
    )

    use_autossh = _has_autossh()
    forward_arg = f"{local_port}:127.0.0.1:{port}"

    if use_autossh:
        cmd = [
            "autossh", "-M", "0", "-f", "-N",
            *SSH_TUNNEL_OPTS,
            "-L", forward_arg,
            ssh_host,
        ]  # fmt: skip
        env = {
            **os.environ,
            "AUTOSSH_GATETIME": "0",
            "AUTOSSH_POLL": "30",
        }
        logger.debug("Using autossh for auto-reconnection")
    else:
        cmd = [
            "ssh", "-f", "-N",
            *SSH_TUNNEL_OPTS,
            "-L", forward_arg,
            ssh_host,
        ]  # fmt: skip
        env = None
        logger.debug("autossh not found, using plain ssh (no auto-reconnection)")

    try:
        subprocess.run(
            cmd,
            timeout=timeout,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        # Record PID for targeted cleanup
        time.sleep(0.5)
        _find_and_record_pid(ssh_host, local_port)

        # Wait for tunnel to bind
        for _attempt in range(6):
            if is_tunnel_active(local_port):
                logger.info(
                    "SSH tunnel established (%s): %s:%d → localhost:%d",
                    "autossh" if use_autossh else "ssh",
                    ssh_host,
                    port,
                    local_port,
                )
                return True
            time.sleep(0.5)

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
        logger.warning("ssh/autossh command not found")
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
                os.kill(int(pid_str), signal.SIGTERM)
                logger.info("Killed orphan tunnel pid %s for %s", pid_str, ssh_host)
                stopped = True
            except (ProcessLookupError, PermissionError, ValueError):
                pass

    if not stopped:
        logger.debug("No active tunnel to %s found", ssh_host)

    return stopped


def is_systemd_tunnel_active(ssh_host: str) -> bool:
    """Check if a systemd-managed tunnel service is active for this host.

    Returns True if the ``imas-codex-tunnel-{host}`` systemd user service
    exists and is in an active state. When True, callers should not start
    ad-hoc tunnels — the systemd service handles reconnection.
    """
    service_name = f"imas-codex-tunnel-{ssh_host}"
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", service_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() == "active"
    except Exception:
        return False


__all__ = [
    "SSH_TUNNEL_OPTS",
    "TUNNEL_OFFSET",
    "_write_pid",
    "ensure_tunnel",
    "is_port_bound_by_ssh",
    "is_systemd_tunnel_active",
    "is_tunnel_active",
    "stop_tunnel",
    "verify_tunnel",
]
