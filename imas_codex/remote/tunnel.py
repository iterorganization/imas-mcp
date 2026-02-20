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

import fcntl
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
    "-o", "ConnectTimeout=10",
    "-o", "ExitOnForwardFailure=yes",
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


# Timeout (seconds) for acquiring the tunnel lock.  If another process
# is already starting a tunnel, we wait up to this long before giving up.
_LOCK_TIMEOUT = 30


def _lock_path(ssh_host: str) -> Path:
    """Return the lock file path for tunnel serialisation."""
    _PID_DIR.mkdir(parents=True, exist_ok=True)
    return _PID_DIR / f"{ssh_host}.lock"


def ensure_tunnel(
    port: int,
    ssh_host: str,
    tunnel_port: int | None = None,
    timeout: float = 15.0,
    remote_bind: str = "127.0.0.1",
) -> bool:
    """Ensure an SSH tunnel is active from localhost to a remote host.

    Prefers ``autossh`` for automatic reconnection. Falls back to plain
    ``ssh -f -N`` if autossh is not installed.

    Uses a file lock to serialise concurrent callers — prevents the race
    where two processes both detect "port unbound" and both try to start
    a tunnel, causing ``bind: Address already in use`` or silent clobber.

    If ``tunnel_port`` is already bound locally, verifies the tunnel
    process is alive and returns immediately (no lock needed).

    Args:
        port: Remote port to forward.
        ssh_host: SSH host alias or hostname.
        tunnel_port: Local port for the tunnel.  Defaults to ``port``
            (same-port forwarding, used for embedding server).
        timeout: Seconds to wait for SSH command and connection probe.
        remote_bind: Hostname on the remote side to bind the forward to.
            Use a SLURM compute node hostname when Neo4j runs on a compute
            node rather than the login node.  Default ``"127.0.0.1"``.

    Returns:
        True if tunnel is active (pre-existing or newly created).
    """
    local_port = tunnel_port if tunnel_port is not None else port

    # Fast path (no lock): port bound AND backed by SSH → tunnel active
    if verify_tunnel(local_port, ssh_host):
        logger.debug("Port %d bound by SSH process, tunnel active", local_port)
        return True

    # Port might be bound by a non-SSH process (local Neo4j) or in
    # TIME_WAIT from a dead tunnel.  Only skip if a real service holds it.
    if is_tunnel_active(local_port) and not is_port_bound_by_ssh(local_port):
        # Something other than SSH is listening — don't interfere
        logger.debug("Port %d bound by non-SSH process, skipping", local_port)
        return True

    # Serialise tunnel startup via file lock.  This prevents two callers
    # from racing through the "port unbound → start ssh" sequence.
    lock_file = _lock_path(ssh_host)
    lock_fd = lock_file.open("w")
    try:
        deadline = time.monotonic() + _LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    logger.warning("Timed out waiting for tunnel lock (%s)", ssh_host)
                    return is_tunnel_active(local_port)
                time.sleep(0.5)

        # Re-check after acquiring lock — another process may have
        # started the tunnel while we were waiting.
        if verify_tunnel(local_port, ssh_host):
            logger.debug("Port %d became active while waiting for lock", local_port)
            return True

        # Port might be bound but not by SSH (dead tunnel in TIME_WAIT).
        # Clean up stale state and proceed with starting a new tunnel.
        if is_tunnel_active(local_port) and not is_port_bound_by_ssh(local_port):
            # Something else holds the port — can't start tunnel
            logger.debug("Port %d bound by non-SSH process after lock", local_port)
            return True

        return _start_tunnel_locked(port, ssh_host, local_port, timeout, remote_bind)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        lock_fd.close()


def _start_tunnel_locked(
    port: int,
    ssh_host: str,
    local_port: int,
    timeout: float,
    remote_bind: str = "127.0.0.1",
) -> bool:
    """Start an SSH tunnel (caller must hold the file lock)."""
    # Port is dead — clean up stale PID if any
    _clear_pid(ssh_host)

    logger.info(
        "Starting SSH tunnel %s:%s:%d → localhost:%d ...",
        ssh_host,
        remote_bind,
        port,
        local_port,
    )

    use_autossh = _has_autossh()
    forward_arg = f"{local_port}:{remote_bind}:{port}"

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
        # ssh -f forks after connection setup; ExitOnForwardFailure=yes
        # ensures the forward is bound before the fork.  Brief sleep for
        # the forked process to appear in the process table.
        time.sleep(0.3)
        _find_and_record_pid(ssh_host, local_port)

        # Wait for tunnel to bind — with ExitOnForwardFailure, this
        # should succeed quickly.  Retry to handle slow networks.
        for _attempt in range(8):
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


def discover_compute_node(ssh_host: str) -> str | None:
    """Discover the SLURM compute node running imas-codex services.

    SSHes to ``ssh_host`` and queries ``squeue`` for the
    ``imas-codex-services`` job to find which compute node it runs on.

    Returns:
        Compute node hostname, or None if no allocation is active.
    """
    try:
        result = subprocess.run(
            [
                "ssh",
                ssh_host,
                "-o",
                "ConnectTimeout=10",
                'squeue -n imas-codex-services -u "$USER" '
                '--format="%N" --noheader 2>/dev/null',
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            node = result.stdout.strip()
            if node:
                return node
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def resolve_remote_bind(ssh_host: str, scheduler: str) -> str:
    """Resolve the remote bind address for a tunnel.

    When the scheduler is ``"slurm"``, discovers the compute node and
    returns its hostname.  Otherwise returns ``"127.0.0.1"`` (login node).

    Args:
        ssh_host: SSH host alias.
        scheduler: Service scheduler (``"slurm"`` or ``"none"``).

    Returns:
        Remote bind hostname for the ``-L`` forward argument.
    """
    if scheduler == "slurm":
        node = discover_compute_node(ssh_host)
        if node:
            logger.info("SLURM compute node for %s: %s", ssh_host, node)
            return node
        logger.debug("No SLURM allocation found, using login node")
    return "127.0.0.1"


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
    "discover_compute_node",
    "ensure_tunnel",
    "is_port_bound_by_ssh",
    "is_systemd_tunnel_active",
    "is_tunnel_active",
    "resolve_remote_bind",
    "stop_tunnel",
    "verify_tunnel",
    "_lock_path",
]
