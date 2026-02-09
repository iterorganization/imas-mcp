"""Centralized embedding server readiness check.

Single entry point for ensuring the remote embedding server is available
before starting long-running operations (wiki discovery, signal enrichment, etc.).

Handles the full lifecycle depending on where the client is running:

On ITER login node (98dci4-*):
  1. Quick health check → return if already healthy
  2. Call ensure_server() → submit SLURM job + TCP proxy + wait
  3. Final health check

Off ITER (workstation):
  1. Quick health check → return if already healthy
  2. Ensure SSH tunnel to ITER login node
  3. Call ensure_server() → submit SLURM job + TCP proxy + wait (via SSH)
  4. Final health check

Usage:
    from imas_codex.embeddings.readiness import ensure_embedding_ready

    ok, msg = ensure_embedding_ready()
    if not ok:
        print(f"Embedding not available: {msg}")
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import time

logger = logging.getLogger(__name__)


def _is_on_iter() -> bool:
    """Check if running on an ITER cluster node."""
    hostname = os.uname().nodename
    if hostname.startswith("98dci4-"):
        return True
    return shutil.which("sbatch") is not None


def _ensure_ssh_tunnel(port: int, ssh_host: str = "iter") -> bool:
    """Ensure SSH tunnel from workstation to ITER login node.

    Only needed when running off-ITER. The tunnel forwards
    localhost:PORT → login:PORT, which chains through the TCP proxy
    to reach the SLURM compute node.

    Args:
        port: Port to forward
        ssh_host: SSH host alias (default: iter)

    Returns:
        True if tunnel is active (already existed or newly created)
    """
    # Check if port is already bound locally
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(("127.0.0.1", port))
            if result == 0:
                logger.debug(
                    "Port %d already bound locally, tunnel likely active", port
                )
                return True
    except OSError:
        pass

    # Try to start SSH tunnel
    logger.info("Starting SSH tunnel to %s (port %d)...", ssh_host, port)
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
                f"{port}:127.0.0.1:{port}",
                ssh_host,
            ],
            timeout=15,
            check=True,
            capture_output=True,
            text=True,
        )
        # Give tunnel a moment to establish
        time.sleep(1.0)
        logger.info("SSH tunnel established to %s:%d", ssh_host, port)
        return True
    except subprocess.TimeoutExpired:
        logger.warning("SSH tunnel start timed out")
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(
            "SSH tunnel start failed: %s", e.stderr.strip() if e.stderr else e
        )
        return False
    except FileNotFoundError:
        logger.warning("ssh command not found")
        return False


def ensure_embedding_ready(
    log_fn: callable | None = None,
    timeout: float = 30.0,
) -> tuple[bool, str]:
    """Ensure the remote embedding server is available and ready.

    This is the **single entry point** for all CLI commands that need
    embeddings. It handles the full lifecycle:

    1. Quick health check — return immediately if server responds
    2. SLURM auto-launch — submit job, set up TCP proxy, wait for ready
    3. SSH tunnel — if running off-ITER, ensure SSH tunnel is active

    Args:
        log_fn: Optional callback for status messages: log_fn(message, style)
                style is "info", "warning", "success", or "dim"
        timeout: Maximum time to wait for server readiness (seconds)

    Returns:
        (success, message) tuple. If success is False, message explains why.
    """
    from imas_codex.embeddings.client import RemoteEmbeddingClient
    from imas_codex.settings import get_embed_remote_url, get_embed_server_port

    def log(msg: str, style: str = "info") -> None:
        if log_fn:
            log_fn(msg, style)
        logger.info(msg)

    remote_url = get_embed_remote_url()
    if not remote_url:
        return False, "Remote embedding URL not configured (embed-remote-url)"

    port = get_embed_server_port()
    client = RemoteEmbeddingClient(remote_url)

    # Step 1: Quick health check — maybe server is already running
    if client.is_available(timeout=5.0):
        info = client.get_info()
        model = info.model if info else "unknown"
        source = _resolve_source_label(info)
        log(f"Embedding server ready: {model} on {source}", "success")
        return True, f"Server ready: {model} on {source}"

    log("Embedding server not responding, attempting auto-start...", "dim")

    # Step 2: If off-ITER, ensure SSH tunnel
    on_iter = _is_on_iter()
    if not on_iter:
        log("Setting up SSH tunnel to ITER...", "dim")
        if not _ensure_ssh_tunnel(port):
            return False, (
                "Cannot establish SSH tunnel to ITER.\n"
                "Start manually: ssh -f -N -L 18765:127.0.0.1:18765 iter\n"
                "Or install autossh service: imas-codex serve tunnel service install"
            )

    # Step 3: SLURM auto-launch via ensure_server()
    # This works both on ITER (direct sbatch) and off-ITER (via SSH)
    log("Ensuring SLURM embedding server is running...", "dim")
    try:
        from imas_codex.embeddings.slurm import ensure_server
        from imas_codex.settings import get_imas_embedding_model

        model_name = get_imas_embedding_model()
        ok = ensure_server(model_name=model_name)
        if not ok:
            return False, (
                "SLURM auto-launch failed. The server could not be started.\n"
                "Check SLURM queue: squeue -u $USER -n imas-codex-embed\n"
                "Check logs: imas-codex serve embed logs\n"
                "Submit manually: imas-codex serve embed submit"
            )
    except Exception as e:
        logger.warning("SLURM auto-launch exception: %s", e)
        # SLURM might not be available — fall through to health check
        # (user might have started server some other way)

    # Step 4: If we're off-ITER and just started the SLURM job, the SSH
    # tunnel needs to be re-established since the TCP proxy destination
    # may have changed (new compute node). Re-check the tunnel.
    if not on_iter:
        # Give a moment for TCP proxy to bind
        time.sleep(1.0)

    # Step 5: Final health check with generous timeout
    deadline = time.time() + timeout
    last_attempt = 0
    while time.time() < deadline:
        if client.is_available(timeout=5.0):
            info = client.get_info()
            model = info.model if info else "unknown"
            source = _resolve_source_label(info)
            log(f"Embedding server ready: {model} on {source}", "success")
            return True, f"Server ready: {model} on {source}"
        remaining = deadline - time.time()
        if remaining > 5:
            last_attempt += 1
            if last_attempt % 3 == 0:
                log(f"Waiting for server... ({remaining:.0f}s remaining)", "dim")
        time.sleep(2.0)

    return False, (
        f"Embedding server not available at {remote_url} after {timeout:.0f}s.\n"
        "Check status: imas-codex serve embed status\n"
        "Check logs: imas-codex serve embed logs\n"
        "Submit manually: imas-codex serve embed submit"
    )


def _resolve_source_label(info) -> str:
    """Create a human-readable source label from server info."""
    if not info:
        return "remote"
    hostname = getattr(info, "hostname", None)
    if not hostname:
        return "remote"
    if hostname.startswith("98dci4-gpu"):
        return f"iter-titan ({hostname})"
    if hostname.startswith("98dci4-srv"):
        return f"iter-login ({hostname})"
    return hostname


__all__ = ["ensure_embedding_ready"]
