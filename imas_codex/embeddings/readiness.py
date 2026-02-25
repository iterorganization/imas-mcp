"""Centralized embedding server readiness check.

Single entry point for ensuring the remote embedding server is available
before starting long-running operations (wiki discovery, signal enrichment, etc.).

Architecture::

    On facility login node:
      Client → compute_node:18765 → embedding server (SLURM GPU node)

    Off facility (WSL/workstation):
      Client → localhost:18765 → SSH tunnel → compute_node:18765 → server

Usage::

    from imas_codex.embeddings.readiness import ensure_embedding_ready

    ok, msg = ensure_embedding_ready()
    if not ok:
        print(f"Embedding not available: {msg}")
"""

from __future__ import annotations

import logging
import re
import socket
import subprocess
import time

from imas_codex.remote.tunnel import ensure_tunnel

logger = logging.getLogger(__name__)


def _get_embed_host() -> str | None:
    """Get the hostname where the embed server runs.

    Delegates to settings which discovers the SLURM compute node.
    Returns None when not resolvable.
    """
    from imas_codex.settings import get_embed_host

    return get_embed_host()


def _is_on_facility() -> bool:
    """Check if running on the embedding server's facility."""
    from imas_codex.remote.locations import is_location_local
    from imas_codex.settings import get_embedding_location

    return is_location_local(get_embedding_location())


def _is_on_login_node() -> bool:
    """Check if running on a login node (has systemd user session)."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-system-running"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.returncode == 0 or "running" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _resolve_url_for_compute(url: str) -> str:
    """On facility nodes, redirect localhost URLs to the embed server host.

    The embedding server runs on a compute node (e.g. Titan).  When the
    resolved URL contains localhost, replace it with the actual compute
    host so facility nodes can reach it directly.
    """
    if not url:
        return url
    embed_host = _get_embed_host()
    if not embed_host:
        return url
    resolved = re.sub(r"localhost|127\.0\.0\.1", embed_host, url)
    if resolved != url:
        logger.info("Redirecting %s → %s", url, resolved)
    return resolved


def _try_start_service() -> bool:
    """Attempt to start the embedding server via systemd user service.

    Only works on ITER where the systemd service is installed.

    Returns:
        True if systemctl start was issued (doesn't guarantee server is ready)
    """
    try:
        result = subprocess.run(
            ["systemctl", "--user", "start", "imas-codex-embed"],
            timeout=10,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Started imas-codex-embed systemd service")
            return True
        logger.warning("systemctl start failed: %s", result.stderr.strip())
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug("Cannot start systemd service: %s", e)
        return False


def _ensure_ssh_tunnel(port: int, ssh_host: str | None = None) -> bool:
    """Ensure SSH tunnel from workstation to remote login node.

    Delegates to the shared tunnel utility in ``imas_codex.remote.tunnel``.

    Args:
        port: Port to forward
        ssh_host: SSH host alias (default: resolved from active graph profile)

    Returns:
        True if tunnel is active (already existed or newly created)
    """
    if ssh_host is None:
        from imas_codex.graph.profiles import get_graph_location
        from imas_codex.remote.locations import resolve_location

        ssh_host = resolve_location(get_graph_location()).ssh_host

    return ensure_tunnel(port=port, ssh_host=ssh_host)


def ensure_embedding_ready(
    log_fn: callable | None = None,
    status_fn: callable | None = None,
    timeout: float = 30.0,
) -> tuple[bool, str]:
    """Ensure the remote embedding server is available and ready.

    This is the **single entry point** for all CLI commands that need
    embeddings. It checks server health and establishes SSH tunnel if
    running off-ITER.

    Args:
        log_fn: Optional callback for status messages: log_fn(message, style)
                style is "info", "warning", "success", or "dim".
                Each call prints a new line.
        status_fn: Optional callback for single-line status updates:
                   status_fn(message). Updates the same line in-place
                   (e.g. a Rich Status spinner). Preferred over log_fn
                   for clean progress display.
        timeout: Maximum time to wait for server readiness (seconds)

    Returns:
        (success, message) tuple. If success is False, message explains why.
    """
    from imas_codex.embeddings.client import RemoteEmbeddingClient
    from imas_codex.settings import get_embed_remote_url, get_embed_server_port

    def log(msg: str, style: str = "info") -> None:
        if status_fn:
            status_fn(msg)
        elif log_fn:
            log_fn(msg, style)
        logger.info(msg)

    remote_url = get_embed_remote_url()
    if not remote_url:
        return False, "Remote embedding URL not configured (embed-remote-url)"

    # On compute nodes, localhost URLs must point to the login node
    remote_url = _resolve_url_for_compute(remote_url)

    port = get_embed_server_port()
    client = RemoteEmbeddingClient(remote_url)

    # Step 1: Quick health check — maybe server is already running
    if client.is_available(timeout=5.0):
        info = client.get_info()
        model = info.model if info else "unknown"
        source = _resolve_source_label(info)
        log(f"Embedding server ready: {model} on {source}", "success")
        return True, f"Server ready: {model} on {source}"

    log("Embedding server not responding, checking connectivity...", "dim")

    # Step 2: If off-facility, ensure SSH tunnel
    on_facility = _is_on_facility()
    if not on_facility:
        log("Setting up SSH tunnel...", "dim")
        if not _ensure_ssh_tunnel(port):
            return False, (
                "Cannot establish SSH tunnel.\n"
                "Start manually: ssh -f -N -L 18765:127.0.0.1:18765 iter\n"
                "Or install autossh service: imas-codex serve tunnel service install"
            )

    # Step 3: On login node, try starting the systemd service.
    # Only if embed server runs on this node (not on a separate compute host).
    embed_host = _get_embed_host()
    embed_on_this_node = (
        embed_host is None or embed_host == socket.gethostname().split(".")[0]
    )
    if on_facility and _is_on_login_node() and embed_on_this_node:
        log("Trying to start embedding service...", "dim")
        _try_start_service()
    elif on_facility and embed_host:
        log(
            f"Embedding server expected on {embed_host}",
            "dim",
        )

    # Step 4: Health check with retries (server might be starting)
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

    start_hint = (
        "Start the server on ITER:\n"
        "  ssh iter\n"
        "  imas-codex serve embed start --gpu 1\n"
        "Or install as systemd service:\n"
        "  imas-codex serve embed service install\n"
        "  imas-codex serve embed service start"
    )

    return False, (
        f"Embedding server not available at {remote_url} after {timeout:.0f}s.\n"
        f"{start_hint}"
    )


def _resolve_source_label(info) -> str:
    """Create a human-readable source label from server info."""
    if not info:
        return "remote"
    hostname = getattr(info, "hostname", None)
    return hostname or "remote"


__all__ = ["ensure_embedding_ready"]
