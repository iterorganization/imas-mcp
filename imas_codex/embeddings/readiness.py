"""Centralized embedding server readiness check.

Single entry point for ensuring the remote embedding server is available
before starting long-running operations (wiki discovery, signal enrichment, etc.).

Architecture::

    On ITER login node:
      Client → localhost:18765 → embedding server (same machine, T4 GPU)

    Off ITER (WSL/workstation):
      Client → localhost:18765 → SSH tunnel → ITER login:18765 → server

Usage::

    from imas_codex.embeddings.readiness import ensure_embedding_ready

    ok, msg = ensure_embedding_ready()
    if not ok:
        print(f"Embedding not available: {msg}")
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time

from imas_codex.remote.tunnel import ensure_tunnel

logger = logging.getLogger(__name__)

# ITER login node hostname — fallback when embed-host is not configured.
ITER_LOGIN_HOST = "98dci4-srv-1001"


def _get_embed_host() -> str:
    """Get the hostname where the embed server runs.

    Priority: settings embed-host → ITER login node fallback.
    """
    from imas_codex.settings import get_embed_host

    return get_embed_host() or ITER_LOGIN_HOST


def _is_on_iter() -> bool:
    """Check if running on an ITER cluster node (login or compute)."""
    return os.uname().nodename.startswith("98dci4-")


def _is_on_iter_login() -> bool:
    """Check if running on an ITER login node."""
    return os.uname().nodename.startswith("98dci4-srv-")


def _is_on_iter_compute() -> bool:
    """Check if running on an ITER compute node.

    Compute nodes have no systemd user session (no D-Bus) and cannot
    run the embedding server locally.  Services are on the login node.
    """
    return os.uname().nodename.startswith("98dci4-clu-")


def _resolve_url_for_compute(url: str) -> str:
    """On compute nodes, redirect localhost URLs to the embed server host.

    The embedding server may run on a compute node (e.g. Titan) or on
    the login node.  When pyproject.toml or env vars specify localhost,
    compute nodes need to reach the actual embed host instead.
    """
    if _is_on_iter_compute() and url:
        embed_host = _get_embed_host()
        resolved = re.sub(r"localhost|127\.0\.0\.1", embed_host, url)
        if resolved != url:
            logger.info("Compute node: redirecting %s → %s", url, resolved)
        return resolved
    return url


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

    # Step 3: On ITER login node, try starting the systemd service.
    # Skip on compute nodes — no D-Bus/systemd user session there.
    # Also skip if embed server is on a separate host (e.g. Titan).
    embed_host = _get_embed_host()
    embed_on_login = embed_host == ITER_LOGIN_HOST or embed_host.startswith(
        "98dci4-srv"
    )
    if on_iter and _is_on_iter_login() and embed_on_login:
        log("Trying to start embedding service...", "dim")
        _try_start_service()
    elif on_iter and _is_on_iter_compute():
        log(
            f"On compute node — embedding server expected on {embed_host}",
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
    if not hostname:
        return "remote"
    if hostname.startswith("98dci4-gpu"):
        return f"iter-gpu ({hostname})"
    if hostname.startswith("98dci4-srv"):
        return f"iter-login ({hostname})"
    return hostname


__all__ = ["ensure_embedding_ready"]
