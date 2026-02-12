"""Service health monitoring for discovery CLI tools.

Provides a composable ``ServiceMonitor`` that polls external dependencies
(Neo4j, embedding server, SSH/VPN connectivity) at low frequency and
exposes live status for progress displays plus a gate that workers can
``await`` before processing.

Design principles:
- **Composable**: Register only the checks a particular CLI needs.
- **Low overhead**: Polls at configurable intervals (default 15s).
- **Worker-friendly**: ``await_services_ready()`` blocks workers when
  services are unhealthy and unblocks automatically on recovery.
- **Display-friendly**: ``get_status()`` returns a snapshot for the
  SERVERS progress row without blocking.

Usage::

    monitor = ServiceMonitor()
    monitor.add_check("graph", neo4j_health_check)
    monitor.add_check("ssh", lambda: ssh_health_check("jt60sa"),
                       auth_label="vpn")

    async with monitor:
        # Start workers that gate on monitor
        while not should_stop():
            await monitor.await_services_ready()
            do_work()

The monitor integrates with the existing ``EmbeddingResilience`` module
by replacing the embed-specific polling with a unified service monitor,
while keeping the embed status accessible for backwards compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Service Status
# =============================================================================


class ServiceState(str, Enum):
    """Health state of a monitored service."""

    unknown = "unknown"  # Not yet checked
    healthy = "healthy"  # Service is operational
    degraded = "degraded"  # Partial availability
    unhealthy = "unhealthy"  # Service is down
    recovering = "recovering"  # Was down, attempting reconnect


@dataclass
class ServiceStatus:
    """Current status snapshot for a single service."""

    name: str
    state: ServiceState = ServiceState.unknown
    auth_label: str | None = None  # e.g., "vpn", "tequila", "ssh"
    detail: str = ""  # Human-readable detail (e.g., "bolt://localhost:7687")
    healthy_detail: str = ""  # Detail to show when grayed (stored on first success)
    last_check: float = 0.0
    last_healthy: float = 0.0
    consecutive_failures: int = 0
    check_latency_ms: float = 0.0

    @property
    def is_healthy(self) -> bool:
        return self.state in (ServiceState.healthy, ServiceState.unknown)

    @property
    def downtime_seconds(self) -> float:
        """Seconds since last healthy check, or 0 if currently healthy."""
        if self.is_healthy or self.last_healthy == 0:
            return 0.0
        return time.time() - self.last_healthy


# =============================================================================
# Health Check Functions
# =============================================================================


def get_graph_location() -> str:
    """Get the configured graph location display name.

    Reads from config/graph.yaml and returns the display name for the
    current location (e.g., 'iter-login', 'local', 'docker').

    Returns:
        Location display name, or 'unknown' if not configured.
    """
    from importlib.resources import files

    import yaml

    try:
        config_path = files("imas_codex.config").joinpath("graph.yaml")
        with config_path.open() as f:
            config = yaml.safe_load(f)
        location = config.get("location", "local")
        locations = config.get("locations", {})
        loc_config = locations.get(location, {})
        return loc_config.get("display_name", location)
    except Exception:
        return "unknown"


def neo4j_health_check() -> tuple[bool, str]:
    """Check Neo4j connectivity via bolt protocol.

    Returns:
        (healthy, detail) tuple where detail is the graph location
        (e.g., 'iter-login') instead of the raw bolt URI.
    """
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            result = gc.query("RETURN 1 AS ok")
            if result and result[0]["ok"] == 1:
                # Return the location display name instead of raw URI
                return True, get_graph_location()
        return False, "query returned unexpected result"
    except Exception as e:
        return False, str(e)[:100]


def _kill_ssh_controlmaster(ssh_host: str) -> None:
    """Kill a stale SSH ControlMaster socket.

    When VPN drops, the ControlMaster socket becomes stale and all new
    ssh connections multiplex through the dead socket.  Killing it
    forces the next ssh to establish a fresh connection.
    """
    try:
        subprocess.run(
            ["ssh", "-O", "exit", ssh_host],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass  # Best-effort cleanup


def ssh_health_check(ssh_host: str, timeout: int = 10) -> tuple[bool, str]:
    """Check SSH connectivity to a remote host.

    For VPN-gated hosts (like jt60sa), this implicitly validates VPN
    status since SSH will fail if the VPN tunnel is down.

    On failure after a previous success, kills the ControlMaster socket
    so recovery can establish a fresh connection.

    Args:
        ssh_host: SSH host alias (from ~/.ssh/config)
        timeout: Connection timeout in seconds

    Returns:
        (healthy, detail) tuple
    """
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                f"ConnectTimeout={timeout}",
                "-o",
                "BatchMode=yes",
                ssh_host,
                "echo ok",
            ],
            capture_output=True,
            timeout=timeout + 5,
        )
        if result.returncode == 0 and b"ok" in result.stdout:
            return True, ssh_host
        # SSH failed — kill stale ControlMaster so recovery can reconnect
        _kill_ssh_controlmaster(ssh_host)
        stderr = result.stderr.decode("utf-8", errors="replace").strip()[:100]
        return False, stderr or f"exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        _kill_ssh_controlmaster(ssh_host)
        return False, f"timeout ({timeout}s)"
    except Exception as e:
        _kill_ssh_controlmaster(ssh_host)
        return False, str(e)[:100]


def wiki_auth_check(
    url: str, ssh_host: str | None = None, timeout: int = 10
) -> tuple[bool, str]:
    """Check that a wiki URL is reachable.

    Tests HTTP access to the primary wiki page.  For tunnel/VPN sites
    this goes through the local port-forward; for SSH sites it curls
    via the remote host.

    Args:
        url: Primary wiki URL to probe
        ssh_host: If set, fetch via ``ssh <host> curl``
        timeout: Connection timeout in seconds

    Returns:
        (healthy, detail) tuple
    """
    try:
        if ssh_host:
            cmd = f'curl -sk --noproxy "*" -o /dev/null -w "%{{http_code}}" "{url}" 2>/dev/null'
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    f"ConnectTimeout={timeout}",
                    "-o",
                    "BatchMode=yes",
                    ssh_host,
                    cmd,
                ],
                capture_output=True,
                timeout=timeout + 15,
            )
            if result.returncode == 0:
                code = result.stdout.decode().strip()
                if code.startswith(("2", "3")):
                    return True, url
                return False, f"HTTP {code}"
            return False, "ssh+curl failed"
        else:
            import urllib.request

            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if 200 <= resp.status < 400:
                    return True, url
                return False, f"HTTP {resp.status}"
    except subprocess.TimeoutExpired:
        return False, f"timeout ({timeout}s)"
    except Exception as e:
        return False, str(e)[:80]


def embed_health_check() -> tuple[bool, str]:
    """Check embedding server health.

    Returns:
        (healthy, detail) tuple with source info
    """
    try:
        from imas_codex.embeddings import get_embedding_source
        from imas_codex.embeddings.client import RemoteEmbeddingClient
        from imas_codex.settings import get_embed_remote_url

        url = get_embed_remote_url()
        if not url:
            # Local embeddings don't need a server
            source = get_embedding_source()
            return True, f"local ({source})"

        client = RemoteEmbeddingClient(url)
        try:
            if client.is_available(timeout=5.0):
                source = get_embedding_source()
                return True, source
            return False, "server unavailable"
        finally:
            client.close()
    except Exception as e:
        return False, str(e)[:100]


# =============================================================================
# Service Check Configuration
# =============================================================================


@dataclass
class ServiceCheck:
    """Configuration for a single service health check."""

    name: str
    check_fn: Any  # Callable[[], tuple[bool, str]]
    auth_label: str | None = None  # Display label: "vpn", "ssh", etc.
    poll_interval: float = 15.0  # Seconds between checks
    critical: bool = True  # If True, workers pause when unhealthy


# =============================================================================
# Service Monitor
# =============================================================================


class ServiceMonitor:
    """Async service health monitor with worker gating.

    Runs periodic health checks in the background and provides:
    - ``get_status()`` for progress display (non-blocking)
    - ``await_services_ready()`` for workers (blocks until all critical
      services are healthy)
    - ``paused`` property for quick boolean check

    Thread-safe: status is read from the display thread, written from
    the async polling loop.
    """

    def __init__(self, poll_interval: float = 15.0) -> None:
        self._checks: list[ServiceCheck] = []
        self._status: dict[str, ServiceStatus] = {}
        self._lock = threading.Lock()
        self._ready_event: asyncio.Event | None = None
        self._poll_task: asyncio.Task | None = None
        self._default_poll_interval = poll_interval
        self._stopped = False

    # ------------------------------------------------------------------
    # Configuration (before start)
    # ------------------------------------------------------------------

    def add_check(
        self,
        name: str,
        check_fn: Any,
        auth_label: str | None = None,
        poll_interval: float | None = None,
        critical: bool = True,
    ) -> None:
        """Register a service health check.

        Args:
            name: Service name (e.g., "graph", "embed", "ssh")
            check_fn: Callable returning (healthy: bool, detail: str)
            auth_label: Auth method label for display (e.g., "vpn")
            poll_interval: Override default poll interval for this check
            critical: If True, workers pause when this service is unhealthy
        """
        check = ServiceCheck(
            name=name,
            check_fn=check_fn,
            auth_label=auth_label,
            poll_interval=poll_interval or self._default_poll_interval,
            critical=critical,
        )
        self._checks.append(check)
        self._status[name] = ServiceStatus(
            name=name,
            auth_label=auth_label,
        )

    # ------------------------------------------------------------------
    # Status queries (thread-safe)
    # ------------------------------------------------------------------

    def get_status(self) -> list[ServiceStatus]:
        """Get current status of all monitored services.

        Thread-safe: can be called from the display thread.
        """
        with self._lock:
            return list(self._status.values())

    def get_service_status(self, name: str) -> ServiceStatus | None:
        """Get status for a specific service."""
        with self._lock:
            return self._status.get(name)

    @property
    def paused(self) -> bool:
        """True if any critical service is unhealthy."""
        with self._lock:
            return any(
                not status.is_healthy
                for check in self._checks
                if check.critical
                for status in [self._status.get(check.name)]
                if status is not None
            )

    @property
    def all_healthy(self) -> bool:
        """True if all services are healthy."""
        return not self.paused

    def is_service_healthy(self, *names: str) -> bool:
        """Check whether specific services are healthy.

        Workers that only depend on a subset of services can check
        just those instead of the global ``all_healthy``.

        Args:
            names: Service names to check (e.g., "ssh", "auth")

        Returns:
            True if ALL named services are currently healthy.
        """
        with self._lock:
            for name in names:
                status = self._status.get(name)
                if status is not None and not status.is_healthy:
                    return False
            return True

    # ------------------------------------------------------------------
    # Worker gating (async)
    # ------------------------------------------------------------------

    async def await_services_ready(self, timeout: float = 0) -> bool:
        """Block until all critical services are healthy.

        Args:
            timeout: Max seconds to wait (0 = wait indefinitely)

        Returns:
            True when services are ready, False if timeout exceeded
        """
        if self.all_healthy:
            return True

        if self._ready_event is None:
            return True  # Not started yet, don't block

        logger.info(
            "Services unhealthy, pausing workers: %s",
            ", ".join(
                s.name
                for s in self.get_status()
                if not s.is_healthy
                and any(c.critical for c in self._checks if c.name == s.name)
            ),
        )

        if timeout > 0:
            try:
                await asyncio.wait_for(self._wait_for_ready(), timeout=timeout)
                return True
            except TimeoutError:
                return False
        else:
            await self._wait_for_ready()
            return True

    async def _wait_for_ready(self) -> None:
        """Wait until the ready event is set."""
        while not self.all_healthy:
            if self._ready_event is not None:
                self._ready_event.clear()
                await self._ready_event.wait()
            else:
                await asyncio.sleep(1.0)

    # ------------------------------------------------------------------
    # Lifecycle (async context manager)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> ServiceMonitor:
        """Start the background polling loop."""
        self._ready_event = asyncio.Event()
        self._ready_event.set()  # Start optimistic
        self._stopped = False

        # Run initial checks synchronously to get baseline
        await self._run_all_checks()

        # Start background polling
        self._poll_task = asyncio.create_task(self._poll_loop())
        return self

    async def __aexit__(self, *args) -> None:
        """Stop the background polling loop."""
        self._stopped = True
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Internal polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Background polling loop with per-check intervals."""
        # Track last check time per service
        last_checked: dict[str, float] = {c.name: 0.0 for c in self._checks}

        while not self._stopped:
            try:
                now = time.time()
                tasks = []

                for check in self._checks:
                    # Check if it's time to poll this service
                    elapsed = now - last_checked[check.name]

                    # Poll more frequently when unhealthy (backoff: min 5s)
                    status = self._status.get(check.name)
                    interval = check.poll_interval
                    if status and not status.is_healthy:
                        interval = max(5.0, interval / 3)

                    if elapsed >= interval:
                        last_checked[check.name] = now
                        tasks.append(self._run_check(check))

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                    # Update ready event
                    if self.all_healthy:
                        if self._ready_event and not self._ready_event.is_set():
                            logger.info("All services healthy, resuming workers")
                            self._ready_event.set()
                    else:
                        if self._ready_event and self._ready_event.is_set():
                            self._ready_event.clear()

                await asyncio.sleep(1.0)  # Check schedule every second

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("Service monitor poll error: %s", e)
                await asyncio.sleep(5.0)

    async def _run_all_checks(self) -> None:
        """Run all checks once (used for initial baseline)."""
        tasks = [self._run_check(check) for check in self._checks]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_check(self, check: ServiceCheck) -> None:
        """Run a single health check in a thread pool."""
        start = time.time()
        try:
            healthy, detail = await asyncio.to_thread(check.check_fn)
            latency_ms = (time.time() - start) * 1000

            with self._lock:
                status = self._status[check.name]
                status.last_check = time.time()
                status.check_latency_ms = latency_ms
                status.detail = detail

                if healthy:
                    if not status.is_healthy:
                        logger.info(
                            "Service %s recovered (was down for %.0fs)",
                            check.name,
                            status.downtime_seconds,
                        )
                    status.state = ServiceState.healthy
                    status.last_healthy = time.time()
                    status.consecutive_failures = 0
                    status.healthy_detail = detail  # Save for grayed display when down
                else:
                    status.consecutive_failures += 1
                    if status.state == ServiceState.healthy:
                        logger.warning(
                            "Service %s became unhealthy: %s",
                            check.name,
                            detail,
                        )
                    status.state = ServiceState.unhealthy

        except Exception as e:
            with self._lock:
                status = self._status[check.name]
                status.state = ServiceState.unhealthy
                status.detail = str(e)[:100]
                status.consecutive_failures += 1
                status.last_check = time.time()


# =============================================================================
# Factory for common configurations
# =============================================================================


def create_service_monitor(
    facility: str | None = None,
    ssh_host: str | None = None,
    access_method: str | None = None,
    auth_type: str | None = None,
    wiki_url: str | None = None,
    check_graph: bool = True,
    check_embed: bool = True,
    check_ssh: bool = True,
    check_auth: bool = True,
    poll_interval: float = 15.0,
) -> ServiceMonitor:
    """Create a ServiceMonitor with standard checks for discovery CLIs.

    Registers separate checks for each concern:

    - ``graph``: Neo4j connectivity
    - ``embed``: Embedding server health
    - ``ssh``: SSH connectivity to remote host (critical — workers pause)
    - ``auth``: Wiki page reachability via HTTP (critical — workers pause)

    Args:
        facility: Facility ID (for loading config)
        ssh_host: SSH host alias for connectivity check
        access_method: Access method ("vpn", "tunnel", "direct", etc.)
        auth_type: Web auth type ("tequila", "keycloak", "basic", "vpn", etc.)
        wiki_url: Primary wiki URL for auth/reachability probing
        check_graph: Include Neo4j health check
        check_embed: Include embedding server health check
        check_ssh: Include SSH connectivity check
        check_auth: Include wiki auth/reachability check
        poll_interval: Default polling interval in seconds

    Returns:
        Configured (but not started) ServiceMonitor
    """
    monitor = ServiceMonitor(poll_interval=poll_interval)

    if check_graph:
        monitor.add_check(
            "graph",
            neo4j_health_check,
            poll_interval=30.0,  # Graph rarely goes down
            critical=True,
        )

    if check_embed:
        monitor.add_check(
            "embed",
            embed_health_check,
            poll_interval=20.0,
            critical=False,  # Ingest workers handle this via EmbeddingResilience
        )

    if check_ssh and ssh_host:
        monitor.add_check(
            "ssh",
            lambda host=ssh_host: ssh_health_check(host),
            poll_interval=poll_interval,
            critical=True,  # Workers need SSH to fetch content
        )

    if check_auth and wiki_url:
        # Determine effective auth label for display
        if access_method in ("vpn", "tunnel"):
            effective_auth = "vpn"
        elif auth_type and auth_type != "none":
            effective_auth = auth_type
        else:
            effective_auth = "http"

        # Determine whether wiki probe goes through SSH or direct HTTP.
        # Tunnel and VPN sites both curl via SSH (workers do the same),
        # so the auth check should validate the same path.
        if access_method in ("tunnel", "vpn") and ssh_host:
            _probe_ssh = ssh_host
        elif ssh_host and access_method not in ("direct",):
            _probe_ssh = ssh_host  # SSH-only sites curl via SSH
        else:
            _probe_ssh = None  # Direct HTTP sites

        def _auth_check(
            url: str = wiki_url,
            host: str | None = _probe_ssh,
            label: str = effective_auth,
        ) -> tuple[bool, str]:
            healthy, detail = wiki_auth_check(url, host)
            if healthy:
                return True, label  # Display label, not URL
            return False, detail

        monitor.add_check(
            "auth",
            _auth_check,
            auth_label=effective_auth,
            poll_interval=poll_interval,
            critical=True,  # Workers need wiki access
        )

    return monitor


# =============================================================================
# Display helpers
# =============================================================================


def build_servers_row(
    statuses: list[ServiceStatus],
    width: int = 80,
) -> str | None:
    """Build a SERVERS row string for progress displays.

    Returns None if no services are registered.

    Format:
      SERVERS  graph:healthy(bolt://localhost:7687)  embed:healthy(iter-login)  ssh:healthy(vpn)
    """
    if not statuses:
        return None

    parts: list[tuple[str, str]] = []  # (text, style)

    for s in statuses:
        # Build compact status string
        if s.state == ServiceState.healthy:
            style = "green"
            state_str = s.detail or "ok"
        elif s.state == ServiceState.unknown:
            style = "dim"
            state_str = "unknown"
        elif s.state == ServiceState.recovering:
            style = "yellow"
            state_str = f"recovering ({int(s.downtime_seconds)}s)"
        else:
            # Unhealthy: show grayed version of healthy state, not red error
            style = "dim"
            if s.healthy_detail:
                state_str = s.healthy_detail
            elif s.auth_label:
                state_str = f"{s.auth_label}"
            else:
                state_str = s.detail[:30] if s.detail else "down"
            if s.downtime_seconds > 0:
                state_str += f" ({int(s.downtime_seconds)}s down)"

        parts.append((f"{s.name}:{state_str}", style))

    return parts  # type: ignore[return-value]


__all__ = [
    "ServiceMonitor",
    "ServiceState",
    "ServiceStatus",
    "build_servers_row",
    "create_service_monitor",
    "embed_health_check",
    "get_graph_location",
    "neo4j_health_check",
    "ssh_health_check",
]
