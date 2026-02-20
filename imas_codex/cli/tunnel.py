"""Unified SSH tunnel management CLI.

Manages SSH tunnels to all remote services (Neo4j graph, embedding server)
from a single top-level command group.  Uses ``autossh`` when available for
automatic reconnection, otherwise falls back to plain ``ssh -f -N``.

All requested port forwards are batched into **one SSH connection** per host
(single ProxyJump, single auth).

Persistent tunnels are managed via systemd user services with autossh,
providing automatic reconnection after network interruptions, VPN drops,
or SSH keepalive timeouts.

Usage::

    imas-codex tunnel start [HOST]           # All services
    imas-codex tunnel start [HOST] --neo4j   # Just graph ports
    imas-codex tunnel start [HOST] --embed   # Just embed port
    imas-codex tunnel start [HOST] --llm     # Just LLM proxy port
    imas-codex tunnel stop [HOST]
    imas-codex tunnel status
    imas-codex tunnel service install [HOST]  # Persistent autossh via systemd
"""

import os
import shutil
import subprocess
from pathlib import Path

import click

# ============================================================================
# Helpers
# ============================================================================


def _record_tunnel_pid(host: str, first_local_port: int) -> None:
    """Find and record the PID of the autossh/ssh process we just started."""
    from imas_codex.remote.tunnel import _find_and_record_pid

    _find_and_record_pid(host, first_local_port)


def _discover_compute_node(host: str) -> str | None:
    """SSH to host and discover the SLURM compute node for imas-codex services.

    Delegates to the shared ``discover_compute_node`` in the tunnel module.
    """
    from imas_codex.remote.tunnel import discover_compute_node

    return discover_compute_node(host)


def _resolve_host(host: str | None) -> str:
    """Resolve HOST from argument, graph profile, or fail."""
    if host:
        return host

    try:
        from imas_codex.graph.profiles import resolve_neo4j

        profile = resolve_neo4j(auto_tunnel=False)
        if profile.host:
            return profile.host
    except Exception:
        pass

    raise click.ClickException(
        "No HOST specified and the active graph profile has no remote host.\n"
        "Provide a host: imas-codex tunnel start <host>"
    )


def _get_tunnel_ports(
    host: str,
    neo4j: bool,
    embed: bool,
    llm: bool = False,
) -> list[tuple[int, int, str, str]]:
    """Return list of (remote_port, local_port, label, remote_bind) for requested services.

    When no service flag is given, returns all known ports.
    ``remote_bind`` is the hostname to bind on the remote side of the tunnel
    (default ``"127.0.0.1"``).  When services run on a SLURM compute node,
    ``remote_bind`` is set to that node's hostname so the tunnel forwards
    ``local_port → compute_node:remote_port`` through the SSH jump host.
    """
    from imas_codex.remote.tunnel import TUNNEL_OFFSET

    ports: list[tuple[int, int, str, str]] = []
    all_services = not neo4j and not embed and not llm

    # Discover SLURM compute node if any service uses slurm scheduler.
    # All services share the same allocation, so one lookup suffices.
    compute_node: str | None = None
    try:
        from imas_codex.settings import (
            get_embed_scheduler,
            get_graph_scheduler,
            get_llm_scheduler,
        )

        if any(
            s() == "slurm"
            for s in [get_embed_scheduler, get_graph_scheduler, get_llm_scheduler]
        ):
            compute_node = _discover_compute_node(host)
            if compute_node:
                click.echo(f"  SLURM compute node: {compute_node}")
    except Exception:
        pass

    if neo4j or all_services:
        try:
            from imas_codex.graph.profiles import resolve_neo4j

            profile = resolve_neo4j(auto_tunnel=False)
            bind = compute_node or "127.0.0.1"
            ports.append(
                (
                    profile.bolt_port,
                    profile.bolt_port + TUNNEL_OFFSET,
                    f"neo4j-bolt ({profile.name})",
                    bind,
                )
            )
            ports.append(
                (
                    profile.http_port,
                    profile.http_port + TUNNEL_OFFSET,
                    f"neo4j-http ({profile.name})",
                    bind,
                )
            )
        except Exception:
            # Fallback: use base ports from convention constants
            from imas_codex.graph.profiles import BOLT_BASE_PORT, HTTP_BASE_PORT

            bind = compute_node or "127.0.0.1"
            ports.append(
                (
                    BOLT_BASE_PORT,
                    BOLT_BASE_PORT + TUNNEL_OFFSET,
                    "neo4j-bolt",
                    bind,
                )
            )
            ports.append(
                (
                    HTTP_BASE_PORT,
                    HTTP_BASE_PORT + TUNNEL_OFFSET,
                    "neo4j-http",
                    bind,
                )
            )

    if embed or all_services:
        from imas_codex.settings import get_embed_server_port

        embed_port = get_embed_server_port()
        # When a SLURM compute node is active, forward through it.
        # Otherwise fall back to localhost (embed on login node).
        remote_bind = compute_node or "127.0.0.1"
        # Embed uses same-port forwarding (no offset)
        ports.append((embed_port, embed_port, "embed", remote_bind))

    if llm or all_services:
        from imas_codex.settings import get_llm_proxy_port

        llm_port = get_llm_proxy_port()
        # LLM proxy runs on login node (needs outbound internet)
        ports.append((llm_port, llm_port, "llm", "127.0.0.1"))

    return ports


def _start_tunnels(
    host: str,
    ports: list[tuple[int, int, str, str]],
    use_autossh: bool,
) -> int:
    """Start port-forward tunnels in a single SSH connection.

    All requested forwards are passed as multiple ``-L`` flags to one
    ``autossh``/``ssh`` process so there is only one ProxyJump negotiation.
    Ports that are already bound locally are skipped and reported.

    Returns the number of active tunnels (including pre-existing ones).
    """
    import time

    from imas_codex.remote.tunnel import is_tunnel_active

    already_active: list[tuple[int, str]] = []
    to_start: list[tuple[int, int, str, str]] = []

    for remote_port, local_port, label, remote_bind in ports:
        if is_tunnel_active(local_port):
            already_active.append((local_port, label))
        else:
            to_start.append((remote_port, local_port, label, remote_bind))

    for local_port, label in already_active:
        click.echo(f"  {label}: already active (localhost:{local_port})")

    if not to_start:
        return len(already_active)

    # Build -L flags for all ports that need tunneling
    forward_args: list[str] = []
    for remote_port, local_port, _label, remote_bind in to_start:
        forward_args.extend(["-L", f"{local_port}:{remote_bind}:{remote_port}"])

    # Use shared SSH options from tunnel module (keepalives, no ControlMaster)
    from imas_codex.remote.tunnel import SSH_TUNNEL_OPTS

    if use_autossh:
        cmd = [
            "autossh", "-M", "0", "-f", "-N",
            *SSH_TUNNEL_OPTS, *forward_args, host,
        ]  # fmt: skip
        env = {
            **os.environ,
            "AUTOSSH_GATETIME": "0",
            "AUTOSSH_POLL": "30",
        }
    else:
        cmd = [
            "ssh", "-f", "-N",
            *SSH_TUNNEL_OPTS,
            *forward_args,
            host,
        ]  # fmt: skip
        env = None

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else str(e)
        click.echo(f"  FAILED: {stderr}")
        return len(already_active)
    except subprocess.TimeoutExpired:
        click.echo("  TIMEOUT waiting for SSH connection")
        return len(already_active)

    # Record the autossh/ssh PID for targeted cleanup later.
    # autossh -f / ssh -f fork into the background, so we find the PID
    # by matching our specific -L forward pattern.
    _record_tunnel_pid(host, to_start[0][1])

    # Wait for the forked ssh to bind the ports.
    # ssh -f forks AFTER connection + forwarding setup, so ports
    # should be available quickly.  Retry to account for slow auth.
    time.sleep(0.5)
    new_ok = 0
    for _attempt in range(5):
        new_ok = 0
        all_up = True
        for _remote_port, local_port, _label, _remote_bind in to_start:
            if is_tunnel_active(local_port):
                new_ok += 1
            else:
                all_up = False
        if all_up:
            break
        time.sleep(1.0)

    for remote_port, local_port, label, remote_bind in to_start:
        if is_tunnel_active(local_port):
            click.echo(
                f"  {label}: localhost:{local_port} → {host}→{remote_bind}:{remote_port}"
            )
        else:
            click.echo(f"  {label}: FAILED (port {local_port} not reachable)")

    return len(already_active) + new_ok


# ============================================================================
# Keyring D-Bus Forwarding
# ============================================================================


def _start_keyring_forward(host: str) -> None:
    """Forward the D-Bus socket from host for keyring access.

    Sets up a unix socket forward from ``host:/run/user/<uid>/bus``
    to a local socket, enabling SecretService keyring access on
    nodes without their own gnome-keyring-daemon.
    """
    from imas_codex.discovery.wiki.auth import (
        _dbus_forward_socket,
        _forward_dbus_socket,
    )

    if _dbus_forward_socket:
        click.echo(f"  keyring: already forwarded ({_dbus_forward_socket})")
        return

    click.echo(f"Forwarding D-Bus socket from {host}...")
    if _forward_dbus_socket(host):
        click.echo(f"✓ Keyring D-Bus forwarded via {host}")
        click.echo(f"  Socket: {_dbus_forward_socket}")

        # Verify keyring works
        try:
            import keyring

            backend = keyring.get_keyring()
            backend_module = type(backend).__module__ or ""
            if "fail" not in backend_module and "null" not in backend_module:
                click.echo(f"  Backend: {type(backend).__name__}")
            else:
                click.echo("  ⚠ Keyring backend still not functional after forwarding")
        except Exception as e:
            click.echo(f"  ⚠ Keyring check failed: {e}")
    else:
        click.echo("✗ D-Bus forwarding failed")
        click.echo(f"  Verify SSH access: ssh -o BatchMode=yes {host} echo ok")
        raise SystemExit(1)


def _get_dbus_forward_status() -> str | None:
    """Check if a D-Bus forward socket exists and return its path."""
    import os
    from pathlib import Path

    uid = os.getuid()
    candidates = [
        Path(f"/run/user/{uid}/dbus-forward.sock"),
        Path(f"/tmp/dbus-forward-{uid}.sock"),  # noqa: S108
    ]
    for sock in candidates:
        if sock.exists():
            return str(sock)
    return None


# ============================================================================
# Command Group
# ============================================================================


@click.group()
def tunnel() -> None:
    """Manage SSH tunnels to remote services.

    Forward remote Neo4j and embedding server ports to localhost
    for transparent access from your workstation.

    \b
      imas-codex tunnel start [HOST]         Start tunnels (all services)
      imas-codex tunnel start HOST --neo4j   Just graph ports
      imas-codex tunnel start HOST --embed   Just embedding port
      imas-codex tunnel start HOST --llm     Just LLM proxy port
      imas-codex tunnel start HOST --keyring D-Bus socket for keyring
      imas-codex tunnel stop [HOST]          Stop tunnels
      imas-codex tunnel status               Show active tunnels
      imas-codex tunnel keyring HOST         Forward keyring D-Bus socket
      imas-codex tunnel service install      Persistent autossh via systemd
    """
    pass


@tunnel.command("start")
@click.argument("host", required=False)
@click.option("--neo4j", "neo4j_only", is_flag=True, help="Tunnel Neo4j ports only")
@click.option("--embed", "embed_only", is_flag=True, help="Tunnel embedding port only")
@click.option("--llm", "llm_only", is_flag=True, help="Tunnel LLM proxy port only")
@click.option(
    "--keyring",
    "keyring_only",
    is_flag=True,
    help="Forward D-Bus socket for keyring access (SLURM compute nodes)",
)
def tunnel_start(
    host: str | None,
    neo4j_only: bool,
    embed_only: bool,
    llm_only: bool = False,
    keyring_only: bool = False,
) -> None:
    """Start SSH tunnels to remote services.

    Uses autossh when available for automatic reconnection, otherwise
    falls back to plain ssh.  HOST defaults to the active graph profile's
    configured host.

    The --keyring flag forwards the D-Bus socket from HOST to the local
    machine, enabling keyring access on SLURM compute nodes that lack
    their own SecretService daemon.

    \b
    Examples:
      imas-codex tunnel start iter           # All services
      imas-codex tunnel start iter --neo4j   # Just graph
      imas-codex tunnel start iter --embed   # Just embedding
      imas-codex tunnel start iter --llm     # Just LLM proxy
      imas-codex tunnel start iter --keyring # D-Bus for keyring
    """
    target = _resolve_host(host)

    # Handle keyring-only mode separately (unix socket, not TCP port)
    if keyring_only:
        _start_keyring_forward(target)
        return

    use_autossh = bool(shutil.which("autossh"))

    if use_autossh:
        click.echo(f"Starting tunnels to {target} (autossh):")
    else:
        click.echo(
            f"Starting tunnels to {target} (ssh — install autossh for auto-reconnect):"
        )

    ports = _get_tunnel_ports(target, neo4j_only, embed_only, llm_only)
    ok = _start_tunnels(target, ports, use_autossh)

    if ok == len(ports):
        click.echo(f"✓ All {ok} tunnel(s) active")
    elif ok > 0:
        click.echo(f"⚠ {ok}/{len(ports)} tunnel(s) active")
    else:
        raise click.ClickException("No tunnels could be established.")


@tunnel.command("stop")
@click.argument("host", required=False)
@click.option("--all", "stop_all", is_flag=True, help="Stop tunnels to all hosts")
def tunnel_stop(host: str | None, stop_all: bool) -> None:
    """Stop SSH tunnels.

    Without HOST, stops tunnels to all configured remote hosts.
    With HOST, stops only tunnels to that specific host.

    \b
    Examples:
      imas-codex tunnel stop           # Stop all tunnels
      imas-codex tunnel stop iter      # Stop only iter tunnels
    """
    from imas_codex.graph.profiles import _get_all_hosts
    from imas_codex.remote.tunnel import stop_tunnel

    if host:
        # Explicit host — stop just that one
        if stop_tunnel(host):
            click.echo(f"✓ Tunnels to {host} stopped")
        else:
            click.echo(f"No active tunnels to {host} found")
        return

    # No host specified — stop all configured remote hosts
    hosts = list(_get_all_hosts().values())
    if not hosts:
        click.echo("No remote hosts configured")
        return

    stopped: list[str] = []
    for h in sorted(set(hosts)):
        if stop_tunnel(h):
            stopped.append(h)

    if stopped:
        click.echo(f"✓ Tunnels stopped: {', '.join(stopped)}")
    else:
        click.echo("No active tunnels found")


@tunnel.command("status")
def tunnel_status() -> None:
    """Show active SSH tunnels across all services.

    Scans known Neo4j and embedding server ports (both direct and
    tunneled) for anything listening.  Ports bound by an SSH process
    are labeled accordingly.
    """
    from imas_codex.remote.tunnel import TUNNEL_OFFSET, is_tunnel_active
    from imas_codex.settings import get_embed_server_port, get_llm_proxy_port

    embed_port = get_embed_server_port()
    llm_port = get_llm_proxy_port()

    # Build port→label map using the same resolution as tunnel_start
    # so that labels match (graph name "codex", not location "iter").
    known_ports: dict[int, str] = {}
    try:
        from imas_codex.graph.profiles import resolve_neo4j

        profile = resolve_neo4j(auto_tunnel=False)
        name = profile.name
        known_ports[profile.bolt_port] = f"neo4j-bolt ({name})"
        known_ports[profile.bolt_port + TUNNEL_OFFSET] = (
            f"neo4j-bolt ({name}, tunneled)"
        )
        known_ports[profile.http_port] = f"neo4j-http ({name})"
        known_ports[profile.http_port + TUNNEL_OFFSET] = (
            f"neo4j-http ({name}, tunneled)"
        )
    except Exception:
        from imas_codex.graph.profiles import BOLT_BASE_PORT, HTTP_BASE_PORT

        known_ports[BOLT_BASE_PORT] = "neo4j-bolt"
        known_ports[BOLT_BASE_PORT + TUNNEL_OFFSET] = "neo4j-bolt (tunneled)"
        known_ports[HTTP_BASE_PORT] = "neo4j-http"
        known_ports[HTTP_BASE_PORT + TUNNEL_OFFSET] = "neo4j-http (tunneled)"
    known_ports[embed_port] = "embed"
    known_ports[llm_port] = "llm"

    # Build port→location map for SSH-forwarded ports so we can show
    # "iter" (or whichever host) instead of a generic "(ssh)" marker.
    port_host: dict[int, str] = {}
    try:
        host = profile.host  # type: ignore[possibly-undefined]
        for p in known_ports:
            if p >= TUNNEL_OFFSET:
                port_host[p] = host
        port_host[embed_port] = host
    except Exception:
        pass

    # Identify which ports are bound by SSH (via ss -tlnp)
    ssh_ports: set[int] = set()
    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "ssh" not in line.lower():
                    continue
                for port in known_ports:
                    if f":{port}" in line:
                        ssh_ports.add(port)
    except Exception:
        pass

    # Check all known ports for liveness
    tunnels: list[tuple[int, str, bool]] = []
    for port, label in sorted(known_ports.items()):
        if is_tunnel_active(port):
            is_ssh = port in ssh_ports
            tunnels.append((port, label, is_ssh))

    if tunnels:
        click.echo("Active tunnels:")
        for port, label, is_ssh in tunnels:
            if is_ssh:
                location = port_host.get(port, "ssh")
                marker = f" ({location})"
            else:
                marker = ""
            click.echo(f"  :{port}  {label}{marker}")
    else:
        click.echo("No active tunnels on known service ports")

    # Check D-Bus keyring forwarding
    dbus_sock = _get_dbus_forward_status()
    if dbus_sock:
        click.echo(f"\nD-Bus keyring forward: {dbus_sock}")


# ============================================================================
# Systemd Service Subcommand
# ============================================================================


@tunnel.command("service")
@click.argument(
    "action",
    type=click.Choice(["install", "uninstall", "status", "start", "stop", "logs"]),
)
@click.argument("host", required=False)
@click.option("--neo4j", "neo4j_only", is_flag=True, help="Tunnel Neo4j ports only")
@click.option("--embed", "embed_only", is_flag=True, help="Tunnel embedding port only")
@click.option("--llm", "llm_only", is_flag=True, help="Tunnel LLM proxy port only")
def tunnel_service(
    action: str,
    host: str | None,
    neo4j_only: bool,
    embed_only: bool,
    llm_only: bool = False,
) -> None:
    """Manage persistent SSH tunnels via systemd + autossh.

    Installs a systemd user service that maintains reconnecting SSH
    tunnels to the specified HOST for Neo4j, embedding, and/or LLM services.

    \b
    Examples:
      imas-codex tunnel service install iter         # All services
      imas-codex tunnel service install iter --neo4j # Just graph
      imas-codex tunnel service install iter --embed # Just embedding
      imas-codex tunnel service install iter --llm   # Just LLM proxy
      imas-codex tunnel service start iter
      imas-codex tunnel service status iter
      imas-codex tunnel service logs iter
    """
    import platform

    if platform.system() != "Linux":
        raise click.ClickException("systemd services only supported on Linux")
    if not shutil.which("systemctl"):
        raise click.ClickException("systemctl not found")

    target = _resolve_host(host)
    service_name = f"imas-codex-tunnel-{target}"
    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_file = service_dir / f"{service_name}.service"

    if action == "install":
        autossh = shutil.which("autossh")
        if not autossh:
            raise click.ClickException(
                "autossh not found. Install with: sudo apt install autossh"
            )

        ports = _get_tunnel_ports(target, neo4j_only, embed_only, llm_only)
        if not ports:
            raise click.ClickException("No services selected for tunneling.")

        forwards_str = " ".join(
            f"-L {local}:{remote_bind}:{remote}"
            for remote, local, _label, remote_bind in ports
        )

        # Log directory for autossh diagnostics
        log_dir = Path.home() / ".local" / "share" / "imas-codex" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        autossh_log = log_dir / f"autossh-{target}.log"

        # Resolve user SSH config for the service environment.
        # systemd user services don't inherit shell profile env; SSH
        # needs to find the user's config for ProxyJump, keys, etc.
        ssh_config_opt = f'-o "UserKnownHostsFile={Path.home()}/.ssh/known_hosts"'

        service_content = f"""\
[Unit]
Description=IMAS Codex SSH tunnels to {target}
Documentation=https://github.com/iterorganization/imas-codex
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=10

[Service]
Type=simple

# autossh environment:
#   GATETIME=0:  don't treat quick ssh exit as failure (retry immediately)
#   POLL=30:     check tunnel health every 30s
#   LOGFILE:     write autossh diagnostics to rotating log
#   PORT=0:      disable autossh's monitoring port (we use ServerAliveInterval)
Environment="AUTOSSH_GATETIME=0"
Environment="AUTOSSH_POLL=30"
Environment="AUTOSSH_LOGFILE={autossh_log}"
Environment="AUTOSSH_PORT=0"
Environment="HOME={Path.home()}"

ExecStart={autossh} -M 0 -N \\
    -o "ControlMaster=no" \\
    -o "ControlPath=none" \\
    -o "TCPKeepAlive=yes" \\
    -o "ServerAliveInterval=15" \\
    -o "ServerAliveCountMax=3" \\
    -o "ConnectTimeout=10" \\
    {ssh_config_opt} \\
    {forwards_str} \\
    {target}

# Restart policy: always restart with backoff
Restart=always
RestartSec=5

# Watchdog: systemd kills the service if it hangs
WatchdogSec=300

[Install]
WantedBy=default.target
"""
        service_dir.mkdir(parents=True, exist_ok=True)
        service_file.write_text(service_content)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", service_name], check=True)
        click.echo(f"✓ Tunnel service installed: {service_name}")
        click.echo(f"  Host: {target}")
        click.echo("  Ports:")
        for remote_port, local_port, label, remote_bind in ports:
            click.echo(
                f"    {label}: localhost:{local_port} → {remote_bind}:{remote_port}"
            )
        click.echo(f"  Log: {autossh_log}")
        click.echo(f"  Start: imas-codex tunnel service start {target}")

        # Check loginctl linger status
        try:
            result = subprocess.run(
                ["loginctl", "show-user", os.environ.get("USER", ""), "-p", "Linger"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "Linger=no" in result.stdout:
                click.echo(
                    "\n  ⚠ User linger is disabled. The service will stop on logout."
                )
                click.echo("  Enable persistence: sudo loginctl enable-linger $USER")
        except Exception:
            pass  # Best-effort linger check

    elif action == "uninstall":
        if not service_file.exists():
            click.echo(f"Service {service_name} not installed")
            return
        subprocess.run(
            ["systemctl", "--user", "stop", service_name], capture_output=True
        )
        subprocess.run(
            ["systemctl", "--user", "disable", service_name], capture_output=True
        )
        service_file.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        click.echo(f"Service {service_name} uninstalled")

    elif action == "status":
        if not service_file.exists():
            click.echo(f"Service {service_name} not installed")
            click.echo(f"  Install: imas-codex tunnel service install {target}")
            return
        result = subprocess.run(
            ["systemctl", "--user", "status", service_name],
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr)

    elif action == "start":
        if not service_file.exists():
            raise click.ClickException(
                f"Service not installed. Run: imas-codex tunnel service install {target}"
            )
        subprocess.run(["systemctl", "--user", "start", service_name], check=True)
        click.echo(f"Tunnel service to {target} started")

    elif action == "stop":
        subprocess.run(["systemctl", "--user", "stop", service_name], check=True)
        click.echo(f"Tunnel service to {target} stopped")

    elif action == "logs":
        subprocess.run(["journalctl", "--user", "-u", service_name, "-n", "50", "-f"])
