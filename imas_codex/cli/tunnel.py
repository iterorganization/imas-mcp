"""Unified SSH tunnel management CLI.

Manages SSH tunnels to all remote services (Neo4j graph, embedding server)
from a single top-level command group.  Uses ``autossh`` when available for
automatic reconnection, otherwise falls back to plain ``ssh -f -N``.

Usage::

    imas-codex tunnel start [HOST]           # All services
    imas-codex tunnel start [HOST] --neo4j   # Just graph ports
    imas-codex tunnel start [HOST] --embed   # Just embed port
    imas-codex tunnel stop [HOST]
    imas-codex tunnel status
    imas-codex tunnel service install [HOST]  # Persistent autossh via systemd
"""

import shutil
import subprocess
from pathlib import Path

import click

# ============================================================================
# Helpers
# ============================================================================


def _resolve_host(host: str | None) -> str:
    """Resolve HOST from argument, graph profile, or fail."""
    if host:
        return host

    try:
        from imas_codex.graph.profiles import resolve_graph

        profile = resolve_graph(auto_tunnel=False)
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
) -> list[tuple[int, int, str]]:
    """Return list of (remote_port, local_port, label) for requested services.

    When neither --neo4j nor --embed is given, returns all known ports.
    """
    from imas_codex.remote.tunnel import TUNNEL_OFFSET

    ports: list[tuple[int, int, str]] = []
    all_services = not neo4j and not embed

    if neo4j or all_services:
        try:
            from imas_codex.graph.profiles import resolve_graph

            profile = resolve_graph(auto_tunnel=False)
            ports.append(
                (
                    profile.bolt_port,
                    profile.bolt_port + TUNNEL_OFFSET,
                    f"neo4j-bolt ({profile.name})",
                )
            )
            ports.append(
                (
                    profile.http_port,
                    profile.http_port + TUNNEL_OFFSET,
                    f"neo4j-http ({profile.name})",
                )
            )
        except Exception:
            # Fallback: use default bolt/http ports
            ports.append((7687, 7687 + TUNNEL_OFFSET, "neo4j-bolt"))
            ports.append((7474, 7474 + TUNNEL_OFFSET, "neo4j-http"))

    if embed or all_services:
        from imas_codex.settings import get_embed_server_port

        embed_port = get_embed_server_port()
        # Embed uses same-port forwarding (no offset)
        ports.append((embed_port, embed_port, "embed"))

    return ports


def _start_single_tunnel(
    host: str,
    remote_port: int,
    local_port: int,
    label: str,
    use_autossh: bool,
) -> bool:
    """Start a single port-forward tunnel. Returns True on success."""
    from imas_codex.remote.tunnel import is_tunnel_active

    if is_tunnel_active(local_port):
        click.echo(f"  {label}: already active (localhost:{local_port})")
        return True

    forward = f"{local_port}:127.0.0.1:{remote_port}"

    if use_autossh:
        cmd = [
            "autossh",
            "-M",
            "0",
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
            "ExitOnForwardFailure=yes",
            "-L",
            forward,
            host,
        ]
    else:
        cmd = [
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
            forward,
            host,
        ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
    except subprocess.CalledProcessError as e:
        click.echo(f"  {label}: FAILED ({e.stderr.strip() if e.stderr else e})")
        return False
    except subprocess.TimeoutExpired:
        click.echo(f"  {label}: TIMEOUT")
        return False

    import time

    time.sleep(0.5)

    # Retry port check — autossh may need time after forking
    for _ in range(3):
        if is_tunnel_active(local_port):
            click.echo(f"  {label}: localhost:{local_port} → {host}:{remote_port}")
            return True
        time.sleep(1.0)

    click.echo(f"  {label}: started but port {local_port} not reachable")
    return False


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
      imas-codex tunnel stop [HOST]          Stop tunnels
      imas-codex tunnel status               Show active tunnels
      imas-codex tunnel service install      Persistent autossh via systemd
    """
    pass


@tunnel.command("start")
@click.argument("host", required=False)
@click.option("--neo4j", "neo4j_only", is_flag=True, help="Tunnel Neo4j ports only")
@click.option("--embed", "embed_only", is_flag=True, help="Tunnel embedding port only")
@click.option(
    "--graph",
    "-g",
    envvar="IMAS_CODEX_GRAPH",
    default=None,
    help="Graph profile (for Neo4j port resolution)",
)
def tunnel_start(
    host: str | None,
    neo4j_only: bool,
    embed_only: bool,
    graph: str | None,
) -> None:
    """Start SSH tunnels to remote services.

    Uses autossh when available for automatic reconnection, otherwise
    falls back to plain ssh.  HOST defaults to the active graph profile's
    configured host.

    \b
    Examples:
      imas-codex tunnel start iter           # All services
      imas-codex tunnel start iter --neo4j   # Just graph
      imas-codex tunnel start iter --embed   # Just embedding
    """
    target = _resolve_host(host)
    use_autossh = bool(shutil.which("autossh"))

    if use_autossh:
        click.echo(f"Starting tunnels to {target} (autossh):")
    else:
        click.echo(
            f"Starting tunnels to {target} (ssh — install autossh for auto-reconnect):"
        )

    ports = _get_tunnel_ports(target, neo4j_only, embed_only)
    ok = 0
    for remote_port, local_port, label in ports:
        if _start_single_tunnel(target, remote_port, local_port, label, use_autossh):
            ok += 1

    if ok == len(ports):
        click.echo(f"✓ All {ok} tunnel(s) active")
    elif ok > 0:
        click.echo(f"⚠ {ok}/{len(ports)} tunnel(s) active")
    else:
        raise click.ClickException("No tunnels could be established.")


@tunnel.command("stop")
@click.argument("host", required=False)
def tunnel_stop(host: str | None) -> None:
    """Stop SSH tunnels to a remote host.

    Tries ControlMaster exit first, falls back to pkill.
    HOST defaults to the active graph profile's host.
    """
    from imas_codex.remote.tunnel import stop_tunnel

    target = _resolve_host(host)

    if stop_tunnel(target):
        click.echo(f"✓ Tunnels to {target} stopped")
    else:
        click.echo(f"No active tunnels to {target} found")


@tunnel.command("status")
def tunnel_status() -> None:
    """Show active SSH tunnels across all services.

    Scans for SSH-bound ports matching known Neo4j and embedding server
    ports across all graph profiles.
    """
    from imas_codex.graph.profiles import BOLT_BASE_PORT, _get_all_offsets
    from imas_codex.remote.tunnel import TUNNEL_OFFSET
    from imas_codex.settings import get_embed_server_port

    offsets = _get_all_offsets()
    embed_port = get_embed_server_port()

    # Build map of all known service ports → label
    known_ports: dict[int, str] = {}
    for name, offset in offsets.items():
        bolt = BOLT_BASE_PORT + offset
        http = bolt - 213  # http = bolt - 213 (7687→7474, 7688→7475, ...)
        known_ports[bolt] = f"neo4j-bolt ({name})"
        known_ports[bolt + TUNNEL_OFFSET] = f"neo4j-bolt ({name}, tunneled)"
        known_ports[http] = f"neo4j-http ({name})"
        known_ports[http + TUNNEL_OFFSET] = f"neo4j-http ({name}, tunneled)"
    known_ports[embed_port] = "embed"

    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            click.echo("Could not check tunnels (ss command failed)")
            return

        seen: set[int] = set()
        tunnels: list[tuple[int, str]] = []
        for line in result.stdout.splitlines():
            if "ssh" not in line.lower():
                continue
            for port, label in known_ports.items():
                if port not in seen and f":{port}" in line:
                    tunnels.append((port, label))
                    seen.add(port)

        if tunnels:
            click.echo("Active SSH tunnels:")
            for port, label in sorted(tunnels):
                click.echo(f"  :{port}  {label}")
        else:
            click.echo("No active SSH tunnels on known service ports")

    except Exception as e:
        click.echo(f"Could not check tunnels: {e}")


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
@click.option(
    "--graph",
    "-g",
    envvar="IMAS_CODEX_GRAPH",
    default=None,
    help="Graph profile (for Neo4j port resolution)",
)
def tunnel_service(
    action: str,
    host: str | None,
    neo4j_only: bool,
    embed_only: bool,
    graph: str | None,
) -> None:
    """Manage persistent SSH tunnels via systemd + autossh.

    Installs a systemd user service that maintains reconnecting SSH
    tunnels to the specified HOST for Neo4j and/or embedding services.

    \b
    Examples:
      imas-codex tunnel service install iter         # All services
      imas-codex tunnel service install iter --neo4j # Just graph
      imas-codex tunnel service install iter --embed # Just embedding
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

        ports = _get_tunnel_ports(target, neo4j_only, embed_only)
        if not ports:
            raise click.ClickException("No services selected for tunneling.")

        # Build -L flags for all ports
        forward_args = []
        for remote_port, local_port, _label in ports:
            forward_args.extend(["-L", f"{local_port}:127.0.0.1:{remote_port}"])

        forwards_str = " ".join(
            f"-L {local}:127.0.0.1:{remote}" for remote, local, _ in ports
        )

        service_content = f"""\
[Unit]
Description=IMAS Codex SSH tunnels to {target}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment="AUTOSSH_GATETIME=0"
Environment="AUTOSSH_POLL=60"
ExecStart={autossh} -M 0 -N \\
    -o "ControlMaster=no" \\
    -o "ControlPath=none" \\
    -o "ServerAliveInterval=30" \\
    -o "ServerAliveCountMax=3" \\
    -o "ExitOnForwardFailure=yes" \\
    {forwards_str} \\
    {target}
ExecStop=/bin/kill $MAINPID
Restart=always
RestartSec=10

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
        for remote_port, local_port, label in ports:
            click.echo(f"    {label}: localhost:{local_port} → {target}:{remote_port}")
        click.echo(f"  Start: imas-codex tunnel service start {target}")

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
