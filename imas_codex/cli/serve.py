"""Serve commands - MCP servers and embedding service."""

import logging
import os
from typing import Literal, cast

import click

from imas_codex import _get_dd_version

logger = logging.getLogger(__name__)


@click.group()
def serve() -> None:
    """Start MCP servers.

    \b
      imas-codex serve imas     Start the IMAS Data Dictionary server
      imas-codex serve agents   Start the Agents server for facility exploration
    """
    pass


@serve.command("imas")
@click.option(
    "--transport",
    envvar="TRANSPORT",
    default="streamable-http",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    help="Transport protocol (env: TRANSPORT)",
)
@click.option(
    "--host",
    envvar="HOST",
    default="127.0.0.1",
    help="Host to bind (env: HOST) for HTTP transports",
)
@click.option(
    "--port",
    envvar="PORT",
    default=8000,
    type=int,
    help="Port to bind (env: PORT) for HTTP transports",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
@click.option(
    "--no-rich",
    is_flag=True,
    help="Disable rich progress output during server initialization",
)
@click.option(
    "--ids-filter",
    envvar="IDS_FILTER",
    type=str,
    help=(
        "Specific IDS names to include as a space-separated string (env: IDS_FILTER) "
        "e.g., 'core_profiles equilibrium'"
    ),
)
@click.option(
    "--dd-version",
    "dd_version_opt",
    envvar="IMAS_DD_VERSION",
    type=str,
    default=None,
    help=(
        "IMAS Data Dictionary version to use (env: IMAS_DD_VERSION). "
        "Defaults to [tool.imas-codex].default-dd-version in pyproject.toml."
    ),
)
def serve_imas(
    transport: str,
    host: str,
    port: int,
    log_level: str,
    no_rich: bool,
    ids_filter: str,
    dd_version_opt: str | None,
) -> None:
    """Start the IMAS Data Dictionary MCP server.

    Examples:
        # Run with default streamable-http transport
        imas-codex serve imas

        # Run with custom host/port
        imas-codex serve imas --host 0.0.0.0 --port 9000

        # Run with stdio transport (for MCP clients)
        imas-codex serve imas --transport stdio

        # Run with debug logging
        imas-codex serve imas --log-level DEBUG

        # Run with specific DD version
        imas-codex serve imas --dd-version 3.42.2
    """
    # Resolve DD version with CLI option taking precedence
    if dd_version_opt:
        os.environ["IMAS_DD_VERSION"] = dd_version_opt

    effective_dd_version = _get_dd_version(dd_version_opt)

    # Configure logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, log_level))

    logger.debug(f"Set logging level to {log_level}")
    logger.debug(f"Starting IMAS MCP server with transport={transport}")

    # Log DD version with source indication
    if dd_version_opt:
        logger.info(f"IMAS DD version: {effective_dd_version} (--dd-version)")
    elif os.environ.get("IMAS_DD_VERSION"):
        logger.info(f"IMAS DD version: {effective_dd_version} (IMAS_DD_VERSION env)")
    else:
        logger.info(f"IMAS DD version: {effective_dd_version} (default)")

    # Parse ids_filter string into a set if provided
    ids_set: set | None = set(ids_filter.split()) if ids_filter else None
    if ids_set:
        logger.info(f"Starting server with IDS filter: {sorted(ids_set)}")
    else:
        logger.info("Starting server with all available IDS")

    # Log transport choice
    match transport:
        case "stdio":
            logger.debug("Using STDIO transport")
        case "streamable-http":
            logger.info(f"Using streamable-http transport on {host}:{port}")
        case _:
            logger.info(f"Using {transport} transport on {host}:{port}")

    # For stdio transport, always disable rich output
    use_rich = not no_rich and transport != "stdio"
    if transport == "stdio" and not no_rich:
        logger.info(
            "Disabled rich output for stdio transport to prevent protocol interference"
        )

    from imas_codex.server import Server

    server_instance = Server(use_rich=use_rich, ids_set=ids_set)

    server_instance.run(
        transport=cast(Literal["stdio", "sse", "streamable-http"], transport),
        host=host,
        port=port,
    )


@serve.command("agents")
@click.option(
    "--transport",
    envvar="AGENTS_TRANSPORT",
    default="stdio",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    help="Transport protocol (env: AGENTS_TRANSPORT)",
)
@click.option(
    "--host",
    envvar="AGENTS_HOST",
    default="127.0.0.1",
    help="Host to bind (env: AGENTS_HOST) for HTTP transports",
)
@click.option(
    "--port",
    envvar="AGENTS_PORT",
    default=8001,
    type=int,
    help="Port to bind (env: AGENTS_PORT) for HTTP transports",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
def serve_agents(
    transport: str,
    host: str,
    port: int,
    log_level: str,
) -> None:
    """Start the Agents MCP server for remote facility exploration.

    This server provides prompts for orchestrating subagents that explore
    remote fusion facilities via SSH.

    Examples:
        # Run with stdio transport (default, for Cursor)
        imas-codex serve agents

        # Run with HTTP transport
        imas-codex serve agents --transport streamable-http --port 8001
    """
    # Configure logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, log_level))

    logger.info("Starting Agents MCP server")

    match transport:
        case "stdio":
            logger.debug("Using STDIO transport")
        case _:
            logger.info(f"Using {transport} transport on {host}:{port}")

    from imas_codex.agentic.server import AgentsServer

    server = AgentsServer()
    server.run(
        transport=cast(Literal["stdio", "sse", "streamable-http"], transport),
        host=host,
        port=port,
    )


# ============================================================================
# Serve Embed Command Group
# ============================================================================


@serve.group("embed")
def serve_embed() -> None:
    """Manage GPU embedding server (local, systemd).

    \b
      imas-codex serve embed start      Start embedding server locally
      imas-codex serve embed status     Check server status
      imas-codex serve embed service    Manage systemd service
    """
    pass


@serve_embed.command("start")
@click.option(
    "--host",
    envvar="EMBED_HOST",
    default="0.0.0.0",
    help="Host to bind (default: all interfaces for SLURM compute node access)",
)
@click.option(
    "--port",
    envvar="EMBED_PORT",
    default=None,
    type=int,
    help="Port to bind (default: from config or 18765)",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
@click.option(
    "--gpu",
    envvar="CUDA_VISIBLE_DEVICES",
    default=None,
    help="CUDA device to use (e.g., 0 or 1)",
)
@click.option(
    "--idle-timeout",
    default=0,
    type=int,
    help="Auto-shutdown after N seconds of inactivity (0=disabled, 1800=30min)",
)
def embed_start(
    host: str,
    port: int | None,
    log_level: str,
    gpu: str | None,
    idle_timeout: int,
) -> None:
    """Start the embedding server.

    The server provides GPU-accelerated embedding via HTTP API.
    Access via SSH tunnel for security.

    Examples:
        # Start with defaults
        imas-codex serve embed start

        # Use specific GPU
        imas-codex serve embed start --gpu 1

        # Custom port
        imas-codex serve embed start --port 18766

        # Auto-shutdown after 30 min idle
        imas-codex serve embed start --idle-timeout 1800
    """
    # Configure logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, log_level))

    # Set GPU if specified
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        logger.info(f"Using CUDA device: {gpu}")

    # Get port from settings if not specified
    if port is None:
        from imas_codex.settings import get_embed_server_port

        port = get_embed_server_port()

    # Set idle timeout in server module
    if idle_timeout > 0:
        import imas_codex.embeddings.server as embed_server

        embed_server._idle_timeout = idle_timeout
        logger.info(f"Idle timeout: {idle_timeout}s")

    logger.info(f"Starting embedding server on {host}:{port}")

    try:
        import uvicorn

        from imas_codex.embeddings.server import app

        uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
    except ImportError as e:
        raise click.ClickException(
            f"Missing dependency: {e}. Install with: uv pip install uvicorn"
        ) from e


@serve_embed.command("status")
@click.option(
    "--url",
    default=None,
    help="Remote server URL (default: from config)",
)
@click.option("--local", is_flag=True, help="Check local embedding capability")
def embed_status(url: str | None, local: bool) -> None:
    """Check embedding server status.

    Shows server health, model info, uptime, and optionally local GPU capability.

    Examples:
        # Check remote server status
        imas-codex serve embed status

        # Check local capability
        imas-codex serve embed status --local
    """
    if local:
        # Check local embedding capability
        click.echo("Local embedding capability:")
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            click.echo("  PyTorch: installed")
            click.echo(f"  CUDA available: {cuda_available}")
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory // (
                    1024 * 1024
                )
                click.echo(f"  GPU: {device_name} ({memory} MB)")
        except ImportError:
            click.echo("  PyTorch: not installed")

        try:
            import importlib.util

            if importlib.util.find_spec("sentence_transformers"):
                click.echo("  SentenceTransformers: installed")
            else:
                click.echo("  SentenceTransformers: not installed")
        except Exception:
            click.echo("  SentenceTransformers: not installed")

        return

    # Check remote server health
    if url is None:
        from imas_codex.settings import get_embed_remote_url

        url = get_embed_remote_url()

    if not url:
        click.echo("No remote embedding server configured.")
        click.echo(
            "Set embed-remote-url in pyproject.toml or IMAS_CODEX_EMBED_REMOTE_URL env var."
        )
        return

    click.echo(f"Server ({url}):")

    from imas_codex.embeddings.client import RemoteEmbeddingClient

    client = RemoteEmbeddingClient(url)

    if client.is_available():
        info = client.get_detailed_info()
        if info:
            click.echo("  ✓ Healthy")
            click.echo(f"  Model: {info['model']['name']}")
            click.echo(f"  Device: {info['model']['device']}")
            click.echo(f"  Dimension: {info['model']['embedding_dimension']}")
            if info["gpu"]["name"]:
                click.echo(
                    f"  GPU: {info['gpu']['name']} ({info['gpu']['memory_mb']} MB)"
                )
            uptime_h = info["server"]["uptime_seconds"] / 3600
            idle_s = info["server"].get("idle_seconds", 0)
            timeout_s = info["server"].get("idle_timeout", 0)
            click.echo(f"  Uptime: {uptime_h:.1f}h")
            if timeout_s > 0:
                click.echo(f"  Idle: {idle_s:.0f}s / {timeout_s}s timeout")
    else:
        click.echo("  ✗ Not available")
        click.echo("  Ensure SSH tunnel is active: ssh -L 18765:127.0.0.1:18765 iter")
        click.echo(
            "  Or start the service on ITER: imas-codex serve embed service start"
        )


@serve_embed.command("service")
@click.argument(
    "action", type=click.Choice(["install", "uninstall", "status", "start", "stop"])
)
@click.option("--gpu", default="1", help="CUDA device to use (default: 1)")
def embed_service(action: str, gpu: str) -> None:
    """Manage embedding server as systemd user service.

    Examples:
        imas-codex serve embed service install
        imas-codex serve embed service start
        imas-codex serve embed service status
    """
    import platform
    import shutil
    import subprocess
    from pathlib import Path

    if platform.system() != "Linux":
        raise click.ClickException("systemd services only supported on Linux")

    if not shutil.which("systemctl"):
        raise click.ClickException("systemctl not found")

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_file = service_dir / "imas-codex-embed.service"

    if action == "install":
        service_dir.mkdir(parents=True, exist_ok=True)

        # Find uv path
        uv_path = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")

        # Find project directory
        project_dir = Path.home() / "imas-codex"
        if not project_dir.exists():
            # Try current working directory
            project_dir = Path.cwd()
            if not (project_dir / "pyproject.toml").exists():
                raise click.ClickException(
                    f"Project not found at {Path.home() / 'imas-codex'} or current directory"
                )

        from imas_codex.settings import get_embed_server_port

        port = get_embed_server_port()

        service_content = f"""[Unit]
Description=IMAS Codex Embedding Service (GPU)
After=network.target

[Service]
Type=simple
WorkingDirectory={Path.home()}
Environment="PATH={Path.home()}/.local/bin:/usr/local/bin:/usr/bin"
Environment="CUDA_VISIBLE_DEVICES={gpu}"
ExecStart={uv_path} run --extra gpu --project {project_dir} imas-codex serve embed start --host 0.0.0.0 --port {port}
ExecStop=/bin/kill -15 $MAINPID
TimeoutStopSec=30
Restart=on-failure
RestartSec=10
CPUQuota=400%
MemoryMax=8G
MemoryHigh=6G
Nice=5

[Install]
WantedBy=default.target
"""
        service_file.write_text(service_content)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "imas-codex-embed"], check=True
        )
        click.echo("✓ Service installed and enabled")
        click.echo("  Start: systemctl --user start imas-codex-embed")
        click.echo("  Or:    imas-codex serve embed service start")

    elif action == "uninstall":
        if not service_file.exists():
            click.echo("Service not installed")
            return
        subprocess.run(
            ["systemctl", "--user", "stop", "imas-codex-embed"], capture_output=True
        )
        subprocess.run(
            ["systemctl", "--user", "disable", "imas-codex-embed"], capture_output=True
        )
        service_file.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        click.echo("Service uninstalled")

    elif action == "status":
        if not service_file.exists():
            click.echo("Service not installed")
            return
        result = subprocess.run(
            ["systemctl", "--user", "status", "imas-codex-embed"],
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout)

    elif action == "start":
        if not service_file.exists():
            raise click.ClickException(
                "Service not installed. Run: imas-codex serve embed service install"
            )
        subprocess.run(["systemctl", "--user", "start", "imas-codex-embed"], check=True)
        click.echo("Service started")

    elif action == "stop":
        subprocess.run(["systemctl", "--user", "stop", "imas-codex-embed"], check=True)
        click.echo("Service stopped")


# ============================================================================
# Serve Tunnel Command Group
# ============================================================================


@serve.group("tunnel")
def serve_tunnel() -> None:
    """Manage SSH tunnel to remote embedding server.

    Uses autossh to maintain a persistent, auto-reconnecting SSH tunnel
    to the ITER embedding server. Required for remote embedding.

    \b
      imas-codex serve tunnel status    Check tunnel status
      imas-codex serve tunnel start     Start tunnel (foreground)
      imas-codex serve tunnel service   Manage systemd service
    """
    pass


@serve_tunnel.command("status")
@click.option("--host", default="iter", help="SSH host alias (default: iter)")
def tunnel_status(host: str) -> None:
    """Check tunnel and embedding server status.

    Examples:
        imas-codex serve tunnel status
        imas-codex serve tunnel status --host iter-gpu
    """
    import subprocess
    from pathlib import Path

    from imas_codex.settings import get_embed_server_port

    port = get_embed_server_port()

    # Check if tunnel port is open
    click.echo(f"SSH tunnel to {host} (port {port}):")

    # Check if port is in use
    result = subprocess.run(
        ["lsof", "-i", f":{port}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        # Parse process info
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            parts = lines[1].split()
            process = parts[0] if parts else "unknown"
            pid = parts[1] if len(parts) > 1 else "?"
            click.echo(f"  ✓ Port {port} in use by {process} (PID {pid})")
    else:
        click.echo(f"  ✗ Port {port} not in use (tunnel not active)")
        click.echo(f"    Start: imas-codex serve tunnel start --host {host}")
        click.echo(f"    Or:    ssh -f -N -L {port}:127.0.0.1:{port} {host}")
        return

    # Check embedding server health
    from imas_codex.embeddings.client import RemoteEmbeddingClient
    from imas_codex.settings import get_embed_remote_url

    url = get_embed_remote_url()
    client = RemoteEmbeddingClient(url)

    if client.is_available():
        info = client.get_detailed_info()
        if info:
            click.echo("  ✓ Embedding server reachable")
            click.echo(f"    Model: {info['model']['name']}")
            click.echo(f"    Uptime: {info['server']['uptime_seconds'] / 3600:.1f}h")
    else:
        click.echo("  ✗ Embedding server not responding")
        click.echo(
            "    Check server on remote: ssh iter systemctl --user status imas-codex-embed"
        )

    # Check if systemd service is installed
    service_file = (
        Path.home() / ".config" / "systemd" / "user" / "imas-codex-tunnel.service"
    )
    if service_file.exists():
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "imas-codex-tunnel"],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() == "active":
            click.echo("  ✓ systemd service active (auto-reconnect enabled)")
        else:
            click.echo(f"  Service installed but {result.stdout.strip()}")


@serve_tunnel.command("start")
@click.option("--host", default="iter", help="SSH host alias (default: iter)")
@click.option("--background", "-b", is_flag=True, help="Run in background")
def tunnel_start(host: str, background: bool) -> None:
    """Start SSH tunnel to embedding server.

    Uses autossh if available for auto-reconnection, otherwise plain ssh.

    Examples:
        # Start in foreground (Ctrl+C to stop)
        imas-codex serve tunnel start

        # Start in background
        imas-codex serve tunnel start -b

        # Use different host
        imas-codex serve tunnel start --host iter-gpu
    """
    import shutil
    import subprocess

    from imas_codex.settings import get_embed_server_port

    port = get_embed_server_port()

    # Prefer autossh if available
    autossh = shutil.which("autossh")
    if autossh:
        click.echo("Using autossh for auto-reconnection")
        # -M 0 disables monitoring port (ServerAliveInterval handles liveness)
        # -o ServerAliveInterval=30 sends keepalive every 30s
        # -o ServerAliveCountMax=3 disconnects after 3 missed keepalives
        cmd = [
            autossh,
            "-M",
            "0",
            "-N",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            f"{port}:127.0.0.1:{port}",
            host,
        ]
    else:
        click.echo("autossh not found, using plain ssh (no auto-reconnect)")
        click.echo("  Install: sudo apt install autossh  # or brew install autossh")
        cmd = [
            "ssh",
            "-N",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            f"{port}:127.0.0.1:{port}",
            host,
        ]

    if background:
        # Add -f flag for background
        if autossh:
            # autossh uses AUTOSSH_GATETIME=0 to background immediately
            env = os.environ.copy()
            env["AUTOSSH_GATETIME"] = "0"
            subprocess.Popen(cmd + ["-f"], env=env)
        else:
            subprocess.Popen(["ssh", "-f"] + cmd[1:])
        click.echo(f"Tunnel started in background (port {port})")
    else:
        click.echo(f"Starting tunnel to {host}:{port} (Ctrl+C to stop)")
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            click.echo("\nTunnel stopped")


@serve_tunnel.command("service")
@click.argument(
    "action",
    type=click.Choice(["install", "uninstall", "status", "start", "stop", "logs"]),
)
@click.option("--host", default="iter", help="SSH host alias (default: iter)")
def tunnel_service(action: str, host: str) -> None:
    """Manage SSH tunnel as systemd user service.

    Installs autossh-based tunnel with auto-reconnection on failure.

    Examples:
        imas-codex serve tunnel service install
        imas-codex serve tunnel service start
        imas-codex serve tunnel service status
        imas-codex serve tunnel service logs
    """
    import platform
    import shutil
    import subprocess
    from pathlib import Path

    if platform.system() != "Linux":
        raise click.ClickException("systemd services only supported on Linux")

    if not shutil.which("systemctl"):
        raise click.ClickException("systemctl not found")

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_file = service_dir / "imas-codex-tunnel.service"

    if action == "install":
        # Check for autossh
        autossh = shutil.which("autossh")
        if not autossh:
            raise click.ClickException(
                "autossh not found. Install with: sudo apt install autossh"
            )

        service_dir.mkdir(parents=True, exist_ok=True)

        from imas_codex.settings import get_embed_server_port

        port = get_embed_server_port()

        # autossh service with auto-reconnection
        # NOTE: ControlMaster=no prevents using existing SSH multiplex sockets
        # This ensures the tunnel has its own dedicated connection
        service_content = f"""[Unit]
Description=IMAS Codex SSH Tunnel (autossh to {host})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment="AUTOSSH_GATETIME=0"
Environment="AUTOSSH_POLL=60"
ExecStart={autossh} -M 0 -N -o "ControlMaster=no" -o "ControlPath=none" -o "ServerAliveInterval=30" -o "ServerAliveCountMax=3" -o "ExitOnForwardFailure=yes" -L {port}:127.0.0.1:{port} {host}
ExecStop=/bin/kill $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""
        service_file.write_text(service_content)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "imas-codex-tunnel"], check=True
        )
        click.echo("✓ Tunnel service installed and enabled")
        click.echo(f"  Host: {host}")
        click.echo(f"  Port: {port}")
        click.echo("  Start: imas-codex serve tunnel service start")

    elif action == "uninstall":
        if not service_file.exists():
            click.echo("Service not installed")
            return
        subprocess.run(
            ["systemctl", "--user", "stop", "imas-codex-tunnel"], capture_output=True
        )
        subprocess.run(
            ["systemctl", "--user", "disable", "imas-codex-tunnel"], capture_output=True
        )
        service_file.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        click.echo("Service uninstalled")

    elif action == "status":
        if not service_file.exists():
            click.echo("Service not installed")
            click.echo("  Install: imas-codex serve tunnel service install")
            return
        result = subprocess.run(
            ["systemctl", "--user", "status", "imas-codex-tunnel"],
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr)

    elif action == "start":
        if not service_file.exists():
            raise click.ClickException(
                "Service not installed. Run: imas-codex serve tunnel service install"
            )
        subprocess.run(
            ["systemctl", "--user", "start", "imas-codex-tunnel"], check=True
        )
        click.echo("Tunnel service started")

    elif action == "stop":
        subprocess.run(["systemctl", "--user", "stop", "imas-codex-tunnel"], check=True)
        click.echo("Tunnel service stopped")

    elif action == "logs":
        # Show recent logs and follow
        subprocess.run(
            ["journalctl", "--user", "-u", "imas-codex-tunnel", "-n", "50", "-f"]
        )
