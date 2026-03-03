"""Serve commands - MCP servers and embedding service."""

import logging
import os
import subprocess
import time
from typing import Literal, cast

import click

from imas_codex import _get_dd_version

logger = logging.getLogger(__name__)


@click.group()
def serve() -> None:
    """MCP servers and embedding service.

    \b
      imas-codex serve imas              Start the IMAS DD MCP server
      imas-codex serve agents            Start the Agents MCP server
      imas-codex serve embed deploy      Deploy embedding server (SLURM/systemd)
      imas-codex serve embed status      Show embed server health and load
      imas-codex serve embed stop        Stop embedding server
      imas-codex serve embed restart     Restart with new GPU allocation
      imas-codex serve embed logs        View embed server logs
      imas-codex serve llm deploy        Deploy LLM proxy (systemd)
      imas-codex serve llm status        Show LLM proxy health
      imas-codex serve tunnel start      Start SSH tunnels
      imas-codex serve tunnel status     Show active tunnels

    \b
    Neo4j is managed separately via 'imas-codex graph start/stop/status'.
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
        "Defaults to [tool.imas-codex.data-dictionary].version in pyproject.toml."
    ),
)
def serve_imas(
    transport: str,
    host: str,
    port: int,
    log_level: str,
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

    # For stdio transport, disable rich output automatically
    if transport == "stdio":
        os.environ["IMAS_CODEX_RICH"] = "0"
        logger.info(
            "Disabled rich output for stdio transport to prevent protocol interference"
        )

    from imas_codex.server import Server

    server_instance = Server(ids_set=ids_set)

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
# Serve LLM Proxy Command Group
# ============================================================================


@serve.group("llm")
def serve_llm() -> None:
    """Manage LiteLLM proxy server for centralized LLM routing.

    Routes all LLM calls through a single proxy with cost tracking via Langfuse.
    Provides OpenAI-compatible endpoint for agent teams and discovery workers.

    Deploy mode is determined by ``[llm].scheduler`` in pyproject.toml:
      slurm → SLURM compute node (shared allocation with embed/neo4j)
      (omit) → systemd service on login node or local start

    \b
      imas-codex serve llm deploy     Deploy per config
      imas-codex serve llm stop       Stop the proxy
      imas-codex serve llm restart    Restart (stop + deploy)
      imas-codex serve llm start      Start LiteLLM proxy (foreground)
      imas-codex serve llm status     Check proxy health
      imas-codex serve llm logs       View service logs
      imas-codex serve llm service    Manage systemd service
    """
    pass


@serve_llm.command("start")
@click.option(
    "--host",
    envvar="LITELLM_HOST",
    default="0.0.0.0",
    help="Host to bind (default: all interfaces for compute node access)",
)
@click.option(
    "--port",
    envvar="LITELLM_PORT",
    default=None,
    type=int,
    help="Port to bind (default: from pyproject.toml [llm].port)",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to LiteLLM config YAML (default: bundled config)",
)
def llm_start(
    host: str,
    port: int | None,
    log_level: str,
    config_path: str | None,
) -> None:
    """Start the LiteLLM proxy server.

    Routes all LLM calls through a centralized proxy with Langfuse observability.
    Requires OPENROUTER_API_KEY and LITELLM_MASTER_KEY environment variables.

    Examples:
        # Start with defaults
        imas-codex serve llm start

        # Custom port
        imas-codex serve llm start --port 19000

        # Custom config
        imas-codex serve llm start --config my_config.yaml
    """
    import shutil
    import subprocess
    from pathlib import Path

    from imas_codex.settings import get_llm_proxy_port

    if port is None:
        port = get_llm_proxy_port()

    uv_path = shutil.which("uv")
    if not uv_path:
        raise click.ClickException(
            "uv CLI not found. Install from: https://docs.astral.sh/uv/"
        )

    # Resolve config path
    if config_path is None:
        config_path = str(
            Path(__file__).parent.parent / "config" / "litellm_config.yaml"
        )
        if not Path(config_path).exists():
            raise click.ClickException(f"Bundled config not found: {config_path}")

    # Check required env vars
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise click.ClickException(
            "OPENROUTER_API_KEY not set. Add to .env or export in shell."
        )

    master_key = os.environ.get("LITELLM_MASTER_KEY")
    if not master_key:
        raise click.ClickException(
            "LITELLM_MASTER_KEY not set. Add to .env or export in shell."
        )

    click.echo(f"Starting LiteLLM proxy on {host}:{port}")
    click.echo(f"Config: {config_path}")
    click.echo(f"Master key: {master_key[:10]}...")

    # Check Langfuse config
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        click.echo("Langfuse: configured (cost tracking enabled)")
        os.environ.setdefault("LITELLM_CALLBACKS", "langfuse")
    else:
        click.echo(
            "Langfuse: not configured (set LANGFUSE_PUBLIC_KEY for cost tracking)"
        )

    cmd = [
        uv_path,
        "tool",
        "run",
        "--with",
        "litellm[proxy]>=1.81.0",
        "--with",
        "langfuse>=2.0.0",
        "--",
        "litellm",
        "--config",
        config_path,
        "--host",
        host,
        "--port",
        str(port),
        "--detailed_debug" if log_level == "DEBUG" else "--drop_params",
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        click.echo("\nProxy stopped")


@serve_llm.command("status")
@click.option(
    "--url",
    default=None,
    help="Proxy URL (default: from pyproject.toml [llm].port)",
)
def llm_status(url: str | None) -> None:
    """Check LiteLLM proxy health.

    Examples:
        imas-codex serve llm status
        imas-codex serve llm status --url http://remote:18400
    """
    import httpx

    from imas_codex.settings import get_llm_proxy_url

    if url is None:
        url = get_llm_proxy_url()

    click.echo(f"LiteLLM Proxy ({url}):")

    try:
        # Use root endpoint for quick alive check (no auth required)
        resp = httpx.get(f"{url}/", timeout=5.0)
        if resp.status_code == 200:
            click.echo("  ✓ Healthy")
            # Show model availability (requires auth)
            headers = {
                "Authorization": f"Bearer {os.environ.get('LITELLM_MASTER_KEY', '')}"
            }
            models_resp = httpx.get(
                f"{url}/v1/models",
                timeout=10.0,
                headers=headers,
            )
            if models_resp.status_code == 200:
                models = models_resp.json().get("data", [])
                click.echo(f"  Models: {len(models)} available")
                for m in models[:10]:
                    click.echo(f"    - {m.get('id', 'unknown')}")
        else:
            click.echo(f"  ✗ Unhealthy (HTTP {resp.status_code})")
    except httpx.ConnectError:
        click.echo("  ✗ Not running")
        click.echo("  Start with: imas-codex serve llm start")
    except httpx.RemoteProtocolError:
        click.echo("  ✗ Not responding (port in use by a non-HTTP process)")
        click.echo("  A stale process may be holding the port.")
        # Extract port from URL for diagnostic hint
        from urllib.parse import urlparse

        from imas_codex.settings import get_llm_proxy_port

        port = urlparse(url).port or get_llm_proxy_port()
        click.echo(f"  Check with: ss -tlnp sport = {port}")
        click.echo("  Then kill the process and restart:")
        click.echo("    imas-codex serve llm service start")
    except Exception as e:
        click.echo(f"  ✗ Error: {e}")


@serve_llm.command("service")
@click.argument(
    "action", type=click.Choice(["install", "uninstall", "status", "start", "stop"])
)
@click.option(
    "--port",
    default=None,
    type=int,
    help="Port for the proxy (default: from pyproject.toml)",
)
def llm_service(action: str, port: int | None) -> None:
    """Manage LiteLLM proxy as systemd user service.

    Examples:
        imas-codex serve llm service install
        imas-codex serve llm service start
        imas-codex serve llm service status
    """
    import platform
    import shutil
    import subprocess
    from pathlib import Path

    if platform.system() != "Linux":
        raise click.ClickException("systemd services only supported on Linux")

    if not shutil.which("systemctl"):
        raise click.ClickException("systemctl not found")

    from imas_codex.settings import get_llm_proxy_port

    if port is None:
        port = get_llm_proxy_port()

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_file = service_dir / "imas-codex-llm.service"

    if action == "install":
        service_dir.mkdir(parents=True, exist_ok=True)

        uv_path = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")

        project_dir = Path.home() / "imas-codex"
        if not project_dir.exists():
            project_dir = Path.cwd()
            if not (project_dir / "pyproject.toml").exists():
                raise click.ClickException(
                    f"Project not found at {Path.home() / 'imas-codex'} or current directory"
                )

        config_path = project_dir / "imas_codex" / "config" / "litellm_config.yaml"

        env_file = project_dir / ".env"

        log_dir = Path.home() / ".local" / "share" / "imas-codex" / "services"
        log_file = log_dir / "llm.log"

        service_content = f"""[Unit]
Description=IMAS Codex LiteLLM Proxy
After=network.target

[Service]
Type=simple
WorkingDirectory={project_dir}
EnvironmentFile=-{env_file}
Environment="PATH={Path.home()}/.local/bin:/usr/local/bin:/usr/bin"
Environment="LITELLM_CALLBACKS=langfuse"
ExecStartPre=/bin/mkdir -p {log_dir}
ExecStart={uv_path} tool run --with 'litellm[proxy]>=1.81.0' --with 'langfuse>=2.0.0' -- litellm --config {config_path} --host 0.0.0.0 --port {port} --drop_params
ExecStop=/bin/kill -15 $MAINPID
StandardOutput=append:{log_file}
StandardError=append:{log_file}
TimeoutStopSec=30
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""
        service_file.write_text(service_content)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "imas-codex-llm"], check=True)
        click.echo("✓ LLM proxy service installed and enabled")
        click.echo(f"  Port: {port}")
        click.echo("  Start: imas-codex serve llm service start")

    elif action == "uninstall":
        if not service_file.exists():
            click.echo("Service not installed")
            return
        subprocess.run(
            ["systemctl", "--user", "stop", "imas-codex-llm"], capture_output=True
        )
        subprocess.run(
            ["systemctl", "--user", "disable", "imas-codex-llm"], capture_output=True
        )
        service_file.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        click.echo("Service uninstalled")

    elif action == "status":
        if not service_file.exists():
            click.echo("Service not installed")
            return
        result = subprocess.run(
            ["systemctl", "--user", "status", "imas-codex-llm"],
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout)

    elif action == "start":
        if not service_file.exists():
            raise click.ClickException(
                "Service not installed. Run: imas-codex serve llm service install"
            )
        result = subprocess.run(
            ["systemctl", "--user", "start", "imas-codex-llm"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Read recent log lines for diagnostics
            log_file = (
                Path.home() / ".local" / "share" / "imas-codex" / "services" / "llm.log"
            )
            log_tail = ""
            if log_file.exists():
                log_tail = log_file.read_text()[-2000:]
            raise click.ClickException(
                "Failed to start LLM proxy service.\n"
                + log_tail
                + "\nCheck full logs: imas-codex serve llm logs"
            )
        click.echo("LLM proxy service started")

    elif action == "stop":
        subprocess.run(["systemctl", "--user", "stop", "imas-codex-llm"], check=True)
        click.echo("LLM proxy service stopped")


# ── LLM deploy/stop/restart/logs commands ────────────────────────────────


@serve_llm.command("deploy")
def llm_deploy() -> None:
    """Deploy LLM proxy on the login node via systemd.

    Installs the systemd user service if needed, then starts it.
    CPU-only service (~50 MB RAM), no GPU required.
    Runs on the login node which has outbound HTTPS for API providers.

    Idempotent: no-op if already running.

    \\b
    Examples:
        imas-codex serve llm deploy
    """
    _deploy_login_llm()
    port = _llm_port()
    _wait_for_health(
        "LLM proxy",
        f"curl -sf http://localhost:{port}/",
        timeout_s=60,
    )
    click.echo(f"  URL: http://localhost:{port}")


@serve_llm.command("stop")
def llm_stop() -> None:
    """Stop the LLM proxy server on the login node.

    \\b
    Examples:
        imas-codex serve llm stop
    """
    try:
        result = _run_remote(
            "systemctl --user is-active imas-codex-llm 2>/dev/null || true",
            timeout=10,
        )
        if "active" in result and "inactive" not in result:
            _run_remote(
                "systemctl --user stop imas-codex-llm",
                timeout=15,
                check=True,
            )
            click.echo("LLM proxy stopped")
        else:
            click.echo("LLM proxy not running")
    except subprocess.CalledProcessError:
        click.echo("Failed to stop LLM proxy")


@serve_llm.command("restart")
def llm_restart() -> None:
    """Restart the LLM proxy server on the login node.

    \\b
    Examples:
        imas-codex serve llm restart
    """
    try:
        _run_remote(
            "systemctl --user restart imas-codex-llm",
            timeout=30,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            "Failed to restart. Is the service installed?\n"
            "  Install: imas-codex serve llm service install"
        ) from exc

    port = _llm_port()
    _wait_for_health(
        "LLM proxy",
        f"curl -sf http://localhost:{port}/",
        timeout_s=60,
    )
    click.echo(f"  URL: http://localhost:{port}")


@serve_llm.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (tail -f)")
@click.option(
    "--lines", "-n", default=50, type=int, help="Number of lines (default: 50)"
)
def llm_logs(follow: bool, lines: int) -> None:
    """View LLM proxy service logs.

    \\b
    Examples:
        imas-codex serve llm logs            # Last 50 lines
        imas-codex serve llm logs -f         # Follow live
        imas-codex serve llm logs -n 100     # Last 100 lines
    """
    log_file = f"{_SERVICES_DIR}/llm.log"
    _tail_log(log_file, follow, lines)


def _deploy_login_llm() -> None:
    """Deploy LLM proxy to login node via systemd.

    Installs the systemd user service if not present, then starts it.
    The LLM proxy needs outbound HTTPS to reach API providers
    (OpenRouter, Anthropic, Google), so it must run on the login
    node which has internet access, not on a compute node.

    Idempotent: no-op if already running.
    """
    # Check if already running
    try:
        result = _run_remote(
            "systemctl --user is-active imas-codex-llm 2>/dev/null || true",
            timeout=10,
        )
        if "active" in result and "inactive" not in result:
            click.echo("  LLM proxy already running on login node")
            return
    except subprocess.CalledProcessError:
        pass

    # Install service if not present, then start
    try:
        _run_remote(
            "systemctl --user cat imas-codex-llm >/dev/null 2>&1",
            timeout=10,
            check=True,
        )
    except subprocess.CalledProcessError:
        # Service not installed — install it remotely
        click.echo("  Installing systemd service...")
        _install_llm_service_remote()

    click.echo("  Starting LLM proxy on login node...")
    try:
        _run_remote(
            "systemctl --user start imas-codex-llm",
            timeout=15,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        log_tail = _run_remote(
            f"tail -20 {_SERVICES_DIR}/llm.log 2>/dev/null || true",
            timeout=10,
        )
        raise click.ClickException(f"Failed to start LLM proxy.\n{log_tail}") from exc


def _install_llm_service_remote() -> None:
    """Install the LLM systemd user service on the remote login node via SSH."""
    import base64

    port = _llm_port()
    # systemd doesn't expand $HOME in any directive — use %h (home dir specifier)
    _project_h = _PROJECT.replace("$HOME", "%h")
    _services_h = _SERVICES_DIR.replace("$HOME", "%h")
    service_content = f"""[Unit]
Description=IMAS Codex LiteLLM Proxy
After=network.target

[Service]
Type=simple
WorkingDirectory={_project_h}
EnvironmentFile=-{_project_h}/.env
Environment="PATH=%h/.local/bin:/usr/local/bin:/usr/bin"
Environment="LITELLM_CALLBACKS=langfuse"
ExecStartPre=/bin/mkdir -p {_services_h}
ExecStart=%h/.local/bin/uv tool run --with 'litellm[proxy]>=1.81.0' --with 'langfuse>=2.0.0' -- litellm --config {_project_h}/imas_codex/config/litellm_config.yaml --host 0.0.0.0 --port {port} --drop_params
ExecStop=/bin/kill -15 $MAINPID
StandardOutput=append:{_services_h}/llm.log
StandardError=append:{_services_h}/llm.log
TimeoutStopSec=30
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""
    content_b64 = base64.b64encode(service_content.encode()).decode()
    _run_remote(
        'mkdir -p "$HOME/.config/systemd/user" && '
        f'echo "{content_b64}" | base64 -d > '
        '"$HOME/.config/systemd/user/imas-codex-llm.service" && '
        "systemctl --user daemon-reload && "
        "systemctl --user enable imas-codex-llm",
        timeout=15,
        check=True,
    )
    click.echo("  Service installed and enabled")


# ============================================================================
# Serve Embed Command Group
# ============================================================================


@serve.group("embed")
def serve_embed() -> None:
    """Manage GPU embedding server.

    Deploy mode is determined by ``[embedding].scheduler`` in pyproject.toml:
      slurm → SLURM batch job on GPU compute node
      (omit) → systemd service on login node

    The embedding location (``[embedding].location``) determines where
    commands are sent (e.g. ``iter``).

    Override with env vars: IMAS_CODEX_EMBED_SCHEDULER, IMAS_CODEX_EMBEDDING_LOCATION

    \b
      imas-codex serve embed deploy     Deploy per config
      imas-codex serve embed stop       Stop the server
      imas-codex serve embed restart    Restart (stop + deploy)
      imas-codex serve embed status     Check server and SLURM job status
      imas-codex serve embed logs       View SLURM job logs
      imas-codex serve embed start      Start embedding server locally
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
@click.option(
    "--deploy-label",
    default=None,
    type=str,
    help="Deployment label (e.g., 'titan', 'login'). Exposed in /health.",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    help="Number of uvicorn worker processes (default: 1). Use with --gpus for multi-GPU.",
)
@click.option(
    "--gpus",
    default=None,
    type=str,
    help="Comma-separated GPU IDs for multi-worker mode (e.g., '0,1,2,3').",
)
def embed_start(
    host: str,
    port: int | None,
    log_level: str,
    gpu: str | None,
    idle_timeout: int,
    deploy_label: str | None,
    workers: int,
    gpus: str | None,
) -> None:
    """Start the embedding server.

    The server provides GPU-accelerated embedding via HTTP API.
    Access via SSH tunnel for security.

    Examples:
        # Start with defaults
        imas-codex serve embed start

        # Use specific GPU
        imas-codex serve embed start --gpu 1

        # Multi-worker on 4 GPUs (Titan)
        imas-codex serve embed start --gpus 0,1,2,3 --workers 4

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

    # Set GPU if specified (single-GPU mode, mutually exclusive with --gpus)
    if gpu is not None and gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        logger.info(f"Using CUDA device: {gpu}")

    # Multi-GPU mode: set GPU pool env var for worker processes to claim
    if gpus is not None:
        os.environ["IMAS_CODEX_GPU_POOL"] = gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        if workers <= 1:
            workers = len(gpus.split(","))
        logger.info(f"Multi-GPU mode: pool={gpus}, workers={workers}")

    # Get port from settings if not specified
    if port is None:
        from imas_codex.settings import get_embed_server_port

        port = get_embed_server_port()

    # Set idle timeout in server module
    if idle_timeout > 0:
        import imas_codex.embeddings.server as embed_server

        embed_server._idle_timeout = idle_timeout
        logger.info(f"Idle timeout: {idle_timeout}s")

    # Set deployment label (exposed in /health)
    if deploy_label:
        import imas_codex.embeddings.server as embed_server

        embed_server._location = deploy_label
        logger.info(f"Deploy label: {deploy_label}")

    logger.info(f"Starting embedding server on {host}:{port}")

    try:
        import uvicorn

        if workers > 1:
            # Multi-worker requires import string (uvicorn forks new processes)
            uvicorn.run(
                "imas_codex.embeddings.server:app",
                host=host,
                port=port,
                workers=workers,
                log_level=log_level.lower(),
            )
        else:
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

    Shows server health, model info, uptime, SLURM job state, and
    optionally local GPU capability.

    \b
    Examples:
        imas-codex serve embed status          # Health + SLURM
        imas-codex serve embed status --local  # Local GPU capability
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

    # Embed service job status
    try:
        job = _get_embed_job()
        if job:
            for line in _format_service_status(job, "embed"):
                click.echo(line)
        else:
            click.echo(click.style("Embed job: none", dim=True))
    except Exception:
        click.echo("Embed job: unavailable")

    # Check remote server health
    if url is None:
        from imas_codex.settings import get_embed_remote_url

        url = get_embed_remote_url()

    if not url:
        click.echo("No remote embedding server configured.")
        click.echo(
            "Set embedding_service.backend=remote in facility YAML or IMAS_CODEX_EMBED_REMOTE_URL env var."
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
            location = info["server"].get("location")
            if location:
                click.echo(f"  Location: {location}")
            click.echo(f"  Uptime: {uptime_h:.1f}h")
            if timeout_s > 0:
                click.echo(f"  Idle: {idle_s:.0f}s / {timeout_s}s timeout")
    else:
        click.echo("  ✗ Not available")
        click.echo("  Deploy with: imas-codex serve embed deploy")
        click.echo("  Check tunnel: imas-codex tunnel status")


@serve_embed.command("service")
@click.argument(
    "action", type=click.Choice(["install", "uninstall", "status", "start", "stop"])
)
@click.option("--gpu", default="1", help="CUDA device to use (default: 1)")
@click.option(
    "--deploy-label", default="login", help="Deployment label (default: login)"
)
def embed_service(action: str, gpu: str, deploy_label: str) -> None:
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

        env_file = project_dir / ".env"

        service_content = f"""[Unit]
Description=IMAS Codex Embedding Service (GPU)
After=network.target

[Service]
Type=simple
WorkingDirectory={project_dir}
EnvironmentFile=-{env_file}
Environment="PATH={Path.home()}/.local/bin:/usr/local/bin:/usr/bin"
Environment="CUDA_VISIBLE_DEVICES={gpu}"
ExecStart={uv_path} run --extra gpu --project {project_dir} imas-codex serve embed start --host 0.0.0.0 --port {port} --deploy-label {deploy_label}
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
# Compute Node Allocation & Service Management
# ============================================================================
#
# Architecture: Each service (Neo4j, embed) runs as its own SLURM job.
# The job script runs the service process directly — no ``sleep infinity``
# allocation, no ``nohup`` wrappers.  SLURM manages the lifecycle:
# ``scancel`` stops the service, cgroup enforcement is automatic, and
# ``squeue`` shows accurate resource accounting.

from imas_codex.cli.services import (  # noqa: E402
    _DEFAULT_GPUS,
    _EMBED_JOB,
    _PROJECT,
    _SERVICES_DIR,
    _cancel_service_job,
    _deploy_login_embed,
    _embed_port,
    _format_service_status,
    _get_embed_job,
    _is_compute_target,
    _llm_port,
    _run_remote,
    _tail_log,
    _wait_for_health,
    deploy_embed,
)

# ── Embed deploy/stop/restart/logs commands ──────────────────────────────


@serve_embed.command("deploy")
@click.option(
    "--gpus",
    "-g",
    default=_DEFAULT_GPUS,
    type=int,
    help=f"Number of GPUs (default: {_DEFAULT_GPUS}, ignored for login)",
)
@click.option(
    "--workers",
    "-w",
    default=None,
    type=int,
    help="Worker processes (default: same as gpus)",
)
def embed_deploy(gpus: int, workers: int | None) -> None:
    """Deploy embedding server.

    When ``[embedding].scheduler = "slurm"``, submits a dedicated SLURM
    job for the embed server with the requested GPU/worker count.

    When scheduler is not set, deploys via systemd on the login node.

    Idempotent: no-op if already running.  Use ``--gpus`` to redeploy
    with different resources (stops existing job first).

    \b
    Examples:
        imas-codex serve embed deploy          # Deploy per config
        imas-codex serve embed deploy -g 2     # 2 GPUs
    """
    if _is_compute_target():
        deploy_embed(gpus, workers)
    else:
        _deploy_login_embed()
        click.echo(f"  URL: http://localhost:{_embed_port()}")


@serve_embed.command("stop")
def embed_stop() -> None:
    """Stop the embedding server.

    Cancels the embed SLURM job (which stops the server process),
    or stops the systemd service on the login node.

    \b
    Examples:
        imas-codex serve embed stop
    """
    stopped = False

    # Cancel embed SLURM job
    if _cancel_service_job(_EMBED_JOB):
        click.echo("Stopped embed server (SLURM job cancelled)")
        stopped = True

    # Stop systemd service
    try:
        _run_remote(
            "systemctl --user stop imas-codex-embed 2>/dev/null",
            timeout=15,
            check=True,
        )
        click.echo("Stopped login embed service")
        stopped = True
    except subprocess.CalledProcessError:
        pass

    if not stopped:
        click.echo("No active embed server found")


@serve_embed.command("restart")
@click.option(
    "--gpus",
    "-g",
    default=_DEFAULT_GPUS,
    type=int,
    help=f"Number of GPUs (default: {_DEFAULT_GPUS}, ignored for login)",
)
@click.option(
    "--workers",
    "-w",
    default=None,
    type=int,
    help="Worker processes (default: same as gpus)",
)
def embed_restart(gpus: int, workers: int | None) -> None:
    """Restart the embedding server (stop + deploy).

    Cancels the existing embed SLURM job and submits a new one.
    Use ``--gpus`` to change GPU allocation on restart.

    \b
    Examples:
        imas-codex serve embed restart         # Restart
        imas-codex serve embed restart -g 2    # Restart with 2 GPUs
    """
    # Stop
    _cancel_service_job(_EMBED_JOB)
    try:
        _run_remote("systemctl --user stop imas-codex-embed 2>/dev/null", timeout=15)
    except subprocess.CalledProcessError:
        pass
    time.sleep(2)

    # Deploy
    if _is_compute_target():
        deploy_embed(gpus, workers)
    else:
        _deploy_login_embed()


@serve_embed.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (tail -f)")
@click.option(
    "--lines", "-n", default=50, type=int, help="Number of lines (default: 50)"
)
def embed_logs(follow: bool, lines: int) -> None:
    """View embedding server logs.

    \b
    Examples:
        imas-codex serve embed logs            # Last 50 lines
        imas-codex serve embed logs -f         # Follow live
        imas-codex serve embed logs -n 100     # Last 100 lines
    """
    log_file = f"{_SERVICES_DIR}/codex-embed.log"
    _tail_log(log_file, follow, lines)


# ============================================================================
# Tunnel Subgroup (delegates to tunnel.py)
# ============================================================================


@serve.group("tunnel")
def serve_tunnel_group() -> None:
    """Manage SSH tunnels to remote services.

    Convenience alias for ``imas-codex tunnel`` commands, grouped
    under ``serve`` for workflow consistency.

    \b
      imas-codex serve tunnel start       Start tunnels (all services)
      imas-codex serve tunnel stop        Stop tunnels
      imas-codex serve tunnel status      Show active tunnels
    """
    pass


@serve_tunnel_group.command("start")
@click.argument("host", required=False)
@click.option("--neo4j", "neo4j_only", is_flag=True, help="Tunnel Neo4j ports only")
@click.option("--embed", "embed_only", is_flag=True, help="Tunnel embedding port only")
@click.option("--llm", "llm_only", is_flag=True, help="Tunnel LLM proxy port only")
def serve_tunnel_start(
    host: str | None,
    neo4j_only: bool,
    embed_only: bool,
    llm_only: bool,
) -> None:
    """Start SSH tunnels to remote services.

    Uses autossh when available for automatic reconnection.
    HOST defaults to the active graph profile's configured host.

    \b
    Examples:
      imas-codex serve tunnel start iter           # All services
      imas-codex serve tunnel start iter --neo4j   # Just graph
    """
    import shutil

    from imas_codex.cli.tunnel import (
        _get_tunnel_ports,
        _resolve_host,
        _start_tunnels,
    )

    target = _resolve_host(host)
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


@serve_tunnel_group.command("stop")
@click.argument("host", required=False)
@click.option("--all", "stop_all", is_flag=True, help="Stop tunnels to all hosts")
def serve_tunnel_stop(host: str | None, stop_all: bool) -> None:
    """Stop SSH tunnels.

    Without HOST, stops tunnels to all configured remote hosts.

    \b
    Examples:
      imas-codex serve tunnel stop           # Stop all tunnels
      imas-codex serve tunnel stop iter      # Stop only iter tunnels
    """
    from imas_codex.cli.tunnel import tunnel_stop

    # Invoke the real tunnel stop command logic
    ctx = click.get_current_context()
    ctx.invoke(tunnel_stop, host=host, stop_all=stop_all)


@serve_tunnel_group.command("status")
def serve_tunnel_status() -> None:
    """Show active SSH tunnels across all services."""
    from imas_codex.cli.tunnel import tunnel_status

    ctx = click.get_current_context()
    ctx.invoke(tunnel_status)
