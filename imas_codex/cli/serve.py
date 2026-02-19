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

    master_key = os.environ.get("LITELLM_MASTER_KEY", "sk-litellm-imas-codex")
    os.environ.setdefault("LITELLM_MASTER_KEY", master_key)

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
                "Authorization": f"Bearer {os.environ.get('LITELLM_MASTER_KEY', 'sk-litellm-imas-codex')}"
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
Restart=on-failure
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
Restart=on-failure
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

    # Allocation / embed service status
    try:
        alloc = _get_allocation()
        if alloc:
            click.echo(
                f"Allocation: {alloc['job_id']} {alloc['state']} on {alloc['node']} ({alloc['gres']}, {alloc['time']})"
            )
            if alloc["state"] == "RUNNING" and _service_running(alloc["node"], "embed"):
                click.echo("  embed: running")
            else:
                click.echo("  embed: not running")
        else:
            click.echo("Allocation: none")
    except Exception:
        click.echo("Allocation: unavailable")

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
# Architecture: A persistent SLURM allocation (sleep infinity) provides a
# "compute lease" on a GPU node.  Individual services (Neo4j, embed server)
# are started/stopped independently on that node via SSH + PID files.
#
# Allocation:  sbatch → sleep infinity         (one job, always running)
# Services:    ssh node → nohup service &      (per-service PID + log files)
#
# Benefits over multi-script approach:
#   - Single SLURM job to manage
#   - Independent service lifecycles (stop neo4j without touching embed)
#   - No combinatorial explosion of batch scripts
#   - Easy to add new services later

_ALLOC_JOB = "imas-codex-services"
_SERVICES_DIR = "$HOME/.local/share/imas-codex/services"
_DEFAULT_GPUS = 4
_PROJECT = "$HOME/Code/imas-codex"


# ── Port helpers ─────────────────────────────────────────────────────────


def _embed_port() -> int:
    """Return the configured embed server port."""
    from imas_codex.settings import get_embed_server_port

    return get_embed_server_port()


def _graph_port() -> int:
    """Return the configured Neo4j bolt port."""
    from imas_codex.graph.profiles import resolve_neo4j

    return resolve_neo4j(auto_tunnel=False).bolt_port


def _graph_http_port() -> int:
    """Return the configured Neo4j HTTP port."""
    from imas_codex.graph.profiles import resolve_neo4j

    return resolve_neo4j(auto_tunnel=False).http_port


def _llm_port() -> int:
    """Return the configured LLM proxy port."""
    from imas_codex.settings import get_llm_proxy_port

    return get_llm_proxy_port()


# ── Scheduler detection ─────────────────────────────────────────────────


def _is_compute_target() -> bool:
    """True when embed scheduler is 'slurm' (deploy via SLURM)."""
    from imas_codex.settings import get_embed_scheduler

    return get_embed_scheduler() == "slurm"


def _is_graph_compute_target() -> bool:
    """True when graph scheduler is 'slurm' (deploy via SLURM)."""
    from imas_codex.settings import get_graph_scheduler

    return get_graph_scheduler() == "slurm"


# ── Facility compute config ─────────────────────────────────────────────


def _compute_config() -> dict:
    """Load compute config from the facility's private YAML.

    The facility is determined by ``[embedding].location`` (e.g. ``"iter"``).
    """
    from imas_codex.discovery.base.facility import get_facility_infrastructure
    from imas_codex.settings import get_embedding_location

    facility_id = get_embedding_location()
    if facility_id == "local":
        raise click.ClickException(
            "Embedding location is 'local' — no compute config available."
        )
    infra = get_facility_infrastructure(facility_id)
    compute = infra.get("compute", {})
    if not compute:
        raise click.ClickException(
            f"No compute config in {facility_id}_private.yaml.\n"
            "Add via: update_facility_infrastructure()"
        )
    return compute


def _gpu_entry() -> dict:
    """Get the GPU resource entry marked for embed_server."""
    compute = _compute_config()
    for gpu in compute.get("gpus", []):
        if gpu.get("current_use") == "embed_server":
            return gpu
    raise click.ClickException(
        "No GPU with current_use=embed_server in facility compute config."
    )


def _gpu_partition() -> dict:
    """Get the first scheduler partition with GPUs."""
    compute = _compute_config()
    for p in compute.get("scheduler", {}).get("partitions", []):
        if p.get("gpus_per_node") or p.get("gpu_type"):
            return p
    raise click.ClickException("No GPU partition found in facility compute config.")


# ── SSH helpers ──────────────────────────────────────────────────────────


def _facility_ssh() -> str | None:
    """SSH host for reaching the facility, or None if local."""
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.remote.executor import is_local_host
    from imas_codex.settings import get_embedding_location

    facility_id = get_embedding_location()
    if facility_id == "local":
        return None
    config = get_facility(facility_id)
    ssh_host = config.get("ssh_host", facility_id)
    if is_local_host(ssh_host):
        return None
    return ssh_host


def _run_remote(cmd: str, timeout: int = 30, check: bool = False) -> str:
    """Run a command on the facility login node (locally if already there)."""
    from imas_codex.remote.executor import run_command

    return run_command(cmd, ssh_host=_facility_ssh(), timeout=timeout, check=check)


def _run_on_node(node: str, cmd: str, timeout: int = 30) -> str:
    """Run a command on a compute node (via SSH from login node).

    Uses base64 encoding to avoid nested shell quoting issues.
    """
    import base64

    cmd_b64 = base64.b64encode(cmd.encode()).decode()
    return _run_remote(
        f'ssh -o StrictHostKeyChecking=no {node} "echo {cmd_b64} | base64 -d | bash"',
        timeout=timeout,
    )


# ── SLURM allocation management ─────────────────────────────────────────


def _get_allocation() -> dict | None:
    """Get the active imas-codex-services SLURM allocation.

    Returns dict with job_id, state, node, gres, time — or None.
    """
    try:
        out = _run_remote(
            f'squeue -n {_ALLOC_JOB} -u "$USER" --format="%A|%T|%M|%N|%b" --noheader'
        )
    except subprocess.CalledProcessError:
        return None
    for line in out.strip().split("\n"):
        line = line.strip()
        if not line or line == "(no output)":
            continue
        parts = line.split("|")
        if len(parts) >= 5:
            return {
                "job_id": parts[0].strip(),
                "state": parts[1].strip(),
                "time": parts[2].strip(),
                "node": parts[3].strip(),
                "gres": parts[4].strip(),
            }
    return None


def _ensure_allocation(gpus: int = _DEFAULT_GPUS) -> dict:
    """Ensure a SLURM allocation exists, creating one if needed.

    The allocation is a ``sleep infinity`` batch job that reserves
    compute resources.  Services are managed separately via SSH.

    Returns the allocation dict (job_id, node, etc.).
    """
    alloc = _get_allocation()
    if alloc and alloc["state"] == "RUNNING":
        return alloc

    # Cancel any PENDING allocation (stuck in queue)
    if alloc:
        _run_remote(f"scancel {alloc['job_id']}", check=False)
        time.sleep(1)

    _submit_allocation(gpus)

    # Wait for RUNNING state
    click.echo("Waiting for allocation to start...")
    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(3)
        alloc = _get_allocation()
        if alloc and alloc["state"] == "RUNNING":
            click.echo(
                f"  Allocation ready: job {alloc['job_id']} on {alloc['node']} "
                f"({alloc['gres']})"
            )
            return alloc
        remaining = int(deadline - time.time())
        if remaining > 0 and remaining % 15 < 5:
            state = alloc["state"] if alloc else "UNKNOWN"
            click.echo(f"  State: {state} ({remaining}s remaining)")

    raise click.ClickException(
        "Allocation did not start within 120s. Check: squeue -u $USER"
    )


def _submit_allocation(gpus: int) -> None:
    """Submit a new allocation job (sleep infinity with cleanup trap)."""
    import base64

    gpu_entry = _gpu_entry()
    partition = _gpu_partition()
    host = gpu_entry["location"]
    partition_name = partition["name"]
    cpus = gpus * 4

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --partition={partition_name}\n"
        f"#SBATCH --gres=gpu:{gpus}\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        "#SBATCH --mem=64G\n"
        "#SBATCH --time=UNLIMITED\n"
        f"#SBATCH --job-name={_ALLOC_JOB}\n"
        f"#SBATCH --nodelist={host}\n"
        f"#SBATCH --output={_SERVICES_DIR}/allocation.log\n"
        "\n"
        f"SERVICES_DIR={_SERVICES_DIR}\n"
        'mkdir -p "$SERVICES_DIR"\n'
        "\n"
        "cleanup() {\n"
        '    echo "Releasing allocation at $(date), stopping services..."\n'
        '    for pidfile in "$SERVICES_DIR"/*.pid; do\n'
        '        [ -f "$pidfile" ] || continue\n'
        '        pid=$(cat "$pidfile")\n'
        '        name=$(basename "$pidfile" .pid)\n'
        '        if kill -0 "$pid" 2>/dev/null; then\n'
        '            echo "Stopping $name (PID $pid)..."\n'
        '            pkill -TERM -P "$pid" 2>/dev/null || true\n'
        '            kill -TERM "$pid" 2>/dev/null || true\n'
        "            sleep 2\n"
        '            if kill -0 "$pid" 2>/dev/null; then\n'
        '                pkill -KILL -P "$pid" 2>/dev/null || true\n'
        '                kill -KILL "$pid" 2>/dev/null || true\n'
        "            fi\n"
        "        fi\n"
        '        rm -f "$pidfile"\n'
        "    done\n"
        '    echo "All services stopped"\n'
        "}\n"
        "trap cleanup EXIT TERM INT\n"
        "\n"
        'echo "Allocation ready on $(hostname) at $(date)"\n'
        'echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"\n'
        "\n"
        "sleep infinity\n"
    )

    # Cancel conflicting legacy jobs (standalone embed, etc.)
    _cancel_legacy_jobs()

    # Stop conflicting login-node services
    _stop_login_services()

    script_b64 = base64.b64encode(script.encode()).decode()
    submit_cmd = (
        f"mkdir -p {_SERVICES_DIR} && "
        f'echo "{script_b64}" | base64 -d > /tmp/codex-alloc.sh && '
        "sbatch /tmp/codex-alloc.sh && "
        "rm -f /tmp/codex-alloc.sh"
    )
    output = _run_remote(submit_cmd, timeout=30, check=True)
    click.echo(output.strip().split("\n")[0])


def _cancel_allocation() -> list[str]:
    """Cancel the allocation job (stopping all services first)."""
    alloc = _get_allocation()
    if not alloc:
        return []

    # Stop services explicitly before canceling the allocation,
    # since nohup'd processes survive SLURM job termination.
    if alloc["state"] == "RUNNING":
        node = alloc["node"]
        for svc in ("neo4j", "embed", "llm"):
            if _service_running(node, svc):
                _stop_service(node, svc)

    try:
        _run_remote(f"scancel {alloc['job_id']}", check=True)
        return [alloc["job_id"]]
    except subprocess.CalledProcessError:
        return []


def _cancel_legacy_jobs() -> None:
    """Cancel any legacy standalone SLURM jobs (codex-embed, etc.)."""
    for name in ("codex-embed", "codex-services"):
        try:
            out = _run_remote(f'squeue -n {name} -u "$USER" --format="%A" --noheader')
            for line in out.strip().split("\n"):
                jid = line.strip()
                if jid and jid != "(no output)" and jid.isdigit():
                    _run_remote(f"scancel {jid}", check=False)
                    click.echo(f"Cancelled legacy job {jid} ({name})")
        except subprocess.CalledProcessError:
            pass


def _stop_login_services() -> None:
    """Stop any conflicting login-node services (systemd, apptainer)."""
    # Embed systemd service
    try:
        result = _run_remote(
            "systemctl --user is-active imas-codex-embed 2>/dev/null || true",
            timeout=10,
        )
        if "active" in result:
            _run_remote(
                "systemctl --user stop imas-codex-embed 2>/dev/null || true",
                timeout=15,
            )
            click.echo("Stopped login embed service")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # Neo4j systemd service
    try:
        result = _run_remote(
            "systemctl --user list-units 'imas-codex-neo4j-*' --no-pager "
            "--plain --no-legend 2>/dev/null | awk '{print $1}' | head -1",
            timeout=10,
        )
        service = result.strip()
        if service and service != "(no output)":
            _run_remote(
                f"systemctl --user stop {service} 2>/dev/null || true", timeout=15
            )
            click.echo(f"Stopped login service ({service})")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # Kill any direct Neo4j apptainer processes (manual starts)
    try:
        result = _run_remote(
            "pgrep -u $USER -f 'neo4j_.*community.*\\.sif' 2>/dev/null | head -1",
            timeout=10,
        )
        pid = result.strip()
        if pid and pid != "(no output)" and pid.isdigit():
            _run_remote(f"kill {pid} 2>/dev/null || true", timeout=10)
            click.echo(f"Killed login Neo4j process (PID {pid})")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # LLM proxy systemd service
    try:
        result = _run_remote(
            "systemctl --user is-active imas-codex-llm 2>/dev/null || true",
            timeout=10,
        )
        if "active" in result:
            _run_remote(
                "systemctl --user stop imas-codex-llm 2>/dev/null || true",
                timeout=15,
            )
            click.echo("Stopped login LLM proxy service")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass


# ── Service management on allocated node ─────────────────────────────────


def _start_service(node: str, name: str, command: str) -> None:
    """Start a named service on the compute node.

    Writes the command to a launcher script on the shared filesystem,
    then runs ``nohup`` on the compute node.  PID tracked in
    ``$SERVICES_DIR/<name>.pid``, logs in ``$SERVICES_DIR/<name>.log``.

    Idempotent: no-op if already running.
    """
    import base64 as b64

    if _service_running(node, name):
        click.echo(f"  {name} already running on {node}")
        return

    # Write service command as a launcher script on shared GPFS.
    # Source .env first for API keys, then exec the service command
    # so the script's PID becomes the actual process PID (not a
    # dead bash wrapper).
    svc_script = f"{_SERVICES_DIR}/{name}.sh"
    script_content = (
        f"#!/bin/bash\n"
        f"cd {_PROJECT}\n"
        f"source {_PROJECT}/.env 2>/dev/null || true\n"
        f"exec {command}\n"
    )
    script_b64 = b64.b64encode(script_content.encode()).decode()
    _run_remote(
        f"mkdir -p {_SERVICES_DIR} && "
        f'echo "{script_b64}" | base64 -d > {svc_script} && '
        f"chmod +x {svc_script}",
        timeout=10,
    )

    # Launch on compute node via nohup (script is on shared GPFS)
    launch = (
        f"nohup {svc_script} > {_SERVICES_DIR}/{name}.log 2>&1 &\n"
        f"echo $! > {_SERVICES_DIR}/{name}.pid\n"
        f'echo "Started {name} (PID $!)"\n'
    )
    result = _run_on_node(node, launch, timeout=30)
    click.echo(f"  {result.strip()}")


def _stop_service(node: str, name: str) -> bool:
    """Stop a named service on the compute node. Returns True if stopped.

    Sends SIGTERM first, waits briefly, then SIGKILL if needed.
    Also kills child processes (process tree).
    """
    stop = (
        f"pid=$(cat {_SERVICES_DIR}/{name}.pid 2>/dev/null) || exit 0\n"
        f'if kill -0 "$pid" 2>/dev/null; then\n'
        f"    # Kill process tree (children first)\n"
        f'    pkill -TERM -P "$pid" 2>/dev/null || true\n'
        f'    kill -TERM "$pid" 2>/dev/null || true\n'
        f"    sleep 3\n"
        f'    if kill -0 "$pid" 2>/dev/null; then\n'
        f'        pkill -KILL -P "$pid" 2>/dev/null || true\n'
        f'        kill -KILL "$pid" 2>/dev/null || true\n'
        f"        sleep 1\n"
        f"    fi\n"
        f'    echo "Stopped {name} (PID $pid)"\n'
        f"else\n"
        f'    echo "{name} not running"\n'
        f"fi\n"
        f"rm -f {_SERVICES_DIR}/{name}.pid {_SERVICES_DIR}/{name}.sh\n"
    )
    result = _run_on_node(node, stop, timeout=30)
    click.echo(f"  {result.strip()}")
    return "Stopped" in result


_SERVICE_PORTS: dict[str, str] = {
    "neo4j": "_graph_port",
    "embed": "_embed_port",
}


def _service_running(node: str, name: str) -> bool:
    """Check if a named service is running on the compute node.

    First checks the PID file.  If the tracked PID is dead, falls back
    to a port-liveness check — this catches orphaned services that were
    started outside the deploy workflow (or whose PID file is stale).
    """
    check = (
        f"pid=$(cat {_SERVICES_DIR}/{name}.pid 2>/dev/null) || exit 1\n"
        f'kill -0 "$pid" 2>/dev/null || exit 1\n'
        f'echo "running"\n'
    )
    try:
        result = _run_on_node(node, check, timeout=10)
        if "running" in result:
            return True
    except subprocess.CalledProcessError:
        pass

    # Fallback: check if the service port is responding
    port_fn_name = _SERVICE_PORTS.get(name)
    if port_fn_name:
        port = globals()[port_fn_name]()
        try:
            result = _run_on_node(
                node,
                f"ss -tlnp sport = :{port} 2>/dev/null | grep -q LISTEN && echo listening",
                timeout=10,
            )
            if "listening" in result:
                # Port is bound — adopt the process into the PID file
                try:
                    pid_result = _run_on_node(
                        node,
                        f"ss -tlnp sport = :{port} 2>/dev/null"
                        f" | grep -oP 'pid=\\K[0-9]+' | head -1",
                        timeout=10,
                    )
                    pid = pid_result.strip()
                    if pid:
                        _run_remote(
                            f"echo {pid} > {_SERVICES_DIR}/{name}.pid",
                            timeout=5,
                        )
                except (subprocess.CalledProcessError, ValueError):
                    pass
                return True
        except subprocess.CalledProcessError:
            pass

    return False


def _clean_neo4j_locks(node: str) -> None:
    """Remove Neo4j coordination lock files on the compute node.

    GPFS can retain stale ``store_lock`` and ``database_lock`` files
    after process death, preventing startup.

    **CRITICAL**: Only remove these two coordination locks.  Never
    delete Lucene ``write.lock`` files (inside ``schema/index/``
    directories) — doing so corrupts vector indexes and can cause
    total data loss via checkpoint failure.
    """
    clean = (
        "DATA=$HOME/.local/share/imas-codex/neo4j/data && "
        'rm -f "$DATA"/databases/store_lock "$DATA"/databases/*/database_lock '
        "2>/dev/null; echo locks_cleaned\n"
    )
    try:
        _run_on_node(node, clean, timeout=15)
    except subprocess.CalledProcessError:
        pass


def _kill_neo4j_on_node(node: str) -> None:
    """Kill any orphaned Neo4j processes on the compute node.

    Handles cases where PID file is missing but process is still running
    (e.g. after a failed deploy with health check timeout).
    """
    kill_cmd = (
        'pids=$(pgrep -u $USER -f "neo4j.*console|Neo4jCommunity" 2>/dev/null)\n'
        'if [ -n "$pids" ]; then\n'
        '    echo "Killing orphaned neo4j: $pids"\n'
        '    echo "$pids" | xargs kill -9 2>/dev/null || true\n'
        "    sleep 2\n"
        "fi\n"
    )
    try:
        result = _run_on_node(node, kill_cmd, timeout=15)
        if "Killing" in result:
            click.echo(f"  {result.strip()}")
    except subprocess.CalledProcessError:
        pass


def _neo4j_service_command() -> str:
    """Build the shell command to start Neo4j on a compute node.

    Binds to ``0.0.0.0`` on the exclusively-allocated SLURM compute
    node.  Port collisions between users are impossible because each
    user's allocation runs on a different node.

    Uses ``neo4j console`` directly with a host-side ``conf/`` bind
    mount for configuration.  **Never** use the Docker entrypoint —
    it calls ``set-initial-password`` and ``rm -rf conf/*`` on every
    start, which can reinitialize an existing database.
    """
    from imas_codex.graph.profiles import resolve_neo4j
    from imas_codex.settings import get_neo4j_image_shell

    profile = resolve_neo4j(auto_tunnel=False)
    bolt_port = profile.bolt_port
    http_port = profile.http_port
    image = get_neo4j_image_shell()

    return (
        "NEO4J_BASE=$HOME/.local/share/imas-codex/neo4j && "
        'mkdir -p "$NEO4J_BASE"/{data,logs,import,conf} && '
        # Write a minimal conf file for listen addresses
        f'printf "server.bolt.listen_address=0.0.0.0:{bolt_port}\\n'
        f"server.http.listen_address=0.0.0.0:{http_port}\\n"
        f'server.default_listen_address=0.0.0.0\\n"'
        f' > "$NEO4J_BASE/conf/neo4j.conf" && '
        "apptainer exec "
        '--bind "$NEO4J_BASE/data:/data" '
        '--bind "$NEO4J_BASE/logs:/logs" '
        '--bind "$NEO4J_BASE/import:/import" '
        '--bind "$NEO4J_BASE/conf:/var/lib/neo4j/conf" '
        "--writable-tmpfs "
        f"{image} "
        "neo4j console"
    )


def _embed_service_command(gpus: int, workers: int) -> str:
    """Build the shell command to start the embed server on a compute node.

    Binds to ``0.0.0.0`` on the exclusively-allocated SLURM compute
    node.  Accessible from the login node via the compute node's
    hostname on the cluster network.
    """
    port = _embed_port()
    gpu_ids = ",".join(str(i) for i in range(gpus))
    partition = _gpu_partition()
    partition_name = partition["name"]

    return (
        f"CUDA_VISIBLE_DEVICES={gpu_ids} "
        "uv run --offline --extra gpu imas-codex serve embed start "
        f"--host 0.0.0.0 --port {port} "
        f"--gpus {gpu_ids} --workers {workers} --deploy-label {partition_name}"
    )


def _wait_for_health(
    label: str,
    check_cmd: str,
    timeout_s: int = 120,
    success_test: str | None = None,
) -> bool:
    """Wait for a service health check to pass. Returns True on success."""
    click.echo(f"  Waiting for {label}...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        try:
            result = _run_remote(check_cmd, timeout=10)
            if result and result != "(no output)":
                if success_test is None or success_test in result:
                    click.echo(f"  {label} healthy")
                    return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        remaining = int(deadline - time.time())
        if remaining > 0 and remaining % 15 < 5:
            click.echo(f"    {remaining}s remaining...")
    click.echo(f"  Warning: {label} not healthy after {timeout_s}s")
    return False


# ── Login-node deploy fallback (systemd) ─────────────────────────────────


def _deploy_login_embed() -> None:
    """Deploy embed server to login node via systemd."""
    click.echo("Deploying to login node via systemd...")
    try:
        _run_remote(
            "systemctl --user restart imas-codex-embed 2>/dev/null || "
            "systemctl --user start imas-codex-embed",
            timeout=15,
            check=True,
        )
        click.echo("  Service started")
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            "Service not installed. Run: imas-codex serve embed service install"
        ) from exc


def _deploy_login_neo4j() -> None:
    """Deploy Neo4j to login node via systemd."""
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j(auto_tunnel=False)
    service_name = f"imas-codex-neo4j-{profile.name}"
    click.echo(f"Deploying Neo4j [{profile.name}] to login node via systemd...")
    try:
        _run_remote(
            f"systemctl --user restart {service_name} 2>/dev/null || "
            f"systemctl --user start {service_name}",
            timeout=30,
            check=True,
        )
        click.echo("  Service started")
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            "Service not installed. Run: imas-codex graph service install"
        ) from exc


def _show_embed_info() -> None:
    """Print embed server info from /info endpoint."""
    import json

    from imas_codex.settings import get_embed_host

    host = get_embed_host() or "localhost"
    port = _embed_port()
    try:
        result = _run_remote(f"curl -sf http://{host}:{port}/info", timeout=10)
        if result and result != "(no output)":
            info = json.loads(result)
            model = info.get("model", {}).get("name", "unknown")
            device = info.get("model", {}).get("device", "unknown")
            click.echo(f"  Model: {model}")
            click.echo(f"  Device: {device}")
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        pass


def _show_embed_info_on_node(node: str, port: int) -> None:
    """Print embed server info by curling /info on the compute node."""
    import json

    try:
        result = _run_on_node(
            node, f"curl -sf http://localhost:{port}/info", timeout=10
        )
        if result and result != "(no output)":
            info = json.loads(result)
            model = info.get("model", {}).get("name", "unknown")
            device = info.get("model", {}).get("device", "unknown")
            click.echo(f"  Model: {model}")
            click.echo(f"  Device: {device}")
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        pass


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

    When ``[embedding].scheduler = "slurm"``, ensures a SLURM allocation
    exists on the compute node and starts the embed service on it.
    The allocation is shared with Neo4j — each service has an independent
    lifecycle.

    When scheduler is not set, deploys via systemd on the login node.

    Idempotent: no-op if already running.

    \b
    Examples:
        imas-codex serve embed deploy          # Deploy per config
        imas-codex serve embed deploy -g 2     # 2 GPUs
    """
    if workers is None:
        workers = gpus

    if _is_compute_target():
        alloc = _ensure_allocation(gpus)
        node = alloc["node"]
        if _service_running(node, "embed"):
            click.echo(
                f"Already running: embed on {node} "
                f"(alloc {alloc['job_id']}, {alloc['gres']}, {alloc['time']})"
            )
            click.echo(f"  URL: http://{node}:{_embed_port()}")
            return

        click.echo(f"Starting embed server on {node}...")
        cmd = _embed_service_command(gpus, workers)
        _start_service(node, "embed", cmd)

        port = _embed_port()
        _wait_for_health(
            "embed",
            f"curl -sf http://{node}:{port}/health",
            success_test='"status"',
        )

        click.echo(f"  URL: http://{node}:{port}")
    else:
        _deploy_login_embed()
        click.echo(f"  URL: http://localhost:{_embed_port()}")


@serve_embed.command("stop")
def embed_stop() -> None:
    """Stop the embedding server.

    Stops the embed service on the compute node (allocation stays alive)
    or stops the systemd service on the login node.

    \b
    Examples:
        imas-codex serve embed stop
    """
    stopped = False

    # Stop on compute node if allocation exists
    alloc = _get_allocation()
    if alloc and alloc["state"] == "RUNNING":
        if _stop_service(alloc["node"], "embed"):
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

    Stops the embed service and redeploys.  The SLURM allocation and
    other services (e.g. Neo4j) are not affected.

    \b
    Examples:
        imas-codex serve embed restart         # Restart
        imas-codex serve embed restart -g 2    # Restart with 2 GPUs
    """
    if workers is None:
        workers = gpus

    # Stop
    alloc = _get_allocation()
    if alloc and alloc["state"] == "RUNNING":
        _stop_service(alloc["node"], "embed")
        time.sleep(2)
    try:
        _run_remote("systemctl --user stop imas-codex-embed 2>/dev/null", timeout=15)
    except subprocess.CalledProcessError:
        pass

    # Deploy
    if _is_compute_target():
        alloc = _ensure_allocation(gpus)
        node = alloc["node"]
        click.echo(f"Starting embed server on {node}...")
        cmd = _embed_service_command(gpus, workers)
        _start_service(node, "embed", cmd)
        port = _embed_port()
        _wait_for_health(
            "embed",
            f"curl -sf http://{node}:{port}/health",
            success_test='"status"',
        )
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
    log_file = f"{_SERVICES_DIR}/embed.log"
    _tail_log(log_file, follow, lines)


# ── Neo4j deploy/stop/status/logs commands ───────────────────────────────


@serve.group("neo4j")
def serve_neo4j_group() -> None:
    """Manage Neo4j graph server deployment.

    When ``[graph].scheduler = "slurm"``, Neo4j runs on a shared SLURM
    allocation alongside the embedding server.  Each service has an
    independent lifecycle — start/stop Neo4j without affecting embed.

    When scheduler is not set, uses systemd on the login node.

    \\b
      imas-codex serve neo4j deploy     Deploy per config
      imas-codex serve neo4j stop       Stop the server
      imas-codex serve neo4j status     Check server status
      imas-codex serve neo4j logs       View service logs
    """
    pass


@serve_neo4j_group.command("deploy")
@click.option(
    "--gpus",
    "-g",
    default=_DEFAULT_GPUS,
    type=int,
    help=f"Number of GPUs for allocation (default: {_DEFAULT_GPUS})",
)
def neo4j_deploy(gpus: int) -> None:
    """Deploy Neo4j graph server.

    Ensures the SLURM allocation exists and starts Neo4j on the
    compute node.  The allocation is shared with the embedding server.

    Idempotent: no-op if already running.

    \\b
    Examples:
        imas-codex serve neo4j deploy          # Deploy per config
        imas-codex serve neo4j deploy -g 2     # With 2-GPU allocation
    """
    if _is_graph_compute_target():
        alloc = _ensure_allocation(gpus)
        node = alloc["node"]
        if _service_running(node, "neo4j"):
            click.echo(
                f"Already running: neo4j on {node} "
                f"(alloc {alloc['job_id']}, {alloc['gres']}, {alloc['time']})"
            )
            click.echo(f"  Bolt: bolt://{node}:{_graph_port()}")
            click.echo(f"  HTTP: http://{node}:{_graph_http_port()}")
            return

        # Pre-deploy cleanup: kill orphaned processes, clear lock files
        _kill_neo4j_on_node(node)
        _clean_neo4j_locks(node)

        click.echo(f"Starting Neo4j on {node}...")
        cmd = _neo4j_service_command()
        _start_service(node, "neo4j", cmd)

        # Health check runs ON the compute node (Neo4j binds to localhost)
        http_port = _graph_http_port()
        _wait_for_health(
            "Neo4j",
            f"curl -sf http://{node}:{http_port}/",
        )

        click.echo(f"  Bolt: bolt://{node}:{_graph_port()}")
        click.echo(f"  HTTP: http://{node}:{_graph_http_port()}")
    else:
        _deploy_login_neo4j()
        click.echo(f"  Bolt: bolt://localhost:{_graph_port()}")
        click.echo(f"  HTTP: http://localhost:{_graph_http_port()}")


@serve_neo4j_group.command("stop")
def neo4j_stop() -> None:
    """Stop the Neo4j graph server.

    Stops the Neo4j service on the compute node.  The SLURM allocation
    and other services (e.g. embed) are not affected.

    \\b
    Examples:
        imas-codex serve neo4j stop
    """
    stopped = False

    alloc = _get_allocation()
    if alloc and alloc["state"] == "RUNNING":
        if _stop_service(alloc["node"], "neo4j"):
            stopped = True

    # Stop systemd service
    try:
        result = _run_remote(
            "systemctl --user list-units 'imas-codex-neo4j-*' --no-pager "
            "--plain --no-legend 2>/dev/null | awk '{print $1}' | head -1",
            timeout=10,
        )
        service = result.strip()
        if service and service != "(no output)":
            _run_remote(f"systemctl --user stop {service}", timeout=15, check=True)
            click.echo(f"  Stopped {service}")
            stopped = True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    if not stopped:
        click.echo("No active Neo4j server found")


@serve_neo4j_group.command("status")
def neo4j_status() -> None:
    """Check Neo4j graph server status.

    \\b
    Examples:
        imas-codex serve neo4j status
    """
    from imas_codex.graph.profiles import resolve_neo4j
    from imas_codex.settings import get_graph_scheduler

    profile = resolve_neo4j(auto_tunnel=False)
    scheduler = get_graph_scheduler()

    click.echo(f"Neo4j [{profile.name}]:")
    click.echo(f"  Location: {profile.location}")
    click.echo(f"  Scheduler: {scheduler}")
    click.echo(f"  Bolt: {profile.bolt_port}, HTTP: {profile.http_port}")

    if scheduler == "slurm":
        alloc = _get_allocation()
        if alloc:
            click.echo(
                f"  Allocation: {alloc['job_id']} {alloc['state']} on {alloc['node']} "
                f"({alloc['gres']}, {alloc['time']})"
            )
            if alloc["state"] == "RUNNING":
                running = _service_running(alloc["node"], "neo4j")
                if running:
                    # Health check
                    try:
                        result = _run_remote(
                            f"curl -sf http://{alloc['node']}:{profile.http_port}/",
                            timeout=10,
                        )
                        if result and result != "(no output)":
                            click.echo("  Status: running, healthy")
                        else:
                            click.echo("  Status: running, not responding")
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                        click.echo("  Status: running, not responding")
                else:
                    click.echo("  Status: not started (allocation available)")
        else:
            click.echo("  Allocation: none")
    else:
        try:
            from imas_codex.graph.health import check_graph_health

            gh = check_graph_health()
            icon = "healthy" if gh.status == "ok" else "unhealthy"
            click.echo(f"  Status: {icon}")
            if gh.neo4j_version:
                click.echo(f"  Version: {gh.neo4j_version}")
            if gh.error:
                click.echo(f"  Error: {gh.error}")
        except Exception as e:
            click.echo(f"  Error: {e}")


@serve_neo4j_group.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (tail -f)")
@click.option(
    "--lines", "-n", default=50, type=int, help="Number of lines (default: 50)"
)
def neo4j_logs(follow: bool, lines: int) -> None:
    """View Neo4j service logs.

    \\b
    Examples:
        imas-codex serve neo4j logs            # Last 50 lines
        imas-codex serve neo4j logs -f         # Follow live
        imas-codex serve neo4j logs -n 100     # Last 100 lines
    """
    log_file = f"{_SERVICES_DIR}/neo4j.log"
    _tail_log(log_file, follow, lines)


# ── Allocation management command ────────────────────────────────────────


@serve.group("alloc")
def serve_alloc_group() -> None:
    """Manage the shared SLURM compute allocation.

    The allocation is a persistent SLURM job that reserves GPU resources.
    Services (embed, neo4j) run independently on the allocated node.

    \\b
      imas-codex serve alloc status     Show allocation status
      imas-codex serve alloc release    Cancel the allocation
    """
    pass


@serve_alloc_group.command("status")
def alloc_status() -> None:
    """Show SLURM allocation status and running services.

    \\b
    Examples:
        imas-codex serve alloc status
    """
    alloc = _get_allocation()
    if not alloc:
        click.echo("No allocation running")
        click.echo("  Create with: imas-codex serve embed deploy")
        click.echo("           or: imas-codex serve neo4j deploy")
        return

    click.echo(f"Allocation: {alloc['job_id']} ({alloc['state']})")
    click.echo(f"  Node: {alloc['node']}")
    click.echo(f"  GPUs: {alloc['gres']}")
    click.echo(f"  Time: {alloc['time']}")

    if alloc["state"] == "RUNNING":
        node = alloc["node"]
        click.echo("Services:")
        for svc, port_fn in [
            ("neo4j", _graph_http_port),
            ("embed", _embed_port),
            ("llm", _llm_port),
        ]:
            running = _service_running(node, svc)
            status = "running" if running else "stopped"
            port = port_fn()
            click.echo(f"  {svc}: {status} (port {port})")


@serve_alloc_group.command("release")
def alloc_release() -> None:
    """Release the SLURM allocation (stops all services).

    \\b
    Examples:
        imas-codex serve alloc release
    """
    cancelled = _cancel_allocation()
    if cancelled:
        click.echo(f"Released allocation (job {cancelled[0]})")
    else:
        click.echo("No allocation to release")


# ── Shared log tail helper ───────────────────────────────────────────────


def _tail_log(log_file: str, follow: bool, lines: int) -> None:
    """Tail a log file on the remote host."""
    # Check if file exists
    try:
        result = _run_remote(f"test -f {log_file} && echo exists", timeout=5)
        if "exists" not in result:
            raise click.ClickException(f"Log file not found: {log_file}")
    except subprocess.CalledProcessError:
        raise click.ClickException(f"Log file not found: {log_file}") from None

    if follow:
        ssh = _facility_ssh()
        if ssh:
            os.execvp("ssh", ["ssh", ssh, f"tail -f {log_file}"])
        else:
            os.execvp("tail", ["tail", "-f", log_file])
    else:
        output = _run_remote(f"tail -n {lines} {log_file}", timeout=10)
        click.echo(output)


# ============================================================================
# Unified Deploy
# ============================================================================


@serve.command("deploy")
@click.option(
    "--gpus",
    "-g",
    default=_DEFAULT_GPUS,
    type=int,
    help=f"Number of GPUs for SLURM allocation (default: {_DEFAULT_GPUS})",
)
@click.option(
    "--workers",
    "-w",
    default=None,
    type=int,
    help="Embed worker processes (default: same as gpus)",
)
@click.option(
    "--skip-tunnels",
    is_flag=True,
    help="Skip SSH tunnel setup after deploying services",
)
def serve_deploy(gpus: int, workers: int | None, skip_tunnels: bool) -> None:
    """Deploy all services (Neo4j, embedding, LLM proxy) and start tunnels.

    Ensures a SLURM allocation exists (when configured), then deploys
    each service that is not already running.  After all services are up,
    starts SSH tunnels so services are accessible from the local machine.

    Idempotent: services already running are skipped.

    \b
    Examples:
        imas-codex serve deploy              # Deploy everything
        imas-codex serve deploy -g 2         # With 2-GPU allocation
        imas-codex serve deploy --skip-tunnels  # Services only
    """
    import shutil

    from imas_codex.cli.tunnel import (
        _get_tunnel_ports,
        _resolve_host,
        _start_tunnels,
    )

    if workers is None:
        workers = gpus

    any_compute = _is_graph_compute_target() or _is_compute_target()

    # ── Step 1: SLURM allocation ─────────────────────────────────────
    node: str | None = None
    if any_compute:
        click.echo("── Allocation ──")
        alloc = _ensure_allocation(gpus)
        node = alloc["node"]
        click.echo(f"  Ready: job {alloc['job_id']} on {node} ({alloc['gres']})")
        click.echo()

    # ── Step 2: Neo4j ────────────────────────────────────────────────
    click.echo("── Neo4j ──")
    if _is_graph_compute_target():
        assert node is not None
        if _service_running(node, "neo4j"):
            click.echo(f"  Already running on {node}")
        else:
            _kill_neo4j_on_node(node)
            _clean_neo4j_locks(node)
            click.echo(f"  Starting Neo4j on {node}...")
            _start_service(node, "neo4j", _neo4j_service_command())
            _wait_for_health(
                "Neo4j",
                f"curl -sf http://{node}:{_graph_http_port()}/",
            )
        click.echo(f"  Bolt: bolt://{node}:{_graph_port()}")
        click.echo(f"  HTTP: http://{node}:{_graph_http_port()}")
    else:
        _deploy_login_neo4j()
        click.echo(f"  Bolt: bolt://localhost:{_graph_port()}")
        click.echo(f"  HTTP: http://localhost:{_graph_http_port()}")
    click.echo()

    # ── Step 3: Embedding server ─────────────────────────────────────
    click.echo("── Embedding ──")
    if _is_compute_target():
        assert node is not None
        if _service_running(node, "embed"):
            click.echo(f"  Already running on {node}")
        else:
            click.echo(f"  Starting embed on {node}...")
            cmd = _embed_service_command(gpus, workers)
            _start_service(node, "embed", cmd)
            _wait_for_health(
                "embed",
                f"curl -sf http://{node}:{_embed_port()}/health",
                success_test='"status"',
            )
        click.echo(f"  URL: http://{node}:{_embed_port()}")
    else:
        _deploy_login_embed()
        click.echo(f"  URL: http://localhost:{_embed_port()}")
    click.echo()

    # ── Step 4: LLM proxy (login node — needs outbound HTTPS) ──────
    click.echo("── LLM Proxy ──")
    _deploy_login_llm()
    port = _llm_port()
    _wait_for_health(
        "LLM proxy",
        f"curl -sf http://localhost:{port}/",
        timeout_s=60,
    )
    click.echo(f"  URL: http://localhost:{port}")
    click.echo()

    # ── Step 5: SSH tunnels ──────────────────────────────────────────
    if skip_tunnels:
        click.echo("── Tunnels (skipped) ──")
        return

    click.echo("── Tunnels ──")
    try:
        host = _resolve_host(None)
    except click.ClickException:
        click.echo("  No remote host configured — tunnels not needed")
        return

    use_autossh = bool(shutil.which("autossh"))
    ports = _get_tunnel_ports(host, neo4j=False, embed=False, llm=False)
    if not ports:
        click.echo("  No tunnel ports configured")
        return

    ok = _start_tunnels(host, ports, use_autossh)
    if ok == len(ports):
        click.echo(f"  ✓ All {ok} tunnel(s) active")
    elif ok > 0:
        click.echo(f"  ⚠ {ok}/{len(ports)} tunnel(s) active")
    else:
        click.echo("  ✗ No tunnels could be established")


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


# ============================================================================
# Unified Serve Status
# ============================================================================


@serve.command("status")
def serve_status() -> None:
    """Show status of all servers (graph, embedding, LLM proxy).

    Checks health, location, deployment mode, and URLs for each service.

    \\b
    Examples:
        imas-codex serve status
    """
    from imas_codex.settings import (
        get_embed_remote_url,
        get_embed_scheduler,
        get_embedding_location,
        get_graph_scheduler,
    )

    # ── Venv Check ────────────────────────────────────────────────────────
    click.echo("Environment:")
    _missing: list[str] = []
    try:
        import torch

        _torch_cuda = torch.cuda.is_available()
        _torch_label = f"torch={torch.__version__}"
        if _torch_cuda:
            _torch_label += " (CUDA)"
        else:
            _torch_label += " (CPU-only)"
        click.echo(f"  {_torch_label}")
    except ImportError:
        _missing.append("torch (cpu/gpu extra)")
        _torch_cuda = False
    if _missing:
        click.echo(f"  ✗ Missing: {', '.join(_missing)}")
        click.echo("  Hint: uv sync --extra gpu  (or --extra cpu)")
    else:
        click.echo("  ✓ All serve dependencies available")
    click.echo()

    # ── SLURM Allocation ─────────────────────────────────────────────────
    alloc = _get_allocation()
    compute_node: str | None = None
    compute_services: dict[str, bool] = {}

    if alloc:
        click.echo(f"Allocation: {alloc['job_id']} ({alloc['state']})")
        click.echo(
            f"  Node: {alloc['node']}, GPUs: {alloc['gres']}, Time: {alloc['time']}"
        )
        if alloc["state"] == "RUNNING":
            compute_node = alloc["node"]
            for svc in ("neo4j", "embed"):
                running = _service_running(compute_node, svc)
                compute_services[svc] = running
                click.echo(f"  {svc}: {'running' if running else 'stopped'}")
        click.echo()

    # ── Graph (Neo4j) ────────────────────────────────────────────────────
    click.echo("Neo4j Graph:")
    scheduler = get_graph_scheduler()

    if compute_node and compute_services.get("neo4j"):
        # SLURM: check health directly on the compute node
        bolt_port = _graph_port()
        http_port = _graph_http_port()
        click.echo(f"  Bolt: bolt://{compute_node}:{bolt_port}")
        click.echo(f"  HTTP: http://{compute_node}:{http_port}")
        click.echo(f"  Scheduler: {scheduler}")
        try:
            result = _run_on_node(
                compute_node,
                f"curl -sf http://localhost:{http_port}/",
                timeout=10,
            )
            if result and result != "(no output)":
                click.echo("  ✓ Status: running")
                # Get detailed stats via curl on compute
                try:
                    stats = _run_on_node(
                        compute_node,
                        f"curl -sf http://localhost:{http_port}/db/neo4j/tx/commit "
                        '-H "Content-Type: application/json" '
                        '-d \'{"statements":[{"statement":"MATCH (n) RETURN count(n) AS nodes"},{"statement":"MATCH ()-[r]->() RETURN count(r) AS rels"}]}\' '
                        f"-u neo4j:imas-codex",
                        timeout=10,
                    )
                    if stats:
                        import json

                        data = json.loads(stats)
                        results = data.get("results", [])
                        if len(results) >= 2:
                            nodes = results[0]["data"][0]["row"][0]
                            rels = results[1]["data"][0]["row"][0]
                            click.echo(f"  Nodes: {nodes}, Relationships: {rels}")
                except Exception:
                    pass
            else:
                click.echo("  ✗ Status: not responding")
        except subprocess.CalledProcessError:
            click.echo("  ✗ Status: not responding on compute node")

        # Check tunnel accessibility
        from imas_codex.remote.tunnel import TUNNEL_OFFSET, is_tunnel_active

        tunnel_bolt = bolt_port + TUNNEL_OFFSET
        tunnel_http = http_port + TUNNEL_OFFSET
        bolt_ok = is_tunnel_active(tunnel_bolt)
        http_ok = is_tunnel_active(tunnel_http)
        if bolt_ok and http_ok:
            click.echo(
                f"  Tunnel: ✓ bolt://localhost:{tunnel_bolt}, "
                f"http://localhost:{tunnel_http}"
            )
        else:
            click.echo(
                f"  Tunnel: ✗ bolt:{tunnel_bolt} "
                f"{'✓' if bolt_ok else '✗'}, "
                f"http:{tunnel_http} {'✓' if http_ok else '✗'}"
            )
    elif compute_node and not compute_services.get("neo4j"):
        click.echo(f"  ✗ Status: stopped (not running on {compute_node})")
        click.echo(f"  Scheduler: {scheduler}")
    else:
        # Non-SLURM or no allocation: use profile-based health check
        try:
            from imas_codex.graph.health import check_graph_health

            gh = check_graph_health()
            status_icon = "✓" if gh.status == "ok" else "✗"
            click.echo(f"  {status_icon} Status: {gh.status}")
            click.echo(f"  Name: {gh.name}")
            click.echo(f"  Location: {gh.location}")
            if gh.host:
                click.echo(f"  Host: {gh.host}")
            click.echo(f"  Bolt URL: {gh.bolt_url}")
            click.echo(f"  HTTP URL: {gh.http_url}")
            click.echo(f"  Scheduler: {scheduler}")
            if gh.status == "ok":
                if gh.neo4j_version:
                    click.echo(f"  Version: {gh.neo4j_version}")
                if gh.node_count is not None:
                    click.echo(
                        f"  Nodes: {gh.node_count}, "
                        f"Relationships: {gh.relationship_count}"
                    )
                if gh.facilities:
                    click.echo(f"  Facilities: {', '.join(gh.facilities)}")
            if gh.error:
                click.echo(f"  Error: {gh.error}")
        except Exception as e:
            click.echo(f"  ✗ Error: {e}")

    click.echo()

    # ── Embedding Server ─────────────────────────────────────────────────
    click.echo("Embedding Server:")
    embed_location = get_embedding_location()
    embed_scheduler = get_embed_scheduler()
    click.echo(f"  Location: {embed_location}")
    click.echo(f"  Scheduler: {embed_scheduler}")

    if compute_node and compute_services.get("embed"):
        # SLURM: check health directly on the compute node
        port = _embed_port()
        click.echo(f"  Compute: http://{compute_node}:{port}")
        try:
            result = _run_on_node(
                compute_node,
                f"curl -sf http://localhost:{port}/health",
                timeout=10,
            )
            if result and '"status"' in result:
                click.echo("  ✓ Status: running")
                _show_embed_info_on_node(compute_node, port)
            else:
                click.echo("  ✗ Status: not responding")
        except subprocess.CalledProcessError:
            click.echo("  ✗ Status: not responding on compute node")

        # Check tunnel / local port accessibility
        from imas_codex.remote.tunnel import is_tunnel_active

        if is_tunnel_active(port):
            click.echo(f"  Tunnel: ✓ localhost:{port}")
        else:
            click.echo(f"  Tunnel: ✗ localhost:{port} not reachable")
    elif compute_node and not compute_services.get("embed"):
        click.echo(f"  ✗ Status: stopped (not running on {compute_node})")
    else:
        # Non-SLURM: check via URL
        embed_url = get_embed_remote_url()
        if embed_url:
            click.echo(f"  URL: {embed_url}")

        if embed_location != "local" and embed_url:
            try:
                from imas_codex.embeddings.client import RemoteEmbeddingClient

                client = RemoteEmbeddingClient(embed_url)
                if client.is_available(timeout=3.0):
                    info = client.get_detailed_info()
                    click.echo("  ✓ Status: ok")
                    if info:
                        click.echo(f"  Model: {info['model']['name']}")
                        click.echo(f"  Device: {info['model']['device']}")
                        if info["gpu"]["name"]:
                            click.echo(f"  GPU: {info['gpu']['name']}")
                        location_label = info["server"].get("location")
                        if location_label:
                            click.echo(f"  Deploy: {location_label}")
                        uptime_h = info["server"]["uptime_seconds"] / 3600
                        click.echo(f"  Uptime: {uptime_h:.1f}h")
                else:
                    click.echo("  ✗ Status: not available")
            except Exception as e:
                click.echo(f"  ✗ Status: error ({e})")
        elif embed_location == "local":
            click.echo("  ✓ Mode: in-process (no server)")

    click.echo()

    # ── LLM Proxy ────────────────────────────────────────────────────────
    click.echo("LLM Proxy:")
    from imas_codex.settings import get_llm_location

    llm_location = get_llm_location()
    click.echo(f"  Location: {llm_location}")
    click.echo("  Deploy: systemd (uv tool run)")

    # Check systemd status on login node
    try:
        result = _run_remote(
            "systemctl --user is-active imas-codex-llm 2>/dev/null || true",
            timeout=10,
        )
        if "active" in result and "inactive" not in result:
            click.echo("  ✓ Status: running (login node)")
        else:
            click.echo("  ✗ Status: not running on login node")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        click.echo("  ✗ Status: could not check (SSH error)")

    # Check local accessibility (direct or via tunnel)
    from imas_codex.settings import get_llm_proxy_url

    llm_url = get_llm_proxy_url()
    click.echo(f"  URL: {llm_url}")
    try:
        import httpx

        resp = httpx.get(f"{llm_url}/", timeout=3.0)
        if resp.status_code == 200:
            click.echo("  ✓ Reachable: ok")
        else:
            click.echo(f"  ✗ Reachable: unhealthy (HTTP {resp.status_code})")
    except httpx.ConnectError:
        click.echo("  ✗ Reachable: no (tunnel needed?)")
    except httpx.RemoteProtocolError:
        click.echo("  ✗ Reachable: port in use by non-HTTP process")
    except Exception:
        click.echo("  ✗ Reachable: no")
