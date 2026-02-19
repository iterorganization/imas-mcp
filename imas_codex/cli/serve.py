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

    \b
      imas-codex serve llm start      Start LiteLLM proxy (foreground)
      imas-codex serve llm status     Check proxy health
      imas-codex serve llm service    Manage systemd service
    """
    pass


@serve_llm.command("start")
@click.option(
    "--host",
    envvar="LITELLM_HOST",
    default="0.0.0.0",
    help="Host to bind (default: all interfaces)",
)
@click.option(
    "--port",
    envvar="LITELLM_PORT",
    default=4000,
    type=int,
    help="Port to bind (default: 4000)",
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
    port: int,
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
        imas-codex serve llm start --port 4001

        # Custom config
        imas-codex serve llm start --config my_config.yaml
    """
    import shutil
    import subprocess
    from pathlib import Path

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
    help="Proxy URL (default: http://localhost:4000)",
)
def llm_status(url: str | None) -> None:
    """Check LiteLLM proxy health.

    Examples:
        imas-codex serve llm status
        imas-codex serve llm status --url http://remote:4000
    """
    import httpx

    if url is None:
        url = os.environ.get("LITELLM_PROXY_URL", "http://localhost:4000")

    click.echo(f"LiteLLM Proxy ({url}):")

    try:
        resp = httpx.get(f"{url}/health", timeout=5.0)
        if resp.status_code == 200:
            click.echo("  ✓ Healthy")
            # Show model availability
            models_resp = httpx.get(
                f"{url}/v1/models",
                timeout=5.0,
                headers={
                    "Authorization": f"Bearer {os.environ.get('LITELLM_MASTER_KEY', 'sk-litellm-imas-codex')}"
                },
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
    except Exception as e:
        click.echo(f"  ✗ Error: {e}")


@serve_llm.command("service")
@click.argument(
    "action", type=click.Choice(["install", "uninstall", "status", "start", "stop"])
)
@click.option(
    "--port", default=4000, type=int, help="Port for the proxy (default: 4000)"
)
def llm_service(action: str, port: int) -> None:
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

        service_content = f"""[Unit]
Description=IMAS Codex LiteLLM Proxy
After=network.target

[Service]
Type=simple
WorkingDirectory={project_dir}
EnvironmentFile=-{env_file}
Environment="PATH={Path.home()}/.local/bin:/usr/local/bin:/usr/bin"
ExecStart={uv_path} tool run --with 'litellm[proxy]>=1.81.0' -- litellm --config {config_path} --host 0.0.0.0 --port {port}
ExecStop=/bin/kill -15 $MAINPID
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
            # Grab recent journal lines for diagnostics
            journal = subprocess.run(
                [
                    "journalctl",
                    "--user",
                    "-eu",
                    "imas-codex-llm",
                    "-n",
                    "10",
                    "--no-pager",
                ],
                capture_output=True,
                text=True,
            )
            raise click.ClickException(
                "Failed to start LLM proxy service.\n"
                + (journal.stdout or journal.stderr or "")
                + "\nCheck full logs: journalctl --user -xeu imas-codex-llm"
            )
        click.echo("LLM proxy service started")

    elif action == "stop":
        subprocess.run(["systemctl", "--user", "stop", "imas-codex-llm"], check=True)
        click.echo("LLM proxy service stopped")


# ============================================================================
# Serve Embed Command Group
# ============================================================================


@serve.group("embed")
def serve_embed() -> None:
    """Manage GPU embedding server.

    \b
      imas-codex serve embed deploy     Deploy to Titan (4 GPUs) or login node
      imas-codex serve embed stop       Stop the server (SLURM or systemd)
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
    "--location",
    default=None,
    type=str,
    help="Deployment location label (e.g., 'titan', 'login'). Exposed in /health.",
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
    location: str | None,
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

    # Set deployment location label
    if location:
        import imas_codex.embeddings.server as embed_server

        embed_server._location = location
        logger.info(f"Location: {location}")

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

    # SLURM job status
    try:
        jobs = _embed_slurm_jobs()
        if jobs:
            click.echo("SLURM Job:")
            for job in jobs:
                click.echo(
                    f"  {job['job_id']} {job['state']} on {job['node']} "
                    f"({job['gres']}, {job['time']})"
                )
        else:
            click.echo("SLURM Job: none")
    except Exception:
        click.echo("SLURM Job: unavailable (not on ITER?)")

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
            location = info["server"].get("location")
            if location:
                click.echo(f"  Location: {location}")
            click.echo(f"  Uptime: {uptime_h:.1f}h")
            if timeout_s > 0:
                click.echo(f"  Idle: {idle_s:.0f}s / {timeout_s}s timeout")
    else:
        click.echo("  ✗ Not available")
        click.echo(
            "  Deploy with: imas-codex serve embed deploy\n"
            "  Or check tunnel: ssh -L 18765:98dci4-gpu-0002:18765 iter"
        )


@serve_embed.command("service")
@click.argument(
    "action", type=click.Choice(["install", "uninstall", "status", "start", "stop"])
)
@click.option("--gpu", default="1", help="CUDA device to use (default: 1)")
@click.option(
    "--location", default="login", help="Deployment location label (default: login)"
)
def embed_service(action: str, gpu: str, location: str) -> None:
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
ExecStart={uv_path} run --extra gpu --project {project_dir} imas-codex serve embed start --host 0.0.0.0 --port {port} --location {location}
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
# Embed Server Deployment (SLURM on Titan, systemd on login)
# ============================================================================

_EMBED_JOB = "codex-embed"
_TITAN_NODE = "98dci4-gpu-0002"
_TITAN_PARTITION = "titan"
_DEFAULT_GPUS = 4
_EMBED_PORT = 18765
_PROJECT = "~/Code/imas-codex"


def _iter_ssh() -> str | None:
    """SSH host for ITER, or None if already on ITER."""
    return None if os.uname().nodename.startswith("98dci4-") else "iter"


def _run_iter(cmd: str, timeout: int = 30, check: bool = False) -> str:
    """Run a command on ITER (locally if on ITER, via SSH otherwise)."""
    from imas_codex.remote.executor import run_command

    return run_command(cmd, ssh_host=_iter_ssh(), timeout=timeout, check=check)


def _embed_slurm_jobs() -> list[dict]:
    """List active codex-embed SLURM jobs."""
    try:
        out = _run_iter(
            f'squeue -n {_EMBED_JOB} -u "$USER" --format="%A|%T|%M|%N|%b" --noheader'
        )
    except subprocess.CalledProcessError:
        return []
    jobs = []
    for line in out.strip().split("\n"):
        line = line.strip()
        if not line or line == "(no output)":
            continue
        parts = line.split("|")
        if len(parts) >= 5:
            jobs.append(
                {
                    "job_id": parts[0].strip(),
                    "state": parts[1].strip(),
                    "time": parts[2].strip(),
                    "node": parts[3].strip(),
                    "gres": parts[4].strip(),
                }
            )
    return jobs


def _cancel_embed_jobs() -> list[str]:
    """Cancel all codex-embed SLURM jobs. Returns cancelled job IDs."""
    jobs = _embed_slurm_jobs()
    cancelled = []
    for job in jobs:
        try:
            _run_iter(f"scancel {job['job_id']}", check=True)
            cancelled.append(job["job_id"])
        except subprocess.CalledProcessError:
            pass
    return cancelled


def _deploy_titan(gpus: int, workers: int, pull: bool) -> None:
    """Deploy embed server to Titan via SLURM."""
    import base64

    # Cancel existing jobs
    jobs = _embed_slurm_jobs()
    if jobs:
        for job in jobs:
            click.echo(
                f"Cancelling job {job['job_id']} ({job['state']} on {job['node']})"
            )
            _run_iter(f"scancel {job['job_id']}")
        time.sleep(2)

    # Git pull
    if pull:
        click.echo("Pulling latest code...")
        output = _run_iter(
            f"cd {_PROJECT} && git pull --no-rebase origin main", timeout=60
        )
        for line in output.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("[stderr]"):
                click.echo(f"  {line}")

    # Generate SLURM script
    gpu_ids = ",".join(str(i) for i in range(gpus))
    cpus = gpus * 4
    script = (
        "#!/bin/bash\n"
        f"#SBATCH --partition={_TITAN_PARTITION}\n"
        f"#SBATCH --gres=gpu:{gpus}\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        "#SBATCH --mem=64G\n"
        "#SBATCH --time=UNLIMITED\n"
        f"#SBATCH --job-name={_EMBED_JOB}\n"
        f"#SBATCH --nodelist={_TITAN_NODE}\n"
        "#SBATCH --output=slurm-embed-%j.log\n"
        "\n"
        "set -euo pipefail\n"
        f"export CUDA_VISIBLE_DEVICES={gpu_ids}\n"
        "mkdir -p ~/.local/share/imas-codex/logs\n"
        f"cd {_PROJECT}\n"
        'echo "Starting embed server on $(hostname) at $(date)"\n'
        f'echo "GPUs: {gpus}, Workers: {workers}, '
        f'CUDA_VISIBLE_DEVICES={gpu_ids}"\n'
        "exec uv run --offline --extra gpu imas-codex serve embed start "
        f"--host 0.0.0.0 --port {_EMBED_PORT} "
        f"--gpus {gpu_ids} --workers {workers} --location titan\n"
    )

    # Submit via base64 to avoid shell quoting issues over SSH
    script_b64 = base64.b64encode(script.encode()).decode()
    submit_cmd = (
        f'echo "{script_b64}" | base64 -d > /tmp/codex-embed-deploy.sh && '
        "sbatch /tmp/codex-embed-deploy.sh && "
        "rm -f /tmp/codex-embed-deploy.sh"
    )
    output = _run_iter(submit_cmd, timeout=30, check=True)
    # First line is "Submitted batch job XXXXX"
    click.echo(output.split("\n")[0])

    # Wait for health
    click.echo(f"Waiting for server on {_TITAN_NODE}:{_EMBED_PORT}...")
    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(5)
        try:
            result = _run_iter(
                f"curl -sf http://{_TITAN_NODE}:{_EMBED_PORT}/health",
                timeout=10,
            )
            if '"status"' in result and "ok" in result.lower():
                click.echo(f"  Server healthy on {_TITAN_NODE}")
                _show_server_info()
                return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        remaining = int(deadline - time.time())
        if remaining > 0 and remaining % 15 < 5:
            click.echo(f"  Waiting... ({remaining}s remaining)")

    click.echo("  Server not healthy after 120s — check logs:")
    click.echo("  imas-codex serve embed logs")


def _deploy_login() -> None:
    """Deploy embed server to login node via systemd."""
    click.echo("Deploying to login node via systemd...")
    try:
        _run_iter(
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


def _show_server_info() -> None:
    """Print embed server info from /info endpoint."""
    import json

    try:
        result = _run_iter(
            f"curl -sf http://{_TITAN_NODE}:{_EMBED_PORT}/info",
            timeout=10,
        )
        if result and result != "(no output)":
            info = json.loads(result)
            model = info.get("model", {}).get("name", "unknown")
            device = info.get("model", {}).get("device", "unknown")
            click.echo(f"  Model: {model}")
            click.echo(f"  Device: {device}")
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        pass


@serve_embed.command("deploy")
@click.option(
    "--target",
    "-t",
    type=click.Choice(["titan", "login"]),
    default="titan",
    help="Deploy target (default: titan)",
)
@click.option(
    "--gpus",
    "-g",
    default=_DEFAULT_GPUS,
    type=int,
    help=f"Number of GPUs for Titan (default: {_DEFAULT_GPUS})",
)
@click.option(
    "--workers",
    "-w",
    default=None,
    type=int,
    help="Worker processes (default: same as gpus)",
)
@click.option(
    "--pull/--no-pull",
    default=True,
    help="Git pull before deploying (default: pull)",
)
def embed_deploy(
    target: str,
    gpus: int,
    workers: int | None,
    pull: bool,
) -> None:
    """Deploy embedding server to Titan GPU node or login node.

    For Titan (default): cancel existing job, git pull, sbatch new job,
    wait for health.

    For login: restart systemd service.

    Works from ITER login node or WSL/workstation (via SSH).

    \b
    Examples:
        imas-codex serve embed deploy              # 4 GPUs on Titan
        imas-codex serve embed deploy -g 2         # 2 GPUs on Titan
        imas-codex serve embed deploy -t login     # systemd on login
        imas-codex serve embed deploy --no-pull    # Skip git pull
    """
    if workers is None:
        workers = gpus

    if target == "titan":
        _deploy_titan(gpus, workers, pull)
    else:
        _deploy_login()


@serve_embed.command("stop")
@click.option(
    "--target",
    "-t",
    type=click.Choice(["titan", "login"]),
    default=None,
    help="Stop target (auto-detected if not specified)",
)
def embed_stop(target: str | None) -> None:
    """Stop the embedding server.

    For Titan: cancel SLURM job.
    For login: stop systemd service.
    Auto-detects target if not specified.

    \b
    Examples:
        imas-codex serve embed stop              # Auto-detect
        imas-codex serve embed stop -t titan     # Cancel SLURM job
        imas-codex serve embed stop -t login     # Stop systemd
    """
    if target is None:
        jobs = _embed_slurm_jobs()
        target = "titan" if jobs else "login"

    if target == "titan":
        jobs = _embed_slurm_jobs()
        if not jobs:
            click.echo("No active embed SLURM jobs found.")
            return
        for job in jobs:
            _run_iter(f"scancel {job['job_id']}")
            click.echo(
                f"Cancelled job {job['job_id']} ({job['state']} on {job['node']})"
            )
    else:
        try:
            _run_iter(
                "systemctl --user stop imas-codex-embed",
                timeout=15,
                check=True,
            )
            click.echo("Service stopped")
        except subprocess.CalledProcessError:
            click.echo("Service not running or not installed")


@serve_embed.command("restart")
@click.option(
    "--target",
    "-t",
    type=click.Choice(["titan", "login"]),
    default="titan",
    help="Deploy target (default: titan)",
)
@click.option(
    "--gpus",
    "-g",
    default=_DEFAULT_GPUS,
    type=int,
    help=f"Number of GPUs for Titan (default: {_DEFAULT_GPUS})",
)
@click.option(
    "--workers",
    "-w",
    default=None,
    type=int,
    help="Worker processes (default: same as gpus)",
)
@click.option(
    "--pull/--no-pull",
    default=True,
    help="Git pull before deploying (default: pull)",
)
def embed_restart(
    target: str,
    gpus: int,
    workers: int | None,
    pull: bool,
) -> None:
    """Restart the embedding server (stop + deploy).

    \b
    Examples:
        imas-codex serve embed restart             # 4 GPUs on Titan
        imas-codex serve embed restart -g 2        # 2 GPUs
        imas-codex serve embed restart -t login    # Restart login service
    """
    if workers is None:
        workers = gpus

    # Stop
    if target == "titan":
        cancelled = _cancel_embed_jobs()
        for jid in cancelled:
            click.echo(f"Cancelled job {jid}")
        if cancelled:
            time.sleep(2)
    else:
        try:
            _run_iter(
                "systemctl --user stop imas-codex-embed",
                timeout=15,
            )
        except subprocess.CalledProcessError:
            pass

    # Deploy
    if target == "titan":
        _deploy_titan(gpus, workers, pull)
    else:
        _deploy_login()


@serve_embed.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (tail -f)")
@click.option(
    "--lines", "-n", default=50, type=int, help="Number of lines (default: 50)"
)
def embed_logs(follow: bool, lines: int) -> None:
    """View embedding server SLURM job logs.

    \b
    Examples:
        imas-codex serve embed logs            # Last 50 lines
        imas-codex serve embed logs -f         # Follow live
        imas-codex serve embed logs -n 100     # Last 100 lines
    """
    # Find the log file: active job or latest
    jobs = _embed_slurm_jobs()
    if jobs:
        log_file = f"{_PROJECT}/slurm-embed-{jobs[0]['job_id']}.log"
    else:
        result = _run_iter(f"ls -t {_PROJECT}/slurm-embed-*.log 2>/dev/null | head -1")
        result = result.strip()
        if not result or result == "(no output)":
            raise click.ClickException("No SLURM log files found")
        log_file = result.split("\n")[0]

    if follow:
        ssh = _iter_ssh()
        if ssh:
            os.execvp("ssh", ["ssh", ssh, f"tail -f {log_file}"])
        else:
            os.execvp("tail", ["tail", "-f", log_file])
    else:
        output = _run_iter(f"tail -n {lines} {log_file}", timeout=10)
        click.echo(output)
