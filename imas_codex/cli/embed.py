"""Embedding server management commands."""

import logging
import os
import subprocess
import time

import click

logger = logging.getLogger(__name__)


@click.group()
def embed():
    """Manage GPU embedding server.

    Deploy mode is determined by ``[embedding].scheduler`` in pyproject.toml:
      slurm → SLURM batch job on GPU compute node
      (omit) → systemd service on login node

    The embedding location (``[embedding].location``) determines where
    commands are sent (e.g. ``iter``).

    Override with env vars: IMAS_CODEX_EMBED_SCHEDULER, IMAS_CODEX_EMBEDDING_LOCATION

    \b
      imas-codex embed deploy     Deploy per config
      imas-codex embed stop       Stop the server
      imas-codex embed restart    Restart (stop + deploy)
      imas-codex embed status     Check server and SLURM job status
      imas-codex embed logs       View SLURM job logs
      imas-codex embed start      Start embedding server locally
      imas-codex embed service    Manage systemd service
    """
    pass


@embed.command("start")
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
        imas-codex embed start

        # Use specific GPU
        imas-codex embed start --gpu 1

        # Multi-worker on 4 GPUs (Titan)
        imas-codex embed start --gpus 0,1,2,3 --workers 4

        # Custom port
        imas-codex embed start --port 18766

        # Auto-shutdown after 30 min idle
        imas-codex embed start --idle-timeout 1800
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


@embed.command("status")
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
        imas-codex embed status          # Health + SLURM
        imas-codex embed status --local  # Local GPU capability
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
        click.echo("  Deploy with: imas-codex embed deploy")
        click.echo("  Check tunnel: imas-codex tunnel status")


@embed.command("service")
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
        imas-codex embed service install
        imas-codex embed service start
        imas-codex embed service status
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
ExecStart={uv_path} run --extra gpu --project {project_dir} imas-codex embed start --host 0.0.0.0 --port {port} --deploy-label {deploy_label}
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
        click.echo("  Or:    imas-codex embed service start")

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
                "Service not installed. Run: imas-codex embed service install"
            )
        subprocess.run(["systemctl", "--user", "start", "imas-codex-embed"], check=True)
        click.echo("Service started")

    elif action == "stop":
        subprocess.run(["systemctl", "--user", "stop", "imas-codex-embed"], check=True)
        click.echo("Service stopped")


# ── Embed deploy/stop/restart/logs commands ──────────────────────────────

from imas_codex.cli.services import (  # noqa: E402
    _DEFAULT_GPUS,
    _EMBED_JOB,
    _SERVICES_DIR,
    _cancel_service_job,
    _deploy_login_embed,
    _embed_port,
    _format_service_status,
    _get_embed_job,
    _is_compute_target,
    _run_remote,
    _tail_log,
    deploy_embed,
)


@embed.command("deploy")
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
        imas-codex embed deploy          # Deploy per config
        imas-codex embed deploy -g 2     # 2 GPUs
    """
    if _is_compute_target():
        deploy_embed(gpus, workers)
    else:
        _deploy_login_embed()
        click.echo(f"  URL: http://localhost:{_embed_port()}")


@embed.command("stop")
def embed_stop() -> None:
    """Stop the embedding server.

    Cancels the embed SLURM job (which stops the server process),
    or stops the systemd service on the login node.

    \b
    Examples:
        imas-codex embed stop
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


@embed.command("restart")
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
        imas-codex embed restart         # Restart
        imas-codex embed restart -g 2    # Restart with 2 GPUs
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


@embed.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (tail -f)")
@click.option(
    "--lines", "-n", default=50, type=int, help="Number of lines (default: 50)"
)
def embed_logs(follow: bool, lines: int) -> None:
    """View embedding server logs.

    \b
    Examples:
        imas-codex embed logs            # Last 50 lines
        imas-codex embed logs -f         # Follow live
        imas-codex embed logs -n 100     # Last 100 lines
    """
    log_file = f"{_SERVICES_DIR}/codex-embed.log"
    _tail_log(log_file, follow, lines)
