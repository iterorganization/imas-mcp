"""Embedding server management commands."""

import logging
import os
import subprocess
import tempfile
import time

import click

logger = logging.getLogger(__name__)


@click.group()
def embed():
    """Manage GPU embedding server.

    Deploy target is derived from ``[embedding].location`` in pyproject.toml.
    When the location maps to a SLURM compute partition in the facility
    YAML (e.g. ``titan`` → ``iter.yaml`` → ``scheduler: slurm``), deploy
    submits a SLURM batch job.  Otherwise deploys via systemd on the
    login node.

    Override: IMAS_CODEX_EMBEDDING_LOCATION env var.

    \b
      imas-codex embed start      Start the server (SLURM/systemd/foreground)
      imas-codex embed stop       Stop the server
      imas-codex embed restart    Restart (stop + start)
      imas-codex embed status     Check server and SLURM job status
      imas-codex embed logs       View SLURM job logs
      imas-codex embed service    Manage systemd service
    """
    pass


@embed.command("start")
@click.option(
    "--foreground",
    "-f",
    is_flag=True,
    help="Run server in foreground (for debugging or SLURM batch scripts)",
)
@click.option(
    "--no-slurm",
    is_flag=True,
    help="Deploy via SSH+nohup instead of SLURM (for when node is draining)",
)
@click.option(
    "--gpus",
    "-g",
    default=None,
    type=str,
    help="GPU allocation: count for deploy (e.g. 4), or comma-separated IDs "
    "for foreground (e.g. '0,1,2,3'). Default: 4 for deploy.",
)
@click.option(
    "--workers",
    "-w",
    default=None,
    type=int,
    help="Worker processes (default: same as GPU count)",
)
@click.option(
    "--host",
    envvar="EMBED_HOST",
    default="0.0.0.0",
    help="Host to bind [foreground only] (default: 0.0.0.0)",
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
    help="Logging level [foreground only]",
)
@click.option(
    "--gpu",
    envvar="CUDA_VISIBLE_DEVICES",
    default=None,
    help="Single CUDA device [foreground only] (e.g., 0 or 1)",
)
@click.option(
    "--idle-timeout",
    default=0,
    type=int,
    help="Auto-shutdown after N seconds idle [foreground only] (0=disabled)",
)
@click.option(
    "--deploy-label",
    default=None,
    type=str,
    help="Deployment label exposed in /health [foreground only]",
)
def embed_start(
    foreground: bool,
    no_slurm: bool,
    gpus: str | None,
    workers: int | None,
    host: str,
    port: int | None,
    log_level: str,
    gpu: str | None,
    idle_timeout: int,
    deploy_label: str | None,
) -> None:
    """Start the embedding server.

    Auto-detects deployment mode from ``[embedding].location``:

    \b
    - SLURM compute (e.g. titan): submits a SLURM batch job
    - Login node: starts via systemd
    - --foreground: runs server directly (for debugging)
    - --no-slurm: SSH+nohup to compute node (when node is draining)

    SLURM batch scripts call this with --foreground internally.

    \b
    Examples:
        imas-codex embed start           # Deploy per config
        imas-codex embed start -g 2      # Deploy with 2 GPUs
        imas-codex embed start -f        # Run in foreground
        imas-codex embed start -f --gpu 1  # Foreground on GPU 1
        imas-codex embed start --no-slurm  # Bypass SLURM
    """
    # Auto-detect foreground: if SLURM_JOB_ID is set, we're inside a batch
    # script and should run the server directly
    if os.environ.get("SLURM_JOB_ID") and not foreground:
        foreground = True

    if foreground:
        _start_foreground(
            host=host,
            port=port,
            log_level=log_level,
            gpu=gpu,
            idle_timeout=idle_timeout,
            deploy_label=deploy_label,
            workers=workers or 1,
            gpus=gpus,
        )
    elif no_slurm:
        from imas_codex.cli.services import deploy_embed_noslurm

        gpu_count = int(gpus) if gpus else _DEFAULT_GPUS
        deploy_embed_noslurm(gpu_count, workers)
    else:
        _start_deploy(gpus=gpus, workers=workers)


def _start_foreground(
    *,
    host: str,
    port: int | None,
    log_level: str,
    gpu: str | None,
    idle_timeout: int,
    deploy_label: str | None,
    workers: int,
    gpus: str | None,
) -> None:
    """Run the embedding server in the foreground."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    for handler in root_logger.handlers:
        handler.setLevel(getattr(logging, log_level))

    if gpu is not None and gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        logger.info(f"Using CUDA device: {gpu}")

    if gpus is not None:
        os.environ["IMAS_CODEX_GPU_POOL"] = gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        if workers <= 1:
            workers = len(gpus.split(","))
        logger.info(f"Multi-GPU mode: pool={gpus}, workers={workers}")

        # Clean up stale lock files from previous server instances.
        # Lock files are keyed by master PID — old ones from dead servers
        # are harmless (flock released on process exit) but clutter /tmp.
        import glob

        my_pid = str(os.getpid())
        for lock_file in glob.glob(
            os.path.join(tempfile.gettempdir(), "codex-embed-gpu-*-slot-*.lock")
        ):
            # Only remove files NOT belonging to this instance
            basename = os.path.basename(lock_file)
            # Format: codex-embed-gpu-{pid}-slot-{n}.lock
            parts = basename.split("-")
            if len(parts) >= 5 and parts[3] != my_pid:
                try:
                    os.unlink(lock_file)
                except OSError:
                    pass

    if port is None:
        from imas_codex.settings import get_embed_server_port

        port = get_embed_server_port()

    if idle_timeout > 0:
        import imas_codex.embeddings.server as embed_server

        embed_server._idle_timeout = idle_timeout
        logger.info(f"Idle timeout: {idle_timeout}s")

    if deploy_label:
        import imas_codex.embeddings.server as embed_server

        embed_server._location = deploy_label
        logger.info(f"Deploy label: {deploy_label}")

    logger.info(f"Starting embedding server on {host}:{port}")

    try:
        import uvicorn

        if workers > 1:
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


def _start_deploy(
    *,
    gpus: str | None,
    workers: int | None,
) -> None:
    """Deploy the embedding server via SLURM or systemd."""
    from imas_codex.cli.services import _embed_port

    if _is_compute_target():
        gpu_count = int(gpus) if gpus else _DEFAULT_GPUS
        deploy_embed(gpu_count, workers)
    else:
        _deploy_login_embed()
        click.echo(f"  URL: http://localhost:{_embed_port()}")


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

    # Show compute node state (draining/down = jobs won't start)
    try:
        from imas_codex.cli.services import _get_node_state, _gpu_entry

        host = _gpu_entry()["location"]
        node_state, reason = _get_node_state(host)
        if node_state in ("drained", "draining", "down", "down*", "drain", "drng"):
            click.echo(
                click.style(f"  ⚠ Node {host} is {node_state}: {reason}", fg="yellow")
            )
            click.echo("    SLURM will not schedule new jobs on this node.")
    except Exception:
        pass

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
            location = info["server"].get("location")
            hostname = info["server"].get("hostname")
            if location:
                loc_str = f"  Location: {location}"
                if hostname:
                    loc_str += f" ({hostname})"
                click.echo(loc_str)
            uptime_h = info["server"]["uptime_seconds"] / 3600
            click.echo(f"  Uptime: {uptime_h:.1f}h")
            idle_s = info["server"].get("idle_seconds", 0)
            timeout_s = info["server"].get("idle_timeout", 0)
            if timeout_s > 0:
                from imas_codex.cli.services import _colored_bar

                idle_bar = _colored_bar(idle_s, timeout_s)
                click.echo(f"  Idle: {idle_bar}  {idle_s:.0f}s / {timeout_s}s timeout")
            elif idle_s > 0:
                idle_min = idle_s / 60
                if idle_min > 60:
                    click.echo(f"  Idle: {idle_min / 60:.1f}h")
                elif idle_min > 1:
                    click.echo(f"  Idle: {idle_min:.0f}m")
                else:
                    click.echo(f"  Idle: {idle_s:.0f}s")

        # Per-worker GPU status from /workers
        workers_info = client.get_workers_info()
        if workers_info and workers_info.get("workers"):
            from imas_codex.cli.services import _colored_bar

            workers = workers_info["workers"]
            click.echo(
                f"  Workers: {len(workers)} / GPUs: {workers_info.get('gpu_pool', [])}"
            )
            total_requests = 0
            total_texts = 0
            for w in workers:
                gpu_idx = w.get("worker_gpu", "?")
                gpu = w.get("gpu", {})
                gpu_name = gpu.get("name", "unknown")
                used = gpu.get("memory_used_mb")
                total = gpu.get("memory_total_mb")
                free = gpu.get("memory_free_mb")
                stats = w.get("stats", {})
                req_count = stats.get("request_count", 0)
                total_requests += req_count
                total_texts += stats.get("total_texts", 0)

                if used is not None and total:
                    bar = _colored_bar(used, total, width=15)
                    click.echo(
                        f"    GPU {gpu_idx}: {bar}  "
                        f"{used} / {total} MB  "
                        f"({req_count} reqs)"
                    )
                elif free is not None and total:
                    used_calc = total - free
                    bar = _colored_bar(used_calc, total, width=15)
                    click.echo(
                        f"    GPU {gpu_idx}: {bar}  "
                        f"{used_calc} / {total} MB  "
                        f"({req_count} reqs)"
                    )
                else:
                    click.echo(f"    GPU {gpu_idx}: {gpu_name}  ({req_count} reqs)")
            if total_requests > 0:
                click.echo(
                    f"  Total: {total_requests} requests, {total_texts} texts embedded"
                )
        elif info and info.get("gpu", {}).get("name"):
            # Fallback: server too old for /workers, show single GPU
            from imas_codex.cli.services import _colored_bar

            gpu = info["gpu"]
            click.echo(f"  GPU: {gpu['name']} ({gpu['memory_mb']} MB)")
            gpu_used = gpu.get("memory_used_mb")
            gpu_total = gpu.get("memory_total_mb")
            if gpu_used is not None and gpu_total:
                gpu_bar = _colored_bar(gpu_used, gpu_total)
                click.echo(f"  VRAM: {gpu_bar}  {gpu_used} MB / {gpu_total} MB")
            stats = info.get("stats", {})
            if stats.get("request_count", 0) > 0:
                click.echo(
                    f"  Requests: {stats['request_count']} "
                    f"({stats['total_texts']} texts, "
                    f"avg {stats['avg_ms_per_request']:.0f}ms)"
                )
    else:
        click.echo("  ✗ Not available")
        click.echo("  Deploy with: imas-codex embed start")
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
ExecStart={uv_path} run --extra gpu --project {project_dir} imas-codex embed start -f --host 0.0.0.0 --port {port} --deploy-label {deploy_label}
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
        try:
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            subprocess.run(
                ["systemctl", "--user", "enable", "imas-codex-embed"], check=True
            )
            click.echo("✓ Service installed and enabled")
            click.echo("  Start: systemctl --user start imas-codex-embed")
            click.echo("  Or:    imas-codex embed service start")
        except subprocess.CalledProcessError:
            click.echo("✓ Service file written but systemd --user unavailable")
            click.echo("  (No D-Bus session bus — common on HPC login nodes)")
            click.echo("  Use instead: imas-codex embed start")
            click.echo("  This will auto-deploy via nohup fallback.")

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


# ── Embed stop/restart/logs commands ─────────────────────────────────────

from imas_codex.cli.services import (  # noqa: E402
    _DEFAULT_GPUS,
    _EMBED_JOB,
    _SERVICES_DIR,
    _cancel_service_job,
    _deploy_login_embed,
    _format_service_status,
    _get_embed_job,
    _is_compute_target,
    _run_remote,
    _tail_log,
    deploy_embed,
)


@embed.command("stop")
def embed_stop() -> None:
    """Stop the embedding server.

    Cancels the embed SLURM job (which stops the server process),
    stops the systemd service, and kills any rogue embed processes
    on the compute node as a safety net.

    \b
    Examples:
        imas-codex embed stop
    """
    stopped = False

    # Cancel embed SLURM job
    slurm_cancelled = _cancel_service_job(_EMBED_JOB)
    if slurm_cancelled:
        click.echo("Stopped embed server (SLURM job cancelled)")
        stopped = True
        # Wait for SLURM to fully terminate processes — scancel is async.
        # uvicorn workers trap SIGTERM; 4s covers typical shutdown.
        time.sleep(4)

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

    # Kill login-node nohup embed processes (non-systemd fallback).
    # Pattern "imas-codex embed start" matches server processes only,
    # not the running stop/restart CLI (which would self-kill).
    try:
        from imas_codex.cli.services import _kill_login_embed

        result = _run_remote(
            'pgrep -u $USER -f "imas-codex embed start" 2>/dev/null || true',
            timeout=10,
        )
        pids = [p.strip() for p in result.strip().split("\n") if p.strip().isdigit()]
        if pids:
            _kill_login_embed()
            pid_list = " ".join(pids)
            click.echo(f"Stopped login embed process(es) (PIDs: {pid_list})")
            stopped = True
    except Exception:
        pass

    # Kill orphan embed processes on the compute node.
    # Match both "imas-codex embed start" AND "uv run.*imas-codex embed"
    # to catch the full process tree.  Uses SIGKILL — torch/uvicorn
    # processes trap SIGTERM and can delay exit for seconds.
    try:
        from imas_codex.cli.services import _gpu_entry, _kill_embed_orphans

        host = _gpu_entry()["location"]
        # Check for orphans before kill to report accurately
        from imas_codex.cli.services import _run_on_node

        result = _run_on_node(
            host,
            'pgrep -u $USER -f "imas-codex embed" 2>/dev/null || true',
            timeout=10,
        )
        pids = [p.strip() for p in result.strip().split("\n") if p.strip().isdigit()]
        if pids:
            _kill_embed_orphans(host)
            if slurm_cancelled:
                # Expected stragglers from the SLURM job we just cancelled
                click.echo(f"  Cleaned up {len(pids)} worker process(es) on {host}")
            else:
                # Genuinely orphaned — no SLURM job was running
                pid_list = " ".join(pids)
                click.echo(
                    click.style(
                        f"⚠ Killed rogue embed process(es) on {host} (PIDs: {pid_list})\n"
                        "  These were NOT managed by SLURM — use 'imas-codex embed start' next time.",
                        fg="yellow",
                    )
                )
            stopped = True
    except Exception:
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
@click.option(
    "--no-slurm",
    is_flag=True,
    help="Deploy via SSH+nohup instead of SLURM (for when node is draining)",
)
def embed_restart(gpus: int, workers: int | None, no_slurm: bool) -> None:
    """Restart the embedding server (stop + start).

    Cancels the existing embed SLURM job and submits a new one.
    Use ``--gpus`` to change GPU allocation on restart.

    \b
    Examples:
        imas-codex embed restart             # Restart via SLURM
        imas-codex embed restart --no-slurm  # Restart via SSH+nohup
        imas-codex embed restart -g 2        # Restart with 2 GPUs
    """
    # Full stop — reuse embed_stop to cancel SLURM, systemd, AND kill rogues
    ctx = click.get_current_context()
    ctx.invoke(embed_stop)
    time.sleep(2)

    # Start
    if no_slurm:
        from imas_codex.cli.services import deploy_embed_noslurm

        deploy_embed_noslurm(gpus, workers)
    elif _is_compute_target():
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
