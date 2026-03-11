"""Graph CLI server commands — start, stop, status, shell, profiles."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click

from imas_codex.graph.ghcr import (
    get_git_info,
    get_local_graph_manifest,
    get_registry,
    require_apptainer,
)
from imas_codex.graph.neo4j_ops import (
    check_database_locks,
    check_stale_neo4j_process,
    is_neo4j_running,
    neo4j_image,
    secure_data_directory,
)

NEO4J_IMAGE = neo4j_image()


# ============================================================================
# Server Commands
# ============================================================================


@click.command()
@click.option("--image", envvar="NEO4J_IMAGE", default=None)
@click.option("--data-dir", envvar="NEO4J_DATA", default=None)
@click.option("--password", envvar="NEO4J_PASSWORD", default=None)
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground")
def graph_start(
    image: str | None,
    data_dir: str | None,
    password: str | None,
    foreground: bool,
) -> None:
    """Start Neo4j server.

    Automatically detects the deployment mode from the graph location:

    - **SLURM compute** (e.g. ``titan``): ensures a SLURM allocation
      and starts Neo4j on the compute node.
    - **Remote**: starts via systemd on the remote host.
    - **Local**: starts via Apptainer directly.
    """
    import platform

    from imas_codex.graph.profiles import get_graph_location, resolve_neo4j

    profile = resolve_neo4j()
    password = password or profile.password

    # ── SLURM compute dispatch ───────────────────────────────────────────
    from imas_codex.remote.locations import resolve_location

    location = get_graph_location()
    loc_info = resolve_location(location)

    if loc_info.is_compute:
        from imas_codex.cli.services import (
            _graph_http_port,
            _graph_port,
            deploy_neo4j,
        )

        job = deploy_neo4j()
        node = job["node"]

        if job.get("_fallback"):
            raise click.ClickException(
                f"SLURM is unavailable and Neo4j is not running on {node}. "
                "Cannot start Neo4j without a SLURM allocation. "
                "Try again when SLURM recovers."
            )

        click.echo(f"  Bolt: bolt://{node}:{_graph_port()}")
        click.echo(f"  HTTP: http://{node}:{_graph_http_port()}")
        return
    # ── End SLURM compute dispatch ───────────────────────────────────────

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            remote_is_neo4j_running,
            remote_service_action,
            resolve_remote_service_name,
        )

        if remote_is_neo4j_running(profile.http_port, profile.host):
            click.echo(
                f"Neo4j [{profile.name}] is already running "
                f"on {profile.host}:{profile.http_port}"
            )
            return

        service = resolve_remote_service_name(profile.name, profile.host)
        click.echo(
            f"Starting Neo4j [{profile.name}] on {profile.host} "
            f"(bolt:{profile.bolt_port}, http:{profile.http_port})..."
        )
        remote_service_action("start", service, profile.host, timeout=60)

        import time

        for _ in range(30):
            if remote_is_neo4j_running(profile.http_port, profile.host):
                click.echo(f"✓ Neo4j [{profile.name}] ready on {profile.host}")
                return
            time.sleep(1)
        click.echo("Warning: Neo4j may still be starting")
        return
    # ── End remote dispatch ──────────────────────────────────────────────

    if platform.system() in ("Windows", "Darwin"):
        click.echo("On Windows/Mac, use Docker: docker compose up -d neo4j", err=True)
        raise SystemExit(1)

    require_apptainer()

    # Check for conflicting tunnel before starting
    from imas_codex.graph.profiles import check_graph_conflict

    conflict = check_graph_conflict(profile.bolt_port)
    if conflict:
        raise click.ClickException(conflict)

    image_path = Path(image) if image else NEO4J_IMAGE
    data_path = Path(data_dir) if data_dir else profile.data_dir

    if not image_path.exists():
        from imas_codex.settings import get_neo4j_version

        raise click.ClickException(
            f"Neo4j image not found: {image_path}\n"
            f"Pull: apptainer pull docker://neo4j:{get_neo4j_version()}"
        )

    if is_neo4j_running(profile.http_port):
        click.echo(
            f"Neo4j [{profile.name}] is already running on port {profile.http_port}"
        )
        return

    # Check for stale processes and locks before starting
    has_stale, stale_msg = check_stale_neo4j_process(data_path)
    if has_stale and stale_msg:
        click.echo(f"Warning: {stale_msg}", err=True)

    has_lock, lock_msg = check_database_locks(data_path)
    if has_lock and lock_msg:
        click.echo(f"Warning: {lock_msg}", err=True)
        return

    for subdir in ["data", "logs", "conf", "import"]:
        (data_path / subdir).mkdir(parents=True, exist_ok=True)

    # Secure permissions to prevent other users accessing our database
    secure_data_directory(data_path)

    # Ensure neo4j.conf exists with proper memory/recovery settings.
    # create_graph_dir writes it at init time, but it may be missing
    # if the directory was created manually.
    conf_file = data_path / "conf" / "neo4j.conf"
    if not conf_file.exists():
        from imas_codex.graph.dirs import DEFAULT_NEO4J_CONF

        conf_file.write_text(
            DEFAULT_NEO4J_CONF.format(
                listen_address="127.0.0.1",
                bolt_port=profile.bolt_port,
                http_port=profile.http_port,
            )
        )

    cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{data_path}/data:/data",
        "--bind",
        f"{data_path}/logs:/logs",
        "--bind",
        f"{data_path}/import:/import",
        "--bind",
        f"{data_path}/conf:/var/lib/neo4j/conf",
        "--writable-tmpfs",
        "--env",
        "NEO4J_server_jvm_additional=-Dfile.encoding=UTF-8 "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        str(image_path),
        "neo4j",
        "console",
    ]

    click.echo(
        f"Starting Neo4j [{profile.name}] (bolt:{profile.bolt_port}, http:{profile.http_port})"
    )

    if foreground:
        subprocess.run(cmd)
    else:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid_file = data_path / "neo4j.pid"
        pid_file.write_text(str(proc.pid))

        click.echo(f"Neo4j starting in background (PID: {proc.pid})")

        import time

        for _ in range(30):
            if is_neo4j_running(profile.http_port):
                click.echo(f"Neo4j [{profile.name}] ready")
                click.echo(f"  Bolt URL: bolt://localhost:{profile.bolt_port}")
                click.echo(f"  HTTP URL: http://localhost:{profile.http_port}")
                return
            time.sleep(1)

        click.echo("Warning: Neo4j may still be starting")


@click.command()
@click.option("--data-dir", envvar="NEO4J_DATA", default=None)
def graph_stop(data_dir: str | None) -> None:
    """Stop Neo4j server.

    Automatically detects the deployment mode from the graph location:

    - **SLURM compute**: stops Neo4j on the compute node.
    - **Remote**: stops via systemd on the remote host.
    - **Local**: stops the local Apptainer process.
    """
    import signal

    from imas_codex.graph.profiles import get_graph_location, resolve_neo4j

    profile = resolve_neo4j()

    # ── SLURM compute dispatch ───────────────────────────────────────────
    from imas_codex.remote.locations import resolve_location

    location = get_graph_location()
    loc_info = resolve_location(location)

    if loc_info.is_compute:
        from imas_codex.cli.services import (
            _NEO4J_JOB,
            _cancel_service_job,
            _get_neo4j_job,
        )

        job = _get_neo4j_job()
        if job and job["state"] == "RUNNING":
            node = job["node"]
            if _cancel_service_job(_NEO4J_JOB):
                click.echo(f"✓ Neo4j [{profile.name}] stopped on {node}")
            else:
                click.echo(f"Failed to stop Neo4j [{profile.name}]")
        else:
            click.echo(f"Neo4j [{profile.name}] not running (no SLURM job)")
        return
    # ── End SLURM compute dispatch ───────────────────────────────────────

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            remote_service_action,
            resolve_remote_service_name,
        )

        service = resolve_remote_service_name(profile.name, profile.host)
        click.echo(f"Stopping Neo4j [{profile.name}] on {profile.host}...")
        remote_service_action("stop", service, profile.host, timeout=60)
        click.echo(f"✓ Neo4j [{profile.name}] stopped on {profile.host}")
        return
    # ── End remote dispatch ──────────────────────────────────────────────

    data_path = Path(data_dir) if data_dir else profile.data_dir
    pid_file = data_path / "neo4j.pid"

    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            click.echo(f"Sent SIGTERM to Neo4j [{profile.name}] (PID: {pid})")
            pid_file.unlink()
        except ProcessLookupError:
            click.echo(f"Neo4j [{profile.name}] process not found (stale PID file)")
            pid_file.unlink()
    else:
        result = subprocess.run(["pkill", "-f", "neo4j.*console"], capture_output=True)
        click.echo(
            f"Neo4j [{profile.name}] stopped"
            if result.returncode == 0
            else f"Neo4j [{profile.name}] not running"
        )


@click.command()
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
def graph_status(registry: str | None) -> None:
    """Show Neo4j graph status with color-coded SLURM resource usage."""
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)

    from imas_codex import __version__

    click.echo("Local status:")
    click.echo(f"  Version: {__version__}")
    click.echo(f"  Git commit: {git_info['commit_short']}")
    click.echo(f"  Git tag: {git_info['tag'] or '(none)'}")
    click.echo(f"  Is fork: {git_info['is_fork']}")
    click.echo(f"  Target registry: {target_registry}")

    manifest = get_local_graph_manifest()
    if manifest:
        click.echo("\nGraph manifest:")
        click.echo(f"  Version: {manifest.get('version', 'unknown')}")
        click.echo(f"  Pushed: {manifest.get('pushed', False)}")
        if manifest.get("pushed_version"):
            click.echo(f"  Pushed as: {manifest['pushed_version']}")
        if manifest.get("pulled_from"):
            click.echo(f"  Pulled from: {manifest['pulled_from']}")
        if manifest.get("loaded_from"):
            click.echo(f"  Loaded from: {manifest['loaded_from']}")
        if manifest.get("git_commit"):
            click.echo(f"  Git commit: {manifest['git_commit'][:7]}")
        if manifest.get("dev_base_version"):
            click.echo(
                f"  Dev revision: {manifest['dev_base_version']}"
                f"-r{manifest.get('dev_revision', '?')}"
            )

    # SLURM job status with color-coded resource usage
    try:
        from imas_codex.cli.services import _format_service_status, _get_neo4j_job

        job = _get_neo4j_job()
        if job:
            click.echo("\nSLURM:")
            for line in _format_service_status(job, "neo4j"):
                click.echo(line)
    except Exception:
        pass

    running = is_neo4j_running()
    state_str = (
        click.style("running", fg="green")
        if running
        else click.style("stopped", fg="red")
    )
    click.echo(f"\nNeo4j: {state_str}")

    from imas_codex.graph.profiles import resolve_neo4j
    from imas_codex.remote.executor import is_local_host

    try:
        profile = resolve_neo4j()
        is_remote = profile.host is not None and not is_local_host(profile.host)
        click.echo(f"  Graph: {profile.name}")
        click.echo(f"  Location: {profile.location}{' (remote)' if is_remote else ''}")
        if is_remote:
            click.echo(f"  URI: {profile.uri}")
        else:
            click.echo(f"  Data: {profile.data_dir}")
        click.echo(f"  Bolt: {profile.bolt_port}, HTTP: {profile.http_port}")
    except Exception:
        pass

    if running:
        try:
            from imas_codex.graph.client import GraphClient
            from imas_codex.graph.meta import get_graph_meta

            gc = GraphClient.from_profile()
            meta = get_graph_meta(gc)
            gc.close()
            if meta:
                click.echo("\nGraph identity (GraphMeta):")
                click.echo(f"  Name: {meta.get('name', '?')}")
                facilities = meta.get("facilities") or []
                click.echo(
                    f"  Facilities: {', '.join(facilities) if facilities else '(none)'}"
                )
                if meta.get("updated_at"):
                    click.echo(f"  Updated: {meta['updated_at']}")
            else:
                click.echo(
                    "\nGraph identity: not initialized"
                    "\n  Run: imas-codex graph init <name> -f <facility>"
                )
        except Exception:
            pass


@click.command()
@click.option("--image", envvar="NEO4J_IMAGE", default=None)
@click.option("--password", envvar="NEO4J_PASSWORD", default=None)
def graph_shell(image: str | None, password: str | None) -> None:
    """Open Cypher shell to Neo4j."""
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()
    password = password or profile.password
    image_path = Path(image) if image else NEO4J_IMAGE

    if not image_path.exists():
        raise click.ClickException(f"Neo4j image not found: {image_path}")

    click.echo(f"Connecting to Neo4j [{profile.name}] at localhost:{profile.bolt_port}")
    subprocess.run(
        [
            "apptainer",
            "exec",
            "--writable-tmpfs",
            str(image_path),
            "cypher-shell",
            "-a",
            f"bolt://localhost:{profile.bolt_port}",
            "-u",
            profile.username,
            "-p",
            password,
        ]
    )


@click.command()
def graph_profiles() -> None:
    """List Neo4j location profiles and their port assignments."""
    from imas_codex.graph.dirs import list_local_graphs
    from imas_codex.graph.profiles import (
        get_graph_location,
        list_profiles,
    )
    from imas_codex.remote.executor import is_local_host

    active_location = get_graph_location()
    profiles = list_profiles()

    click.echo("Locations:")
    for p in profiles:
        marker = "→" if p.location == active_location else " "
        running = "running" if is_neo4j_running(p.http_port) else "stopped"
        is_remote = p.host is not None and not is_local_host(p.host)
        location_label = f"remote ({p.host})" if is_remote else "local"
        click.echo(
            f"  {marker} {p.location:<10s}  bolt:{p.bolt_port}  http:{p.http_port}  "
            f"{location_label:<20s}  [{running}]"
        )

    graphs = list_local_graphs()
    if graphs:
        click.echo("\nLocal graphs:")
        for g in graphs:
            marker = "→" if g.active else " "
            click.echo(f"  {marker} {g.name}")
