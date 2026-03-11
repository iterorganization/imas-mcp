"""Graph lifecycle CLI for Neo4j database management.

This module provides the ``imas-codex graph`` command group for:
- Neo4j server management (start, stop, shell, service)
- Graph database export/load/push/pull to GHCR (with per-facility federation)
- Graph lifecycle: init, switch, list, clear
- Security: password rotation (secure)
- Registry: tags, prune
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click

from imas_codex import __version__

# ============================================================================
# Imports from extracted library modules
# ============================================================================
from imas_codex.graph.ghcr import (
    LOCAL_GRAPH_MANIFEST,
    SCOPE_FIX_HINT as _SCOPE_FIX_HINT,
    GITHUB_API as _GITHUB_API,
    delete_tag as _delete_tag,
    dispatch_graph_quality as _dispatch_graph_quality,
    ensure_fresh_version as _ensure_fresh_version,
    fetch_tag_messages as _fetch_tag_messages,
    get_ghcr_owner_and_type as _get_ghcr_owner_and_type,
    get_git_info,
    get_local_graph_manifest,
    get_package_name,
    get_package_version_id as _get_package_version_id,
    get_registry,
    get_version_tag,
    github_api_paginated as _github_api_paginated,
    github_api_request as _github_api_request,
    list_registry_tags as _list_registry_tags,
    login_to_ghcr,
    next_dev_revision as _next_dev_revision,
    require_apptainer,
    require_clean_git,
    require_gh,
    require_oras,
    resolve_latest_tag as _resolve_latest_tag,
    resolve_token as _resolve_token,
    save_dev_revision as _save_dev_revision,
    save_local_graph_manifest,
)
from imas_codex.graph.neo4j_ops import (
    DATA_DIR,
    NEO4J_LOCK_FILE,
    RECOVERY_DIR,
    SERVICES_DIR,
    Neo4jOperation,
    backup_existing_data,
    backup_graph_dump,
    check_database_lock as _check_database_lock,
    check_database_locks,
    check_graph_exists,
    check_stale_neo4j_process,
    is_neo4j_running,
    neo4j_image,
    neo4j_process_info as _neo4j_process_info,
    parse_dump_error as _parse_dump_error,
    run_neo4j_dump as _run_neo4j_dump,
    secure_data_directory,
)
from imas_codex.graph.temp_neo4j import (
    IMAS_DD_LABELS as _IMAS_DD_LABELS,
    create_facility_dump as _create_facility_dump,
    create_imas_only_dump as _create_imas_only_dump,
    dump_temp_neo4j as _dump_temp_neo4j,
    start_temp_neo4j as _start_temp_neo4j,
    stop_temp_neo4j as _stop_temp_neo4j,
    write_temp_neo4j_conf as _write_temp_neo4j_conf,
)

if TYPE_CHECKING:
    from imas_codex.graph.profiles import Neo4jProfile

# Keep module-level name for backward compat with CLI --image defaults
NEO4J_IMAGE = neo4j_image()


# Main Command Group
# ============================================================================


@click.group()
def graph() -> None:
    """Manage graph database lifecycle.

    \b
    Server:
      imas-codex graph start               Start Neo4j (SLURM/systemd/local)
      imas-codex graph stop                Stop Neo4j
      imas-codex graph status              Show status with SLURM resource usage

    \b
    Setup:
      imas-codex graph init NAME           Create a new graph
      imas-codex graph list                List local graph instances
      imas-codex graph switch NAME         Activate a different graph
      imas-codex graph shell               Interactive Cypher REPL

    \b
    Archive & Registry:
      imas-codex graph export              Export graph to archive
      imas-codex graph load <file>         Load graph archive
      imas-codex graph push                Push archive to GHCR
      imas-codex graph pull                Fetch + load from GHCR
      imas-codex graph fetch               Download archive (no load)
      imas-codex graph tags                List GHCR versions
      imas-codex graph prune               Remove GHCR tags

    \b
    Maintenance:
      imas-codex graph clear               Clear all graph data
      imas-codex graph secure              Rotate Neo4j password
    """
    pass


# ============================================================================
# Neo4j Server Commands
# ============================================================================


@graph.command("start")
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


@graph.command("stop")
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


@graph.command("status")
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


@graph.command("shell")
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


@graph.command("profiles")
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


# ============================================================================
# Graph Secure Command
# ============================================================================


@graph.command("secure")
def graph_secure() -> None:
    """Rotate the Neo4j server password.

    Connects to the running Neo4j via Cypher ``ALTER CURRENT USER SET
    PASSWORD`` — no restart required.  Updates ``.env`` and (for remote
    hosts) syncs via SCP.

    Auth lives in the Neo4j system database, not in service files or
    env vars.  ``graph init`` sets the initial password via
    ``set-initial-password`` before first start.

    Falls back to ``set-initial-password`` only when Neo4j is stopped
    (first setup or post-dump-load).
    """
    import secrets

    from imas_codex.graph.profiles import resolve_neo4j
    from imas_codex.settings import get_graph_uri

    profile = resolve_neo4j(auto_tunnel=False)
    new_password = secrets.token_urlsafe(24)

    env_file = Path(".env")
    if not env_file.exists():
        raise click.ClickException(
            ".env file not found in project root.\n"
            "Copy from env.example: cp env.example .env"
        )

    old_password = profile.password

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            remote_is_neo4j_running,
            remote_set_initial_password,
        )

        was_running = remote_is_neo4j_running(profile.http_port, profile.host)

        if was_running:
            # Rotate via Cypher through the SSH tunnel.  get_graph_uri()
            # resolves to the tunnelled bolt port (e.g. bolt://localhost:17687).
            bolt_uri = get_graph_uri()
            _rotate_password_cypher(bolt_uri, old_password, new_password)
        else:
            click.echo("Neo4j not running, using set-initial-password...")
            try:
                remote_set_initial_password(profile.host, new_password, clear_auth=True)
                click.echo("✓ Set Neo4j initial password")
            except Exception as e:
                click.echo(f"Warning: Password set issue: {e}", err=True)

        _update_env_file(env_file, new_password)

        # Sync .env to remote host
        try:
            remote_env = "~/Code/imas-codex/.env"
            result = subprocess.run(
                ["scp", "-q", str(env_file), f"{profile.host}:{remote_env}"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                subprocess.run(
                    ["ssh", profile.host, f"chmod 600 {remote_env}"],
                    capture_output=True,
                    timeout=10,
                )
                click.echo("✓ Synced .env to remote host")
            else:
                click.echo(
                    f"Warning: .env sync failed: {result.stderr.strip()}\n"
                    f"  Run manually: imas-codex config secrets push {profile.host}",
                    err=True,
                )
        except Exception as e:
            click.echo(
                f"Warning: .env sync failed: {e}\n"
                f"  Run manually: imas-codex config secrets push {profile.host}",
                err=True,
            )

        click.echo(f"\n✓ Neo4j server password rotated on {profile.host}")
        return
    # ── End remote dispatch ──────────────────────────────────────────────

    # ── Local path ───────────────────────────────────────────────────────
    was_running = is_neo4j_running(profile.http_port)

    if was_running:
        bolt_uri = f"bolt://localhost:{profile.bolt_port}"
        _rotate_password_cypher(bolt_uri, old_password, new_password)
    else:
        if shutil.which("apptainer") and NEO4J_IMAGE.exists():
            auth_file = profile.data_dir / "data" / "dbms" / "auth.ini"
            if auth_file.exists():
                auth_file.unlink()
            cmd = [
                "apptainer",
                "exec",
                "--bind",
                f"{profile.data_dir}/data:/data",
                "--writable-tmpfs",
                str(NEO4J_IMAGE),
                "neo4j-admin",
                "dbms",
                "set-initial-password",
                new_password,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 and "already set" not in result.stderr.lower():
                click.echo(
                    f"Warning: Password reset issue: {result.stderr.strip()}",
                    err=True,
                )

    _update_env_file(env_file, new_password)

    click.echo("\n✓ Neo4j server password rotated")


def _rotate_password_cypher(
    bolt_uri: str, old_password: str, new_password: str
) -> None:
    """Rotate Neo4j password via Cypher ALTER CURRENT USER SET PASSWORD."""
    try:
        from neo4j import GraphDatabase

        with GraphDatabase.driver(bolt_uri, auth=("neo4j", old_password)) as driver:
            driver.verify_connectivity()
            with driver.session() as session:
                session.run(
                    "ALTER CURRENT USER SET PASSWORD FROM $old TO $new",
                    old=old_password,
                    new=new_password,
                )
        click.echo("✓ Rotated Neo4j server password")
    except Exception as e:
        raise click.ClickException(
            f"Failed to rotate password via Cypher: {e}\n"
            "Ensure Neo4j is running and the current password in .env is correct."
        ) from e


def _update_env_file(env_file: Path, new_password: str) -> None:
    """Update NEO4J_PASSWORD in the .env file."""
    import re

    env_content = env_file.read_text()
    if re.search(r"^NEO4J_PASSWORD=", env_content, re.MULTILINE):
        env_content = re.sub(
            r"^NEO4J_PASSWORD=.*$",
            f"NEO4J_PASSWORD={new_password}",
            env_content,
            flags=re.MULTILINE,
        )
    else:
        env_content = env_content.rstrip() + f"\nNEO4J_PASSWORD={new_password}\n"
    env_file.write_text(env_content)
    env_file.chmod(0o600)
    click.echo("✓ Updated .env")


# Archive & Distribution Commands
# ============================================================================


@graph.command("export")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output archive path (default: auto-named)",
)
@click.option("--no-restart", is_flag=True, help="Don't restart Neo4j after export")
@click.option(
    "--facility",
    "-F",
    "facilities",
    multiple=True,
    help="Facility to include (repeatable). Filters out other facilities.",
)
@click.option(
    "--no-imas",
    is_flag=True,
    help="Exclude IMAS Data Dictionary nodes from export",
)
@click.option(
    "--imas-only",
    is_flag=True,
    help="Export only IMAS Data Dictionary nodes (no facility data)",
)
@click.option(
    "--local",
    is_flag=True,
    help="Also transfer the archive locally (remote graphs only).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show full error output from neo4j-admin.",
)
def graph_export(
    output: str | None,
    no_restart: bool,
    facilities: tuple[str, ...],
    no_imas: bool,
    imas_only: bool,
    local: bool,
    verbose: bool = False,
) -> None:
    """Export graph database to archive.

    When the configured location is remote, the dump is performed on
    the remote host and the archive stays in the remote exports
    directory.  Use ``--local`` to also transfer it back via SCP.
    """
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()

    git_info = get_git_info()
    version_label = git_info["tag"] or f"dev-{git_info['commit_short']}"
    pkg_name = get_package_name(
        facilities=list(facilities), no_imas=no_imas, imas_only=imas_only
    )

    if output:
        output_path = Path(output)
    else:
        from imas_codex.graph.dirs import ensure_exports_dir

        exports = ensure_exports_dir()
        output_path = exports / f"{pkg_name}-{version_label}.tar.gz"

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.cli.graph_progress import (
            GraphProgress,
            remote_operation_streaming,
        )
        from imas_codex.graph.remote import (
            build_remote_export_script,
            scp_from_remote,
        )

        if facilities:
            click.echo(
                "Warning: --facility filtering is not supported for remote export. "
                "The full graph will be exported.",
                err=True,
            )

        _remote_markers_export = {
            "STOPPING": f"Stopping Neo4j on {profile.host}",
            "DUMPING": "Dumping graph database",
            "RECOVERY": "Recovery cycle (clean start/stop)",
            "ARCHIVING": "Creating archive",
            "STARTING": f"Starting Neo4j on {profile.host}",
            "COMPLETE": "Export complete",
        }

        with GraphProgress("export") as gp:
            gp.set_total_phases(2 if (local or output) else 1)

            gp.start_phase(f"Exporting graph [{profile.name}] on {profile.host}")
            script = build_remote_export_script(profile.name)
            try:
                export_output = remote_operation_streaming(
                    script,
                    profile.host,
                    progress=gp,
                    progress_markers=_remote_markers_export,
                    timeout=600,
                )
            except Exception as e:
                gp.fail_phase(str(e))
                raise click.ClickException(
                    f"Remote export on {profile.host} failed: {e}"
                ) from e

            remote_archive = None
            size_str = None
            for line in export_output.strip().splitlines():
                if line.startswith("ARCHIVE_PATH="):
                    remote_archive = line.split("=", 1)[1].strip()
                elif line.startswith("SIZE="):
                    size_str = line.split("=", 1)[1].strip()
            if not remote_archive:
                gp.fail_phase("No archive path in output")
                raise click.ClickException(
                    f"Could not find archive path:\n{export_output}"
                )
            gp.complete_phase(size_str)

            try:
                if local or output:
                    gp.start_phase(f"Transferring from {profile.host}")
                    try:
                        scp_from_remote(remote_archive, output_path, profile.host)
                    except Exception as e:
                        gp.fail_phase(str(e))
                        raise click.ClickException(
                            f"Transfer from {profile.host} failed: {e}"
                        ) from e
                    size_mb = output_path.stat().st_size / 1024 / 1024
                    gp.complete_phase(f"{size_mb:.1f} MB")
            finally:
                # Always clean up remote archive
                from imas_codex.graph.remote import remote_cleanup_archive

                remote_cleanup_archive(remote_archive, profile.host)

        return
    # ── End remote dispatch ──────────────────────────────────────────────

    require_apptainer()

    with Neo4jOperation("graph dump", require_stopped=True) as op:
        if no_restart:
            op.was_running = False

        click.echo(f"Creating archive [{profile.name}]: {output_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            archive_dir = tmp / f"{pkg_name}-{version_label}"
            archive_dir.mkdir()

            click.echo("  Dumping graph database...")
            dumps_dir = profile.data_dir / "dumps"
            dumps_dir.mkdir(parents=True, exist_ok=True)

            _run_neo4j_dump(profile, dumps_dir, verbose=verbose)

            dump_file = dumps_dir / "neo4j.dump"
            if dump_file.exists():
                shutil.move(str(dump_file), str(archive_dir / "graph.dump"))
                size_mb = (archive_dir / "graph.dump").stat().st_size / 1024 / 1024
                click.echo(f"    Graph: {size_mb:.1f} MB")
            else:
                raise click.ClickException("Graph dump file not created")

            # If facilities specified, filter the dump
            if facilities:
                for fac in facilities:
                    click.echo(f"  Filtering dump for facility: {fac}")
                    _create_facility_dump(
                        archive_dir / "graph.dump",
                        fac,
                        archive_dir / "graph.dump",
                    )

            # If imas-only, remove all facility nodes
            if imas_only:
                _create_imas_only_dump(
                    archive_dir / "graph.dump",
                    archive_dir / "graph.dump",
                )

            manifest = {
                "version": __version__,
                "git_commit": git_info["commit"],
                "git_tag": git_info["tag"],
                "timestamp": datetime.now(UTC).isoformat(),
            }
            (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

            click.echo("  Creating archive...")
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(archive_dir, arcname=archive_dir.name)

    size_mb = output_path.stat().st_size / 1024 / 1024
    click.echo(f"Archive created: {output_path} ({size_mb:.1f} MB)")


@graph.command("load")
@click.argument("archive", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Overwrite existing data")
@click.option("--no-restart", is_flag=True, help="Don't restart Neo4j after load")
@click.option("--password", envvar="NEO4J_PASSWORD", default=None)
def graph_load(
    archive: str,
    force: bool,
    no_restart: bool,
    password: str | None,
) -> None:
    """Load graph database from archive.

    When the configured location is remote, the archive is transferred
    via SCP and loaded on the remote host.
    """
    from imas_codex.graph.profiles import resolve_neo4j
    from imas_codex.settings import get_graph_password

    profile = resolve_neo4j()
    password = password or get_graph_password()

    archive_path = Path(archive)

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            remote_cleanup_archive,
            remote_load_archive,
            scp_to_remote,
        )

        remote_archive = f"/tmp/imas-codex-load-{archive_path.name}"
        click.echo(
            f"Loading archive into [{profile.name}] on {profile.host}: {archive_path}"
        )

        click.echo(f"  Transferring archive to {profile.host}...")
        scp_to_remote(archive_path, remote_archive, profile.host)

        try:
            click.echo("  Loading on remote host...")
            output = remote_load_archive(
                remote_archive,
                profile.name,
                profile.host,
                password=password,
            )
            if "LOAD_COMPLETE" in output:
                click.echo("✓ Load complete (remote)")
            else:
                click.echo(f"Warning: Unexpected output: {output}", err=True)
        finally:
            remote_cleanup_archive(remote_archive, profile.host)

        # Update local manifest (extract version info from archive)
        manifest = {"pushed": False, "loaded_from": str(archive_path)}
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("manifest.json"):
                        f = tar.extractfile(member)
                        if f:
                            archive_manifest = json.loads(f.read())
                            manifest.update(archive_manifest)
                            manifest["pushed"] = False
                            manifest["loaded_from"] = str(archive_path)
                        break
        except Exception:
            pass
        save_local_graph_manifest(manifest)
        return
    # ── End remote dispatch ──────────────────────────────────────────────

    require_apptainer()
    click.echo(f"Loading archive into [{profile.name}]: {archive_path}")

    with Neo4jOperation(
        "graph load",
        require_stopped=True,
        reset_password_on_restart=True,
        password=password,
    ) as op:
        if no_restart:
            op.was_running = False

        backup_existing_data("pre-load", data_dir=profile.data_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            click.echo("  Extracting archive...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(tmp)

            extracted_dirs = list(tmp.iterdir())
            if not extracted_dirs:
                raise click.ClickException("Empty archive")
            archive_dir = extracted_dirs[0]

            manifest_file = archive_dir / "manifest.json"
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
                click.echo(f"  Version: {manifest.get('version')}")
                click.echo(f"  Commit: {manifest.get('git_commit', 'unknown')[:7]}")

            dump_file = archive_dir / "graph.dump"
            if dump_file.exists():
                click.echo("  Loading graph database...")
                dumps_dir = profile.data_dir / "dumps"
                dumps_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(dump_file, dumps_dir / "neo4j.dump")

                cmd = [
                    "apptainer",
                    "exec",
                    "--bind",
                    f"{profile.data_dir}/data:/data",
                    "--bind",
                    f"{dumps_dir}:/dumps",
                    "--writable-tmpfs",
                    str(NEO4J_IMAGE),
                    "neo4j-admin",
                    "database",
                    "load",
                    "neo4j",
                    "--from-path=/dumps",
                    "--overwrite-destination=true",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise click.ClickException(f"Graph load failed: {result.stderr}")

            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
                manifest["pushed"] = False
                manifest["loaded_from"] = str(archive_path)
                save_local_graph_manifest(manifest)

    click.echo("✓ Load complete")


@graph.command("push")
@click.option("--dev", is_flag=True, help="Push as dev-{commit} tag")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option("--dry-run", is_flag=True, help="Show what would be pushed")
@click.option(
    "--facility",
    "-F",
    "facilities",
    multiple=True,
    help="Facility to include (repeatable). Filters the dump.",
)
@click.option(
    "--no-imas",
    is_flag=True,
    help="Exclude IMAS Data Dictionary nodes",
)
@click.option(
    "--imas-only",
    is_flag=True,
    help="Push only IMAS Data Dictionary nodes (no facility data)",
)
@click.option(
    "-m",
    "--message",
    default=None,
    help="Short description to attach to this push (shown by 'graph tags').",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show full error output from neo4j-admin.",
)
def graph_push(
    dev: bool,
    registry: str | None,
    token: str | None,
    dry_run: bool,
    facilities: tuple[str, ...],
    no_imas: bool,
    imas_only: bool,
    message: str | None,
    verbose: bool = False,
) -> None:
    """Push graph archive to GHCR.

    Use --facility/-f (repeatable) to push a filtered per-facility graph.
    Use --imas-only to push only IMAS Data Dictionary nodes.
    Use -m/--message to attach a short description (like a git commit message).
    """
    from imas_codex.cli.graph_progress import GraphProgress, run_oras_with_progress

    git_info = get_git_info()

    if not dev:
        require_clean_git(git_info)

    # Ensure __version__ reflects current git state (hatch-vcs freezes at
    # uv sync time — without this, the GHCR tag embeds a stale commit hash).
    fresh_version = _ensure_fresh_version()

    target_registry = get_registry(git_info, registry)
    version_tag = get_version_tag(git_info, dev, version_override=fresh_version)
    pkg_name = get_package_name(
        list(facilities) or None, no_imas=no_imas, imas_only=imas_only
    )

    click.echo(f"Push target: {target_registry}/{pkg_name}:{version_tag}")
    if git_info["is_fork"]:
        click.echo(f"  Detected fork: {git_info['remote_owner']}")

    if dry_run:
        click.echo("\n[DRY RUN] Would:")
        click.echo("  1. Dump graph (auto stop/start Neo4j)")
        click.echo(f"  2. Push to {target_registry}/{pkg_name}:{version_tag}")
        return

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.profiles import resolve_neo4j
    from imas_codex.graph.remote import is_remote_location

    profile = resolve_neo4j()

    if is_remote_location(profile.host):
        from imas_codex.cli.graph_progress import remote_operation_streaming
        from imas_codex.graph.remote import (
            build_remote_push_script,
            remote_check_imas_codex,
            remote_check_oras,
        )

        if not remote_check_oras(profile.host):
            raise click.ClickException(
                f"oras not found on {profile.host}. "
                "Install with: imas-codex tools install"
            )

        codex_cli_path: str | None = None
        if imas_only:
            codex_cli_path = remote_check_imas_codex(profile.host)
            if not codex_cli_path:
                raise click.ClickException(
                    f"imas-codex CLI not found on {profile.host}. "
                    "Install with: cd ~/Code/imas-codex && uv sync"
                )

        if facilities:
            click.echo(
                "Warning: --facility filtering is not supported for remote push. "
                "The full graph will be pushed.",
                err=True,
            )

        artifact_ref = f"{target_registry}/{pkg_name}:{version_tag}"

        _remote_markers_push = {
            "STOPPING": f"Stopping Neo4j on {profile.host}",
            "DUMPING": "Dumping graph database",
            "RECOVERY": "Recovery cycle (clean start/stop)",
            "EXPORTING": "Exporting IMAS-only graph via imas-codex CLI",
            "FILTERING": "Filtering to IMAS DD nodes only",
            "ARCHIVING": "Creating archive",
            "STARTING": f"Starting Neo4j on {profile.host}",
            "LOGIN": "Authenticating with GHCR",
            "PUSHING": f"Pushing to GHCR ({artifact_ref})",
            "TAGGING": "Tagging as latest",
            "COMPLETE": "Push complete",
        }

        phases = 1  # single streaming operation
        with GraphProgress("push") as gp:
            gp.set_total_phases(phases)
            gp.start_phase(f"Pushing [{profile.name}] from {profile.host}")

            script = build_remote_push_script(
                profile.name,
                artifact_ref,
                version_tag=version_tag,
                git_commit=git_info["commit"],
                message=message,
                token=token,
                is_dev=dev,
                imas_only=imas_only,
                codex_cli_path=codex_cli_path,
            )

            try:
                push_output = remote_operation_streaming(
                    script,
                    profile.host,
                    progress=gp,
                    progress_markers=_remote_markers_push,
                    timeout=900,
                )
            except Exception as e:
                gp.fail_phase(str(e))
                raise click.ClickException(
                    f"Remote push on {profile.host} failed: {e}"
                ) from e

            size_str = None
            for line in push_output.strip().splitlines():
                if line.startswith("SIZE="):
                    size_str = line.split("=", 1)[1].strip()
            gp.complete_phase(size_str)

        # Update local manifest
        manifest = get_local_graph_manifest() or {}
        manifest["pushed"] = True
        manifest["pushed_version"] = version_tag
        manifest["pushed_to"] = artifact_ref
        manifest["pushed_at"] = datetime.now(UTC).isoformat()
        if message:
            manifest["pushed_message"] = message
        save_local_graph_manifest(manifest)

        if dev:
            base = __version__.replace("+", "-")
            rev_str = version_tag.rsplit("-r", 1)[-1]
            _save_dev_revision(base, int(rev_str))

        _dispatch_graph_quality(git_info, version_tag, target_registry)
        return
    # ── End remote dispatch ──────────────────────────────────────────────

    require_oras()

    with tempfile.TemporaryDirectory() as push_tmpdir:
        archive_path = Path(push_tmpdir) / f"{pkg_name}-{version_tag}.tar.gz"

        with GraphProgress("push") as gp:
            gp.set_total_phases(3 if not dev else 2)

            gp.start_phase("Exporting graph database")
            from click.testing import CliRunner

            runner = CliRunner()
            dump_args = ["-o", str(archive_path)]
            for fac in facilities:
                dump_args.extend(["--facility", fac])
            if no_imas:
                dump_args.append("--no-imas")
            if imas_only:
                dump_args.append("--imas-only")
            if verbose:
                dump_args.append("--verbose")
            result = runner.invoke(graph_export, dump_args)
            if result.exit_code != 0:
                if result.exception and not isinstance(result.exception, SystemExit):
                    detail = f"{type(result.exception).__name__}: {result.exception}"
                else:
                    # Extract the error block from click output.
                    # Click formats ClickException as "Error: <message>"
                    # where <message> may be multi-line.  Capture
                    # everything from the last "Error: " to the end.
                    output_lines = result.output.strip().splitlines()
                    error_start = None
                    for i, line in enumerate(output_lines):
                        if line.startswith("Error: "):
                            error_start = i
                    if error_start is not None:
                        error_block = output_lines[error_start:]
                        error_block[0] = error_block[0].removeprefix("Error: ")
                        detail = "\n".join(error_block)
                    else:
                        detail = result.output.strip()
                gp.fail_phase(detail.splitlines()[0])
                raise click.ClickException(detail)
            size_mb = archive_path.stat().st_size / 1024 / 1024
            gp.complete_phase(f"{size_mb:.1f} MB")

            login_to_ghcr(token)

            artifact_ref = f"{target_registry}/{pkg_name}:{version_tag}"
            push_cmd = [
                "oras",
                "push",
                artifact_ref,
                f"{archive_path}:application/gzip",
                "--disable-path-validation",
                "--annotation",
                f"org.opencontainers.image.version={version_tag}",
                "--annotation",
                f"io.imas-codex.git-commit={git_info['commit']}",
            ]
            if message:
                push_cmd.extend(
                    [
                        "--annotation",
                        f"org.opencontainers.image.description={message}",
                    ]
                )

            gp.start_phase(f"Pushing to GHCR ({artifact_ref})")
            run_oras_with_progress(push_cmd, progress=gp)
            gp.complete_phase()

            manifest = get_local_graph_manifest() or {}
            manifest["pushed"] = True
            manifest["pushed_version"] = version_tag
            manifest["pushed_to"] = artifact_ref
            manifest["pushed_at"] = datetime.now(UTC).isoformat()
            if message:
                manifest["pushed_message"] = message
            save_local_graph_manifest(manifest)

            # Save dev revision for auto-increment on next push
            if dev:
                base = __version__.replace("+", "-")
                rev_str = version_tag.rsplit("-r", 1)[-1]
                _save_dev_revision(base, int(rev_str))

            if not dev:
                gp.start_phase("Tagging as latest")
                result = subprocess.run(
                    ["oras", "tag", artifact_ref, "latest"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    gp.complete_phase()
                else:
                    gp.fail_phase(result.stderr.strip())

    # Dispatch graph quality CI
    _dispatch_graph_quality(git_info, version_tag, target_registry)


@graph.command("fetch")
@click.option("-v", "--version", "version", default="latest")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Save archive to this path (default: auto-named in current directory)",
)
@click.option(
    "--facility",
    "-F",
    "facilities",
    multiple=True,
    help="Facility to filter (repeatable). Selects GHCR package name.",
)
@click.option(
    "--no-imas",
    is_flag=True,
    help="Fetch no-imas variant",
)
@click.option(
    "--imas-only",
    is_flag=True,
    help="Fetch IMAS-only variant (DD nodes only)",
)
@click.option(
    "--local",
    is_flag=True,
    help="Also transfer the archive locally (remote graphs only).",
)
def graph_fetch(
    version: str,
    registry: str | None,
    token: str | None,
    output: str | None,
    facilities: tuple[str, ...],
    no_imas: bool,
    imas_only: bool,
    local: bool,
) -> Path:
    """Fetch graph archive from GHCR without loading.

    Downloads the archive to disk but does NOT load it into Neo4j.
    Use 'graph load <archive>' to load it afterwards, or use
    'graph pull' as a convenience for fetch + load.

    When the configured location is remote and ``oras`` is available
    there, the fetch runs directly on the remote host.  Use
    ``--local`` to also transfer the archive back via SCP.

    When no --version is specified, fetches 'latest'. If 'latest' doesn't
    exist, falls back to the most recent tag in the registry.
    """
    from imas_codex.cli.graph_progress import (
        GraphProgress,
        remote_operation_streaming,
        run_oras_with_progress,
    )
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    pkg_name = get_package_name(
        list(facilities) or None, no_imas=no_imas, imas_only=imas_only
    )

    # Resolve version: if "latest" doesn't exist, find most recent tag
    resolved_version = version
    if version == "latest":
        resolved_version = _resolve_latest_tag(target_registry, token, pkg_name)

    artifact_ref = f"{target_registry}/{pkg_name}:{resolved_version}"

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            build_remote_fetch_script,
            remote_check_oras,
            scp_from_remote,
        )

        if remote_check_oras(profile.host):
            with GraphProgress("fetch") as gp:
                gp.set_total_phases(2 if (local or output) else 1)

                # Build output name for remote file
                ref_parts = artifact_ref.rsplit("/", 1)[-1]
                output_name = ref_parts.replace(":", "-") + ".tar.gz"

                gp.start_phase(f"Fetching on {profile.host} via ORAS")
                script = build_remote_fetch_script(
                    artifact_ref, output_name, token=token
                )
                fetch_output = remote_operation_streaming(
                    script,
                    profile.host,
                    progress=gp,
                    progress_markers={
                        "LOGIN": f"Authenticating on {profile.host}",
                        "PULLING": f"Downloading from GHCR on {profile.host}",
                        "MOVING": "Saving archive",
                        "DONE": "Fetch complete",
                    },
                    timeout=300,
                )

                # Extract archive path and size from output
                remote_archive = None
                size_str = None
                for line in fetch_output.strip().splitlines():
                    if line.startswith("ARCHIVE_PATH="):
                        remote_archive = line.split("=", 1)[1].strip()
                    elif line.startswith("SIZE="):
                        size_str = line.split("=", 1)[1].strip()
                if not remote_archive:
                    gp.fail_phase("No archive path in output")
                    raise click.ClickException(
                        f"Could not find archive path in output:\n{fetch_output}"
                    )
                gp.complete_phase(size_str)

                if local or output:
                    from imas_codex.graph.dirs import ensure_exports_dir

                    if output:
                        dest = Path(output)
                    else:
                        exports = ensure_exports_dir()
                        dest = exports / f"{pkg_name}-{resolved_version}.tar.gz"

                    gp.start_phase(f"Transferring from {profile.host}")
                    scp_from_remote(remote_archive, dest, profile.host)
                    size_mb = dest.stat().st_size / 1024 / 1024
                    gp.complete_phase(f"{size_mb:.1f} MB")
                    gp.print(f"  Load locally: imas-codex graph load {dest}")
                    return dest

                gp.print(f"  Load remotely: imas-codex graph load {remote_archive}")
                return Path(remote_archive)
        else:
            click.echo(f"oras not on {profile.host}, fetching locally...")
    # ── End remote dispatch ──────────────────────────────────────────────

    require_oras()

    with GraphProgress("fetch") as gp:
        gp.set_total_phases(1)

        gp.start_phase("Fetching from GHCR")
        login_to_ghcr(token)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            run_oras_with_progress(
                ["oras", "pull", artifact_ref, "-o", str(tmp)],
                progress=gp,
                phase_description=f"Fetching {artifact_ref}",
            )

            archives = list(tmp.glob("*.tar.gz"))
            if not archives:
                gp.fail_phase("No archive found")
                raise click.ClickException("No archive found in fetched artifact")

            src_archive = archives[0]
            if output:
                dest = Path(output)
            else:
                from imas_codex.graph.dirs import ensure_exports_dir

                exports = ensure_exports_dir()
                dest = exports / f"{pkg_name}-{resolved_version}.tar.gz"

            shutil.move(str(src_archive), str(dest))

        size_mb = dest.stat().st_size / 1024 / 1024
        gp.complete_phase(f"{size_mb:.1f} MB")
        gp.print(f"  Load with: imas-codex graph load {dest}")
    return dest


@graph.command("pull")
@click.option("-v", "--version", "version", default="latest")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option("--force", is_flag=True, help="Overwrite existing graph without checks")
@click.option("--no-backup", is_flag=True, help="Skip backup marker")
@click.option(
    "--facility",
    "-F",
    "facilities",
    multiple=True,
    help="Facility to filter (repeatable). Selects GHCR package name.",
)
@click.option(
    "--no-imas",
    is_flag=True,
    help="Pull no-imas variant",
)
@click.option(
    "--imas-only",
    is_flag=True,
    help="Pull IMAS-only variant (DD nodes only)",
)
def graph_pull(
    version: str,
    registry: str | None,
    token: str | None,
    force: bool,
    no_backup: bool,
    facilities: tuple[str, ...],
    no_imas: bool,
    imas_only: bool,
) -> None:
    """Pull graph from GHCR and load it (convenience for fetch + load).

    This is equivalent to running 'graph fetch' followed by 'graph load'.
    Use 'graph fetch' if you only want to download without loading.

    When the configured location is remote:
    - If ``oras`` is available on the remote host, the archive is fetched
      directly there (no SCP transfer needed).
    - Otherwise, the archive is fetched locally and transferred via SCP.

    When no --version is specified, pulls 'latest'. If 'latest' doesn't
    exist, falls back to the most recent tag in the registry.

    Use --facility/-f (repeatable) to pull a per-facility graph.
    """
    from imas_codex.cli.graph_progress import (
        GraphProgress,
        remote_operation_streaming,
        run_oras_with_progress,
    )
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    pkg_name = get_package_name(
        list(facilities) or None, no_imas=no_imas, imas_only=imas_only
    )

    # Resolve version: if "latest" doesn't exist, find most recent tag
    resolved_version = version
    if version == "latest":
        resolved_version = _resolve_latest_tag(target_registry, token, pkg_name)

    artifact_ref = f"{target_registry}/{pkg_name}:{resolved_version}"

    # ── Pull compatibility check ─────────────────────────────────────────
    if not force:
        try:
            from imas_codex.graph.client import GraphClient
            from imas_codex.graph.meta import check_pull_compatibility, get_graph_meta

            gc = GraphClient.from_profile()
            meta = get_graph_meta(gc)
            gc.close()
            if meta:
                pull_errors = check_pull_compatibility(
                    meta,
                    imas_only=imas_only,
                    no_imas=no_imas,
                    facilities=list(facilities) or None,
                )
                if pull_errors:
                    msg = "\n".join(pull_errors)
                    raise click.ClickException(f"{msg}\nUse --force to override.")
        except click.ClickException:
            raise
        except Exception:
            pass  # Can't reach Neo4j — skip check

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            REMOTE_EXPORTS,
            build_remote_fetch_script,
            build_remote_load_script,
            remote_check_oras,
            remote_cleanup_archive,
            scp_to_remote,
        )
        from imas_codex.settings import get_graph_password

        password = get_graph_password()

        _remote_markers_fetch = {
            "LOGIN": f"Authenticating on {profile.host}",
            "PULLING": f"Downloading from GHCR on {profile.host}",
            "MOVING": "Saving archive",
            "DONE": "Fetch complete",
        }
        _remote_markers_load = {
            "STOPPING": f"Stopping Neo4j on {profile.host}",
            "EXTRACTING": "Extracting archive",
            "LOADING_DUMP": "Loading graph dump into Neo4j",
            "PASSWORD": "Resetting password",
            "STARTING": f"Starting Neo4j on {profile.host}",
            "COMPLETE": "Load complete",
        }

        with GraphProgress("pull") as gp:
            click.echo(f"Pulling: {artifact_ref}")

            if remote_check_oras(profile.host):
                gp.set_total_phases(3)

                # Build output name
                ref_parts = artifact_ref.rsplit("/", 1)[-1]
                output_name = ref_parts.replace(":", "-") + ".tar.gz"

                gp.start_phase(f"Fetching on {profile.host} via ORAS")
                script = build_remote_fetch_script(
                    artifact_ref, output_name, token=token
                )
                fetch_output = remote_operation_streaming(
                    script,
                    profile.host,
                    progress=gp,
                    progress_markers=_remote_markers_fetch,
                    timeout=300,
                )
                remote_archive = None
                for line in fetch_output.strip().splitlines():
                    if line.startswith("ARCHIVE_PATH="):
                        remote_archive = line.split("=", 1)[1].strip()
                if not remote_archive:
                    gp.fail_phase("No archive path in output")
                    raise click.ClickException(
                        f"Could not find archive path:\n{fetch_output}"
                    )
                gp.complete_phase()
            else:
                gp.set_total_phases(4)

                gp.start_phase("Fetching from GHCR locally")
                require_oras()
                login_to_ghcr(token)

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    run_oras_with_progress(
                        ["oras", "pull", artifact_ref, "-o", str(tmp)],
                        progress=gp,
                    )

                    archives = list(tmp.glob("*.tar.gz"))
                    if not archives:
                        gp.fail_phase("No archive found")
                        raise click.ClickException("No archive found")
                    gp.complete_phase()

                    local_archive = archives[0]
                    remote_archive = f"{REMOTE_EXPORTS}/{local_archive.name}"

                    gp.start_phase(f"Transferring to {profile.host}")
                    scp_to_remote(local_archive, remote_archive, profile.host)
                    gp.complete_phase()

            # Load on remote (streaming)
            gp.start_phase(f"Loading on {profile.host}")
            load_script = build_remote_load_script(
                remote_archive, profile.name, password
            )
            try:
                load_output = remote_operation_streaming(
                    load_script,
                    profile.host,
                    progress=gp,
                    progress_markers=_remote_markers_load,
                    timeout=600,
                )
            finally:
                remote_cleanup_archive(remote_archive, profile.host)

            if "LOAD_COMPLETE" not in load_output:
                gp.fail_phase("Unexpected output")
                click.echo(f"Warning: Unexpected output: {load_output}", err=True)
            else:
                gp.complete_phase()

            # Update local manifest
            manifest = {
                "version": resolved_version,
                "pulled_from": artifact_ref,
                "pulled_version": resolved_version,
                "pushed": True,
                "pushed_version": resolved_version,
            }
            save_local_graph_manifest(manifest)

            gp.print("[green]✓[/] Graph pull complete (remote)")
        return
    # ── End remote dispatch ──────────────────────────────────────────────

    require_oras()

    if check_graph_exists(data_dir=profile.data_dir) and not force:
        manifest = get_local_graph_manifest()
        if manifest is None:
            raise click.ClickException(
                "Local graph exists but has no manifest (unknown origin).\n"
                "Either:\n"
                "  1. Push current graph first: imas-codex graph push --dev\n"
                "  2. Use --force to overwrite (data will be lost)"
            )
        elif not manifest.get("pushed"):
            raise click.ClickException(
                f"Local graph (loaded {manifest.get('loaded_at', 'unknown')}) "
                "has not been pushed.\n"
                "Either:\n"
                "  1. Push current graph: imas-codex graph push --dev\n"
                "  2. Use --force to overwrite (data will be lost)"
            )
        else:
            pushed_version = manifest.get("pushed_version", "unknown")
            click.echo(f"Local graph was pushed as: {pushed_version}")

    click.echo(f"Pulling: {artifact_ref}")

    if not no_backup:
        backup_existing_data("pre-pull", data_dir=profile.data_dir)

    with GraphProgress("pull") as gp:
        gp.set_total_phases(2)

        gp.start_phase("Fetching from GHCR")
        login_to_ghcr(token)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            run_oras_with_progress(
                ["oras", "pull", artifact_ref, "-o", str(tmp)],
                progress=gp,
            )

            archives = list(tmp.glob("*.tar.gz"))
            if not archives:
                gp.fail_phase("No archive found")
                raise click.ClickException("No archive found")
            gp.complete_phase()

            gp.start_phase("Loading into Neo4j")
            from click.testing import CliRunner

            runner = CliRunner()
            load_args = [str(archives[0]), "--force"]
            result = runner.invoke(graph_load, load_args)
            if result.exit_code != 0:
                gp.fail_phase(result.output.strip())
                raise click.ClickException(f"Load failed: {result.output}")
            gp.complete_phase()

            with tarfile.open(archives[0], "r:gz") as tar:
                tar.extractall(tmp / "extracted")
            extracted_dirs = list((tmp / "extracted").iterdir())
            if extracted_dirs:
                manifest_file = extracted_dirs[0] / "manifest.json"
                if manifest_file.exists():
                    manifest = json.loads(manifest_file.read_text())
                    manifest["pulled_from"] = artifact_ref
                    manifest["pulled_version"] = resolved_version
                    manifest["pushed"] = True
                    manifest["pushed_version"] = resolved_version
                    save_local_graph_manifest(manifest)

        gp.print("[green]✓[/] Graph pull complete")


# ============================================================================
# Graph List Command
# ============================================================================


@graph.command("list")
def graph_list() -> None:
    """List graph instances.

    Scans the .neo4j/ store directory for graph instances and shows
    their name and whether they are active.  Works on both local and
    remote (SSH) graph locations.

    \b
    Examples:
      imas-codex graph list
    """
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j(auto_tunnel=False)

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import remote_list_graphs

        output = remote_list_graphs(profile.host)

        if "NO_STORE" in output:
            click.echo(f"No graph store on {profile.host}.")
            click.echo("Create one: imas-codex graph init <name> -f <facility>")
            return

        click.echo(f"Graphs on {profile.host}:\n")
        count = 0
        for line in output.strip().splitlines():
            line = line.strip()
            if line.startswith("[stderr]"):
                continue
            if line.startswith("ACTIVE:"):
                click.echo(f"→ {line.removeprefix('ACTIVE:')}")
                count += 1
            elif line.startswith("GRAPH:"):
                click.echo(f"  {line.removeprefix('GRAPH:')}")
                count += 1
        click.echo(f"\n{count} graph(s)")
        return
    # ── End remote dispatch ──────────────────────────────────────────────

    from imas_codex.graph.dirs import list_local_graphs

    graphs = list_local_graphs()
    if not graphs:
        click.echo("No local graphs found.")
        click.echo("Create one: imas-codex graph init <name> -f <facility> ...")
        return

    click.echo("Local graphs:\n")
    for g in graphs:
        marker = "→ " if g.active else "  "
        click.echo(f"{marker}{g.name}")
        for warn in g.warnings:
            click.echo(f"    ⚠ {warn}")

    click.echo(f"\n{len(graphs)} graph(s)")


# ============================================================================
# Graph Switch Command
# ============================================================================


@graph.command("switch")
@click.argument("name")
def graph_switch(name: str) -> None:
    """Switch the active graph.

    Stops Neo4j if running, repoints the neo4j/ symlink to the
    target graph directory, and restarts Neo4j.

    Works on both local and remote graph locations.

    \b
    Examples:
      imas-codex graph switch codex
      imas-codex graph switch dev
    """
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j(auto_tunnel=False)

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            remote_create_graph_dir,
            remote_is_neo4j_running,
            remote_list_graphs,
            remote_service_action,
            remote_switch_active_graph,
            resolve_remote_service_name,
        )

        service = resolve_remote_service_name(profile.name, profile.host)

        if remote_is_neo4j_running(profile.http_port, profile.host):
            click.echo(f"Stopping Neo4j [{profile.name}] on {profile.host}...")
            remote_service_action("stop", service, profile.host, timeout=60)

            import time

            for _ in range(15):
                if not remote_is_neo4j_running(profile.http_port, profile.host):
                    break
                time.sleep(1)

        # Auto-init if graph doesn't exist
        listing = remote_list_graphs(profile.host)
        graph_names = []
        for line in listing.strip().splitlines():
            line = line.strip()
            if line.startswith("[stderr]"):
                continue
            if line.startswith("ACTIVE:"):
                graph_names.append(line.removeprefix("ACTIVE:"))
            elif line.startswith("GRAPH:"):
                graph_names.append(line.removeprefix("GRAPH:"))

        if name not in graph_names:
            click.echo(f"Graph '{name}' not found — creating...")
            remote_create_graph_dir(
                name,
                profile.host,
                bolt_port=profile.bolt_port,
                http_port=profile.http_port,
            )

        try:
            remote_switch_active_graph(name, profile.host)
        except Exception as e:
            raise click.ClickException(str(e)) from e

        click.echo(f"✓ Switched to '{name}' on {profile.host}")

        # Restart Neo4j
        click.echo("Restarting Neo4j...")
        service = resolve_remote_service_name(name, profile.host)
        remote_service_action("start", service, profile.host, timeout=60)

        import time

        for _ in range(30):
            if remote_is_neo4j_running(profile.http_port, profile.host):
                click.echo(f"✓ Neo4j [{name}] ready on {profile.host}")
                break
            time.sleep(1)
        else:
            click.echo("Warning: Neo4j may still be starting")
        return
    # ── End remote dispatch ──────────────────────────────────────────────

    from imas_codex.graph.dirs import (
        create_graph_dir,
        find_graph,
        get_active_graph,
        switch_active_graph,
    )

    try:
        target = find_graph(name)
    except LookupError:
        click.echo(f"Graph '{name}' not found — creating...")
        create_graph_dir(name)
        target = find_graph(name)

    # Check if already active
    active = get_active_graph()
    if active and active.name == target.name:
        click.echo(f"Graph '{target.name}' is already active.")
        return

    profile = resolve_neo4j(auto_tunnel=False)
    was_running = is_neo4j_running(profile.http_port)

    if was_running:
        click.echo(f"Stopping Neo4j [{profile.name}]...")
        _stop_neo4j_for_switch(profile)

    try:
        switch_active_graph(name)
    except (FileExistsError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"✓ Switched to '{target.name}'")

    if was_running:
        click.echo("Restarting Neo4j...")
        # Re-resolve profile after switch (name may have changed)
        new_profile = resolve_neo4j(auto_tunnel=False)
        _start_neo4j_after_switch(new_profile)


def _stop_neo4j_for_switch(profile: Neo4jProfile) -> None:
    """Stop Neo4j for a graph switch operation."""
    import signal

    data_path = profile.data_dir
    pid_file = data_path / "neo4j.pid"

    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            pid_file.unlink()
        except ProcessLookupError:
            pid_file.unlink()
    else:
        subprocess.run(["pkill", "-f", "neo4j.*console"], capture_output=True)

    # Wait for shutdown
    import time

    for _ in range(15):
        if not is_neo4j_running(profile.http_port):
            return
        time.sleep(1)
    click.echo("Warning: Neo4j may still be shutting down", err=True)


def _start_neo4j_after_switch(profile: Neo4jProfile) -> None:
    """Start Neo4j after a graph switch operation."""
    data_path = profile.data_dir

    if not NEO4J_IMAGE.exists():
        click.echo("Warning: Neo4j image not found, cannot auto-start", err=True)
        return

    for subdir in ["data", "logs", "conf", "import"]:
        (data_path / subdir).mkdir(parents=True, exist_ok=True)

    secure_data_directory(data_path)

    # Ensure neo4j.conf exists with proper memory/recovery settings
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
        str(NEO4J_IMAGE),
        "neo4j",
        "console",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_file = data_path / "neo4j.pid"
    pid_file.write_text(str(proc.pid))

    import time

    for _ in range(30):
        if is_neo4j_running(profile.http_port):
            click.echo(
                f"✓ Neo4j [{profile.name}] ready "
                f"at http://localhost:{profile.http_port}"
            )
            return
        time.sleep(1)

    click.echo("Warning: Neo4j may still be starting", err=True)


# ============================================================================
# Graph Init Command
# ============================================================================


@graph.command("init")
@click.argument("name")
@click.option(
    "--facility",
    "-F",
    "facilities",
    multiple=True,
    help="Facility ID to include (repeatable). Omit for DD-only graphs.",
)
@click.option("--force", is_flag=True, help="Allow using an existing directory")
@click.option(
    "--no-imas",
    is_flag=True,
    help="Mark graph as not containing IMAS DD data",
)
def graph_init(
    name: str, facilities: tuple[str, ...], force: bool, no_imas: bool
) -> None:
    """Initialize a new graph instance.

    Creates a name-based directory in .neo4j/<NAME>/, points the neo4j/
    symlink to it, starts Neo4j, and initializes the (:GraphMeta) node.

    Works on both local and remote graph locations.  When the configured
    location is remote, directories are created via SSH and the service
    is started via systemctl.

    \b
    Examples:
      imas-codex graph init codex -F iter -F tcv -F jt-60sa
      imas-codex graph init imas              # DD-only, no facilities
      imas-codex graph init dev -F tcv --no-imas
    """
    from imas_codex.graph.profiles import resolve_neo4j

    facility_list = sorted(set(facilities))
    include_imas = not no_imas

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    profile = resolve_neo4j(auto_tunnel=False)

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            remote_create_graph_dir,
            remote_is_neo4j_running,
            remote_service_action,
            remote_set_initial_password,
            remote_switch_active_graph,
            resolve_remote_service_name,
        )

        click.echo(f"Initializing graph on {profile.host}...")
        try:
            remote_create_graph_dir(
                name,
                profile.host,
                force=force,
                bolt_port=profile.bolt_port,
                http_port=profile.http_port,
            )
        except Exception as e:
            raise click.ClickException(str(e)) from e

        # Stop Neo4j before switching (must restart on new data)
        service = resolve_remote_service_name(name, profile.host)
        if remote_is_neo4j_running(profile.http_port, profile.host):
            click.echo("  Stopping Neo4j for graph switch...")
            remote_service_action("stop", service, profile.host, timeout=60)
            import time

            time.sleep(3)

        try:
            remote_switch_active_graph(name, profile.host)
        except Exception as e:
            raise click.ClickException(str(e)) from e

        click.echo(f"  Name: {name}")
        click.echo(f"  Facilities: {', '.join(facility_list)}")
        click.echo(f"  Host: {profile.host}")

        # Set initial password before first start (reads from .env)
        try:
            remote_set_initial_password(profile.host)
        except Exception:
            pass  # OK if database already initialized

        # Start Neo4j on the new graph
        click.echo("\nStarting Neo4j...")
        remote_service_action("start", service, profile.host, timeout=60)

        import time

        for _ in range(30):
            if remote_is_neo4j_running(profile.http_port, profile.host):
                break
            time.sleep(1)

        # Re-resolve with auto-tunnel to get the bolt URI
        profile = resolve_neo4j(auto_tunnel=True)

        # Init GraphMeta via Bolt (through tunnel)
        try:
            from imas_codex.graph.client import GraphClient
            from imas_codex.graph.meta import init_graph_meta

            gc = GraphClient.from_profile()
            init_graph_meta(gc, name, facility_list, imas=include_imas)
            gc.close()
            click.echo("\n✓ GraphMeta node initialized")
        except Exception as e:
            click.echo(
                f"\nWarning: Cannot reach Neo4j via tunnel: {e}\n"
                "Ensure tunnel is active and run 'graph init' again.",
                err=True,
            )
        return
    # ── End remote dispatch ──────────────────────────────────────────────

    from imas_codex.graph.dirs import (
        create_graph_dir,
        switch_active_graph,
    )

    # Create new graph directory
    profile = resolve_neo4j(auto_tunnel=False)
    try:
        info = create_graph_dir(
            name,
            force=force,
            bolt_port=profile.bolt_port,
            http_port=profile.http_port,
        )
    except FileExistsError as e:
        raise click.ClickException(str(e)) from e

    # Point symlink to the new directory
    try:
        switch_active_graph(name)
    except FileExistsError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"  Name: {name}")
    click.echo(f"  Facilities: {', '.join(facility_list)}")
    click.echo(f"  Path: {info.path}")

    # Start Neo4j and create GraphMeta node
    if not is_neo4j_running(profile.http_port):
        click.echo("\nStarting Neo4j...")
        _start_neo4j_after_switch(profile)

    if is_neo4j_running(profile.http_port):
        from imas_codex.graph.client import GraphClient
        from imas_codex.graph.meta import init_graph_meta

        gc = GraphClient.from_profile()
        init_graph_meta(gc, name, facility_list, imas=include_imas)
        gc.close()
        click.echo("\n✓ GraphMeta node initialized")
    else:
        click.echo(
            "\nWarning: Neo4j not running — GraphMeta node not created.\n"
            "Start Neo4j and run 'graph init' again to create the GraphMeta node."
        )


# ============================================================================
# Graph Facility Subcommand Group
# ============================================================================


@graph.group("facility")
def graph_facility_group() -> None:
    """Manage facilities in the graph identity.

    \b
      imas-codex graph facility list          Show facilities
      imas-codex graph facility add <fac>     Add a facility
      imas-codex graph facility remove <fac>  Remove a facility
    """
    pass


@graph_facility_group.command("list")
def facility_list() -> None:
    """List facilities in the graph identity."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.graph.meta import get_graph_meta

    gc = GraphClient.from_profile()
    meta = get_graph_meta(gc)
    gc.close()

    if meta is None:
        click.echo("Graph identity not initialized.")
        click.echo("Run: imas-codex graph init <name> -f <facility>")
        return

    facilities = meta.get("facilities") or []
    click.echo(f"Graph: {meta.get('name', '?')}")
    if facilities:
        for f in sorted(facilities):
            click.echo(f"  - {f}")
    else:
        click.echo("  (no facilities)")


@graph_facility_group.command("add")
@click.argument("facility_id")
def facility_add(facility_id: str) -> None:
    """Add a facility to the graph identity."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.graph.meta import add_facility_to_meta, get_graph_meta

    gc = GraphClient.from_profile()

    meta = get_graph_meta(gc)
    if meta is None:
        gc.close()
        raise click.ClickException(
            "Graph identity not initialized.\n"
            "Run: imas-codex graph init <name> -f <facility>"
        )

    add_facility_to_meta(gc, facility_id)

    meta = get_graph_meta(gc)
    gc.close()

    facilities = meta.get("facilities") or [] if meta else []
    click.echo(
        f"✓ Added '{facility_id}' to graph '{meta.get('name', '?') if meta else '?'}'"
    )
    click.echo(f"  Facilities: {', '.join(facilities)}")


@graph_facility_group.command("remove")
@click.argument("facility_id")
@click.option("--force", is_flag=True, help="Skip confirmation")
def facility_remove(facility_id: str, force: bool) -> None:
    """Remove a facility from the graph identity."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.graph.meta import get_graph_meta, remove_facility_from_meta

    gc = GraphClient.from_profile()

    meta = get_graph_meta(gc)
    if meta is None:
        gc.close()
        raise click.ClickException("Graph identity not initialized.")

    if not force:
        click.echo(
            f"WARNING: This will delete the '{facility_id}' Facility node, "
            f"detach all its relationships, and remove any orphaned nodes "
            f"that were exclusively linked to this facility."
        )
        if not click.confirm(
            f"Remove '{facility_id}' from graph '{meta.get('name', '?')}'?"
        ):
            gc.close()
            click.echo("Aborted.")
            return

    remove_facility_from_meta(gc, facility_id)

    meta = get_graph_meta(gc)
    gc.close()

    facilities = meta.get("facilities") or [] if meta else []
    click.echo(f"✓ Removed '{facility_id}'")
    click.echo(f"  Facilities: {', '.join(facilities)}")


# ============================================================================
# Graph Lifecycle Commands
# ============================================================================


@graph.command("clear")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def graph_clear(force: bool) -> None:
    """Clear all data from the graph database.

    Requires Neo4j to be running.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()

    if not is_neo4j_running(profile.http_port):
        raise click.ClickException(
            f"Neo4j [{profile.name}] is not running on port {profile.http_port}.\n"
            f"Start it: imas-codex graph start"
        )

    # Show current stats
    try:
        gc = GraphClient.from_profile()
        stats = gc.get_stats()
        gc.close()
        click.echo(
            f"Graph [{profile.name}] has {stats['nodes']} nodes "
            f"and {stats['relationships']} relationships."
        )
    except Exception as e:
        click.echo(f"Warning: Could not get stats: {e}", err=True)
        stats = {"nodes": "?", "relationships": "?"}

    if not force:
        if not click.confirm("Delete ALL data? This cannot be undone."):
            click.echo("Aborted.")
            return

    click.echo(f"Clearing graph [{profile.name}]...")
    try:
        gc = GraphClient.from_profile()
        deleted = gc.drop_all()
        gc.close()
        click.echo(f"✓ Cleared {deleted} nodes from [{profile.name}]")
    except Exception as e:
        raise click.ClickException(f"Clear failed: {e}") from e


# ============================================================================
# Registry Tag Commands
# ============================================================================


@graph.command("tags")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option(
    "--facility",
    "-F",
    default=None,
    help="List tags for a facility-specific graph package.",
)
def graph_tags(registry: str | None, facility: str | None) -> None:
    """List available graph versions in GHCR."""
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    pkg_name = f"imas-codex-graph-{facility}" if facility else "imas-codex-graph"

    tags = _list_registry_tags(target_registry, pkg_name=pkg_name)
    if not tags:
        click.echo(f"No tags found for {target_registry}/{pkg_name}")
        return

    # Fetch messages for each tag from OCI annotations
    tag_messages = _fetch_tag_messages(target_registry, tags, pkg_name=pkg_name)

    click.echo(f"Tags in {target_registry}/{pkg_name}:")
    for tag in sorted(tags):
        msg = tag_messages.get(tag)
        if msg:
            # Clip long messages to keep output tidy
            display_msg = msg if len(msg) <= 72 else msg[:69] + "..."
            click.echo(f"  {tag}  — {display_msg}")
        else:
            click.echo(f"  {tag}")
    click.echo(f"\n{len(tags)} tag(s) total")


@graph.command("prune")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option(
    "--facility",
    "-F",
    default=None,
    help="Prune tags for a facility-specific graph package.",
)
@click.option("--keep", default=5, help="Number of recent tags to keep.")
@click.option("--dev-only", is_flag=True, help="Only prune dev tags.")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted.")
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
def graph_prune(
    registry: str | None,
    facility: str | None,
    keep: int,
    dev_only: bool,
    dry_run: bool,
    force: bool,
) -> None:
    """Prune old graph versions from GHCR."""
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    pkg_name = f"imas-codex-graph-{facility}" if facility else "imas-codex-graph"

    tags = _list_registry_tags(target_registry, pkg_name=pkg_name)
    if not tags:
        click.echo(f"No tags found for {target_registry}/{pkg_name}")
        return

    # Separate release and dev tags
    dev_tags = [t for t in tags if "dev" in t or "-r" in t]
    release_tags = [t for t in tags if t not in dev_tags and t != "latest"]

    if dev_only:
        candidates = dev_tags
    else:
        candidates = dev_tags + release_tags

    # Sort candidates: dev tags by revision descending, release by semver
    def _sort_key(tag: str) -> tuple[int, int]:
        is_dev = 0 if ("dev" in tag or "-r" in tag) else 1
        rev = 0
        if "-r" in tag:
            try:
                rev = int(tag.rsplit("-r", 1)[-1])
            except ValueError:
                pass
        return (is_dev, -rev)

    candidates.sort(key=_sort_key)

    # Keep the most recent N, delete the rest
    to_keep = set(candidates[:keep])
    to_keep.add("latest")  # Never prune 'latest'
    to_delete = [t for t in candidates if t not in to_keep]

    if not to_delete:
        click.echo(f"Nothing to prune (keeping {keep} most recent)")
        return

    click.echo(
        f"Will delete {len(to_delete)} tag(s) from {target_registry}/{pkg_name}:"
    )
    for tag in to_delete:
        click.echo(f"  {tag}")

    if dry_run:
        click.echo("\n(dry-run — no changes made)")
        return

    if not force:
        if not click.confirm(f"Delete {len(to_delete)} tag(s)?"):
            click.echo("Aborted.")
            return

    deleted = 0
    for tag in to_delete:
        if _delete_tag(target_registry, tag, pkg_name=pkg_name):
            click.echo(f"  ✓ Deleted {tag}")
            deleted += 1
        else:
            click.echo(f"  ✗ Failed to delete {tag}")

    click.echo(f"\n✓ Pruned {deleted}/{len(to_delete)} tags")
