"""Graph CLI data commands — export, load, init, switch, list, clear, secure, facility."""

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
from imas_codex.graph.ghcr import (
    get_git_info,
    get_package_name,
    save_local_graph_manifest,
)
from imas_codex.graph.neo4j_ops import (
    Neo4jOperation,
    backup_existing_data,
    is_neo4j_running,
    neo4j_image,
    run_neo4j_dump as _run_neo4j_dump,
    secure_data_directory,
)
from imas_codex.graph.temp_neo4j import (
    create_dd_only_dump as _create_dd_only_dump,
    create_facility_dump as _create_facility_dump,
)

if TYPE_CHECKING:
    from imas_codex.graph.profiles import Neo4jProfile

NEO4J_IMAGE = neo4j_image()


def _resolve_scheduler(profile: Neo4jProfile) -> str:
    """Resolve the job scheduler for a Neo4j profile's location."""
    try:
        from imas_codex.remote.locations import resolve_location

        return resolve_location(profile.location).scheduler
    except Exception:
        return "none"


# ============================================================================
# Graph Secure Command
# ============================================================================


@click.command()
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


# ============================================================================
# Archive & Distribution Commands
# ============================================================================


@click.command()
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
    "--without-dd",
    is_flag=True,
    help="Exclude IMAS Data Dictionary nodes from export",
)
@click.option(
    "--dd-only",
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
@click.option(
    "--source-dump",
    type=click.Path(exists=True),
    default=None,
    help="Use pre-existing dump file instead of dumping (avoids Neo4j stop/start).",
)
@click.option(
    "--version-label",
    default=None,
    help="Override version label for archive directory name.",
)
def graph_export(
    output: str | None,
    no_restart: bool,
    facilities: tuple[str, ...],
    without_dd: bool,
    dd_only: bool,
    local: bool,
    verbose: bool = False,
    source_dump: str | None = None,
    version_label: str | None = None,
) -> None:
    """Export graph database to archive.

    When the configured location is remote, the dump is performed on
    the remote host and the archive stays in the remote exports
    directory.  Use ``--local`` to also transfer it back via SCP.
    """
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()

    git_info = get_git_info()
    if not version_label:
        version_label = git_info["tag"] or f"dev-{git_info['commit_short']}"
    pkg_name = get_package_name(
        facilities=list(facilities), without_dd=without_dd, dd_only=dd_only
    )

    if output:
        output_path = Path(output)
    else:
        from imas_codex.graph.dirs import ensure_exports_dir

        exports = ensure_exports_dir()
        output_path = exports / f"{pkg_name}-{version_label}.tar.gz"

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host) and not source_dump:
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
            script = build_remote_export_script(
                profile.name,
                scheduler=_resolve_scheduler(profile),
            )
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

    from imas_codex.graph.ghcr import require_apptainer

    require_apptainer()

    def _build_archive(archive_dir: Path) -> None:
        """Build the archive contents: dump with optional filtering."""
        if source_dump:
            click.echo(f"  Using cached dump: {source_dump}")
            shutil.copy(source_dump, str(archive_dir / "graph.dump"))
        else:
            # Create dump from live graph
            click.echo("  Dumping graph database...")
            dumps_dir = profile.data_dir / "dumps"
            dumps_dir.mkdir(parents=True, exist_ok=True)

            _run_neo4j_dump(profile, dumps_dir, verbose=verbose)

            dump_file = dumps_dir / "neo4j.dump"
            if dump_file.exists():
                shutil.move(str(dump_file), str(archive_dir / "graph.dump"))
            else:
                raise click.ClickException("Graph dump file not created")

        size_mb = (archive_dir / "graph.dump").stat().st_size / 1024 / 1024
        click.echo(f"    Graph: {size_mb:.1f} MB")

        # Apply filtering to the dump
        if facilities:
            for fac in facilities:
                click.echo(f"  Filtering dump for facility: {fac}")
                _create_facility_dump(
                    archive_dir / "graph.dump",
                    fac,
                    archive_dir / "graph.dump",
                )
        if dd_only:
            click.echo("  Filtering dump to DD-only...")
            _create_dd_only_dump(
                archive_dir / "graph.dump",
                archive_dir / "graph.dump",
            )

        manifest = {
            "version": __version__,
            "git_commit": git_info["commit"],
            "git_tag": git_info["tag"],
            "timestamp": datetime.now(UTC).isoformat(),
            "format": "csv" if (archive_dir / "csv").is_dir() else "dump",
        }
        (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    use_csv = not source_dump and (dd_only or facilities)

    if source_dump or use_csv:
        # No Neo4j stop/start needed — either cached dump or live CSV export
        click.echo(f"Creating archive [{profile.name}]: {output_path}")

        # Use GPFS-visible temp dir when SLURM dispatch is possible,
        # otherwise /run/user tmpfs is not visible from compute nodes.
        tmp_base = str(profile.data_dir) if shutil.which("srun") else None
        with tempfile.TemporaryDirectory(dir=tmp_base) as tmpdir:
            tmp = Path(tmpdir)
            archive_dir = tmp / f"{pkg_name}-{version_label}"
            archive_dir.mkdir()

            _build_archive(archive_dir)

            click.echo("  Creating archive...")
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(archive_dir, arcname=archive_dir.name)
    else:
        with Neo4jOperation("graph dump", require_stopped=True) as op:
            if no_restart:
                op.was_running = False

            click.echo(f"Creating archive [{profile.name}]: {output_path}")

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                archive_dir = tmp / f"{pkg_name}-{version_label}"
                archive_dir.mkdir()

                _build_archive(archive_dir)

                click.echo("  Creating archive...")
                with tarfile.open(output_path, "w:gz") as tar:
                    tar.add(archive_dir, arcname=archive_dir.name)

    size_mb = output_path.stat().st_size / 1024 / 1024
    click.echo(f"Archive created: {output_path} ({size_mb:.1f} MB)")


@click.command()
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
                scheduler=_resolve_scheduler(profile),
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

    from imas_codex.graph.ghcr import require_apptainer

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
            manifest = {}
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
            else:
                raise click.ClickException("Archive does not contain graph.dump")

            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
                manifest["pushed"] = False
                manifest["loaded_from"] = str(archive_path)
                save_local_graph_manifest(manifest)

    click.echo("✓ Load complete")


# ============================================================================
# Graph List Command
# ============================================================================


@click.command()
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


@click.command()
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


@click.command()
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
    "--without-dd",
    is_flag=True,
    help="Mark graph as not containing IMAS DD data",
)
def graph_init(
    name: str, facilities: tuple[str, ...], force: bool, without_dd: bool
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
      imas-codex graph init dev -F tcv --without-dd
    """
    from imas_codex.graph.profiles import resolve_neo4j

    facility_list = sorted(set(facilities))
    include_imas = not without_dd

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


@click.group()
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


@click.command()
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
# Graph Repair Subcommand Group
# ============================================================================


@click.group()
def graph_repair_group() -> None:
    """Repair graph data without a full rebuild.

    \b
      imas-codex graph repair cocos-labels    Re-run COCOS label backfill
    """
    pass


@graph_repair_group.command("cocos-labels")
def repair_cocos_labels() -> None:
    """Re-run COCOS label backfill against the live graph.

    Fixes IMASNode nodes that have cocos_label_source set but
    cocos_label_transformation is null (the "half-state" bug).

    This extracts path data from all DD versions, then runs the
    backfill inference pipeline (forward-port from 3.x, expression
    parsing, sign-flip detection).
    """
    from imas_codex.graph.build_dd import (
        _backfill_cocos_labels,
        extract_paths_for_version,
        get_all_dd_versions,
    )
    from imas_codex.graph.client import GraphClient

    gc = GraphClient.from_profile()

    # Check for half-state nodes before repair
    pre_check = gc.query(
        "MATCH (p:IMASNode) "
        "WHERE p.cocos_label_source IS NOT NULL "
        "AND p.cocos_label_transformation IS NULL "
        "RETURN count(*) AS n"
    )
    half_state = pre_check[0]["n"] if pre_check else 0
    click.echo(f"Half-state nodes (source set, label null): {half_state}")

    # Build version data from all DD versions
    versions = get_all_dd_versions()
    click.echo(f"Extracting path data from {len(versions)} DD versions...")

    version_data: dict[str, dict] = {}
    with click.progressbar(versions, label="Extracting") as bar:
        for v in bar:
            version_data[v] = extract_paths_for_version(v)

    # Run backfill
    click.echo("Running COCOS label backfill...")
    count = _backfill_cocos_labels(gc, version_data)

    # Post-repair check
    post_check = gc.query(
        "MATCH (p:IMASNode) "
        "WHERE p.cocos_label_source IS NOT NULL "
        "AND p.cocos_label_transformation IS NULL "
        "RETURN count(*) AS n"
    )
    remaining = post_check[0]["n"] if post_check else 0

    # Breakdown by source
    breakdown = gc.query(
        "MATCH (p:IMASNode) "
        "WHERE p.cocos_label_source IS NOT NULL "
        "RETURN p.cocos_label_source AS source, count(*) AS cnt "
        "ORDER BY cnt DESC"
    )

    gc.close()

    click.echo(f"\n✓ Backfilled {count} COCOS labels")
    click.echo(f"  Half-state nodes remaining: {remaining}")
    if breakdown:
        click.echo("  Breakdown by source:")
        for row in breakdown:
            click.echo(f"    {row['source']}: {row['cnt']}")
    if remaining > 0:
        click.echo(
            "\n⚠ Some half-state nodes remain. These may require manual investigation."
        )
