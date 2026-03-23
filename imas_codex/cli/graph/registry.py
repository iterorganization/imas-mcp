"""Graph CLI registry commands — push, fetch, pull, tags, prune."""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import click

from imas_codex import __version__
from imas_codex.graph.ghcr import (
    delete_tag as _delete_tag,
    dispatch_graph_quality as _dispatch_graph_quality,
    ensure_fresh_version as _ensure_fresh_version,
    fetch_tag_messages as _fetch_tag_messages,
    get_git_info,
    get_local_graph_manifest,
    get_package_name,
    get_registry,
    get_version_tag,
    list_registry_tags as _list_registry_tags,
    login_to_ghcr,
    require_clean_git,
    require_oras,
    resolve_latest_tag as _resolve_latest_tag,
    save_dev_revision as _save_dev_revision,
    save_local_graph_manifest,
)
from imas_codex.graph.neo4j_ops import (
    backup_existing_data,
    check_graph_exists,
)


@click.command()
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
@click.option(
    "--version",
    "version_tag_override",
    default=None,
    help="Override version tag (e.g. v5.0.0-rc2). Bypasses git tag detection.",
)
@click.option(
    "--source-dump",
    type=click.Path(exists=True),
    default=None,
    help="Use pre-existing dump file (avoids Neo4j stop/start per variant).",
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
    version_tag_override: str | None = None,
    source_dump: str | None = None,
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

    target_registry = get_registry(git_info, registry)

    if version_tag_override:
        version_tag = version_tag_override
    else:
        # Ensure __version__ reflects current git state (hatch-vcs freezes at
        # uv sync time — without this, the GHCR tag embeds a stale commit hash).
        fresh_version = _ensure_fresh_version()
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

            from imas_codex.cli.graph.data import graph_export

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
            if source_dump:
                dump_args.extend(["--source-dump", source_dump])
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
                f"{archive_path.name}:application/gzip",
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
            run_oras_with_progress(push_cmd, progress=gp, cwd=archive_path.parent)
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


@click.command()
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


@click.command()
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

            from imas_codex.cli.graph.data import graph_load

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


@click.command()
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


@click.command()
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
