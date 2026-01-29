"""Data management CLI for graph database and private facility data.

This module provides the `imas-codex data` command group for:
- Graph database dump/load/push/pull to GHCR
- Neo4j database server management (under `data db`)
- Private YAML file management via GitHub Gist (under `data private`)
"""

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import click

from imas_codex import __version__

# ============================================================================
# Constants and Helpers
# ============================================================================

PRIVATE_YAML_GLOB = "imas_codex/config/facilities/*_private.yaml"
PRIVATE_YAML_DIR = Path("imas_codex/config/facilities")
RECOVERY_DIR = Path.home() / ".local" / "share" / "imas-codex" / "recovery"
DATA_DIR = Path.home() / ".local" / "share" / "imas-codex" / "neo4j"
NEO4J_IMAGE = Path.home() / "apptainer" / "neo4j_2025.11-community.sif"
GIST_ID_FILE = Path.home() / ".config" / "imas-codex" / "private-gist-id"


def get_git_info() -> dict:
    """Get current git state: commit, tag, remote, dirty status."""
    info = {
        "commit": None,
        "commit_short": None,
        "tag": None,
        "is_dirty": False,
        "remote_owner": None,
        "remote_url": None,
        "is_fork": False,
    }

    # Get current commit
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if result.returncode == 0:
        info["commit"] = result.stdout.strip()
        info["commit_short"] = info["commit"][:7]

    # Check if on a tag
    result = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        info["tag"] = result.stdout.strip()

    # Check dirty state
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    info["is_dirty"] = bool(result.stdout.strip())

    # Get origin remote URL to detect fork
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, text=True
    )
    if result.returncode == 0:
        url = result.stdout.strip()
        info["remote_url"] = url
        # Extract owner from GitHub URL
        if "github.com" in url:
            if url.startswith("git@"):
                parts = url.split(":")[-1].replace(".git", "").split("/")
            else:
                parts = url.replace(".git", "").split("/")
            if len(parts) >= 2:
                info["remote_owner"] = parts[-2]

        info["is_fork"] = (
            info["remote_owner"] is not None
            and info["remote_owner"].lower() != "iterorganization"
        )

    return info


def get_registry(git_info: dict, force_registry: str | None = None) -> str:
    """Determine GHCR registry based on git remote or explicit override."""
    if force_registry:
        return force_registry

    if git_info["is_fork"] and git_info["remote_owner"]:
        return f"ghcr.io/{git_info['remote_owner'].lower()}"
    return "ghcr.io/iterorganization"


def get_version_tag(git_info: dict, dev: bool = False) -> str:
    """Determine version tag for push."""
    if dev:
        return f"dev-{git_info['commit_short']}"
    if git_info["tag"]:
        return git_info["tag"]
    raise click.ClickException(
        "Not on a git tag. Use --dev for development push, or create a tag first."
    )


def require_clean_git(git_info: dict) -> None:
    """Raise error if git working tree is dirty."""
    if git_info["is_dirty"]:
        raise click.ClickException(
            "Working tree has uncommitted changes. Commit or stash first."
        )


def require_oras() -> None:
    """Raise error if oras is not installed."""
    if not shutil.which("oras"):
        raise click.ClickException(
            "oras not found in PATH. Install from: "
            "https://github.com/oras-project/oras/releases"
        )


def require_apptainer() -> None:
    """Raise error if apptainer is not installed."""
    if not shutil.which("apptainer"):
        raise click.ClickException("apptainer not found in PATH")


def require_gh() -> None:
    """Raise error if gh CLI is not installed."""
    if not shutil.which("gh"):
        raise click.ClickException(
            "gh CLI not found. Install from: https://cli.github.com/"
        )


def is_neo4j_running() -> bool:
    """Check if Neo4j is responding on localhost."""
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:7474/", timeout=2)
        return True
    except Exception:
        return False


def backup_existing_data(reason: str) -> Path | None:
    """Backup current graph state marker to recovery directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    recovery_path = RECOVERY_DIR / f"{timestamp}-{reason}"

    if DATA_DIR.exists():
        recovery_path.mkdir(parents=True, exist_ok=True)
        (recovery_path / "graph_data_existed.marker").touch()
        return recovery_path

    return None


def login_to_ghcr(token: str | None) -> None:
    """Login to GHCR using provided token."""
    if not token:
        return

    result = subprocess.run(
        ["oras", "login", "ghcr.io", "-u", "token", "--password-stdin"],
        input=token,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"GHCR login failed: {result.stderr}")


def get_private_files() -> list[Path]:
    """Get list of private YAML files."""
    return list(Path(".").glob(PRIVATE_YAML_GLOB))


def get_saved_gist_id() -> str | None:
    """Get saved gist ID from config file."""
    if GIST_ID_FILE.exists():
        return GIST_ID_FILE.read_text().strip()
    return os.environ.get("IMAS_PRIVATE_GIST_ID")


def save_gist_id(gist_id: str) -> None:
    """Save gist ID to config file."""
    GIST_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    GIST_ID_FILE.write_text(gist_id)


# ============================================================================
# Main Command Group
# ============================================================================


@click.group()
def data() -> None:
    """Manage graph database and private facility data.

    \b
      imas-codex data dump         Export graph to archive
      imas-codex data load <file>  Load graph archive
      imas-codex data push         Push graph to GHCR
      imas-codex data pull         Pull graph from GHCR
      imas-codex data list         List available versions
      imas-codex data status       Show local status

    \b
      imas-codex data db start     Start Neo4j server
      imas-codex data db stop      Stop Neo4j server
      imas-codex data db status    Check Neo4j status

    \b
      imas-codex data private push   Push private YAML to GitHub Gist
      imas-codex data private pull   Pull private YAML from Gist
      imas-codex data private status Show Gist status
    """
    pass


# ============================================================================
# Database Subgroup
# ============================================================================


@data.group("db")
def data_db() -> None:
    """Manage Neo4j graph database server.

    \b
      imas-codex data db start     Start Neo4j via Apptainer
      imas-codex data db stop      Stop Neo4j
      imas-codex data db status    Check status
      imas-codex data db shell     Open Cypher shell
      imas-codex data db service   Manage systemd service
    """
    pass


@data_db.command("start")
@click.option("--image", envvar="NEO4J_IMAGE", default=None)
@click.option("--data-dir", envvar="NEO4J_DATA", default=None)
@click.option("--password", envvar="NEO4J_PASSWORD", default="imas-codex")
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground")
def db_start(
    image: str | None, data_dir: str | None, password: str, foreground: bool
) -> None:
    """Start Neo4j server via Apptainer."""
    import platform

    if platform.system() in ("Windows", "Darwin"):
        click.echo("On Windows/Mac, use Docker: docker compose up -d neo4j", err=True)
        raise SystemExit(1)

    require_apptainer()

    image_path = Path(image) if image else NEO4J_IMAGE
    data_path = Path(data_dir) if data_dir else DATA_DIR

    if not image_path.exists():
        raise click.ClickException(
            f"Neo4j image not found: {image_path}\n"
            "Pull: apptainer pull docker://neo4j:2025.11-community"
        )

    if is_neo4j_running():
        click.echo("Neo4j is already running on port 7474")
        return

    for subdir in ["data", "logs", "conf", "import"]:
        (data_path / subdir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{data_path}/data:/data",
        "--bind",
        f"{data_path}/logs:/logs",
        "--bind",
        f"{data_path}/import:/import",
        "--writable-tmpfs",
        "--env",
        f"NEO4J_AUTH=neo4j/{password}",
        str(image_path),
        "neo4j",
        "console",
    ]

    click.echo(f"Starting Neo4j from {image_path}")

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
            if is_neo4j_running():
                click.echo("Neo4j ready at http://localhost:7474")
                return
            time.sleep(1)

        click.echo("Warning: Neo4j may still be starting")


@data_db.command("stop")
@click.option("--data-dir", envvar="NEO4J_DATA", default=None)
def db_stop(data_dir: str | None) -> None:
    """Stop Neo4j server."""
    import signal

    data_path = Path(data_dir) if data_dir else DATA_DIR
    pid_file = data_path / "neo4j.pid"

    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            click.echo(f"Sent SIGTERM to Neo4j (PID: {pid})")
            pid_file.unlink()
        except ProcessLookupError:
            click.echo("Neo4j process not found (stale PID file)")
            pid_file.unlink()
    else:
        result = subprocess.run(["pkill", "-f", "neo4j.*console"], capture_output=True)
        click.echo("Neo4j stopped" if result.returncode == 0 else "Neo4j not running")


@data_db.command("status")
def db_status() -> None:
    """Check Neo4j server status."""
    try:
        import urllib.request

        with urllib.request.urlopen("http://localhost:7474/", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            click.echo("Neo4j is running")
            click.echo(f"  Version: {data.get('neo4j_version', 'unknown')}")
            click.echo(f"  Edition: {data.get('neo4j_edition', 'unknown')}")
    except Exception:
        click.echo("Neo4j is not responding on port 7474")


@data_db.command("shell")
@click.option("--image", envvar="NEO4J_IMAGE", default=None)
@click.option("--password", envvar="NEO4J_PASSWORD", default="imas-codex")
def db_shell(image: str | None, password: str) -> None:
    """Open Cypher shell to Neo4j."""
    image_path = Path(image) if image else NEO4J_IMAGE

    if not image_path.exists():
        raise click.ClickException(f"Neo4j image not found: {image_path}")

    subprocess.run(
        [
            "apptainer",
            "exec",
            "--writable-tmpfs",
            str(image_path),
            "cypher-shell",
            "-u",
            "neo4j",
            "-p",
            password,
        ]
    )


@data_db.command("service")
@click.argument("action", type=click.Choice(["install", "uninstall", "status"]))
@click.option("--image", envvar="NEO4J_IMAGE", default=None)
@click.option("--data-dir", envvar="NEO4J_DATA", default=None)
@click.option("--password", envvar="NEO4J_PASSWORD", default="imas-codex")
def db_service(
    action: str, image: str | None, data_dir: str | None, password: str
) -> None:
    """Manage Neo4j as a systemd user service."""
    import platform

    if platform.system() != "Linux":
        raise click.ClickException("systemd services only supported on Linux")

    if not shutil.which("systemctl"):
        raise click.ClickException("systemctl not found")

    require_apptainer()

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_file = service_dir / "imas-codex-neo4j.service"
    image_path = Path(image) if image else NEO4J_IMAGE
    data_path = Path(data_dir) if data_dir else DATA_DIR
    apptainer_path = shutil.which("apptainer")

    if action == "install":
        if not image_path.exists():
            raise click.ClickException(f"Neo4j image not found: {image_path}")

        service_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["data", "logs", "conf", "import"]:
            (data_path / subdir).mkdir(parents=True, exist_ok=True)

        service_content = f"""[Unit]
Description=Neo4j Graph Database (IMAS Codex)
After=network.target

[Service]
Type=simple
ExecStart={apptainer_path} exec \\
    --bind {data_path}/data:/data \\
    --bind {data_path}/logs:/logs \\
    --bind {data_path}/import:/import \\
    --writable-tmpfs \\
    --env NEO4J_AUTH=neo4j/{password} \\
    {image_path} \\
    neo4j console
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
        service_file.write_text(service_content)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "imas-codex-neo4j"], check=True
        )
        click.echo("Service installed")

    elif action == "uninstall":
        if not service_file.exists():
            click.echo("Service not installed")
            return
        subprocess.run(
            ["systemctl", "--user", "stop", "imas-codex-neo4j"], capture_output=True
        )
        subprocess.run(
            ["systemctl", "--user", "disable", "imas-codex-neo4j"], capture_output=True
        )
        service_file.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        click.echo("Service uninstalled")

    elif action == "status":
        if not service_file.exists():
            click.echo("Service not installed")
            return
        result = subprocess.run(
            ["systemctl", "--user", "status", "imas-codex-neo4j"],
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout)


# ============================================================================
# Private Data Subgroup (GitHub Gist)
# ============================================================================


@data.group("private")
def data_private() -> None:
    """Manage private facility YAML via GitHub Gist.

    Uses the `gh` CLI to create and manage a secret gist containing
    your private facility configuration files.

    \b
      imas-codex data private push    Create/update secret gist
      imas-codex data private pull    Download and restore files
      imas-codex data private status  Show gist URL and file status
    """
    pass


@data_private.command("push")
@click.option("--gist-id", envvar="IMAS_PRIVATE_GIST_ID", help="Existing gist ID")
@click.option("--dry-run", is_flag=True, help="Show what would be pushed")
def private_push(gist_id: str | None, dry_run: bool) -> None:
    """Push private YAML files to a secret GitHub Gist.

    Creates a new secret gist on first run, updates existing on subsequent runs.
    The gist ID is saved to ~/.config/imas-codex/private-gist-id

    Examples:
        imas-codex data private push
        imas-codex data private push --dry-run
    """
    require_gh()

    private_files = get_private_files()
    if not private_files:
        click.echo("No private YAML files found")
        click.echo(f"  Expected pattern: {PRIVATE_YAML_GLOB}")
        return

    # Use saved gist ID if not provided
    effective_gist_id = gist_id or get_saved_gist_id()

    click.echo(f"Private files to push: {len(private_files)}")
    for f in private_files:
        click.echo(f"  - {f.name}")

    if dry_run:
        if effective_gist_id:
            click.echo(f"\n[DRY RUN] Would update gist: {effective_gist_id}")
        else:
            click.echo("\n[DRY RUN] Would create new secret gist")
        return

    if effective_gist_id:
        # Update existing gist
        click.echo(f"\nUpdating gist {effective_gist_id}...")
        cmd = ["gh", "gist", "edit", effective_gist_id]
        for f in private_files:
            cmd.extend(["-a", str(f)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Gist might not exist anymore, create new one
            click.echo("Gist not found, creating new one...")
            effective_gist_id = None

    if not effective_gist_id:
        # Create new secret gist
        click.echo("\nCreating secret gist...")
        cmd = [
            "gh",
            "gist",
            "create",
            "--desc",
            "imas-codex private facility configs",
            *[str(f) for f in private_files],
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(f"Failed to create gist: {result.stderr}")

        # Extract gist URL/ID from output
        gist_url = result.stdout.strip()
        effective_gist_id = gist_url.split("/")[-1]

        # Save gist ID for future use
        save_gist_id(effective_gist_id)
        click.echo(f"Gist ID saved to: {GIST_ID_FILE}")

    click.echo(f"\n✓ Pushed to: https://gist.github.com/{effective_gist_id}")
    click.echo("  (This is a secret gist - only accessible with the URL)")


@data_private.command("pull")
@click.option("--gist-id", envvar="IMAS_PRIVATE_GIST_ID", help="Gist ID to pull from")
@click.option("--force", is_flag=True, help="Overwrite existing files without backup")
def private_pull(gist_id: str | None, force: bool) -> None:
    """Pull private YAML files from GitHub Gist.

    Downloads files from the gist and places them in the facilities config directory.
    Existing files are backed up to the recovery directory first.

    Examples:
        imas-codex data private pull
        imas-codex data private pull --gist-id abc123def456
    """
    require_gh()

    effective_gist_id = gist_id or get_saved_gist_id()
    if not effective_gist_id:
        raise click.ClickException(
            "No gist ID configured. Either:\n"
            "  1. Run 'imas-codex data private push' first, or\n"
            "  2. Provide --gist-id, or\n"
            "  3. Set IMAS_PRIVATE_GIST_ID environment variable"
        )

    click.echo(f"Pulling from gist: {effective_gist_id}")

    # Backup existing files
    existing_files = get_private_files()
    if existing_files and not force:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        backup_dir = RECOVERY_DIR / f"{timestamp}-private-pull"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for f in existing_files:
            shutil.copy(f, backup_dir / f.name)
        click.echo(f"Backed up {len(existing_files)} files to: {backup_dir}")

    # Get gist files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Clone the gist
        result = subprocess.run(
            ["gh", "gist", "clone", effective_gist_id, str(tmp / "gist")],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise click.ClickException(f"Failed to clone gist: {result.stderr}")

        gist_dir = tmp / "gist"
        yaml_files = list(gist_dir.glob("*_private.yaml"))

        if not yaml_files:
            click.echo("No *_private.yaml files found in gist")
            return

        # Copy files to target directory
        PRIVATE_YAML_DIR.mkdir(parents=True, exist_ok=True)
        for f in yaml_files:
            target = PRIVATE_YAML_DIR / f.name
            shutil.copy(f, target)
            click.echo(f"  ✓ {f.name}")

    # Save gist ID if we didn't have it
    if not get_saved_gist_id():
        save_gist_id(effective_gist_id)

    click.echo(f"\n✓ Pulled {len(yaml_files)} files")


@data_private.command("status")
def private_status() -> None:
    """Show private YAML file status and gist configuration."""
    private_files = get_private_files()
    gist_id = get_saved_gist_id()

    click.echo("Private YAML files:")
    if private_files:
        for f in private_files:
            size = f.stat().st_size
            click.echo(f"  - {f.name} ({size} bytes)")
    else:
        click.echo("  (none found)")
        click.echo(f"  Expected pattern: {PRIVATE_YAML_GLOB}")

    click.echo(f"\nGist ID: {gist_id or '(not configured)'}")
    if gist_id:
        click.echo(f"  URL: https://gist.github.com/{gist_id}")
        click.echo(f"  Config: {GIST_ID_FILE}")


# ============================================================================
# Graph Data Operations
# ============================================================================


@data.command("dump")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output archive path (default: imas-codex-graph-{version}.tar.gz)",
)
def data_dump(output: str | None) -> None:
    """Export graph database to archive.

    Creates a tarball containing the Neo4j database dump.
    Neo4j must be stopped before dumping.

    Examples:
        imas-codex data dump
        imas-codex data dump -o backup.tar.gz
    """
    if is_neo4j_running():
        raise click.ClickException(
            "Neo4j is running. Stop it first: imas-codex data db stop"
        )

    require_apptainer()

    git_info = get_git_info()
    version_label = git_info["tag"] or f"dev-{git_info['commit_short']}"

    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"imas-codex-graph-{version_label}.tar.gz")

    click.echo(f"Creating archive: {output_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / f"imas-codex-graph-{version_label}"
        archive_dir.mkdir()

        # Dump graph
        click.echo("  Dumping graph database...")
        dumps_dir = DATA_DIR / "dumps"
        dumps_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{DATA_DIR}/data:/data",
            "--bind",
            f"{dumps_dir}:/dumps",
            "--writable-tmpfs",
            str(NEO4J_IMAGE),
            "neo4j-admin",
            "database",
            "dump",
            "neo4j",
            "--to-path=/dumps",
            "--overwrite-destination=true",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(f"Graph dump failed: {result.stderr}")

        graph_dump = dumps_dir / "neo4j.dump"
        if graph_dump.exists():
            shutil.move(str(graph_dump), str(archive_dir / "graph.dump"))
            size_mb = (archive_dir / "graph.dump").stat().st_size / 1024 / 1024
            click.echo(f"    Graph: {size_mb:.1f} MB")
        else:
            raise click.ClickException("Graph dump file not created")

        # Write manifest
        manifest = {
            "version": __version__,
            "git_commit": git_info["commit"],
            "git_tag": git_info["tag"],
            "timestamp": datetime.now(UTC).isoformat(),
        }
        (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Create tarball
        click.echo("  Creating archive...")
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(archive_dir, arcname=archive_dir.name)

    size_mb = output_path.stat().st_size / 1024 / 1024
    click.echo(f"Archive created: {output_path} ({size_mb:.1f} MB)")


@data.command("load")
@click.argument("archive", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Overwrite existing data")
def data_load(archive: str, force: bool) -> None:
    """Load graph database from archive.

    Neo4j must be stopped before loading.

    Examples:
        imas-codex data load imas-codex-graph-v1.0.0.tar.gz
    """
    if is_neo4j_running():
        raise click.ClickException(
            "Neo4j is running. Stop it first: imas-codex data db stop"
        )

    require_apptainer()

    archive_path = Path(archive)
    click.echo(f"Loading archive: {archive_path}")

    backup_existing_data("pre-load")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        click.echo("  Extracting archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmp)

        extracted_dirs = list(tmp.iterdir())
        if not extracted_dirs:
            raise click.ClickException("Empty archive")
        archive_dir = extracted_dirs[0]

        # Read manifest
        manifest_file = archive_dir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            click.echo(f"  Version: {manifest.get('version')}")
            click.echo(f"  Commit: {manifest.get('git_commit', 'unknown')[:7]}")

        # Load graph
        graph_dump = archive_dir / "graph.dump"
        if graph_dump.exists():
            click.echo("  Loading graph database...")
            dumps_dir = DATA_DIR / "dumps"
            dumps_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(graph_dump, dumps_dir / "neo4j.dump")

            cmd = [
                "apptainer",
                "exec",
                "--bind",
                f"{DATA_DIR}/data:/data",
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

    click.echo("✓ Load complete. Start Neo4j: imas-codex data db start")


@data.command("push")
@click.option("--dev", is_flag=True, help="Push as dev-{commit} tag")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option("--dry-run", is_flag=True, help="Show what would be pushed")
def data_push(
    dev: bool, registry: str | None, token: str | None, dry_run: bool
) -> None:
    """Push graph archive to GHCR.

    Auto-detects fork and pushes to your GHCR registry.
    Requires clean git state for release pushes.

    Examples:
        imas-codex data push              # Push release (requires git tag)
        imas-codex data push --dev        # Push dev version
    """
    require_oras()

    git_info = get_git_info()

    if not dev:
        require_clean_git(git_info)

    target_registry = get_registry(git_info, registry)
    version_tag = get_version_tag(git_info, dev)

    click.echo(f"Push target: {target_registry}/imas-codex-graph:{version_tag}")
    if git_info["is_fork"]:
        click.echo(f"  Detected fork: {git_info['remote_owner']}")

    if dry_run:
        click.echo("\n[DRY RUN] Would:")
        click.echo("  1. Stop Neo4j, dump graph")
        click.echo(f"  2. Push to {target_registry}/imas-codex-graph:{version_tag}")
        return

    archive_path = Path(f"imas-codex-graph-{version_tag}.tar.gz")

    neo4j_was_running = is_neo4j_running()
    if neo4j_was_running:
        click.echo("Stopping Neo4j...")
        subprocess.run(["pkill", "-f", "neo4j.*console"], capture_output=True)
        import time

        time.sleep(2)

    try:
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(data_dump, ["-o", str(archive_path)])
        if result.exit_code != 0:
            raise click.ClickException(f"Dump failed: {result.output}")

        login_to_ghcr(token)

        artifact_ref = f"{target_registry}/imas-codex-graph:{version_tag}"
        push_cmd = [
            "oras",
            "push",
            artifact_ref,
            f"{archive_path}:application/gzip",
            "--annotation",
            f"org.opencontainers.image.version={version_tag}",
            "--annotation",
            f"io.imas-codex.git-commit={git_info['commit']}",
        ]

        click.echo(f"Pushing to {artifact_ref}...")
        result = subprocess.run(push_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(f"Push failed: {result.stderr}")

        click.echo(f"✓ Pushed: {artifact_ref}")

        if not dev:
            subprocess.run(
                [
                    "oras",
                    "tag",
                    artifact_ref,
                    f"{target_registry}/imas-codex-graph:latest",
                ],
                capture_output=True,
            )
            click.echo("✓ Tagged as latest")

    finally:
        if neo4j_was_running:
            from click.testing import CliRunner

            runner = CliRunner()
            runner.invoke(db_start, [])

        if archive_path.exists():
            archive_path.unlink()


@data.command("pull")
@click.option("-v", "--version", "version", default="latest")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
def data_pull(version: str, registry: str | None, token: str | None) -> None:
    """Pull graph archive from GHCR and load.

    Examples:
        imas-codex data pull                 # Pull latest
        imas-codex data pull -v v1.0.0       # Pull specific version
        imas-codex data pull -v dev-abc1234  # Pull dev version
    """
    require_oras()

    if is_neo4j_running():
        raise click.ClickException(
            "Neo4j is running. Stop it first: imas-codex data db stop"
        )

    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    artifact_ref = f"{target_registry}/imas-codex-graph:{version}"

    click.echo(f"Pulling: {artifact_ref}")

    backup_existing_data("pre-pull")
    login_to_ghcr(token)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        result = subprocess.run(
            ["oras", "pull", artifact_ref, "-o", str(tmp)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise click.ClickException(f"Pull failed: {result.stderr}")

        archives = list(tmp.glob("*.tar.gz"))
        if not archives:
            raise click.ClickException("No archive found")

        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(data_load, [str(archives[0]), "--force"])
        if result.exit_code != 0:
            raise click.ClickException(f"Load failed: {result.output}")

    click.echo("✓ Pull complete. Start Neo4j: imas-codex data db start")


@data.command("list")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
def data_list(registry: str | None, token: str | None) -> None:
    """List available graph versions in GHCR."""
    require_oras()

    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    repo_ref = f"{target_registry}/imas-codex-graph"

    login_to_ghcr(token)

    click.echo(f"Available versions in {repo_ref}:")

    result = subprocess.run(
        ["oras", "repo", "tags", repo_ref],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if "not found" in result.stderr.lower():
            click.echo("  (no versions published yet)")
        else:
            raise click.ClickException(f"Failed: {result.stderr}")
    else:
        for tag in sorted(result.stdout.strip().split("\n"), reverse=True):
            if tag:
                click.echo(f"  {'→' if tag == 'latest' else ' '} {tag}")


@data.command("status")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
def data_status(registry: str | None) -> None:
    """Show local and registry status."""
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    gist_id = get_saved_gist_id()
    private_files = get_private_files()

    click.echo("Local status:")
    click.echo(f"  Git commit: {git_info['commit_short']}")
    click.echo(f"  Git tag: {git_info['tag'] or '(none)'}")
    click.echo(f"  Is fork: {git_info['is_fork']}")
    click.echo(f"  Target registry: {target_registry}")

    click.echo(f"\nNeo4j: {'running' if is_neo4j_running() else 'stopped'}")

    click.echo(f"\nPrivate YAML: {len(private_files)} files")
    click.echo(f"  Gist: {gist_id or '(not configured)'}")
    if not gist_id and private_files:
        click.echo("  → Run 'imas-codex data private push' to backup")
