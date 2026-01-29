"""Data management CLI for graph database and private facility data.

This module provides the `imas-codex data` command group for:
- Unified dump/load of graph database + private YAML files
- Push/pull to GHCR with automatic fork detection
- Neo4j database server management (under `data db`)

Private data is encrypted using age before pushing to GHCR.
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
RECOVERY_DIR = Path.home() / ".local" / "share" / "imas-codex" / "recovery"
DATA_DIR = Path.home() / ".local" / "share" / "imas-codex" / "neo4j"
NEO4J_IMAGE = Path.home() / "apptainer" / "neo4j_2025.11-community.sif"


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
        # Formats: git@github.com:owner/repo.git or https://github.com/owner/repo.git
        if "github.com" in url:
            if url.startswith("git@"):
                # git@github.com:owner/repo.git
                parts = url.split(":")[-1].replace(".git", "").split("/")
            else:
                # https://github.com/owner/repo.git
                parts = url.replace(".git", "").split("/")
            if len(parts) >= 2:
                info["remote_owner"] = parts[-2]

        # Check if this is a fork (not iterorganization)
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


def get_age_key_path() -> Path | None:
    """Get age key file path from environment."""
    key_file = os.environ.get("IMAS_AGE_KEY_FILE")
    if key_file:
        path = Path(key_file).expanduser()
        if path.exists():
            return path
    # Default location
    default = Path.home() / ".config" / "imas-codex" / "age-key.txt"
    if default.exists():
        return default
    return None


def get_age_public_key(key_path: Path) -> str:
    """Extract public key from age key file."""
    content = key_path.read_text()
    for line in content.splitlines():
        if line.startswith("# public key:"):
            return line.split(":")[-1].strip()
    raise click.ClickException(f"Could not find public key in {key_path}")


def encrypt_file(source: Path, dest: Path, public_key: str) -> None:
    """Encrypt a file using age."""
    if not shutil.which("age"):
        raise click.ClickException(
            "age not found in PATH. Install with: brew install age"
        )
    result = subprocess.run(
        ["age", "-r", public_key, "-o", str(dest), str(source)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"age encryption failed: {result.stderr}")


def decrypt_file(source: Path, dest: Path, key_path: Path) -> None:
    """Decrypt a file using age."""
    if not shutil.which("age"):
        raise click.ClickException("age not found in PATH")
    result = subprocess.run(
        ["age", "-d", "-i", str(key_path), "-o", str(dest), str(source)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"age decryption failed: {result.stderr}")


def is_neo4j_running() -> bool:
    """Check if Neo4j is responding on localhost."""
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:7474/", timeout=2)
        return True
    except Exception:
        return False


def backup_existing_data(reason: str) -> Path | None:
    """Backup current graph dump and private YAML to recovery directory.

    Returns path to recovery directory, or None if nothing to backup.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    recovery_path = RECOVERY_DIR / f"{timestamp}-{reason}"

    has_content = False

    # Check if there's anything to backup
    private_files = list(Path(".").glob(PRIVATE_YAML_GLOB))

    if private_files or DATA_DIR.exists():
        recovery_path.mkdir(parents=True, exist_ok=True)

        # Backup private YAML files
        if private_files:
            private_dir = recovery_path / "private"
            private_dir.mkdir(exist_ok=True)
            for f in private_files:
                shutil.copy(f, private_dir / f.name)
            has_content = True

        # Note about graph - we don't copy 300MB each time, just record state
        if DATA_DIR.exists():
            (recovery_path / "graph_data_existed.marker").touch()
            has_content = True

    return recovery_path if has_content else None


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


# ============================================================================
# Main Command Group
# ============================================================================


@click.group()
def data() -> None:
    """Manage graph database and private facility data.

    \b
      imas-codex data dump         Export graph + private YAML to archive
      imas-codex data load <file>  Load archive (backs up existing first)
      imas-codex data push         Push to GHCR (auto-detects fork)
      imas-codex data pull         Pull from GHCR (backs up existing first)
      imas-codex data list         List available versions in registry
      imas-codex data status       Compare local vs registry

    \b
      imas-codex data db start     Start Neo4j server
      imas-codex data db stop      Stop Neo4j server
      imas-codex data db status    Check Neo4j status
      imas-codex data db shell     Open Cypher shell
      imas-codex data db service   Manage systemd service
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
@click.option(
    "--image",
    envvar="NEO4J_IMAGE",
    default=None,
    help="Path to Neo4j SIF image (env: NEO4J_IMAGE)",
)
@click.option(
    "--data-dir",
    envvar="NEO4J_DATA",
    default=None,
    help="Data directory (env: NEO4J_DATA)",
)
@click.option(
    "--password",
    envvar="NEO4J_PASSWORD",
    default="imas-codex",
    help="Neo4j password (env: NEO4J_PASSWORD)",
)
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground")
def db_start(
    image: str | None,
    data_dir: str | None,
    password: str,
    foreground: bool,
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

    # Create directories
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
    click.echo(f"Data directory: {data_path}")

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
        click.echo("Waiting for server...")

        import time

        for _ in range(30):
            if is_neo4j_running():
                click.echo("Neo4j ready at http://localhost:7474")
                click.echo("Bolt: bolt://localhost:7687")
                click.echo(f"Credentials: neo4j / {password}")
                return
            time.sleep(1)

        click.echo(
            "Warning: Neo4j may still be starting. Check with: imas-codex data db status"
        )


@data_db.command("stop")
@click.option(
    "--data-dir",
    envvar="NEO4J_DATA",
    default=None,
    help="Data directory (env: NEO4J_DATA)",
)
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
        result = subprocess.run(
            ["pkill", "-f", "neo4j.*console"],
            capture_output=True,
        )
        if result.returncode == 0:
            click.echo("Neo4j stopped")
        else:
            click.echo("Neo4j not running")


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
            click.echo(f"  Bolt: {data.get('bolt_direct', 'unknown')}")
    except Exception:
        click.echo("Neo4j is not responding on port 7474")


@data_db.command("shell")
@click.option(
    "--image",
    envvar="NEO4J_IMAGE",
    default=None,
    help="Path to Neo4j SIF image (env: NEO4J_IMAGE)",
)
@click.option(
    "--password",
    envvar="NEO4J_PASSWORD",
    default="imas-codex",
    help="Neo4j password (env: NEO4J_PASSWORD)",
)
def db_shell(image: str | None, password: str) -> None:
    """Open Cypher shell to Neo4j."""
    image_path = Path(image) if image else NEO4J_IMAGE

    if not image_path.exists():
        raise click.ClickException(f"Neo4j image not found: {image_path}")

    cmd = [
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
    subprocess.run(cmd)


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
        click.echo(
            "Service installed. Control with: systemctl --user start/stop imas-codex-neo4j"
        )

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
# Data Operations
# ============================================================================


@data.command("dump")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output archive path (default: imas-codex-data-{version}.tar.gz)",
)
@click.option(
    "--no-private",
    is_flag=True,
    help="Exclude private YAML files from archive",
)
def data_dump(output: str | None, no_private: bool) -> None:
    """Export graph database and private YAML to archive.

    Creates a tarball containing:
    - graph.dump: Neo4j database dump
    - manifest.json: Git commit, timestamp, schema version
    - private/*.yaml: Encrypted private facility files (unless --no-private)

    Neo4j must be stopped before dumping.

    Examples:
        imas-codex data dump
        imas-codex data dump -o backup.tar.gz
        imas-codex data dump --no-private
    """
    if is_neo4j_running():
        raise click.ClickException(
            "Neo4j is running. Stop it first: imas-codex data db stop"
        )

    require_apptainer()

    git_info = get_git_info()
    version_label = git_info["tag"] or f"dev-{git_info['commit_short']}"

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"imas-codex-data-{version_label}.tar.gz")

    click.echo(f"Creating archive: {output_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / f"imas-codex-data-{version_label}"
        archive_dir.mkdir()

        # Step 1: Dump graph
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
            click.echo(
                f"    Graph: {(archive_dir / 'graph.dump').stat().st_size / 1024 / 1024:.1f} MB"
            )
        else:
            raise click.ClickException("Graph dump file not created")

        # Step 2: Include private YAML (encrypted if key available)
        if not no_private:
            private_files = list(Path(".").glob(PRIVATE_YAML_GLOB))
            if private_files:
                private_dir = archive_dir / "private"
                private_dir.mkdir()

                age_key = get_age_key_path()
                if age_key:
                    public_key = get_age_public_key(age_key)
                    click.echo(f"  Encrypting {len(private_files)} private files...")
                    for f in private_files:
                        encrypted = private_dir / f"{f.name}.age"
                        encrypt_file(f, encrypted, public_key)
                    click.echo("    Private files encrypted with age")
                else:
                    click.echo(
                        "  Including private files (unencrypted - no age key found)"
                    )
                    click.echo("    Warning: Set IMAS_AGE_KEY_FILE for encryption")
                    for f in private_files:
                        shutil.copy(f, private_dir / f.name)

        # Step 3: Write manifest
        manifest = {
            "version": __version__,
            "git_commit": git_info["commit"],
            "git_tag": git_info["tag"],
            "timestamp": datetime.now(UTC).isoformat(),
            "has_private": not no_private
            and bool(list(Path(".").glob(PRIVATE_YAML_GLOB))),
            "private_encrypted": get_age_key_path() is not None,
        }
        (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Step 4: Create tarball
        click.echo("  Creating archive...")
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(archive_dir, arcname=archive_dir.name)

    click.echo(
        f"Archive created: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)"
    )


@data.command("load")
@click.argument("archive", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Overwrite existing data without prompt")
def data_load(archive: str, force: bool) -> None:
    """Load graph and private YAML from archive.

    Backs up existing data to ~/.local/share/imas-codex/recovery/ first.
    Neo4j must be stopped before loading.

    Examples:
        imas-codex data load imas-codex-data-v1.0.0.tar.gz
        imas-codex data load backup.tar.gz --force
    """
    if is_neo4j_running():
        raise click.ClickException(
            "Neo4j is running. Stop it first: imas-codex data db stop"
        )

    require_apptainer()

    archive_path = Path(archive)
    click.echo(f"Loading archive: {archive_path}")

    # Backup existing data
    recovery_path = backup_existing_data("pre-load")
    if recovery_path:
        click.echo(f"  Backed up existing data to: {recovery_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Extract archive
        click.echo("  Extracting archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmp)

        # Find extracted directory
        extracted_dirs = list(tmp.iterdir())
        if not extracted_dirs:
            raise click.ClickException("Empty archive")
        archive_dir = extracted_dirs[0]

        # Read manifest
        manifest_file = archive_dir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            click.echo(f"  Archive version: {manifest.get('version')}")
            click.echo(f"  Git commit: {manifest.get('git_commit', 'unknown')[:7]}")
            click.echo(f"  Created: {manifest.get('timestamp', 'unknown')}")

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
            click.echo("    Graph loaded successfully")

        # Load private files
        private_dir = archive_dir / "private"
        if private_dir.exists():
            age_key = get_age_key_path()
            target_dir = Path("imas_codex/config/facilities")
            target_dir.mkdir(parents=True, exist_ok=True)

            for f in private_dir.iterdir():
                if f.suffix == ".age":
                    if not age_key:
                        click.echo(f"    Skipping encrypted {f.stem} (no age key)")
                        continue
                    # Decrypt
                    target = target_dir / f.stem  # Remove .age suffix
                    click.echo(f"    Decrypting {f.stem}...")
                    decrypt_file(f, target, age_key)
                else:
                    # Plain copy
                    shutil.copy(f, target_dir / f.name)
                    click.echo(f"    Copied {f.name}")

    click.echo("Load complete. Start Neo4j: imas-codex data db start")


@data.command("push")
@click.option("--dev", is_flag=True, help="Push as dev-{commit} tag")
@click.option(
    "--registry",
    envvar="IMAS_DATA_REGISTRY",
    default=None,
    help="Override registry (default: auto-detect from git remote)",
)
@click.option("--token", envvar="GHCR_TOKEN", help="GHCR token (env: GHCR_TOKEN)")
@click.option("--dry-run", is_flag=True, help="Show what would be pushed")
def data_push(
    dev: bool, registry: str | None, token: str | None, dry_run: bool
) -> None:
    """Push data archive to GHCR.

    Automatically:
    - Detects fork and pushes to your GHCR (not iterorganization)
    - Requires clean git working tree (unless --dev)
    - Uses git tag as version, or --dev for dev-{commit}

    Examples:
        imas-codex data push              # Push release (requires git tag)
        imas-codex data push --dev        # Push dev version
        imas-codex data push --dry-run    # Preview what would happen
    """
    require_oras()

    git_info = get_git_info()

    # Enforce clean git for non-dev pushes
    if not dev:
        require_clean_git(git_info)

    target_registry = get_registry(git_info, registry)
    version_tag = get_version_tag(git_info, dev)

    click.echo(f"Push target: {target_registry}/imas-codex-data:{version_tag}")
    if git_info["is_fork"]:
        click.echo(f"  Detected fork: {git_info['remote_owner']}")
    click.echo(f"  Git commit: {git_info['commit_short']}")
    click.echo(f"  Git dirty: {git_info['is_dirty']}")

    if dry_run:
        click.echo("\n[DRY RUN] Would perform:")
        click.echo("  1. Stop Neo4j")
        click.echo("  2. Create data archive")
        click.echo(f"  3. Push to {target_registry}/imas-codex-data:{version_tag}")
        if not dev:
            click.echo(f"  4. Tag as {target_registry}/imas-codex-data:latest")
        return

    # Create archive
    archive_path = Path(f"imas-codex-data-{version_tag}.tar.gz")

    # Check if Neo4j needs stopping
    neo4j_was_running = is_neo4j_running()
    if neo4j_was_running:
        click.echo("Stopping Neo4j for dump...")
        subprocess.run(
            ["pkill", "-f", "neo4j.*console"],
            capture_output=True,
        )
        import time

        time.sleep(2)

    try:
        # Use dump command logic (invoke directly)
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(data_dump, ["-o", str(archive_path)])
        if result.exit_code != 0:
            raise click.ClickException(f"Dump failed: {result.output}")

        # Login and push
        login_to_ghcr(token)

        artifact_ref = f"{target_registry}/imas-codex-data:{version_tag}"
        push_cmd = [
            "oras",
            "push",
            artifact_ref,
            f"{archive_path}:application/gzip",
            "--annotation",
            f"org.opencontainers.image.version={version_tag}",
            "--annotation",
            f"io.imas-codex.git-commit={git_info['commit']}",
            "--annotation",
            f"io.imas-codex.schema-version={__version__}",
        ]

        click.echo(f"Pushing to {artifact_ref}...")
        result = subprocess.run(push_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(f"Push failed: {result.stderr}")

        click.echo(f"Successfully pushed: {artifact_ref}")

        # Tag as latest for releases
        if not dev:
            tag_cmd = [
                "oras",
                "tag",
                artifact_ref,
                f"{target_registry}/imas-codex-data:latest",
            ]
            subprocess.run(tag_cmd, capture_output=True)
            click.echo(f"Tagged as: {target_registry}/imas-codex-data:latest")

    finally:
        # Restart Neo4j if it was running
        if neo4j_was_running:
            click.echo("Restarting Neo4j...")
            from click.testing import CliRunner

            runner = CliRunner()
            runner.invoke(db_start, [])

        # Clean up archive
        if archive_path.exists():
            archive_path.unlink()


@data.command("pull")
@click.option("-v", "--version", "version", default="latest", help="Version to pull")
@click.option(
    "--registry",
    envvar="IMAS_DATA_REGISTRY",
    default=None,
    help="Override registry (default: auto-detect)",
)
@click.option("--token", envvar="GHCR_TOKEN", help="GHCR token")
@click.option("--graph-only", is_flag=True, help="Only pull graph, skip private files")
def data_pull(
    version: str, registry: str | None, token: str | None, graph_only: bool
) -> None:
    """Pull data archive from GHCR and load.

    Backs up existing data to ~/.local/share/imas-codex/recovery/ first.

    Examples:
        imas-codex data pull                   # Pull latest
        imas-codex data pull -v v1.0.0         # Pull specific version
        imas-codex data pull -v dev-abc1234    # Pull dev version
    """
    require_oras()

    if is_neo4j_running():
        raise click.ClickException(
            "Neo4j is running. Stop it first: imas-codex data db stop"
        )

    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    artifact_ref = f"{target_registry}/imas-codex-data:{version}"

    click.echo(f"Pulling: {artifact_ref}")

    # Backup existing data
    recovery_path = backup_existing_data("pre-pull")
    if recovery_path:
        click.echo(f"Backed up existing data to: {recovery_path}")

    login_to_ghcr(token)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Pull artifact
        pull_cmd = ["oras", "pull", artifact_ref, "-o", str(tmp)]
        result = subprocess.run(pull_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(f"Pull failed: {result.stderr}")

        # Find downloaded file
        archives = list(tmp.glob("*.tar.gz"))
        if not archives:
            raise click.ClickException("No archive found in pulled artifact")

        archive_path = archives[0]
        click.echo(f"Downloaded: {archive_path.name}")

        # Load using load command
        from click.testing import CliRunner

        runner = CliRunner()
        args = [str(archive_path), "--force"]
        result = runner.invoke(data_load, args)
        if result.exit_code != 0:
            raise click.ClickException(f"Load failed: {result.output}")

    click.echo("Pull complete. Start Neo4j: imas-codex data db start")


@data.command("list")
@click.option(
    "--registry",
    envvar="IMAS_DATA_REGISTRY",
    default=None,
    help="Override registry",
)
@click.option("--token", envvar="GHCR_TOKEN", help="GHCR token")
def data_list(registry: str | None, token: str | None) -> None:
    """List available versions in GHCR.

    Examples:
        imas-codex data list
    """
    require_oras()

    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    repo_ref = f"{target_registry}/imas-codex-data"

    login_to_ghcr(token)

    click.echo(f"Available versions in {repo_ref}:")

    result = subprocess.run(
        ["oras", "repo", "tags", repo_ref],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if "not found" in result.stderr.lower() or "NAME_UNKNOWN" in result.stderr:
            click.echo("  (no versions published yet)")
        else:
            raise click.ClickException(f"Failed to list: {result.stderr}")
    else:
        tags = result.stdout.strip().split("\n")
        for tag in sorted(tags, reverse=True):
            if tag:
                prefix = "→" if tag == "latest" else " "
                click.echo(f"  {prefix} {tag}")


@data.command("status")
@click.option(
    "--registry",
    envvar="IMAS_DATA_REGISTRY",
    default=None,
    help="Override registry",
)
def data_status(registry: str | None) -> None:
    """Show local vs registry status.

    Examples:
        imas-codex data status
    """
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)

    click.echo("Local status:")
    click.echo(f"  Git commit: {git_info['commit_short']}")
    click.echo(f"  Git tag: {git_info['tag'] or '(none)'}")
    click.echo(f"  Dirty: {git_info['is_dirty']}")
    click.echo(f"  Is fork: {git_info['is_fork']}")
    click.echo(f"  Target registry: {target_registry}")

    # Check Neo4j
    click.echo(f"\nNeo4j: {'running' if is_neo4j_running() else 'stopped'}")

    # Check private files
    private_files = list(Path(".").glob(PRIVATE_YAML_GLOB))
    click.echo(f"Private YAML files: {len(private_files)}")

    # Check age key
    age_key = get_age_key_path()
    click.echo(f"Age encryption: {'configured' if age_key else 'not configured'}")
    if not age_key:
        click.echo("  Set IMAS_AGE_KEY_FILE or create ~/.config/imas-codex/age-key.txt")
