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
SERVICES_DIR = Path("imas_codex/config/services")
RECOVERY_DIR = Path.home() / ".local" / "share" / "imas-codex" / "recovery"
DATA_DIR = Path.home() / ".local" / "share" / "imas-codex" / "neo4j"
NEO4J_IMAGE = Path.home() / "apptainer" / "neo4j_2025.11-community.sif"
GIST_ID_FILE = Path.home() / ".config" / "imas-codex" / "private-gist-id"
LOCAL_GRAPH_MANIFEST = Path.home() / ".config" / "imas-codex" / "graph-manifest.json"
NEO4J_LOCK_FILE = Path.home() / ".config" / "imas-codex" / "neo4j-operation.lock"


# ============================================================================
# Neo4j Operation Context Manager
# ============================================================================


class Neo4jOperation:
    """Context manager for Neo4j operations requiring stop/start.

    Handles:
    - Checking for concurrent operations (via lock file)
    - Stopping Neo4j if running
    - Performing the operation
    - Restarting Neo4j if it was running

    Usage:
        with Neo4jOperation("dump graph") as op:
            if op.acquired:
                # Do operation...
                pass
    """

    def __init__(self, operation_name: str, require_stopped: bool = True):
        self.operation_name = operation_name
        self.require_stopped = require_stopped
        self.acquired = False
        self.was_running = False
        self._lock_file: Path = NEO4J_LOCK_FILE

    def __enter__(self) -> "Neo4jOperation":
        # Check for existing lock
        if self._lock_file.exists():
            lock_info = json.loads(self._lock_file.read_text())
            pid = lock_info.get("pid")
            operation = lock_info.get("operation", "unknown")
            started = lock_info.get("started", "unknown")

            # Check if process is still running
            if pid and self._process_exists(pid):
                raise click.ClickException(
                    f"Another operation is in progress:\n"
                    f"  Operation: {operation}\n"
                    f"  Started: {started}\n"
                    f"  PID: {pid}\n\n"
                    f"If the process crashed, remove lock: rm {self._lock_file}"
                )
            else:
                # Stale lock, remove it
                self._lock_file.unlink()

        # Acquire lock
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_info = {
            "operation": self.operation_name,
            "pid": os.getpid(),
            "started": datetime.now(UTC).isoformat(),
        }
        self._lock_file.write_text(json.dumps(lock_info))
        self.acquired = True

        # Stop Neo4j if needed
        if self.require_stopped and is_neo4j_running():
            self.was_running = True
            click.echo(f"Stopping Neo4j for {self.operation_name}...")
            self._stop_neo4j()

            # Wait for stop
            import time

            for _ in range(30):
                if not is_neo4j_running():
                    break
                time.sleep(1)
            else:
                self._release_lock()
                raise click.ClickException("Failed to stop Neo4j within 30 seconds")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            # Restart Neo4j if it was running
            if self.was_running:
                click.echo("Restarting Neo4j...")
                self._start_neo4j()
        finally:
            self._release_lock()
        return False

    def _process_exists(self, pid: int) -> bool:
        """Check if a process with given PID exists."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _stop_neo4j(self) -> None:
        """Stop Neo4j via systemd or pkill."""
        # Try systemd first
        result = subprocess.run(
            ["systemctl", "--user", "stop", "imas-codex-neo4j"],
            capture_output=True,
        )
        if result.returncode == 0:
            return

        # Fallback to pkill - try multiple patterns that may match
        # Pattern 1: Apptainer container with neo4j image
        subprocess.run(
            ["pkill", "-15", "-f", "neo4j_2025.*community.sif"],
            capture_output=True,
        )
        # Pattern 2: Neo4j Java process
        subprocess.run(
            ["pkill", "-15", "-f", "Neo4jCommunity"],
            capture_output=True,
        )
        # Pattern 3: Legacy pattern
        subprocess.run(["pkill", "-15", "-f", "neo4j.*console"], capture_output=True)

    def _start_neo4j(self) -> None:
        """Start Neo4j via systemd or direct command."""
        # Try systemd first
        result = subprocess.run(
            ["systemctl", "--user", "start", "imas-codex-neo4j"],
            capture_output=True,
        )
        if result.returncode == 0:
            # Wait for startup
            import time

            for _ in range(30):
                if is_neo4j_running():
                    click.echo("Neo4j ready")
                    return
                time.sleep(1)
            click.echo("Warning: Neo4j may still be starting")
            return

        # Fallback to direct start (would need to invoke db_start)
        click.echo("Note: Restart Neo4j manually: imas-codex data db start")

    def _release_lock(self) -> None:
        """Release the lock file."""
        if self._lock_file.exists():
            self._lock_file.unlink()
        self.acquired = False


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
    """Determine version tag for push.

    For dev pushes, uses full package version (e.g., 3.2.1.dev588-g2d31208df.d20260129)
    which includes base version, distance from tag, commit hash, and dirty date.
    OCI tags don't allow '+', so we replace with '-'.
    """
    if dev:
        # Use full PEP 440 version, replacing '+' with '-' for OCI compatibility
        return __version__.replace("+", "-")
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
    """Get list of private facility YAML files."""
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


def get_file_hash(path: Path) -> str:
    """Get SHA256 hash of file contents."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def get_local_graph_manifest() -> dict | None:
    """Get manifest of currently loaded graph."""
    if LOCAL_GRAPH_MANIFEST.exists():
        return json.loads(LOCAL_GRAPH_MANIFEST.read_text())
    return None


def save_local_graph_manifest(manifest: dict) -> None:
    """Save manifest for currently loaded graph."""
    LOCAL_GRAPH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest["loaded_at"] = datetime.now(UTC).isoformat()
    LOCAL_GRAPH_MANIFEST.write_text(json.dumps(manifest, indent=2))


def show_file_diff(local_path: Path, remote_path: Path) -> bool:
    """Show diff between local and remote file, return True if different."""
    if not local_path.exists():
        click.echo(f"  + {local_path.name} (new file)")
        return True
    if not remote_path.exists():
        click.echo(f"  - {local_path.name} (would be removed)")
        return True

    local_hash = get_file_hash(local_path)
    remote_hash = get_file_hash(remote_path)

    if local_hash == remote_hash:
        click.echo(f"  = {local_path.name} (unchanged)")
        return False

    # Show line count diff
    local_lines = len(local_path.read_text().splitlines())
    remote_lines = len(remote_path.read_text().splitlines())
    diff = remote_lines - local_lines
    diff_str = f"+{diff}" if diff > 0 else str(diff)
    click.echo(
        f"  ~ {local_path.name} ({diff_str} lines, {local_hash} → {remote_hash})"
    )
    return True


def check_graph_exists() -> bool:
    """Check if Neo4j data directory has graph data."""
    data_path = DATA_DIR / "data" / "databases" / "neo4j"
    return data_path.exists() and any(data_path.iterdir())


# ============================================================================
# Main Command Group
# ============================================================================


@click.group()
def data() -> None:
    """Manage graph database and private facility data.

    \b
      imas-codex data dump         Export graph to archive
      imas-codex data load <file>  Load graph archive
      imas-codex data push         Push graph + private YAML (full backup)
      imas-codex data pull         Pull graph + private YAML (full restore)
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
@click.option("--image", envvar="NEO4J_IMAGE", default=None, help="Custom image path")
@click.option("--data-dir", envvar="NEO4J_DATA", default=None, help="Custom data dir")
@click.option("--password", envvar="NEO4J_PASSWORD", default="imas-codex")
@click.option(
    "--minimal", is_flag=True, help="Use minimal service (no resource limits)"
)
def db_service(
    action: str,
    image: str | None,
    data_dir: str | None,
    password: str,
    minimal: bool,
) -> None:
    """Manage Neo4j as a systemd user service.

    Uses the template from imas_codex/config/services/imas-codex-db.service
    which includes cleanup, graceful shutdown, and resource limits.

    The template uses systemd %h specifier for home directory.

    Examples:
        imas-codex data db service install      # Install with defaults
        imas-codex data db service install --minimal  # Basic service
        imas-codex data db service status       # Check service status
        imas-codex data db service uninstall    # Remove service
    """
    import platform

    if platform.system() != "Linux":
        raise click.ClickException("systemd services only supported on Linux")

    if not shutil.which("systemctl"):
        raise click.ClickException("systemctl not found")

    require_apptainer()

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_file = service_dir / "imas-codex-neo4j.service"
    template_file = SERVICES_DIR / "imas-codex-db.service"
    image_path = Path(image) if image else NEO4J_IMAGE
    data_path = Path(data_dir) if data_dir else DATA_DIR
    apptainer_path = shutil.which("apptainer")

    if action == "install":
        if not image_path.exists():
            raise click.ClickException(
                f"Neo4j image not found: {image_path}\n"
                "Pull: apptainer pull docker://neo4j:2025.11-community"
            )

        service_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["data", "logs", "conf", "import"]:
            (data_path / subdir).mkdir(parents=True, exist_ok=True)

        if minimal or not template_file.exists():
            # Minimal service without resource limits
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
            click.echo("Installed minimal service")
        else:
            # Use production template with resource limits
            # Template uses %h for home dir (systemd resolves this)
            shutil.copy(template_file, service_file)
            click.echo(f"Installed from template: {template_file}")
            click.echo("  Includes: cleanup, graceful shutdown, resource limits")

        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "imas-codex-neo4j"], check=True
        )
        click.echo("\n✓ Service installed and enabled")
        click.echo("  Start: systemctl --user start imas-codex-neo4j")
        click.echo("  Or:    imas-codex data db start")

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
        # Update existing gist by cloning, replacing files, and pushing
        click.echo(f"\nUpdating gist {effective_gist_id}...")

        with tempfile.TemporaryDirectory() as tmpdir:
            gist_dir = Path(tmpdir) / "gist"

            # Clone the gist
            result = subprocess.run(
                ["gh", "gist", "clone", effective_gist_id, str(gist_dir)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                click.echo("Gist not found, creating new one...")
                effective_gist_id = None
            else:
                # Set SSH remote URL for push (gh clone uses HTTPS which fails for secret gists)
                subprocess.run(
                    [
                        "git",
                        "remote",
                        "set-url",
                        "origin",
                        f"git@gist.github.com:{effective_gist_id}.git",
                    ],
                    cwd=gist_dir,
                    capture_output=True,
                )

                # Copy local files over (replaces existing)
                for f in private_files:
                    shutil.copy(f, gist_dir / f.name)

                # Commit and push
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=gist_dir,
                    capture_output=True,
                )
                result = subprocess.run(
                    ["git", "commit", "-m", f"Update from imas-codex {__version__}"],
                    cwd=gist_dir,
                    capture_output=True,
                    text=True,
                )
                if "nothing to commit" in result.stdout + result.stderr:
                    click.echo("  No changes to push")
                else:
                    result = subprocess.run(
                        ["git", "push"],
                        cwd=gist_dir,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        raise click.ClickException(f"Push failed: {result.stderr}")

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
@click.option("--url", "gist_url", help="Gist URL (extracts ID and saves for future)")
@click.option("--force", is_flag=True, help="Overwrite without diff/prompt")
@click.option("--no-backup", is_flag=True, help="Skip backup of existing files")
def private_pull(
    gist_id: str | None, gist_url: str | None, force: bool, no_backup: bool
) -> None:
    """Pull private YAML files from GitHub Gist.

    Shows diff and prompts for confirmation before overwriting.
    Use --force to skip prompt, --no-backup to skip backup.

    For onboarding, use --url to specify a gist URL (ID will be saved):
        imas-codex data private pull --url https://gist.github.com/abc123

    Examples:
        imas-codex data private pull
        imas-codex data private pull --url https://gist.github.com/abc123
        imas-codex data private pull --force
    """
    require_gh()

    # Extract gist ID from URL if provided
    if gist_url:
        # URL format: https://gist.github.com/[user/]<gist_id>
        gist_id = gist_url.rstrip("/").split("/")[-1]
        click.echo(f"Extracted gist ID: {gist_id}")
        save_gist_id(gist_id)
        click.echo(f"  Saved to: {GIST_ID_FILE}")

    effective_gist_id = gist_id or get_saved_gist_id()
    if not effective_gist_id:
        raise click.ClickException(
            "No gist ID configured. Either:\n"
            "  1. Run 'imas-codex data private push' first, or\n"
            "  2. Provide --url https://gist.github.com/<id>, or\n"
            "  3. Provide --gist-id, or\n"
            "  4. Set IMAS_PRIVATE_GIST_ID environment variable"
        )

    click.echo(f"Pulling from gist: {effective_gist_id}")

    existing_files = get_private_files()

    # Clone gist to temp dir for comparison
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

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

        # Show diff if not forcing
        if existing_files and not force:
            click.echo("\nChanges:")
            has_changes = False
            for gist_file in yaml_files:
                local_file = PRIVATE_YAML_DIR / gist_file.name
                if show_file_diff(local_file, gist_file):
                    has_changes = True

            # Check for files in local but not in gist
            gist_names = {f.name for f in yaml_files}
            for local_file in existing_files:
                if local_file.name not in gist_names:
                    click.echo(f"  ? {local_file.name} (local only, not in gist)")

            if has_changes:
                click.echo("")
                if not click.confirm("Apply these changes?"):
                    click.echo("Aborted.")
                    return
            else:
                click.echo("\nNo changes to apply.")
                return

        # Backup existing files
        if existing_files and not no_backup:
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            backup_dir = RECOVERY_DIR / f"{timestamp}-private-pull"
            backup_dir.mkdir(parents=True, exist_ok=True)
            for f in existing_files:
                shutil.copy(f, backup_dir / f.name)
            click.echo(f"Backed up {len(existing_files)} files to: {backup_dir}")

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
# Secrets Subgroup (SSH-based transfer)
# ============================================================================

SECRETS_DEFAULT_PROJECT_PATH = "~/Code/imas-codex"


@data.group("secrets")
def data_secrets() -> None:
    """Sync .env between project clones via SSH.

    Transfer .env directly into the imas-codex project directory on remote hosts.
    Uses SSH/SCP with strict file permissions (0600).

    \b
      imas-codex data secrets push iter    Push .env to iter:~/Code/imas-codex
      imas-codex data secrets pull iter    Pull .env from iter:~/Code/imas-codex
      imas-codex data secrets status       Show local .env status

    Default remote path: ~/Code/imas-codex (override with --path)
    """
    pass


def _get_secrets_host() -> str | None:
    """Get saved secrets host."""
    host_file = Path.home() / ".config" / "imas-codex" / "secrets-host"
    if host_file.exists():
        return host_file.read_text().strip()
    return os.environ.get("IMAS_SECRETS_HOST")


def _save_secrets_host(host: str) -> None:
    """Save secrets host."""
    host_file = Path.home() / ".config" / "imas-codex" / "secrets-host"
    host_file.parent.mkdir(parents=True, exist_ok=True)
    host_file.write_text(host)


@data_secrets.command("push")
@click.argument("host", required=False, envvar="IMAS_SECRETS_HOST")
@click.option(
    "--path",
    default=SECRETS_DEFAULT_PROJECT_PATH,
    show_default=True,
    help="Remote project path",
)
@click.option("--dry-run", is_flag=True, help="Show what would be transferred")
def secrets_push(host: str | None, path: str, dry_run: bool) -> None:
    """Push .env to remote project directory.

    HOST is an SSH host alias (from ~/.ssh/config) or user@hostname.
    If omitted, uses previously saved host or IMAS_SECRETS_HOST env var.

    Examples:
        imas-codex data secrets push iter
        imas-codex data secrets push iter --path ~/projects/imas-codex
        imas-codex data secrets push iter --dry-run
    """
    effective_host = host or _get_secrets_host()
    if not effective_host:
        raise click.ClickException(
            "No host specified. Provide HOST argument or set IMAS_SECRETS_HOST"
        )

    env_file = Path(".env")
    if not env_file.exists():
        raise click.ClickException(".env file not found in project root")

    remote_env = f"{path}/.env"
    remote_path = f"{effective_host}:{remote_env}"

    click.echo(f"Push target: {effective_host}")
    click.echo(f"  Remote project: {path}")
    click.echo(f"  File: .env ({env_file.stat().st_size} bytes)")

    if dry_run:
        click.echo("\n[DRY RUN] Would:")
        click.echo(f"  1. Verify {path} exists on {effective_host}")
        click.echo(f"  2. Copy .env to {remote_path}")
        click.echo("  3. Set .env permissions to 0600")
        return

    # Verify remote project exists
    click.echo("\nVerifying remote project...")
    result = subprocess.run(
        ["ssh", effective_host, f"test -d {path} && echo exists"],
        capture_output=True,
        text=True,
    )
    if "exists" not in result.stdout:
        raise click.ClickException(
            f"Project not found on {effective_host}\n"
            f"  Expected: {path}\n"
            f"  Use --path to specify a different location"
        )

    # Copy file
    click.echo("Transferring .env...")
    result = subprocess.run(
        ["scp", "-q", str(env_file), remote_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"Transfer failed: {result.stderr}")

    # Set strict permissions
    result = subprocess.run(
        ["ssh", effective_host, f"chmod 600 {remote_env}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Warning: Could not set permissions: {result.stderr}", err=True)

    # Save host for future use
    if host:
        _save_secrets_host(host)

    click.echo(f"\n✓ Pushed .env to {effective_host}:{path}")


@data_secrets.command("pull")
@click.argument("host", required=False, envvar="IMAS_SECRETS_HOST")
@click.option(
    "--path",
    default=SECRETS_DEFAULT_PROJECT_PATH,
    show_default=True,
    help="Remote project path",
)
@click.option("--force", is_flag=True, help="Overwrite existing .env without prompt")
def secrets_pull(host: str | None, path: str, force: bool) -> None:
    """Pull .env from remote project directory.

    HOST is an SSH host alias (from ~/.ssh/config) or user@hostname.
    If omitted, uses previously saved host or IMAS_SECRETS_HOST env var.

    Examples:
        imas-codex data secrets pull iter
        imas-codex data secrets pull iter --path ~/projects/imas-codex
        imas-codex data secrets pull iter --force
    """
    effective_host = host or _get_secrets_host()
    if not effective_host:
        raise click.ClickException(
            "No host specified. Provide HOST argument or set IMAS_SECRETS_HOST"
        )

    env_file = Path(".env")
    remote_env = f"{path}/.env"
    remote_path = f"{effective_host}:{remote_env}"

    # Check if .env exists locally
    if env_file.exists() and not force:
        click.echo("Local .env exists. Overwrite? (use --force to skip prompt)")
        if not click.confirm("Continue?"):
            return

    click.echo(f"Pulling from: {effective_host}")
    click.echo(f"  Remote project: {path}")

    # Check remote file exists
    result = subprocess.run(
        ["ssh", effective_host, f"test -f {remote_env} && echo exists"],
        capture_output=True,
        text=True,
    )
    if "exists" not in result.stdout:
        raise click.ClickException(
            f"No .env found on {effective_host}\n"
            f"  Expected: {remote_env}\n"
            f"  Use --path to specify a different location"
        )

    # Pull file
    click.echo("Transferring .env...")
    result = subprocess.run(
        ["scp", "-q", remote_path, str(env_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"Transfer failed: {result.stderr}")

    # Set strict local permissions
    env_file.chmod(0o600)

    # Save host for future use
    if host:
        _save_secrets_host(host)

    click.echo(f"\n✓ Pulled .env from {effective_host}:{path}")
    click.echo(f"  Size: {env_file.stat().st_size} bytes")


@data_secrets.command("status")
def secrets_status() -> None:
    """Show local .env status."""
    env_file = Path(".env")
    secrets_host = _get_secrets_host()

    click.echo("Local .env:")
    if env_file.exists():
        stat = env_file.stat()
        perms = oct(stat.st_mode)[-3:]
        click.echo(f"  Size: {stat.st_size} bytes")
        click.echo(f"  Permissions: {perms}")
        if perms != "600":
            click.echo("  ⚠ Permissions should be 600")
    else:
        click.echo("  (not found)")

    click.echo(f"\nDefault remote path: {SECRETS_DEFAULT_PROJECT_PATH}")
    click.echo(f"Last used host: {secrets_host or '(none)'}")


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
@click.option("--no-restart", is_flag=True, help="Don't restart Neo4j after dump")
def data_dump(output: str | None, no_restart: bool) -> None:
    """Export graph database to archive.

    Creates a tarball containing the Neo4j database dump.
    Automatically stops Neo4j, dumps, and restarts.

    Examples:
        imas-codex data dump
        imas-codex data dump -o backup.tar.gz
        imas-codex data dump --no-restart
    """
    require_apptainer()

    git_info = get_git_info()
    version_label = git_info["tag"] or f"dev-{git_info['commit_short']}"

    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"imas-codex-graph-{version_label}.tar.gz")

    with Neo4jOperation("graph dump", require_stopped=True) as op:
        if no_restart:
            op.was_running = False  # Don't restart

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
@click.option("--no-restart", is_flag=True, help="Don't restart Neo4j after load")
def data_load(archive: str, force: bool, no_restart: bool) -> None:
    """Load graph database from archive.

    Automatically stops Neo4j, loads, and restarts.

    Examples:
        imas-codex data load imas-codex-graph-v1.0.0.tar.gz
        imas-codex data load backup.tar.gz --no-restart
    """
    require_apptainer()

    archive_path = Path(archive)
    click.echo(f"Loading archive: {archive_path}")

    with Neo4jOperation("graph load", require_stopped=True) as op:
        if no_restart:
            op.was_running = False

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

            # Save manifest for loaded graph (not pushed until explicitly pushed)
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
                manifest["pushed"] = False  # Local load = needs push to backup
                manifest["loaded_from"] = str(archive_path)
                save_local_graph_manifest(manifest)

    click.echo("✓ Load complete")


@data.command("push")
@click.option("--dev", is_flag=True, help="Push as dev-{commit} tag")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option("--dry-run", is_flag=True, help="Show what would be pushed")
@click.option("--skip-private", is_flag=True, help="Skip private YAML gist sync")
def data_push(
    dev: bool,
    registry: str | None,
    token: str | None,
    dry_run: bool,
    skip_private: bool,
) -> None:
    """Push graph archive to GHCR and private facility configs to Gist.

    Auto-detects fork and pushes to your GHCR registry.
    Also syncs private facility files (*_private.yaml) to GitHub Gist.
    Requires clean git state for release pushes.

    Examples:
        imas-codex data push              # Push release (requires git tag)
        imas-codex data push --dev        # Push dev version
        imas-codex data push --skip-private  # Graph only, no gist sync
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
        click.echo("  1. Dump graph (auto stop/start Neo4j)")
        click.echo(f"  2. Push to {target_registry}/imas-codex-graph:{version_tag}")
        return

    archive_path = Path(f"imas-codex-graph-{version_tag}.tar.gz")

    try:
        # data_dump uses Neo4jOperation internally for stop/start
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

        # Update manifest to mark as pushed
        manifest = get_local_graph_manifest() or {}
        manifest["pushed"] = True
        manifest["pushed_version"] = version_tag
        manifest["pushed_to"] = artifact_ref
        manifest["pushed_at"] = datetime.now(UTC).isoformat()
        save_local_graph_manifest(manifest)

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
        if archive_path.exists():
            archive_path.unlink()

    # Sync private YAML files to Gist
    if not skip_private and not dry_run:
        private_files = get_private_files()
        if private_files:
            click.echo("\nSyncing private YAML to Gist...")
            from click.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(private_push, [])
            if result.exit_code != 0:
                click.echo(f"Warning: Private sync failed: {result.output}", err=True)
            else:
                click.echo("✓ Private YAML synced to Gist")
    elif skip_private:
        click.echo("\nSkipped private YAML sync (--skip-private)")


@data.command("pull")
@click.option("-v", "--version", "version", default="latest")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option("--force", is_flag=True, help="Overwrite existing graph without checks")
@click.option("--no-backup", is_flag=True, help="Skip backup marker")
@click.option("--skip-private", is_flag=True, help="Skip private YAML gist sync")
@click.option("--gist-url", help="Gist URL for private files (for onboarding)")
def data_pull(
    version: str,
    registry: str | None,
    token: str | None,
    force: bool,
    no_backup: bool,
    skip_private: bool,
    gist_url: str | None,
) -> None:
    """Pull graph archive from GHCR and private facility configs from Gist.

    Neo4j is automatically stopped/restarted during load.
    Also pulls private facility files (*_private.yaml) from Gist.

    For onboarding, use --gist-url to configure the private gist:
        imas-codex data pull --gist-url https://gist.github.com/abc123

    Safety: Refuses to overwrite existing graph unless:
    - Graph was previously pushed (tracked in manifest), or
    - --force is specified

    Examples:
        imas-codex data pull                 # Pull latest
        imas-codex data pull -v v1.0.0       # Pull specific version
        imas-codex data pull --gist-url https://gist.github.com/abc123
        imas-codex data pull --force         # Overwrite without checks
    """
    require_oras()

    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    artifact_ref = f"{target_registry}/imas-codex-graph:{version}"

    # Safety check: if graph exists, verify it's been pushed or require --force
    if check_graph_exists() and not force:
        manifest = get_local_graph_manifest()
        if manifest is None:
            # Graph exists but no manifest - unsafe to overwrite
            raise click.ClickException(
                "Local graph exists but has no manifest (unknown origin).\n"
                "Either:\n"
                "  1. Push current graph first: imas-codex data push --dev\n"
                "  2. Use --force to overwrite (data will be lost)"
            )
        elif not manifest.get("pushed"):
            raise click.ClickException(
                f"Local graph (loaded {manifest.get('loaded_at', 'unknown')}) "
                "has not been pushed.\n"
                "Either:\n"
                "  1. Push current graph: imas-codex data push --dev\n"
                "  2. Use --force to overwrite (data will be lost)"
            )
        else:
            pushed_version = manifest.get("pushed_version", "unknown")
            click.echo(f"Local graph was pushed as: {pushed_version}")

    click.echo(f"Pulling: {artifact_ref}")

    if not no_backup:
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

        # Extract manifest from archive and save
        with tarfile.open(archives[0], "r:gz") as tar:
            tar.extractall(tmp / "extracted")
        extracted_dirs = list((tmp / "extracted").iterdir())
        if extracted_dirs:
            manifest_file = extracted_dirs[0] / "manifest.json"
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
                manifest["pulled_from"] = artifact_ref
                manifest["pulled_version"] = version
                manifest["pushed"] = True  # Pulled = already in registry
                manifest["pushed_version"] = version
                save_local_graph_manifest(manifest)

    click.echo("✓ Graph pull complete")

    # Sync private YAML files from Gist
    if not skip_private:
        # If gist_url provided, extract and save ID first
        if gist_url:
            extracted_id = gist_url.rstrip("/").split("/")[-1]
            save_gist_id(extracted_id)
            click.echo(f"\nSaved gist ID: {extracted_id}")

        gist_id = get_saved_gist_id()
        if gist_id or gist_url:
            click.echo("\nPulling private YAML from Gist...")
            from click.testing import CliRunner

            runner = CliRunner()
            args = ["--force"] if force else []
            if gist_url:
                args.extend(["--url", gist_url])
            result = runner.invoke(private_pull, args)
            if result.exit_code != 0:
                click.echo(f"Warning: Private sync failed: {result.output}", err=True)
            else:
                click.echo("✓ Private YAML restored from Gist")
        else:
            click.echo("\nNo Gist configured - skipping private YAML sync")
            click.echo("  Configure with: --gist-url https://gist.github.com/<id>")
            click.echo("  Or run: imas-codex data private push")
    else:
        click.echo("\nSkipped private YAML sync (--skip-private)")


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
