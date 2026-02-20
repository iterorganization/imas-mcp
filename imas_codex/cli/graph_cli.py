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

if TYPE_CHECKING:
    from imas_codex.graph.profiles import Neo4jProfile

# ============================================================================
# Constants and Helpers
# ============================================================================

SERVICES_DIR = Path("imas_codex/config/services")
RECOVERY_DIR = Path.home() / ".local" / "share" / "imas-codex" / "recovery"
DATA_DIR = Path.home() / ".local" / "share" / "imas-codex" / "neo4j"
LOCAL_GRAPH_MANIFEST = Path.home() / ".config" / "imas-codex" / "graph-manifest.json"
NEO4J_LOCK_FILE = Path.home() / ".config" / "imas-codex" / "neo4j-operation.lock"


def _neo4j_image() -> Path:
    """Resolve the Neo4j Apptainer SIF image path."""
    from imas_codex.settings import get_neo4j_image_path

    return get_neo4j_image_path()


# Keep module-level name for backward compat with CLI --image defaults
NEO4J_IMAGE = _neo4j_image()


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
    - Resetting password after database load (which clears auth)
    """

    def __init__(
        self,
        operation_name: str,
        require_stopped: bool = True,
        reset_password_on_restart: bool = False,
        password: str | None = None,
    ):
        from imas_codex.graph.profiles import DEFAULT_PASSWORD, resolve_neo4j

        self.profile = resolve_neo4j()
        self.operation_name = operation_name
        self.require_stopped = require_stopped
        self.reset_password_on_restart = reset_password_on_restart
        self.password = password or DEFAULT_PASSWORD
        self.acquired = False
        self.was_running = False
        self._lock_file: Path = NEO4J_LOCK_FILE

    def __enter__(self) -> Neo4jOperation:
        # Check for existing lock
        if self._lock_file.exists():
            lock_info = json.loads(self._lock_file.read_text())
            pid = lock_info.get("pid")
            operation = lock_info.get("operation", "unknown")
            started = lock_info.get("started", "unknown")

            if pid and self._process_exists(pid):
                raise click.ClickException(
                    f"Another operation is in progress:\n"
                    f"  Operation: {operation}\n"
                    f"  Started: {started}\n"
                    f"  PID: {pid}\n\n"
                    f"If the process crashed, remove lock: rm {self._lock_file}"
                )
            else:
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
        if self.require_stopped and is_neo4j_running(self.profile.http_port):
            self.was_running = True
            click.echo(
                f"Stopping Neo4j [{self.profile.name}] for {self.operation_name}..."
            )
            self._stop_neo4j()

            import time

            for _ in range(30):
                if not is_neo4j_running(self.profile.http_port):
                    break
                time.sleep(1)
            else:
                self._release_lock()
                raise click.ClickException(
                    f"Failed to stop Neo4j [{self.profile.name}] within 30 seconds"
                )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            # Always reset password after load operations, regardless of whether
            # Neo4j was running before. The dump replaces the auth database.
            if self.reset_password_on_restart:
                click.echo("Resetting Neo4j password after load...")
                self._reset_password()

            if self.was_running:
                click.echo("Restarting Neo4j...")
                self._start_neo4j()
        finally:
            self._release_lock()
        return False

    def _reset_password(self) -> None:
        """Reset Neo4j password after database load.

        After loading a database dump, Neo4j's auth database may be
        replaced.  Clear the auth file first so ``set-initial-password``
        always succeeds regardless of prior auth state.
        """
        auth_file = self.profile.data_dir / "data" / "dbms" / "auth.ini"
        if auth_file.exists():
            auth_file.unlink()
        cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{self.profile.data_dir}/data:/data",
            "--writable-tmpfs",
            str(NEO4J_IMAGE),
            "neo4j-admin",
            "dbms",
            "set-initial-password",
            self.password,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if "already set" not in result.stderr.lower():
                click.echo(f"Warning: Password reset issue: {result.stderr.strip()}")
        else:
            click.echo("  Password reset successful")

    def _process_exists(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _stop_neo4j(self) -> None:
        service_name = f"imas-codex-neo4j-{self.profile.name}"
        result = subprocess.run(
            ["systemctl", "--user", "stop", service_name],
            capture_output=True,
        )
        if result.returncode == 0:
            return

        subprocess.run(
            ["pkill", "-15", "-f", "neo4j_.*community.sif"],
            capture_output=True,
        )
        subprocess.run(
            ["pkill", "-15", "-f", "Neo4jCommunity"],
            capture_output=True,
        )
        subprocess.run(["pkill", "-15", "-f", "neo4j.*console"], capture_output=True)

    def _start_neo4j(self) -> None:
        service_name = f"imas-codex-neo4j-{self.profile.name}"
        result = subprocess.run(
            ["systemctl", "--user", "start", service_name],
            capture_output=True,
        )
        if result.returncode == 0:
            import time

            for _ in range(30):
                if is_neo4j_running(self.profile.http_port):
                    click.echo(f"Neo4j [{self.profile.name}] ready")
                    return
                time.sleep(1)
            click.echo(f"Warning: Neo4j [{self.profile.name}] may still be starting")
            return

        click.echo("Note: Restart Neo4j manually: imas-codex graph start")

    def _release_lock(self) -> None:
        if self._lock_file.exists():
            self._lock_file.unlink()
        self.acquired = False


# ============================================================================
# Helper Functions
# ============================================================================


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

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if result.returncode == 0:
        info["commit"] = result.stdout.strip()
        info["commit_short"] = info["commit"][:7]

    result = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        info["tag"] = result.stdout.strip()

    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    info["is_dirty"] = bool(result.stdout.strip())

    result = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, text=True
    )
    if result.returncode == 0:
        url = result.stdout.strip()
        info["remote_url"] = url
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

    For dev pushes, auto-increments a revision counter per base version
    so that successive pushes from the same commit produce unique tags:
      0.5.0.dev123-abc1234-r1, 0.5.0.dev123-abc1234-r2, ...

    The revision counter is tracked in the local graph manifest and
    resets when the base version changes (new commit or version bump).
    """
    if dev:
        base = __version__.replace("+", "-")
        revision = _next_dev_revision(base)
        return f"{base}-r{revision}"
    if git_info["tag"]:
        return git_info["tag"]
    raise click.ClickException(
        "Not on a git tag. Use --dev for development push, or create a tag first."
    )


def _next_dev_revision(base_version: str) -> int:
    """Get next revision number for a dev push.

    Reads the graph manifest to find the last pushed revision for this
    base version. Returns last_revision + 1, or 1 if no previous push.
    """
    manifest = get_local_graph_manifest()
    if manifest:
        last_base = manifest.get("dev_base_version")
        last_rev = manifest.get("dev_revision", 0)
        if last_base == base_version:
            return last_rev + 1
    return 1


def _save_dev_revision(base_version: str, revision: int) -> None:
    """Save the current dev revision to the graph manifest."""
    manifest = get_local_graph_manifest() or {}
    manifest["dev_base_version"] = base_version
    manifest["dev_revision"] = revision
    save_local_graph_manifest(manifest)


def require_clean_git(git_info: dict) -> None:
    if git_info["is_dirty"]:
        raise click.ClickException(
            "Working tree has uncommitted changes. Commit or stash first."
        )


def require_oras() -> None:
    if not shutil.which("oras"):
        raise click.ClickException(
            "oras not found in PATH. Install from: "
            "https://github.com/oras-project/oras/releases"
        )


def require_apptainer() -> None:
    if not shutil.which("apptainer"):
        raise click.ClickException("apptainer not found in PATH")


def require_gh() -> None:
    if not shutil.which("gh"):
        raise click.ClickException(
            "gh CLI not found. Install from: https://cli.github.com/"
        )


def is_neo4j_running(http_port: int | None = None) -> bool:
    from imas_codex.graph.profiles import HTTP_BASE_PORT

    if http_port is None:
        http_port = HTTP_BASE_PORT
    try:
        import urllib.request

        urllib.request.urlopen(f"http://localhost:{http_port}/", timeout=2)
        return True
    except Exception:
        return False


def check_stale_neo4j_process(data_dir: Path) -> tuple[bool, str | None]:
    """Check for stale Neo4j processes that may hold locks.

    Returns (has_stale, message) where has_stale indicates a hung/stale
    process was detected and message describes the issue.
    """
    pid_file = data_dir / "neo4j.pid"
    if not pid_file.exists():
        return False, None

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return False, None

    # Check if process exists
    try:
        os.kill(pid, 0)  # Signal 0 just checks if process exists
    except ProcessLookupError:
        # Process gone, clean up stale PID file
        try:
            pid_file.unlink()
        except OSError:
            pass
        return False, None
    except PermissionError:
        # Process exists but owned by another user
        return True, f"Neo4j process (PID {pid}) exists but is owned by another user"

    # Process exists - check if it's responding
    # Read http_port from proc cmdline if possible
    try:
        cmdline_file = Path(f"/proc/{pid}/cmdline")
        if cmdline_file.exists():
            # Process is running - we confirmed it exists
            return False, None
    except (OSError, PermissionError):
        pass

    return False, None


def check_database_locks(data_dir: Path) -> tuple[bool, str | None]:
    """Check for database lock files that indicate another process.

    Returns (has_lock, message).
    """
    lock_file = data_dir / "data" / "databases" / "store_lock"
    if lock_file.exists():
        mtime = datetime.fromtimestamp(lock_file.stat().st_mtime, tz=UTC)
        age_hours = (datetime.now(tz=UTC) - mtime).total_seconds() / 3600
        if age_hours > 24:
            return True, (
                f"Stale database lock file detected (age: {age_hours:.1f}h).\n"
                f"This may indicate a previous Neo4j crash.\n"
                f"If Neo4j won't start, try: rm {lock_file}"
            )
    return False, None


def secure_data_directory(data_path: Path) -> None:
    """Set restrictive permissions on Neo4j data directory.

    Ensures only the owner can access the database files, preventing
    accidental conflicts on shared filesystems where multiple users
    might run their own Neo4j instances.

    Sets directories to 700 (rwx------) and files to 600 (rw-------).
    """
    import stat

    if not data_path.exists():
        return

    # Secure the root directory
    try:
        data_path.chmod(stat.S_IRWXU)  # 700
    except OSError:
        pass  # May fail if we don't own it

    # Recursively secure subdirectories and files
    for item in data_path.rglob("*"):
        try:
            if item.is_dir():
                item.chmod(stat.S_IRWXU)  # 700
            else:
                item.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
        except OSError:
            pass  # Skip files we can't chmod


def get_package_name(
    facilities: list[str] | None = None,
    *,
    no_imas: bool = False,
    imas_only: bool = False,
) -> str:
    """Get the GHCR package name, optionally scoped to facilities.

    Args:
        facilities: If given, appends sorted facility IDs to the name.
        no_imas: If True, appends ``-no-imas`` suffix.
        imas_only: If True, uses ``imas-codex-graph-imas`` (DD-only graph).

    Returns:
        Package name, e.g. ``"imas-codex-graph-iter-tcv-no-imas"``.
    """
    if imas_only:
        return "imas-codex-graph-imas"
    parts = ["imas-codex-graph"]
    if facilities:
        parts.extend(sorted(facilities))
    if no_imas:
        parts.append("no-imas")
    return "-".join(parts)


def backup_existing_data(reason: str, data_dir: Path | None = None) -> Path | None:
    """Create a marker for pre-operation recovery (lightweight)."""
    target = data_dir or DATA_DIR
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    recovery_path = RECOVERY_DIR / f"{timestamp}-{reason}"

    if target.exists():
        recovery_path.mkdir(parents=True, exist_ok=True)
        (recovery_path / "graph_data_existed.marker").touch()
        return recovery_path

    return None


def backup_graph_dump(
    output: Path | None = None,
) -> Path:
    """Create a real neo4j-admin dump backup of the current graph.

    Args:
        output: Output path override.  Defaults to
            ``~/.local/share/imas-codex/backups/{profile}-{timestamp}.dump``.

    Returns:
        Path to the created dump file.
    """
    from imas_codex.graph.profiles import BACKUPS_DIR, resolve_neo4j

    profile = resolve_neo4j()
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    dump_path = output or (BACKUPS_DIR / f"{profile.name}-{timestamp}.dump")

    dumps_dir = profile.data_dir / "dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)

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
        "dump",
        "neo4j",
        "--to-path=/dumps",
        "--overwrite-destination=true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"Backup dump failed: {result.stderr}")

    neo4j_dump = dumps_dir / "neo4j.dump"
    if neo4j_dump.exists():
        shutil.move(str(neo4j_dump), str(dump_path))
    else:
        raise click.ClickException("Dump file not created by neo4j-admin")

    return dump_path


def login_to_ghcr(token: str | None) -> None:
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


def get_local_graph_manifest() -> dict | None:
    if LOCAL_GRAPH_MANIFEST.exists():
        return json.loads(LOCAL_GRAPH_MANIFEST.read_text())
    return None


def save_local_graph_manifest(manifest: dict) -> None:
    LOCAL_GRAPH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest["loaded_at"] = datetime.now(UTC).isoformat()
    LOCAL_GRAPH_MANIFEST.write_text(json.dumps(manifest, indent=2))


def check_graph_exists(data_dir: Path | None = None) -> bool:
    target = data_dir or DATA_DIR
    data_path = target / "data" / "databases" / "neo4j"
    return data_path.exists() and any(data_path.iterdir())


# ============================================================================
# Main Command Group
# ============================================================================


@click.group()
def graph() -> None:
    """Manage graph database lifecycle.

    \b
    Setup:
      imas-codex graph init NAME           Create a new graph
      imas-codex graph status              Show graph and server status
      imas-codex graph list                List local graph instances
      imas-codex graph switch NAME         Activate a different graph

    \b
    Server:
      imas-codex graph start               Start Neo4j
      imas-codex graph stop                Stop Neo4j
      imas-codex graph shell               Interactive Cypher REPL
      imas-codex graph service install     Install systemd service

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
    """Start Neo4j server via Apptainer."""
    import platform

    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()
    password = password or profile.password

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
        f"NEO4J_server_bolt_listen__address=127.0.0.1:{profile.bolt_port}",
        "--env",
        f"NEO4J_server_http_listen__address=127.0.0.1:{profile.http_port}",
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
    """Stop Neo4j server."""
    import signal

    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()

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
    """Show local and registry status."""
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

    click.echo(f"\nNeo4j: {'running' if is_neo4j_running() else 'stopped'}")

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

    if is_neo4j_running():
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


@graph.group("service")
def graph_service_group() -> None:
    """Manage Neo4j systemd service.

    \\b
      imas-codex graph service install   Install systemd unit
      imas-codex graph service start     Start service
      imas-codex graph service stop      Stop service
      imas-codex graph service status    Check service status
    """
    pass


@graph_service_group.command("install")
@click.option("--image", envvar="NEO4J_IMAGE", default=None)
@click.option("--data-dir", envvar="NEO4J_DATA", default=None)
@click.option("--password", envvar="NEO4J_PASSWORD", default=None)
@click.option(
    "--minimal", is_flag=True, help="Use minimal service (no resource limits)"
)
def graph_service_install(
    image: str | None,
    data_dir: str | None,
    password: str | None,
    minimal: bool,
) -> None:
    """Install Neo4j as a systemd user service."""
    import platform

    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()
    password = password or profile.password

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        raise click.ClickException(
            f"Service install must be run directly on {profile.host}.\n"
            f"SSH in and run:\n"
            f"  ssh {profile.host}\n"
            f"  cd ~/Code/imas-codex && uv run imas-codex graph service install"
        )
    # ── End remote dispatch ──────────────────────────────────────────────

    if platform.system() != "Linux":
        raise click.ClickException("systemd services only supported on Linux")

    if not shutil.which("systemctl"):
        raise click.ClickException("systemctl not found")

    require_apptainer()

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_name = f"imas-codex-neo4j-{profile.name}"
    service_file = service_dir / f"{service_name}.service"
    template_file = SERVICES_DIR / "imas-codex-db.service"
    image_path = Path(image) if image else NEO4J_IMAGE
    data_path = Path(data_dir) if data_dir else profile.data_dir
    apptainer_path = shutil.which("apptainer")

    from imas_codex.graph.profiles import check_graph_conflict

    conflict = check_graph_conflict(profile.bolt_port)
    if conflict:
        raise click.ClickException(conflict)

    if not image_path.exists():
        from imas_codex.settings import get_neo4j_version

        raise click.ClickException(
            f"Neo4j image not found: {image_path}\n"
            f"Pull: apptainer pull docker://neo4j:{get_neo4j_version()}"
        )

    service_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["data", "logs", "conf", "import"]:
        (data_path / subdir).mkdir(parents=True, exist_ok=True)

    if minimal or not template_file.exists():
        service_content = f"""[Unit]
Description=Neo4j Graph Database - {profile.name} (IMAS Codex)
After=network.target

[Service]
Type=simple
ExecStart={apptainer_path} exec \\
    --bind {data_path}/data:/data \\
    --bind {data_path}/logs:/logs \\
    --bind {data_path}/import:/import \\
    --writable-tmpfs \\
    --env NEO4J_server_bolt_listen__address=127.0.0.1:{profile.bolt_port} \\
    --env NEO4J_server_http_listen__address=127.0.0.1:{profile.http_port} \\
    {image_path} \\
    neo4j console
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
        service_file.write_text(service_content)
        click.echo(f"Installed minimal service for [{profile.name}]")
    else:
        shutil.copy(template_file, service_file)
        click.echo(f"Installed from template: {template_file}")
        click.echo("  Includes: cleanup, graceful shutdown, resource limits")

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", service_name], check=True)
    click.echo(f"\n✓ Service [{profile.name}] installed and enabled")
    click.echo(
        f"  Bolt: localhost:{profile.bolt_port}  HTTP: localhost:{profile.http_port}"
    )
    click.echo(f"  Start: systemctl --user start {service_name}")
    click.echo("  Or:    imas-codex graph service start")


@graph_service_group.command("start")
def graph_service_start() -> None:
    """Start Neo4j systemd service."""
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j(auto_tunnel=False)
    service_name = f"imas-codex-neo4j-{profile.name}"

    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            remote_service_action,
            resolve_remote_service_name,
        )

        service_name = resolve_remote_service_name(profile.name, profile.host)
        remote_service_action("start", service_name, profile.host)
        click.echo(f"Started {service_name} on {profile.host}")
        return

    result = subprocess.run(
        ["systemctl", "--user", "start", service_name], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise click.ClickException(f"Failed to start service: {result.stderr}")
    click.echo(f"Started {service_name}")


@graph_service_group.command("stop")
def graph_service_stop() -> None:
    """Stop Neo4j systemd service."""
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j(auto_tunnel=False)
    service_name = f"imas-codex-neo4j-{profile.name}"

    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            remote_service_action,
            resolve_remote_service_name,
        )

        service_name = resolve_remote_service_name(profile.name, profile.host)
        remote_service_action("stop", service_name, profile.host)
        click.echo(f"Stopped {service_name} on {profile.host}")
        return

    result = subprocess.run(
        ["systemctl", "--user", "stop", service_name], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise click.ClickException(f"Failed to stop service: {result.stderr}")
    click.echo(f"Stopped {service_name}")


@graph_service_group.command("status")
def graph_service_status() -> None:
    """Check Neo4j systemd service status."""
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j(auto_tunnel=False)
    service_name = f"imas-codex-neo4j-{profile.name}"

    from imas_codex.graph.remote import is_remote_location

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import resolve_remote_service_name
        from imas_codex.remote.executor import run_command

        service_name = resolve_remote_service_name(profile.name, profile.host)
        output = run_command(
            f"systemctl --user status {service_name}",
            ssh_host=profile.host,
            timeout=15,
        )
        click.echo(output)
        return

    subprocess.run(["systemctl", "--user", "status", service_name])


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


# ============================================================================
# DD node labels to keep for --imas-only exports
_IMAS_DD_LABELS = [
    "DDVersion",
    "IDS",
    "IMASPath",
    "IMASCoordinateSpec",
    "IMASSemanticCluster",
    "IdentifierSchema",
    "IMASPathChange",
    "CoordinateRelationship",
    "ClusterMembership",
    "EmbeddingChange",
    "Unit",
    "PhysicsDomain",
    "SignConvention",
]


# ============================================================================
# Graph Data Operations
# ============================================================================


def _create_imas_only_dump(source_dump_path: Path, output_path: Path) -> None:
    """Create an IMAS-only dump by filtering out facility nodes.

    Loads the full dump into a temporary Neo4j instance, deletes all
    nodes that are not IMAS Data Dictionary types, then dumps the
    filtered graph.
    """
    import time
    import urllib.request

    temp_bolt_port = 27687
    temp_http_port = 27474

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir) / "neo4j-temp"
        for subdir in ("data", "logs", "dumps"):
            (temp_dir / subdir).mkdir(parents=True)

        shutil.copy(source_dump_path, temp_dir / "dumps" / "neo4j.dump")

        click.echo("  Loading full dump into temp instance for IMAS-only filtering...")

        load_cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{temp_dir}/data:/data",
            "--bind",
            f"{temp_dir}/dumps:/dumps",
            "--writable-tmpfs",
            str(NEO4J_IMAGE),
            "neo4j-admin",
            "database",
            "load",
            "neo4j",
            "--from-path=/dumps",
            "--overwrite-destination=true",
        ]
        result = subprocess.run(load_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(
                f"Failed to load dump into temp instance: {result.stderr}"
            )

        pw_cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{temp_dir}/data:/data",
            "--writable-tmpfs",
            str(NEO4J_IMAGE),
            "neo4j-admin",
            "dbms",
            "set-initial-password",
            "temp-password",
        ]
        subprocess.run(pw_cmd, capture_output=True, text=True)

        click.echo("  Starting temp Neo4j instance...")
        start_cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{temp_dir}/data:/data",
            "--bind",
            f"{temp_dir}/logs:/logs",
            "--writable-tmpfs",
            "--env",
            "NEO4J_AUTH=neo4j/temp-password",
            "--env",
            f"NEO4J_server_bolt_listen__address=127.0.0.1:{temp_bolt_port}",
            "--env",
            f"NEO4J_server_http_listen__address=127.0.0.1:{temp_http_port}",
            str(NEO4J_IMAGE),
            "neo4j",
            "console",
        ]
        proc = subprocess.Popen(
            start_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        try:
            ready = False
            for _ in range(60):
                try:
                    urllib.request.urlopen(
                        f"http://localhost:{temp_http_port}/", timeout=2
                    )
                    ready = True
                    break
                except Exception:
                    time.sleep(1)

            if not ready:
                raise click.ClickException(
                    "Temp Neo4j instance did not start within 60 seconds"
                )

            click.echo("  Filtering graph: keeping only IMAS DD nodes...")

            from neo4j import GraphDatabase

            label_check = " AND ".join(f"NOT n:{label}" for label in _IMAS_DD_LABELS)
            driver = GraphDatabase.driver(
                f"bolt://localhost:{temp_bolt_port}",
                auth=("neo4j", "temp-password"),
            )
            with driver.session() as session:
                result = session.run(
                    f"MATCH (n) WHERE {label_check} "
                    "DETACH DELETE n "
                    "RETURN count(*) AS deleted"
                )
                deleted = result.single()["deleted"]
                click.echo(f"    Removed {deleted} non-DD nodes")

            driver.close()

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        click.echo("  Dumping filtered graph...")
        (temp_dir / "dumps" / "neo4j.dump").unlink(missing_ok=True)

        dump_cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{temp_dir}/data:/data",
            "--bind",
            f"{temp_dir}/dumps:/dumps",
            "--writable-tmpfs",
            str(NEO4J_IMAGE),
            "neo4j-admin",
            "database",
            "dump",
            "neo4j",
            "--to-path=/dumps",
            "--overwrite-destination=true",
        ]
        result = subprocess.run(dump_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(
                f"Failed to dump filtered graph: {result.stderr}"
            )

        filtered_dump = temp_dir / "dumps" / "neo4j.dump"
        if not filtered_dump.exists():
            raise click.ClickException("Filtered dump file not created")

        shutil.move(str(filtered_dump), str(output_path))
        size_mb = output_path.stat().st_size / 1024 / 1024
        click.echo(f"    IMAS-only dump: {size_mb:.1f} MB")


def _create_facility_dump(
    source_dump_path: Path, facility: str, output_path: Path
) -> None:
    """Create a per-facility dump by filtering a full graph dump.

    Loads the full dump into a temporary Neo4j instance, deletes nodes
    belonging to other facilities and orphaned non-DD nodes, then dumps
    the filtered graph.

    Args:
        source_dump_path: Path to the full ``neo4j.dump`` file.
        facility: Facility ID to keep (e.g. ``"tcv"``).
        output_path: Where to write the filtered dump file.
    """
    import time
    import urllib.request

    temp_bolt_port = 27687
    temp_http_port = 27474

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir) / "neo4j-temp"
        for subdir in ("data", "logs", "dumps"):
            (temp_dir / subdir).mkdir(parents=True)

        # Copy source dump into temp dumps dir
        shutil.copy(source_dump_path, temp_dir / "dumps" / "neo4j.dump")

        click.echo(
            f"  Loading full dump into temp instance for {facility} filtering..."
        )

        # Load dump into temp data dir
        load_cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{temp_dir}/data:/data",
            "--bind",
            f"{temp_dir}/dumps:/dumps",
            "--writable-tmpfs",
            str(NEO4J_IMAGE),
            "neo4j-admin",
            "database",
            "load",
            "neo4j",
            "--from-path=/dumps",
            "--overwrite-destination=true",
        ]
        result = subprocess.run(load_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(
                f"Failed to load dump into temp instance: {result.stderr}"
            )

        # Set initial password for temp instance
        pw_cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{temp_dir}/data:/data",
            "--writable-tmpfs",
            str(NEO4J_IMAGE),
            "neo4j-admin",
            "dbms",
            "set-initial-password",
            "temp-password",
        ]
        subprocess.run(pw_cmd, capture_output=True, text=True)

        # Start temp Neo4j
        click.echo("  Starting temp Neo4j instance...")
        start_cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{temp_dir}/data:/data",
            "--bind",
            f"{temp_dir}/logs:/logs",
            "--writable-tmpfs",
            "--env",
            "NEO4J_AUTH=neo4j/temp-password",
            "--env",
            f"NEO4J_server_bolt_listen__address=127.0.0.1:{temp_bolt_port}",
            "--env",
            f"NEO4J_server_http_listen__address=127.0.0.1:{temp_http_port}",
            str(NEO4J_IMAGE),
            "neo4j",
            "console",
        ]
        proc = subprocess.Popen(
            start_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        try:
            # Wait for temp instance to be ready
            ready = False
            for _ in range(60):
                try:
                    urllib.request.urlopen(
                        f"http://localhost:{temp_http_port}/", timeout=2
                    )
                    ready = True
                    break
                except Exception:
                    time.sleep(1)

            if not ready:
                raise click.ClickException(
                    "Temp Neo4j instance did not start within 60 seconds"
                )

            click.echo(f"  Filtering graph: keeping facility={facility}...")

            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                f"bolt://localhost:{temp_bolt_port}",
                auth=("neo4j", "temp-password"),
            )
            with driver.session() as session:
                # Delete non-target facility nodes
                result = session.run(
                    "MATCH (n) "
                    "WHERE n.facility_id IS NOT NULL "
                    "AND n.facility_id <> $facility "
                    "DETACH DELETE n "
                    "RETURN count(*) AS deleted",
                    facility=facility,
                )
                deleted_facility = result.single()["deleted"]
                click.echo(f"    Removed {deleted_facility} non-{facility} nodes")

                # Delete orphan nodes (no relationships, not IMAS DD types)
                result = session.run(
                    "MATCH (n) WHERE NOT (n)--() "
                    "AND NOT n:IMASPath AND NOT n:DDVersion AND NOT n:Unit "
                    "AND NOT n:IMASCoordinateSpec AND NOT n:PhysicsDomain "
                    "AND NOT n:IMASSemanticCluster "
                    "DELETE n "
                    "RETURN count(*) AS deleted"
                )
                deleted_orphans = result.single()["deleted"]
                click.echo(f"    Removed {deleted_orphans} orphan nodes")

            driver.close()

        finally:
            # Stop temp Neo4j
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # Dump filtered graph from temp instance
        click.echo("  Dumping filtered graph...")
        # Clear old dump file
        (temp_dir / "dumps" / "neo4j.dump").unlink(missing_ok=True)

        dump_cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{temp_dir}/data:/data",
            "--bind",
            f"{temp_dir}/dumps:/dumps",
            "--writable-tmpfs",
            str(NEO4J_IMAGE),
            "neo4j-admin",
            "database",
            "dump",
            "neo4j",
            "--to-path=/dumps",
            "--overwrite-destination=true",
        ]
        result = subprocess.run(dump_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(
                f"Failed to dump filtered graph: {result.stderr}"
            )

        filtered_dump = temp_dir / "dumps" / "neo4j.dump"
        if not filtered_dump.exists():
            raise click.ClickException("Filtered dump file not created")

        shutil.move(str(filtered_dump), str(output_path))
        size_mb = output_path.stat().st_size / 1024 / 1024
        click.echo(f"    Filtered dump: {size_mb:.1f} MB")


# ============================================================================
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
    "-f",
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
def graph_export(
    output: str | None,
    no_restart: bool,
    facilities: tuple[str, ...],
    no_imas: bool,
    imas_only: bool,
    local: bool,
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

                # Clean up remote archive after successful transfer
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
                "dump",
                "neo4j",
                "--to-path=/dumps",
                "--overwrite-destination=true",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise click.ClickException(f"Graph dump failed: {result.stderr}")

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


# ============================================================================
# Embedding Update (pre-push hook)
# ============================================================================


def _dispatch_graph_quality(git_info: dict, version_tag: str, registry: str) -> None:
    """Fire a repository_dispatch event to trigger graph quality CI.

    Uses the GitHub CLI (gh) to dispatch a graph-pushed event.
    Silently skips if gh is not available or the dispatch fails.
    """
    if not shutil.which("gh"):
        return

    owner = git_info.get("remote_owner", "iterorganization")
    repo = "imas-codex"

    payload = json.dumps(
        {
            "tag": version_tag,
            "registry": registry,
            "commit": git_info.get("commit", ""),
        }
    )

    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{owner}/{repo}/dispatches",
                "-f",
                "event_type=graph-pushed",
                "-f",
                f"client_payload={payload}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            click.echo("✓ Dispatched graph-quality CI")
        else:
            click.echo(
                f"Warning: graph-quality dispatch failed: {result.stderr.strip()}",
                err=True,
            )
    except (subprocess.TimeoutExpired, Exception) as e:
        click.echo(f"Warning: graph-quality dispatch skipped: {e}", err=True)


@graph.command("push")
@click.option("--dev", is_flag=True, help="Push as dev-{commit} tag")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option("--dry-run", is_flag=True, help="Show what would be pushed")
@click.option(
    "--facility",
    "-f",
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
def graph_push(
    dev: bool,
    registry: str | None,
    token: str | None,
    dry_run: bool,
    facilities: tuple[str, ...],
    no_imas: bool,
    imas_only: bool,
    message: str | None,
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
    version_tag = get_version_tag(git_info, dev)
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
            remote_check_oras,
        )

        if not remote_check_oras(profile.host):
            raise click.ClickException(
                f"oras not found on {profile.host}. "
                "Install with: imas-codex tools install"
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
            result = runner.invoke(graph_export, dump_args)
            if result.exit_code != 0:
                if result.exception and not isinstance(result.exception, SystemExit):
                    detail = f"{type(result.exception).__name__}: {result.exception}"
                else:
                    detail = result.output.strip()
                gp.fail_phase(detail)
                raise click.ClickException(f"Export failed: {detail}")
            size_mb = archive_path.stat().st_size / 1024 / 1024
            gp.complete_phase(f"{size_mb:.1f} MB")

            login_to_ghcr(token)

            artifact_ref = f"{target_registry}/{pkg_name}:{version_tag}"
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
    "-f",
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
    "-f",
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
            load_output = remote_operation_streaming(
                load_script,
                profile.host,
                progress=gp,
                progress_markers=_remote_markers_load,
                timeout=600,
            )
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
            remote_is_neo4j_running,
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
        find_graph,
        get_active_graph,
        switch_active_graph,
    )

    try:
        target = find_graph(name)
    except LookupError as e:
        raise click.ClickException(str(e)) from e

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
        f"NEO4J_server_bolt_listen__address=127.0.0.1:{profile.bolt_port}",
        "--env",
        f"NEO4J_server_http_listen__address=127.0.0.1:{profile.http_port}",
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
    "-f",
    "facilities",
    multiple=True,
    required=True,
    help="Facility ID to include (repeatable)",
)
@click.option("--force", is_flag=True, help="Allow using an existing directory")
def graph_init(name: str, facilities: tuple[str, ...], force: bool) -> None:
    """Initialize a new graph instance.

    Creates a name-based directory in .neo4j/<NAME>/, points the neo4j/
    symlink to it, starts Neo4j, and initializes the (:GraphMeta) node.

    Works on both local and remote graph locations.  When the configured
    location is remote, directories are created via SSH and the service
    is started via systemctl.

    \b
    Examples:
      imas-codex graph init codex -f iter -f tcv -f jt-60sa
      imas-codex graph init dev -f tcv
    """
    from imas_codex.graph.profiles import resolve_neo4j

    facility_list = sorted(set(facilities))

    # ── Remote dispatch ──────────────────────────────────────────────────
    from imas_codex.graph.remote import is_remote_location

    profile = resolve_neo4j(auto_tunnel=False)

    if is_remote_location(profile.host):
        from imas_codex.graph.remote import (
            remote_create_graph_dir,
            remote_is_legacy_data_dir,
            remote_is_neo4j_running,
            remote_service_action,
            remote_set_initial_password,
            remote_switch_active_graph,
            resolve_remote_service_name,
        )

        if remote_is_legacy_data_dir(profile.host):
            raise click.ClickException(
                f"neo4j/ on {profile.host} is a real directory (pre-migration).\n"
                f"Migrate manually via SSH:\n"
                f"  ssh {profile.host} 'cd ~/.local/share/imas-codex && "
                f"mkdir -p .neo4j && mv neo4j .neo4j/{name} && "
                f"ln -s .neo4j/{name} neo4j'"
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
            init_graph_meta(gc, name, facility_list)
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
        is_legacy_data_dir,
        switch_active_graph,
    )

    if is_legacy_data_dir():
        raise click.ClickException(
            "neo4j/ exists as a real directory (pre-migration layout).\n"
            "Move it manually into .neo4j/ first:\n"
            f"  mkdir -p ~/.local/share/imas-codex/.neo4j\n"
            f"  mv ~/.local/share/imas-codex/neo4j "
            f"~/.local/share/imas-codex/.neo4j/{name}\n"
            f"  ln -s .neo4j/{name} ~/.local/share/imas-codex/neo4j"
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
        init_graph_meta(gc, name, facility_list)
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
# Delete and Cleanup Commands
# ============================================================================


def _resolve_latest_tag(
    registry: str,
    token: str | None = None,
    pkg_name: str = "imas-codex-graph",
) -> str:
    """Resolve the most recent tag when 'latest' doesn't exist.

    Checks for 'latest' first. If not found, picks the most recent tag
    by sorting: release tags (semver) first, then dev tags by revision.
    """
    tags = _list_registry_tags(registry, token, pkg_name)
    if not tags:
        raise click.ClickException(
            f"No graph versions found in {registry}/{pkg_name}.\n"
            "Push a graph first: imas-codex graph push --dev"
        )

    if "latest" in tags:
        return "latest"

    # Sort: prefer release tags (no 'dev'), then by revision number descending
    def _tag_sort_key(tag: str) -> tuple[int, int]:
        is_dev = 1 if ("dev" in tag or "-r" in tag) else 0
        rev = 0
        if "-r" in tag:
            try:
                rev = int(tag.rsplit("-r", 1)[-1])
            except ValueError:
                pass
        return (is_dev, -rev)

    tags.sort(key=_tag_sort_key)
    best = tags[0]
    click.echo(f"No 'latest' tag found. Using most recent: {best}")
    return best


def _list_registry_tags(
    registry: str,
    token: str | None = None,
    pkg_name: str = "imas-codex-graph",
) -> list[str]:
    """List all tags in the GHCR registry."""
    login_to_ghcr(token)
    repo_ref = f"{registry}/{pkg_name}"
    result = subprocess.run(
        ["oras", "repo", "tags", repo_ref],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "not found" in result.stderr.lower():
            return []
        raise click.ClickException(f"Failed to list tags: {result.stderr}")
    return [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]


def _fetch_tag_messages(
    registry: str,
    tags: list[str],
    *,
    pkg_name: str = "imas-codex-graph",
) -> dict[str, str | None]:
    """Fetch the push message for each tag from OCI manifest annotations.

    Returns a mapping of tag -> message (None if no message was set).
    """
    messages: dict[str, str | None] = {}
    for tag in tags:
        ref = f"{registry}/{pkg_name}:{tag}"
        result = subprocess.run(
            ["oras", "manifest", "fetch", ref],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            messages[tag] = None
            continue
        try:
            manifest = json.loads(result.stdout)
            annotations = manifest.get("annotations", {})
            messages[tag] = annotations.get("org.opencontainers.image.description")
        except (json.JSONDecodeError, AttributeError):
            messages[tag] = None
    return messages


_GITHUB_API = "https://api.github.com"

_SCOPE_FIX_HINT = (
    "\n  Your token lacks the required GitHub API scopes for package management."
    "\n  Fix: set GHCR_TOKEN to a PAT with read:packages + delete:packages scopes."
    "\n  Create one at: https://github.com/settings/tokens/new"
    "\n    Required scopes: read:packages, write:packages, delete:packages"
)


def _github_api_request(
    path: str,
    token: str,
    method: str = "GET",
) -> tuple[int, dict | list | None]:
    """Make a GitHub REST API request. Returns (status_code, json_body)."""
    import urllib.error
    import urllib.request

    url = f"{_GITHUB_API}{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode()
            return resp.status, json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            return e.code, json.loads(body) if body else None
        except json.JSONDecodeError:
            return e.code, {"message": body}


def _github_api_paginated(
    path: str,
    token: str,
) -> tuple[int, list]:
    """Fetch all pages from a paginated GitHub API endpoint."""
    import urllib.error
    import urllib.request

    all_items: list = []
    url = f"{_GITHUB_API}{path}?per_page=100"

    while url:
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")

        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read().decode())
                if isinstance(body, list):
                    all_items.extend(body)
                else:
                    return resp.status, all_items

                # Parse Link header for next page
                link_header = resp.headers.get("Link", "")
                url = None
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        url = part.split("<")[1].split(">")[0]
        except urllib.error.HTTPError as e:
            body_str = e.read().decode() if e.fp else ""
            try:
                err = json.loads(body_str)
            except json.JSONDecodeError:
                err = {"message": body_str}
            return e.code, err if isinstance(err, list) else [err]

    return 200, all_items


def _resolve_token(token: str | None) -> str:
    """Resolve a GitHub token from argument, env var, or gh CLI."""
    if token:
        return token

    # Try GHCR_TOKEN env var
    env_token = os.environ.get("GHCR_TOKEN")
    if env_token:
        return env_token

    # Fall back to gh auth token
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()

    raise click.ClickException(
        "No GitHub token found. Provide --token, set GHCR_TOKEN,"
        " or run 'gh auth login'."
    )


def _get_ghcr_owner_and_type(registry: str, token: str) -> tuple[str, str]:
    """Extract the owner and API type from a GHCR registry string.

    Returns (owner, api_type) where api_type is 'orgs' or 'users'.
    """
    # registry is like "ghcr.io/owner-name"
    parts = registry.split("/")
    owner = parts[-1] if len(parts) >= 2 else parts[0]

    status, _ = _github_api_request(f"/orgs/{owner}", token)
    api_type = "orgs" if status == 200 else "users"
    return owner, api_type


def _get_package_version_id(
    owner: str,
    api_type: str,
    tag: str,
    token: str,
    pkg_name: str = "imas-codex-graph",
) -> int | None:
    """Find the GHCR package version ID for a given tag."""
    path = f"/{api_type}/{owner}/packages/container/{pkg_name}/versions"
    status, data = _github_api_paginated(path, token)

    if status == 403:
        msg = ""
        if isinstance(data, list) and data:
            msg = data[0].get("message", "") if isinstance(data[0], dict) else ""
        click.echo(f"  Permission denied listing package versions: {msg}", err=True)
        click.echo(_SCOPE_FIX_HINT, err=True)
        return None

    if status != 200:
        msg = ""
        if isinstance(data, list) and data:
            msg = (
                data[0].get("message", "")
                if isinstance(data[0], dict)
                else str(data[0])
            )
        click.echo(
            f"  Failed to query package versions (HTTP {status}): {msg}", err=True
        )
        return None

    if not isinstance(data, list):
        click.echo("  Unexpected API response format", err=True)
        return None

    for version in data:
        tags = version.get("metadata", {}).get("container", {}).get("tags", [])
        if tag in tags:
            return version["id"]

    return None


def _delete_tag(
    registry: str,
    tag: str,
    token: str | None = None,
    pkg_name: str = "imas-codex-graph",
) -> bool:
    """Delete a specific tag from GHCR using the GitHub Packages API.

    GHCR does not support the OCI manifest delete endpoint (`oras manifest
    delete` returns "unsupported: The operation is unsupported"). We use
    the GitHub REST API directly to find and delete the package version.
    """
    resolved_token = _resolve_token(token)
    owner, api_type = _get_ghcr_owner_and_type(registry, resolved_token)
    version_id = _get_package_version_id(owner, api_type, tag, resolved_token, pkg_name)

    if version_id is None:
        click.echo(f"  Could not find version for tag: {tag}", err=True)
        return False

    path = f"/{api_type}/{owner}/packages/container/{pkg_name}/versions/{version_id}"
    status, resp = _github_api_request(path, resolved_token, method="DELETE")

    if status == 403:
        msg = resp.get("message", "") if isinstance(resp, dict) else ""
        click.echo(f"  Permission denied deleting {tag}: {msg}", err=True)
        click.echo(_SCOPE_FIX_HINT, err=True)
        return False

    if status not in (200, 204):
        msg = resp.get("message", "") if isinstance(resp, dict) else str(resp)
        click.echo(f"  Failed to delete {tag} (HTTP {status}): {msg}", err=True)
        return False

    return True


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
    "-f",
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
    "-f",
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
