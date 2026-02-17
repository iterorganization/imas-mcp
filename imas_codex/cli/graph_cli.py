"""Graph lifecycle CLI for Neo4j database management.

This module provides the ``imas-codex graph`` command group for:
- Graph database export/load/push/pull to GHCR (with per-facility federation)
- Graph lifecycle: clear, backup, restore, clean

Neo4j server management is under ``imas-codex serve neo4j``.
SSH tunnel management is under ``imas-codex tunnel``.
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
NEO4J_IMAGE = Path.home() / "apptainer" / "neo4j_2025.11-community.sif"
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

        After loading a database dump, Neo4j's auth database is replaced and
        the password must be re-initialized before the first start.
        """
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
            # May fail if password already set - not an error
            if "password is already set" not in result.stderr.lower():
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
            ["pkill", "-15", "-f", "neo4j_2025.*community.sif"],
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

        click.echo(
            f"Note: Restart Neo4j manually: "
            f"imas-codex serve neo4j start --graph {self.profile.name}"
        )

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


def get_package_name(facility: str | None = None) -> str:
    """Get the GHCR package name, optionally scoped to a facility.

    Args:
        facility: If given, returns ``"imas-codex-graph-{facility}"``.
            Otherwise returns ``"imas-codex-graph"`` (unified package).
    """
    if facility:
        return f"imas-codex-graph-{facility}"
    return "imas-codex-graph"


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
    Archive & Registry:
      imas-codex graph export         Export graph to archive
      imas-codex graph load <file>    Load graph archive
      imas-codex graph push           Push archive to GHCR
      imas-codex graph pull           Fetch + load from GHCR
      imas-codex graph fetch          Download archive (no load)
      imas-codex graph list           List GHCR versions

    \b
    Local Operations:
      imas-codex graph status         Show graph and registry status
      imas-codex graph backup         Create local backup (.dump)
      imas-codex graph restore        Restore from local backup
      imas-codex graph clear          Clear all graph data
      imas-codex graph clean          Remove GHCR tags or old backups
    """
    pass


# ============================================================================
# Neo4j Server Group (registered under 'serve' in CLI __init__)
# ============================================================================


@click.group("neo4j")
def neo4j() -> None:
    """Manage Neo4j graph database server.

    \b
      imas-codex serve neo4j start     Start Neo4j via Apptainer
      imas-codex serve neo4j stop      Stop Neo4j
      imas-codex serve neo4j status    Check status
      imas-codex serve neo4j profiles  List profiles and ports
      imas-codex serve neo4j shell     Open Cypher shell
      imas-codex serve neo4j service   Manage systemd service
    """
    pass


@neo4j.command("start")
@click.option("--image", envvar="NEO4J_IMAGE", default=None)
@click.option("--data-dir", envvar="NEO4J_DATA", default=None)
@click.option("--password", envvar="NEO4J_PASSWORD", default=None)
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground")
def neo4j_start(
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
        raise click.ClickException(
            f"Neo4j image not found: {image_path}\n"
            "Pull: apptainer pull docker://neo4j:2025.11-community"
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
        f"NEO4J_AUTH=neo4j/{password}",
        "--env",
        f"NEO4J_server_bolt_listen__address=:{profile.bolt_port}",
        "--env",
        f"NEO4J_server_http_listen__address=:{profile.http_port}",
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
                click.echo(
                    f"Neo4j [{profile.name}] ready at http://localhost:{profile.http_port}"
                )
                return
            time.sleep(1)

        click.echo("Warning: Neo4j may still be starting")


@neo4j.command("stop")
@click.option("--data-dir", envvar="NEO4J_DATA", default=None)
def neo4j_stop(data_dir: str | None) -> None:
    """Stop Neo4j server."""
    import signal

    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()
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


@neo4j.command("status")
def neo4j_status() -> None:
    """Check Neo4j server status."""
    from imas_codex.graph.profiles import resolve_neo4j
    from imas_codex.remote.executor import is_local_host
    from imas_codex.remote.tunnel import TUNNEL_OFFSET, is_tunnel_active

    profile = resolve_neo4j()

    # Determine connection topology
    is_remote = profile.host is not None and not is_local_host(profile.host)
    tunneled_http = profile.http_port + TUNNEL_OFFSET
    tunneled_bolt = profile.bolt_port + TUNNEL_OFFSET

    # Choose the HTTP port to probe — prefer tunneled for remote graphs
    if is_remote and is_tunnel_active(tunneled_http):
        probe_port = tunneled_http
        connection_method = "tunnel"
    elif is_tunnel_active(profile.http_port):
        probe_port = profile.http_port
        # Port is active — but is it an SSH tunnel or local server?
        connection_method = "tunnel" if is_remote else "local"
    else:
        probe_port = profile.http_port
        connection_method = "local"

    try:
        import urllib.request

        with urllib.request.urlopen(
            f"http://localhost:{probe_port}/", timeout=5
        ) as resp:
            resp_data = json.loads(resp.read().decode())
            click.echo(f"Neo4j [{profile.name}] is running")
            click.echo(f"  Version: {resp_data.get('neo4j_version', 'unknown')}")
            click.echo(f"  Edition: {resp_data.get('neo4j_edition', 'unknown')}")

            # Connection info — show actual URI and location
            if is_remote:
                click.echo(f"  Location: {profile.location} (remote)")
                if connection_method == "tunnel":
                    click.echo(
                        f"  Bolt: localhost:{tunneled_bolt} → "
                        f"{profile.host}:{profile.bolt_port}"
                    )
                    click.echo(
                        f"  HTTP: localhost:{tunneled_http} → "
                        f"{profile.host}:{profile.http_port}"
                    )
                else:
                    click.echo(f"  Bolt: localhost:{profile.bolt_port}")
                    click.echo(f"  HTTP: localhost:{profile.http_port}")
            else:
                click.echo("  Location: local")
                click.echo(f"  Bolt: localhost:{profile.bolt_port}")
                click.echo(f"  HTTP: localhost:{profile.http_port}")
                click.echo(f"  Data: {profile.data_dir}")
            click.echo(f"  URI: {profile.uri}")
    except Exception:
        click.echo(f"Neo4j [{profile.name}] is not responding on port {probe_port}")
        if is_remote:
            click.echo(f"  Location: {profile.location} (remote)")
            has_tunnel = is_tunnel_active(tunneled_bolt)
            if has_tunnel:
                click.echo(f"  Tunnel: active (localhost:{tunneled_bolt})")
            else:
                click.echo("  Tunnel: not active")
                click.echo(f"  Start tunnel: imas-codex tunnel start {profile.host}")
        else:
            # Local graph - check for hung process
            pid_file = profile.data_dir / "neo4j.pid"
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    os.kill(pid, 0)  # Check if process exists
                    click.echo(
                        f"  WARNING: Neo4j process exists (PID {pid}) but is not responding",
                        err=True,
                    )
                    click.echo(
                        "  This may indicate a hung process. To recover:", err=True
                    )
                    click.echo(
                        f"    kill {pid} && imas-codex serve neo4j start", err=True
                    )
                except (ProcessLookupError, ValueError, OSError):
                    pass
            # Check for stale locks
            has_lock, lock_msg = check_database_locks(profile.data_dir)
            if has_lock and lock_msg:
                click.echo(f"  {lock_msg}", err=True)


@neo4j.command("profiles")
def neo4j_profiles() -> None:
    """List Neo4j location profiles and their port assignments.

    Also shows locally stored graph instances from .neo4j/.
    """
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
            marker = "→" if g.is_active else " "
            drift = " ⚠ hash drift" if not g.hash_ok else ""
            click.echo(
                f"  {marker} {g.name} [{g.hash}] {','.join(g.facilities)}{drift}"
            )


@neo4j.command("secure")
def neo4j_secure() -> None:
    """Secure Neo4j data directory permissions.

    Sets restrictive permissions (700 for dirs, 600 for files) on the
    Neo4j data directory to prevent other users on shared filesystems
    from accessing your database.
    """
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()
    data_path = profile.data_dir

    if not data_path.exists():
        click.echo(f"Data directory does not exist: {data_path}")
        return

    click.echo(f"Securing {data_path} ...")
    secure_data_directory(data_path)
    click.echo(f"Permissions set to owner-only (700/600) for [{profile.name}]")


@neo4j.command("shell")
@click.option("--image", envvar="NEO4J_IMAGE", default=None)
@click.option("--password", envvar="NEO4J_PASSWORD", default=None)
def neo4j_shell(image: str | None, password: str | None) -> None:
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


@neo4j.command("service")
@click.argument("action", type=click.Choice(["install", "uninstall", "status"]))
@click.option("--image", envvar="NEO4J_IMAGE", default=None, help="Custom image path")
@click.option("--data-dir", envvar="NEO4J_DATA", default=None, help="Custom data dir")
@click.option("--password", envvar="NEO4J_PASSWORD", default=None)
@click.option(
    "--minimal", is_flag=True, help="Use minimal service (no resource limits)"
)
def neo4j_service(
    action: str,
    image: str | None,
    data_dir: str | None,
    password: str | None,
    minimal: bool,
) -> None:
    """Manage Neo4j as a systemd user service."""
    import platform

    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j()
    password = password or profile.password

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

    if action == "install":
        # Check for conflicting tunnel before installing
        from imas_codex.graph.profiles import check_graph_conflict

        conflict = check_graph_conflict(profile.bolt_port)
        if conflict:
            raise click.ClickException(conflict)

        if not image_path.exists():
            raise click.ClickException(
                f"Neo4j image not found: {image_path}\n"
                "Pull: apptainer pull docker://neo4j:2025.11-community"
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
    --env NEO4J_AUTH=neo4j/{password} \\
    --env NEO4J_server_bolt_listen__address=:{profile.bolt_port} \\
    --env NEO4J_server_http_listen__address=:{profile.http_port} \\
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
            f"  Bolt: localhost:{profile.bolt_port}  "
            f"HTTP: localhost:{profile.http_port}"
        )
        click.echo(f"  Start: systemctl --user start {service_name}")
        click.echo(f"  Or:    imas-codex serve neo4j start --graph {profile.name}")

    elif action == "uninstall":
        if not service_file.exists():
            click.echo(f"Service [{profile.name}] not installed")
            return
        subprocess.run(
            ["systemctl", "--user", "stop", service_name], capture_output=True
        )
        subprocess.run(
            ["systemctl", "--user", "disable", service_name], capture_output=True
        )
        service_file.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        click.echo(f"Service [{profile.name}] uninstalled")

    elif action == "status":
        if not service_file.exists():
            click.echo(f"Service [{profile.name}] not installed")
            return
        result = subprocess.run(
            ["systemctl", "--user", "status", service_name],
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout)


# ============================================================================
# Graph Data Operations
# ============================================================================


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
            f"NEO4J_server_bolt_listen__address=:{temp_bolt_port}",
            "--env",
            f"NEO4J_server_http_listen__address=:{temp_http_port}",
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


@graph.command("export")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output archive path (default: imas-codex-graph[-{facility}]-{version}.tar.gz)",
)
@click.option("--no-restart", is_flag=True, help="Don't restart Neo4j after export")
@click.option(
    "--facility",
    "-f",
    default=None,
    help="Export per-facility graph (e.g. tcv)",
)
def graph_export(
    output: str | None,
    no_restart: bool,
    facility: str | None,
) -> None:
    """Export graph database to archive."""
    from imas_codex.graph.profiles import resolve_neo4j

    require_apptainer()

    profile = resolve_neo4j()

    git_info = get_git_info()
    version_label = git_info["tag"] or f"dev-{git_info['commit_short']}"
    pkg_name = get_package_name(facility)

    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"{pkg_name}-{version_label}.tar.gz")

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

            # If facility specified, filter the dump to keep only that facility
            if facility:
                click.echo(f"  Filtering dump for facility: {facility}")
                _create_facility_dump(
                    archive_dir / "graph.dump",
                    facility,
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
    """Load graph database from archive."""
    from imas_codex.graph.profiles import resolve_neo4j
    from imas_codex.settings import get_graph_password

    profile = resolve_neo4j()
    password = password or get_graph_password()
    require_apptainer()

    archive_path = Path(archive)
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

# Node types with description + embedding fields, updated before push
_DESCRIPTION_EMBEDDABLE_LABELS = [
    "FacilitySignal",
    "FacilityPath",
    "TreeNode",
    "WikiArtifact",
]


def _update_description_embeddings() -> None:
    """Update description embeddings for all embeddable node types.

    Called before graph dump in `data push` to ensure all description
    fields have up-to-date embeddings. Uses the same logic as
    `imas-codex embed update` but runs for all labels in sequence.
    """
    from imas_codex.embeddings.description import embed_descriptions_batch
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        for label in _DESCRIPTION_EMBEDDABLE_LABELS:
            # Count nodes needing update
            result = gc.query(
                f"MATCH (n:{label}) "
                f"WHERE n.description IS NOT NULL "
                f"  AND n.description <> '' "
                f"  AND n.embedding IS NULL "
                f"RETURN count(n) AS total"
            )
            total = result[0]["total"] if result else 0
            if total == 0:
                click.echo(f"  {label}: all descriptions embedded ✓")
                continue

            click.echo(f"  {label}: embedding {total} descriptions...")
            processed = 0
            batch_size = 100

            while True:
                rows = gc.query(
                    f"MATCH (n:{label}) "
                    f"WHERE n.description IS NOT NULL "
                    f"  AND n.description <> '' "
                    f"  AND n.embedding IS NULL "
                    f"RETURN n.id AS id, n.description AS description "
                    f"LIMIT $batch_size",
                    batch_size=batch_size,
                )
                if not rows:
                    break

                items = [{"id": r["id"], "description": r["description"]} for r in rows]
                items = embed_descriptions_batch(items)
                gc.query(
                    f"UNWIND $items AS item "
                    f"MATCH (n:{label} {{id: item.id}}) "
                    f"SET n.embedding = item.embedding",
                    items=items,
                )
                processed += len(items)

            click.echo(f"  {label}: embedded {processed} descriptions ✓")


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
@click.option("--skip-embed", is_flag=True, help="Skip description embedding update")
@click.option(
    "--facility",
    "-f",
    default=None,
    help="Push per-facility graph (e.g. tcv) to imas-codex-graph-{facility}",
)
def graph_push(
    dev: bool,
    registry: str | None,
    token: str | None,
    dry_run: bool,
    skip_embed: bool,
    facility: str | None,
) -> None:
    """Push graph archive to GHCR.

    Before dumping, updates description embeddings for all node types
    that have description fields (FacilitySignal, FacilityPath, TreeNode,
    WikiArtifact). Use --skip-embed to skip this step.

    Use --facility to push a per-facility graph to a separate GHCR package.
    """
    require_oras()

    git_info = get_git_info()

    if not dev:
        require_clean_git(git_info)

    target_registry = get_registry(git_info, registry)
    version_tag = get_version_tag(git_info, dev)
    pkg_name = get_package_name(facility)

    click.echo(f"Push target: {target_registry}/{pkg_name}:{version_tag}")
    if git_info["is_fork"]:
        click.echo(f"  Detected fork: {git_info['remote_owner']}")

    if dry_run:
        click.echo("\n[DRY RUN] Would:")
        click.echo("  1. Update description embeddings")
        click.echo("  2. Dump graph (auto stop/start Neo4j)")
        click.echo(f"  3. Push to {target_registry}/{pkg_name}:{version_tag}")
        return

    # Step 1: Update description embeddings before dump
    if not skip_embed:
        _update_description_embeddings()
    else:
        click.echo("Skipped embedding update (--skip-embed)")

    archive_path = Path(f"{pkg_name}-{version_tag}.tar.gz")

    try:
        from click.testing import CliRunner

        runner = CliRunner()
        dump_args = ["-o", str(archive_path)]
        if graph:
            dump_args.extend(["--graph", graph])
        if facility:
            dump_args.extend(["--facility", facility])
        result = runner.invoke(graph_export, dump_args)
        if result.exit_code != 0:
            raise click.ClickException(f"Export failed: {result.output}")

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

        click.echo(f"Pushing to {artifact_ref}...")
        result = subprocess.run(push_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(f"Push failed: {result.stderr}")

        click.echo(f"✓ Pushed: {artifact_ref}")

        manifest = get_local_graph_manifest() or {}
        manifest["pushed"] = True
        manifest["pushed_version"] = version_tag
        manifest["pushed_to"] = artifact_ref
        manifest["pushed_at"] = datetime.now(UTC).isoformat()
        save_local_graph_manifest(manifest)

        # Save dev revision for auto-increment on next push
        if dev:
            base = __version__.replace("+", "-")
            # Extract revision from version_tag (format: base-rN)
            rev_str = version_tag.rsplit("-r", 1)[-1]
            _save_dev_revision(base, int(rev_str))

        if not dev:
            result = subprocess.run(
                ["oras", "tag", artifact_ref, "latest"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                click.echo("✓ Tagged as latest")
            else:
                click.echo(
                    f"Warning: Failed to tag as latest: {result.stderr.strip()}",
                    err=True,
                )

    finally:
        if archive_path.exists():
            archive_path.unlink()

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
    default=None,
    help="Fetch per-facility graph (e.g. tcv) from imas-codex-graph-{facility}",
)
def graph_fetch(
    version: str,
    registry: str | None,
    token: str | None,
    output: str | None,
    facility: str | None,
) -> Path:
    """Fetch graph archive from GHCR without loading.

    Downloads the archive to disk but does NOT load it into Neo4j.
    Use 'graph load <archive>' to load it afterwards, or use
    'graph pull' as a convenience for fetch + load.

    When no --version is specified, fetches 'latest'. If 'latest' doesn't
    exist, falls back to the most recent tag in the registry.
    """
    require_oras()

    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    pkg_name = get_package_name(facility)

    # Resolve version: if "latest" doesn't exist, find most recent tag
    resolved_version = version
    if version == "latest":
        resolved_version = _resolve_latest_tag(target_registry, token, pkg_name)

    artifact_ref = f"{target_registry}/{pkg_name}:{resolved_version}"
    click.echo(f"Fetching: {artifact_ref}")

    login_to_ghcr(token)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        result = subprocess.run(
            ["oras", "pull", artifact_ref, "-o", str(tmp)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise click.ClickException(f"Fetch failed: {result.stderr}")

        archives = list(tmp.glob("*.tar.gz"))
        if not archives:
            raise click.ClickException("No archive found in fetched artifact")

        src_archive = archives[0]
        if output:
            dest = Path(output)
        else:
            dest = Path(f"{pkg_name}-{resolved_version}.tar.gz")

        shutil.move(str(src_archive), str(dest))

    size_mb = dest.stat().st_size / 1024 / 1024
    click.echo(f"✓ Fetched: {dest} ({size_mb:.1f} MB)")
    click.echo(f"  Load with: imas-codex graph load {dest}")
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
    default=None,
    help="Pull per-facility graph (e.g. tcv) from imas-codex-graph-{facility}",
)
def graph_pull(
    version: str,
    registry: str | None,
    token: str | None,
    force: bool,
    no_backup: bool,
    facility: str | None,
) -> None:
    """Pull graph from GHCR and load it (convenience for fetch + load).

    This is equivalent to running 'graph fetch' followed by 'graph load'.
    Use 'graph fetch' if you only want to download without loading.

    When no --version is specified, pulls 'latest'. If 'latest' doesn't
    exist, falls back to the most recent tag in the registry.

    Use --facility to pull a per-facility graph from a separate GHCR package.
    """
    from imas_codex.graph.profiles import resolve_neo4j

    require_oras()

    profile = resolve_neo4j()
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    pkg_name = get_package_name(facility)

    # Resolve version: if "latest" doesn't exist, find most recent tag
    resolved_version = version
    if version == "latest":
        resolved_version = _resolve_latest_tag(target_registry, token, pkg_name)

    artifact_ref = f"{target_registry}/{pkg_name}:{resolved_version}"

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
        load_args = [str(archives[0]), "--force", "--graph", profile.name]
        result = runner.invoke(graph_load, load_args)
        if result.exit_code != 0:
            raise click.ClickException(f"Load failed: {result.output}")

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

    click.echo("✓ Graph pull complete")


@graph.command("remote-list")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option(
    "--facility",
    "-f",
    default=None,
    help="List per-facility graph versions",
)
def graph_remote_list(
    registry: str | None, token: str | None, facility: str | None
) -> None:
    """List available graph versions in GHCR."""
    require_oras()

    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    pkg_name = get_package_name(facility)
    repo_ref = f"{target_registry}/{pkg_name}"

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


@graph.command("status")
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
def graph_status(registry: str | None) -> None:
    """Show local and registry status."""
    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)

    click.echo("Local status:")
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
        if manifest.get("dev_base_version"):
            click.echo(
                f"  Dev revision: {manifest['dev_base_version']}"
                f"-r{manifest.get('dev_revision', '?')}"
            )

    click.echo(f"\nNeo4j: {'running' if is_neo4j_running() else 'stopped'}")

    # Show graph profiles with location awareness
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

    # Show in-graph identity (GraphMeta node)
    if is_neo4j_running():
        try:
            from imas_codex.graph.client import GraphClient
            from imas_codex.graph.meta import get_graph_meta

            gc = GraphClient.from_profile()  # type: ignore[possibly-undefined]
            meta = get_graph_meta(gc)
            gc.close()
            if meta:
                click.echo("\nGraph identity (GraphMeta):")
                click.echo(f"  Name: {meta.get('name', '?')}")
                facilities = meta.get("facilities") or []
                click.echo(
                    f"  Facilities: {', '.join(facilities) if facilities else '(none)'}"
                )
                click.echo(f"  Hash: {meta.get('hash', '?')}")
                if meta.get("updated_at"):
                    click.echo(f"  Updated: {meta['updated_at']}")
            else:
                click.echo(
                    "\nGraph identity: not initialized"
                    "\n  Run: imas-codex graph init --name <name> --facility <fac>"
                )
        except Exception:
            pass


# ============================================================================
# Graph List Command
# ============================================================================


@graph.command("list")
def graph_list() -> None:
    """List local graph instances.

    Scans the .neo4j/ store directory for graph instances and shows
    their name, facilities, hash, and whether they are active.
    Validates hash consistency and warns about drift.

    \b
    Examples:
      imas-codex graph list
    """
    from imas_codex.graph.dirs import get_active_graph, list_local_graphs

    graphs = list_local_graphs()
    if not graphs:
        click.echo("No local graphs found.")
        click.echo("Create one: imas-codex graph init -n <name> -f <facility> ...")
        return

    active = get_active_graph()
    active_hash = active.hash if active else None

    click.echo("Local graphs:\n")
    for g in graphs:
        marker = "→ " if g.hash == active_hash else "  "
        facs = ", ".join(g.facilities) if g.facilities else "(none)"
        click.echo(f"{marker}{g.name}  [{facs}]  hash={g.hash}")
        if g.created_at:
            click.echo(f"    created: {g.created_at}")
        for warn in g.warnings:
            click.echo(f"    ⚠ {warn}")

    click.echo(f"\n{len(graphs)} graph(s)")


# ============================================================================
# Graph Switch Command
# ============================================================================


@graph.command("switch")
@click.argument("identifier")
def graph_switch(identifier: str) -> None:
    """Switch the active graph.

    Stops Neo4j if running, repoints the neo4j/ symlink to the
    target graph directory, and restarts Neo4j.

    IDENTIFIER can be a graph name (e.g. "codex") or a hash prefix
    (e.g. "a3f8").  If the name matches multiple graphs, use the
    hash prefix to disambiguate.

    \b
    Examples:
      imas-codex graph switch codex
      imas-codex graph switch dev
      imas-codex graph switch a3f8
    """
    from imas_codex.graph.dirs import (
        find_graph,
        get_active_graph,
        switch_active_graph,
    )
    from imas_codex.graph.profiles import resolve_neo4j

    try:
        target = find_graph(identifier)
    except LookupError as e:
        raise click.ClickException(str(e)) from e

    # Check if already active
    active = get_active_graph()
    if active and active.hash == target.hash:
        click.echo(f"Graph '{target.name}' (hash={target.hash}) is already active.")
        return

    profile = resolve_neo4j(auto_tunnel=False)
    was_running = is_neo4j_running(profile.http_port)

    if was_running:
        click.echo(f"Stopping Neo4j [{profile.name}]...")
        _stop_neo4j_for_switch(profile)

    try:
        switch_active_graph(target.hash)
    except (FileExistsError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    click.echo(
        f"✓ Switched to '{target.name}' "
        f"[{', '.join(target.facilities)}] "
        f"hash={target.hash}"
    )

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
        f"NEO4J_AUTH=neo4j/{profile.password}",
        "--env",
        f"NEO4J_server_bolt_listen__address=:{profile.bolt_port}",
        "--env",
        f"NEO4J_server_http_listen__address=:{profile.http_port}",
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
@click.option(
    "--name",
    "-n",
    required=True,
    help="Graph name (e.g. 'codex', 'dev')",
)
@click.option(
    "--facility",
    "-f",
    "facilities",
    multiple=True,
    required=True,
    help="Facility ID to include (repeatable)",
)
@click.option(
    "--migrate",
    is_flag=True,
    help="Migrate existing neo4j/ directory into the .neo4j/ store",
)
def graph_init(name: str, facilities: tuple[str, ...], migrate: bool) -> None:
    """Initialize a new graph instance.

    Creates a hash-named directory in .neo4j/, writes .meta.json,
    creates the neo4j/ symlink, starts Neo4j, and initializes the
    (:GraphMeta) node.

    If neo4j/ is an existing real directory (pre-migration), use
    --migrate to move it into the .neo4j/ store automatically.

    \b
    Examples:
      imas-codex graph init -n codex -f imas -f iter -f tcv -f jt60sa
      imas-codex graph init -n dev -f imas -f tcv
      imas-codex graph init -n codex -f imas -f iter --migrate
    """
    from imas_codex.graph.dirs import (
        create_graph_dir,
        is_legacy_data_dir,
        migrate_legacy_dir,
        switch_active_graph,
    )
    from imas_codex.graph.profiles import resolve_neo4j

    facility_list = sorted(set(facilities))

    # Handle legacy migration
    if is_legacy_data_dir():
        if migrate:
            click.echo("Migrating existing neo4j/ directory into .neo4j/ store...")
            profile = resolve_neo4j(auto_tunnel=False)
            was_running = is_neo4j_running(profile.http_port)
            if was_running:
                click.echo("Stopping Neo4j for migration...")
                _stop_neo4j_for_switch(profile)

            try:
                info = migrate_legacy_dir(name, facility_list)
            except (FileExistsError, ValueError) as e:
                raise click.ClickException(str(e)) from e

            click.echo(f"✓ Migrated to .neo4j/{info.hash}/")
        else:
            raise click.ClickException(
                "neo4j/ exists as a real directory (pre-migration layout).\n"
                "Use --migrate to move it into the .neo4j/ store:\n"
                f"  imas-codex graph init -n {name} "
                + " ".join(f"-f {f}" for f in facility_list)
                + " --migrate"
            )
    else:
        # Create new graph directory
        try:
            info = create_graph_dir(name, facility_list)
        except FileExistsError as e:
            raise click.ClickException(str(e)) from e

        # Point symlink to the new directory
        try:
            switch_active_graph(info.hash)
        except FileExistsError as e:
            raise click.ClickException(str(e)) from e

    click.echo(f"  Name: {name}")
    click.echo(f"  Facilities: {', '.join(facility_list)}")
    click.echo(f"  Hash: {info.hash}")

    # Start Neo4j and create GraphMeta node
    profile = resolve_neo4j(auto_tunnel=False)

    if not is_neo4j_running(profile.http_port):
        click.echo("\nStarting Neo4j...")
        _start_neo4j_after_switch(profile)

    if is_neo4j_running(profile.http_port):
        from imas_codex.graph.client import GraphClient
        from imas_codex.graph.meta import init_graph_meta

        gc = GraphClient.from_profile()
        result = init_graph_meta(gc, name, facility_list)
        gc.close()
        click.echo(f"\n✓ GraphMeta node initialized (hash={result['hash']})")
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
        click.echo("Run: imas-codex graph init --name <name> --facility <fac>")
        return

    facilities = meta.get("facilities") or []
    click.echo(f"Graph: {meta.get('name', '?')}")
    click.echo(f"Hash: {meta.get('hash', '?')}")
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
            "Run: imas-codex graph init --name <name> --facility <fac>"
        )

    add_facility_to_meta(gc, facility_id)

    meta = get_graph_meta(gc)
    gc.close()

    facilities = meta.get("facilities") or [] if meta else []
    click.echo(
        f"✓ Added '{facility_id}' to graph '{meta.get('name', '?') if meta else '?'}'"
    )
    click.echo(f"  Facilities: {', '.join(facilities)}")
    click.echo(f"  Hash: {meta.get('hash', '?') if meta else '?'}")


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
    click.echo(f"  Hash: {meta.get('hash', '?') if meta else '?'}")


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


@graph.command("clean")
@click.argument("tags", nargs=-1)
@click.option("--dev", "dev_only", is_flag=True, help="Remove all dev tags from GHCR")
@click.option(
    "--before",
    "before_version",
    default=None,
    help="Remove tags older than this semver version",
)
@click.option("--pattern", default=None, help="Glob pattern to match GHCR tags")
@click.option(
    "--backups", is_flag=True, help="Operate on local backup files instead of GHCR"
)
@click.option(
    "--older-than",
    "older_than",
    default=None,
    help="With --backups: remove backups older than N days (e.g. 30d)",
)
@click.option(
    "--keep-latest",
    type=int,
    default=0,
    help="With --backups: retain N most recent backup files",
)
@click.option(
    "--facility",
    "-f",
    default=None,
    help="Target per-facility GHCR package",
)
@click.option("--registry", envvar="IMAS_DATA_REGISTRY", default=None)
@click.option("--token", envvar="GHCR_TOKEN")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.option("--dry-run", is_flag=True, help="Show what would be removed")
def graph_clean(
    tags: tuple[str, ...],
    dev_only: bool,
    before_version: str | None,
    pattern: str | None,
    backups: bool,
    older_than: str | None,
    keep_latest: int,
    facility: str | None,
    registry: str | None,
    token: str | None,
    force: bool,
    dry_run: bool,
) -> None:
    """Remove GHCR tags or clean local backups.

    \b
    GHCR examples:
      imas-codex graph clean tag1 tag2         # Delete specific tags
      imas-codex graph clean --dev              # Remove all dev tags
      imas-codex graph clean --before 0.5.0    # Remove tags older than v0.5.0
      imas-codex graph clean --pattern "*.dev*" # Glob match on tags

    \b
    Backup examples:
      imas-codex graph clean --backups --older-than 30d
      imas-codex graph clean --backups --keep-latest 5
    """
    if backups:
        _remove_backups(older_than, keep_latest, force, dry_run)
        return

    # GHCR mode
    require_oras()

    git_info = get_git_info()
    target_registry = get_registry(git_info, registry)
    pkg_name = get_package_name(facility)

    available = _list_registry_tags(target_registry, token, pkg_name)
    if not available:
        click.echo("No tags in registry.")
        return

    to_delete: list[str] = []

    # Explicit tags
    if tags:
        missing = [t for t in tags if t not in available]
        if missing:
            click.echo(f"Tags not found: {', '.join(missing)}", err=True)
        to_delete.extend(t for t in tags if t in available)

    # --dev: all dev tags
    if dev_only:
        to_delete.extend(
            t
            for t in available
            if t != "latest" and ("dev" in t or "-r" in t) and t not in to_delete
        )

    # --before VERSION: tags with semver < given version
    if before_version:
        from packaging.version import InvalidVersion, Version

        try:
            threshold = Version(before_version)
        except InvalidVersion:
            raise click.ClickException(f"Invalid version: {before_version}") from None

        for t in available:
            if t == "latest" or t in to_delete:
                continue
            # Extract base version from tag (e.g. "0.4.0.dev123-abc-r1" -> "0.4.0")
            base = t.split(".dev")[0].split("-")[0]
            try:
                if Version(base) < threshold:
                    to_delete.append(t)
            except InvalidVersion:
                continue

    # --pattern GLOB
    if pattern:
        import fnmatch

        to_delete.extend(
            t
            for t in available
            if t != "latest" and fnmatch.fnmatch(t, pattern) and t not in to_delete
        )

    if not to_delete:
        click.echo("No tags matched for removal.")
        return

    # Deduplicate preserving order
    seen: set[str] = set()
    unique_delete: list[str] = []
    for t in to_delete:
        if t not in seen:
            seen.add(t)
            unique_delete.append(t)
    to_delete = unique_delete

    click.echo(f"Will remove {len(to_delete)} tag(s) from {target_registry}:")
    for t in to_delete:
        click.echo(f"  - {t}")

    if dry_run:
        click.echo("\n[DRY RUN] No tags were removed.")
        return

    if not force:
        if not click.confirm("\nProceed?"):
            click.echo("Aborted.")
            return

    deleted = 0
    for t in to_delete:
        if _delete_tag(target_registry, t, token, pkg_name):
            click.echo(f"  Removed: {t}")
            deleted += 1

    click.echo(f"\n✓ Removed {deleted}/{len(to_delete)} tags")


def _remove_backups(
    older_than: str | None,
    keep_latest: int,
    force: bool,
    dry_run: bool,
) -> None:
    """Remove local backup dump files."""
    from imas_codex.graph.profiles import BACKUPS_DIR

    if not BACKUPS_DIR.exists():
        click.echo("No backups directory found.")
        return

    all_backups = sorted(
        BACKUPS_DIR.glob("*.dump"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not all_backups:
        click.echo("No backup files found.")
        return

    to_remove: list[Path] = []

    if older_than:
        # Parse "30d" format
        value = older_than.rstrip("d")
        try:
            days = int(value)
        except ValueError:
            raise click.ClickException(
                f"Invalid --older-than format: {older_than} (expected e.g. 30d)"
            ) from None

        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(days=days)
        cutoff_ts = cutoff.timestamp()

        to_remove = [b for b in all_backups if b.stat().st_mtime < cutoff_ts]

    elif keep_latest > 0:
        if len(all_backups) > keep_latest:
            to_remove = all_backups[keep_latest:]
        else:
            click.echo(
                f"Only {len(all_backups)} backup(s), "
                f"fewer than --keep-latest={keep_latest}. Nothing to remove."
            )
            return
    else:
        click.echo("Specify --older-than or --keep-latest for backup cleanup.")
        return

    if not to_remove:
        click.echo("No backups matched for removal.")
        return

    click.echo(f"Will remove {len(to_remove)} backup file(s):")
    for b in to_remove:
        mb = b.stat().st_size / 1024 / 1024
        click.echo(f"  - {b.name} ({mb:.1f} MB)")

    if dry_run:
        click.echo("\n[DRY RUN] No backups were removed.")
        return

    if not force:
        if not click.confirm("\nProceed?"):
            click.echo("Aborted.")
            return

    removed = 0
    for b in to_remove:
        b.unlink()
        removed += 1

    click.echo(f"\n✓ Removed {removed} backup file(s)")


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
            f"Start it: imas-codex serve neo4j start"
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


@graph.command("backup")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output dump file path (default: auto-named in backups dir)",
)
def graph_backup(output: str | None) -> None:
    """Create a neo4j-admin dump backup of the graph.

    Backup files are stored in ~/.local/share/imas-codex/backups/ by default.
    Use 'graph restore' to reload a backup.
    """
    require_apptainer()

    output_path = Path(output) if output else None

    with Neo4jOperation("graph backup", require_stopped=True):
        dump_path = backup_graph_dump(output=output_path)

    size_mb = dump_path.stat().st_size / 1024 / 1024
    click.echo(f"✓ Backup created: {dump_path} ({size_mb:.1f} MB)")

    # List recent backups
    from imas_codex.graph.profiles import BACKUPS_DIR

    if BACKUPS_DIR.exists():
        backups = sorted(BACKUPS_DIR.glob("*.dump"), reverse=True)[:5]
        if len(backups) > 1:
            click.echo(f"\nRecent backups ({len(backups)} shown):")
            for b in backups:
                mb = b.stat().st_size / 1024 / 1024
                click.echo(f"  {b.name} ({mb:.1f} MB)")


@graph.command("restore")
@click.argument("backup_file", type=click.Path(exists=True), required=False)
@click.option("--password", envvar="NEO4J_PASSWORD", default=None)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def graph_restore(
    backup_file: str | None,
    password: str | None,
    force: bool,
) -> None:
    """Restore graph from a neo4j-admin dump backup.

    If no BACKUP_FILE is provided, lists available backups and prompts
    for selection.
    """
    from imas_codex.graph.profiles import BACKUPS_DIR, resolve_neo4j

    require_apptainer()
    profile = resolve_neo4j()
    password = password or profile.password

    if backup_file is None:
        # List available backups
        if not BACKUPS_DIR.exists():
            raise click.ClickException(
                "No backups directory found.\n"
                "Create a backup first: imas-codex graph backup"
            )

        backups = sorted(BACKUPS_DIR.glob("*.dump"), reverse=True)
        if not backups:
            raise click.ClickException(
                "No backup files found.\nCreate a backup first: imas-codex graph backup"
            )

        click.echo("Available backups:")
        for i, b in enumerate(backups[:10], 1):
            mb = b.stat().st_size / 1024 / 1024
            click.echo(f"  {i}. {b.name} ({mb:.1f} MB)")

        choice = click.prompt("Select backup number", type=int, default=1)
        if choice < 1 or choice > len(backups[:10]):
            raise click.ClickException("Invalid selection")

        backup_path = backups[choice - 1]
    else:
        backup_path = Path(backup_file)

    click.echo(f"Restore [{profile.name}] from: {backup_path.name}")

    if not force:
        if not click.confirm("This will REPLACE the current graph. Continue?"):
            click.echo("Aborted.")
            return

    with Neo4jOperation(
        "graph restore",
        require_stopped=True,
        reset_password_on_restart=True,
        password=password,
    ):
        # Copy dump to expected location
        dumps_dir = profile.data_dir / "dumps"
        dumps_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(backup_path, dumps_dir / "neo4j.dump")

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
            raise click.ClickException(f"Restore failed: {result.stderr}")

    click.echo(f"✓ Restored [{profile.name}] from {backup_path.name}")
