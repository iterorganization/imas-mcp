"""Neo4j operational infrastructure.

Provides:
- Neo4jOperation context manager (stop/start bracketing)
- Database dump with recovery retry
- Lock detection (POSIX + GPFS)
- Process diagnostics
- Data directory security
- Backup and existence checks
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from imas_codex.graph.profiles import Neo4jProfile

# ============================================================================
# Constants
# ============================================================================

SERVICES_DIR = Path("imas_codex/config/services")
RECOVERY_DIR = Path.home() / ".local" / "share" / "imas-codex" / "recovery"
DATA_DIR = Path.home() / ".local" / "share" / "imas-codex" / "neo4j"
NEO4J_LOCK_FILE = Path.home() / ".config" / "imas-codex" / "neo4j-operation.lock"


def neo4j_image() -> Path:
    """Resolve the Neo4j Apptainer SIF image path."""
    from imas_codex.settings import get_neo4j_image_path

    return get_neo4j_image_path()


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
        if self.require_stopped:
            is_locked, is_verified, _ = check_database_lock(self.profile)
            needs_stop = is_neo4j_running(self.profile.http_port) or (
                is_locked and is_verified
            )

            # GPFS doesn't propagate POSIX locks across nodes, so the
            # check above misses Neo4j running on a SLURM compute node.
            if not needs_stop:
                try:
                    from imas_codex.cli.services import (
                        _get_neo4j_job,
                        _is_graph_compute_target,
                    )

                    if _is_graph_compute_target():
                        job = _get_neo4j_job()
                        if job and job["state"] == "RUNNING":
                            needs_stop = True
                except Exception:
                    pass

            if needs_stop:
                self.was_running = True
                click.echo(
                    f"Stopping Neo4j [{self.profile.name}] for {self.operation_name}..."
                )
                self._stop_neo4j()

                import time

                # _stop_neo4j already waits for process exit (up to 90s
                # for SLURM) and cleans stale POSIX locks. Just confirm
                # the HTTP endpoint is down. Don't call
                # check_database_lock here — fcntl.lockf hangs on GPFS
                # stale locks, causing this loop to block indefinitely.
                for _ in range(15):
                    if not is_neo4j_running(self.profile.http_port):
                        break
                    time.sleep(1)
                else:
                    self._release_lock()
                    raise click.ClickException(
                        f"Failed to stop Neo4j [{self.profile.name}] — "
                        "still responding on HTTP port after shutdown"
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
        image = neo4j_image()
        auth_file = self.profile.data_dir / "data" / "dbms" / "auth.ini"
        if auth_file.exists():
            auth_file.unlink()
        cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{self.profile.data_dir}/data:/data",
            "--writable-tmpfs",
            str(image),
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
        # Try SLURM-based stop first (Neo4j on compute node)
        try:
            from imas_codex.cli.services import _is_graph_compute_target

            if _is_graph_compute_target():
                from imas_codex.cli.services import (
                    _NEO4J_JOB,
                    _cancel_service_job,
                    _clean_neo4j_locks,
                    _get_neo4j_job,
                )

                job = _get_neo4j_job()
                if job and job["state"] == "RUNNING":
                    node = job["node"]
                    _cancel_service_job(_NEO4J_JOB)
                    _clean_neo4j_locks(node)
        except Exception:
            pass

        # Always kill local orphans — recovery cycles or manual starts
        # can leave Neo4j processes running directly on the login node
        service_name = f"imas-codex-neo4j-{self.profile.name}"
        subprocess.run(
            ["systemctl", "--user", "stop", service_name],
            capture_output=True,
        )
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
        # Try SLURM-based start first (Neo4j on compute node)
        try:
            from imas_codex.cli.services import _is_graph_compute_target

            if _is_graph_compute_target():
                from imas_codex.cli.services import deploy_neo4j

                job = deploy_neo4j()
                node = job["node"]
                click.echo(f"  Neo4j [{self.profile.name}] ready on {node}")
                return
        except Exception:
            pass

        # Fall back to systemd
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
# Neo4j process and lock helpers
# ============================================================================


def is_neo4j_running(http_port: int | None = None) -> bool:
    from imas_codex.graph.profiles import HTTP_BASE_PORT

    if http_port is None:
        http_port = HTTP_BASE_PORT

    # For SLURM compute locations, check the resolved service URL
    # instead of blindly hitting localhost.
    try:
        from imas_codex.remote.locations import resolve_service_url
        from imas_codex.settings import get_graph_location

        location = get_graph_location()
        url = resolve_service_url(location, http_port, protocol="http")
        if url:
            import urllib.request

            urllib.request.urlopen(f"{url}/", timeout=2)
            return True
    except Exception:
        pass

    # Fallback: check localhost directly
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


def check_graph_exists(data_dir: Path | None = None) -> bool:
    target = data_dir or DATA_DIR
    data_path = target / "data" / "databases" / "neo4j"
    return data_path.exists() and any(data_path.iterdir())


# ============================================================================
# Backup helpers
# ============================================================================


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

    run_neo4j_dump(profile, dumps_dir)

    neo4j_dump = dumps_dir / "neo4j.dump"
    if neo4j_dump.exists():
        shutil.move(str(neo4j_dump), str(dump_path))
    else:
        raise click.ClickException("Dump file not created by neo4j-admin")

    return dump_path


# ============================================================================
# Dump error parsing
# ============================================================================


def parse_dump_error(stderr: str) -> tuple[str, bool]:
    """Parse neo4j-admin dump stderr into a user-friendly message.

    Returns (message, is_active_lock) where is_active_lock indicates
    another process holds the database lock (recovery won't help).
    """
    lower = stderr.lower()

    # Explicit lock / in-use signals
    if "database is in use" in lower or "filelockexception" in lower:
        return (
            "Neo4j database is currently in use by another process.\n"
            "Stop Neo4j first: imas-codex graph stop",
            True,
        )

    # Non-lock errors that also produce "dump failed for databases" —
    # check these BEFORE the generic "dump failed" catch-all.
    if "unable to find store id" in lower:
        return "Unable to find store id (store format issue)", False
    if "classformaterror" in lower or "incompatible magic value" in lower:
        return "JVM class format error (Apptainer overlay issue)", False

    # "Dump failed for databases: 'neo4j'" without a specific cause
    # usually means the database lock couldn't be acquired.
    if "dump failed for databases" in lower:
        return (
            "Neo4j database dump failed (likely locked by another process).\n"
            "Stop Neo4j first: imas-codex graph stop",
            True,
        )

    # Generic failure — extract the first Caused-by or CommandFailedException
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("Caused by:") or "CommandFailedException" in stripped:
            return stripped, False
    return stderr.splitlines()[0].strip() if stderr.strip() else "Unknown error", False


# ============================================================================
# Database lock detection
# ============================================================================


def check_database_lock(
    profile: Neo4jProfile,
) -> tuple[bool, bool, str | None]:
    """Check if the Neo4j database is locked and verify the holder.

    Returns (is_locked, is_verified, holder_info):
    - is_locked: POSIX file lock or HTTP endpoint indicates active use
    - is_verified: a running process was found that holds the lock
    - holder_info: diagnostic string describing the lock holder

    On GPFS, POSIX locks survive across nodes, so a lock may appear
    held even after Neo4j on the compute node has exited (stale lock).
    We verify by checking for actual running processes.

    **GPFS caveat**: ``fcntl.lockf`` with ``LOCK_NB`` can hang
    indefinitely on stale cross-node locks despite being non-blocking.
    A 5-second timeout via ``signal.alarm`` prevents deadlock.
    """
    if is_neo4j_running(profile.http_port):
        return True, True, "Neo4j responding on localhost"

    # Check the database_lock file on disk (visible on GPFS).
    # Neo4j uses Java FileChannel.lock() which creates POSIX (fcntl)
    # locks — BSD flock() won't detect them.  We must use fcntl.lockf
    # with O_RDWR (POSIX exclusive-lock test needs write access).
    lock_file = profile.data_dir / "data" / "databases" / "neo4j" / "database_lock"
    posix_locked = False
    if lock_file.exists():
        import fcntl
        import signal

        def _lock_timeout_handler(signum, frame):
            raise TimeoutError("fcntl.lockf hung on GPFS stale lock")

        try:
            fd = os.open(str(lock_file), os.O_RDWR)
            try:
                old_handler = signal.signal(signal.SIGALRM, _lock_timeout_handler)
                signal.alarm(5)
                try:
                    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.lockf(fd, fcntl.LOCK_UN)
                except OSError:
                    posix_locked = True
                except TimeoutError:
                    # fcntl hung — GPFS stale lock, treat as locked
                    posix_locked = True
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            finally:
                os.close(fd)
        except OSError:
            pass

    if not posix_locked:
        return False, False, None

    # Lock is held — verify a process actually owns it
    proc_info = neo4j_process_info(profile)
    if proc_info:
        return True, True, proc_info

    # POSIX lock held but no process found — likely stale on GPFS
    return True, False, None


def neo4j_process_info(profile: Neo4jProfile) -> str | None:
    """Return a diagnostic string about running Neo4j processes, or None."""
    lines: list[str] = []
    # Check systemd service
    service_name = f"imas-codex-neo4j-{profile.name}"
    result = subprocess.run(
        ["systemctl", "--user", "is-active", service_name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip() == "active":
        lines.append(f"  systemd service: {service_name} (active)")

    # Check SLURM allocation (Neo4j may be on a compute node)
    slurm_node = None
    try:
        from imas_codex.cli.services import _is_graph_compute_target

        if _is_graph_compute_target():
            result = subprocess.run(
                [
                    "squeue",
                    "--me",
                    "--noheader",
                    "--format=%j %N %T",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    if "codex" in line.lower() or "neo4j" in line.lower():
                        parts = line.split()
                        if len(parts) >= 2:
                            slurm_node = parts[1]
                        lines.append(f"  SLURM: {line.strip()}")
    except Exception:
        pass

    # If SLURM node found, SSH to verify Neo4j is actually running there
    if slurm_node:
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "ConnectTimeout=5",
                    slurm_node,
                    f"pgrep -u {os.getuid()} -af Neo4jCommunity",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                for proc_line in result.stdout.strip().splitlines():
                    pid = proc_line.split()[0]
                    lines.append(
                        f"  {slurm_node} PID {pid}: "
                        f"{proc_line[len(pid) :].strip()[:60]}"
                    )
        except (subprocess.TimeoutExpired, OSError):
            pass

    # Check for local neo4j processes (exclude our own Python processes)
    result = subprocess.run(
        ["pgrep", "-u", str(os.getuid()), "-af", "neo4j"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        for proc_line in result.stdout.strip().splitlines():
            # Skip Python/uv processes — those are CLI tools, not Neo4j
            if "python" in proc_line.lower() or "uv run" in proc_line:
                continue
            pid = proc_line.split()[0]
            lines.append(f"  PID {pid}: {proc_line[len(pid) :].strip()[:80]}")

    return "\n".join(lines) if lines else None


# ============================================================================
# Neo4j dump with recovery
# ============================================================================


def run_neo4j_dump(
    profile: Neo4jProfile,
    dumps_dir: Path,
    *,
    verbose: bool = False,
) -> None:
    """Run neo4j-admin dump with optional recovery retry.

    If the initial dump fails due to stale locks (unclean shutdown),
    performs a recovery cycle: start Neo4j briefly to replay transaction
    logs and release locks, then stop cleanly and retry.

    If the database is actively in use by another process, fails
    immediately with a clear message.
    """
    image = neo4j_image()

    # Pre-check: fail fast if Neo4j is verified running, or warn on stale lock
    is_locked, is_verified, holder_info = check_database_lock(profile)
    if is_locked and is_verified:
        msg = (
            "Cannot dump: Neo4j database is locked by another process.\n"
            "Stop Neo4j first: imas-codex graph stop"
        )
        if holder_info:
            msg += f"\n\nRunning processes:\n{holder_info}"
        raise click.ClickException(msg)
    if is_locked and not is_verified:
        click.echo(
            "  POSIX lock held but no Neo4j process found — "
            "treating as stale lock, attempting dump..."
        )

    # Bind logs dir so neo4j-admin can write crash reports without
    # needing --writable-tmpfs (which corrupts the FUSE overlay on GPFS).
    logs_dir = profile.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{profile.data_dir}/data:/data",
        "--bind",
        f"{dumps_dir}:/dumps",
        "--bind",
        f"{logs_dir}:/var/lib/neo4j/logs",
        str(image),
        "neo4j-admin",
        "database",
        "dump",
        "neo4j",
        "--to-path=/dumps",
        "--overwrite-destination=true",
    ]
    # Always use --verbose internally so parse_dump_error can detect
    # the error type; only show full output to user with verbose=True
    cmd_verbose = [*cmd, "--verbose"]

    result = subprocess.run(cmd_verbose, capture_output=True, text=True)
    if result.returncode == 0:
        return

    # The dump may succeed but exit non-zero due to Apptainer cleanup
    # errors (e.g. fuse-overlayfs ClassFormatError).  If the dump file
    # exists and neo4j-admin reported completion, accept it.
    dump_file = dumps_dir / "neo4j.dump"
    if dump_file.exists() and "dump completed successfully" in result.stderr.lower():
        if verbose:
            click.echo("  Dump completed (ignoring non-zero exit from cleanup)")
        return

    message, is_active_lock = parse_dump_error(result.stderr)

    if is_active_lock:
        # Database locked by another process — no point attempting recovery
        _, _, holder_info = check_database_lock(profile)
        err_msg = (
            "Cannot dump: Neo4j database is locked by another process.\n"
            "Stop Neo4j first: imas-codex graph stop"
        )
        if holder_info:
            err_msg += f"\n\nRunning processes:\n{holder_info}"
        if verbose:
            click.echo(result.stderr, err=True)
        raise click.ClickException(err_msg)

    # Stale lock / unclean shutdown — attempt recovery cycle
    if verbose:
        click.echo(f"  Initial dump stderr:\n{result.stderr}", err=True)
    click.echo(f"  Initial dump failed ({message}) — performing recovery cycle...")

    # Start Neo4j so it replays transaction logs and releases stale locks
    logs_dir = profile.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _recovery_start_cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{profile.data_dir}/data:/data",
        "--bind",
        f"{logs_dir}:/logs",
        "--bind",
        f"{logs_dir}:/var/lib/neo4j/logs",
        "--writable-tmpfs",
        "--env",
        f"NEO4J_server_bolt_listen__address=127.0.0.1:{profile.bolt_port}",
        "--env",
        f"NEO4J_server_http_listen__address=127.0.0.1:{profile.http_port}",
        str(image),
        "neo4j",
        "console",
    ]
    recovery_proc = subprocess.Popen(
        _recovery_start_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    import time

    click.echo("  Waiting for Neo4j recovery start...")
    for _ in range(60):
        if is_neo4j_running(profile.http_port):
            break
        time.sleep(1)
    else:
        recovery_proc.terminate()
        recovery_proc.wait(timeout=10)
        raise click.ClickException(
            f"Graph dump failed and recovery start timed out: {message}\n"
            "Run with --verbose for full error output."
        )

    # Stop cleanly — Neo4j may take time to checkpoint on GPFS
    click.echo("  Recovery start successful — stopping cleanly...")
    recovery_proc.terminate()
    try:
        recovery_proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        recovery_proc.kill()
        recovery_proc.wait(timeout=10)

    for _ in range(30):
        if not is_neo4j_running(profile.http_port):
            break
        time.sleep(1)

    # Retry dump
    click.echo("  Retrying dump after recovery...")
    result = subprocess.run(cmd_verbose, capture_output=True, text=True)
    if result.returncode != 0:
        message, _ = parse_dump_error(result.stderr)
        if verbose:
            click.echo(result.stderr, err=True)
        raise click.ClickException(
            f"Graph dump failed after recovery: {message}\n"
            "Run with --verbose for full error output."
        )
    click.echo("  Dump succeeded after recovery")
