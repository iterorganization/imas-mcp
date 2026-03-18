"""CLI logging configuration with file output.

Provides a shared ``configure_cli_logging`` function that sets up both
console and file logging for CLI commands.  Log files are split by
CLI tool *and* facility under ``~/.local/share/imas-codex/logs/``.

Also provides structured logging utilities for worker diagnostics
and log reading functions for the MCP logs tools.

Log location routing:
    MCP log tools automatically read from the correct host.
    ``[tool.imas-codex.logs].location`` in pyproject.toml controls where
    logs are fetched from.  When set to a facility name (e.g. ``"iter"``),
    log tools SSH to that host.  ``"local"`` reads from the local filesystem.
    Override: ``IMAS_CODEX_LOG_LOCATION`` env var.

Naming convention::

    <command>_<facility>.log   # e.g. paths_tcv.log, wiki_jet.log
    <command>.log              # fallback when no facility is specified

Usage from any CLI command::

    from imas_codex.cli.logging import configure_cli_logging

    configure_cli_logging("wiki", facility="tcv", verbose=verbose)

Agents can access the logs via MCP tools (get_logs, list_logs, tail_logs)
which automatically route to local or remote based on configuration.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Standard log directory follows XDG convention
LOG_DIR = Path.home() / ".local" / "share" / "imas-codex" / "logs"
REMOTE_LOG_DIR = "~/.local/share/imas-codex/logs"

# Structured log format with worker/batch fields
STRUCTURED_FORMAT = (
    "%(asctime)s %(levelname)-8s %(name)s"
    " [%(worker_name)s%(batch_id_fmt)s]:"
    " %(message)s"
)
STANDARD_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class StructuredFormatter(logging.Formatter):
    """Formatter that includes worker/batch extras when present.

    Falls back to standard format for log records without structured fields.
    Produces output like::

        2026-03-13 10:15:23 INFO     imas_codex.signals [check_worker_2 batch=abc123]: checked 20 signals
        2026-03-13 10:15:24 WARNING  imas_codex.graph: regular log message

    """

    def __init__(self) -> None:
        super().__init__(STANDARD_FORMAT, datefmt=DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        worker = getattr(record, "worker_name", None) or getattr(
            record, "worker", None
        )
        batch = getattr(record, "batch_id", None) or getattr(
            record, "batch", None
        )
        signal = getattr(record, "signal_id", None)

        if worker or batch or signal:
            parts = []
            if worker:
                parts.append(str(worker))
            if batch:
                parts.append(f"batch={batch}")
            if signal:
                parts.append(f"signal={signal}")
            extra_str = " ".join(parts)
            self._style._fmt = (
                f"%(asctime)s %(levelname)-8s %(name)s [{extra_str}]: %(message)s"
            )
        else:
            self._style._fmt = STANDARD_FORMAT

        return super().format(record)


# Log line timestamp regex for parsing
_LOG_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+"
)
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def get_log_dir() -> Path:
    """Return the CLI log directory, creating it if needed."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def get_log_file(command: str, facility: str | None = None) -> Path:
    """Return the log file path for a given CLI command and facility.

    Args:
        command: CLI command name (e.g., "paths", "wiki").
        facility: Optional facility ID.  When provided the file is named
            ``<command>_<facility>.log`` so parallel facility runs don't
            clobber each other.
    """
    stem = f"{command}_{facility}" if facility else command
    return get_log_dir() / f"{stem}.log"


def configure_cli_logging(
    command: str,
    *,
    facility: str | None = None,
    verbose: bool = False,
    console_level: int | None = None,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 3,
) -> Path:
    """Configure logging for a CLI command with file output.

    Sets up:
    - File handler: DEBUG-level rotating log at
      ``~/.local/share/imas-codex/logs/<command>_<facility>.log``
    - Console handler: WARNING (or VERBOSE if requested)

    Args:
        command: CLI command name (e.g., "wiki", "paths", "ingest")
        facility: Facility ID (e.g., "tcv").  Splits the log file per
            facility so parallel runs don't interleave.
        verbose: If True, set console to INFO level
        console_level: Override console level (takes precedence over verbose)
        file_level: File log level (default: DEBUG)
        max_bytes: Max size per log file before rotation
        backup_count: Number of rotated backups to keep

    Returns:
        Path to the log directory
    """
    log_file = get_log_file(command, facility=facility)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Root logger gets file handler at DEBUG level
    root_logger = logging.getLogger("imas_codex")

    # Remove existing file handlers to avoid duplicates on repeated calls
    for handler in root_logger.handlers[:]:
        if isinstance(handler, (RotatingFileHandler, logging.FileHandler)):
            root_logger.removeHandler(handler)

    # File handler: captures everything for diagnosis
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)

    # Ensure root logger threshold allows file handler to receive events.
    # NOTSET (0) means "inherit from parent" which defaults to WARNING,
    # so we must explicitly set the level when it's NOTSET or too high.
    if root_logger.level == logging.NOTSET or root_logger.level > file_level:
        root_logger.setLevel(file_level)

    return log_file.parent


# =============================================================================
# Structured Worker Logging (Phase 4.1)
# =============================================================================


class WorkerLogAdapter(logging.LoggerAdapter):
    """Log adapter that injects worker name and batch ID into log records.

    Usage::

        logger = logging.getLogger(__name__)
        wlog = WorkerLogAdapter(logger, worker_name="check_worker_2")
        wlog.info("checked 20 signals (18 success, 2 failed)")
        # → 2026-03-13 10:15:23 INFO check_worker_2: checked 20 signals ...

        wlog.set_batch("abc123")
        wlog.info("processing batch")
        # → 2026-03-13 10:15:24 INFO check_worker_2 [batch=abc123]: processing batch
    """

    def __init__(
        self,
        logger: logging.Logger,
        worker_name: str,
        batch_id: str | None = None,
    ) -> None:
        super().__init__(logger, {"worker_name": worker_name, "batch_id": batch_id})

    def set_batch(self, batch_id: str | None) -> None:
        """Update the batch ID for subsequent log messages."""
        self.extra["batch_id"] = batch_id

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        worker = self.extra.get("worker_name", "")
        batch = self.extra.get("batch_id")
        batch_str = f" [batch={batch}]" if batch else ""
        return f"{worker}{batch_str}: {msg}", kwargs


def log_worker_error(
    logger: logging.Logger | logging.LoggerAdapter,
    *,
    worker_name: str,
    signal_id: str | None = None,
    error: Exception,
    error_type: str = "application",
    retry_count: int = 0,
    max_retries: int = 0,
    batch_id: str | None = None,
) -> None:
    """Log a worker error with consistent structure (Phase 4.2).

    All worker errors should use this function for uniform log format.

    Args:
        logger: Logger or LoggerAdapter instance.
        worker_name: Name of the worker (e.g. "check_worker_2").
        signal_id: ID of the signal/item being processed.
        error: The exception that occurred.
        error_type: Classification: "infrastructure" or "application".
        retry_count: Current retry attempt number.
        max_retries: Maximum retries configured.
        batch_id: Batch identifier if applicable.
    """
    parts = [worker_name]
    if batch_id:
        parts.append(f"batch={batch_id}")
    if signal_id:
        parts.append(f"signal={signal_id}")
    parts.append(f"type={error_type}")
    if max_retries > 0:
        parts.append(f"retry={retry_count}/{max_retries}")

    context = " ".join(parts)
    level = logging.WARNING if error_type == "infrastructure" else logging.ERROR
    logger.log(level, "%s: %s", context, error)


# =============================================================================
# Log Reading Utilities (Phase 3 support)
# =============================================================================


def list_log_files() -> list[dict[str, str | int | float]]:
    """List available log files with sizes and last-modified times.

    Automatically routes to remote host based on ``[logs].location`` config.

    Returns:
        List of dicts with keys: name, path, size_bytes, modified_iso, age_hours.
    """
    ssh_host = _get_log_ssh_host()
    if ssh_host:
        return _remote_list_log_files(ssh_host)

    log_dir = get_log_dir()
    results = []
    for path in sorted(log_dir.glob("*.log*")):
        if not path.is_file():
            continue
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        age_hours = (datetime.now(tz=UTC) - modified).total_seconds() / 3600
        results.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "modified_iso": modified.isoformat(),
                "age_hours": round(age_hours, 1),
            }
        )
    return results


def read_log(
    command: str = "signals",
    facility: str | None = None,
    lines: int = 100,
    level: str = "WARNING",
    grep: str | None = None,
    since: str | None = None,
) -> str:
    """Read log file with filtering.

    Automatically routes to remote host based on ``[logs].location`` config.

    Args:
        command: CLI command name (e.g. "signals", "wiki", "paths").
        facility: Facility ID (e.g. "jet", "tcv"). If None, uses command.log.
        lines: Maximum lines to return.
        level: Minimum log level to include ("DEBUG", "INFO", "WARNING", "ERROR").
        grep: Text substring filter (case-insensitive).
        since: Time filter. Accepts:
            - Relative: "1h", "30m", "2d"
            - Absolute ISO: "2024-03-13T10:00"

    Returns:
        Filtered log content as string. Empty string if log file not found.
    """
    ssh_host = _get_log_ssh_host()
    if ssh_host:
        return _remote_read_log(
            ssh_host, command=command, facility=facility,
            lines=lines, level=level, grep=grep, since=since,
        )

    log_file = get_log_file(command, facility=facility)
    if not log_file.exists():
        # Try with rotated backup
        for suffix in [".1", ".2", ".3"]:
            backup = Path(str(log_file) + suffix)
            if backup.exists():
                log_file = backup
                break
        else:
            return f"Log file not found: {log_file}"

    min_level = _LOG_LEVELS.get(level.upper(), logging.WARNING)
    since_dt = _parse_since(since) if since else None
    grep_lower = grep.lower() if grep else None

    matching_lines: list[str] = []
    try:
        with open(log_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                # Parse timestamp and level from log line
                m = _LOG_TIMESTAMP_RE.match(line)
                if m:
                    ts_str, lvl_str = m.group(1), m.group(2)
                    line_level = _LOG_LEVELS.get(lvl_str, logging.DEBUG)

                    # Level filter
                    if line_level < min_level:
                        continue

                    # Time filter
                    if since_dt is not None:
                        try:
                            line_dt = datetime.strptime(ts_str, DATE_FORMAT).replace(
                                tzinfo=UTC
                            )
                            if line_dt < since_dt:
                                continue
                        except ValueError:
                            pass

                # Grep filter
                if grep_lower and grep_lower not in line.lower():
                    continue

                matching_lines.append(line.rstrip())
    except OSError as e:
        return f"Error reading log file: {e}"

    # Return last N lines
    result_lines = matching_lines[-lines:]
    return "\n".join(result_lines)


def tail_log(
    command: str = "signals",
    facility: str | None = None,
    lines: int = 50,
) -> str:
    """Get the most recent log entries (tail -n).

    Automatically routes to remote host based on ``[logs].location`` config.

    Returns the last ``lines`` lines from the log file, regardless of
    level or content. Efficient: reads from end of file.

    Args:
        command: CLI command name.
        facility: Facility ID.
        lines: Number of lines from the end.

    Returns:
        Last N lines of the log file.
    """
    ssh_host = _get_log_ssh_host()
    if ssh_host:
        return _remote_tail_log(ssh_host, command=command, facility=facility, lines=lines)

    log_file = get_log_file(command, facility=facility)
    if not log_file.exists():
        return f"Log file not found: {log_file}"

    try:
        with open(log_file, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        result = all_lines[-lines:]
        return "".join(result).rstrip()
    except OSError as e:
        return f"Error reading log file: {e}"


def _parse_since(since: str) -> datetime | None:
    """Parse a 'since' time string to a datetime.

    Supports:
    - Relative: "1h", "30m", "2d", "45s"
    - Absolute ISO: "2024-03-13T10:00"
    """
    since = since.strip()

    # Try relative format: 1h, 30m, 2d
    m = re.match(r"^(\d+)\s*(s|m|h|d)$", since, re.IGNORECASE)
    if m:
        value = int(m.group(1))
        unit = m.group(2).lower()
        delta_map = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
        delta = timedelta(**{delta_map[unit]: value})
        return datetime.now(tz=UTC) - delta

    # Try absolute ISO format
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"]:
        try:
            return datetime.strptime(since, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Remote log routing
# ═══════════════════════════════════════════════════════════════════════════


def _get_log_ssh_host() -> str | None:
    """Return the SSH host for remote log access, or None for local.

    Uses ``get_log_location()`` from settings. If the location is a
    facility name (not ``"local"``), resolves it to an SSH host via
    ``is_local_host`` to handle on-facility vs off-facility access.
    """
    from imas_codex.settings import get_log_location

    location = get_log_location()
    if location == "local":
        return None

    # Check if 'location' actually points to the local machine
    from imas_codex.remote.executor import is_local_host

    if is_local_host(location):
        return None

    return location


def _remote_log_file(command: str, facility: str | None = None) -> str:
    """Return the remote log file path string."""
    stem = f"{command}_{facility}" if facility else command
    return f"{REMOTE_LOG_DIR}/{stem}.log"


def _remote_list_log_files(ssh_host: str) -> list[dict[str, str | int | float]]:
    """List log files on a remote host via SSH."""
    import json

    from imas_codex.remote.executor import run_command

    # Use a Python one-liner for reliable JSON output
    script = (
        "import json, os, time; "
        f"d = os.path.expanduser('{REMOTE_LOG_DIR}'); "
        "files = []; "
        "[files.append({'name': f, "
        "'path': os.path.join(d, f), "
        "'size_bytes': os.stat(os.path.join(d, f)).st_size, "
        "'modified_epoch': os.stat(os.path.join(d, f)).st_mtime}) "
        "for f in sorted(os.listdir(d)) "
        "if os.path.isfile(os.path.join(d, f)) and f.endswith(('.log', '.log.1', '.log.2', '.log.3'))]; "
        "print(json.dumps(files))"
    )

    try:
        output = run_command(f"python3 -c {_shell_quote(script)}", ssh_host=ssh_host, timeout=15)
        # Parse the JSON output (last line)
        for line in reversed(output.strip().split("\n")):
            line = line.strip()
            if line.startswith("["):
                raw = json.loads(line)
                break
        else:
            return []

        now = time.time()
        results = []
        for f in raw:
            mtime = f["modified_epoch"]
            modified = datetime.fromtimestamp(mtime, tz=UTC)
            age_hours = (now - mtime) / 3600
            results.append({
                "name": f["name"],
                "path": f["path"],
                "size_bytes": f["size_bytes"],
                "modified_iso": modified.isoformat(),
                "age_hours": round(age_hours, 1),
            })
        return results
    except Exception as e:
        logging.getLogger(__name__).warning("Remote log listing failed: %s", e)
        return []


def _remote_read_log(
    ssh_host: str,
    command: str = "signals",
    facility: str | None = None,
    lines: int = 100,
    level: str = "WARNING",
    grep: str | None = None,
    since: str | None = None,
) -> str:
    """Read and filter log file on remote host via SSH.

    Uses grep/awk on the remote side for efficiency — only matching lines
    are transferred over the network.
    """
    from imas_codex.remote.executor import run_command

    log_file = _remote_log_file(command, facility)

    # Build a pipeline: cat | grep level | grep text | grep since | tail
    parts = [f"cat {log_file} 2>/dev/null"]

    # Level filter — use grep to match log level prefixes at or above minimum
    level_upper = level.upper()
    levels_above = _levels_at_or_above(level_upper)
    if levels_above and level_upper != "DEBUG":
        level_pattern = "|".join(levels_above)
        parts.append(f"grep -E '({level_pattern})'")

    # Since filter — use awk to compare timestamps
    if since:
        since_dt = _parse_since(since)
        if since_dt:
            since_str = since_dt.strftime("%Y-%m-%d %H:%M:%S")
            # awk: extract timestamp, compare lexicographically
            parts.append(
                f"awk -v since='{since_str}' '"
                '{ts=substr($0,1,19); if(ts >= since) print}'
                "'"
            )

    # Text filter
    if grep:
        parts.append(f"grep -i {_shell_quote(grep)}")

    # Take last N lines
    parts.append(f"tail -n {lines}")

    cmd = " | ".join(parts)

    try:
        output = run_command(cmd, ssh_host=ssh_host, timeout=30)
        if "(no output)" in output and "not found" not in output.lower():
            return ""
        return output
    except Exception as e:
        return f"Error reading remote log: {e}"


def _remote_tail_log(
    ssh_host: str,
    command: str = "signals",
    facility: str | None = None,
    lines: int = 50,
) -> str:
    """Tail log file on remote host via SSH."""
    from imas_codex.remote.executor import run_command

    log_file = _remote_log_file(command, facility)

    try:
        output = run_command(
            f"tail -n {lines} {log_file} 2>/dev/null",
            ssh_host=ssh_host,
            timeout=15,
        )
        return output if output != "(no output)" else f"Log file not found: {log_file}"
    except Exception as e:
        return f"Error reading remote log: {e}"


def _levels_at_or_above(level: str) -> list[str]:
    """Return log level names at or above the given level."""
    ordered = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    try:
        idx = ordered.index(level)
        return ordered[idx:]
    except ValueError:
        return ordered


def _shell_quote(s: str) -> str:
    """Shell-quote a string for remote execution."""
    import shlex
    return shlex.quote(s)
