"""CLI logging configuration with file output.

Provides a shared ``configure_cli_logging`` function that sets up both
console and file logging for CLI commands.  Log files are split by
CLI tool *and* facility under ``~/.local/share/imas-codex/logs/``.

Naming convention::

    <command>_<facility>.log   # e.g. paths_tcv.log, wiki_jet.log
    <command>.log              # fallback when no facility is specified

Usage from any CLI command::

    from imas_codex.cli.logging import configure_cli_logging

    configure_cli_logging("wiki", facility="tcv", verbose=verbose)

Agents can access the logs via::

    tail -f ~/.local/share/imas-codex/logs/paths_tcv.log   # Follow live
    ls -la ~/.local/share/imas-codex/logs/                  # List all logs
    rg ERROR ~/.local/share/imas-codex/logs/                # Search errors
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Standard log directory follows XDG convention
LOG_DIR = Path.home() / ".local" / "share" / "imas-codex" / "logs"


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
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)

    # Ensure root logger threshold allows file handler to receive events.
    # NOTSET (0) means "inherit from parent" which defaults to WARNING,
    # so we must explicitly set the level when it's NOTSET or too high.
    if root_logger.level == logging.NOTSET or root_logger.level > file_level:
        root_logger.setLevel(file_level)

    return log_file.parent
