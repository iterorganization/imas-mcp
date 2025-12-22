"""
Remote facility exploration via SSH.

This package provides LLM-driven exploration of remote fusion facilities.
The Cursor chat LLM orchestrates exploration via terminal commands,
with session logging and artifact capture.

Usage:
    uv run imas-codex epfl "python --version"
    uv run imas-codex epfl --status
    uv run imas-codex epfl --capture environment << 'EOF'
    python:
      version: "3.9.21"
    EOF
"""

from imas_codex.remote.capture import capture_artifact
from imas_codex.remote.executor import run_command, run_script
from imas_codex.remote.session import (
    CommandRecord,
    discard_session,
    get_session_log_path,
    get_session_status,
    read_session_log,
)

__all__ = [
    "CommandRecord",
    "capture_artifact",
    "discard_session",
    "get_session_log_path",
    "get_session_status",
    "read_session_log",
    "run_command",
    "run_script",
]
