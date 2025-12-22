"""
Remote facility exploration via SSH.

This package provides LLM-driven exploration of remote fusion facilities.
The Cursor chat LLM orchestrates exploration via terminal commands,
with session logging and learning persistence.

Usage:
    uv run imas-codex epfl "python --version"
    uv run imas-codex epfl --status
    uv run imas-codex epfl --finish < learnings.yaml
"""

from imas_codex.remote.executor import run_command, run_script
from imas_codex.remote.finish import finish_session
from imas_codex.remote.session import (
    CommandRecord,
    discard_session,
    get_session_log_path,
    get_session_status,
    read_session_log,
)

__all__ = [
    "CommandRecord",
    "discard_session",
    "finish_session",
    "get_session_log_path",
    "get_session_status",
    "read_session_log",
    "run_command",
    "run_script",
]
