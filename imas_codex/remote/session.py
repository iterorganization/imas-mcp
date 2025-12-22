"""
Session log management for facility exploration.

Sessions track commands executed during an exploration, stored as
append-only JSON Lines files. This enables concurrent command execution
without locks.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def get_cache_dir() -> Path:
    """Get the cache directory for session logs."""
    cache_dir = Path.home() / ".cache" / "imas-codex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_session_log_path(facility: str) -> Path:
    """Get the session log path for a facility."""
    return get_cache_dir() / f"{facility}.jsonl"


def get_large_output_dir(facility: str) -> Path:
    """Get directory for large output files."""
    output_dir = get_cache_dir() / facility
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@dataclass
class CommandRecord:
    """A single command execution record."""

    timestamp: datetime
    command: str
    exit_code: int
    stdout: str
    stderr: str | None = None
    truncated: bool = False
    output_file: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "ts": self.timestamp.isoformat(),
            "cmd": self.command,
            "exit": self.exit_code,
            "stdout": self.stdout,
        }
        if self.stderr:
            data["stderr"] = self.stderr
        if self.truncated:
            data["truncated"] = True
        if self.output_file:
            data["output_file"] = self.output_file
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CommandRecord":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["ts"]),
            command=data["cmd"],
            exit_code=data["exit"],
            stdout=data["stdout"],
            stderr=data.get("stderr"),
            truncated=data.get("truncated", False),
            output_file=data.get("output_file"),
        )


def append_to_log(facility: str, record: CommandRecord) -> None:
    """Append a command record to the session log (atomic for reasonable sizes)."""
    log_path = get_session_log_path(facility)
    with open(log_path, "a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def read_session_log(facility: str) -> list[CommandRecord]:
    """Read all command records from the session log."""
    log_path = get_session_log_path(facility)
    if not log_path.exists():
        return []

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    records.append(CommandRecord.from_dict(data))
                except (json.JSONDecodeError, KeyError):
                    # Skip malformed lines
                    continue
    return records


def discard_session(facility: str) -> bool:
    """Clear the session log without persisting learnings."""
    log_path = get_session_log_path(facility)
    if log_path.exists():
        log_path.unlink()
        return True
    return False


@dataclass
class SessionStatus:
    """Summary of a session's state."""

    facility: str
    exists: bool
    command_count: int
    started_at: datetime | None
    last_command_at: datetime | None
    commands: list[CommandRecord]

    def format(self) -> str:
        """Format as human-readable string."""
        if not self.exists:
            return f"No active session for {self.facility}"

        lines = [
            f"Session: {self.facility}",
            f"Commands: {self.command_count}",
        ]

        if self.started_at:
            ago = datetime.now(UTC) - self.started_at
            minutes = int(ago.total_seconds() / 60)
            lines.append(
                f"Started: {minutes} minutes ago ({self.started_at.isoformat()})"
            )

        if self.last_command_at:
            lines.append(f"Last command: {self.last_command_at.isoformat()}")

        lines.append("")
        lines.append("Commands:")
        for i, cmd in enumerate(self.commands, 1):
            stdout_len = len(cmd.stdout)
            size_str = (
                f"{stdout_len} bytes"
                if stdout_len < 1024
                else f"{stdout_len / 1024:.1f} KB"
            )
            status = "✓" if cmd.exit_code == 0 else f"✗ ({cmd.exit_code})"
            lines.append(
                f"  {i}. {cmd.command[:60]}{'...' if len(cmd.command) > 60 else ''}"
            )
            lines.append(f"     {status} | {size_str}")

        return "\n".join(lines)


def get_session_status(facility: str) -> SessionStatus:
    """Get the current session status for a facility."""
    records = read_session_log(facility)

    if not records:
        return SessionStatus(
            facility=facility,
            exists=False,
            command_count=0,
            started_at=None,
            last_command_at=None,
            commands=[],
        )

    return SessionStatus(
        facility=facility,
        exists=True,
        command_count=len(records),
        started_at=records[0].timestamp,
        last_command_at=records[-1].timestamp,
        commands=records,
    )
