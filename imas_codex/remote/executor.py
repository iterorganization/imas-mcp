"""
Command execution on remote facilities with session logging.

Executes commands via SSH and logs results to the session log.
Large outputs are saved to separate files to keep logs manageable.

Supports local execution when running on the target facility itself.
"""

import socket
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime

from imas_codex.discovery.config import get_config
from imas_codex.discovery.connection import FacilityConnection
from imas_codex.discovery.sandbox import CommandSandbox
from imas_codex.remote.session import (
    CommandRecord,
    append_to_log,
    get_large_output_dir,
)

# Threshold for saving large outputs to separate files
LARGE_OUTPUT_THRESHOLD = 100_000  # 100 KB

# Sandbox for command validation
_sandbox = CommandSandbox()


def _is_local_host(facility: str) -> bool:
    """
    Check if we're running on the target facility.

    Compares current hostname against known hostnames for the facility.
    """
    config = get_config(facility)

    # Get hostnames to check against
    hostnames = getattr(config, "hostnames", None) or []

    # Also check ssh_host if it looks like a hostname (not an alias)
    ssh_host = config.ssh_host
    if "." in ssh_host:
        hostnames.append(ssh_host)

    if not hostnames:
        return False

    # Get current hostname (short and FQDN)
    current_short = socket.gethostname()
    try:
        current_fqdn = socket.getfqdn()
    except Exception:
        current_fqdn = current_short

    # Check if any match
    for hostname in hostnames:
        hostname_lower = hostname.lower()
        if (
            current_short.lower() == hostname_lower
            or current_fqdn.lower() == hostname_lower
            or current_short.lower().startswith(hostname_lower.split(".")[0])
        ):
            return True

    return False


def _run_local(command: str, validate: bool = True) -> tuple[int, str, str]:
    """
    Execute a command locally.

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    if validate:
        is_valid, error = _sandbox.validate(command)
        if not is_valid:
            raise ValueError(f"Command rejected by sandbox: {error}")

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=_sandbox.timeout,
    )
    return result.returncode, result.stdout, result.stderr


@dataclass
class CommandResult:
    """Result of executing a command on a remote facility."""

    command: str
    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.exit_code == 0


def run_command(facility: str, command: str) -> CommandResult:
    """
    Execute a command on a facility and log it.

    Automatically detects if running locally on the target facility
    and skips SSH for faster execution.

    Args:
        facility: Facility identifier (e.g., 'epfl')
        command: Shell command to execute

    Returns:
        CommandResult with output and exit code
    """
    # Check if we're running locally on this facility
    if _is_local_host(facility):
        return_code, stdout, stderr = _run_local(command, validate=True)
    else:
        config = get_config(facility)
        conn = FacilityConnection(config.ssh_host)

        with conn.session():
            result = conn.run(command, validate=True)

        return_code = result.return_code
        stdout = result.stdout or ""
        stderr = result.stderr or ""
    truncated = False
    output_file = None

    # Handle large outputs
    stdout_for_log = stdout
    if len(stdout) > LARGE_OUTPUT_THRESHOLD:
        # Save full output to file
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = get_large_output_dir(facility)
        output_file = output_dir / f"output_{timestamp}.txt"
        output_file.write_text(stdout)

        # Truncate for log
        stdout_for_log = (
            f"[Output saved to {output_file}]\n"
            f"First 1000 chars:\n{stdout[:1000]}\n...\n"
            f"Last 500 chars:\n{stdout[-500:]}"
        )
        truncated = True
        output_file = str(output_file)

    # Create and append record
    record = CommandRecord(
        timestamp=datetime.now(UTC),
        command=command,
        exit_code=return_code,
        stdout=stdout_for_log,
        stderr=stderr[:2000] if stderr else None,
        truncated=truncated,
        output_file=output_file,
    )
    append_to_log(facility, record)

    return CommandResult(
        command=command,
        exit_code=return_code,
        stdout=stdout,
        stderr=stderr,
    )


def _run_script_local(script: str, validate: bool = True) -> tuple[int, str, str]:
    """
    Execute a script locally via bash.

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    if validate:
        _sandbox.validate_script(script)

    result = subprocess.run(
        ["bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=_sandbox.timeout,
    )
    return result.returncode, result.stdout, result.stderr


def run_script(facility: str, script: str) -> CommandResult:
    """
    Execute a multiline script on a facility via bash -s.

    This bypasses the single-command sandbox and instead validates
    the entire script for dangerous patterns. Useful for batching
    multiple commands in a single round-trip.

    Automatically detects if running locally on the target facility
    and skips SSH for faster execution.

    Args:
        facility: Facility identifier (e.g., 'epfl')
        script: Multiline bash script content

    Returns:
        CommandResult with output and exit code

    Raises:
        ValueError: If script contains dangerous patterns
    """
    # Check if we're running locally on this facility
    if _is_local_host(facility):
        return_code, stdout, stderr = _run_script_local(script, validate=True)
    else:
        config = get_config(facility)
        conn = FacilityConnection(config.ssh_host)

        with conn.session():
            result = conn.run_script(script, validate=True)

        return_code = result.return_code
        stdout = result.stdout or ""
        stderr = result.stderr or ""
    truncated = False
    output_file = None

    # Handle large outputs
    stdout_for_log = stdout
    if len(stdout) > LARGE_OUTPUT_THRESHOLD:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = get_large_output_dir(facility)
        output_file = output_dir / f"script_output_{timestamp}.txt"
        output_file.write_text(stdout)

        stdout_for_log = (
            f"[Output saved to {output_file}]\n"
            f"First 1000 chars:\n{stdout[:1000]}\n...\n"
            f"Last 500 chars:\n{stdout[-500:]}"
        )
        truncated = True
        output_file = str(output_file)

    # Log as a single script execution
    # Truncate script in log if very long
    script_summary = script if len(script) < 500 else f"{script[:500]}...[truncated]"
    record = CommandRecord(
        timestamp=datetime.now(UTC),
        command=f"[SCRIPT]\n{script_summary}",
        exit_code=return_code,
        stdout=stdout_for_log,
        stderr=stderr[:2000] if stderr else None,
        truncated=truncated,
        output_file=output_file,
    )
    append_to_log(facility, record)

    return CommandResult(
        command=script,
        exit_code=return_code,
        stdout=stdout,
        stderr=stderr,
    )
