"""
Low-level command execution for local and remote systems.

This module provides the core SSH/local execution primitives WITHOUT any
dependency on facility configuration. This allows discovery modules to
import these functions without creating circular imports.

For facility-aware execution, use the higher-level functions in tools.py
which wrap these with automatic ssh_host resolution.

Functions:
- run_command(): Execute command locally or via SSH
- run_script_via_stdin(): Execute multi-line script via stdin
- is_local_host(): Check if ssh_host refers to local machine
"""

import logging
import re
import socket
import subprocess
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_local_hostnames() -> set[str]:
    """Get all hostnames that refer to this machine."""
    hostnames = {"localhost", "127.0.0.1", "::1"}

    # Current hostname
    hostname = socket.gethostname()
    hostnames.add(hostname)
    hostnames.add(hostname.lower())

    # FQDN
    try:
        fqdn = socket.getfqdn()
        hostnames.add(fqdn)
        hostnames.add(fqdn.lower())
        # Also add short name from FQDN
        short = fqdn.split(".")[0]
        hostnames.add(short)
        hostnames.add(short.lower())
    except Exception:
        pass

    return hostnames


@lru_cache(maxsize=16)
def _parse_ssh_config_host(ssh_host: str) -> str | None:
    """Parse ~/.ssh/config to get HostName for an alias.

    Args:
        ssh_host: SSH host alias (e.g., 'epfl')

    Returns:
        Resolved HostName or None if not found
    """
    ssh_config_path = Path.home() / ".ssh" / "config"
    if not ssh_config_path.exists():
        return None

    try:
        content = ssh_config_path.read_text()
    except Exception:
        return None

    # Parse SSH config - look for Host block matching ssh_host
    current_host = None
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Match "Host <pattern>" line
        host_match = re.match(r"^Host\s+(.+)$", line, re.IGNORECASE)
        if host_match:
            hosts = host_match.group(1).split()
            current_host = ssh_host if ssh_host in hosts else None
            continue

        # If we're in the right Host block, look for HostName
        if current_host:
            hostname_match = re.match(r"^HostName\s+(.+)$", line, re.IGNORECASE)
            if hostname_match:
                return hostname_match.group(1).strip()

    return None


def is_local_host(ssh_host: str | None) -> bool:
    """Determine if an ssh_host refers to the local machine.

    This is a low-level check based on hostname matching.
    For facility-aware locality detection, use is_local_facility() from tools.py.

    Args:
        ssh_host: SSH host alias or hostname (None = local)

    Returns:
        True if ssh_host refers to local machine
    """
    if ssh_host is None:
        return True

    local_hostnames = _get_local_hostnames()

    # Check if ssh_host is explicitly localhost
    if ssh_host.lower() in local_hostnames:
        return True

    # Resolve ssh_host via SSH config
    resolved = _parse_ssh_config_host(ssh_host)
    if resolved and resolved.lower() in local_hostnames:
        return True

    return False


def run_command(
    cmd: str,
    ssh_host: str | None = None,
    timeout: int = 60,
    check: bool = False,
) -> str:
    """Execute command locally or via SSH.

    Low-level execution primitive. Does NOT do facility lookup.
    For facility-aware execution, use run() from tools.py.

    Args:
        cmd: Shell command to execute
        ssh_host: SSH host to connect to (None = local)
        timeout: Command timeout in seconds
        check: Raise exception on non-zero exit

    Returns:
        Command output (stdout + stderr)
    """
    is_local = is_local_host(ssh_host)

    if is_local:
        # Local execution
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    else:
        # SSH execution with -T to disable pseudo-terminal allocation
        # This avoids triggering .bashrc on systems where it's loaded for PTY sessions
        result = subprocess.run(
            ["ssh", "-T", ssh_host, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]: {result.stderr}"

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    return output.strip() or "(no output)"


def run_script_via_stdin(
    script: str,
    ssh_host: str | None = None,
    timeout: int = 60,
    check: bool = False,
    interpreter: str = "bash",
) -> str:
    """Execute a multi-line script via stdin to avoid shell -c overhead.

    Unlike run_command(), this passes the script via stdin which avoids the ~11s
    overhead of bash loading .bashrc when invoked with 'bash -c'.

    This is 2-3x faster for complex scripts on remote facilities with
    slow bashrc initialization (e.g., ITER with module system).

    Low-level execution primitive. Does NOT do facility lookup.
    For facility-aware execution, use run_script() from tools.py.

    Args:
        script: Multi-line script to execute
        ssh_host: SSH host to connect to (None = local)
        timeout: Command timeout in seconds
        check: Raise exception on non-zero exit
        interpreter: Interpreter to use ("bash", "python3", etc.)

    Returns:
        Command output (stdout + stderr)
    """
    is_local = is_local_host(ssh_host)

    # Build interpreter command
    # For bash: use 'bash -s' to read from stdin
    # For python: use 'python3 -' to read from stdin
    if interpreter == "bash":
        interp_cmd = ["bash", "-s"]
    elif interpreter in ("python3", "python"):
        interp_cmd = ["python3", "-"]
    else:
        interp_cmd = [interpreter]

    if is_local:
        # Local execution via stdin
        result = subprocess.run(
            interp_cmd,
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    else:
        # SSH execution via stdin - avoids bash -c overhead
        # Use 'interpreter [args]' to read script from stdin
        result = subprocess.run(
            ["ssh", "-T", ssh_host, " ".join(interp_cmd)],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]: {result.stderr}"

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, script[:100], result.stdout, result.stderr
        )

    return output.strip() or "(no output)"


def run_python_script(
    script_name: str,
    input_data: dict | list | None = None,
    ssh_host: str | None = None,
    timeout: int = 60,
) -> str:
    """Execute a Python script from the remote/scripts package.

    Loads the script from imas_codex/remote/scripts/ and executes it
    via SSH (or locally). Input data is passed as JSON on stdin.

    The script is sent via stdin to Python, followed by the JSON input
    using a heredoc-style approach.

    Args:
        script_name: Script filename (e.g., "scan_directories.py")
        input_data: Dict/list to pass as JSON on stdin (None = no input)
        ssh_host: SSH host to connect to (None = local)
        timeout: Command timeout in seconds

    Returns:
        Script output (stdout)

    Raises:
        FileNotFoundError: If script doesn't exist
        subprocess.CalledProcessError: On non-zero exit
    """
    import base64
    import importlib.resources
    import json

    # Load script from package (Python 3.12+ required)
    script_path = importlib.resources.files("imas_codex.remote.scripts").joinpath(
        script_name
    )
    script_content = script_path.read_text()

    is_local = is_local_host(ssh_host)

    # Prepare JSON input
    json_input = json.dumps(input_data) if input_data is not None else "{}"

    # Encode script as base64 to avoid quoting issues
    script_b64 = base64.b64encode(script_content.encode()).decode()

    # Single-line Python that decodes script, runs it with JSON on stdin
    # This avoids nested quoting and subprocess overhead
    runner = (
        f"import base64,subprocess,sys;"
        f's=base64.b64decode("{script_b64}");'
        f'exec(compile(s,"script","exec"))'
    )

    if is_local:
        result = subprocess.run(
            ["python3", "-c", runner],
            input=json_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    else:
        # SSH execution with JSON piped through
        result = subprocess.run(
            ["ssh", "-T", ssh_host, f"python3 -c '{runner}'"],
            input=json_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]: {result.stderr}"

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, script_name, result.stdout, result.stderr
        )

    return output.strip()
