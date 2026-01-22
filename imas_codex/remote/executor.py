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
        # SSH execution
        result = subprocess.run(
            ["ssh", ssh_host, cmd],
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
) -> str:
    """Execute a multi-line script via stdin to avoid bash -c overhead.

    Unlike run_command(), this passes the script via stdin which avoids the ~11s
    overhead of bash loading .bashrc when invoked with 'bash -c'.

    This is 2-3x faster for complex scripts on remote facilities with
    slow bashrc initialization (e.g., ITER with module system).

    Low-level execution primitive. Does NOT do facility lookup.
    For facility-aware execution, use run_script() from tools.py.

    Args:
        script: Multi-line bash script to execute
        ssh_host: SSH host to connect to (None = local)
        timeout: Command timeout in seconds
        check: Raise exception on non-zero exit

    Returns:
        Command output (stdout + stderr)
    """
    is_local = is_local_host(ssh_host)

    if is_local:
        # Local execution via stdin
        result = subprocess.run(
            ["bash"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    else:
        # SSH execution via stdin - avoids bash -c overhead
        # Use 'bash -s' to read script from stdin (non-login shell, no bashrc)
        result = subprocess.run(
            ["ssh", "-T", ssh_host, "bash -s"],
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
