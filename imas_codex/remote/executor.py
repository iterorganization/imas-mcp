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
- check_ssh_socket(): Verify SSH control master is healthy
- cleanup_stale_sockets(): Remove stale SSH control master sockets
"""

import logging
import os
import re
import shlex
import socket
import subprocess
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# PATH prefix applied to all remote SSH commands so that tools installed
# by ``imas-codex tools install`` (uv, rg, fd, python via uv) are found.
# This is a safety net — the tools CLI also configures ~/.bashrc directly,
# but explicit PATH ensures correctness even before that setup runs.
_REMOTE_PATH_PREFIX = 'export PATH="$HOME/bin:$HOME/.local/bin:$PATH"'


# ============================================================================
# Per-host Nice Level Registry
# ============================================================================

# Maps ssh_host (lowercase) → nice level (0-19). Registered by facility-aware
# code in tools.py when resolving facility → ssh_host. All execution functions
# check this registry and apply `nice -n {level}` for remote commands.
_host_nice_levels: dict[str, int] = {}


def configure_host_nice(ssh_host: str, level: int) -> None:
    """Register a nice level for all remote commands to an SSH host.

    Once registered, all ``run_command()``, ``run_python_script()``,
    ``run_script_via_stdin()``, and ``async_run_python_script()`` calls
    targeting this host will run at the specified nice level.

    Args:
        ssh_host: SSH host alias (e.g., "jt-60sa", "iter")
        level: Unix nice level (0-19). 19 = lowest priority.
    """
    if not 0 <= level <= 19:
        logger.warning("nice_level %d out of range [0, 19], clamping", level)
        level = max(0, min(19, level))
    _host_nice_levels[ssh_host.lower()] = level
    logger.debug("Registered nice level %d for host %s", level, ssh_host)


def _get_host_nice_level(ssh_host: str | None) -> int | None:
    """Look up registered nice level for an SSH host."""
    if ssh_host is None:
        return None
    return _host_nice_levels.get(ssh_host.lower())


# ============================================================================
# SSH Connection Errors
# ============================================================================


class SSHConnectionError(Exception):
    """Raised when SSH connection to a facility fails.

    This indicates network-level issues like:
    - VPN not connected
    - Host unreachable
    - Connection refused
    - DNS resolution failure
    """

    def __init__(self, ssh_host: str, message: str, suggestion: str | None = None):
        self.ssh_host = ssh_host
        self.suggestion = suggestion
        super().__init__(message)


def check_ssh_connection(ssh_host: str, timeout: int = 10) -> dict:
    """Test SSH connectivity to a host.

    This performs a quick connection test WITHOUT executing any commands.
    Use this to provide clear error messages when network issues occur.

    Args:
        ssh_host: SSH host alias (e.g., 'tcv', 'jt-60sa')
        timeout: Connection timeout in seconds

    Returns:
        Dict with 'connected' (bool), 'error' (str or None), 'suggestion' (str or None)
    """
    if is_local_host(ssh_host):
        return {"connected": True, "error": None, "suggestion": None}

    try:
        # Use `ssh -o ConnectTimeout=X -o BatchMode=yes host true`
        # This tests connection without prompts and exits quickly
        result = subprocess.run(
            [
                "ssh",
                "-o",
                f"ConnectTimeout={timeout}",
                "-o",
                "BatchMode=yes",
                ssh_host,
                "true",
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,  # Allow a bit more than ConnectTimeout
        )

        if result.returncode == 0:
            return {"connected": True, "error": None, "suggestion": None}

        # Parse common SSH errors
        stderr = result.stderr.lower()
        if "connection timed out" in stderr or "operation timed out" in stderr:
            return {
                "connected": False,
                "error": f"Connection to {ssh_host} timed out",
                "suggestion": "Check VPN connection or network access",
            }
        elif "connection refused" in stderr:
            return {
                "connected": False,
                "error": f"Connection to {ssh_host} refused",
                "suggestion": "SSH service may not be running on remote host",
            }
        elif "no route to host" in stderr or "network is unreachable" in stderr:
            return {
                "connected": False,
                "error": f"Cannot reach {ssh_host} - network unreachable",
                "suggestion": "Check VPN connection",
            }
        elif "could not resolve hostname" in stderr:
            return {
                "connected": False,
                "error": f"Cannot resolve hostname for {ssh_host}",
                "suggestion": "Check DNS or VPN connection",
            }
        elif "permission denied" in stderr:
            return {
                "connected": False,
                "error": f"Permission denied for {ssh_host}",
                "suggestion": "Check SSH key or credentials",
            }
        else:
            return {
                "connected": False,
                "error": f"SSH connection failed: {result.stderr.strip()}",
                "suggestion": None,
            }

    except subprocess.TimeoutExpired:
        return {
            "connected": False,
            "error": f"Connection to {ssh_host} timed out (>{timeout}s)",
            "suggestion": "Check VPN connection or network access",
        }
    except Exception as e:
        return {
            "connected": False,
            "error": f"SSH error: {e}",
            "suggestion": None,
        }


def require_ssh_connection(ssh_host: str, timeout: int = 10) -> None:
    """Verify SSH connection is possible, raising SSHConnectionError if not.

    Call this at the start of operations that require SSH to provide
    clear, actionable error messages instead of generic timeouts.

    Args:
        ssh_host: SSH host alias

    Raises:
        SSHConnectionError: If connection is not possible
    """
    if is_local_host(ssh_host):
        return

    result = check_ssh_connection(ssh_host, timeout=timeout)
    if not result["connected"]:
        raise SSHConnectionError(
            ssh_host=ssh_host,
            message=result["error"],
            suggestion=result["suggestion"],
        )


# ============================================================================
# SSH Socket Health Management
# ============================================================================


def get_ssh_socket_dir() -> Path:
    """Get the SSH control socket directory from config or default."""
    ssh_config_path = Path.home() / ".ssh" / "config"

    # Default socket directory
    socket_dir = Path.home() / ".ssh" / "sockets"

    if ssh_config_path.exists():
        try:
            content = ssh_config_path.read_text()
            # Look for ControlPath directive
            for line in content.splitlines():
                line = line.strip()
                if line.lower().startswith("controlpath"):
                    # Extract directory from path pattern
                    # e.g., "ControlPath ~/.ssh/sockets/%r@%h-%p"
                    path_pattern = line.split(None, 1)[1]
                    # Expand ~ and get directory portion
                    path_pattern = path_pattern.replace("~", str(Path.home()))
                    socket_dir = Path(path_pattern).parent
                    break
        except Exception:
            pass

    return socket_dir


def check_ssh_socket(ssh_host: str, timeout: int = 5) -> bool:
    """Check if SSH control master socket is healthy.

    Args:
        ssh_host: SSH host alias (e.g., 'tcv', 'iter')
        timeout: Timeout in seconds for check

    Returns:
        True if socket is healthy or doesn't exist, False if stale
    """
    try:
        result = subprocess.run(
            ["ssh", "-O", "check", ssh_host],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Exit code 0 means master is running and healthy
        # Exit code 255 means no master (also fine - will create new one)
        return result.returncode in (0, 255)
    except subprocess.TimeoutExpired:
        # Timeout usually means stale socket
        logger.warning(f"SSH socket check timed out for {ssh_host}")
        return False
    except Exception as e:
        logger.debug(f"SSH socket check failed for {ssh_host}: {e}")
        return True  # Unknown state, let SSH attempt handle it


def cleanup_stale_socket(ssh_host: str) -> bool:
    """Clean up a stale SSH control master socket.

    Args:
        ssh_host: SSH host alias

    Returns:
        True if cleanup was performed, False otherwise
    """
    socket_dir = get_ssh_socket_dir()

    # Try graceful exit first
    try:
        result = subprocess.run(
            ["ssh", "-O", "exit", ssh_host],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            logger.info(f"Gracefully closed SSH master for {ssh_host}")
            return True
    except subprocess.TimeoutExpired:
        pass  # Master is stuck, need to remove socket file
    except Exception:
        pass

    # Find and remove socket files matching this host
    if socket_dir.exists():
        for socket_file in socket_dir.glob(f"*{ssh_host}*"):
            if socket_file.is_socket():
                try:
                    socket_file.unlink()
                    logger.info(f"Removed stale socket: {socket_file}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to remove socket {socket_file}: {e}")

    return False


def cleanup_stale_sockets(ssh_hosts: list[str] | None = None) -> dict[str, bool]:
    """Check and cleanup stale SSH sockets for given hosts.

    Args:
        ssh_hosts: List of SSH host aliases to check. If None, check all.

    Returns:
        Dict of {host: was_cleaned} for hosts that needed cleanup
    """
    socket_dir = get_ssh_socket_dir()
    cleaned = {}

    if ssh_hosts is None and socket_dir.exists():
        # Get all hosts from socket files
        ssh_hosts = []
        for socket_file in socket_dir.iterdir():
            if socket_file.is_socket():
                # Extract host from socket name (e.g., "user@host-22")
                name = socket_file.name
                if "@" in name:
                    host = name.split("@")[1].rsplit("-", 1)[0]
                    ssh_hosts.append(host)

    if not ssh_hosts:
        return cleaned

    for host in ssh_hosts:
        if not check_ssh_socket(host):
            if cleanup_stale_socket(host):
                cleaned[host] = True
            else:
                cleaned[host] = False

    return cleaned


def ensure_ssh_healthy(ssh_host: str) -> None:
    """Ensure SSH connection to host is healthy, cleaning stale sockets if needed.

    This is called automatically by run_command() on first SSH access to each host.
    Uses caching to only check once per host per process lifetime.

    Args:
        ssh_host: SSH host alias to check
    """
    if is_local_host(ssh_host):
        return

    if not check_ssh_socket(ssh_host):
        logger.info(f"Detected stale SSH socket for {ssh_host}, cleaning up...")
        cleanup_stale_socket(ssh_host)


# Track which hosts we've already verified this session
_verified_hosts: set[str] = set()


def _ensure_ssh_healthy_once(ssh_host: str) -> None:
    """Check SSH health once per host per process lifetime."""
    if ssh_host in _verified_hosts:
        return
    ensure_ssh_healthy(ssh_host)
    _verified_hosts.add(ssh_host)


# PATH directories to prepend for SSH commands (non-interactive shells don't load full profile)
SSH_PATH_DIRS = ["$HOME/bin", "$HOME/.local/bin"]


def _prepend_path_setup(cmd: str) -> str:
    """Prepend PATH setup to command for non-interactive SSH shells.

    Non-interactive SSH shells (bash without -l) don't source ~/.profile or
    the interactive section of ~/.bashrc, so ~/bin and ~/.local/bin are often
    not in PATH. This ensures tools installed by our installer are available.

    Args:
        cmd: Original command

    Returns:
        Command with PATH setup prepended
    """
    path_dirs = ":".join(SSH_PATH_DIRS)
    return f'export PATH="{path_dirs}:$PATH" && {cmd}'


# ============================================================================
# Hostname Resolution
# ============================================================================


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
def _parse_ssh_config_host(ssh_host: str) -> tuple[str | None, bool]:
    """Parse ~/.ssh/config to get HostName and proxy status for an alias.

    Args:
        ssh_host: SSH host alias (e.g., 'tcv')

    Returns:
        Tuple of (resolved HostName or None, has_proxy)
        has_proxy is True if ProxyJump or ProxyCommand is configured
    """
    ssh_config_path = Path.home() / ".ssh" / "config"
    if not ssh_config_path.exists():
        return None, False

    try:
        content = ssh_config_path.read_text()
    except Exception:
        return None, False

    # Parse SSH config - look for Host block matching ssh_host
    current_host = None
    hostname = None
    has_proxy = False

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Match "Host <pattern>" line
        host_match = re.match(r"^Host\s+(.+)$", line, re.IGNORECASE)
        if host_match:
            # If we were in a matching block and found hostname, return it
            if current_host and hostname is not None:
                return hostname, has_proxy
            hosts = host_match.group(1).split()
            current_host = ssh_host if ssh_host in hosts else None
            hostname = None
            has_proxy = False
            continue

        # If we're in the right Host block, look for HostName and Proxy
        if current_host:
            hostname_match = re.match(r"^HostName\s+(.+)$", line, re.IGNORECASE)
            if hostname_match:
                hostname = hostname_match.group(1).strip()
                continue

            proxy_match = re.match(r"^Proxy(Jump|Command)\s+", line, re.IGNORECASE)
            if proxy_match:
                has_proxy = True

    # Handle case where matching block was at end of file
    if current_host and hostname is not None:
        return hostname, has_proxy

    return None, False


def _is_local_address(hostname: str) -> bool:
    """Check if *hostname* resolves to an IP bound on a local interface.

    Attempts ``socket.bind()`` for each resolved address — only succeeds if
    the OS owns the address.  Useful when the machine sits behind a VIP or
    load-balancer whose external IPs differ from the login node's own.
    """
    try:
        addrs = {info[4][0] for info in socket.getaddrinfo(hostname, None)}
    except Exception:
        return False

    for addr in addrs:
        family = socket.AF_INET6 if ":" in addr else socket.AF_INET
        try:
            s = socket.socket(family, socket.SOCK_STREAM)
            s.bind((addr, 0))
            s.close()
            return True
        except OSError:
            continue
    return False


# ─── Facility-based local host detection ────────────────────────────────────


@lru_cache(maxsize=1)
def _get_facility_local_hosts() -> frozenset[str]:
    """Return host aliases declared local via facility configs.

    Reads ``login_nodes`` (FQDN glob patterns) and ``local_hosts`` (SSH aliases)
    from each facility's public YAML. If the current machine's FQDN matches
    any ``login_nodes`` pattern, returns that facility's ``local_hosts``.
    """
    import fnmatch

    from imas_codex.discovery.base.facility import (
        get_facility,
        list_facilities,
    )

    fqdn = socket.getfqdn().lower()
    hostname = socket.gethostname().lower()
    hosts: set[str] = set()

    for facility_id in list_facilities():
        try:
            config = get_facility(facility_id)
            login_nodes = config.get("login_nodes", [])
            local_hosts = config.get("local_hosts", [])

            # Check if current FQDN or hostname matches any login_nodes
            # pattern.  SLURM compute nodes may not resolve a full FQDN
            # (e.g. "98dci4-gpu-0002" without ".iter.org"), so we test
            # both the FQDN and the short hostname.
            for pattern in login_nodes:
                pat_lower = pattern.lower()
                if fnmatch.fnmatch(fqdn, pat_lower) or fnmatch.fnmatch(
                    hostname, pat_lower
                ):
                    # We're on this facility — add its local_hosts
                    hosts.update(h.lower() for h in local_hosts)
                    break
        except Exception:
            # Config loading errors shouldn't break execution
            pass

    return frozenset(hosts)


@lru_cache(maxsize=1)
def _get_configured_local_hosts() -> frozenset[str]:
    """Return host aliases declared local via facility config or env var.

    Sources (merged, case-insensitive):

    1. Facility configs — ``login_nodes`` (FQDN patterns) and
       ``local_hosts`` (SSH aliases) in public facility YAML.
    2. ``IMAS_CODEX_LOCAL_HOSTS`` env var — comma-separated,
       session-level override for debugging/testing only (do NOT put in .env).
    """
    hosts: set[str] = set()

    # Facility configs (primary source)
    hosts.update(_get_facility_local_hosts())

    # Env var (session-level override — NOT for .env files)
    raw = os.environ.get("IMAS_CODEX_LOCAL_HOSTS", "")
    hosts.update(h.strip().lower() for h in raw.split(",") if h.strip())

    return frozenset(hosts)


def is_local_host(ssh_host: str | None) -> bool:
    """Determine if an ssh_host refers to the local machine.

    Resolution order:

    1. ``None`` / ``"local"`` → always local
    2. Facility-configured local hosts — ``login_nodes`` (FQDN patterns) and
       ``local_hosts`` (SSH aliases) in ``<facility>.yaml``. If the
       current machine's FQDN matches a facility's ``login_nodes`` pattern,
       that facility's ``local_hosts`` are treated as local.
    3. Session override — ``IMAS_CODEX_LOCAL_HOSTS`` env var (debugging only,
       do NOT put in ``.env`` as it travels with ``config secrets push``).
    4. Hostname / FQDN / short-name match via :func:`_get_local_hostnames`
    5. SSH config resolution (``HostName`` via ``~/.ssh/config``)
    6. Bind probe — resolves the candidate via DNS and tries
       ``socket.bind()`` to check if any resulting IP is on a local
       interface (catches FQDN-to-interface mismatches)

    Args:
        ssh_host: SSH host alias or hostname (None = local).
            Special value "local" explicitly means local execution.

    Returns:
        True if ssh_host refers to local machine
    """
    if ssh_host is None:
        return True

    host_lower = ssh_host.lower()

    # Explicit "local" specifier means run locally
    if host_lower == "local":
        return True

    # XDG config file + env var override
    if host_lower in _get_configured_local_hosts():
        return True

    local_hostnames = _get_local_hostnames()

    # Check if ssh_host is explicitly localhost
    if host_lower in local_hostnames:
        return True

    # Resolve ssh_host via SSH config
    resolved, has_proxy = _parse_ssh_config_host(ssh_host)

    # If there's a proxy (ProxyJump/ProxyCommand), the resolved hostname
    # is relative to the jump host, not local - so never treat as local
    if has_proxy:
        return False

    if resolved and resolved.lower() in local_hostnames:
        return True

    # Last resort: resolve via DNS and check if any IP is bound locally.
    # Check the SSH-resolved name first (more specific), then the raw alias.
    for candidate in dict.fromkeys(filter(None, [resolved, ssh_host])):
        if _is_local_address(candidate):
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

    Automatically checks SSH socket health on first access to each host
    to prevent hangs from stale control master sockets.

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
        # Check SSH health once per host per session (prevents stale socket hangs)
        _ensure_ssh_healthy_once(ssh_host)

        # Wrap command with nice if configured for this host
        nice_level = _get_host_nice_level(ssh_host)
        if nice_level is not None and nice_level > 0:
            cmd = f"nice -n {nice_level} bash -c {shlex.quote(cmd)}"

        # Prepend PATH setup for non-interactive SSH shells
        remote_cmd = _prepend_path_setup(cmd)

        # SSH execution with -T to disable pseudo-terminal allocation
        # This avoids triggering .bashrc on systems where it's loaded for PTY sessions
        result = subprocess.run(
            ["ssh", "-T", ssh_host, remote_cmd],
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

    # For remote bash scripts, prepend PATH setup
    remote_script = script
    if not is_local and interpreter == "bash":
        path_dirs = ":".join(SSH_PATH_DIRS)
        remote_script = f'export PATH="{path_dirs}:$PATH"\n{script}'

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
        # Check SSH health once per host per session (prevents stale socket hangs)
        _ensure_ssh_healthy_once(ssh_host)

        # Wrap interpreter with nice if configured for this host
        nice_level = _get_host_nice_level(ssh_host)
        if nice_level is not None and nice_level > 0:
            interp_cmd = ["nice", "-n", str(nice_level)] + interp_cmd

        # SSH execution via stdin - avoids bash -c overhead
        # Use 'interpreter [args]' to read script from stdin
        result = subprocess.run(
            ["ssh", "-T", ssh_host, " ".join(interp_cmd)],
            input=remote_script,
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
    python_command: str = "python3",
    setup_commands: list[str] | None = None,
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
        python_command: Python interpreter to use (default: "python3").
            Can be full path (e.g., "/path/to/python3").
        setup_commands: Shell commands to run before Python (e.g., module loads).
            Only used for remote (SSH) execution.

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
            [python_command, "-c", runner],
            input=json_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    else:
        # Check SSH health once per host per session (prevents stale socket hangs)
        _ensure_ssh_healthy_once(ssh_host)

        # Build remote command: PATH prefix + optional setup + python
        python_cmd = f"{python_command} -c '{runner}'"

        # Wrap Python command with nice if configured for this host
        nice_level = _get_host_nice_level(ssh_host)
        if nice_level is not None and nice_level > 0:
            python_cmd = f"nice -n {nice_level} {python_cmd}"

        parts = [_REMOTE_PATH_PREFIX]
        if setup_commands:
            parts.extend(setup_commands)
        parts.append(python_cmd)
        remote_cmd = " && ".join(parts)

        # SSH execution with JSON piped through
        result = subprocess.run(
            ["ssh", "-T", ssh_host, remote_cmd],
            input=json_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    # Return only stdout — stderr must not contaminate structured JSON output.
    # Remote Python scripts (check_signals_batch, extract_tdi_functions, etc.)
    # return JSON on stdout. MDSplus and other libraries may print warnings to
    # stderr (e.g., libvaccess.so loading, TDI function debug output).
    output = result.stdout

    if result.stderr:
        logger.debug(
            "run_python_script(%s) stderr: %s",
            script_name,
            result.stderr[:500],
        )

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, script_name, result.stdout, result.stderr
        )

    return output.strip()


async def async_run_python_script(
    script_name: str,
    input_data: dict | list | None = None,
    ssh_host: str | None = None,
    timeout: int = 60,
    python_command: str = "python3",
    setup_commands: list[str] | None = None,
) -> str:
    """Async version of run_python_script using asyncio subprocesses.

    Uses asyncio.create_subprocess_exec() instead of subprocess.run(),
    making the call fully cancellable by asyncio task cancellation.

    Args:
        script_name: Script filename (e.g., "scan_directories.py")
        input_data: Dict/list to pass as JSON on stdin (None = no input)
        ssh_host: SSH host to connect to (None = local)
        timeout: Command timeout in seconds
        python_command: Python interpreter to use (default: "python3").
            Can be full path (e.g., "/path/to/python3").
        setup_commands: Shell commands to run before Python (e.g., module loads).
            Only used for remote (SSH) execution.

    Returns:
        Script output (stdout)

    Raises:
        FileNotFoundError: If script doesn't exist
        subprocess.CalledProcessError: On non-zero exit
        TimeoutError: If command exceeds timeout
        asyncio.CancelledError: If task is cancelled
    """
    import asyncio
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
    runner = (
        f"import base64,subprocess,sys;"
        f's=base64.b64decode("{script_b64}");'
        f'exec(compile(s,"script","exec"))'
    )

    if is_local:
        cmd = [python_command, "-c", runner]
    else:
        # Check SSH health once per host per session (sync, cached)
        _ensure_ssh_healthy_once(ssh_host)
        # Build remote command: PATH prefix + optional setup + python
        python_cmd = f"{python_command} -c '{runner}'"

        # Wrap Python command with nice if configured for this host
        nice_level = _get_host_nice_level(ssh_host)
        if nice_level is not None and nice_level > 0:
            python_cmd = f"nice -n {nice_level} {python_cmd}"

        parts = [_REMOTE_PATH_PREFIX]
        if setup_commands:
            parts.extend(setup_commands)
        parts.append(python_cmd)
        remote_cmd = " && ".join(parts)
        cmd = ["ssh", "-T", ssh_host, remote_cmd]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(json_input.encode()),
            timeout=timeout,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise subprocess.TimeoutExpired(cmd, timeout) from None
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise

    stdout = stdout_bytes.decode()
    stderr = stderr_bytes.decode()

    if stderr:
        logger.debug(
            "async_run_python_script(%s) stderr: %s",
            script_name,
            stderr[:500],
        )

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, script_name, stdout, stderr
        )

    return stdout.strip()
