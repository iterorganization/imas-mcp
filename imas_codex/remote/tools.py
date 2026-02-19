"""
Fast CLI tools management for local and remote facilities.

This module provides:
- Tool configuration loading from remote_tools.yaml
- Architecture detection (local and remote)
- Tool availability checking
- Installation script generation
- Unified execution interface (local vs SSH)

The key abstraction is `run()` which executes commands either locally
or via SSH depending on whether we're on the target facility.

Auto-detection logic:
1. If facility is None, run locally
2. Parse ~/.ssh/config to get HostName for the facility's ssh_host
3. Compare resolved hostname to current machine's hostname/FQDN
4. If they match (or ssh_host is localhost), run locally

Module Structure:
- executor.py: Low-level run primitives (no facility imports, avoids cycles)
- tools.py: Facility-aware wrappers + tool management (this file)
"""

import logging
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Import low-level executor (no circular import risk)
from imas_codex.remote.executor import (
    configure_host_nice,
    is_local_host,
    run_command,
    run_script_via_stdin,
)

logger = logging.getLogger(__name__)


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings.

    Args:
        v1: First version string (e.g., '2.35.2')
        v2: Second version string (e.g., '2.30.0')

    Returns:
        -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """

    # Parse versions into numeric tuples
    def parse(v: str) -> tuple[int, ...]:
        parts = []
        for part in v.split("."):
            # Extract leading digits only (handle versions like '2.35.2-ubuntu')
            digits = ""
            for c in part:
                if c.isdigit():
                    digits += c
                else:
                    break
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    p1, p2 = parse(v1), parse(v2)
    # Pad shorter tuple with zeros
    max_len = max(len(p1), len(p2))
    p1 = p1 + (0,) * (max_len - len(p1))
    p2 = p2 + (0,) * (max_len - len(p2))

    if p1 < p2:
        return -1
    elif p1 > p2:
        return 1
    return 0


def _resolve_ssh_host(facility: str | None) -> str | None:
    """Resolve facility ID to SSH host.

    Args:
        facility: Facility ID or None.
            Special value "local" explicitly means local execution.

    Returns:
        SSH host string or None for local execution
    """
    if facility is None:
        return None

    # Explicit "local" pseudo-facility means run locally
    if facility.lower() == "local":
        return None

    # Import here to avoid circular import at module load time
    from imas_codex.discovery.base.facility import get_facility

    try:
        config = get_facility(facility)
        ssh_host = config.get("ssh_host", facility)

        # Auto-register nice level for this host from facility config
        nice_level = config.get("nice_level")
        if nice_level is not None:
            configure_host_nice(ssh_host, nice_level)

        return ssh_host
    except ValueError:
        return facility


def is_local_facility(facility: str | None) -> bool:
    """Determine if a facility should be accessed locally (no SSH).

    Args:
        facility: Facility ID or None.
            Special value "local" explicitly means local execution.

    Returns:
        True if commands should run locally, False if SSH needed
    """
    # Explicit "local" pseudo-facility
    if facility is not None and facility.lower() == "local":
        return True

    ssh_host = _resolve_ssh_host(facility)
    return is_local_host(ssh_host)


# PATH prefix to ensure tools in ~/bin are accessible via SSH
PATH_PREFIX = 'export PATH="$HOME/bin:$HOME/.local/bin:$PATH" && '


def run(
    cmd: str,
    facility: str | None = None,
    timeout: int = 60,
    check: bool = False,
) -> str:
    """Execute command locally or via SSH depending on facility.

    For remote execution, automatically prepends PATH to ensure
    tools installed in ~/bin (like uv) are accessible.

    Args:
        cmd: Shell command to execute
        facility: Facility ID (None = local)
        timeout: Command timeout in seconds
        check: Raise exception on non-zero exit

    Returns:
        Command output (stdout + stderr)
    """
    is_local = is_local_host(facility)
    ssh_host = _resolve_ssh_host(facility) if not is_local else None

    # Prepend PATH for remote execution to find ~/bin tools
    if not is_local:
        cmd = PATH_PREFIX + cmd

    return run_command(cmd, ssh_host=ssh_host, timeout=timeout, check=check)


def run_script(
    script: str,
    facility: str | None = None,
    timeout: int = 60,
    check: bool = False,
) -> str:
    """Execute a multi-line script via stdin to avoid bash -c overhead.

    Args:
        script: Multi-line bash script to execute
        facility: Facility ID (None = local)
        timeout: Command timeout in seconds
        check: Raise exception on non-zero exit

    Returns:
        Command output (stdout + stderr)
    """
    ssh_host = _resolve_ssh_host(facility) if not is_local_host(facility) else None
    return run_script_via_stdin(script, ssh_host=ssh_host, timeout=timeout, check=check)


@dataclass
class ToolRelease:
    """Release configuration for a tool."""

    repo: str
    version: str
    asset_pattern: str
    strip_components: int = 0
    binary_path: str = ""
    is_binary: bool = False
    arch_map: dict[str, str] = field(default_factory=dict)
    tag_prefix: str = "v"  # Most repos use 'v', set to '' for repos like ripgrep


@dataclass
class RemoteTool:
    """Configuration for a remote CLI tool."""

    key: str  # e.g., 'rg', 'fd'
    name: str  # e.g., 'ripgrep', 'fd'
    purpose: str
    binary: str
    fallback: str | None
    releases: ToolRelease
    examples: dict[str, str | None]
    required: bool = False
    min_version: str | None = None  # Minimum acceptable system version
    version_command: str | None = None  # Custom version command (default: --version)
    system_only: bool = (
        False  # True if tool requires system package manager (no auto-install)
    )

    @property
    def github_url(self) -> str:
        """Get GitHub releases URL."""
        return f"https://github.com/{self.releases.repo}/releases"

    def get_download_url(self, arch: str = "x86_64") -> str:
        """Get download URL for specific architecture."""
        # Map architecture if needed
        arch_mapped = arch
        if self.releases.arch_map:
            arch_mapped = self.releases.arch_map.get(arch, arch)

        # Format asset pattern
        asset = self.releases.asset_pattern.format(
            version=self.releases.version,
            arch=self._get_musl_arch(arch),
            arch_simple=arch_mapped,
        )

        return f"https://github.com/{self.releases.repo}/releases/download/{self.releases.tag_prefix}{self.releases.version}/{asset}"

    def _get_musl_arch(self, arch: str) -> str:
        """Get musl architecture string."""
        arch_map = {
            "x86_64": "x86_64-unknown-linux-musl",
            "aarch64": "aarch64-unknown-linux-musl",
        }
        return arch_map.get(arch, arch)

    def get_install_command(self, arch: str = "x86_64") -> str:
        """Generate install command for this tool."""
        url = self.get_download_url(arch)
        install_dir = "~/bin"

        # Map architecture for simplified naming (amd64/arm64)
        arch_simple = (
            self.releases.arch_map.get(arch, arch) if self.releases.arch_map else arch
        )

        if self.releases.is_binary:
            # Direct binary download
            return f'curl -sL "{url}" -o {install_dir}/{self.binary} && chmod +x {install_dir}/{self.binary}'

        # Tarball extraction
        # Only filter specific files if binary_path is explicitly set
        file_filter = ""
        if self.releases.binary_path:
            binary_path = self.releases.binary_path.format(
                version=self.releases.version,
                arch=self._get_musl_arch(arch),
                arch_simple=arch_simple,
            )
            file_filter = f" {binary_path}"

        if self.releases.strip_components > 0:
            strip = f"--strip-components={self.releases.strip_components}"
            return f'curl -sL "{url}" | tar xz {strip} -C {install_dir}{file_filter}'
        else:
            return f'curl -sL "{url}" | tar xz -C {install_dir}{file_filter}'


@dataclass
class RemoteToolsConfig:
    """Complete remote tools configuration."""

    version: str
    install_dir: str
    architectures: dict[str, str]
    required: dict[str, RemoteTool]
    optional: dict[str, RemoteTool]

    @property
    def all_tools(self) -> dict[str, RemoteTool]:
        """Get all tools (required + optional)."""
        return {**self.required, **self.optional}

    def get_tool(self, key: str) -> RemoteTool | None:
        """Get tool by key."""
        return self.all_tools.get(key)


@lru_cache(maxsize=1)
def load_remote_tools() -> RemoteToolsConfig:
    """Load remote tools configuration from YAML.

    Returns:
        RemoteToolsConfig with all tool definitions
    """
    config_path = Path(__file__).parent.parent / "config" / "remote_tools.yaml"

    with config_path.open() as f:
        data = yaml.safe_load(f)

    def parse_tool(key: str, tool_data: dict[str, Any], required: bool) -> RemoteTool:
        releases_data = tool_data.get("releases", {})
        return RemoteTool(
            key=key,
            name=tool_data.get("name", key),
            purpose=tool_data.get("purpose", ""),
            binary=tool_data.get("binary", key),
            fallback=tool_data.get("fallback"),
            releases=ToolRelease(
                repo=releases_data.get("repo", ""),
                version=releases_data.get("version", ""),
                asset_pattern=releases_data.get("asset_pattern", ""),
                strip_components=releases_data.get("strip_components", 0),
                binary_path=releases_data.get("binary_path", ""),
                is_binary=releases_data.get("is_binary", False),
                arch_map=releases_data.get("arch_map", {}),
                tag_prefix=releases_data.get("tag_prefix", "v"),
            ),
            examples=tool_data.get("examples", {}),
            required=required,
            min_version=tool_data.get("min_version"),
            version_command=tool_data.get("version_command"),
            system_only=tool_data.get("system_only", False),
        )

    required = {
        key: parse_tool(key, tool_data, required=True)
        for key, tool_data in data.get("required", {}).items()
    }

    optional = {
        key: parse_tool(key, tool_data, required=False)
        for key, tool_data in data.get("optional", {}).items()
    }

    install_config = data.get("install", {})

    return RemoteToolsConfig(
        version=data.get("version", "1.0"),
        install_dir=install_config.get("directory", "~/bin"),
        architectures=install_config.get("architectures", {}),
        required=required,
        optional=optional,
    )


def detect_architecture(facility: str | None = None) -> str:
    """Detect CPU architecture of target system.

    Args:
        facility: Facility ID (None = local)

    Returns:
        Architecture string: 'x86_64' or 'aarch64'
    """
    output = run("uname -m", facility=facility)
    arch = output.strip().split("\n")[0]

    # Normalize architecture names
    arch_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }

    return arch_map.get(arch, arch)


def check_tool(
    tool_key: str,
    facility: str | None = None,
) -> dict[str, Any]:
    """Check if a specific tool is available and meets version requirements.

    Args:
        tool_key: Tool key (e.g., 'rg', 'fd', 'git')
        facility: Facility ID (None = local)

    Returns:
        Dict with available, version, path, meets_min_version
    """
    import re

    config = load_remote_tools()
    tool = config.get_tool(tool_key)

    if not tool:
        return {"error": f"Unknown tool: {tool_key}"}

    # Build version check command
    # Use custom version_command if specified, otherwise default to --version
    version_arg = tool.version_command or "--version"
    # Expand ~ explicitly and try multiple paths
    check_cmd = (
        f"eval ~/bin/{tool.binary} {version_arg} 2>/dev/null || "
        f"$HOME/bin/{tool.binary} {version_arg} 2>/dev/null || "
        f"{tool.binary} {version_arg} 2>/dev/null"
    )

    try:
        output = run(check_cmd, facility=facility, timeout=10)
        # Check for valid version output (not empty/error placeholder)
        if (
            output
            and "(no output)" not in output.lower()
            and "[stderr]" not in output.lower()
        ):
            # Extract version number - must be a real version
            version_match = re.search(r"(\d+\.\d+\.?\d*)", output)
            if not version_match:
                # No valid version found
                raise ValueError("No version number in output")

            version = version_match.group(1)

            # Check which path - expand ~ explicitly
            which_cmd = (
                f"eval echo ~/bin/{tool.binary} 2>/dev/null || "
                f"which {tool.binary} 2>/dev/null"
            )
            path_output = run(which_cmd, facility=facility, timeout=5)
            path = (
                path_output.split("\n")[0]
                if path_output and "(no output)" not in path_output.lower()
                else None
            )

            # Check if version meets minimum requirement
            meets_min_version = True
            if tool.min_version:
                meets_min_version = compare_versions(version, tool.min_version) >= 0

            return {
                "available": True,
                "version": version,
                "path": path,
                "required": tool.required,
                "purpose": tool.purpose,
                "min_version": tool.min_version,
                "meets_min_version": meets_min_version,
            }
    except Exception as e:
        logger.debug(f"Tool check failed for {tool_key}: {e}")

    return {
        "available": False,
        "version": None,
        "path": None,
        "required": tool.required,
        "purpose": tool.purpose,
        "fallback": tool.fallback,
        "min_version": tool.min_version,
        "meets_min_version": False,
    }


def check_all_tools(facility: str | None = None) -> dict[str, Any]:
    """Check availability of all remote tools.

    Args:
        facility: Facility ID (None = local)

    Returns:
        Dict with tool statuses and summary
    """
    config = load_remote_tools()
    results: dict[str, Any] = {
        "facility": facility or "local",
        "tools": {},
        "required_ok": True,
        "missing_required": [],
        "missing_optional": [],
        "version_too_old": [],  # Tools present but below min_version
    }

    for key in config.all_tools:
        status = check_tool(key, facility=facility)
        results["tools"][key] = status

        if not status.get("available"):
            if status.get("required"):
                results["required_ok"] = False
                results["missing_required"].append(key)
            else:
                results["missing_optional"].append(key)
        elif not status.get("meets_min_version", True):
            # Tool is available but version is too old
            results["version_too_old"].append(
                {
                    "tool": key,
                    "version": status.get("version"),
                    "min_version": status.get("min_version"),
                }
            )
            if status.get("required"):
                results["required_ok"] = False

    return results


# Marker comment used to detect if PATH block is already in .bashrc/.profile
_PATH_MARKER = "# imas-codex PATH setup"

# The PATH block inserted into shell rc files
_PATH_BLOCK = f"""{_PATH_MARKER}
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
"""


def ensure_path(facility: str | None = None) -> str:
    """Ensure ~/bin and ~/.local/bin are in PATH for all shell sessions.

    Inserts a PATH export block directly into ~/.bashrc (near the top,
    before any non-interactive guard) and ~/.profile. This ensures tools
    installed by ``imas-codex tools install`` (uv, rg, fd, Python via uv)
    are available for:

    * Interactive login shells  (bash_profile → bashrc)
    * Interactive non-login     (bashrc)
    * Non-interactive SSH       (ssh host 'cmd' → bashrc, before guard)
    * Login shells via sh/dash  (~/.profile)

    Uses a marker comment for idempotency — safe to call repeatedly.
    Also cleans up the legacy ~/.imas-codex.env approach if present.

    Args:
        facility: Facility ID (None = local)

    Returns:
        Status message describing what was configured
    """
    messages = []

    # --- ~/.bashrc -----------------------------------------------------------
    messages.append(_ensure_path_in_rc("~/.bashrc", facility))

    # --- ~/.profile ----------------------------------------------------------
    # Covers login shells that use sh/dash (and some systems where
    # .bash_profile doesn't source .bashrc).
    messages.append(_ensure_path_in_rc("~/.profile", facility))

    # --- Clean up legacy .imas-codex.env approach ----------------------------
    _cleanup_legacy_env(facility, messages)

    return "; ".join(messages)


def _ensure_path_in_rc(rc_file: str, facility: str | None) -> str:
    """Insert the PATH block into a shell rc file if not already present.

    For ~/.bashrc, inserts before the non-interactive guard so that
    ``ssh host 'command'`` inherits the PATH.

    Args:
        rc_file: Shell rc file path (e.g. "~/.bashrc")
        facility: Facility ID (None = local)

    Returns:
        Human-readable status message
    """
    # Check if already present
    check = (
        f'grep -F "{_PATH_MARKER}" {rc_file} >/dev/null 2>&1 && echo "yes" || echo "no"'
    )
    if run(check, facility=facility, timeout=10).strip() == "yes":
        return f"{rc_file} already configured"

    # Check if the file exists at all
    exists = f'test -f {rc_file} && echo "yes" || echo "no"'
    if run(exists, facility=facility, timeout=5).strip() != "yes":
        # Create it with just the PATH block
        create = f"cat > {rc_file} << 'IMASEOF'\n{_PATH_BLOCK}IMASEOF"
        run(create, facility=facility)
        return f"Created {rc_file} with PATH"

    # For .bashrc: insert before the non-interactive guard if present
    if rc_file == "~/.bashrc":
        insert_script = f"""
# Look for the standard non-interactive guard patterns
for pattern in '\\[\\[ \\$- != \\*i\\*' '\\[ -z "\\$PS1" \\]' 'case \\$-'; do
    LINE=$(grep -n "$pattern" {rc_file} 2>/dev/null | head -1 | cut -d: -f1)
    if [ -n "$LINE" ]; then
        sed -i "${{LINE}}i\\\\{_PATH_MARKER}\\nexport PATH=\\"\\$HOME/bin:\\$HOME/.local/bin:\\$PATH\\"\\n" {rc_file}
        echo "inserted:$LINE"
        exit 0
    fi
done
# No guard found — prepend after first comment block
awk 'NR==1{{print; print ""; print "{_PATH_MARKER}"; print "export PATH=\\"$HOME/bin:$HOME/.local/bin:$PATH\\""; print ""; next}}1' {rc_file} > {rc_file}.tmp && mv {rc_file}.tmp {rc_file}
echo "prepended"
"""
        result = run(insert_script, facility=facility).strip()
        if result.startswith("inserted"):
            return f"{rc_file} PATH inserted before guard (line {result.split(':')[1]})"
        return f"{rc_file} PATH prepended"

    # For .profile and others: just append
    append = f'echo "" >> {rc_file} && echo "{_PATH_MARKER}" >> {rc_file} && echo \'export PATH="$HOME/bin:$HOME/.local/bin:$PATH"\' >> {rc_file}'
    run(append, facility=facility)
    return f"{rc_file} PATH appended"


def _cleanup_legacy_env(facility: str | None, messages: list[str]) -> None:
    """Remove the old ~/.imas-codex.env indirection if present."""
    # Remove source line from .bashrc
    check = (
        'grep -F "imas-codex.env" ~/.bashrc >/dev/null 2>&1 && echo "yes" || echo "no"'
    )
    if run(check, facility=facility, timeout=5).strip() == "yes":
        # Remove the source line and its preceding comment
        run(
            "sed -i '/# Source imas-codex environment/d; /imas-codex\\.env/d' ~/.bashrc",
            facility=facility,
        )
        messages.append("Removed legacy .imas-codex.env source from .bashrc")

    # Remove the env file itself
    check_file = 'test -f ~/.imas-codex.env && echo "yes" || echo "no"'
    if run(check_file, facility=facility, timeout=5).strip() == "yes":
        run("rm -f ~/.imas-codex.env", facility=facility)
        messages.append("Removed ~/.imas-codex.env")


def check_internet_access(facility: str | None = None, timeout: int = 5) -> bool:
    """Check if facility has outbound internet access to GitHub.

    Args:
        facility: Facility ID (None = local)
        timeout: Connection timeout in seconds

    Returns:
        True if GitHub is reachable, False otherwise
    """
    # Use curl with connect timeout to check GitHub reachability
    check_cmd = f"curl -sI --connect-timeout {timeout} https://github.com >/dev/null 2>&1 && echo 'ok' || echo 'fail'"
    try:
        result = run(check_cmd, facility=facility, timeout=timeout + 5)
        return result.strip() == "ok"
    except Exception as e:
        logger.debug(f"Internet access check failed: {e}")
        return False


def _download_locally(url: str, dest_path: Path) -> None:
    """Download a file locally using curl.

    Args:
        url: URL to download
        dest_path: Local destination path

    Raises:
        RuntimeError: If download fails
    """
    result = subprocess.run(
        ["curl", "-sL", "-o", str(dest_path), url],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr}")
    if not dest_path.exists() or dest_path.stat().st_size == 0:
        raise RuntimeError("Download produced empty file")


def _scp_to_remote(local_path: Path, remote_path: str, facility: str) -> None:
    """SCP a file to a remote facility.

    Args:
        local_path: Local file path
        remote_path: Remote destination path (e.g., ~/bin/rg)
        facility: Facility ID (used as SSH host)

    Raises:
        RuntimeError: If SCP fails
    """
    ssh_host = _resolve_ssh_host(facility)
    result = subprocess.run(
        ["scp", "-q", str(local_path), f"{ssh_host}:{remote_path}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr}")


def install_tool_via_scp(
    tool: "RemoteTool",
    arch: str,
    facility: str,
) -> dict[str, Any]:
    """Install a tool by downloading locally and SCPing to remote.

    This is used when the remote facility has no internet access.
    Downloads to a temp file, extracts if needed, SCPs binary to ~/bin,
    and cleans up local temp files.

    Args:
        tool: RemoteTool configuration
        arch: Target architecture (x86_64 or aarch64)
        facility: Remote facility ID

    Returns:
        Dict with success status and details
    """
    url = tool.get_download_url(arch)
    binary_name = tool.binary

    # Map architecture for simplified naming
    arch_simple = (
        tool.releases.arch_map.get(arch, arch) if tool.releases.arch_map else arch
    )

    with tempfile.TemporaryDirectory(prefix="imas_codex_tool_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        try:
            if tool.releases.is_binary:
                # Direct binary download
                local_binary = tmpdir_path / binary_name
                logger.debug(f"Downloading binary: {url}")
                _download_locally(url, local_binary)
                local_binary.chmod(0o755)
            else:
                # Tarball download and extraction
                tarball = tmpdir_path / "download.tar.gz"
                logger.debug(f"Downloading tarball: {url}")
                _download_locally(url, tarball)

                # Extract tarball locally
                extract_dir = tmpdir_path / "extracted"
                extract_dir.mkdir()

                strip = tool.releases.strip_components
                result = subprocess.run(
                    ["tar", "xzf", str(tarball), "-C", str(extract_dir)]
                    + ([f"--strip-components={strip}"] if strip > 0 else []),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Extraction failed: {result.stderr}")

                # Find the binary
                if tool.releases.binary_path:
                    # Use configured binary path
                    binary_rel_path = tool.releases.binary_path.format(
                        version=tool.releases.version,
                        arch=tool._get_musl_arch(arch),
                        arch_simple=arch_simple,
                    )
                    # After strip, the binary should be at root level or close
                    if strip > 0:
                        # With strip, binary is at top level
                        local_binary = extract_dir / binary_name
                    else:
                        local_binary = extract_dir / binary_rel_path
                else:
                    local_binary = extract_dir / binary_name

                if not local_binary.exists():
                    # Search for binary in extracted files
                    found = list(extract_dir.rglob(binary_name))
                    if found:
                        local_binary = found[0]
                    else:
                        raise RuntimeError(
                            f"Binary {binary_name} not found in extracted archive"
                        )

                local_binary.chmod(0o755)

            # Ensure ~/bin exists on remote
            run("mkdir -p ~/bin", facility=facility)

            # SCP binary to remote
            remote_path = f"~/bin/{binary_name}"
            logger.debug(f"SCPing {local_binary} to {facility}:{remote_path}")
            _scp_to_remote(local_binary, remote_path, facility)

            # Make executable on remote (in case permissions weren't preserved)
            run(f"chmod +x ~/bin/{binary_name}", facility=facility)

            return {"success": True, "method": "scp"}

        except Exception as e:
            logger.error(f"SCP install failed for {tool.key}: {e}")
            return {"success": False, "error": str(e), "method": "scp"}
        # Temp directory auto-cleaned by context manager


def install_tool(
    tool_key: str,
    facility: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Install a specific tool.

    Only installs if:
    - Tool is not available, OR
    - Tool is available but below min_version, OR
    - force=True

    Args:
        tool_key: Tool key (e.g., 'rg', 'fd', 'git')
        facility: Facility ID (None = local)
        force: Reinstall even if already present and meets requirements

    Returns:
        Dict with success status and details
    """
    config = load_remote_tools()
    tool = config.get_tool(tool_key)

    if not tool:
        return {"success": False, "error": f"Unknown tool: {tool_key}"}

    # Check if already installed and meets requirements
    if not force:
        status = check_tool(tool_key, facility=facility)
        if status.get("available"):
            if status.get("meets_min_version", True):
                # System version is good enough
                return {
                    "success": True,
                    "action": "system_sufficient",
                    "version": status.get("version"),
                    "path": status.get("path"),
                    "min_version": tool.min_version,
                }
            else:
                # Version too old
                if tool.system_only:
                    # Cannot auto-install system-only tools
                    return {
                        "success": False,
                        "action": "version_too_old",
                        "error": f"{tool_key} v{status.get('version')} < min v{tool.min_version}. "
                        f"Requires system package manager upgrade (apt/yum/module load).",
                        "version": status.get("version"),
                        "min_version": tool.min_version,
                        "system_only": True,
                    }
                # Can auto-install, proceed
                logger.info(
                    f"{tool_key} v{status.get('version')} < min v{tool.min_version}, installing newer"
                )
        elif tool.system_only:
            # Tool not available and cannot be auto-installed
            return {
                "success": False,
                "action": "not_available",
                "error": f"{tool_key} not found. Requires system package manager "
                f"installation (apt install {tool.binary}, yum install {tool.binary}, "
                f"or module load {tool.binary}).",
                "min_version": tool.min_version,
                "system_only": True,
            }

    # Detect architecture
    arch = detect_architecture(facility=facility)
    logger.info(f"Installing {tool_key} for {arch} on {facility or 'local'}")

    # Ensure ~/bin exists
    run("mkdir -p ~/bin", facility=facility)

    # Ensure PATH is configured
    ensure_path(facility=facility)

    # For remote facilities, check internet access and use SCP fallback if needed
    is_local = is_local_facility(facility)
    use_scp = False

    if not is_local:
        # Check if remote has internet access
        has_internet = check_internet_access(facility=facility, timeout=5)
        if not has_internet:
            logger.info(
                f"No internet access from {facility}, using local download + SCP"
            )
            use_scp = True

    if use_scp:
        # Use SCP fallback: download locally, then SCP to remote
        result = install_tool_via_scp(tool, arch, facility)
        if result.get("success"):
            # Verify installation
            status = check_tool(tool_key, facility=facility)
            if status.get("available"):
                return {
                    "success": True,
                    "action": "installed",
                    "method": "scp",
                    "version": status.get("version"),
                    "path": status.get("path"),
                }
            else:
                return {
                    "success": False,
                    "error": "SCP completed but tool not found",
                    "method": "scp",
                }
        else:
            return {
                "success": False,
                "error": result.get("error", "SCP installation failed"),
                "method": "scp",
            }

    # Direct installation (local or remote with internet)
    try:
        install_cmd = tool.get_install_command(arch)
    except Exception as e:
        return {"success": False, "error": f"Failed to generate install command: {e}"}

    # Execute installation with shorter timeout for remote
    install_timeout = 30 if not is_local else 120
    try:
        output = run(install_cmd, facility=facility, timeout=install_timeout)
        logger.debug(f"Install output: {output}")
    except Exception as e:
        # If direct install fails on remote, try SCP fallback
        if not is_local:
            logger.info(f"Direct install failed, trying SCP fallback: {e}")
            result = install_tool_via_scp(tool, arch, facility)
            if result.get("success"):
                status = check_tool(tool_key, facility=facility)
                if status.get("available"):
                    return {
                        "success": True,
                        "action": "installed",
                        "method": "scp",
                        "version": status.get("version"),
                        "path": status.get("path"),
                    }
            return {
                "success": False,
                "error": f"Both direct and SCP installation failed: {e}",
            }
        return {"success": False, "error": f"Installation failed: {e}"}

    # Verify installation
    status = check_tool(tool_key, facility=facility)
    if status.get("available"):
        return {
            "success": True,
            "action": "installed",
            "version": status.get("version"),
            "path": status.get("path"),
        }
    else:
        return {
            "success": False,
            "error": "Installation completed but tool not found",
            "install_output": output,
        }


def install_all_tools(
    facility: str | None = None,
    required_only: bool = False,
    force: bool = False,
    on_progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Install all remote tools.

    Args:
        facility: Facility ID (None = local)
        required_only: Only install required tools
        force: Reinstall even if already present
        on_progress: Optional callback called after each tool with (tool_key, result)

    Returns:
        Dict mapping tool_key -> installation result dict
    """
    config = load_remote_tools()
    results: dict[str, Any] = {}

    tools_to_install = config.required if required_only else config.all_tools

    for key in tools_to_install:
        result = install_tool(key, facility=facility, force=force)
        results[key] = result
        if on_progress:
            on_progress(key, result)

    return results


def check_outdated_tools(
    facility: str | None = None,
) -> list[dict[str, Any]]:
    """Find tools that are installed but older than the configured version.

    Compares the installed version of each tool against the version
    specified in remote_tools.yaml. System-only tools are skipped.

    Args:
        facility: Facility ID (None = local)

    Returns:
        List of dicts with tool key, installed version, configured version
    """
    config = load_remote_tools()
    outdated = []

    for key, tool in config.all_tools.items():
        if tool.system_only:
            continue
        if not tool.releases.version:
            continue

        status = check_tool(key, facility=facility)
        if not status.get("available"):
            continue

        installed_version = status.get("version", "")
        configured_version = tool.releases.version

        if (
            installed_version
            and compare_versions(installed_version, configured_version) < 0
        ):
            outdated.append(
                {
                    "key": key,
                    "name": tool.name,
                    "installed": installed_version,
                    "configured": configured_version,
                    "required": tool.required,
                }
            )

    return outdated
