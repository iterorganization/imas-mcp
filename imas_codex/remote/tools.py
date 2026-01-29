"""
Fast CLI tools management for local and remote facilities.

This module provides:
- Tool configuration loading from fast_tools.yaml
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
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Import low-level executor (no circular import risk)
from imas_codex.remote.executor import (
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
        facility: Facility ID or None

    Returns:
        SSH host string or None for local execution
    """
    if facility is None:
        return None

    # Import here to avoid circular import at module load time
    from imas_codex.discovery.facility import get_facility

    try:
        config = get_facility(facility)
        return config.get("ssh_host", facility)
    except ValueError:
        return facility


def is_local_facility(facility: str | None) -> bool:
    """Determine if a facility should be accessed locally (no SSH).

    Args:
        facility: Facility ID or None

    Returns:
        True if commands should run locally, False if SSH needed
    """
    ssh_host = _resolve_ssh_host(facility)
    return is_local_host(ssh_host)


def run(
    cmd: str,
    facility: str | None = None,
    timeout: int = 60,
    check: bool = False,
) -> str:
    """Execute command locally or via SSH depending on facility.

    Args:
        cmd: Shell command to execute
        facility: Facility ID (None = local)
        timeout: Command timeout in seconds
        check: Raise exception on non-zero exit

    Returns:
        Command output (stdout + stderr)
    """
    ssh_host = _resolve_ssh_host(facility) if not is_local_host(facility) else None
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


@dataclass
class FastTool:
    """Configuration for a fast CLI tool."""

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

        return f"https://github.com/{self.releases.repo}/releases/download/v{self.releases.version}/{asset}"

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
        if self.releases.strip_components > 0:
            strip = f"--strip-components={self.releases.strip_components}"
            # Format binary path with version and arch
            binary_path = self.releases.binary_path.format(
                version=self.releases.version,
                arch=self._get_musl_arch(arch),
                arch_simple=arch_simple,
            )
            return f'curl -sL "{url}" | tar xz {strip} -C {install_dir} {binary_path}'
        else:
            return f'curl -sL "{url}" | tar xz -C {install_dir} {self.binary}'


@dataclass
class FastToolsConfig:
    """Complete fast tools configuration."""

    version: str
    install_dir: str
    architectures: dict[str, str]
    required: dict[str, FastTool]
    optional: dict[str, FastTool]

    @property
    def all_tools(self) -> dict[str, FastTool]:
        """Get all tools (required + optional)."""
        return {**self.required, **self.optional}

    def get_tool(self, key: str) -> FastTool | None:
        """Get tool by key."""
        return self.all_tools.get(key)


@lru_cache(maxsize=1)
def load_fast_tools() -> FastToolsConfig:
    """Load fast tools configuration from YAML.

    Returns:
        FastToolsConfig with all tool definitions
    """
    config_path = Path(__file__).parent.parent / "config" / "fast_tools.yaml"

    with config_path.open() as f:
        data = yaml.safe_load(f)

    def parse_tool(key: str, tool_data: dict[str, Any], required: bool) -> FastTool:
        releases_data = tool_data.get("releases", {})
        return FastTool(
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

    return FastToolsConfig(
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

    config = load_fast_tools()
    tool = config.get_tool(tool_key)

    if not tool:
        return {"error": f"Unknown tool: {tool_key}"}

    # Build version check command
    # Use custom version_command if specified, otherwise default to --version
    version_arg = tool.version_command or "--version"
    check_cmd = f"~/bin/{tool.binary} {version_arg} 2>/dev/null || {tool.binary} {version_arg} 2>/dev/null"

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

            # Check which path
            which_cmd = f"which ~/bin/{tool.binary} 2>/dev/null || which {tool.binary} 2>/dev/null"
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
    """Check availability of all fast tools.

    Args:
        facility: Facility ID (None = local)

    Returns:
        Dict with tool statuses and summary
    """
    config = load_fast_tools()
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


def ensure_path(facility: str | None = None) -> str:
    """Ensure ~/bin is in PATH, adding to .bashrc if needed.

    Args:
        facility: Facility ID (None = local)

    Returns:
        Status message
    """
    # Check if ~/bin is in PATH
    check_cmd = 'echo $PATH | grep -q "$HOME/bin" && echo "yes" || echo "no"'
    in_path = run(check_cmd, facility=facility).strip() == "yes"

    if in_path:
        return "~/bin already in PATH"

    # Check if already in .bashrc
    check_bashrc = (
        'grep -q \'export PATH="$HOME/bin:$PATH"\' ~/.bashrc && echo "yes" || echo "no"'
    )
    in_bashrc = run(check_bashrc, facility=facility).strip() == "yes"

    if in_bashrc:
        return "~/bin configured in .bashrc (reload shell to activate)"

    # Add to .bashrc
    add_cmd = "echo 'export PATH=\"$HOME/bin:$PATH\"' >> ~/.bashrc"
    run(add_cmd, facility=facility)

    return "Added ~/bin to PATH in .bashrc (reload shell to activate)"


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
    config = load_fast_tools()
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

    # Get install command
    try:
        install_cmd = tool.get_install_command(arch)
    except Exception as e:
        return {"success": False, "error": f"Failed to generate install command: {e}"}

    # Execute installation
    try:
        output = run(install_cmd, facility=facility, timeout=120)
        logger.debug(f"Install output: {output}")
    except Exception as e:
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
) -> dict[str, Any]:
    """Install all fast tools.

    Args:
        facility: Facility ID (None = local)
        required_only: Only install required tools
        force: Reinstall even if already present

    Returns:
        Dict with installation results
    """
    config = load_fast_tools()
    results: dict[str, Any] = {
        "facility": facility or "local",
        "installed": [],
        "already_present": [],
        "failed": [],
    }

    tools_to_install = config.required if required_only else config.all_tools

    for key in tools_to_install:
        result = install_tool(key, facility=facility, force=force)

        if result.get("success"):
            if result.get("action") == "already_installed":
                results["already_present"].append(key)
            else:
                results["installed"].append(key)
        else:
            results["failed"].append({"tool": key, "error": result.get("error")})

    results["success"] = len(results["failed"]) == 0

    return results
