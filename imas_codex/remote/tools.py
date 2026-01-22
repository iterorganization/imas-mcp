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
    """Check if a specific tool is available.

    Args:
        tool_key: Tool key (e.g., 'rg', 'fd')
        facility: Facility ID (None = local)

    Returns:
        Dict with available, version, path
    """
    config = load_fast_tools()
    tool = config.get_tool(tool_key)

    if not tool:
        return {"error": f"Unknown tool: {tool_key}"}

    # Check ~/bin first, then PATH
    check_cmd = f"~/bin/{tool.binary} --version 2>/dev/null || {tool.binary} --version 2>/dev/null"

    try:
        output = run(check_cmd, facility=facility, timeout=10)
        if output and "[stderr]" not in output.lower():
            # Extract version number
            import re

            version_match = re.search(r"(\d+\.\d+\.?\d*)", output)
            version = version_match.group(1) if version_match else output.split()[0]

            # Check which path
            which_cmd = f"which ~/bin/{tool.binary} 2>/dev/null || which {tool.binary} 2>/dev/null"
            path_output = run(which_cmd, facility=facility, timeout=5)
            path = path_output.split("\n")[0] if path_output else None

            return {
                "available": True,
                "version": version,
                "path": path,
                "required": tool.required,
                "purpose": tool.purpose,
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

    Args:
        tool_key: Tool key (e.g., 'rg', 'fd')
        facility: Facility ID (None = local)
        force: Reinstall even if already present

    Returns:
        Dict with success status and details
    """
    config = load_fast_tools()
    tool = config.get_tool(tool_key)

    if not tool:
        return {"success": False, "error": f"Unknown tool: {tool_key}"}

    # Check if already installed
    if not force:
        status = check_tool(tool_key, facility=facility)
        if status.get("available"):
            return {
                "success": True,
                "action": "already_installed",
                "version": status.get("version"),
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
