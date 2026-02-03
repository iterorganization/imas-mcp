"""
Remote Python environment management.

This module provides Python version and venv management on local and remote facilities
using uv (astral-sh/uv). It enables modern Python development across facilities with
different system Python versions.

Key capabilities:
- Detect Python versions (system and uv-managed)
- Install modern Python via uv (no root required)
- Create project venvs for imas-codex
- Handle airgapped facilities (pre-cached Python binaries)

Strategy per facility type:
1. PyPI accessible (TCV, ITER, JET): Install uv, then uv python install
2. Airgapped (JT60SA): uv already installed, Python pre-cached from GitHub releases

Usage:
    from imas_codex.remote.python import get_python_status, install_python, create_venv

    # Check what's available
    status = get_python_status(facility='tcv')

    # Install modern Python
    result = install_python(facility='tcv', version='3.12')

    # Create project venv
    result = create_venv(facility='tcv')
"""

import logging
import re
from dataclasses import dataclass

from imas_codex.remote.tools import check_tool, install_tool, run

logger = logging.getLogger(__name__)

# Recommended Python version for imas-codex
RECOMMENDED_PYTHON = "3.12"

# Minimum Python version for imas-codex features
MIN_PYTHON = "3.10"

# Default venv path on remotes
DEFAULT_VENV_PATH = "~/.local/share/imas-codex/venv"


@dataclass
class PythonVersion:
    """Parsed Python version info."""

    major: int
    minor: int
    patch: int
    source: str  # 'system', 'uv', or path

    @property
    def version_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def short_version(self) -> str:
        return f"{self.major}.{self.minor}"

    def meets_minimum(self, min_version: str = MIN_PYTHON) -> bool:
        """Check if this version meets the minimum requirement."""
        min_parts = [int(p) for p in min_version.split(".")]
        current = [self.major, self.minor, self.patch]
        return current[:2] >= min_parts[:2]


@dataclass
class PythonStatus:
    """Python environment status for a facility."""

    facility: str
    uv_available: bool
    uv_version: str | None
    system_python: PythonVersion | None
    uv_pythons: list[PythonVersion]
    recommended_action: str
    has_modern_python: bool
    venv_path: str | None
    venv_python: PythonVersion | None


def _parse_python_version(
    version_output: str, source: str = "system"
) -> PythonVersion | None:
    """Parse Python version from --version output.

    Args:
        version_output: Output from python3 --version
        source: Source identifier

    Returns:
        PythonVersion or None if parsing fails
    """
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_output)
    if match:
        return PythonVersion(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            source=source,
        )
    return None


def _parse_uv_python_list(output: str) -> list[PythonVersion]:
    """Parse uv python list output to find installed versions.

    uv python list shows:
    - cpython-3.12.12-linux-x86_64-gnu    /home/user/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3
    - cpython-3.13.9-linux-x86_64-gnu     .local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/bin/python3  (relative)

    Returns:
        List of installed Python versions
    """
    versions = []
    seen = set()  # Dedupe by version
    for line in output.splitlines():
        # Look for cpython-X.Y.Z patterns with a path (absolute or relative)
        # Path can start with / or . (for relative paths like .local/)
        match = re.search(r"cpython-(\d+)\.(\d+)\.(\d+)-\S+\s+([/.]\S+)", line)
        if match:
            version_key = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
            if version_key not in seen:
                seen.add(version_key)
                versions.append(
                    PythonVersion(
                        major=version_key[0],
                        minor=version_key[1],
                        patch=version_key[2],
                        source=match.group(4),
                    )
                )
    return versions


def get_python_status(facility: str | None = None) -> PythonStatus:
    """Get comprehensive Python environment status for a facility.

    Checks:
    - System Python version
    - uv availability and version
    - uv-managed Python installations
    - Existing imas-codex venv
    - Recommended action

    Args:
        facility: Facility ID (None = local)

    Returns:
        PythonStatus with complete environment info
    """
    facility_name = facility or "local"

    # Check uv availability
    uv_status = check_tool("uv", facility=facility)
    uv_available = uv_status.get("available", False)
    uv_version = uv_status.get("version")

    # Get system Python version
    system_python = None
    try:
        output = run(
            "python3 --version 2>/dev/null || python --version 2>/dev/null",
            facility=facility,
            timeout=10,
        )
        if output and "Python" in output:
            system_python = _parse_python_version(output, source="system")
    except Exception as e:
        logger.debug(f"Failed to get system Python: {e}")

    # Get uv-managed Pythons
    uv_pythons: list[PythonVersion] = []
    if uv_available:
        try:
            output = run(
                "uv python list --only-installed 2>/dev/null || true",
                facility=facility,
                timeout=15,
            )
            uv_pythons = _parse_uv_python_list(output)
        except Exception as e:
            logger.debug(f"Failed to list uv Pythons: {e}")

    # Check for existing venv
    venv_path = None
    venv_python = None
    try:
        venv_check = run(
            f"test -f {DEFAULT_VENV_PATH}/bin/python && echo exists || echo missing",
            facility=facility,
            timeout=5,
        )
        if "exists" in venv_check:
            venv_path = DEFAULT_VENV_PATH
            # Get venv Python version
            venv_version_output = run(
                f"{DEFAULT_VENV_PATH}/bin/python --version 2>/dev/null || true",
                facility=facility,
                timeout=5,
            )
            if venv_version_output:
                venv_python = _parse_python_version(
                    venv_version_output, source=DEFAULT_VENV_PATH
                )
    except Exception as e:
        logger.debug(f"Failed to check venv: {e}")

    # Determine if we have modern Python
    has_modern = False
    if system_python and system_python.meets_minimum():
        has_modern = True
    if any(p.meets_minimum() for p in uv_pythons):
        has_modern = True

    # Determine recommended action
    if venv_python and venv_python.meets_minimum():
        recommended_action = "ready"
    elif has_modern and not venv_path:
        recommended_action = "create_venv"
    elif not uv_available:
        recommended_action = "install_uv"
    elif not has_modern:
        recommended_action = "install_python"
    else:
        recommended_action = "create_venv"

    return PythonStatus(
        facility=facility_name,
        uv_available=uv_available,
        uv_version=uv_version,
        system_python=system_python,
        uv_pythons=uv_pythons,
        recommended_action=recommended_action,
        has_modern_python=has_modern,
        venv_path=venv_path,
        venv_python=venv_python,
    )


def install_uv(facility: str | None = None, force: bool = False) -> dict:
    """Install uv on a facility.

    Args:
        facility: Facility ID (None = local)
        force: Reinstall even if already present

    Returns:
        Dict with success status and details
    """
    return install_tool("uv", facility=facility, force=force)


def install_python(
    facility: str | None = None,
    version: str = RECOMMENDED_PYTHON,
) -> dict:
    """Install Python via uv on a facility.

    uv downloads Python from python-build-standalone (GitHub releases),
    not PyPI, so this works even on PyPI-airgapped facilities as long
    as GitHub is accessible.

    Args:
        facility: Facility ID (None = local)
        version: Python version to install (e.g., "3.12")

    Returns:
        Dict with success status and details
    """
    # Ensure uv is available
    uv_status = check_tool("uv", facility=facility)
    if not uv_status.get("available"):
        return {
            "success": False,
            "error": "uv not installed. Run 'imas-codex tools install <facility> --tool uv' first.",
        }

    # Check if Python is already cached
    try:
        existing = run(
            f"uv python list --only-installed 2>/dev/null | grep 'cpython-{version}' || true",
            facility=facility,
            timeout=15,
        )
        if f"cpython-{version}" in existing:
            # Extract exact version
            match = re.search(rf"cpython-({version}\.\d+)", existing)
            exact_version = match.group(1) if match else version
            return {
                "success": True,
                "action": "already_installed",
                "version": exact_version,
            }
    except Exception:
        pass

    # Install Python via uv
    try:
        output = run(
            f"uv python install {version} 2>&1", facility=facility, timeout=300
        )
        logger.info(f"uv python install output: {output}")

        # Verify installation
        verify = run(
            f"uv python list --only-installed 2>/dev/null | grep 'cpython-{version}' || true",
            facility=facility,
            timeout=15,
        )
        if f"cpython-{version}" in verify:
            match = re.search(rf"cpython-({version}\.\d+)", verify)
            exact_version = match.group(1) if match else version
            return {
                "success": True,
                "action": "installed",
                "version": exact_version,
                "output": output,
            }
        else:
            return {
                "success": False,
                "error": "Installation completed but Python not found in uv list",
                "output": output,
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def create_venv(
    facility: str | None = None,
    python_version: str = RECOMMENDED_PYTHON,
    venv_path: str = DEFAULT_VENV_PATH,
    force: bool = False,
) -> dict:
    """Create a venv for imas-codex on a facility.

    Uses uv venv for fast creation with the best available Python.

    Args:
        facility: Facility ID (None = local)
        python_version: Preferred Python version
        venv_path: Path for the venv
        force: Recreate even if exists

    Returns:
        Dict with success status and details
    """
    # Check if venv already exists
    if not force:
        try:
            check = run(
                f"test -f {venv_path}/bin/python && echo exists || echo missing",
                facility=facility,
                timeout=5,
            )
            if "exists" in check:
                # Get version
                version_out = run(
                    f"{venv_path}/bin/python --version 2>/dev/null || true",
                    facility=facility,
                    timeout=5,
                )
                version = _parse_python_version(version_out, source=venv_path)
                return {
                    "success": True,
                    "action": "already_exists",
                    "venv_path": venv_path,
                    "python_version": version.version_string if version else "unknown",
                }
        except Exception:
            pass

    # Ensure uv is available
    uv_status = check_tool("uv", facility=facility)
    if not uv_status.get("available"):
        return {
            "success": False,
            "error": "uv not installed. Run 'imas-codex tools install <facility> --tool uv' first.",
        }

    # Check available Pythons
    status = get_python_status(facility=facility)

    # Determine which Python to use
    python_arg = ""
    used_version = None

    # Prefer uv-managed Python at requested version
    matching_uv = [p for p in status.uv_pythons if p.short_version == python_version]
    if matching_uv:
        used_version = matching_uv[0]
        python_arg = f"--python {python_version}"
    # Fall back to any modern uv Python
    elif status.uv_pythons:
        modern_uv = [p for p in status.uv_pythons if p.meets_minimum()]
        if modern_uv:
            used_version = modern_uv[0]
            python_arg = f"--python {used_version.short_version}"
    # Fall back to system Python if modern
    elif status.system_python and status.system_python.meets_minimum():
        used_version = status.system_python
        python_arg = "--python python3"

    if not used_version:
        return {
            "success": False,
            "error": f"No Python >= {MIN_PYTHON} available. Run 'imas-codex tools python install <facility>' first.",
        }

    # Create parent directory
    parent_dir = "/".join(venv_path.split("/")[:-1])
    run(f"mkdir -p {parent_dir}", facility=facility)

    # Create venv with uv
    try:
        cmd = f"uv venv {python_arg} {venv_path} 2>&1"
        output = run(cmd, facility=facility, timeout=60)
        logger.info(f"uv venv output: {output}")

        # Verify
        check = run(
            f"test -f {venv_path}/bin/python && echo exists || echo missing",
            facility=facility,
            timeout=5,
        )
        if "exists" in check:
            version_out = run(
                f"{venv_path}/bin/python --version 2>/dev/null || true",
                facility=facility,
                timeout=5,
            )
            version = _parse_python_version(version_out, source=venv_path)
            return {
                "success": True,
                "action": "created",
                "venv_path": venv_path,
                "python_version": version.version_string
                if version
                else used_version.version_string,
                "output": output,
            }
        else:
            return {
                "success": False,
                "error": "venv creation completed but venv not found",
                "output": output,
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def setup_python_env(
    facility: str | None = None,
    python_version: str = RECOMMENDED_PYTHON,
    force: bool = False,
) -> dict:
    """Complete Python environment setup for a facility.

    This is the main entry point for setting up a modern Python environment.
    It performs all necessary steps:
    1. Install uv (if missing)
    2. Install Python via uv (if no modern Python available)
    3. Create imas-codex venv

    Args:
        facility: Facility ID (None = local)
        python_version: Preferred Python version
        force: Force reinstall/recreate

    Returns:
        Dict with complete setup status
    """
    results = {
        "facility": facility or "local",
        "steps": [],
    }

    # Step 1: Check/install uv
    uv_status = check_tool("uv", facility=facility)
    if not uv_status.get("available") or force:
        uv_result = install_uv(facility=facility, force=force)
        results["steps"].append(
            {
                "step": "install_uv",
                "result": uv_result,
            }
        )
        if not uv_result.get("success"):
            results["success"] = False
            results["error"] = "Failed to install uv"
            return results
    else:
        results["steps"].append(
            {
                "step": "install_uv",
                "result": {
                    "success": True,
                    "action": "already_installed",
                    "version": uv_status.get("version"),
                },
            }
        )

    # Step 2: Check/install Python
    status = get_python_status(facility=facility)
    needs_python = not status.has_modern_python

    if needs_python or force:
        python_result = install_python(facility=facility, version=python_version)
        results["steps"].append(
            {
                "step": "install_python",
                "result": python_result,
            }
        )
        if not python_result.get("success"):
            results["success"] = False
            results["error"] = "Failed to install Python"
            return results
    else:
        # Report what Python is available
        best_python = None
        if status.system_python and status.system_python.meets_minimum():
            best_python = status.system_python
        for p in status.uv_pythons:
            if p.meets_minimum():
                if best_python is None or p.minor > best_python.minor:
                    best_python = p
        results["steps"].append(
            {
                "step": "install_python",
                "result": {
                    "success": True,
                    "action": "already_available",
                    "version": best_python.version_string if best_python else "unknown",
                    "source": best_python.source if best_python else "unknown",
                },
            }
        )

    # Step 3: Create venv
    venv_result = create_venv(
        facility=facility, python_version=python_version, force=force
    )
    results["steps"].append(
        {
            "step": "create_venv",
            "result": venv_result,
        }
    )

    results["success"] = venv_result.get("success", False)
    if venv_result.get("success"):
        results["venv_path"] = venv_result.get("venv_path")
        results["python_version"] = venv_result.get("python_version")
    else:
        results["error"] = venv_result.get("error", "Failed to create venv")

    return results
