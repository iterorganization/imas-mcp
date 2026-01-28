"""
Graph-led directory scanner.

Scans directories at remote facilities using os.scandir and optional rg.
Uses the remote/scripts/scan_directories.py script for actual scanning.

Key design:
- Uses Python's os.scandir for fast directory enumeration
- Uses rg for pattern detection (IMAS, MDSplus, physics keywords)
- Single SSH call per batch minimizes latency (~1.8s network overhead)
- Outputs JSON for reliable parsing (handles control chars, unicode)
- All data collected for grounded LLM scoring decisions
- Facility-specific exclusions merged from *_private.yaml

The remote script (scan_directories.py) is pure Python 3.8+ stdlib.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from imas_codex.config.discovery_config import (
    get_discovery_config,
    get_exclusion_config_for_facility,
)
from imas_codex.discovery.facility import get_facility
from imas_codex.remote.executor import run_python_script

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ChildDirInfo:
    """Information about a child directory including symlink status."""

    path: str
    is_symlink: bool = False
    realpath: str | None = None  # Only set if is_symlink=True
    device_inode: str | None = None  # "device:inode" for deduplication


def _get_rg_patterns() -> dict[str, str]:
    """Get regex patterns for rg detection from config.

    Returns a dict of category -> pattern string for ripgrep.
    Categories are used as keys in the rg_matches output.
    All patterns are included since rg runs server-side and is fast.
    """
    config = get_discovery_config()

    # Build category patterns from data systems - use ALL patterns
    patterns = {}

    for name, ds in config.scoring.data_systems.items():
        if ds.patterns:
            patterns[name] = "|".join(p.pattern for p in ds.patterns)

    # Add ALL physics domains
    for name, pd in config.scoring.physics_domains.items():
        if pd.patterns:
            patterns[name] = "|".join(p.pattern for p in pd.patterns)

    return patterns


@dataclass
class DirStats:
    """Statistics about a directory's contents for grounded scoring."""

    total_files: int = 0
    total_dirs: int = 0
    has_readme: bool = False
    has_makefile: bool = False
    has_git: bool = False
    git_remote_url: str | None = None  # Git remote origin URL
    git_head_commit: str | None = None  # Git HEAD commit hash
    git_branch: str | None = None  # Current branch name
    git_root_commit: str | None = None  # First commit in history (for fork detection)
    child_names: list[str] | None = None  # First 30 child file/dir names
    file_type_counts: dict[str, int] = field(default_factory=dict)
    patterns_detected: list[str] = field(default_factory=list)
    # Fast tool data for grounded scoring
    size_bytes: int | None = None  # Directory size from du
    rg_matches: dict[str, int] = field(default_factory=dict)  # pattern -> match count

    def to_dict(self) -> dict[str, Any]:
        result = {
            "total_files": self.total_files,
            "total_dirs": self.total_dirs,
            "has_readme": self.has_readme,
            "has_makefile": self.has_makefile,
            "has_git": self.has_git,
        }
        if self.git_remote_url:
            result["git_remote_url"] = self.git_remote_url
        if self.git_head_commit:
            result["git_head_commit"] = self.git_head_commit
        if self.git_branch:
            result["git_branch"] = self.git_branch
        if self.git_root_commit:
            result["git_root_commit"] = self.git_root_commit
        if self.child_names:
            result["child_names"] = self.child_names
        if self.file_type_counts:
            result["file_type_counts"] = json.dumps(self.file_type_counts)
        if self.patterns_detected:
            result["patterns_detected"] = self.patterns_detected
        if self.size_bytes is not None:
            result["size_bytes"] = self.size_bytes
        if self.rg_matches:
            result["rg_matches"] = json.dumps(self.rg_matches)
        return result


@dataclass
class ScanResult:
    """Result of scanning a directory."""

    path: str
    stats: DirStats
    child_dirs: list[ChildDirInfo]
    excluded_dirs: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None
    # Symlink info for the scanned path itself
    is_symlink: bool = False
    realpath: str | None = None  # Only set if is_symlink=True
    device_inode: str | None = None  # "device:inode" for deduplication


def scan_paths(
    facility: str,
    paths: list[str],
    timeout: int = 300,
    enable_rg: bool = True,
    enable_size: bool = False,
) -> list[ScanResult]:
    """Scan multiple paths using the remote scan_directories.py script.

    Uses run_python_script() for transparent local/SSH execution.
    Applies exclusion patterns from DiscoveryConfig merged with facility-specific
    exclusions from *_private.yaml.

    Args:
        facility: Facility identifier for SSH/local execution
        paths: List of directory paths to scan
        timeout: Command timeout in seconds
        enable_rg: If True, run rg pattern detection (slower).
                   If False, skip rg for faster enumeration-only scans.
        enable_size: If True, calculate directory size (can be very slow).
                     If False, skip size calculation for speed.
    """
    if not paths:
        return []

    # Resolve ssh_host from facility config
    try:
        config = get_facility(facility)
        ssh_host = config.get("ssh_host", facility)
    except ValueError:
        ssh_host = facility

    # Use facility-specific exclusion config (merges base + facility excludes)
    exclusion_config = get_exclusion_config_for_facility(facility)

    # Build input data for the remote script
    rg_patterns = _get_rg_patterns() if enable_rg else {}
    input_data = {
        "paths": paths,
        "rg_patterns": rg_patterns,
        "enable_rg": enable_rg,
        "enable_size": enable_size,
    }

    try:
        output = run_python_script(
            "scan_directories.py",
            input_data=input_data,
            ssh_host=ssh_host,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # Timeout is transient - re-raise so paths stay in 'listing' for retry
        logger.warning(
            f"SSH scan timed out after {timeout}s for {facility}. "
            f"Paths will be retried via orphan recovery."
        )
        raise
    except subprocess.CalledProcessError as e:
        # SSH connection failures (exit 255) are transient - re-raise for retry
        if e.returncode == 255:
            logger.warning(
                f"Scan failed for {facility}: SSH connection failed. "
                f"Verify connectivity with 'ssh {facility}'. "
                f"Check: VPN connected, SSH key loaded (ssh-add), host in ~/.ssh/config. "
                f"Paths will be retried via orphan recovery."
            )
            raise
        # Other CalledProcessError: mark paths as errors (actual script failures)
        error_str = str(e)[:200]
        logger.warning(f"Scan script failed for {facility}: {error_str}")
        return [
            ScanResult(path=p, stats=DirStats(), child_dirs=[], error=error_str)
            for p in paths
        ]
    except Exception as e:
        # Unknown errors - log and return as path errors
        error_str = str(e)
        if len(error_str) > 200:
            error_str = error_str[:200] + "..."
        logger.warning(f"Scan failed for {facility}: {error_str}")
        return [
            ScanResult(path=p, stats=DirStats(), child_dirs=[], error=error_str)
            for p in paths
        ]

    # Parse JSON output
    try:
        # Handle stderr output mixed in
        if "[stderr]:" in output:
            output = output.split("[stderr]:")[0].strip()

        results_data = json.loads(output)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse scan output: {e}")
        logger.debug(f"Raw output: {output[:500]}")
        return [
            ScanResult(path=p, stats=DirStats(), child_dirs=[], error="parse error")
            for p in paths
        ]

    # Convert to ScanResult objects with exclusion filtering
    results = []
    for data in results_data:
        path = data["path"]
        error = data.get("error")

        if error:
            results.append(
                ScanResult(path=path, stats=DirStats(), child_dirs=[], error=error)
            )
            continue

        stats_data = data.get("stats", {})

        # Extract rg_matches if available
        rg_matches = stats_data.get("rg_matches", {})

        # Extract size_bytes
        size_bytes = stats_data.get("size_bytes")
        if isinstance(size_bytes, str):
            try:
                size_bytes = int(size_bytes)
            except ValueError:
                size_bytes = None

        stats = DirStats(
            total_files=stats_data.get("total_files", 0),
            total_dirs=stats_data.get("total_dirs", 0),
            has_readme=stats_data.get("has_readme", False),
            has_makefile=stats_data.get("has_makefile", False),
            has_git=stats_data.get("has_git", False),
            git_remote_url=stats_data.get("git_remote_url"),
            git_head_commit=stats_data.get("git_head_commit"),
            git_branch=stats_data.get("git_branch"),
            git_root_commit=stats_data.get("git_root_commit"),
            child_names=data.get("child_names", []),
            file_type_counts=stats_data.get("file_type_counts", {}),
            size_bytes=size_bytes,
            rg_matches=rg_matches,
        )

        # Parse child directories - now includes symlink info
        raw_child_dirs = data.get("child_dirs", [])

        # Handle both old format (list of strings) and new format (list of dicts)
        child_dir_infos: list[ChildDirInfo] = []
        for item in raw_child_dirs:
            if isinstance(item, str):
                # Legacy format: just path string
                child_dir_infos.append(ChildDirInfo(path=item))
            elif isinstance(item, dict):
                # New format: {path, is_symlink, realpath, device_inode}
                child_dir_infos.append(
                    ChildDirInfo(
                        path=item.get("path", ""),
                        is_symlink=item.get("is_symlink", False),
                        realpath=item.get("realpath"),
                        device_inode=item.get("device_inode"),
                    )
                )

        # Filter child directories using facility-specific exclusion config
        # Extract just paths for exclusion check
        all_paths = [c.path for c in child_dir_infos]
        _, excluded_dirs = exclusion_config.filter_paths(all_paths)
        excluded_path_set = {p for p, _ in excluded_dirs}

        # Filter to included dirs, preserving symlink info
        included_dirs = [c for c in child_dir_infos if c.path not in excluded_path_set]

        # Log exclusions at debug level
        if excluded_dirs:
            logger.debug(
                f"Excluded {len(excluded_dirs)} dirs in {path}: "
                f"{[d for d, _ in excluded_dirs[:5]]}"
            )

        # Extract symlink info for the scanned path itself
        path_is_symlink = data.get("is_symlink", False)
        path_realpath = data.get("realpath")
        path_device_inode = data.get("device_inode")

        results.append(
            ScanResult(
                path=path,
                stats=stats,
                child_dirs=included_dirs,
                excluded_dirs=excluded_dirs,
                error=None,
                is_symlink=path_is_symlink,
                realpath=path_realpath,
                device_inode=path_device_inode,
            )
        )

    # Fill missing paths
    result_paths = {r.path for r in results}
    for path in paths:
        if path not in result_paths:
            results.append(
                ScanResult(path=path, stats=DirStats(), child_dirs=[], error="missing")
            )

    return results
