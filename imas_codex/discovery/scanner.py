"""
Graph-led directory scanner.

Scans directories at remote facilities via SSH. Uses single SSH calls
with batched commands to minimize latency impact from ProxyJump connections.

Key design: One SSH call scans 100+ paths. With 6s SSH latency, this means
~0.06s per path instead of 6s per path.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.frontier import (
    get_frontier,
    persist_scan_results,
    seed_facility_roots,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _get_ssh_host(facility: str) -> str:
    """Get SSH host for facility."""
    from imas_codex.discovery import get_facility

    try:
        config = get_facility(facility)
        return config.get("ssh_host", facility)
    except ValueError:
        return facility


def _run_ssh(facility: str, cmd: str, timeout: int = 300) -> tuple[str, int]:
    """Run command via SSH, return (stdout, returncode)."""
    ssh_host = _get_ssh_host(facility)
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", ssh_host, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        return "", -1
    except Exception as e:
        logger.warning(f"SSH error: {e}")
        return "", -1


@dataclass
class DirStats:
    """Statistics about a directory's contents."""

    total_files: int = 0
    total_dirs: int = 0
    has_readme: bool = False
    has_makefile: bool = False
    has_git: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_dirs": self.total_dirs,
            "has_readme": self.has_readme,
            "has_makefile": self.has_makefile,
            "has_git": self.has_git,
        }


@dataclass
class ScanResult:
    """Result of scanning a directory."""

    path: str
    stats: DirStats
    child_dirs: list[str]
    error: str | None = None


def scan_paths(
    facility: str,
    paths: list[str],
    timeout: int = 300,
) -> list[ScanResult]:
    """Scan multiple paths in ONE SSH call.

    Uses find (universal) instead of fd. One SSH call for all paths.
    """
    if not paths:
        return []

    # Build shell script
    script_lines = []
    for path in paths:
        escaped = path.replace("'", "'\\''")
        script_lines.append(f"""
echo "===PATH:{escaped}==="
if [ -d '{escaped}' ]; then
  echo "===DIRS==="
  find '{escaped}' -maxdepth 1 -type d 2>/dev/null | tail -n +2
  echo "===FILES==="
  find '{escaped}' -maxdepth 1 -type f 2>/dev/null | wc -l
  echo "===LS==="
  ls -1a '{escaped}' 2>/dev/null | head -30
else
  echo "===ERROR==="
  echo "not a directory"
fi""")

    script_lines.append('echo "===DONE==="')
    script = "\n".join(script_lines)

    stdout, rc = _run_ssh(facility, script, timeout=timeout)

    if not stdout:
        return [
            ScanResult(path=p, stats=DirStats(), child_dirs=[], error=f"SSH rc={rc}")
            for p in paths
        ]

    # Parse output
    results = []
    current_path = None
    current_section = None
    sections: dict[str, list[str]] = {}

    for line in stdout.split("\n"):
        if line.startswith("===PATH:") and line.endswith("==="):
            if current_path:
                results.append(_parse_sections(current_path, sections))
            current_path = line[8:-3]
            sections = {}
            current_section = None
        elif line.startswith("===") and line.endswith("==="):
            current_section = line.strip("=")
            sections[current_section] = []
        elif current_section and current_path:
            sections[current_section].append(line)

    if current_path:
        results.append(_parse_sections(current_path, sections))

    # Fill missing
    result_paths = {r.path for r in results}
    for path in paths:
        if path not in result_paths:
            results.append(
                ScanResult(path=path, stats=DirStats(), child_dirs=[], error="missing")
            )

    return results


def _parse_sections(path: str, sections: dict[str, list[str]]) -> ScanResult:
    """Parse sections for a single path."""
    stats = DirStats()
    child_dirs = []

    if "ERROR" in sections:
        error = sections["ERROR"][0] if sections["ERROR"] else "unknown"
        return ScanResult(path=path, stats=stats, child_dirs=[], error=error)

    if "DIRS" in sections:
        child_dirs = [d.strip() for d in sections["DIRS"] if d.strip()]
        stats.total_dirs = len(child_dirs)

    if "FILES" in sections and sections["FILES"]:
        try:
            stats.total_files = int(sections["FILES"][0].strip())
        except ValueError:
            pass

    if "LS" in sections:
        files = [f.lower() for f in sections["LS"] if f.strip()]
        stats.has_readme = any("readme" in f for f in files)
        stats.has_makefile = any("makefile" in f for f in files)
        stats.has_git = ".git" in files

    return ScanResult(path=path, stats=stats, child_dirs=child_dirs, error=None)


def scan_facility_sync(
    facility: str,
    limit: int = 100,
    dry_run: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    batch_size: int = 200,
) -> dict[str, int]:
    """Scan frontier paths. One SSH call per batch (default 200 paths)."""
    frontier = get_frontier(facility, limit=limit)

    if not frontier:
        logger.info(f"No frontier, seeding {facility}")
        seed_facility_roots(facility)
        frontier = get_frontier(facility, limit=limit)
        if not frontier:
            return {"scanned": 0, "children_created": 0, "errors": 0}

    paths = [p["path"] for p in frontier]
    total = len(paths)
    logger.info(f"Scanning {total} paths for {facility}")

    scanned = 0
    children_created = 0
    errors = 0

    for batch_idx, start in enumerate(range(0, total, batch_size)):
        batch = paths[start : start + batch_size]
        n_batches = (total + batch_size - 1) // batch_size

        if progress_callback:
            progress_callback(start + 1, total, f"batch {batch_idx + 1}/{n_batches}")

        results = scan_paths(facility, batch)

        if dry_run:
            # Just count, don't persist
            for r in results:
                if r.error:
                    errors += 1
                else:
                    scanned += 1
        else:
            # Batch persist all results in one transaction
            batch_data = [
                (r.path, r.stats.to_dict(), r.child_dirs, r.error) for r in results
            ]
            stats = persist_scan_results(facility, batch_data)
            scanned += stats["scanned"]
            children_created += stats["children_created"]
            errors += stats["errors"]

    return {"scanned": scanned, "children_created": children_created, "errors": errors}
