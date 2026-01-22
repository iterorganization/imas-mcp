"""
Graph-led directory scanner.

Scans directories at remote facilities, collecting DirStats and creating
child FacilityPath nodes. The scanner is graph-led: it queries the graph
for paths to scan rather than accepting CLI parameters.

Key features:
    - Uses fast tools (fd, rg) for efficient scanning
    - Collects file type counts, sizes, and patterns
    - Creates child paths for discovered directories
    - Handles large directories gracefully (skips size calc if >10k files)
    - Idempotent: re-running with no score changes does nothing
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.executor import ParallelExecutor
from imas_codex.discovery.frontier import (
    create_child_paths,
    get_frontier,
    mark_path_scanned,
    mark_path_skipped,
    seed_facility_roots,
)
from imas_codex.remote.tools import run

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# IMAS-relevant patterns to search for with rg
IMAS_PATTERNS = [
    r"imas\.",          # IMAS Python imports
    r"imas_",           # IMAS module prefixes
    r"ids\.",           # IDS access patterns
    r"write_ids",       # IDS writing
    r"read_ids",        # IDS reading
    r"put_slice",       # IMAS put operations
    r"get_slice",       # IMAS get operations
    r"equilibrium",     # Common IDS names
    r"core_profiles",   # Common IDS names
    r"MDSconnect",      # MDSplus connection
    r"tdi\(",           # TDI expressions
]


@dataclass
class DirStats:
    """Statistics about a directory's contents."""

    total_files: int = 0
    total_dirs: int = 0
    total_size_bytes: int | None = None
    size_skipped: bool = False
    file_type_counts: dict[str, int] = field(default_factory=dict)
    has_readme: bool = False
    has_makefile: bool = False
    has_git: bool = False
    patterns_detected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for graph storage."""
        return {
            "total_files": self.total_files,
            "total_dirs": self.total_dirs,
            "total_size_bytes": self.total_size_bytes,
            "size_skipped": self.size_skipped,
            "file_type_counts": json.dumps(self.file_type_counts),
            "has_readme": self.has_readme,
            "has_makefile": self.has_makefile,
            "has_git": self.has_git,
            "patterns_detected": self.patterns_detected,
        }


@dataclass
class ScanResult:
    """Result of scanning a directory."""

    path: str
    stats: DirStats
    child_dirs: list[str]
    error: str | None = None


class DirectoryScanner:
    """Scans directories at remote facilities.

    The scanner is graph-led: it queries the graph for paths to scan,
    collects statistics via SSH/local commands, and persists results
    back to the graph.

    Usage:
        scanner = DirectoryScanner(facility="iter")
        stats = await scanner.scan_all(limit=100)
    """

    def __init__(
        self,
        facility: str,
        max_sessions: int = 4,
        timeout: int = 30,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        self.facility = facility
        self.max_sessions = max_sessions
        self.timeout = timeout
        self.progress_callback = progress_callback
        self._executor = ParallelExecutor(
            facility=facility,
            max_sessions=max_sessions,
            timeout=timeout,
        )

    async def scan_path(self, path: str) -> ScanResult:
        """Scan a single directory, collecting stats.

        Args:
            path: Absolute directory path

        Returns:
            ScanResult with stats and child directories
        """
        stats = DirStats()
        child_dirs = []
        error = None

        try:
            # Step 1: Count files with fd (bounded, fast)
            file_count_cmd = f"fd -t f . '{path}' 2>/dev/null | head -10001 | wc -l"
            result = await self._executor.run_one(file_count_cmd, path)
            
            if result.returncode == 0:
                count = int(result.stdout.strip() or "0")
                stats.total_files = min(count, 10000)  # Cap at 10k
                stats.size_skipped = count > 10000

            # Step 2: List immediate subdirectories
            dir_list_cmd = f"fd -t d --max-depth 1 . '{path}' 2>/dev/null"
            result = await self._executor.run_one(dir_list_cmd, path)
            
            if result.returncode == 0 and result.stdout.strip():
                dirs = [d.strip() for d in result.stdout.strip().split("\n") if d.strip()]
                # Filter out the path itself
                child_dirs = [d for d in dirs if d != path and d.startswith(path)]
                stats.total_dirs = len(child_dirs)

            # Step 3: Get file extension counts (sample first 1000 files)
            ext_cmd = f"fd -t f . '{path}' 2>/dev/null | head -1000 | sed 's/.*\\.//' | sort | uniq -c | sort -rn | head -20"
            result = await self._executor.run_one(ext_cmd, path)
            
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    match = re.match(r"\s*(\d+)\s+(\S+)", line)
                    if match:
                        count, ext = match.groups()
                        stats.file_type_counts[ext] = int(count)

            # Step 4: Check for key files
            key_files_cmd = f"ls -1 '{path}' 2>/dev/null | grep -iE '^(readme|makefile|\.git)$' | head -5"
            result = await self._executor.run_one(key_files_cmd, path)
            
            if result.returncode == 0 and result.stdout.strip():
                files = result.stdout.strip().lower().split("\n")
                stats.has_readme = any("readme" in f for f in files)
                stats.has_makefile = any("makefile" in f for f in files)
                stats.has_git = ".git" in files

            # Step 5: Quick pattern search (limited to first 100 matches)
            if stats.total_files > 0 and stats.total_files <= 10000:
                patterns_found = []
                for pattern in IMAS_PATTERNS[:5]:  # Check first 5 patterns only
                    rg_cmd = f"rg -l --max-count=1 '{pattern}' '{path}' 2>/dev/null | head -1"
                    result = await self._executor.run_one(rg_cmd, path)
                    if result.returncode == 0 and result.stdout.strip():
                        patterns_found.append(pattern)
                stats.patterns_detected = patterns_found

            # Step 6: Get size if not too many files
            if not stats.size_skipped:
                size_cmd = f"du -sb '{path}' 2>/dev/null | cut -f1"
                result = await self._executor.run_one(size_cmd, path)
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        stats.total_size_bytes = int(result.stdout.strip())
                    except ValueError:
                        pass

        except Exception as e:
            error = str(e)
            logger.warning(f"Error scanning {path}: {e}")

        return ScanResult(
            path=path,
            stats=stats,
            child_dirs=child_dirs,
            error=error,
        )

    async def scan_all(
        self,
        limit: int = 100,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Scan all paths in the frontier.

        This is the main entry point for graph-led scanning:
        1. Query graph for pending paths
        2. If none, seed with facility roots
        3. Scan each path, collect stats
        4. Create child paths
        5. Mark as scanned

        Args:
            limit: Maximum paths to scan this run
            dry_run: If True, don't persist to graph

        Returns:
            Dict with stats: scanned, children_created, errors
        """
        # Get frontier from graph
        frontier = get_frontier(self.facility, limit=limit)

        # If no frontier, seed with roots
        if not frontier:
            logger.info(f"No frontier found, seeding {self.facility}")
            seed_facility_roots(self.facility)
            frontier = get_frontier(self.facility, limit=limit)

            if not frontier:
                logger.warning(f"No paths to scan for {self.facility}")
                return {"scanned": 0, "children_created": 0, "errors": 0}

        logger.info(f"Scanning {len(frontier)} paths for {self.facility}")

        scanned = 0
        children_created = 0
        errors = 0

        for i, path_info in enumerate(frontier):
            path = path_info["path"]

            if self.progress_callback:
                self.progress_callback(i + 1, len(frontier), path)

            # Scan the path
            result = await self.scan_path(path)

            if result.error:
                errors += 1
                if not dry_run:
                    mark_path_skipped(self.facility, path, result.error)
                continue

            scanned += 1

            if not dry_run:
                # Persist stats
                mark_path_scanned(self.facility, path, result.stats.to_dict())

                # Create child paths
                if result.child_dirs:
                    created = create_child_paths(self.facility, path, result.child_dirs)
                    children_created += created

        return {
            "scanned": scanned,
            "children_created": children_created,
            "errors": errors,
        }


async def scan_facility(
    facility: str,
    limit: int = 100,
    max_sessions: int = 4,
    dry_run: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    """Convenience function to scan a facility.

    Args:
        facility: Facility ID
        limit: Maximum paths to scan
        max_sessions: Concurrent SSH sessions
        dry_run: If True, don't persist to graph
        progress_callback: Optional (current, total, path) callback

    Returns:
        Dict with stats: scanned, children_created, errors
    """
    scanner = DirectoryScanner(
        facility=facility,
        max_sessions=max_sessions,
        progress_callback=progress_callback,
    )
    return await scanner.scan_all(limit=limit, dry_run=dry_run)


def scan_facility_sync(
    facility: str,
    limit: int = 100,
    max_sessions: int = 4,
    dry_run: bool = False,
) -> dict[str, int]:
    """Synchronous wrapper for scan_facility."""
    return asyncio.run(
        scan_facility(
            facility=facility,
            limit=limit,
            max_sessions=max_sessions,
            dry_run=dry_run,
        )
    )
