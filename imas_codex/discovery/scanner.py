"""
Graph-led directory scanner.

Scans directories at remote facilities using fast tools (fd, rg, dust).
Uses the DiscoveryConfig for exclusion patterns and the run() function
from remote/tools.py for transparent local/SSH execution.

Key design:
- Uses fd for directory enumeration (5x faster than find)
- Uses rg for pattern detection (IMAS, MDSplus, physics keywords)
- Uses dust for directory size estimation
- Single SSH call per batch minimizes latency (~8s network overhead)
- Outputs JSON for reliable parsing
- All data collected for grounded LLM scoring decisions
"""

from __future__ import annotations

import json
import logging
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from imas_codex.config.discovery_config import get_discovery_config
from imas_codex.discovery.frontier import (
    get_frontier,
    persist_scan_results,
    seed_facility_roots,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


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
    child_names: list[str] | None = None  # First 30 child file/dir names
    file_type_counts: dict[str, int] = field(default_factory=dict)
    patterns_detected: list[str] = field(default_factory=list)
    # New: fast tool data for grounded scoring
    size_bytes: int | None = None  # Directory size from dust
    rg_matches: dict[str, int] = field(default_factory=dict)  # pattern -> match count

    def to_dict(self) -> dict[str, Any]:
        result = {
            "total_files": self.total_files,
            "total_dirs": self.total_dirs,
            "has_readme": self.has_readme,
            "has_makefile": self.has_makefile,
            "has_git": self.has_git,
        }
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
    child_dirs: list[str]
    excluded_dirs: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None


def _build_scan_script(
    paths: list[str], enable_rg: bool = True, enable_size: bool = True
) -> str:
    """Build a bash script that uses fast tools for comprehensive scanning.

    Uses fd, rg, dust with fallbacks. Outputs JSON for parsing.
    All data collection happens in a SINGLE SSH call for efficiency.
    Patterns are loaded from config/patterns/scoring/*.yaml.

    Args:
        paths: List of directory paths to scan
        enable_rg: If True, run rg pattern detection (slower but more data).
                   If False, skip rg for faster enumeration-only scans.
        enable_size: If True, calculate directory size with dust/du.
                     If False, skip size calculation (much faster).
    """
    # Escape paths for shell
    escaped_paths = " ".join(shlex.quote(p) for p in paths)

    # Build rg commands only if enabled
    rg_block = ""
    rg_json = ""
    if enable_rg:
        # Get patterns from config
        rg_patterns = _get_rg_patterns()

        # Combine ALL patterns into a single rg call for speed
        # This is 14x faster than running rg separately per category
        all_patterns = "|".join(rg_patterns.values())
        escaped_all = shlex.quote(all_patterns)

        # Single rg call gets total match count, plus per-category breakdown
        # We use --max-depth 1 for speed (depth 2 is 2-10x slower on large dirs)
        rg_commands = [
            f"        local total_matches=$(rg -c --max-depth 1 {escaped_all} \"$p\" 2>/dev/null | awk -F: '{{s+=$2}} END {{print s+0}}')"
        ]
        rg_json_parts = ['\\"total\\":$total_matches']

        # Only get per-category breakdown if there are matches (optimization)
        for cat, pattern in rg_patterns.items():
            var_name = f"{cat}_cnt"
            escaped_pattern = shlex.quote(pattern)
            # Use -l to just count files, faster than -c
            rg_commands.append(
                f'        local {var_name}=$(rg -l --max-depth 1 {escaped_pattern} "$p" 2>/dev/null | wc -l)'
            )
            rg_json_parts.append(f'\\"{cat}\\":${var_name}')

        rg_block = "\n".join(rg_commands)
        rg_json = ",".join(rg_json_parts)

    # Size calculation block - skip in fast mode
    if enable_size:
        size_block = """
    # Get directory size using dust or du (can be slow on large dirs)
    local size_bytes=0
    if [ "$HAS_DUST" = "1" ]; then
        size_bytes=$(dust -sb -d 0 "$p" 2>/dev/null | awk '{print $1}' | head -1)
    else
        size_bytes=$(du -sb "$p" 2>/dev/null | awk '{print $1}')
    fi
    [ -z "$size_bytes" ] && size_bytes=0"""
    else:
        size_block = """
    # Size calculation disabled for speed
    local size_bytes=0"""

    # Bash script using fast tools with fallbacks
    script = f"""#!/bin/bash
set -o pipefail

# Detect available tools (cached for session)
HAS_FD=$(command -v fd >/dev/null 2>&1 && echo 1 || echo 0)
HAS_RG=$(command -v rg >/dev/null 2>&1 && echo 1 || echo 0)
HAS_DUST=$(command -v dust >/dev/null 2>&1 && echo 1 || echo 0)

scan_dir() {{
    local p="$1"
    # Check accessibility
    if [ ! -d "$p" ]; then
        printf '{{"path":"%s","error":"not a directory"}}\\n' "$p"
        return
    fi
    if [ ! -r "$p" ]; then
        printf '{{"path":"%s","error":"permission denied"}}\\n' "$p"
        return
    fi

    # Use fd for file/dir enumeration, fallback to find
    local files dirs
    if [ "$HAS_FD" = "1" ]; then
        files=$(fd -t f -d 1 . "$p" 2>/dev/null | wc -l)
        dirs=$(fd -t d -d 1 . "$p" 2>/dev/null | wc -l)
    else
        files=$(find "$p" -maxdepth 1 -type f 2>/dev/null | wc -l)
        dirs=$(find "$p" -maxdepth 1 -type d 2>/dev/null | tail -n +2 | wc -l)
    fi

    # Get child directories (full paths)
    local child_dirs=""
    if [ "$HAS_FD" = "1" ]; then
        while IFS= read -r d; do
            [ -z "$d" ] && continue
            [ -n "$child_dirs" ] && child_dirs="$child_dirs,"
            child_dirs="$child_dirs\\"$d\\""
        done < <(fd -t d -d 1 . "$p" 2>/dev/null)
    else
        while IFS= read -r d; do
            [ -z "$d" ] && continue
            [ -n "$child_dirs" ] && child_dirs="$child_dirs,"
            child_dirs="$child_dirs\\"$d\\""
        done < <(find "$p" -maxdepth 1 -type d 2>/dev/null | tail -n +2)
    fi

    # Get first 30 entry names for context
    local names=""
    local count=0
    for e in $(ls -1A "$p" 2>/dev/null | head -30); do
        [ -n "$names" ] && names="$names,"
        names="$names\\"$e\\""
        count=$((count + 1))
    done

    # Check for quality indicators
    local has_readme=false has_makefile=false has_git=false
    [ -f "$p/README.md" ] || [ -f "$p/README.rst" ] || [ -f "$p/README" ] && has_readme=true
    [ -f "$p/Makefile" ] || [ -f "$p/CMakeLists.txt" ] && has_makefile=true
    [ -d "$p/.git" ] && has_git=true

    # Count file extensions (for scoring)
    local ext_counts=""
    local ext_data=$(ls -1 "$p" 2>/dev/null | grep -E '\\.[a-zA-Z0-9]+$' | sed 's/.*\\.//' | sort | uniq -c | head -10)
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local cnt=$(echo "$line" | awk '{{print $1}}')
        local ext=$(echo "$line" | awk '{{print $2}}')
        [ -n "$ext_counts" ] && ext_counts="$ext_counts,"
        ext_counts="$ext_counts\\"$ext\\":$cnt"
    done <<< "$ext_data"

    # Use rg for pattern detection (quick, depth-limited)
    local rg_matches=""
    if [ "$HAS_RG" = "1" ]; then
{rg_block}
        rg_matches="{rg_json}"
    fi
{size_block}

    # Output JSON
    printf '{{"path":"%s","stats":{{"total_files":%d,"total_dirs":%d,"has_readme":%s,"has_makefile":%s,"has_git":%s,"size_bytes":%s,"file_type_counts":{{%s}},"rg_matches":{{%s}}}},"child_dirs":[%s],"child_names":[%s]}}\\n' \\
        "$p" "$files" "$dirs" "$has_readme" "$has_makefile" "$has_git" "$size_bytes" "$ext_counts" "$rg_matches" "$child_dirs" "$names"
}}

# Output as JSON array
echo "["
first=1
for p in {escaped_paths}; do
    [ $first -eq 0 ] && echo ","
    first=0
    scan_dir "$p"
done
echo "]"
"""
    return f"bash -c {shlex.quote(script)}"


def scan_paths(
    facility: str,
    paths: list[str],
    timeout: int = 300,
    enable_rg: bool = True,
    enable_size: bool = True,
) -> list[ScanResult]:
    """Scan multiple paths using bash script with ls.

    Uses run() for transparent local/SSH execution.
    Applies exclusion patterns from DiscoveryConfig.

    Args:
        facility: Facility identifier for SSH/local execution
        paths: List of directory paths to scan
        timeout: Command timeout in seconds
        enable_rg: If True, run rg pattern detection (slower).
                   If False, skip rg for faster enumeration-only scans.
        enable_size: If True, calculate directory size (can be very slow).
                     If False, skip size calculation for speed.
    """
    # Late import to avoid circular dependency
    from imas_codex.remote.tools import run

    if not paths:
        return []

    config = get_discovery_config()

    # Build and execute the scan script
    script = _build_scan_script(paths, enable_rg=enable_rg, enable_size=enable_size)

    try:
        output = run(script, facility=facility, timeout=timeout)
    except Exception as e:
        logger.warning(f"Scan failed for {facility}: {e}")
        return [
            ScanResult(path=p, stats=DirStats(), child_dirs=[], error=str(e))
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
            child_names=data.get("child_names", []),
            file_type_counts=stats_data.get("file_type_counts", {}),
            size_bytes=size_bytes,
            rg_matches=rg_matches,
        )

        # Filter child directories using exclusion config
        all_child_dirs = data.get("child_dirs", [])
        included_dirs, excluded_dirs = config.exclusions.filter_paths(all_child_dirs)

        # Log exclusions at debug level
        if excluded_dirs:
            logger.debug(
                f"Excluded {len(excluded_dirs)} dirs in {path}: "
                f"{[d for d, _ in excluded_dirs[:5]]}"
            )

        results.append(
            ScanResult(
                path=path,
                stats=stats,
                child_dirs=included_dirs,
                excluded_dirs=excluded_dirs,
                error=None,
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


def scan_paths_parallel(
    facility: str,
    paths: list[str],
    batch_size: int = 20,
    max_workers: int = 8,
    timeout: int = 60,
    enable_rg: bool = False,
    enable_size: bool = False,
) -> list[ScanResult]:
    """Scan paths using parallel SSH connections for higher throughput.

    Key insight: Each SSH call to ITER has ~8s network overhead due to ProxyJump.
    By running multiple SSH connections in parallel, we overlap this latency.

    Performance (ITER via ProxyJump, batch_size=20):
    - 4 workers: ~0.6s/path, 1.6 paths/sec
    - 8 workers: ~0.4s/path, 2.6 paths/sec (optimal)
    - 12 workers: ~0.4s/path, 2.6 paths/sec
    - 16 workers: ~0.5s/path (contention)

    Args:
        facility: Facility identifier
        paths: List of directory paths to scan
        batch_size: Paths per SSH call (default 20)
        max_workers: Number of parallel SSH connections (default 8)
        timeout: Timeout per batch in seconds
        enable_rg: Run ripgrep pattern detection (slower)
        enable_size: Calculate directory sizes (very slow)

    Returns:
        Combined list of ScanResult from all batches
    """
    if not paths:
        return []

    # Split into batches
    batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]

    logger.info(
        f"Parallel scan: {len(paths)} paths in {len(batches)} batches "
        f"({max_workers} workers)"
    )

    all_results: list[ScanResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(
                scan_paths, facility, batch, timeout, enable_rg, enable_size
            ): (i, batch)
            for i, batch in enumerate(batches)
        }

        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_idx, batch = future_to_batch[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.debug(f"Batch {batch_idx + 1}/{len(batches)} completed")
            except Exception as e:
                logger.warning(f"Batch {batch_idx + 1} failed: {e}")
                # Add error results for failed batch
                all_results.extend(
                    ScanResult(path=p, stats=DirStats(), child_dirs=[], error=str(e))
                    for p in batch
                )

    return all_results


def scan_facility_sync(
    facility: str,
    limit: int = 100,
    dry_run: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    batch_size: int = 20,
    max_workers: int = 8,
    enable_rg: bool = False,
    enable_size: bool = False,
) -> dict[str, int]:
    """Scan frontier paths using parallel SSH connections for high throughput.

    Key features:
    - Uses bash + fd/ls for fast directory scanning
    - Parallel SSH connections overlap network latency
    - Applies exclusion patterns from DiscoveryConfig before creating children
    - Tracks excluded directories in graph with status='excluded'

    Performance (ITER via ProxyJump):
    - 8 workers, batch_size=20: ~0.4s/path, 2.6 paths/sec
    - 150 paths/minute sustained throughput

    Args:
        facility: Facility identifier
        limit: Maximum number of paths to scan
        dry_run: If True, don't persist results
        progress_callback: Optional callback for progress updates
        batch_size: Paths per SSH call (default 20)
        max_workers: Number of parallel SSH connections (default 8)
        enable_rg: If True, run rg pattern detection. Default False for speed.
        enable_size: If True, calculate directory size. Default False (slow).
    """
    frontier = get_frontier(facility, limit=limit)

    if not frontier:
        logger.info(f"No frontier, seeding {facility}")
        seed_facility_roots(facility)
        frontier = get_frontier(facility, limit=limit)
        if not frontier:
            return {"scanned": 0, "children_created": 0, "excluded": 0, "errors": 0}

    paths = [p["path"] for p in frontier]
    total = len(paths)
    n_batches = (total + batch_size - 1) // batch_size
    logger.info(
        f"Scanning {total} paths for {facility} "
        f"({n_batches} batches, {max_workers} parallel workers)"
    )

    if progress_callback:
        progress_callback(0, total, f"starting {n_batches} batches...")

    # Use parallel scanning for throughput
    results = scan_paths_parallel(
        facility,
        paths,
        batch_size=batch_size,
        max_workers=max_workers,
        timeout=60,
        enable_rg=enable_rg,
        enable_size=enable_size,
    )

    if progress_callback:
        progress_callback(total, total, "processing results...")

    scanned = 0
    children_created = 0
    excluded_count = 0
    errors = 0

    if dry_run:
        # Just count, don't persist
        for r in results:
            if r.error:
                errors += 1
            else:
                scanned += 1
                excluded_count += len(r.excluded_dirs)
    else:
        # Batch persist all results in one transaction
        batch_data = [
            (r.path, r.stats.to_dict(), r.child_dirs, r.error) for r in results
        ]

        # Collect excluded directories with parent paths and reasons
        excluded_data = []
        for r in results:
            for excluded_path, reason in r.excluded_dirs:
                excluded_data.append((excluded_path, r.path, reason))

        stats = persist_scan_results(facility, batch_data, excluded=excluded_data)
        scanned += stats["scanned"]
        children_created += stats["children_created"]
        excluded_count += stats.get("excluded", 0)
        errors += stats["errors"]

    return {
        "scanned": scanned,
        "children_created": children_created,
        "excluded": excluded_count,
        "errors": errors,
    }
