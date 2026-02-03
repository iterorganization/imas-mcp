#!/usr/bin/env python3
"""Remote directory enrichment script.

This script is executed on remote facilities via SSH. It runs deep analysis
on directories to collect enrichment data for high-value paths.

Category-aware pattern selection:
- Code paths (modeling_code, analysis_code, etc.): Full code + data patterns
- Documentation paths: Skip code pattern matching (focus on size/structure)
- Data paths (experimental_data, modeling_data): Focus on data format patterns

Requirements:
- Python 3.8+ (stdlib only, no external dependencies)
- du (standard Unix utility, always available)
- Optional: rg (ripgrep) for pattern matching
- Optional: tokei for lines of code analysis

Usage:
    echo '{"paths": ["/home/codes/project"], "path_purposes": {"/home/codes/project": "modeling_code"}}' | python3 enrich_directories.py

Input (JSON on stdin):
    {
        "paths": ["/path/to/enrich", ...],
        "path_purposes": {"/path": "purpose", ...}  # Optional
    }

Output (JSON on stdout):
    [
        {
            "path": "/path/to/enrich",
            "read_matches": 5,
            "write_matches": 3,
            "total_bytes": 1048576,
            "total_lines": 5000,
            "language_breakdown": {"Python": 3000, "Fortran": 2000}
        },
        ...
    ]
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

# Purposes that are primarily documentation - skip code pattern matching
DOC_PURPOSES = {"documentation"}

# Purposes that are primarily data - focus on data patterns, skip code LOC
DATA_PURPOSES = {"experimental_data", "modeling_data"}

# Purposes that are containers/system - minimal enrichment
SKIP_PATTERNS_PURPOSES = {"container", "system", "build_artifact", "archive"}


# ============================================================================
# Categorized Data Access Patterns
# ============================================================================
# Goal: Discover how data is accessed NATIVELY at each facility.
# We intentionally weight all data systems equally - IMAS is just one option.
# Understanding native patterns helps design better integration strategies.

# Pattern categories for data access detection
# Each category is searched independently to provide a breakdown
PATTERN_CATEGORIES = {
    # Native/facility-specific data access (highest priority - this is what we want to discover)
    "mdsplus": r"mdsconnect|mdsopen|mdsvalue|MDSplus|TdiExecute|connection\.openTree",
    "ppf": r"ppf\.read|ppfget|jet\.ppf|ppfgo|ppfuid",  # JET PPF system
    "ufile": r"ufile\.read|read_ufile|ufiles|rdufile",  # UFILE format
    # Standard scientific formats
    "hdf5": r"h5py\.File|hdf5|create_dataset|to_hdf|\.h5",
    "netcdf": r"netCDF4|xr\.open|xarray\.open|to_netcdf|\.nc",
    # Equilibrium formats
    "eqdsk": r"read_eqdsk|load_eqdsk|from_eqdsk|write_eqdsk|to_eqdsk|geqdsk",
    # Standard formats
    "pickle": r"pickle\.load|pickle\.dump|\.pkl",
    "csv": r"\.csv|read_csv|to_csv|csv\.reader|csv\.writer",
    "mat": r"scipy\.io\.loadmat|savemat|sio\.loadmat|\.mat",
    # IMAS (one of many options, not privileged)
    "imas": r"imas\.database|get_ids|put_slice|ids\.put|imas\.create|get_slice",
}

# Read vs Write pattern suffixes (for determining data flow direction)
READ_SUFFIXES = r"read|load|open|get|from|import|fetch"
WRITE_SUFFIXES = r"write|save|put|create|to|export|dump"


def sanitize_str(s: str) -> str:
    """Remove surrogate characters that cannot be encoded as JSON."""
    return s.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")


def has_command(cmd: str) -> bool:
    """Check if command exists in PATH."""
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for path_dir in path_dirs:
        if os.path.isfile(os.path.join(path_dir, cmd)):
            return True
    return False


def count_pattern_matches(path: str, pattern: str, timeout: int = 30) -> int:
    """Count matches for a pattern using rg.

    Args:
        path: Directory to search
        pattern: Regex pattern to search for
        timeout: Command timeout in seconds

    Returns:
        Total match count across all files
    """
    try:
        proc = subprocess.run(
            ["rg", "-c", "--no-messages", "--max-depth", "3", "-e", pattern, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            total = 0
            for line in proc.stdout.strip().split("\n"):
                if ":" in line:
                    try:
                        total += int(line.rsplit(":", 1)[-1])
                    except ValueError:
                        pass
            return total
    except (subprocess.TimeoutExpired, Exception):
        pass
    return 0


def enrich_directory(
    path: str,
    has_rg: bool,
    has_tokei: bool,
    purpose: Optional[str] = None,
) -> Dict[str, Any]:
    """Enrich a single directory and return results dict.

    Uses purpose to target pattern matching:
    - Documentation: Skip code pattern matching (wastes time)
    - Data: Skip LOC analysis (not code)
    - Container/system: Minimal enrichment (size only)
    - Code: Full analysis

    Returns pattern_categories dict with per-category match counts,
    allowing discovery of native data access patterns at each facility.

    Args:
        path: Directory path to enrich
        has_rg: Whether rg command is available
        has_tokei: Whether tokei command is available
        purpose: Path purpose category (e.g., "modeling_code", "documentation")

    Returns:
        Dict with path, pattern_categories, size, and lines of code
    """
    result: Dict[str, Any] = {"path": sanitize_str(path)}

    if not os.path.isdir(path):
        result["error"] = "not a directory"
        return result

    if not os.access(path, os.R_OK):
        result["error"] = "permission denied"
        return result

    # Determine what to analyze based on purpose
    skip_patterns = purpose in SKIP_PATTERNS_PURPOSES
    skip_code_patterns = purpose in DOC_PURPOSES
    skip_loc = purpose in DATA_PURPOSES

    # Categorized pattern matching with rg (skip for docs/containers)
    pattern_categories: Dict[str, int] = {}
    read_matches = 0
    write_matches = 0

    if has_rg and not skip_patterns and not skip_code_patterns:
        # Search each pattern category independently
        for category, pattern in PATTERN_CATEGORIES.items():
            matches = count_pattern_matches(path, pattern)
            if matches > 0:
                pattern_categories[category] = matches
                # Classify as read or write based on pattern content
                # This is approximate but helps understand data flow
                if any(
                    r in pattern.lower()
                    for r in ["read", "load", "open", "get", "from"]
                ):
                    read_matches += matches
                if any(
                    w in pattern.lower()
                    for w in ["write", "save", "put", "create", "to"]
                ):
                    write_matches += matches

    result["pattern_categories"] = pattern_categories
    result["read_matches"] = read_matches
    result["write_matches"] = write_matches

    # Storage size analysis using du -sb (reliable byte output)
    # Note: dust is a visual tool without simple bytes-only output,
    # so we use du -sb which outputs "BYTES\tPATH" format
    total_bytes = 0
    try:
        proc = subprocess.run(
            ["du", "-sb", path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                total_bytes = int(proc.stdout.split()[0])
            except (ValueError, IndexError):
                pass
    except (subprocess.TimeoutExpired, Exception):
        pass

    result["total_bytes"] = total_bytes

    # Lines of code analysis with tokei (skip for data paths - not code)
    total_lines = 0
    language_breakdown: Dict[str, int] = {}

    if has_tokei and not skip_loc and not skip_patterns:
        try:
            proc = subprocess.run(
                ["tokei", path, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                try:
                    tokei_data = json.loads(proc.stdout)
                    # tokei JSON format: {"Python": {"code": 1000, ...}, ...}
                    for lang, stats in tokei_data.items():
                        if lang == "Total":
                            continue
                        if isinstance(stats, dict):
                            code = stats.get("code", 0)
                            if code > 0:
                                language_breakdown[lang] = code
                                total_lines += code
                except json.JSONDecodeError:
                    pass
        except (subprocess.TimeoutExpired, Exception):
            pass

    result["total_lines"] = total_lines
    result["language_breakdown"] = language_breakdown

    return result


def main() -> None:
    """Read config from stdin, enrich directories, output JSON to stdout."""
    # Read configuration from stdin
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}), file=sys.stderr)
        sys.exit(1)

    paths: List[str] = config.get("paths", [])
    path_purposes: Dict[str, str] = config.get("path_purposes", {})

    # Check for tools once
    has_rg = has_command("rg")
    has_tokei = has_command("tokei")

    # Enrich all paths with their purpose (for targeted pattern selection)
    results = [
        enrich_directory(
            p,
            has_rg,
            has_tokei,
            purpose=path_purposes.get(p),
        )
        for p in paths
    ]

    # Output JSON (handles all escaping correctly)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
