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
#
# IMPORTANT: These are FALLBACK patterns used only when the caller does not
# pass pattern_categories in the input config. The canonical patterns are
# defined in imas_codex/config/patterns/scoring/data_systems.yaml and
# imas_codex/discovery/paths/enrichment.py PATTERN_REGISTRY. The enrichment
# caller (enrichment.py) builds the combined pattern dict from those sources
# and passes it via stdin, so these fallbacks are only used for standalone
# script execution.

# Fallback pattern categories (used when caller doesn't provide patterns)
DEFAULT_PATTERN_CATEGORIES = {
    # Native/facility-specific data access
    "mdsplus": r"MDSplus|mdsplus|from\s+MDSplus|Tree\(|openTree|TdiCompile|TdiExecute|tdi\(|MdsValue|Connection\(|makeConnection|MdsConnect",
    "ppf": r"ppf\.|ppfgo|ppfget|ppfuid|pulse_file|JETPPF",
    "ufile": r"ufile\.read|read_ufile|ufiles|rdufile|UFILE",
    # Standard scientific formats
    "hdf5": r"h5py|import\s+h5py|from\s+h5py|HDFStore|to_hdf|read_hdf|\.h5|\.hdf5|\.hdf",
    "netcdf": r"netCDF4|import\s+netCDF4|from\s+netCDF4|xarray.*open_dataset|\.nc|\.nc4",
    # Equilibrium formats
    "eqdsk": r"read_eqdsk|load_eqdsk|from_eqdsk|write_eqdsk|to_eqdsk|geqdsk",
    # IMAS
    "imas": r"imas\.imasdef|imas\.DBEntry|from\s+imas\s+import|ids_properties|homogeneous_time|put_slice|get_slice|write_ids|read_ids|DDEntry|IMASpy",
    # Convention / coordinate / sign patterns
    "cocos": r"COCOS|cocos_[0-9]+|cocos_transform|cocos_identify|cocosify|set_cocos|get_cocos",
    "sign_convention": r"sign_convention|ip_sign|bt_sign|sign_bp|sign_b0|sigma_ip|sigma_b0|sigma_rphiz|sign_q",
    "coord_convention": r"coordinate_convention|coord_system|phi_convention|theta_convention|rho_tor|rho_pol|psi_norm|psi_sign",
    "unit_conversion": r"unit_convert|units_to|to_si|from_si|pint\.Unit|ureg\.|convert_units",
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
    pattern_categories: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Enrich a single directory and return results dict.

    Runs du and rg concurrently for speed, then tokei (which needs du result
    for timeout calculation). Each command has independent timeout handling —
    a du timeout doesn't block rg results and vice versa.

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
        pattern_categories: Optional mapping of category name to rg regex pattern.
            Falls back to DEFAULT_PATTERN_CATEGORIES if not provided.

    Returns:
        Dict with path, pattern_categories, size, and lines of code
    """
    from concurrent.futures import Future, ThreadPoolExecutor

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

    warnings: List[str] = []

    # --- Run du and rg concurrently ---
    # These are independent I/O operations that benefit from parallelism.
    # tokei runs after du because its timeout scales by total_bytes.

    def _run_du() -> tuple:
        """Run du -sb and return (total_bytes, timed_out).

        Parse stdout regardless of return code: du returns non-zero
        when it encounters permission-denied subdirectories (common on
        NFS/GPFS) but still writes the valid total to stdout.
        """
        try:
            proc = subprocess.run(
                ["du", "-sb", path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.stdout.strip():
                try:
                    return int(proc.stdout.split()[0]), False
                except (ValueError, IndexError):
                    pass
        except subprocess.TimeoutExpired:
            return 0, True
        except Exception:
            pass
        return 0, False

    def _run_rg() -> tuple:
        """Run rg pattern matching and return (matched_categories, read_matches, write_matches)."""
        matched_categories: Dict[str, int] = {}
        read_matches = 0
        write_matches = 0

        if not has_rg or skip_patterns or skip_code_patterns:
            return matched_categories, read_matches, write_matches

        cats = pattern_categories or DEFAULT_PATTERN_CATEGORIES
        for category, pattern in cats.items():
            matches = count_pattern_matches(path, pattern)
            if matches > 0:
                matched_categories[category] = matches
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

        return matched_categories, read_matches, write_matches

    # Run du and rg in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        du_future: Future = executor.submit(_run_du)
        rg_future: Future = executor.submit(_run_rg)

        total_bytes, du_timed_out = du_future.result()
        matched_categories, read_matches, write_matches = rg_future.result()

    if du_timed_out:
        warnings.append("du_timeout")

    result["pattern_categories"] = matched_categories
    result["read_matches"] = read_matches
    result["write_matches"] = write_matches
    result["total_bytes"] = total_bytes

    # --- Run tokei sequentially (needs total_bytes for timeout scaling) ---
    total_lines = 0
    language_breakdown: Dict[str, int] = {}

    if has_tokei and not skip_loc and not skip_patterns:
        tokei_timeout = (
            30 + int(total_bytes / 1_000_000_000) * 15 if total_bytes > 0 else 60
        )
        tokei_timeout = min(tokei_timeout, 120)
        try:
            proc = subprocess.run(
                ["tokei", path, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=tokei_timeout,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                try:
                    tokei_data = json.loads(proc.stdout)
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
        except subprocess.TimeoutExpired:
            warnings.append(f"tokei_timeout({total_bytes}B)")
        except Exception:
            pass

    result["total_lines"] = total_lines
    result["language_breakdown"] = language_breakdown

    if warnings:
        result["warnings"] = warnings

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
    # Accept patterns from caller; fall back to built-in defaults
    pattern_categories: Dict[str, str] = config.get(
        "pattern_categories", DEFAULT_PATTERN_CATEGORIES
    )

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
            pattern_categories=pattern_categories,
        )
        for p in paths
    ]

    # Output JSON (handles all escaping correctly)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
