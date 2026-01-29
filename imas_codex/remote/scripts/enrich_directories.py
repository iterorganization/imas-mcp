#!/usr/bin/env python3
"""Remote directory enrichment script.

This script is executed on remote facilities via SSH. It runs deep analysis
on directories to collect enrichment data for high-value paths.

Requirements:
- Python 3.8+ (stdlib only, no external dependencies)
- Optional: rg (ripgrep) for pattern matching
- Optional: dust or du for size analysis
- Optional: tokei for lines of code analysis

Usage:
    echo '{"paths": ["/home/codes/project"]}' | python3 enrich_directories.py

Input (JSON on stdin):
    {
        "paths": ["/path/to/enrich", ...]
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
from typing import Any


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


# Simplified patterns for format detection (kept short to avoid arg length issues)
READ_PATTERN = (
    r"read_eqdsk|load_eqdsk|from_eqdsk|"
    r"mdsconnect|mdsopen|MDSplus|TdiExecute|"
    r"h5py\.File|netCDF4|hdf5|"
    r"xr\.open|xarray\.open|"
    r"json\.load|pickle\.load|"
    r"imas\.database|get_ids|get_slice|"
    r"ppf\.read|ppfget"
)

WRITE_PATTERN = (
    r"put_slice|ids\.put|imas\.create|"
    r"to_hdf|hdf5\.write|create_dataset|"
    r"to_netcdf|netcdf\.create|"
    r"json\.dump|pickle\.dump|"
    r"write_eqdsk|to_eqdsk"
)


def enrich_directory(
    path: str,
    has_rg: bool,
    has_dust: bool,
    has_tokei: bool,
) -> dict[str, Any]:
    """Enrich a single directory and return results dict.

    Args:
        path: Directory path to enrich
        has_rg: Whether rg command is available
        has_dust: Whether dust command is available
        has_tokei: Whether tokei command is available

    Returns:
        Dict with path, pattern matches, size, and lines of code
    """
    result: dict[str, Any] = {"path": sanitize_str(path)}

    if not os.path.isdir(path):
        result["error"] = "not a directory"
        return result

    if not os.access(path, os.R_OK):
        result["error"] = "permission denied"
        return result

    # Pattern matching with rg
    read_matches = 0
    write_matches = 0

    if has_rg:
        # Count read pattern matches
        try:
            proc = subprocess.run(
                [
                    "rg",
                    "-c",
                    "--no-messages",
                    "--max-depth",
                    "3",
                    "-e",
                    READ_PATTERN,
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                for line in proc.stdout.strip().split("\n"):
                    if ":" in line:
                        try:
                            read_matches += int(line.rsplit(":", 1)[-1])
                        except ValueError:
                            pass
        except (subprocess.TimeoutExpired, Exception):
            pass

        # Count write pattern matches
        try:
            proc = subprocess.run(
                [
                    "rg",
                    "-c",
                    "--no-messages",
                    "--max-depth",
                    "3",
                    "-e",
                    WRITE_PATTERN,
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                for line in proc.stdout.strip().split("\n"):
                    if ":" in line:
                        try:
                            write_matches += int(line.rsplit(":", 1)[-1])
                        except ValueError:
                            pass
        except (subprocess.TimeoutExpired, Exception):
            pass

    result["read_matches"] = read_matches
    result["write_matches"] = write_matches

    # Storage size analysis
    total_bytes = 0
    if has_dust:
        try:
            proc = subprocess.run(
                ["dust", "-sb", path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                # dust -sb outputs: "1234 /path"
                first_line = proc.stdout.strip().split("\n")[0]
                parts = first_line.split()
                if parts:
                    try:
                        total_bytes = int(parts[0])
                    except ValueError:
                        pass
        except (subprocess.TimeoutExpired, Exception):
            pass
    else:
        # Fallback to du
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

    # Lines of code analysis with tokei
    total_lines = 0
    language_breakdown: dict[str, int] = {}

    if has_tokei:
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

    paths: list[str] = config.get("paths", [])

    # Check for tools once
    has_rg = has_command("rg")
    has_dust = has_command("dust")
    has_tokei = has_command("tokei")

    # Enrich all paths
    results = [enrich_directory(p, has_rg, has_dust, has_tokei) for p in paths]

    # Output JSON (handles all escaping correctly)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
