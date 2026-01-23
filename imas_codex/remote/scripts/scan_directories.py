#!/usr/bin/env python3
"""Remote directory scanner script.

This script is executed on remote facilities via SSH. It scans directories
and returns JSON with file counts, child directories, and optional rg matches.

Requirements:
- Python 3.8+ (stdlib only, no external dependencies)
- Optional: rg (ripgrep) for pattern matching

Usage:
    echo '{"paths": ["/home/user"], "enable_rg": false}' | python3 scan_directories.py

Input (JSON on stdin):
    {
        "paths": ["/path/to/scan", ...],
        "rg_patterns": {"category": "pattern|pattern", ...},
        "enable_rg": true,
        "enable_size": false
    }

Output (JSON on stdout):
    [
        {
            "path": "/path/to/scan",
            "stats": {
                "total_files": 10,
                "total_dirs": 5,
                "has_readme": true,
                "has_makefile": false,
                "has_git": true,
                "size_bytes": 0,
                "file_type_counts": {"py": 5, "md": 2},
                "rg_matches": {"total": 15}
            },
            "child_dirs": ["/path/to/scan/subdir", ...],
            "child_names": ["file1.py", "subdir", ...]
        },
        ...
    ]
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def sanitize_str(s: str) -> str:
    """Remove surrogate characters that cannot be encoded as JSON.

    Some filesystems have filenames with invalid UTF-8 bytes. Python represents
    these as surrogate characters which json.dumps() cannot encode.
    """
    return s.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")


def has_command(cmd: str) -> bool:
    """Check if command exists in PATH."""
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for path_dir in path_dirs:
        if os.path.isfile(os.path.join(path_dir, cmd)):
            return True
    return False


def scan_directory(
    path: str,
    rg_patterns: dict[str, str],
    enable_rg: bool,
    enable_size: bool,
    has_rg: bool,
) -> dict[str, Any]:
    """Scan a single directory and return results dict.

    Args:
        path: Directory path to scan
        rg_patterns: Dict of category -> regex pattern for ripgrep
        enable_rg: Whether to run rg pattern matching
        enable_size: Whether to calculate directory size
        has_rg: Whether rg command is available

    Returns:
        Dict with path, stats, child_dirs, child_names, and optional error
    """
    result: dict[str, Any] = {"path": path}

    # Check accessibility
    if not os.path.isdir(path):
        result["error"] = "not a directory"
        return result
    if not os.access(path, os.R_OK):
        result["error"] = "permission denied"
        return result

    # Enumerate directory entries using scandir (fast, single syscall)
    try:
        entries = list(os.scandir(path))
    except PermissionError:
        result["error"] = "permission denied"
        return result
    except OSError as e:
        result["error"] = str(e)
        return result

    # Separate files and directories (avoid following symlinks)
    files: list[os.DirEntry] = []
    dirs: list[os.DirEntry] = []
    for entry in entries:
        try:
            if entry.is_file(follow_symlinks=False):
                files.append(entry)
            elif entry.is_dir(follow_symlinks=False):
                dirs.append(entry)
        except OSError:
            # Entry may have been deleted or permission denied
            pass

    # Child directories (full paths) - sanitize for JSON encoding
    child_dirs = [sanitize_str(entry.path) for entry in dirs]

    # First 30 names for context (for LLM scoring) - sanitize for JSON encoding
    child_names = [sanitize_str(entry.name) for entry in entries[:30]]

    # Quality indicators (case-insensitive check)
    names_lower: set[str] = set()
    for entry in entries:
        names_lower.add(entry.name.lower())

    has_readme = any(
        n in names_lower for n in ("readme.md", "readme.rst", "readme", "readme.txt")
    )
    has_makefile = "makefile" in names_lower or "cmakelists.txt" in names_lower
    has_git = ".git" in names_lower

    # Extension counts (for file type distribution) - sanitize keys for JSON
    ext_counts: dict[str, int] = {}
    for f in files:
        suffix = Path(f.name).suffix
        if suffix:
            ext = sanitize_str(suffix.lstrip("."))
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    # rg pattern matching (if enabled and available)
    rg_matches: dict[str, int] = {}
    if enable_rg and has_rg and rg_patterns:
        all_patterns = "|".join(rg_patterns.values())
        try:
            proc = subprocess.run(
                ["rg", "-c", "--max-depth", "1", all_patterns, path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                total = 0
                for line in proc.stdout.strip().split("\n"):
                    if ":" in line:
                        try:
                            total += int(line.rsplit(":", 1)[-1])
                        except ValueError:
                            pass
                rg_matches["total"] = total
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

    # Size calculation (if enabled)
    size_bytes = 0
    if enable_size:
        try:
            proc = subprocess.run(
                ["du", "-sb", path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                size_bytes = int(proc.stdout.split()[0])
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        except Exception:
            pass

    result["stats"] = {
        "total_files": len(files),
        "total_dirs": len(dirs),
        "has_readme": has_readme,
        "has_makefile": has_makefile,
        "has_git": has_git,
        "size_bytes": size_bytes,
        "file_type_counts": ext_counts,
        "rg_matches": rg_matches,
    }
    result["child_dirs"] = child_dirs
    result["child_names"] = child_names

    return result


def main() -> None:
    """Read config from stdin, scan directories, output JSON to stdout."""
    # Read configuration from stdin
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}), file=sys.stderr)
        sys.exit(1)

    paths: list[str] = config.get("paths", [])
    rg_patterns: dict[str, str] = config.get("rg_patterns", {})
    enable_rg: bool = config.get("enable_rg", False)
    enable_size: bool = config.get("enable_size", False)

    # Check for rg once
    has_rg = has_command("rg")

    # Scan all paths
    results = [
        scan_directory(p, rg_patterns, enable_rg, enable_size, has_rg) for p in paths
    ]

    # Output JSON (handles all escaping correctly)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
