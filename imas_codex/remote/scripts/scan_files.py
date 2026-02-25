#!/usr/bin/env python3
"""Remote file scanner script.

This script is executed on remote facilities via SSH. It scans directories
for source files matching supported extensions and returns JSON with file lists.

Requirements:
- Python 3.8+ (stdlib only, no external dependencies)
- Optional: fd (fast-find) for faster file enumeration

Usage:
    echo '{"paths": [...], "extensions": [...], "max_depth": 5}' | python3 scan_files.py

Input (JSON on stdin):
    {
        "paths": ["/path/to/scan", ...],
        "extensions": ["py", "f90", "c", ...],
        "max_depth": 5,
        "max_files_per_path": 5000
    }

Output (JSON on stdout):
    [
        {
            "path": "/path/to/scan",
            "files": ["/path/to/scan/code.py", ...],
            "error": null
        },
        ...
    ]
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, List

# Default file size limit: 1 MB (user code is typically small)
DEFAULT_MAX_FILE_SIZE = 1 * 1024 * 1024


def has_command(cmd: str) -> bool:
    """Check if command exists in PATH."""
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for path_dir in path_dirs:
        if os.path.isfile(os.path.join(path_dir, cmd)):
            return True
    return False


def sanitize_str(s: str) -> str:
    """Remove surrogate characters that cannot be encoded as JSON."""
    return s.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")


def _filter_by_size(files: List[str], max_size: int) -> tuple:
    """Filter files by size, returning (kept, skipped_count)."""
    if max_size <= 0:
        return files, 0
    kept = []
    skipped = 0
    for f in files:
        try:
            if os.path.getsize(f) <= max_size:
                kept.append(f)
            else:
                skipped += 1
        except OSError:
            kept.append(f)  # Keep if stat fails (permission etc.)
    return kept, skipped


# Directories that indicate machine data, caches, or build artifacts
_SKIP_DIR_PATTERNS = {
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    ".tox",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    ".local",
    "site-packages",
    "dist-packages",
    ".venv",
    "venv",
}


def _should_skip_path(path: str) -> bool:
    """Check if a file path contains skip-worthy directory components."""
    parts = path.split(os.sep)
    return any(p in _SKIP_DIR_PATTERNS for p in parts)


def scan_path_fd(
    path: str,
    extensions: List[str],
    max_depth: int,
    max_files: int,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> Dict[str, Any]:
    """Scan a path using fd (fast-find)."""
    ext_args = []
    for ext in extensions:
        ext_args.extend(["-e", ext])

    # fd --size requires human-readable units: -1m, -500k, etc.
    # Convert bytes to a suitable unit string for fd
    if max_file_size > 0:
        if max_file_size >= 1024 * 1024 and max_file_size % (1024 * 1024) == 0:
            size_str = f"-{max_file_size // (1024 * 1024)}m"
        elif max_file_size >= 1024 and max_file_size % 1024 == 0:
            size_str = f"-{max_file_size // 1024}k"
        else:
            size_str = f"-{max_file_size}b"
        size_args = ["--size", size_str]
    else:
        size_args = []

    cmd = (
        ["fd", ".", "--type", "f", "--max-depth", str(max_depth)]
        + size_args
        + ext_args
        + [path]
    )
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        files = [
            sanitize_str(line.strip())
            for line in result.stdout.strip().splitlines()
            if line.strip() and not _should_skip_path(line.strip())
        ]
        truncated = len(files) > max_files
        if truncated:
            files = files[:max_files]
        return {
            "path": sanitize_str(path),
            "files": files,
            "truncated": truncated,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "path": sanitize_str(path),
            "files": [],
            "truncated": False,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "path": sanitize_str(path),
            "files": [],
            "truncated": False,
            "error": str(e)[:200],
        }


def scan_path_find(
    path: str,
    extensions: List[str],
    max_depth: int,
    max_files: int,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> Dict[str, Any]:
    """Scan a path using find (fallback)."""
    ext_predicates = " -o ".join(f'-name "*.{ext}"' for ext in extensions)
    # Add -size filter when max_file_size is set
    size_filter = f"-size -{max_file_size}c" if max_file_size > 0 else ""
    cmd = (
        f"find {path} -maxdepth {max_depth} -type f {size_filter} "
        f"\\( {ext_predicates} \\) 2>/dev/null"
    )

    try:
        result = subprocess.run(
            ["sh", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=60,
        )
        files = [
            sanitize_str(line.strip())
            for line in result.stdout.strip().splitlines()
            if line.strip() and not _should_skip_path(line.strip())
        ]
        truncated = len(files) > max_files
        if truncated:
            files = files[:max_files]
        return {
            "path": sanitize_str(path),
            "files": files,
            "truncated": truncated,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "path": sanitize_str(path),
            "files": [],
            "truncated": False,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "path": sanitize_str(path),
            "files": [],
            "truncated": False,
            "error": str(e)[:200],
        }


def main():
    input_data = json.loads(sys.stdin.read())
    paths = input_data.get("paths", [])
    extensions = input_data.get("extensions", ["py"])
    max_depth = input_data.get("max_depth", 5)
    max_files_per_path = input_data.get("max_files_per_path", 5000)
    max_file_size = input_data.get("max_file_size", DEFAULT_MAX_FILE_SIZE)

    use_fd = has_command("fd")
    results = []

    for path in paths:
        if not os.path.isdir(path):
            results.append(
                {
                    "path": sanitize_str(path),
                    "files": [],
                    "truncated": False,
                    "error": "not_a_directory",
                }
            )
            continue

        if use_fd:
            result = scan_path_fd(
                path, extensions, max_depth, max_files_per_path, max_file_size
            )
        else:
            result = scan_path_find(
                path, extensions, max_depth, max_files_per_path, max_file_size
            )

        results.append(result)

    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
