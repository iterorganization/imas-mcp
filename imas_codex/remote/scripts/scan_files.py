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


def scan_path_fd(
    path: str,
    extensions: List[str],
    max_depth: int,
    max_files: int,
) -> Dict[str, Any]:
    """Scan a path using fd (fast-find)."""
    ext_args = []
    for ext in extensions:
        ext_args.extend(["-e", ext])

    cmd = ["fd", ".", "--type", "f", "--max-depth", str(max_depth)] + ext_args + [path]
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
            if line.strip()
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
) -> Dict[str, Any]:
    """Scan a path using find (fallback)."""
    ext_predicates = " -o ".join(f'-name "*.{ext}"' for ext in extensions)
    cmd = f"find {path} -maxdepth {max_depth} -type f \\( {ext_predicates} \\) 2>/dev/null"

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
            if line.strip()
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
            result = scan_path_fd(path, extensions, max_depth, max_files_per_path)
        else:
            result = scan_path_find(path, extensions, max_depth, max_files_per_path)

        results.append(result)

    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
