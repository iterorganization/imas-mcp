#!/usr/bin/env python3
"""Remote file discovery script with pattern pre-filtering.

Combines file enumeration (fd/find) with rg pattern matching in a single
SSH call. Only files at depth=1 (directly in the directory) are scanned
since the paths pipeline has already walked subdirectories.

For each discovered file, runs rg pattern matching to provide enrichment
evidence that feeds into LLM scoring. Files with zero pattern matches
are still returned but marked as unenriched — the triage LLM decides
whether to keep them based on path/naming signals.

Requirements:
- Python 3.8+ (stdlib only, no external dependencies)
- Optional: fd (fast-find) for faster file enumeration
- Optional: rg (ripgrep) for pattern matching

Usage:
    echo '{"paths": [...], "extensions": [...], "pattern_categories": {...}}' | python3 discover_files.py

Input (JSON on stdin):
    {
        "paths": ["/path/to/scan", ...],
        "extensions": ["py", "f90", "c", ...],
        "max_depth": 1,
        "max_files_per_path": 500,
        "max_file_size": 1048576,
        "pattern_categories": {
            "mdsplus": "MDSplus|mdsplus|Tree\\(",
            "imas_read": "get_ids|imas\\.open|DBEntry",
            ...
        }
    }

Output (JSON on stdout):
    [
        {
            "path": "/path/to/scan",
            "files": [
                {
                    "path": "/path/to/scan/code.py",
                    "patterns": {"mdsplus": 3, "imas_read": 1},
                    "total_matches": 4,
                    "line_count": 142
                },
                ...
            ],
            "truncated": false,
            "error": null
        },
        ...
    ]
"""

import json
import os
import subprocess
import sys

# Default file size limit: 1 MB
DEFAULT_MAX_FILE_SIZE = 1 * 1024 * 1024

# Directories to skip
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


def has_command(cmd):
    # type: (str) -> bool
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for d in path_dirs:
        if os.path.isfile(os.path.join(d, cmd)):
            return True
    return False


def sanitize_str(s):
    # type: (str) -> str
    return s.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")


def _should_skip_path(path):
    # type: (str) -> bool
    parts = path.split(os.sep)
    return any(p in _SKIP_DIR_PATTERNS for p in parts)


def _enumerate_files_fd(path, extensions, max_depth, max_files, max_file_size):
    # type: (str, List[str], int, int, int) -> tuple
    ext_args = []
    for ext in extensions:
        ext_args.extend(["-e", ext])

    size_args = []
    if max_file_size > 0:
        if max_file_size >= 1024 * 1024 and max_file_size % (1024 * 1024) == 0:
            size_str = f"-{max_file_size // (1024 * 1024)}m"
        elif max_file_size >= 1024 and max_file_size % 1024 == 0:
            size_str = f"-{max_file_size // 1024}k"
        else:
            size_str = f"-{max_file_size}b"
        size_args = ["--size", size_str]

    cmd = (
        ["fd", ".", "--type", "f", "--max-depth", str(max_depth)]
        + size_args
        + ext_args
        + [path]
    )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        files = [
            sanitize_str(line.strip())
            for line in result.stdout.strip().splitlines()
            if line.strip() and not _should_skip_path(line.strip())
        ]
        truncated = len(files) > max_files
        if truncated:
            files = files[:max_files]
        return files, truncated
    except (subprocess.TimeoutExpired, Exception):
        return [], False


def _enumerate_files_find(path, extensions, max_depth, max_files, max_file_size):
    # type: (str, List[str], int, int, int) -> tuple
    ext_predicates = " -o ".join(f'-name "*.{ext}"' for ext in extensions)
    size_filter = f"-size -{max_file_size}c" if max_file_size > 0 else ""
    cmd = f"find {path} -maxdepth {max_depth} -type f {size_filter} \\( {ext_predicates} \\) 2>/dev/null"
    try:
        result = subprocess.run(
            ["sh", "-c", cmd], capture_output=True, text=True, timeout=30
        )
        files = [
            sanitize_str(line.strip())
            for line in result.stdout.strip().splitlines()
            if line.strip() and not _should_skip_path(line.strip())
        ]
        truncated = len(files) > max_files
        if truncated:
            files = files[:max_files]
        return files, truncated
    except (subprocess.TimeoutExpired, Exception):
        return [], False


def _count_lines(path):
    # type: (str) -> int
    try:
        result = subprocess.run(
            ["wc", "-l", path], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split()[0])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return 0


def _rg_count(pattern, path):
    # type: (str, str) -> int
    try:
        result = subprocess.run(
            ["rg", "-c", "--no-filename", pattern, path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return 0


def _enrich_file(path, pattern_categories, has_rg):
    # type: (str, Dict[str, str], bool) -> Dict[str, Any]
    info = {
        "path": sanitize_str(path),
        "patterns": {},
        "total_matches": 0,
        "line_count": _count_lines(path),
    }
    if has_rg and pattern_categories:
        total = 0
        for category, pattern in pattern_categories.items():
            count = _rg_count(pattern, path)
            if count > 0:
                info["patterns"][category] = count
                total += count
        info["total_matches"] = total
    return info


def discover_path(
    path,
    extensions,
    max_depth,
    max_files,
    max_file_size,
    pattern_categories,
    use_fd,
    has_rg,
):
    # type: (str, List[str], int, int, int, Dict[str, str], bool, bool) -> Dict[str, Any]
    if not os.path.isdir(path):
        return {
            "path": sanitize_str(path),
            "files": [],
            "truncated": False,
            "error": "not_a_directory",
        }

    if use_fd:
        files, truncated = _enumerate_files_fd(
            path, extensions, max_depth, max_files, max_file_size
        )
    else:
        files, truncated = _enumerate_files_find(
            path, extensions, max_depth, max_files, max_file_size
        )

    # Enrich each file with pattern matching
    enriched_files = []
    for f in files:
        enriched_files.append(_enrich_file(f, pattern_categories, has_rg))

    return {
        "path": sanitize_str(path),
        "files": enriched_files,
        "truncated": truncated,
        "error": None,
    }


def main():
    input_data = json.loads(sys.stdin.read())
    paths = input_data.get("paths", [])
    extensions = input_data.get("extensions", ["py"])
    max_depth = input_data.get("max_depth", 1)
    max_files_per_path = input_data.get("max_files_per_path", 500)
    max_file_size = input_data.get("max_file_size", DEFAULT_MAX_FILE_SIZE)
    pattern_categories = input_data.get("pattern_categories", {})

    use_fd = has_command("fd")
    has_rg = has_command("rg")

    results = []
    for path in paths:
        result = discover_path(
            path,
            extensions,
            max_depth,
            max_files_per_path,
            max_file_size,
            pattern_categories,
            use_fd,
            has_rg,
        )
        results.append(result)

    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
