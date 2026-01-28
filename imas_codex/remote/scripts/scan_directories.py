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
                "total_symlink_dirs": 2,
                "has_readme": true,
                "has_makefile": false,
                "has_git": true,
                "size_bytes": 0,
                "file_type_counts": {"py": 5, "md": 2},
                "rg_matches": {"total": 15}
            },
            "child_dirs": ["/path/to/scan/subdir", ...],
            "symlink_dirs": [
                {"path": "/path/to/scan/link", "realpath": "/real/target"}
            ],
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
from typing import Any, Dict, List


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
    rg_patterns: Dict[str, str],
    enable_rg: bool,
    enable_size: bool,
    has_rg: bool,
) -> Dict[str, Any]:
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
    result: Dict[str, Any] = {"path": path}

    # Check if path itself is a symlink and resolve it
    path_is_symlink = os.path.islink(path)
    path_realpath: str | None = None
    if path_is_symlink:
        try:
            path_realpath = sanitize_str(os.path.realpath(path))
        except OSError:
            pass
    result["is_symlink"] = path_is_symlink
    result["realpath"] = path_realpath

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
    files: List[os.DirEntry] = []
    dirs: List[os.DirEntry] = []
    symlink_dirs: List[os.DirEntry] = []
    for entry in entries:
        try:
            if entry.is_file(follow_symlinks=False):
                files.append(entry)
            elif entry.is_dir(follow_symlinks=False):
                # Check if it's a symlink pointing to a directory
                if entry.is_symlink():
                    symlink_dirs.append(entry)
                else:
                    dirs.append(entry)
        except OSError:
            # Entry may have been deleted or permission denied
            pass

    # Child directories (full paths) - sanitize for JSON encoding
    # Only include non-symlink directories for expansion
    child_dirs = [sanitize_str(entry.path) for entry in dirs]

    # Symlink directories with their resolved targets
    # Format: [{path: "/symlink/path", realpath: "/real/target"}]
    symlink_info: List[Dict[str, str]] = []
    for entry in symlink_dirs:
        try:
            resolved = os.path.realpath(entry.path)
            symlink_info.append(
                {
                    "path": sanitize_str(entry.path),
                    "realpath": sanitize_str(resolved),
                }
            )
        except OSError:
            # Can't resolve symlink, skip it
            pass

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
    ext_counts: Dict[str, int] = {}
    for f in files:
        suffix = Path(f.name).suffix
        if suffix:
            ext = sanitize_str(suffix.lstrip("."))
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    # rg pattern matching (if enabled and available)
    rg_matches: Dict[str, int] = {}
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

    # Git metadata extraction (if .git directory exists)
    git_remote_url: str | None = None
    git_head_commit: str | None = None
    git_branch: str | None = None
    git_root_commit: str | None = None

    if has_git:
        git_dir = os.path.join(path, ".git")
        if os.path.isdir(git_dir):
            # Git base command with safe.directory bypass for cross-user access
            # This is safe for read-only discovery operations
            git_base = ["git", "-c", "safe.directory=*", "-C", path]

            # Extract remote origin URL
            try:
                proc = subprocess.run(
                    git_base + ["config", "--get", "remote.origin.url"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    git_remote_url = sanitize_str(proc.stdout.strip())
            except (subprocess.TimeoutExpired, Exception):
                pass

            # Extract HEAD commit hash
            try:
                proc = subprocess.run(
                    git_base + ["rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    git_head_commit = sanitize_str(proc.stdout.strip())
            except (subprocess.TimeoutExpired, Exception):
                pass

            # Extract current branch name
            try:
                proc = subprocess.run(
                    git_base + ["symbolic-ref", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    git_branch = sanitize_str(proc.stdout.strip())
            except (subprocess.TimeoutExpired, Exception):
                pass

            # Extract root commit (first commit in history)
            try:
                proc = subprocess.run(
                    git_base + ["rev-list", "--max-parents=0", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    # May return multiple root commits (rare), take first
                    git_root_commit = sanitize_str(proc.stdout.strip().split("\n")[0])
            except (subprocess.TimeoutExpired, Exception):
                pass

    result["stats"] = {
        "total_files": len(files),
        "total_dirs": len(dirs),
        "total_symlink_dirs": len(symlink_dirs),
        "has_readme": has_readme,
        "has_makefile": has_makefile,
        "has_git": has_git,
        "git_remote_url": git_remote_url,
        "git_head_commit": git_head_commit,
        "git_branch": git_branch,
        "git_root_commit": git_root_commit,
        "size_bytes": size_bytes,
        "file_type_counts": ext_counts,
        "rg_matches": rg_matches,
    }
    result["child_dirs"] = child_dirs
    result["symlink_dirs"] = symlink_info
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

    paths: List[str] = config.get("paths", [])
    rg_patterns: Dict[str, str] = config.get("rg_patterns", {})
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
