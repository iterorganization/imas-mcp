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
from typing import Any, Dict, List, Optional


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

    # Resolve symlink status and realpath for the scanned path itself
    is_symlink = os.path.islink(path)
    try:
        realpath = os.path.realpath(path) if is_symlink else None
    except OSError:
        realpath = None

    # Get device:inode for deduplication (detects bind mounts)
    # Format: "device:inode" e.g. "64:9468985"
    device_inode: Optional[str] = None
    try:
        stat_info = os.stat(path)
        device_inode = f"{stat_info.st_dev}:{stat_info.st_ino}"
    except OSError:
        pass

    # Check accessibility
    if not os.path.isdir(path):
        result["error"] = "not a directory"
        result["is_symlink"] = is_symlink
        result["realpath"] = realpath
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
    for entry in entries:
        try:
            if entry.is_file(follow_symlinks=False):
                files.append(entry)
            elif entry.is_dir(follow_symlinks=False):
                dirs.append(entry)
        except OSError:
            # Entry may have been deleted or permission denied
            pass

    # Child directories with symlink and device_inode info - enables deduplication
    # Each entry is {path, is_symlink, realpath, device_inode} for graph relationship creation
    child_dirs: List[Dict[str, Any]] = []
    for entry in dirs:
        child_path = sanitize_str(entry.path)
        child_device_inode: Optional[str] = None
        try:
            child_is_symlink = entry.is_symlink()
            if child_is_symlink:
                try:
                    child_realpath = sanitize_str(os.path.realpath(entry.path))
                except OSError:
                    child_realpath = None
            else:
                child_realpath = None
            # Get device:inode for deduplication
            try:
                child_stat = os.stat(entry.path)  # follows symlinks
                child_device_inode = f"{child_stat.st_dev}:{child_stat.st_ino}"
            except OSError:
                pass
        except OSError:
            child_is_symlink = False
            child_realpath = None
        child_dirs.append(
            {
                "path": child_path,
                "is_symlink": child_is_symlink,
                "realpath": child_realpath,
                "device_inode": child_device_inode,
            }
        )

    # Collect file and directory names separately for LLM scoring context
    # Using terminal names only (not full paths) to save tokens
    # Directories get trailing "/" to distinguish from files
    # Sort by mtime (most recently modified first) to show active content
    def get_mtime(entry: os.DirEntry) -> float:
        try:
            return entry.stat(follow_symlinks=False).st_mtime
        except OSError:
            return 0.0

    files_sorted = sorted(files, key=get_mtime, reverse=True)
    dirs_sorted = sorted(dirs, key=get_mtime, reverse=True)

    file_names = [sanitize_str(f.name) for f in files_sorted[:20]]
    dir_names = [sanitize_str(d.name) + "/" for d in dirs_sorted[:20]]
    # Combine for legacy child_names field (first 30 total with trailing / on dirs)
    child_names = dir_names[:15] + file_names[:15]

    # Detect numeric directories (shot IDs, run numbers) - signal data container
    numeric_dir_count = 0
    for d in dirs:
        name = d.name
        # Consider dir numeric if it's all digits or has numeric prefix/suffix
        if name.isdigit() or (len(name) > 3 and name[:4].isdigit()):
            numeric_dir_count += 1
    numeric_dir_ratio = numeric_dir_count / len(dirs) if dirs else 0.0

    # Tree context for hierarchical view (eza preferred, tree fallback)
    tree_context: Optional[str] = None
    if len(dirs) > 5:
        # Prefer eza --tree (fast tool), fall back to tree if unavailable
        if has_command("eza"):
            try:
                proc = subprocess.run(
                    ["eza", "--tree", "--level", "2", "--only-dirs", path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    lines = proc.stdout.strip().split("\n")
                    # Cap to 25 lines to avoid token explosion
                    if len(lines) > 25:
                        tree_context = (
                            "\n".join(lines[:25]) + f"\n... ({len(lines)} dirs total)"
                        )
                    else:
                        tree_context = proc.stdout.strip()
            except (subprocess.TimeoutExpired, Exception):
                pass
        elif has_command("tree"):
            try:
                # Fallback to tree if eza not available
                proc = subprocess.run(
                    ["tree", "-L", "2", "-d", "--dirsfirst", "--noreport", path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    lines = proc.stdout.strip().split("\n")
                    if len(lines) > 25:
                        tree_context = (
                            "\n".join(lines[:25]) + f"\n... ({len(lines)} dirs total)"
                        )
                    else:
                        tree_context = proc.stdout.strip()
            except (subprocess.TimeoutExpired, Exception):
                pass

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

    # Size calculation (if enabled) using du -sb (reliable byte output)
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
    git_remote_url: Optional[str] = None
    git_head_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_root_commit: Optional[str] = None

    if has_git:
        git_dir = os.path.join(path, ".git")
        if os.path.isdir(git_dir):
            # Common git args to bypass ownership checks (scanning other users' repos)
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
        "numeric_dir_ratio": numeric_dir_ratio,
    }
    result["child_dirs"] = child_dirs
    result["child_names"] = child_names
    if tree_context:
        result["tree_context"] = tree_context
    # Include symlink status and device_inode for the scanned path itself
    result["is_symlink"] = is_symlink
    result["realpath"] = realpath
    result["device_inode"] = device_inode

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

    # Check for tools once
    has_rg = has_command("rg")

    # Scan all paths
    results = [
        scan_directory(p, rg_patterns, enable_rg, enable_size, has_rg) for p in paths
    ]

    # Output JSON (handles all escaping correctly)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
