#!/usr/bin/env python3
"""Remote file enrichment script.

Executed on remote facilities via SSH. Runs rg pattern matching on
individual files (non-recursive) and counts lines. Much lighter than
directory enrichment since each file is a single target.

Requirements:
- Python 3.8+ (stdlib only)
- rg (ripgrep) for pattern matching
- wc for line counting

Usage:
    echo '{"files": ["/path/a.py", "/path/b.py"], "pattern_categories": {...}}' | python3 enrich_files.py

Input (JSON on stdin):
    {
        "files": ["/path/to/file.py", ...],
        "pattern_categories": {
            "mdsplus": "MDSplus|mdsplus|Tree\\(",
            "imas_read": "get_ids|imas\\.open|DBEntry",
            ...
        }
    }

Output (JSON on stdout):
    [
        {
            "path": "/path/to/file.py",
            "pattern_categories": {"mdsplus": 5, "imas_read": 0},
            "total_pattern_matches": 5,
            "line_count": 142
        },
        ...
    ]
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, List


def count_lines(path: str) -> int:
    """Count lines in a file using wc -l."""
    try:
        result = subprocess.run(
            ["wc", "-l", path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split()[0])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return 0


def run_rg_on_file(pattern: str, path: str) -> int:
    """Run rg -c on a single file and return match count."""
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


def enrich_file(path: str, pattern_categories: Dict[str, str]) -> Dict[str, Any]:
    """Enrich a single file with pattern matching and line count."""
    result = {
        "path": path,
        "pattern_categories": {},
        "total_pattern_matches": 0,
        "line_count": 0,
    }

    if not os.path.isfile(path):
        result["error"] = "file_not_found"
        return result

    # Line count
    result["line_count"] = count_lines(path)

    # Pattern matching — batch all categories via single rg call where possible
    # For accuracy, run per-category to get per-category counts
    has_rg = subprocess.run(["which", "rg"], capture_output=True).returncode == 0

    if has_rg and pattern_categories:
        total = 0
        for category, pattern in pattern_categories.items():
            count = run_rg_on_file(pattern, path)
            if count > 0:
                result["pattern_categories"][category] = count
                total += count
        result["total_pattern_matches"] = total

    return result


def enrich_files_batch(
    files: List[str], pattern_categories: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Enrich a batch of files."""
    results = []
    for path in files:
        try:
            r = enrich_file(path, pattern_categories)
            results.append(r)
        except Exception as e:
            results.append(
                {
                    "path": path,
                    "pattern_categories": {},
                    "total_pattern_matches": 0,
                    "line_count": 0,
                    "error": str(e)[:200],
                }
            )
    return results


def main():
    """Read input from stdin, enrich files, write JSON to stdout."""
    input_data = json.loads(sys.stdin.read())
    files = input_data.get("files", [])
    pattern_categories = input_data.get("pattern_categories", {})

    results = enrich_files_batch(files, pattern_categories)
    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
