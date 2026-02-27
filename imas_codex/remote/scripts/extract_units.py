#!/usr/bin/env python3
"""Extract units from MDSplus tree nodes in batches.

Runs on the facility host where MDSplus is available.
Extracts ``node.units`` for a batch of nodes specified by offset/limit,
returning a path→units mapping. Designed to be called multiple times
with different offsets for progress tracking and fault tolerance.

Requirements:
- Python 3.12+ (installed via ``imas-codex tools install``)
- MDSplus Python bindings

Usage:
    echo '{"tree_name": "static", "version": 8}' | python3 extract_units.py
    echo '{"tree_name": "static", "version": 8, "offset": 5000, "limit": 5000}' | python3 extract_units.py

Input (JSON on stdin):
    {
        "tree_name": "static",
        "version": 8,
        "node_types": ["NUMERIC", "SIGNAL"],
        "offset": 0,
        "limit": 0
    }

    offset: Starting index into the filtered node list (default: 0)
    limit:  Max nodes to process in this batch (default: 0 = all)

Output (JSON on stdout):
    {
        "version": 8,
        "units": {
            "\\\\STATIC::TOP.C.R": "m",
            "\\\\STATIC::TOP.C.Z": "m"
        },
        "count": 2,
        "total_nodes": 34082,
        "batch_checked": 5000,
        "empty": 3200
    }
"""

import json
import sys
from typing import Any


def extract_units(
    tree_name: str,
    version: int,
    node_types: list[str] | None = None,
    offset: int = 0,
    limit: int = 0,
) -> dict[str, Any]:
    """Extract units for a batch of data-bearing nodes.

    Args:
        tree_name: MDSplus tree name (e.g., "static")
        version: Version number (tree opened with this as shot number)
        node_types: Node usage types to extract units for, in deterministic
            order. Must be consistent across batches for correct offset/limit
            slicing. Defaults to ["NUMERIC", "SIGNAL"].
        offset: Starting index into the filtered node list.
        limit: Max nodes to process (0 = all remaining from offset).

    Returns:
        Dict with units mapping, counts, and total node count.
    """
    import MDSplus

    if node_types is None:
        node_types = ["NUMERIC", "SIGNAL"]

    try:
        tree = MDSplus.Tree(tree_name, version, "readonly")
    except Exception as e:
        return {"error": str(e)[:300], "version": version}

    # Get data-bearing nodes by usage type directly from MDSplus.
    target_nodes = []
    for usage_type in node_types:
        try:
            nodes = list(tree.getNodeWild("***", usage_type))
            target_nodes.extend(nodes)
        except Exception:
            pass

    total_nodes = len(target_nodes)

    # Slice to requested batch
    end = offset + limit if limit > 0 else total_nodes
    batch = target_nodes[offset:end]

    units: dict[str, str] = {}
    empty_count = 0

    for node in batch:
        try:
            path = str(node.path)
            unit_str = str(node.units).strip()

            if unit_str and unit_str.lower() not in ("", "none", "unknown", " "):
                units[path] = unit_str
            else:
                empty_count += 1
        except Exception:
            pass

    return {
        "version": version,
        "units": units,
        "count": len(units),
        "total_nodes": total_nodes,
        "batch_checked": len(batch),
        "empty": empty_count,
    }


def main() -> None:
    """Read config from stdin, extract units, output JSON."""
    import os

    # Read stdin BEFORE redirecting fds
    config = json.loads(sys.stdin.read())

    tree_name = config["tree_name"]
    version = config.get("version", 1)
    node_types = sorted(config.get("node_types", ["NUMERIC", "SIGNAL"]))
    offset = config.get("offset", 0)
    limit = config.get("limit", 0)

    # Suppress MDSplus C library warnings that go to stdout/stderr at fd level.
    # MDSplus libvaccess.so prints directly to file descriptor 1, bypassing
    # Python's sys.stdout. node.units evaluates TDI expressions which trigger
    # these C library outputs. Redirect fd 1 and 2 to /dev/null during
    # tree operations, then restore fd 1 only for our JSON output.
    saved_stdout_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    result = extract_units(
        tree_name=tree_name,
        version=version,
        node_types=node_types,
        offset=offset,
        limit=limit,
    )

    # Restore stdout fd for JSON output
    os.dup2(saved_stdout_fd, 1)
    os.close(saved_stdout_fd)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
