#!/usr/bin/env python3
"""Enumerate MDSplus tree nodes for signal discovery.

This script runs on the facility host where MDSplus is available.
It walks specified subtrees and extracts data-bearing nodes (SIGNAL,
NUMERIC, AXIS) with metadata for signal discovery.

Filters:
- Only SIGNAL usage nodes (physics measurements, not config/numeric params)
- Must have data for the reference shot (getLength() > 0)
- Excludes metadata nodes (COMMENTS, DATE, ROUTINE, etc.)
- Deduplicates array channels (BPOL_003..BPOL_255 → single BPOL entry)

Requirements:
- Python 3.8+ (stdlib only except MDSplus)
- MDSplus Python bindings

Usage:
    echo '{"trees": ["results", "magnetics"], "shot": 85000}' | python3 enumerate_mdsplus.py

Input (JSON on stdin):
    {
        "trees": ["results", "magnetics", "diagz"],
        "shot": 85000,
        "exclude_names": ["COMMENTS", "DATE", "ROUTINE"],
        "max_nodes_per_tree": 5000
    }

Output (JSON on stdout):
    {
        "signals": [
            {
                "tree": "results",
                "path": "\\\\RESULTS::ASTRA:TE",
                "name": "TE",
                "group": "ASTRA",
                "usage": "SIGNAL",
                "units": "eV",
                "has_data": true,
                "depth": 2
            },
            ...
        ],
        "tree_stats": {
            "results": {"total_nodes": 11463, "data_nodes": 8198, "signals_found": 1548},
            "magnetics": {"total_nodes": 514, "data_nodes": 479, "signals_found": 215}
        }
    }
"""

import json
import re
import sys
from collections import defaultdict
from typing import Any


def enumerate_tree(
    tree_name: str,
    shot: int,
    exclude_names: set[str],
    max_nodes: int = 5000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Walk an MDSplus tree and extract physics-relevant signal nodes.

    Strategy:
    - Walk all nodes with *** wildcard
    - Keep only SIGNAL usage nodes (the physics-meaningful data)
    - Filter out metadata/config nodes by name
    - Deduplicate numbered array channels into single entries
    - Record group (parent structure) for context

    Args:
        tree_name: MDSplus tree name (e.g., "results")
        shot: Reference shot number
        exclude_names: Set of node names to skip
        max_nodes: Safety limit per tree

    Returns:
        (signals list, stats dict)
    """
    import MDSplus

    stats = {"total_nodes": 0, "data_nodes": 0, "signals_found": 0, "skipped": 0}

    try:
        tree = MDSplus.Tree(tree_name, int(shot), "readonly")
    except Exception as e:
        return [], {"error": str(e)[:200]}

    nodes = list(tree.getNodeWild("***"))
    stats["total_nodes"] = len(nodes)

    # First pass: collect SIGNAL nodes with data
    raw_signals: list[dict[str, Any]] = []
    for node in nodes:
        usage = str(node.usage)
        if usage not in ("SIGNAL",):
            # Only SIGNAL usage - NUMERIC is mostly config params, AXIS is
            # typically time bases. We want physics measurement signals.
            continue

        stats["data_nodes"] += 1
        name = str(node.node_name).strip()

        if name.upper() in exclude_names:
            stats["skipped"] += 1
            continue

        # Check if node has data
        try:
            has_data = node.getLength() > 0
        except Exception:
            has_data = False

        if not has_data:
            stats["skipped"] += 1
            continue

        path = str(node.path)

        # Extract units
        try:
            units = str(node.units).strip() if hasattr(node, "units") else ""
        except Exception:
            units = ""

        # Extract group (parent structure name)
        # Path like \RESULTS::ASTRA:TE -> group = ASTRA
        # Path like \MAGNETICS::BPOL_003 -> group = TOP
        parts = path.split("::")
        if len(parts) >= 2:
            after_tree = parts[-1]
            segments = re.split(r"[.:]", after_tree)
            # Group is the first meaningful segment after TOP
            group = "TOP"
            for seg in segments:
                if seg and seg != "TOP" and seg != name:
                    group = seg
                    break
        else:
            group = "TOP"

        # Calculate depth
        depth = path.count(":") + path.count(".") - 1  # -1 for the :: separator

        raw_signals.append(
            {
                "tree": tree_name,
                "path": path,
                "name": name,
                "group": group,
                "usage": usage,
                "units": units,
                "has_data": has_data,
                "depth": depth,
            }
        )

    # Second pass: deduplicate numbered array channels
    # E.g., BPOL_003, BPOL_011, BPOL_255 -> single BPOL entry with channel_count
    channel_pattern = re.compile(r"^(.+?)_(\d{2,})$")
    grouped: dict[str, list[dict]] = defaultdict(list)
    standalone: list[dict[str, Any]] = []

    for sig in raw_signals:
        match = channel_pattern.match(sig["name"])
        if match:
            base_name = match.group(1)
            key = f"{sig['tree']}:{sig['group']}:{base_name}"
            grouped[key].append(sig)
        else:
            standalone.append(sig)

    signals: list[dict[str, Any]] = list(standalone)

    # For channel groups, emit a single representative entry
    for _key, channel_sigs in grouped.items():
        rep = channel_sigs[0].copy()
        base_match = channel_pattern.match(rep["name"])
        if base_match:
            rep["name"] = base_match.group(1)
        rep["channel_count"] = len(channel_sigs)
        rep["channel_paths"] = [s["path"] for s in channel_sigs]
        signals.append(rep)

    stats["signals_found"] = len(signals)

    # Apply safety limit
    if len(signals) > max_nodes:
        signals = signals[:max_nodes]
        stats["truncated"] = True

    return signals, stats


def main() -> None:
    """Read config from stdin, enumerate trees, output JSON."""
    import os

    # Read stdin BEFORE redirecting fds
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}", "signals": []}))
        sys.exit(0)

    trees = config.get("trees", [])
    shot = config.get("shot")
    exclude_names = {n.upper() for n in config.get("exclude_names", [])}
    max_nodes = config.get("max_nodes_per_tree", 5000)

    if not trees or not shot:
        print(
            json.dumps(
                {
                    "error": "Missing required fields: trees, shot",
                    "signals": [],
                }
            )
        )
        sys.exit(0)

    # Suppress MDSplus C library warnings that go to stdout/stderr at fd level.
    # MDSplus libvaccess.so prints "Error loading libvaccess.so" directly to
    # file descriptor 1 (stdout), bypassing Python's sys.stdout. This happens
    # lazily during Tree() constructor calls, not just at import time.
    # We redirect fd 1 and 2 to /dev/null for the entire tree enumeration,
    # then restore fd 1 only for our JSON output.
    saved_stdout_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    all_signals: list[dict] = []
    tree_stats: dict[str, Any] = {}

    for tree_name in trees:
        signals, stats = enumerate_tree(tree_name, shot, exclude_names, max_nodes)
        all_signals.extend(signals)
        tree_stats[tree_name] = stats

    # Restore stdout fd for JSON output (was suppressed during tree ops)
    os.dup2(saved_stdout_fd, 1)
    os.close(saved_stdout_fd)

    print(
        json.dumps(
            {
                "signals": all_signals,
                "tree_stats": tree_stats,
                "shot": shot,
                "trees_scanned": len(trees),
            }
        )
    )


if __name__ == "__main__":
    main()
