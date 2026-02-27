#!/usr/bin/env python3
"""Extract static/machine-description tree structure and values.

This script runs on the facility host where MDSplus is available.
It walks a static tree (versioned by machine configuration, not shot)
and extracts both the tree structure and optionally the numerical values
(R/Z coordinates, Green's functions, etc.).

Static trees differ from shot-dependent trees:
- Opened by version number (shot=1..N), not experimental shot number
- Contain constructional geometry: coil positions, vessel filaments,
  tile contours, magnetic probe positions, flux loop positions, mesh grids
- Data accessed via TDI functions like static("tag")[$shot]

Requirements:
- Python 3.12+ (installed via ``imas-codex tools install``)
- MDSplus Python bindings

Usage:
    echo '{"tree_name": "static", "versions": [1,2,3]}' | python3 extract_static_tree.py

Input (JSON on stdin):
    {
        "tree_name": "static",
        "versions": [1, 2, 3, 4, 5, 6, 7, 8],
        "extract_values": true,
        "exclude_names": ["COMMENTS", "DATE"]
    }

Output (JSON on stdout):
    {
        "versions": {
            "1": {
                "nodes": [...],
                "node_count": 123,
                "tags": {"\\R_C": "\\STATIC::TOP.C:R", ...}
            },
            ...
        },
        "diff": {
            "added": {"2": [paths...], "3": [paths...]},
            "removed": {"3": [paths...]}
        }
    }
"""

import json
import sys
from typing import Any


def extract_version(
    tree_name: str,
    version: int,
    extract_values: bool = False,
    exclude_names: set | None = None,
) -> dict[str, Any]:
    """Extract all nodes and optionally values from a static tree version.

    Args:
        tree_name: MDSplus tree name (e.g., "static")
        version: Version number (tree is opened with this as shot number)
        extract_values: Whether to extract actual numerical data
        exclude_names: Node names to skip

    Returns:
        Dict with nodes list, node_count, tags mapping
    """
    import MDSplus
    import numpy as np

    if exclude_names is None:
        exclude_names = set()

    try:
        tree = MDSplus.Tree(tree_name, version, "readonly")
    except Exception as e:
        return {"error": str(e)[:300], "version": version}

    # Get all nodes
    all_nodes = list(tree.getNodeWild("***"))

    # Build tag mapping: tag -> path
    tags = {}
    try:
        # Walk all tags in the tree
        tag_node = tree.findTags("*")
        for tag in tag_node:
            try:
                tag_name = str(tag)
                tag_path = str(tree.getNode(tag_name).path)
                tags[tag_name] = tag_path
            except Exception:
                pass
    except Exception:
        pass

    nodes = []
    for node in all_nodes:
        try:
            path = str(node.path)
            name = str(node.node_name).strip()

            if name.upper() in exclude_names:
                continue

            usage = str(node.usage)
            usage_map = {
                "STRUCTURE": "STRUCTURE",
                "SIGNAL": "SIGNAL",
                "NUMERIC": "NUMERIC",
                "TEXT": "TEXT",
                "AXIS": "AXIS",
                "SUBTREE": "SUBTREE",
            }
            node_type = usage_map.get(usage, "STRUCTURE")

            # Get units
            try:
                units = str(node.units).strip()
                if not units or units == " ":
                    units = ""
            except Exception:
                units = ""

            # Get tags for this node
            try:
                node_tags = [str(t) for t in node.tags] if hasattr(node, "tags") else []
            except Exception:
                node_tags = []

            # Get description from COMMENT child
            description = ""
            try:
                comment_node = node.getNode(":COMMENT")
                data = comment_node.data()
                if data:
                    description = str(data)[:500]
            except Exception:
                pass

            record: dict[str, Any] = {
                "path": path,
                "name": name,
                "node_type": node_type,
            }

            if units:
                record["units"] = units
            if node_tags:
                record["tags"] = node_tags
            if description:
                record["description"] = description

            # Extract values if requested and node has data
            if extract_values and node_type in ("NUMERIC", "SIGNAL"):
                try:
                    has_data = node.getLength() > 0
                    if has_data:
                        data = node.data()
                        if isinstance(data, np.ndarray):
                            record["shape"] = list(data.shape)
                            record["dtype"] = str(data.dtype)
                            # For small arrays, include actual values
                            if data.size <= 2048:
                                record["value"] = data.tolist()
                            else:
                                # Too large — store summary only
                                record["value_summary"] = {
                                    "min": float(np.nanmin(data)),
                                    "max": float(np.nanmax(data)),
                                    "mean": float(np.nanmean(data)),
                                    "size": int(data.size),
                                }
                        elif isinstance(data, int | float):
                            record["value"] = data
                            record["shape"] = []
                            record["dtype"] = type(data).__name__
                        elif isinstance(data, str):
                            record["value"] = data[:500]
                            record["dtype"] = "str"
                except Exception as e:
                    record["value_error"] = str(e)[:200]

            nodes.append(record)

        except Exception:
            pass

    return {
        "version": version,
        "node_count": len(nodes),
        "nodes": nodes,
        "tags": tags,
    }


def diff_versions(
    version_data: dict[str, dict],
) -> dict[str, dict[str, list[str]]]:
    """Compute structural differences between consecutive versions.

    Returns:
        Dict with "added" and "removed" keys, each mapping version
        number (str) to list of paths added/removed in that version.
    """
    sorted_versions = sorted(version_data.keys(), key=int)
    added: dict[str, list[str]] = {}
    removed: dict[str, list[str]] = {}

    prev_paths: set[str] | None = None
    for ver in sorted_versions:
        data = version_data[ver]
        if "error" in data:
            continue
        current_paths = {n["path"] for n in data.get("nodes", [])}
        if prev_paths is not None:
            new_paths = sorted(current_paths - prev_paths)
            gone_paths = sorted(prev_paths - current_paths)
            if new_paths:
                added[ver] = new_paths
            if gone_paths:
                removed[ver] = gone_paths
        prev_paths = current_paths

    return {"added": added, "removed": removed}


def main() -> None:
    """Read config from stdin, extract static tree versions, output JSON."""
    config = json.loads(sys.stdin.read())

    tree_name = config["tree_name"]
    versions = config.get("versions", [1])
    extract_values = config.get("extract_values", False)
    exclude_names = {n.upper() for n in config.get("exclude_names", [])}

    version_data: dict[str, dict] = {}
    for ver in versions:
        result = extract_version(
            tree_name=tree_name,
            version=ver,
            extract_values=extract_values,
            exclude_names=exclude_names,
        )
        version_data[str(ver)] = result

    # Compute structural diffs
    diff = diff_versions(version_data)

    output = {
        "tree_name": tree_name,
        "versions": version_data,
        "diff": diff,
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
