#!/usr/bin/env python3
"""Extract static/machine-description tree structure.

This script runs on the facility host where MDSplus is available.
It walks a static tree (versioned by machine configuration, not shot)
and extracts the tree structure (paths, node types, tags).

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
    exclude_names: set | None = None,
) -> dict[str, Any]:
    """Extract all nodes from a static tree version.

    Args:
        tree_name: MDSplus tree name (e.g., "static")
        version: Version number (tree is opened with this as shot number)
        exclude_names: Node names to skip

    Returns:
        Dict with nodes list, node_count, tags mapping
    """
    import MDSplus

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

    # Build reverse tag mapping: path -> list of tags
    path_to_tags: dict[str, list[str]] = {}
    for tag_name, tag_path in tags.items():
        path_to_tags.setdefault(tag_path, []).append(tag_name)

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

            # NOTE: node.units is NOT accessed here — it's extremely slow
            # (~1ms per node × 48k nodes = 48s). Units can be fetched
            # selectively during enrichment if needed.

            # Get tags from pre-built reverse mapping (avoids per-node MDSplus lookups)
            node_tags = path_to_tags.get(path, [])

            record: dict[str, Any] = {
                "path": path,
                "name": name,
                "node_type": node_type,
            }

            if node_tags:
                record["tags"] = node_tags

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
    exclude_names = {n.upper() for n in config.get("exclude_names", [])}

    version_data: dict[str, dict] = {}
    for ver in versions:
        result = extract_version(
            tree_name=tree_name,
            version=ver,
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
