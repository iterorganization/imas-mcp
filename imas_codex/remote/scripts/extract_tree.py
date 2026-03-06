#!/usr/bin/env python3
"""Extract MDSplus tree structure.

This script runs on the facility host where MDSplus is available.
It walks an MDSplus tree and extracts the full structure (paths, node
types, tags). Works for any tree type:

- **Versioned trees** (e.g., static/machine-description): opened by
  version number, contain constructional geometry.
- **Shot-scoped trees** (e.g., results, magnetics): opened by shot
  number, contain experimental data.

The ``shots`` parameter is the unified entry point — it accepts version
numbers for versioned trees or experimental shot numbers for dynamic
trees. MDSplus ``Tree(name, shot)`` works identically for both.

Requirements:
- Python 3.12+ (installed via ``imas-codex tools install``)
- MDSplus Python bindings

Usage:
    echo '{"data_source_name": "static", "shots": [1,2,3]}' | python3 extract_tree.py
    echo '{"data_source_name": "results", "shots": [85000]}' | python3 extract_tree.py

Input (JSON on stdin):
    {
        "data_source_name": "static",
        "shots": [1, 2, 3, 4, 5, 6, 7, 8],
        "exclude_names": ["COMMENTS", "DATE"],
        "node_usages": null
    }

    - ``shots``: list of shot/version numbers to extract (also accepts
      legacy ``versions`` key for backwards compatibility)
    - ``exclude_names``: node names to skip (optional)
    - ``node_usages``: if set, only include nodes with these usage types
      (e.g., ["SIGNAL", "NUMERIC"]). If null/absent, all nodes are included.

Output (JSON on stdout):
    {
        "data_source_name": "static",
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
    data_source_name: str,
    version: int,
    exclude_names: set | None = None,
    node_usages: set | None = None,
) -> dict[str, Any]:
    """Extract all nodes from a tree at a given shot/version number.

    Args:
        data_source_name: MDSplus tree name (e.g., "static", "results")
        version: Shot or version number (Tree is opened with this)
        exclude_names: Node names to skip
        node_usages: If set, only include nodes with these usage types.
            If None, all nodes are included (full structure extraction).

    Returns:
        Dict with nodes list, node_count, tags mapping
    """
    import MDSplus

    if exclude_names is None:
        exclude_names = set()

    try:
        tree = MDSplus.Tree(data_source_name, version, "readonly")
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

            # Filter by node usage if requested
            if node_usages and node_type not in node_usages:
                continue

            # NOTE: node.units is NOT accessed here — it's extremely slow
            # (~1ms per node × 48k nodes = 48s). Units can be fetched
            # selectively during enrichment if needed.

            # Get tags from pre-built reverse mapping (avoids per-node MDSplus lookups)
            node_tags = path_to_tags.get(path, [])

            # fullpath preserves the structural hierarchy (uses . and : separators)
            # while path may return a flat tag alias (e.g. \STATIC::DBRDR_A_A)
            # that loses parent-child context.
            fullpath = str(node.fullpath)

            record: dict[str, Any] = {
                "path": path,
                "name": name,
                "node_type": node_type,
            }

            if fullpath != path:
                record["fullpath"] = fullpath

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
    """Read config from stdin, extract tree structure, output JSON."""
    config = json.loads(sys.stdin.read())

    data_source_name = config["data_source_name"]
    # Accept "shots" (preferred) or "versions" (legacy compat)
    shots = config.get("shots") or config.get("versions", [1])
    exclude_names = {n.upper() for n in config.get("exclude_names", [])}
    # Optional: filter by node usage types (None = all nodes)
    raw_usages = config.get("node_usages")
    node_usages = {u.upper() for u in raw_usages} if raw_usages else None

    version_data: dict[str, dict] = {}
    for shot in shots:
        result = extract_version(
            data_source_name=data_source_name,
            version=shot,
            exclude_names=exclude_names,
            node_usages=node_usages,
        )
        version_data[str(shot)] = result

    # Compute structural diffs
    diff = diff_versions(version_data)

    output = {
        "data_source_name": data_source_name,
        "versions": version_data,
        "diff": diff,
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
