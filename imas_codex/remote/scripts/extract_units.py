#!/usr/bin/env python3
"""Extract units from MDSplus tree nodes in batches.

Runs on the facility host where MDSplus is available.
Reads raw records via ``node.getRecord()`` and extracts units only from
value nodes (stored data wrapped in ``WithUnits``). Expression nodes
(``EXT_FUNCTION``, ``DIVIDE``, etc.) are skipped entirely to avoid
triggering expensive TDI evaluation (e.g. Green's function computation).

Requirements:
- Python 3.12+ (installed via ``imas-codex tools install``)
- MDSplus Python bindings

Usage:
    echo '{"tree_name": "static", "version": 8}' | python3 extract_units.py
    echo '{"tree_name": "static", "version": 8, "offset": 500, "limit": 500}' | python3 extract_units.py

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
        "batch_checked": 500,
        "empty": 200,
        "skipped_expressions": 150
    }
"""

import json
import sys
from typing import Any


def _extract_units_from_record(record) -> str | None:
    """Extract units string from an MDSplus record without evaluating data.

    Checks for WithUnits compounds (direct or nested inside Parameter).
    Returns the units string if found, None otherwise.
    """
    import MDSplus

    if record is None:
        return None

    # Direct WithUnits: BUILD_WITH_UNITS(value, units)
    if isinstance(record, MDSplus.compound.WithUnits):
        units = record.getUnits()
        if units is not None:
            unit_str = str(units).strip()
            if unit_str and unit_str.lower() not in ("", "none", "unknown", " "):
                return unit_str
        return None

    # Parameter: BUILD_PARAM(value, help, validation)
    # The value inside may be wrapped in WithUnits
    if isinstance(record, MDSplus.compound.Parameter):
        try:
            val = record.getValue()
            if isinstance(val, MDSplus.compound.WithUnits):
                units = val.getUnits()
                if units is not None:
                    unit_str = str(units).strip()
                    if unit_str and unit_str.lower() not in (
                        "",
                        "none",
                        "unknown",
                        " ",
                    ):
                        return unit_str
        except Exception:
            pass
        return None

    # Signal: BUILD_SIGNAL(value, raw, dimension)
    # The value or raw inside may be wrapped in WithUnits
    if isinstance(record, MDSplus.compound.Signal):
        try:
            val = record.getValue()
            if isinstance(val, MDSplus.compound.WithUnits):
                units = val.getUnits()
                if units is not None:
                    unit_str = str(units).strip()
                    if unit_str and unit_str.lower() not in (
                        "",
                        "none",
                        "unknown",
                        " ",
                    ):
                        return unit_str
        except Exception:
            pass
        return None

    # Stored value types (Array, Scalar) — no units wrapper
    return None


def extract_units(
    tree_name: str,
    version: int,
    node_types: list[str] | None = None,
    offset: int = 0,
    limit: int = 0,
) -> dict[str, Any]:
    """Extract units from value nodes only, skipping expression nodes.

    Uses ``node.getRecord()`` to read the raw record descriptor without
    evaluating TDI expressions. Only extracts units from nodes whose
    record is (or contains) a ``WithUnits`` compound.

    Expression nodes (EXT_FUNCTION, DIVIDE, ADD, etc.) are skipped
    entirely — they would trigger expensive computation (e.g.
    Green's function evaluation in static trees) via ``node.units``.

    Args:
        tree_name: MDSplus tree name (e.g., "static")
        version: Version number (tree opened with this as shot number)
        node_types: Node usage types to filter. Defaults to
            ["NUMERIC", "SIGNAL"].
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
    skipped_expressions = 0

    # Expression types that trigger expensive TDI evaluation
    _EXPRESSION_TYPES = (
        MDSplus.compound.Function,  # Base class for EXT_FUNCTION, etc.
    )
    # Also check by class name for types that may not share a common base
    _EXPRESSION_NAMES = frozenset(
        {
            "EXT_FUNCTION",
            "DIVIDE",
            "ADD",
            "SUBTRACT",
            "MULTIPLY",
            "CONCAT",
            "TRANSLATE",
            "VECTOR",
            "DATA",
        }
    )

    for node in batch:
        try:
            record = node.getRecord()
            if record is None:
                empty_count += 1
                continue

            # Skip expression nodes — they evaluate TDI on access
            rec_name = type(record).__name__
            if isinstance(record, _EXPRESSION_TYPES) or rec_name in _EXPRESSION_NAMES:
                skipped_expressions += 1
                continue

            unit_str = _extract_units_from_record(record)
            if unit_str:
                path = str(node.path)
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
        "skipped_expressions": skipped_expressions,
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
