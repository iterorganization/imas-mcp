#!/usr/bin/env python3
"""Enumerate JET JPF (JET Processing Facility) subsystems and signals.

This script runs on the JET host where MDSplus thin-client access is available.
It connects to the MDSplus server and uses TDI functions to enumerate JPF
subsystem signals.

Requirements:
- Python 3.8+ (stdlib only except MDSplus)
- MDSplus Python bindings available on JET via module load jet/1.0

Usage:
    echo '{"server": "mdsplus.jet.uk", "shot": 99896, "subsystems": ["DA","DB"]}' | python3 enumerate_jpf.py

Input (JSON on stdin):
    {
        "server": "mdsplus.jet.uk",
        "shot": 99896,
        "subsystems": ["DA", "DB", "PF", ...],
        "max_signals_per_subsystem": 200
    }

Output (JSON on stdout):
    {
        "signals": [
            {"subsystem": "DA", "signal": "C2-IPLA", "path": "da/c2-ipla"},
            ...
        ],
        "shot": 99896,
        "subsystems_scanned": 5,
        "subsystems_failed": 1,
        "errors": ["PF: connection timeout"]
    }
"""

import json
import sys


def main():
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(0)

    server = config.get("server")
    shot = config.get("shot")
    subsystems = config.get("subsystems", [])
    max_per_sub = config.get("max_signals_per_subsystem", 200)

    if not server:
        print(json.dumps({"error": "No server specified"}))
        sys.exit(0)
    if not shot:
        print(json.dumps({"error": "No shot specified"}))
        sys.exit(0)

    try:
        import MDSplus
    except ImportError:
        print(json.dumps({"error": "MDSplus module not available"}))
        sys.exit(0)

    try:
        conn = MDSplus.Connection(server)
    except Exception as e:
        print(json.dumps({"error": f"Connection failed: {e}"}))
        sys.exit(0)

    results = []
    errors = []
    subsystems_scanned = 0

    for subsystem in subsystems:
        sub_lower = subsystem.lower()
        try:
            # Use jpfnodes() TDI function to enumerate nodes in this subsystem
            # jpfnodes(subsystem) returns an array of signal node names
            try:
                nodes_raw = conn.get(f'jpfnodes("{subsystem}")')
                if hasattr(nodes_raw, "data"):
                    nodes_raw = nodes_raw.data()
                if hasattr(nodes_raw, "tolist"):
                    nodes_raw = nodes_raw.tolist()
            except Exception:
                # Fallback: try listing via dpf pattern matching
                # Some subsystems may not support jpfnodes
                try:
                    nodes_raw = conn.get(f'jpfnodenames("{subsystem}/*")')
                    if hasattr(nodes_raw, "data"):
                        nodes_raw = nodes_raw.data()
                    if hasattr(nodes_raw, "tolist"):
                        nodes_raw = nodes_raw.tolist()
                except Exception:
                    # Last resort: try reading the subsystem node list
                    errors.append(f"{subsystem}: enumeration not supported")
                    continue

            if not nodes_raw:
                continue

            # Decode bytes to strings if needed
            signal_names = []
            if isinstance(nodes_raw, (list, tuple)):
                for node in nodes_raw:
                    if isinstance(node, bytes):
                        node = node.decode("utf-8", errors="replace")
                    name = str(node).strip()
                    if name:
                        signal_names.append(name)
            elif isinstance(nodes_raw, bytes):
                name = nodes_raw.decode("utf-8", errors="replace").strip()
                if name:
                    signal_names.append(name)
            elif isinstance(nodes_raw, str):
                signal_names.append(nodes_raw.strip())

            # Limit per subsystem
            for name in signal_names[:max_per_sub]:
                path = f"{sub_lower}/{name.lower()}"
                results.append(
                    {
                        "subsystem": subsystem,
                        "signal": name,
                        "path": path,
                    }
                )

            subsystems_scanned += 1

        except Exception as e:
            errors.append(f"{subsystem}: {str(e)[:200]}")

    # If jpfnodes didn't work for any subsystem, try alternative approach:
    # validate the subsystems by testing a known sample signal per subsystem
    if subsystems_scanned == 0 and not results:
        for subsystem in subsystems:
            sub_lower = subsystem.lower()
            try:
                # Try to get active subsystem signals via jpfincludedsubsystems
                active_subs = conn.get(f"jpfincludedsubsystems({shot})")
                if hasattr(active_subs, "data"):
                    active_subs = active_subs.data()
                if hasattr(active_subs, "tolist"):
                    active_subs = active_subs.tolist()

                if isinstance(active_subs, (list, tuple)):
                    for sub in active_subs:
                        if isinstance(sub, bytes):
                            sub = sub.decode("utf-8", errors="replace")
                        sub = str(sub).strip()
                        if sub:
                            results.append(
                                {
                                    "subsystem": sub,
                                    "signal": "*",
                                    "path": f"{sub.lower()}/",
                                    "enumeration_pending": True,
                                }
                            )
                    subsystems_scanned = len(results)
                break
            except Exception as e:
                errors.append(f"jpfincludedsubsystems: {str(e)[:200]}")
                break

    print(
        json.dumps(
            {
                "signals": results,
                "shot": shot,
                "subsystems_scanned": subsystems_scanned,
                "subsystems_failed": len(errors),
                "errors": errors,
            }
        )
    )


if __name__ == "__main__":
    main()
