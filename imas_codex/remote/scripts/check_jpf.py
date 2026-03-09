#!/usr/bin/env python3
"""Validate JET JPF signals return data for a reference pulse.

This script runs on the JET host where MDSplus thin-client access is available.

Requirements:
- Python 3.8+ (stdlib only except MDSplus)
- MDSplus Python bindings available on JET via module load jet/1.0

Usage:
    echo '{"signals": [...], "server": "mdsplus.jet.uk", "shot": 99896}' | python3 check_jpf.py

Input (JSON on stdin):
    {
        "signals": [
            {"id": "jet:magnetics/da_c2-ipla", "path": "da/c2-ipla"},
            ...
        ],
        "server": "mdsplus.jet.uk",
        "shot": 99896
    }

Output (JSON on stdout):
    {
        "results": [
            {"id": "jet:magnetics/da_c2-ipla", "success": true, "shape": [8192], "dtype": "float32"},
            {"id": "jet:other/invalid", "success": false, "error": "No data"},
            ...
        ]
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

    signals = config.get("signals", [])
    server = config.get("server")
    shot = config.get("shot")

    if not server or not shot:
        print(json.dumps({"error": "Missing server or shot"}))
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
    for sig in signals:
        sig_id = sig.get("id", "")
        path = sig.get("path", "")

        if not path:
            results.append({"id": sig_id, "success": False, "error": "empty path"})
            continue

        try:
            data = conn.get(f'dpf("{path}", {shot})')
            if hasattr(data, "data"):
                data = data.data()

            if data is None:
                results.append({"id": sig_id, "success": False, "error": "No data"})
                continue

            shape = list(data.shape) if hasattr(data, "shape") else [len(data)]
            dtype = str(data.dtype) if hasattr(data, "dtype") else type(data).__name__

            results.append(
                {
                    "id": sig_id,
                    "success": True,
                    "shape": shape,
                    "dtype": dtype,
                }
            )
        except Exception as e:
            results.append(
                {
                    "id": sig_id,
                    "success": False,
                    "error": str(e)[:200],
                }
            )

    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
