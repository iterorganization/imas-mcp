#!/usr/bin/env python3
"""Check signals via MDSplus thin-client TDI function access.

Validates that signals are accessible via a remote MDSplus server
using TDI function calls (dpf/jpf/ppf). Used by the MDSplus scanner's
check() method for JET-style thin-client facilities.

Python 3.8+ (stdlib + MDSplus)

Usage:
    echo '{"server": "mdsplus.jet.uk", "signals": [...]}' | python3 check_mdsplus_tdi.py

Input (JSON on stdin):
    {
        "server": "mdsplus.jet.uk",
        "shot": 99896,
        "signals": [
            {"id": "jet:magnetics/da_c2-ipla", "path": "da/c2-ipla", "type": "jpf"},
            {"id": "jet:equilibrium/efit_rbnd", "path": "EFIT/RBND", "type": "ppf"}
        ]
    }

Output (JSON on stdout):
    {
        "results": [
            {"id": "jet:...", "success": true, "shape": [11957], "dtype": "float32"},
            {"id": "jet:...", "success": false, "error": "connection failed"}
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

    server = config.get("server")
    shot = config.get("shot")
    signals = config.get("signals", [])

    if not server or not shot:
        print(json.dumps({"error": "server and shot required"}))
        sys.exit(0)

    try:
        import MDSplus
    except ImportError:
        print(json.dumps({"error": "MDSplus module not available"}))
        sys.exit(0)

    try:
        conn = MDSplus.Connection(server)
    except Exception as e:
        results = [
            {"id": s["id"], "success": False, "error": str(e)[:100]} for s in signals
        ]
        print(json.dumps({"results": results}))
        sys.exit(0)

    results = []
    for sig in signals:
        sig_id = sig.get("id", "")
        path = sig.get("path", "")
        sig_type = sig.get("type", "jpf")
        entry = {"id": sig_id, "success": False}

        try:
            if sig_type == "ppf":
                data = conn.get('ppf("%s", %d)' % (path, shot))
            else:
                data = conn.get('dpf("%s", %d)' % (path, shot))

            if hasattr(data, "shape") and data.size > 0:
                entry["success"] = True
                entry["shape"] = list(data.shape)
                entry["dtype"] = str(data.dtype)
            elif isinstance(data, (int, float)) and data != 0:
                entry["success"] = True
                entry["shape"] = []
                entry["dtype"] = type(data).__name__
        except Exception as e:
            entry["error"] = str(e)[:200]

        results.append(entry)

    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
