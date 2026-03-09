#!/usr/bin/env python3
"""Enumerate signals via MDSplus thin-client TDI function access.

Connects to a remote MDSplus server and uses TDI functions to discover
available JPF subsystems, validate data access, and enumerate EFIT
geometry data. Works at JET (mdsplus.jet.uk) where MDSplus uses
TDI function-based access rather than traditional tree traversal.

Python 3.8+ (stdlib + MDSplus)

Usage:
    echo '{"server": "mdsplus.jet.uk", "shot": 99896}' | python3 enumerate_mdsplus_tdi.py

Input (JSON on stdin):
    {
        "server": "mdsplus.jet.uk",
        "shot": 99896,
        "subsystem_codes": ["DA", "DB", ...],
        "sample_signals": [
            {"path": "da/c2-ipla", "type": "jpf"},
            {"path": "EFIT/RBND", "type": "ppf"}
        ]
    }

Output (JSON on stdout):
    {
        "server": "mdsplus.jet.uk",
        "shot": 99896,
        "connected": true,
        "active_subsystems": ["PF", "DA", ...],
        "all_subsystems": ["PL", "SS", ...],
        "signal_checks": [
            {"path": "da/c2-ipla", "type": "jpf", "valid": true, "shape": [11957], "dtype": "float32"},
            ...
        ],
        "geometry": {
            "rlim": {"shape": [1, 251], "r_range": [1.836, 3.891]},
            "zlim": {"shape": [1, 251], "z_range": [-1.746, 1.984]},
            "rbnd": {"shape": [947, 105], "available": true},
            "zbnd": {"shape": [947, 105], "available": true}
        },
        "tdi_function_count": 1448
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
    subsystem_codes = config.get("subsystem_codes", [])
    sample_signals = config.get("sample_signals", [])

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

    # Connect to server
    try:
        conn = MDSplus.Connection(server)
    except Exception as e:
        print(json.dumps({"error": f"Connection failed: {e}"}))
        sys.exit(0)

    result = {
        "server": server,
        "shot": shot,
        "connected": True,
        "all_subsystems": [],
        "active_subsystems": [],
        "signal_checks": [],
        "geometry": {},
        "tdi_function_count": 0,
    }

    # Count TDI functions
    try:
        help_result = conn.get("HELP()")
        if hasattr(help_result, "__len__"):
            result["tdi_function_count"] = len(help_result)
    except Exception:
        pass

    # Enumerate JPF subsystems
    try:
        raw = conn.get("jpfsubsystems()")
        rawstr = (
            raw.decode("utf-8").strip() if isinstance(raw, bytes) else str(raw).strip()
        )
        all_subs = [rawstr[i : i + 2] for i in range(0, len(rawstr), 2)]
        result["all_subsystems"] = all_subs
    except Exception:
        result["all_subsystems"] = subsystem_codes

    # Get active subsystems for this shot
    try:
        raw = conn.get("jpfincludedsubsystems(%d)" % shot)
        rawstr = (
            raw.decode("utf-8").strip() if isinstance(raw, bytes) else str(raw).strip()
        )
        active = [rawstr[i : i + 2] for i in range(0, len(rawstr), 2)]
        result["active_subsystems"] = active
    except Exception:
        pass

    # Check sample signals
    for sig in sample_signals:
        path = sig.get("path", "")
        sig_type = sig.get("type", "jpf")
        check = {"path": path, "type": sig_type, "valid": False}

        try:
            if sig_type == "jpf":
                data = conn.get('dpf("%s", %d)' % (path, shot))
            elif sig_type == "ppf":
                data = conn.get('ppf("%s", %d)' % (path, shot))
            else:
                data = conn.get('dpf("%s", %d)' % (path, shot))

            if hasattr(data, "shape") and data.size > 0:
                check["valid"] = True
                check["shape"] = list(data.shape)
                check["dtype"] = str(data.dtype)
            elif isinstance(data, (int, float)) and data != 0:
                check["valid"] = True
                check["shape"] = []
                check["dtype"] = type(data).__name__
        except Exception as e:
            check["error"] = str(e)[:100]

        result["signal_checks"].append(check)

    # Check EFIT geometry availability
    geometry = {}
    for name, expr in [
        ("rlim", 'ppf("EFIT/RLIM", %d)' % shot),
        ("zlim", 'ppf("EFIT/ZLIM", %d)' % shot),
        ("rbnd", 'ppf("EFIT/RBND", %d)' % shot),
        ("zbnd", 'ppf("EFIT/ZBND", %d)' % shot),
    ]:
        try:
            data = conn.get(expr)
            if hasattr(data, "shape") and data.size > 1:
                entry = {
                    "shape": list(data.shape),
                    "available": True,
                }
                if name in ("rlim", "rbnd"):
                    entry["r_range"] = [
                        round(float(data.min()), 3),
                        round(float(data.max()), 3),
                    ]
                elif name in ("zlim", "zbnd"):
                    entry["z_range"] = [
                        round(float(data.min()), 3),
                        round(float(data.max()), 3),
                    ]
                geometry[name] = entry
            else:
                geometry[name] = {"available": False}
        except Exception as e:
            geometry[name] = {"available": False, "error": str(e)[:100]}

    result["geometry"] = geometry

    print(json.dumps(result))


if __name__ == "__main__":
    main()
