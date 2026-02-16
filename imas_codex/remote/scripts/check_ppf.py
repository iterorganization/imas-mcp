#!/usr/bin/env python3
"""Validate JET PPF signals return data for a reference pulse.

This script runs on the JET host where libppf.so and ppf.py are available.
It calls ppfdata() for each signal in the batch to verify data exists.

Requirements:
- Python 3.8+ (stdlib only except ppf)
- ppf Python bindings

Usage:
    echo '{"signals": [...], "pulse": 99999, "owner": "jetppf"}' | python3 check_ppf.py

Input (JSON on stdin):
    {
        "signals": [
            {"id": "jet:equilibrium/efit_bvac", "dda": "EFIT", "dtype": "BVAC"},
            ...
        ],
        "pulse": 99999,
        "owner": "jetppf"
    }

Output (JSON on stdout):
    {
        "results": [
            {"id": "jet:equilibrium/efit_bvac", "success": true, "shape": [1000], "dtype": "float64"},
            {"id": "jet:magnetics/magn_ipla", "success": false, "error": "ppfdata ier=1"},
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
    pulse = config.get("pulse")
    owner = config.get("owner", "jetppf")

    if not pulse:
        print(
            json.dumps(
                {
                    "results": [
                        {"id": s["id"], "success": False, "error": "no pulse"}
                        for s in signals
                    ]
                }
            )
        )
        sys.exit(0)

    try:
        sys.path.insert(0, "/jet/share/lib/python")
        import ppf
    except ImportError:
        try:
            sys.path.insert(0, "/jet/share/DEPOT/pyppf/21260/lib/python")
            import ppf
        except ImportError:
            print(
                json.dumps(
                    {
                        "results": [
                            {
                                "id": s["id"],
                                "success": False,
                                "error": "ppf module not available",
                            }
                            for s in signals
                        ]
                    }
                )
            )
            sys.exit(0)

    ppf.ppfuid(owner, rw="R")
    ier = ppf.ppfgo(pulse, seq=0)
    if ier != 0:
        print(
            json.dumps(
                {
                    "results": [
                        {
                            "id": s["id"],
                            "success": False,
                            "error": f"ppfgo failed: ier={ier}",
                        }
                        for s in signals
                    ]
                }
            )
        )
        sys.exit(0)

    results = []
    for sig in signals:
        try:
            data, x, t, ier = ppf.ppfdata(pulse, sig["dda"], sig["dtype"], uid=owner)
            if ier == 0 and data is not None:
                results.append(
                    {
                        "id": sig["id"],
                        "success": True,
                        "shape": list(data.shape)
                        if hasattr(data, "shape")
                        else [len(data)],
                        "dtype": str(data.dtype)
                        if hasattr(data, "dtype")
                        else "unknown",
                    }
                )
            else:
                results.append(
                    {
                        "id": sig["id"],
                        "success": False,
                        "error": f"ppfdata ier={ier}",
                    }
                )
        except Exception as e:
            results.append(
                {
                    "id": sig["id"],
                    "success": False,
                    "error": str(e)[:200],
                }
            )

    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
