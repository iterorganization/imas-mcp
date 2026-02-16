#!/usr/bin/env python3
"""Validate JT-60SA EDAS signals return data for a reference shot.

This script runs on the JT-60SA host where eddb_pwrapper is available.
It calls eddbreadOne() for each signal to verify data exists.

Requirements:
- Python 3.8+ (stdlib only except eddb_pwrapper)
- eddb_pwrapper (available at /analysis/lib)

Usage:
    echo '{"signals": [...], "ref_shot": "E012345"}' | python3 check_edas.py

Input (JSON on stdin):
    {
        "signals": [
            {"id": "jt60sa:general/eddb_testime", "category": "EDDB", "data_name": "tesTime"},
            ...
        ],
        "ref_shot": "E012345"
    }

Output (JSON on stdout):
    {
        "results": [
            {"id": "jt60sa:general/eddb_testime", "success": true, "dtype": "ndarray"},
            {"id": "jt60sa:general/eddb_bad", "success": false, "error": "eddbreadOne returned None"},
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
    ref_shot = config.get("ref_shot", "")

    if not ref_shot:
        print(json.dumps({"results": [
            {"id": s["id"], "success": False, "error": "no ref_shot"}
            for s in signals
        ]}))
        sys.exit(0)

    try:
        sys.path.insert(0, "/analysis/lib")
        from eddb_pwrapper import eddbWrapper
    except ImportError:
        print(json.dumps({"results": [
            {"id": s["id"], "success": False, "error": "eddb_pwrapper not available"}
            for s in signals
        ]}))
        sys.exit(0)

    db = eddbWrapper()
    db.opendb("EDDB")

    results = []
    for sig in signals:
        try:
            data = db._eddbreadOne(ref_shot, sig["category"], sig["data_name"], 1)
            if data is not None:
                results.append({
                    "id": sig["id"],
                    "success": True,
                    "dtype": str(type(data).__name__),
                })
            else:
                results.append({
                    "id": sig["id"],
                    "success": False,
                    "error": "eddbreadOne returned None",
                })
        except Exception as e:
            results.append({
                "id": sig["id"],
                "success": False,
                "error": str(e)[:200],
            })

    db.closedb()
    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
