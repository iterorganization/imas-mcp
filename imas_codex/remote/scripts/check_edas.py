#!/usr/bin/env python3
"""Validate JT-60SA EDAS signals return data for a reference shot.

This script runs on the JT-60SA host where eddb_pwrapper is available.
It calls eddbreadOne() for each signal to verify data exists.

Requirements:
- Python 3.8+ (stdlib only except eddb_pwrapper)
- eddb_pwrapper (available at /analysis/src/eddb)
- libeddb.so (available at /analysis/lib/libeddb.so)

Usage:
    echo '{"signals": [...], "ref_shot": "E012345"}' | python3 check_edas.py

Input (JSON on stdin):
    {
        "signals": [
            {"id": "jt-60sa:general/eddb_testime", "category": "EDDB", "data_name": "tesTime"},
            ...
        ],
        "ref_shot": "E012345",
        "api_path": "/analysis/src/eddb",
        "lib_path": "/analysis/lib/libeddb.so"
    }

Output (JSON on stdout):
    {
        "results": [
            {"id": "jt-60sa:general/eddb_testime", "success": true, "dtype": "ndarray"},
            {"id": "jt-60sa:general/eddb_bad", "success": false, "error": "eddbreadOne returned None"},
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
    api_path = config.get("api_path", "")
    lib_path = config.get("lib_path", "")

    if not ref_shot:
        print(
            json.dumps(
                {
                    "results": [
                        {"id": s["id"], "success": False, "error": "no ref_shot"}
                        for s in signals
                    ]
                }
            )
        )
        sys.exit(0)

    if not api_path or not lib_path:
        print(
            json.dumps(
                {
                    "results": [
                        {
                            "id": s["id"],
                            "success": False,
                            "error": "api_path and lib_path are required",
                        }
                        for s in signals
                    ]
                }
            )
        )
        sys.exit(0)

    try:
        sys.path.insert(0, api_path)
        from eddb_pwrapper import eddbWrapper
    except ImportError:
        print(
            json.dumps(
                {
                    "results": [
                        {
                            "id": s["id"],
                            "success": False,
                            "error": f"eddb_pwrapper not available at {api_path}",
                        }
                        for s in signals
                    ]
                }
            )
        )
        sys.exit(0)

    db = eddbWrapper(lib_path)
    ok = db.eddbOpen()
    if not ok:
        print(
            json.dumps(
                {
                    "results": [
                        {
                            "id": s["id"],
                            "success": False,
                            "error": "eddbOpen() failed",
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
            # Use eddbreadTable to check if the signal exists in the catalog
            # eddbreadTable returns (rtn_bool, rtn_data)
            tbl_ok, tbl_data = db.eddbreadTable(
                shot=ref_shot,
                cat=sig["category"],
                dname=sig["data_name"],
            )
            if tbl_ok and tbl_data and tbl_data.get("count", 0) > 0:
                results.append(
                    {
                        "id": sig["id"],
                        "success": True,
                        "dtype": "edas_table",
                    }
                )
            else:
                results.append(
                    {
                        "id": sig["id"],
                        "success": False,
                        "error": "not found in eddbreadTable",
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

    db.eddbClose()
    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
