#!/usr/bin/env python3
"""Enumerate JT-60SA EDAS categories and data names.

This script runs on the JT-60SA host where the eddb_pwrapper module
is available. It uses eddbreadCatTable()/eddbreadTable() to enumerate
all categories and their data names with metadata.

Requirements:
- Python 3.8+ (stdlib only except eddb_pwrapper)
- eddb_pwrapper or edas_eddb_api (available at /analysis/lib)

Usage:
    echo '{"ref_shot": "E012345"}' | python3 enumerate_edas.py

Input (JSON on stdin):
    {
        "ref_shot": "E012345"
    }

Output (JSON on stdout):
    {
        "signals": [
            {
                "category": "EDDB",
                "data_name": "tesTime",
                "alias": "",
                "units": "s",
                "description": "Time base"
            },
            ...
        ],
        "shot": "E012345",
        "categories": ["EDDB", ...],
        "ncats": 5
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

    ref_shot = config.get("ref_shot", "")
    if not ref_shot:
        print(json.dumps({"error": "No ref_shot specified"}))
        sys.exit(0)

    # Import eddb_pwrapper — available on JT-60SA compute nodes
    # Primary path: /analysis/src/eddb (eddb_pwrapper.py)
    # The wrapper needs libeddb.so at /analysis/lib/libeddb.so
    try:
        sys.path.insert(0, "/analysis/src/eddb")
        from eddb_pwrapper import eddbWrapper
    except ImportError:
        try:
            sys.path.insert(0, "/analysis/lib")
            from eddb_pwrapper import eddbWrapper
        except ImportError:
            try:
                sys.path.insert(
                    0, "/analysis/src/edas2/v2.1.15/edas_share/database_api"
                )
                import edas_eddb_api  # noqa: F401

                print(
                    json.dumps(
                        {
                            "error": "eddb_pwrapper not available, edas_eddb_api loaded but unsupported"
                        }
                    )
                )
                sys.exit(0)
            except ImportError:
                print(
                    json.dumps(
                        {
                            "error": "No EDAS Python API available. Tried /analysis/src/eddb and /analysis/lib"
                        }
                    )
                )
                sys.exit(0)

    # eddbWrapper requires the path to libeddb.so
    db = eddbWrapper("/analysis/lib/libeddb.so")
    # eddbOpen takes no arguments (connects to default EDDB)
    db.eddbOpen()

    # Step 1: Get category listing
    try:
        cat_result = db.eddbreadCatTable()
        categories = (
            cat_result.get("categories", []) if isinstance(cat_result, dict) else []
        )
    except Exception:
        # Fallback: use known categories from exploration
        categories = ["EDDB"]

    signals = []
    for cat in categories:
        if not cat:
            continue
        try:
            # Step 2: Read data table for this category
            table = db.eddbreadTable(shot=ref_shot, cat=cat)
            if table is None:
                continue

            dnames = table.get("dnamelist", [])
            aliases = table.get("aliaslist", [])
            units = table.get("unitlist", [])
            descs = table.get("desclist", [])

            for i, dname in enumerate(dnames):
                if not dname or not dname.strip():
                    continue
                signals.append(
                    {
                        "category": cat,
                        "data_name": dname.strip(),
                        "alias": aliases[i].strip() if i < len(aliases) else "",
                        "units": units[i].strip() if i < len(units) else "",
                        "description": descs[i].strip() if i < len(descs) else "",
                    }
                )
        except Exception:
            pass

    db.eddbClose()
    print(
        json.dumps(
            {
                "signals": signals,
                "shot": ref_shot,
                "categories": categories,
                "ncats": len(categories),
            }
        )
    )


if __name__ == "__main__":
    main()
