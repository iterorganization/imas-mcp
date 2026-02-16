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

    # eddbWrapper Python wrapper — no constructor arguments needed
    # (the C API path is handled internally by the wrapper)
    db = eddbWrapper()
    # eddbOpen returns rtn_bool — True on success
    ok = db.eddbOpen()
    if not ok:
        print(json.dumps({"error": "eddbOpen() failed"}))
        sys.exit(0)

    # Step 1: Get category listing
    # eddbreadCatTable returns (rtn_bool, rtn_data) where rtn_data is dict
    # with keys: count, catlist, desclist, rolist, ircgrp, irc
    try:
        cat_ok, cat_data = db.eddbreadCatTable()
        if cat_ok and cat_data:
            categories = cat_data.get("catlist", [])
        else:
            categories = []
    except Exception:
        # Fallback: use known categories from exploration
        categories = []

    signals = []
    for cat in categories:
        if not cat or not cat.strip():
            continue
        cat = cat.strip()
        try:
            # Step 2: Read data table for this category
            # eddbreadTable returns (rtn_bool, rtn_data) where rtn_data is dict
            # with keys: count, data, dnamelist, aliaslist, udpidlist,
            #            classlist, shotlist, unitlist, desclist, ircgrp, irc
            tbl_ok, tbl_data = db.eddbreadTable(shot=ref_shot, cat=cat)
            if not tbl_ok or tbl_data is None:
                continue

            dnames = tbl_data.get("dnamelist", [])
            aliases = tbl_data.get("aliaslist", [])
            units = tbl_data.get("unitlist", [])
            descs = tbl_data.get("desclist", [])

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
