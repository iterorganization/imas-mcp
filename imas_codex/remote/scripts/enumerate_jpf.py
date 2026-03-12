#!/usr/bin/env python3
"""Enumerate JET JPF (JET Processing Facility) subsystems and signals.

This script runs on the JET host where the ``getdat`` Python module is
available (``/jet/share/lib/python/getdat.py``).  It uses the JPF pulse
file API to enumerate every signal node within each subsystem.

Key getdat functions used:
    zadop(pulse, system=sub, type="JPF")  — returns (status, pptab, pulse)
    adlnod(pptab, stub="{sub}/*")         — returns (status, handle, names_list, count)
    getcom(name, pulse)                   — returns (description, length, status)
    adcl(pptab)                           — close pulse file handle

Requirements:
- Python 3.8+ (stdlib only except getdat)
- getdat module available on JET (loaded via ``module load jet/1.0``)

Usage:
    echo '{"shot": 99896, "subsystems": ["DA","DB"]}' | python3 enumerate_jpf.py

Input (JSON on stdin):
    {
        "shot": 99896,
        "subsystems": ["DA", "DB", "PF", ...],
        "max_signals_per_subsystem": 0
    }

Output (JSON on stdout):
    {
        "signals": [
            {"subsystem": "DA", "signal": "C2-IPLA", "path": "da/c2-ipla",
             "description": "Plasma Current"},
            ...
        ],
        "shot": 99896,
        "subsystems_scanned": 5,
        "subsystems_failed": 1,
        "errors": ["PL: zadop status=6801"]
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

    shot = config.get("shot")
    subsystems = config.get("subsystems", [])
    max_per_sub = config.get("max_signals_per_subsystem", 0)

    if not shot:
        print(json.dumps({"error": "No shot specified"}))
        sys.exit(0)

    # Import getdat from JET shared python libs
    sys.path.insert(0, "/jet/share/lib/python")
    try:
        import getdat
    except ImportError:
        print(json.dumps({"error": "getdat module not available"}))
        sys.exit(0)

    results = []
    errors = []
    subsystems_scanned = 0

    for subsystem in subsystems:
        sub_upper = subsystem.upper().strip()
        sub_lower = sub_upper.lower()

        try:
            # Open the JPF pulse file for this subsystem
            status, pptab, _pulse = getdat.zadop(shot, system=sub_upper, type="JPF")
            if status != 0:
                errors.append(f"{sub_upper}: zadop status={status}")
                continue

            try:
                # Enumerate all signal nodes in this subsystem
                # adlnod returns (status, handle, names_list, count)
                ad_status, _handle, node_names, _count = getdat.adlnod(
                    pptab, stub=f"{sub_upper}/*"
                )

                if ad_status != 0 or not node_names:
                    errors.append(
                        f"{sub_upper}: adlnod status={ad_status}, "
                        f"nodes={len(node_names) if node_names else 0}"
                    )
                    continue

                signal_names = []
                for node in node_names:
                    name = str(node).strip()
                    if name:
                        signal_names.append(name)

                # Apply per-subsystem limit if set (0 = unlimited)
                if max_per_sub > 0:
                    signal_names = signal_names[:max_per_sub]

                for name in signal_names:
                    # Node names from adlnod are like "DA/C2-IPLA"
                    # Extract signal part after the subsystem prefix
                    if "/" in name:
                        signal_part = name.split("/", 1)[1]
                    else:
                        signal_part = name

                    path = f"{sub_lower}/{signal_part.lower()}"

                    # Try to get description via getcom
                    # getcom returns (description, length, status)
                    description = ""
                    try:
                        desc_text, _dlen, desc_status = getdat.getcom(name, shot)
                        if desc_status == 0 and desc_text:
                            description = str(desc_text).strip()
                    except Exception:
                        pass

                    entry = {
                        "subsystem": sub_upper,
                        "signal": signal_part,
                        "path": path,
                    }
                    if description:
                        entry["description"] = description

                    results.append(entry)

                subsystems_scanned += 1

            finally:
                # Always close the pulse file handle
                try:
                    getdat.adcl(pptab)
                except Exception:
                    pass

        except Exception as e:
            errors.append(f"{sub_upper}: {str(e)[:200]}")

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
