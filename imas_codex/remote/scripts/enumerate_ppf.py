#!/usr/bin/env python3
"""Enumerate JET PPF DDAs and Dtypes for a reference pulse.

This script runs on the JET host where libppf.so and ppf.py are available.
It uses ppfdda()/ppfdti() to enumerate all DDAs and their Dtypes.

Requirements:
- Python 3.8+ (stdlib only except ppf)
- ppf Python bindings (available on JET via /jet/share/DEPOT/pyppf/)

Usage:
    echo '{"pulse": 99999, "owner": "jetppf"}' | python3 enumerate_ppf.py

Input (JSON on stdin):
    {
        "pulse": 99999,
        "owner": "jetppf",
        "exclude_ddas": ["PRIV"]
    }

Output (JSON on stdout):
    {
        "signals": [
            {"dda": "EFIT", "dtype": "BVAC"},
            {"dda": "EFIT", "dtype": "FBND"},
            ...
        ],
        "pulse": 99999,
        "owner": "jetppf",
        "ndda": 42
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

    pulse = config.get("pulse")
    owner = config.get("owner", "jetppf")
    exclude_ddas = set(config.get("exclude_ddas", []))

    if not pulse:
        print(json.dumps({"error": "No pulse specified"}))
        sys.exit(0)

    # Import ppf — available on JET compute nodes
    # Primary path: /jet/share/lib/python (set by 'module load jet/1.0')
    # Fallback: legacy DEPOT path
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
                        "error": "ppf module not available. Tried /jet/share/lib/python and /jet/share/DEPOT/pyppf/"
                    }
                )
            )
            sys.exit(0)

    # Set default user
    ppf.ppfuid(owner, rw="R")

    # Open pulse context
    ier = ppf.ppfgo(pulse, seq=0)
    if ier != 0:
        print(json.dumps({"error": f"ppfgo failed: ier={ier}"}))
        sys.exit(0)

    # Enumerate DDAs for this pulse
    # New API: ppfdds(pulse) returns (count, dda_list, dtype_list, seq_array, ier)
    # Old API: ppfdda(pulse) returns (dda_list, ndda, ier)
    try:
        ndds, ddas_list, dtypes_list, _seqs, ier = ppf.ppfdds(pulse)
        if ier != 0:
            print(json.dumps({"error": f"ppfdds failed: ier={ier}"}))
            sys.exit(0)

        # Build unique DDA/Dtype pairs from the flat lists
        seen = set()
        results = []
        for i in range(ndds):
            dda = ddas_list[i].strip()
            dtype = dtypes_list[i].strip()
            if not dda or not dtype or dda in exclude_ddas:
                continue
            key = (dda, dtype)
            if key in seen:
                continue
            seen.add(key)
            results.append({"dda": dda, "dtype": dtype})

        ndda = len({r["dda"] for r in results})
    except (TypeError, ValueError):
        # Fallback: Old API with ppfdda + ppfdti
        ddas, ndda, ier = ppf.ppfdda(pulse)
        if ier != 0:
            print(json.dumps({"error": f"ppfdda failed: ier={ier}"}))
            sys.exit(0)

        results = []
        for dda in ddas[:ndda]:
            dda = dda.strip()
            if not dda or dda in exclude_ddas:
                continue
            try:
                dtypes, ndtype, ier = ppf.ppfdti(pulse, dda)
                if ier != 0:
                    continue
                for dtype in dtypes[:ndtype]:
                    dtype = dtype.strip()
                    if not dtype:
                        continue
                    results.append({"dda": dda, "dtype": dtype})
            except Exception:
                pass

    print(
        json.dumps(
            {
                "signals": results,
                "pulse": pulse,
                "owner": owner,
                "ndda": ndda,
            }
        )
    )


if __name__ == "__main__":
    main()
