#!/usr/bin/env python3
"""Parse JET PF coil circuit turns file.

Python 3.12+ (runs via venv interpreter).

Parses the pfcu/cturns file which maps PF coil circuit names to their
turns count and JPF data node addresses.

Usage:
    echo '{"base_dir": "/home/chain1/input/pfcu"}' | python3 parse_pf_coil_turns.py

Input (JSON on stdin):
    {"base_dir": "/home/chain1/input/pfcu"}

Output (JSON on stdout):
    {
        "coils": [
            {
                "name": "P2SUOI",
                "turns": 1,
                "node_id": "CNOD1",
                "jpf_address": "PF/UD-P2SUO<TRN"
            }
        ]
    }
"""

import json
import os
import re
import sys


def parse_cturns(text: str) -> list[dict]:
    """Parse PF coil turns file.

    Format per line:
        TURNS  / COIL_NAME  CNODnn ='JPF_ADDRESS'
    Example:
        1                   / P2SUOI     CNOD1 ='PF/UD-P2SUO<TRN'
    """
    coils = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Match: turns / coil_name CNODnn ='jpf_address'
        match = re.match(
            r"\s*(\d+)\s+/\s+(\S+)\s+(CNO\w+)\s*=\s*'([^']*)'",
            line,
        )
        if match:
            coils.append(
                {
                    "name": match.group(2),
                    "turns": int(match.group(1)),
                    "node_id": match.group(3),
                    "jpf_address": match.group(4),
                }
            )

    return coils


def main():
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(0)

    base_dir = config.get("base_dir", "")
    if not base_dir:
        print(json.dumps({"error": "No base_dir specified"}))
        sys.exit(0)

    cturns_path = os.path.join(base_dir, "cturns")

    try:
        with open(cturns_path) as f:
            text = f.read()
        coils = parse_cturns(text)
        print(json.dumps({"coils": coils, "file_path": cturns_path}))
    except FileNotFoundError:
        print(json.dumps({"error": f"File not found: {cturns_path}"}))
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {str(e)[:300]}"}))


if __name__ == "__main__":
    main()
