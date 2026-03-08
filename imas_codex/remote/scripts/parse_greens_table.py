#!/usr/bin/env python3
"""Parse JET Greens table version-to-shot mapping.

Python 3.12+ (runs via venv interpreter).

Parses the efit/green_list file which maps shot ranges to precomputed
Greens function table directories used by EFIT.

Usage:
    echo '{"base_dir": "/home/chain1/input/efit"}' | python3 parse_greens_table.py

Input (JSON on stdin):
    {"base_dir": "/home/chain1/input/efit"}

Output (JSON on stdout):
    {
        "entries": [
            {
                "first_shot": 1,
                "last_shot": 63278,
                "greens_dir": "/GREENS/pre_JET_EP_DMSS_091_T_285C_nbpol_95/GREEN3333",
                "version": "pre_JET_EP_DMSS_091_T_285C",
                "comment": "DMSS_91 as a standard"
            }
        ]
    }
"""

import json
import os
import re
import sys


def parse_green_list(text: str) -> list[dict]:
    """Parse green_list file.

    Format per line:
        first_shot last_shot /GREENS/version_dir/GREEN3333  ! comment
    or single-shot entry:
        first_shot /GREENS/version_dir/GREEN3333  ! comment
    """
    entries = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Split off comment
        comment = ""
        if "!" in line:
            main_part, comment = line.split("!", 1)
            comment = comment.strip()
        else:
            main_part = line

        parts = main_part.split()
        if len(parts) < 2:
            continue

        try:
            first_shot = int(parts[0])
        except ValueError:
            continue

        # Check if second field is a number (shot range) or path
        try:
            last_shot = int(parts[1])
            greens_path = parts[2] if len(parts) > 2 else ""
        except ValueError:
            last_shot = None
            greens_path = parts[1]

        # Extract version name from path
        # e.g. /GREENS/JET_EP_DMSS_091_T_200C_nbpol_95/GREEN3333
        version = greens_path
        match = re.search(r"/GREENS/([^/]+)/", greens_path)
        if match:
            version = match.group(1)
            # Strip common suffixes for cleaner version names
            version = re.sub(r"_nbpol_\d+$", "", version)

        entry = {
            "first_shot": first_shot,
            "greens_dir": greens_path,
            "version": version,
        }
        if last_shot is not None:
            entry["last_shot"] = last_shot
        if comment:
            entry["comment"] = comment

        entries.append(entry)

    return entries


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

    green_list_path = os.path.join(base_dir, "green_list")

    try:
        with open(green_list_path) as f:
            text = f.read()
        entries = parse_green_list(text)
        print(json.dumps({"entries": entries, "file_path": green_list_path}))
    except FileNotFoundError:
        print(json.dumps({"error": f"File not found: {green_list_path}"}))
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {str(e)[:300]}"}))


if __name__ == "__main__":
    main()
