#!/usr/bin/env python3
"""Parse MCFG sensor position and calibration epoch files.

Python 3.12+ (runs via venv interpreter).

Parses magnetics sensor configuration files from the MAGNW group:
- sensors_200c_*.txt: Canonical sensor R,Z,angle positions from CATIA CAD
- MCFG.ix: Calibration epoch index (pulse-to-config mapping)

Usage:
    echo '{"base_dir": "/home/MAGNW/chain1/input", "files": [...]}' | python3 parse_mcfg_sensors.py

Input (JSON on stdin):
    {
        "base_dir": "/home/MAGNW/chain1/input",
        "files": [
            {"path": "PPFcfg/sensors_200c_2019-03-11.txt", "role": "sensors"},
            {"path": "magn_ep_2019-05-14/MCFG.ix", "role": "calibration_index"}
        ]
    }

Output (JSON on stdout):
    {
        "sensors": {
            "coils": [...],
            "hall_probes": [...],
            "other": [...]
        },
        "calibration_index": {
            "epochs": [...]
        }
    }
"""

import json
import os
import re
import sys


def parse_sensors(text: str) -> dict:
    """Parse sensor position file into structured sections.

    File has three sections separated by blank/comment lines:
    1. Coils (id 1-238): pick-up coils with R, Z, angle, gain, errors
    2. Hall probes (id 1-8): ex-vessel probes
    3. Other: Rogowski coils, TF current integrators

    Lines starting with # or ! are comments. Data lines have
    tab/space-separated fields with varying formats per section.
    """
    coils: list[dict] = []
    hall_probes: list[dict] = []
    other: list[dict] = []

    lines = text.strip().split("\n")
    section = "coils"  # Start in coils section
    end_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Section separator or comment
        if stripped.startswith("#"):
            if stripped.upper() == "#END" or "END" in stripped.upper():
                end_count += 1
                if end_count == 1:
                    section = "hall_probes"
                elif end_count >= 2:
                    section = "other"
            continue

        if stripped.startswith("!") or stripped.startswith("*"):
            continue

        # Parse data line based on section
        parts = stripped.split()
        if len(parts) < 2:
            continue

        try:
            if section == "coils" and len(parts) >= 9:
                entry = {
                    "id": int(parts[0]),
                    "r": float(parts[1]),
                    "z": float(parts[2]),
                    "angle": float(parts[3]),
                    "gain": float(parts[4]),
                    "rel_error": float(parts[5]),
                    "abs_error": float(parts[6]),
                    "jpf_name": parts[7],
                    "description": " ".join(parts[9:]) if len(parts) > 9 else "",
                }
                # parts[8] might be a poloidal angle or annotation
                if len(parts) > 8:
                    try:
                        entry["poloidal_angle"] = float(parts[8])
                    except ValueError:
                        entry["description"] = " ".join(parts[8:])
                coils.append(entry)

            elif section == "hall_probes" and len(parts) >= 9:
                entry = {
                    "id": int(parts[0]),
                    "r": float(parts[1]),
                    "z": float(parts[2]),
                    "offset": float(parts[3]),
                    "gain": float(parts[4]),
                    "rel_error": float(parts[5]),
                    "abs_error": float(parts[6]),
                    "jpf_name": parts[7],
                    "description": " ".join(parts[8:]),
                }
                hall_probes.append(entry)

            elif section == "other":
                # Rogowski coils, TFCI — simpler format
                entry = {
                    "name": parts[0],
                    "values": parts[1:],
                }
                other.append(entry)

        except (ValueError, IndexError):
            continue

    return {"coils": coils, "hall_probes": hall_probes, "other": other}


def parse_calibration_index(text: str) -> dict:
    """Parse MCFG.ix calibration epoch index.

    Format: $:YYYYMMDD PPPPPP user MCFG:NNNN/MAGNW type ! comment
    """
    epochs: list[dict] = []

    for line in text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Match $:YYYYMMDD PPPPPP pattern
        match = re.match(
            r"^\$:(\d{8})\s+0*(\d+)\s+(\S+)\s+(MCFG:\d+/\S+)\s+(\S+)\s*!?\s*(.*)",
            stripped,
        )
        if match:
            epochs.append(
                {
                    "date": match.group(1),
                    "first_shot": int(match.group(2)),
                    "user": match.group(3),
                    "config_id": match.group(4),
                    "config_type": match.group(5),
                    "description": match.group(6).strip(),
                }
            )

    return {"epochs": epochs}


def main():
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(0)

    base_dir = config.get("base_dir", "")
    files = config.get("files", [])

    if not base_dir:
        print(json.dumps({"error": "No base_dir specified"}))
        sys.exit(0)

    result: dict = {}

    role_parsers = {
        "sensors": parse_sensors,
        "calibration_index": parse_calibration_index,
    }

    for file_entry in files:
        path = file_entry.get("path", "")
        role = file_entry.get("role", "")

        if not path or not role:
            continue

        full_path = os.path.join(base_dir, path)
        parser = role_parsers.get(role)
        if not parser:
            result[role] = {"error": f"Unknown role: {role}"}
            continue

        try:
            with open(full_path) as f:
                text = f.read()
            result[role] = parser(text)
        except FileNotFoundError:
            result[role] = {"error": f"File not found: {full_path}"}
        except Exception as e:
            result[role] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
