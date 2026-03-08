#!/usr/bin/env python3
"""Parse JET magnetics configuration files (sensor geometry per epoch).

Python 3.12+ (runs via venv interpreter).

Parses the magn/config/ directory structure:
- indexr: Shot-range to config file mapping
- Config files (limves, pdvessri, pdvessra, etc.): Two-line sensor records
  with JPF address, PPF signal, index, calibration, R, Z, angle.

Each config file defines the complete set of magnetic sensors available
for a range of JET pulses, from the earliest operations (1983) through
the pre-EFIT++ era (2006).

Usage:
    echo '{"config_dir": "/home/chain1/input/magn/config"}' | python3 parse_magnetics_config.py

Input (JSON on stdin):
    {
        "config_dir": "/home/chain1/input/magn/config",
        "index_file": "/home/chain1/input/magn/config/indexr"
    }

Output (JSON on stdout):
    {
        "index": [
            {"first_shot": 1, "last_shot": 27968, "config_file": "limves"}
        ],
        "configs": {
            "limves": {
                "file_path": "/home/chain1/input/magn/config/limves",
                "sensors": [
                    {
                        "jpf_address": "DA/C2-CX01",
                        "ppf_signal": "MAGNBPOL",
                        "ppf_index": 1,
                        "sensor_type": "BPOL",
                        "index": 1,
                        "cal1": 0.0,
                        "cal2": 0.0,
                        "r": 4.2971,
                        "z": 0.6050,
                        "angle": -1.2933,
                        "gains": [1.0, 1.0, 1.0, 1.00, 1.0],
                        "flags": [0, 0],
                        "weight": 1.0,
                        "flag3": 0
                    }
                ],
                "sensor_counts": {"BPOL": 18, "FLUX": 17, ...}
            }
        }
    }
"""

import json
import os
import re
import sys

# Map PPF signal prefix to sensor type code
SENSOR_TYPE_MAP = {
    "MAGNBPOL": "BPOL",
    "MAGNFLUX": "FLUX",
    "MAGNSADX": "SADX",
    "MAGNTPC": "TPC",
    "MAGNTNC": "TNC",
    "MAGNTS": "TS",
    "MAGNTSFL": "TSFL",
    "MAGNTSRR": "TSRR",
    "MAGNPC": "PC",
    "MAGNBSAD": "BSAD",
    "MAGNIC": "IC",
    "MAGNDVC": "DVC",
    "MAGNXPOL": "XPOL",
    "MAGNXNOR": "XNOR",
    "MAGNITOR": "ITOR",
    "MAGNIPLA": "IPLA",
    "MAGNXTOR": "XTOR",
    "MAGNDL05": "DL05",
    "MAGNIMK": "IMK",
    "MAGOIPLA": "OIPLA",
    "MAGNVL": "VL",
    "MAGNFL": "FL",
    "MAGNFLRR": "FLRR",
    "MAGNVLRR": "VLRR",
    "MAGNUNC": "UNC",
    "MAGNFLD": "FLD",
    "MAGNUPC": "UPC",
    "MAGNP": "P",
    "MAGNIEXR": "IEXR",
}

# Human-readable descriptions for each sensor type
SENSOR_TYPE_DESCRIPTIONS = {
    "BPOL": "Magnetic pick-up coil (Bpol probe)",
    "FLUX": "Full flux loop",
    "SADX": "Saddle loop (cross-measurement)",
    "TPC": "Divertor target probe coil",
    "TNC": "TN divertor coil",
    "TS": "TS saddle loop",
    "TSFL": "TS flux loop",
    "TSRR": "TS RR measurement",
    "PC": "P8xx probe",
    "BSAD": "Broadband saddle loop",
    "IC": "ICRF-area coil",
    "DVC": "Diamagnetic/vertical field coil",
    "XPOL": "External poloidal coil",
    "XNOR": "External normal coil",
    "ITOR": "Toroidal field current (Rogowski)",
    "IPLA": "Plasma current (Rogowski)",
    "XTOR": "External toroidal coil",
    "DL05": "DL05 integrator",
    "IMK": "Mk2 measurement",
    "OIPLA": "Outer plasma current (alternate Rogowski)",
    "VL": "Voltage loop",
    "FL": "Flux (alternate)",
    "FLRR": "Flux RR measurement",
    "VLRR": "Voltage loop RR measurement",
    "UNC": "Uncertainty measurement",
    "FLD": "Field measurement",
    "UPC": "UP coil",
    "P": "P-series probe",
    "IEXR": "External Rogowski current",
}


def classify_sensor(ppf_signal_raw: str) -> tuple[str, str, int]:
    """Extract sensor type and index from PPF signal name.

    PPF signal names follow: MAGNTYPE INDEX (e.g., 'MAGNBPOL  1', 'MAGNFLUX10').
    Returns (ppf_signal_base, sensor_type, ppf_index).
    """
    # Strip quotes and whitespace
    sig = ppf_signal_raw.strip().strip("'").strip()

    # Match pattern: MAGN<TYPE><whitespace?><INDEX>
    match = re.match(r"(MAGN\w+?)\s*(\d+)$", sig)
    if match:
        ppf_base = match.group(1)
        ppf_index = int(match.group(2))
    else:
        # Fallback: whole string is the signal name, no numeric index
        ppf_base = sig
        ppf_index = 0

    sensor_type = SENSOR_TYPE_MAP.get(ppf_base, ppf_base.replace("MAGN", ""))
    return ppf_base, sensor_type, ppf_index


def parse_index_file(text: str) -> list[dict]:
    """Parse the indexr file mapping shot ranges to config files.

    Format: first_shot last_shot /full/path/to/config/file
    Header lines (no shot numbers) are skipped.
    """
    entries = []
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Match: first_shot [last_shot] /path/to/file
        match = re.match(r"(\d+)\s+(\d+)\s+(\S+)", stripped)
        if match:
            config_path = match.group(3)
            config_name = os.path.basename(config_path)
            entries.append(
                {
                    "first_shot": int(match.group(1)),
                    "last_shot": int(match.group(2)),
                    "config_file": config_name,
                    "config_path": config_path,
                }
            )
    return entries


def parse_config_file(text: str, file_path: str) -> dict:
    """Parse a magnetics config file into structured sensor records.

    Each sensor is defined by two consecutive lines:
    Line 1: 'JPF_ADDRESS' 'PPF_SIGNAL INDEX' index cal1 cal2 R Z angle
    Line 2: gain1 gain2 gain3 weight gain4 flag1 flag2 gain5 flag3

    Header/comment lines at the top are non-data and skipped.
    """
    sensors: list[dict] = []
    sensor_counts: dict[str, int] = {}

    lines = text.strip().split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Data lines start with a quoted JPF address
        if not line.startswith("'"):
            i += 1
            continue

        # Parse line 1: sensor definition
        # Format: 'DA/C2-CX01' 'MAGNBPOL  1'     1 0.0E0 0.0E0   4.2971   0.6050  -1.2933
        # Extract quoted strings first
        quotes = re.findall(r"'([^']*)'", line)
        if len(quotes) < 2:
            i += 1
            continue

        jpf_address = quotes[0].strip()
        ppf_signal_raw = quotes[1].strip()

        # Get everything after the second quoted string
        after_quotes = line
        for q in quotes:
            after_quotes = after_quotes.replace(f"'{q}'", "", 1)
        parts = after_quotes.strip().split()

        if len(parts) < 6:
            i += 1
            continue

        try:
            index = int(parts[0])
            cal1 = float(parts[1])
            cal2 = float(parts[2])
            r = float(parts[3])
            z = float(parts[4])
            angle = float(parts[5]) if len(parts) > 5 else 0.0
        except (ValueError, IndexError):
            i += 1
            continue

        # Parse line 2: gains and flags
        gains_line = {}
        if i + 1 < len(lines):
            gline = lines[i + 1].strip()
            if gline and not gline.startswith("'"):
                gparts = gline.split()
                try:
                    gains_line = {
                        "gain1": float(gparts[0]) if len(gparts) > 0 else 1.0,
                        "gain2": float(gparts[1]) if len(gparts) > 1 else 1.0,
                        "gain3": float(gparts[2]) if len(gparts) > 2 else 1.0,
                        "weight": float(gparts[3]) if len(gparts) > 3 else 1.0,
                        "gain4": float(gparts[4]) if len(gparts) > 4 else 1.0,
                        "flag1": int(float(gparts[5])) if len(gparts) > 5 else 0,
                        "flag2": int(float(gparts[6])) if len(gparts) > 6 else 0,
                        "gain5": float(gparts[7]) if len(gparts) > 7 else 1.0,
                        "flag3": int(float(gparts[8])) if len(gparts) > 8 else 0,
                    }
                except (ValueError, IndexError):
                    pass
                i += 1  # Skip gains line

        ppf_base, sensor_type, ppf_index = classify_sensor(ppf_signal_raw)
        sensor_counts[sensor_type] = sensor_counts.get(sensor_type, 0) + 1

        sensor = {
            "jpf_address": jpf_address,
            "ppf_signal": ppf_base,
            "ppf_index": ppf_index,
            "sensor_type": sensor_type,
            "index": index,
            "cal1": cal1,
            "cal2": cal2,
            "r": r,
            "z": z,
            "angle": angle,
        }
        sensor.update(gains_line)

        sensors.append(sensor)
        i += 1

    return {
        "file_path": file_path,
        "sensors": sensors,
        "sensor_counts": sensor_counts,
        "total_sensors": len(sensors),
    }


def main():
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(0)

    config_dir = config.get("config_dir", "")
    index_file = config.get("index_file", "")

    if not config_dir:
        print(json.dumps({"error": "No config_dir specified"}))
        sys.exit(0)

    if not index_file:
        index_file = os.path.join(config_dir, "indexr")

    result: dict = {"index": [], "configs": {}}

    # Parse index file
    try:
        with open(index_file) as f:
            result["index"] = parse_index_file(f.read())
    except FileNotFoundError:
        result["index_error"] = f"Index file not found: {index_file}"
    except Exception as e:
        result["index_error"] = f"{type(e).__name__}: {str(e)[:300]}"

    # Collect unique config files from index
    config_files: set[str] = set()
    for entry in result["index"]:
        config_files.add(entry["config_file"])

    # Parse each config file
    for config_name in sorted(config_files):
        config_path = os.path.join(config_dir, config_name)
        try:
            with open(config_path) as f:
                text = f.read()
            result["configs"][config_name] = parse_config_file(text, config_path)
        except FileNotFoundError:
            result["configs"][config_name] = {"error": f"File not found: {config_path}"}
        except Exception as e:
            result["configs"][config_name] = {
                "error": f"{type(e).__name__}: {str(e)[:300]}"
            }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
