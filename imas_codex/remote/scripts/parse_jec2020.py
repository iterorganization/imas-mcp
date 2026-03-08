#!/usr/bin/env python3
"""Parse JEC2020 XML files for EFIT++ equilibrium geometry.

Python 3.12+ (runs via venv interpreter).

Parses JEC2020 XML files containing next-generation geometry data:
- magnetics.xml: 95 magnetic probes + 36 flux loops with dual PPF/JPF sources
- pfSystems.xml: 20 PF coils + 10 circuits with multi-element geometry
- ironBoundaries3.xml: 96 iron core boundary segments with permeabilities
- limiter.xml: High-resolution ILW first wall contour at T=200°C

Usage:
    echo '{"base_dir": "/home/chain1/jec2020", "files": [...]}' | python3 parse_jec2020.py

Input (JSON on stdin):
    {
        "base_dir": "/home/chain1/jec2020",
        "files": [
            {"path": "magnetics.xml", "role": "magnetics"},
            {"path": "pfSystems.xml", "role": "pf_coils"},
            {"path": "ironBoundaries3.xml", "role": "iron_core"},
            {"path": "limiter.xml", "role": "limiter"}
        ]
    }

Output (JSON on stdout):
    {
        "magnetics": {
            "probes": [...],
            "flux_loops": [...]
        },
        "pf_coils": {
            "coils": [...],
            "circuits": [...]
        },
        "iron_core": {
            "segments": [...],
            "boundary_length": 32.962
        },
        "limiter": {
            "r": [...],
            "z": [...],
            "n_points": 248
        }
    }
"""

import json
import os
import sys
import xml.etree.ElementTree as ET


def _parse_float_list(s: str) -> list:
    """Parse a string of floats separated by commas or whitespace."""
    if "," in s:
        return [float(v.strip()) for v in s.split(",") if v.strip()]
    return [float(v) for v in s.split() if v]


def parse_magnetics(xml_bytes: bytes) -> dict:
    """Parse magnetics.xml: probes and flux loops with dual data sources.

    Each probe has geometry (R, Z, angle) plus dual PPF/JPF signal references.
    Each flux loop has geometry plus dual signal references.
    """
    root = ET.fromstring(xml_bytes)
    probes: list[dict] = []
    flux_loops: list[dict] = []

    # Parse magneticProbe elements
    for probe in root.iter("magneticProbe"):
        entry: dict = {
            "id": probe.get("id", ""),
            "description": probe.get("description", ""),
        }

        # Geometry sub-element
        geom = probe.find("geometry")
        if geom is not None:
            for attr in ("rCentre", "zCentre", "poloidalOrientation"):
                val = geom.get(attr)
                if val is not None:
                    try:
                        entry[attr] = float(val)
                    except ValueError:
                        entry[attr] = val
            if geom.get("angleUnits"):
                entry["angle_units"] = geom.get("angleUnits")

        # TimeTrace sub-element — signal references and data sources
        tt = probe.find("timeTrace")
        if tt is not None:
            entry["ppf_signal"] = tt.get("signalName", "")
            entry["jpf_signal"] = tt.get("signalName2", "")
            entry["ppf_data_source"] = tt.get("dataSource", "")
            entry["jpf_data_source"] = tt.get("dataSource2", "")
            entry["error_type"] = tt.get("errorType", "")
            error_val = tt.get("errorRelativeAbsolute", "")
            if error_val:
                # Parse "[rel,abs]" format
                try:
                    vals = error_val.strip("[]").split(",")
                    if len(vals) == 2:
                        entry["rel_error"] = float(vals[0])
                        entry["abs_error"] = float(vals[1])
                except (ValueError, IndexError):
                    entry["error_raw"] = error_val

        probes.append(entry)

    # Parse fluxLoop elements
    for loop in root.iter("fluxLoop"):
        entry = {"id": loop.get("id", ""), "description": loop.get("description", "")}

        geom = loop.find("geometry")
        if geom is not None:
            for attr in ("rCentre", "zCentre"):
                val = geom.get(attr)
                if val is not None:
                    try:
                        entry[attr] = float(val)
                    except ValueError:
                        entry[attr] = val

        tt = loop.find("timeTrace")
        if tt is not None:
            entry["ppf_signal"] = tt.get("signalName", "")
            entry["jpf_signal"] = tt.get("signalName2", "")
            entry["ppf_data_source"] = tt.get("dataSource", "")
            entry["jpf_data_source"] = tt.get("dataSource2", "")

        flux_loops.append(entry)

    return {"probes": probes, "flux_loops": flux_loops}


def parse_pf_systems(xml_bytes: bytes) -> dict:
    """Parse pfSystems.xml: PF coils and circuits.

    Some coils have multi-element geometry with comma-separated arrays.
    """
    root = ET.fromstring(xml_bytes)
    coils: list[dict] = []
    circuits: list[dict] = []

    for coil in root.iter("pfCoil"):
        entry: dict = {"id": coil.get("id", ""), "name": coil.get("name", "")}

        geom = coil.find("geometry")
        if geom is not None:
            for attr in (
                "rCentre",
                "zCentre",
                "dR",
                "dZ",
                "angle1",
                "angle2",
                "turnCount",
            ):
                val = geom.get(attr)
                if val is not None:
                    # Handle comma-separated arrays (multi-element coils)
                    if "," in val:
                        try:
                            entry[attr] = [
                                float(v.strip()) for v in val.split(",") if v.strip()
                            ]
                        except ValueError:
                            entry[attr] = val
                    else:
                        try:
                            entry[attr] = float(val)
                        except ValueError:
                            entry[attr] = val

        coils.append(entry)

    for circuit in root.iter("pfCircuit"):
        entry = {
            "id": circuit.get("id", ""),
            "name": circuit.get("name", ""),
        }
        # Parse coil connections
        connections = circuit.find("connections")
        if connections is not None:
            coil_refs: list[str] = []
            for conn in connections:
                coil_refs.append(conn.get("coilId", ""))
            entry["coil_ids"] = coil_refs

        circuits.append(entry)

    return {"coils": coils, "circuits": circuits}


def _get_attr(parent, child, attr):
    """Get attribute from child element, falling back to parent."""
    if child is not None:
        val = child.get(attr, "")
        if val:
            return val
    return parent.get(attr, "")


def parse_iron_boundaries(xml_bytes: bytes) -> dict:
    """Parse ironBoundaries3.xml: iron core boundary segments.

    Structure: <ironBoundaries> → <ironBoundary> with children:
      - <knotSet>: basisFunctionCount, spline knot parameters
      - <observationPoints>: initialPermeabilities
      - <geometry>: boundaryCoordsR, boundaryCoordsZ, segmentLengths, boundaryLength
    """
    root = ET.fromstring(xml_bytes)
    result: dict = {}

    boundary = root.find(".//ironBoundary")
    if boundary is None:
        if root.tag == "ironBoundary":
            boundary = root
        else:
            return {"error": "No ironBoundary element found"}

    result["material_id"] = boundary.get("materialId", "")
    result["material2_id"] = boundary.get("material2Id", "")

    # Geometry element has R,Z coordinates, segment lengths, boundary length
    geom = boundary.find("geometry")

    # Coordinates and segment data from <geometry>
    for attr, key in [
        ("boundaryCoordsR", "r"),
        ("boundaryCoordsZ", "z"),
        ("segmentLengths", "segment_lengths"),
    ]:
        val = _get_attr(boundary, geom, attr)
        if val:
            try:
                result[key] = _parse_float_list(val)
            except ValueError:
                result[key] = val

    # boundaryLength from <geometry>
    bl = _get_attr(boundary, geom, "boundaryLength")
    if bl:
        try:
            result["boundary_length"] = float(bl)
        except ValueError:
            result["boundary_length"] = bl

    # Permeabilities from <observationPoints>
    obs = boundary.find("observationPoints")
    perm_val = _get_attr(boundary, obs, "initialPermeabilities")
    if perm_val:
        try:
            result["permeabilities"] = _parse_float_list(perm_val)
        except ValueError:
            result["permeabilities"] = perm_val

    # Segment count from <knotSet>
    knot_set = boundary.find("knotSet")
    bfc = _get_attr(boundary, knot_set, "basisFunctionCount")
    if bfc:
        try:
            result["n_segments"] = float(bfc)
        except ValueError:
            result["n_segments"] = bfc

    return result


def parse_limiter(xml_bytes: bytes) -> dict:
    """Parse limiter.xml: ILW first wall contour at T=200°C.

    The XML has rValues and zValues attributes. Values may be comma-separated
    or whitespace-separated depending on the file version.
    """
    root = ET.fromstring(xml_bytes)

    # Look for limiter element with rValues/zValues
    limiter = root.find(".//limiter")
    if limiter is None:
        limiter = root  # might be root element

    r_str = limiter.get("rValues", "")
    z_str = limiter.get("zValues", "")

    if not r_str or not z_str:
        # Try child elements
        for child in limiter:
            if child.get("rValues"):
                r_str = child.get("rValues", "")
                z_str = child.get("zValues", "")
                break

    if not r_str or not z_str:
        return {"error": "No rValues/zValues attributes found"}

    try:
        r_vals = _parse_float_list(r_str)
        z_vals = _parse_float_list(z_str)
    except ValueError as e:
        return {"error": f"Failed to parse coordinates: {e}"}

    return {"r": r_vals, "z": z_vals, "n_points": len(r_vals)}


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
        "magnetics": parse_magnetics,
        "pf_coils": parse_pf_systems,
        "iron_core": parse_iron_boundaries,
        "limiter": parse_limiter,
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
            with open(full_path, "rb") as f:
                xml_bytes = f.read()
            result[role] = parser(xml_bytes)
        except FileNotFoundError:
            result[role] = {"error": f"File not found: {full_path}"}
        except Exception as e:
            result[role] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
