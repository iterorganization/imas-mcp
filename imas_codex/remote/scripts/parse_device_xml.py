#!/usr/bin/env python3
"""Parse JET device XML files from a bare git repo.

Python 3.12+ (runs via venv interpreter).

Parses device XML files containing geometry for PF coils, passive structures,
magnetic probes (BPME), flux loops, circuits, and scalar diagnostics.
Also parses EFITSNAP files for probe enable/disable masks and limiter
contour files for first-wall R,Z geometry.

Usage:
    echo '{"git_repo": "/home/chain1/git/efit_f90.git", ...}' | python3 parse_device_xml.py

Input (JSON on stdin):
    {
        "git_repo": "/home/chain1/git/efit_f90.git",
        "input_prefix": "JET/input",
        "versions": [
            {
                "version": "p72258",
                "device_xml": "Devices/device_p72258.xml",
                "snap_file": "Snap_files/EFITSNAP/efitsnap_p72258_bound0"
            }
        ],
        "limiter_files": [
            {"name": "Mk2ILW", "file": "Limiters/limiter.mk2ilw_cc"}
        ]
    }

Output (JSON on stdout):
    {
        "versions": {
            "p72258": {
                "magprobes": [...],
                "flux": [...],
                "pfcoils": [...],
                "pfcircuits": [...],
                "pfsupplies": [...],
                "pfpassive": [...],
                "toroidalfield": [...],
                "plasmacurrent": [...],
                "diamagneticflux": [...],
                "enabled_probes": [...],
                "disabled_probes": [...]
            }
        },
        "limiters": {
            "Mk2ILW": {"r": [...], "z": [...], "n_points": 251}
        }
    }
"""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET


def git_show(git_repo: str, path: str) -> bytes:
    """Read a file from a bare git repo via git show, resolving symlinks."""
    # Check if the path is a symlink (git mode 120000)
    ls_result = subprocess.run(
        ["git", "-C", git_repo, "ls-tree", "HEAD", path],
        capture_output=True,
        timeout=30,
    )
    if ls_result.returncode == 0 and ls_result.stdout:
        mode = ls_result.stdout.split()[0].decode()
        if mode == "120000":
            # Symlink — read target and resolve relative to parent dir
            target_result = subprocess.run(
                ["git", "-C", git_repo, "show", f"HEAD:{path}"],
                capture_output=True,
                timeout=30,
            )
            target = target_result.stdout.decode().strip()
            parent = "/".join(path.split("/")[:-1])
            resolved = f"{parent}/{target}" if parent else target
            path = resolved

    result = subprocess.run(
        ["git", "-C", git_repo, "show", f"HEAD:{path}"],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git show HEAD:{path} failed: {result.stderr.decode()[:200]}"
        )
    return result.stdout


def parse_instance(elem) -> dict:
    """Parse a single XML <instance> element into a dict of attributes."""
    data: dict = {}
    # XML attributes (id, file, signal, archive, owner, status, seq)
    for key in ("id", "file", "signal", "archive", "owner", "status", "seq"):
        val = elem.get(key)
        if val is not None:
            data[key] = val

    # Numeric sub-elements (r, z, angle, dr, dz, ang1, ang2, etc.)
    for child in elem:
        tag = child.tag.lower()
        text = child.text
        if text is not None:
            text = text.strip()
            try:
                data[tag] = float(text)
            except ValueError:
                data[tag] = text
    return data


def parse_device_xml(xml_bytes: bytes) -> dict:
    """Parse a device XML file and return structured geometry data."""
    root = ET.fromstring(xml_bytes)

    sections: dict = {}

    # Section mapping: XML parent tag -> output key
    section_map = {
        "magprobes": "magprobes",
        "fluxloops": "flux",
        "pfcoils": "pfcoils",
        "pfcircuits": "pfcircuits",
        "pfsupplies": "pfsupplies",
        "pfpassive": "pfpassive",
    }

    for xml_tag, output_key in section_map.items():
        parent = root.find(xml_tag)
        if parent is None:
            # Try alternate casing
            parent = root.find(xml_tag.lower())
        if parent is not None:
            instances = []
            for inst in parent.findall("instance"):
                instances.append(parse_instance(inst))
            sections[output_key] = instances

    # Scalar sections (single instance each)
    for scalar_tag in ("toroidalfield", "plasmacurrent", "diamagneticflux"):
        parent = root.find(scalar_tag)
        if parent is not None:
            instances = []
            for inst in parent.findall("instance"):
                instances.append(parse_instance(inst))
            sections[scalar_tag] = instances

    return sections


def parse_snap_file(snap_bytes: bytes) -> tuple[list[str], list[str]]:
    """Parse an EFITSNAP file to extract enabled/disabled probe lists.

    EFITSNAP files are Fortran namelists with probe enable/disable masks.
    Lines with probe identifiers and their status (0=disabled, 1=enabled).
    The format is: each line has probe_name and a 0/1 flag.
    """
    enabled: list[str] = []
    disabled: list[str] = []

    text = snap_bytes.decode("utf-8", errors="replace")
    lines = text.strip().split("\n")

    # Parse Fortran namelist style: look for probe enable/disable patterns
    # The EFITSNAP format has sections with probe status indicators
    in_probes = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("!") or line.startswith("#"):
            continue

        # Look for probe status lines: typically "BPME(N) = 0/1" or similar
        # Also handle "probe_name status" format
        if "bpme" in line.lower() or "flme" in line.lower():
            in_probes = True

        if in_probes:
            # Try to parse "name = value" format
            if "=" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    name = parts[0].strip().strip("'\"")
                    val = parts[1].strip().rstrip(",").strip()
                    try:
                        status = int(float(val))
                        if status == 0:
                            disabled.append(name)
                        else:
                            enabled.append(name)
                    except (ValueError, IndexError):
                        pass

    return enabled, disabled


def parse_limiter_file(data: bytes) -> dict:
    """Parse a limiter contour file with segment-count headers.

    Format: Each segment starts with an integer point-count on its own line,
    followed by that many R,Z coordinate pairs.  Comment lines (C/c/!/# prefix)
    and blank lines are skipped.  Only the first segment (the primary first-wall
    contour) is returned; secondary segments (inner boundaries, septum outlines)
    are discarded.
    """
    text = data.decode("utf-8", errors="replace")
    r_vals: list[float] = []
    z_vals: list[float] = []

    expected = 0  # remaining points in current segment
    segment_done = False

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip comment lines (Fortran-style C/c, or #/!)
        if line[0] in ("C", "c", "!", "#"):
            continue

        parts = line.split()

        # A single integer on a line = segment point count
        if len(parts) == 1:
            try:
                count = int(parts[0])
            except ValueError:
                continue
            if segment_done:
                # We already have the first segment — stop
                break
            expected = count
            continue

        # R,Z coordinate pair
        if len(parts) >= 2 and expected > 0:
            try:
                r_vals.append(float(parts[0]))
                z_vals.append(float(parts[1]))
                expected -= 1
                if expected == 0:
                    segment_done = True
            except ValueError:
                continue

    return {"r": r_vals, "z": z_vals, "n_points": len(r_vals)}


def main():
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(0)

    git_repo = config.get("git_repo")
    input_prefix = config.get("input_prefix", "")
    versions = config.get("versions", [])
    limiter_files = config.get("limiter_files", [])

    if not git_repo:
        print(json.dumps({"error": "No git_repo specified"}))
        sys.exit(0)

    result: dict = {"versions": {}, "limiters": {}}

    # Parse each device XML version
    for ver in versions:
        version_id = ver.get("version", "unknown")
        device_xml_path = ver.get("device_xml")
        snap_path = ver.get("snap_file")

        if not device_xml_path:
            result["versions"][version_id] = {"error": "no device_xml path"}
            continue

        full_xml_path = (
            f"{input_prefix}/{device_xml_path}" if input_prefix else device_xml_path
        )

        try:
            xml_bytes = git_show(git_repo, full_xml_path)
            parsed = parse_device_xml(xml_bytes)

            # Parse snap file if provided
            if snap_path:
                full_snap_path = (
                    f"{input_prefix}/{snap_path}" if input_prefix else snap_path
                )
                try:
                    snap_bytes = git_show(git_repo, full_snap_path)
                    enabled, disabled = parse_snap_file(snap_bytes)
                    parsed["enabled_probes"] = enabled
                    parsed["disabled_probes"] = disabled
                except Exception as e:
                    parsed["snap_error"] = str(e)[:200]

            result["versions"][version_id] = parsed
        except Exception as e:
            result["versions"][version_id] = {"error": str(e)[:300]}

    # Parse limiter contour files
    for lim in limiter_files:
        name = lim.get("name", "unknown")
        file_path = lim.get("file")
        if not file_path:
            continue

        full_path = f"{input_prefix}/{file_path}" if input_prefix else file_path
        try:
            data = git_show(git_repo, full_path)
            result["limiters"][name] = parse_limiter_file(data)
        except Exception as e:
            result["limiters"][name] = {"error": str(e)[:200]}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
