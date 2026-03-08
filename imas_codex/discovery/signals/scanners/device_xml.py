"""Device XML scanner plugin for EFIT-format geometry files.

Parses device XML files from a git bare repo to discover machine description
geometry: PF coils, passive structures, magnetic probes, flux loops, circuits,
and limiter contours. Creates DataSource, StructuralEpoch, DataNode, and
FacilitySignal graph nodes.

Config key: data_systems.device_xml
Facility: JET (may be extended to other EFIT-based facilities)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from imas_codex.discovery.signals.scanners.base import (
    ScanResult,
    register_scanner,
)
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import (
    DataAccess,
    DataNodeSource,
    DataNodeType,
    DataSourceType,
    FacilitySignal,
    FacilitySignalStatus,
    IngestionStatus,
)
from imas_codex.remote.executor import run_python_script

logger = logging.getLogger(__name__)

# XML section -> (physics_domain, imas_ids_name, unit, description_template)
SECTION_METADATA: dict[str, dict[str, Any]] = {
    "magprobes": {
        "physics_domain": "magnetic_field_diagnostics",
        "imas_ids": "magnetics.bpol_probe",
        "system": "MP",
        "fields": {
            "r": {"unit": "m", "desc": "Radial position"},
            "z": {"unit": "m", "desc": "Vertical position"},
            "angle": {"unit": "deg", "desc": "Orientation angle"},
        },
        "label": "magnetic probe",
    },
    "flux": {
        "physics_domain": "magnetic_field_diagnostics",
        "imas_ids": "magnetics.flux_loop",
        "system": "FL",
        "fields": {
            "r": {"unit": "m", "desc": "Radial position"},
            "z": {"unit": "m", "desc": "Vertical position"},
            "dphi": {"unit": "deg", "desc": "Toroidal angle extent"},
        },
        "label": "flux loop",
    },
    "pfcoils": {
        "physics_domain": "magnetic_field_diagnostics",
        "imas_ids": "pf_active.coil",
        "system": "PF",
        "fields": {
            "r": {"unit": "m", "desc": "Radial position"},
            "z": {"unit": "m", "desc": "Vertical position"},
            "dr": {"unit": "m", "desc": "Radial width"},
            "dz": {"unit": "m", "desc": "Vertical height"},
            "turnsperelement": {"unit": "", "desc": "Turns per element"},
        },
        "label": "PF coil",
    },
    "pfcircuits": {
        "physics_domain": "magnetic_field_diagnostics",
        "imas_ids": "pf_active.circuit",
        "system": "CI",
        "fields": {
            "coil_connect": {"unit": "", "desc": "Coil connection string"},
            "supply_connect": {"unit": "", "desc": "Supply connection string"},
        },
        "label": "PF circuit",
    },
    "pfpassive": {
        "physics_domain": "magnetic_field_diagnostics",
        "imas_ids": "pf_passive.loop",
        "system": "PS",
        "fields": {
            "r": {"unit": "m", "desc": "Radial position"},
            "z": {"unit": "m", "desc": "Vertical position"},
            "dr": {"unit": "m", "desc": "Radial width"},
            "dz": {"unit": "m", "desc": "Vertical height"},
            "ang1": {"unit": "deg", "desc": "Start angle"},
            "ang2": {"unit": "deg", "desc": "End angle"},
            "resistance": {"unit": "ohm", "desc": "Electrical resistance"},
        },
        "label": "passive structure element",
    },
}


def _make_signal_name(section: str, instance_id: str, field: str) -> str:
    """Create a normalized signal name from section/instance/field."""
    section_labels = {
        "magprobes": "bpme",
        "flux": "flux_loop",
        "pfcoils": "pf_coil",
        "pfcircuits": "pf_circuit",
        "pfpassive": "passive",
    }
    prefix = section_labels.get(section, section)
    return f"{prefix}_{instance_id}_{field}"


def _make_signal_id(facility: str, section: str, instance_id: str, field: str) -> str:
    """Create signal ID following the convention facility:domain/signal_name."""
    meta = SECTION_METADATA.get(section, {})
    domain = meta.get("physics_domain", "magnetic_field_diagnostics")
    name = _make_signal_name(section, instance_id, field)
    return f"{facility}:{domain}/{name}"


def _build_data_access(facility: str, config: dict[str, Any]) -> DataAccess:
    """Build the DataAccess node for device XML git access."""
    git_repo = config.get("git_repo", "")
    return DataAccess(
        id=f"{facility}:device_xml:git",
        facility_id=facility,
        name="JET Device XML (git)",
        method_type="device_xml",
        library="xml.etree.ElementTree",
        access_type="local",
        data_source=git_repo,
        imports_template="import subprocess\nimport xml.etree.ElementTree as ET",
        connection_template=(
            "xml_bytes = subprocess.check_output([\n"
            "    'git', '-C', '{data_source}',\n"
            "    'show', 'HEAD:JET/input/Devices/{device_xml}'\n"
            "])\n"
            "root = ET.fromstring(xml_bytes)"
        ),
        data_template=(
            "elements = root.findall('.//{section}/instance')\n"
            "data = []\n"
            "for e in elements:\n"
            "    d = {c.tag: float(c.text) for c in e if c.text}\n"
            "    d['id'] = e.get('id')\n"
            "    data.append(d)"
        ),
        full_example=(
            "import subprocess\n"
            "import xml.etree.ElementTree as ET\n\n"
            "xml_bytes = subprocess.check_output([\n"
            f"    'git', '-C', '{git_repo}',\n"
            "    'show', 'HEAD:JET/input/Devices/device_p89440.xml'\n"
            "])\n"
            "root = ET.fromstring(xml_bytes)\n\n"
            "# Magnetic probes\n"
            "for probe in root.findall('.//magprobes/instance'):\n"
            "    r = probe.find('r').text\n"
            "    z = probe.find('z').text\n"
            "    print(f'Probe {probe.get(\"id\")}: R={r}, Z={z}')\n\n"
            "# PF coils\n"
            "for coil in root.findall('.//pfcoils/instance'):\n"
            "    r = coil.find('r').text\n"
            "    z = coil.find('z').text\n"
            "    print(f'Coil {coil.get(\"id\")}: R={r}, Z={z}')"
        ),
    )


def _persist_graph_nodes(
    facility: str,
    config: dict[str, Any],
    parsed_versions: dict[str, dict],
    parsed_limiters: dict[str, dict],
) -> dict[str, int]:
    """Persist DataSource, StructuralEpoch, DataNode, and FacilitySignal to graph."""
    versions_config = config.get("versions", [])
    data_access_id = f"{facility}:device_xml:git"

    stats = {
        "epochs": 0,
        "data_nodes": 0,
        "signals": 0,
        "limiter_nodes": 0,
    }

    # Track which XML files map to which version groups (for dedup)
    xml_to_versions: dict[str, list[str]] = {}
    for vc in versions_config:
        xml_file = vc.get("device_xml", "")
        xml_to_versions.setdefault(xml_file, []).append(vc["version"])

    # Find the canonical version for each XML (first one that has parsed data)
    xml_canonical: dict[str, str] = {}
    for xml_file, vers in xml_to_versions.items():
        for v in vers:
            if v in parsed_versions and "error" not in parsed_versions[v]:
                xml_canonical[xml_file] = v
                break

    # Deduplicated set of signals (across versions sharing same XML)
    all_signals: dict[str, FacilitySignal] = {}

    with GraphClient() as gc:
        # 1. Create DataSource
        gc.query(
            """
            MERGE (ds:DataSource {name: $name})
            ON CREATE SET
                ds.facility_id = $facility,
                ds.source_type = $source_type,
                ds.source_format = $source_format,
                ds.description = $description,
                ds.shot_dependent = false
            WITH ds
            MATCH (f:Facility {id: $facility})
            MERGE (ds)-[:AT_FACILITY]->(f)
            """,
            name="device_xml",
            facility=facility,
            source_type=DataSourceType.xml.value,
            source_format="git_xml",
            description=(
                "JET tokamak geometry: PF coils, passive structures, "
                "magnetic probes, flux loops, circuits, and limiter contours. "
                f"Versioned in {config.get('git_repo', '')}."
            ),
        )

        # 2. Create StructuralEpoch nodes
        epoch_records = []
        for i, vc in enumerate(versions_config):
            version = vc["version"]
            rec: dict[str, Any] = {
                "id": f"{facility}:device_xml:{version}",
                "facility_id": facility,
                "data_source_name": "device_xml",
                "version": i + 1,
                "first_shot": vc.get("first_shot", 0),
                "last_shot": vc.get("last_shot"),
                "description": vc.get("description", ""),
                "status": IngestionStatus.ingested.value,
            }
            if vc.get("wall_configuration"):
                rec["wall_configuration"] = vc["wall_configuration"]
            # Store uses_limiter as property for dual property+relationship model
            lim_name = vc.get("uses_limiter")
            if lim_name:
                rec["uses_limiter"] = f"{facility}:device_xml:limiter:{lim_name}"
            epoch_records.append(rec)

        if epoch_records:
            gc.query(
                """
                UNWIND $records AS rec
                MERGE (se:StructuralEpoch {id: rec.id})
                SET se.facility_id = rec.facility_id,
                    se.data_source_name = rec.data_source_name,
                    se.version = rec.version,
                    se.first_shot = rec.first_shot,
                    se.last_shot = rec.last_shot,
                    se.description = rec.description,
                    se.status = rec.status
                SET se.wall_configuration = rec.wall_configuration
                SET se.uses_limiter = rec.uses_limiter
                WITH se, rec
                MATCH (f:Facility {id: rec.facility_id})
                MERGE (se)-[:AT_FACILITY]->(f)
                WITH se, rec
                MERGE (ds:DataSource {name: rec.data_source_name})
                MERGE (se)-[:IN_DATA_SOURCE]->(ds)
                """,
                records=epoch_records,
            )
            stats["epochs"] = len(epoch_records)

        # 3. Create DataNode and FacilitySignal nodes per version
        for vc in versions_config:
            version = vc["version"]
            xml_file = vc.get("device_xml", "")

            # Versions without device_xml (pre-EFIT++ limiter-only epochs)
            # have no DataNodes from XML parsing — skip gracefully.
            if not xml_file:
                continue

            # Use parsed data from this version or from canonical version
            parsed = parsed_versions.get(version)
            if not parsed or "error" in parsed:
                canonical = xml_canonical.get(xml_file)
                if canonical:
                    parsed = parsed_versions.get(canonical)
                if not parsed or "error" in parsed:
                    logger.warning(
                        "No parsed data for version %s (XML: %s)",
                        version,
                        xml_file,
                    )
                    continue

            epoch_id = f"{facility}:device_xml:{version}"
            data_nodes: list[dict] = []

            for section, meta in SECTION_METADATA.items():
                instances = parsed.get(section, [])
                for inst in instances:
                    inst_id = str(inst.get("id", ""))
                    if not inst_id:
                        continue

                    node_path = f"{facility}:device_xml:{version}:{section}:{inst_id}"

                    # Build description from instance attributes
                    desc_parts = [f"{meta['label'].title()} {inst_id}"]
                    for field, field_meta in meta.get("fields", {}).items():
                        val = inst.get(field)
                        if val is not None and field_meta.get("unit"):
                            if isinstance(val, list):
                                desc_parts.append(
                                    f"{field_meta['desc']}: {len(val)} elements"
                                )
                            else:
                                desc_parts.append(
                                    f"{field_meta['desc']}={val}{field_meta['unit']}"
                                )

                    dn = {
                        "path": node_path,
                        "data_source_name": "device_xml",
                        "facility_id": facility,
                        "node_type": DataNodeType.NUMERIC.value,
                        "source": DataNodeSource.introspection.value,
                        "description": ", ".join(desc_parts),
                        "introduced_version": epoch_id,
                        "system": meta.get("system", ""),
                        "first_shot": vc.get("first_shot"),
                    }
                    if vc.get("last_shot"):
                        dn["last_shot"] = vc["last_shot"]

                    # Store geometry values as node properties
                    for field in meta.get("fields", {}):
                        val = inst.get(field)
                        if val is not None:
                            dn[field] = val

                    # Store error bounds
                    for err_field in ("abs_error", "rel_error"):
                        val = inst.get(err_field)
                        if val is not None:
                            dn[err_field] = val

                    data_nodes.append(dn)

                    # Create FacilitySignal per field (deduplicated across versions)
                    for field, field_meta in meta.get("fields", {}).items():
                        val = inst.get(field)
                        if val is None:
                            continue

                        sig_id = _make_signal_id(facility, section, inst_id, field)
                        if sig_id not in all_signals:
                            all_signals[sig_id] = FacilitySignal(
                                id=sig_id,
                                facility_id=facility,
                                status=FacilitySignalStatus.discovered,
                                physics_domain=meta["physics_domain"],
                                name=f"{meta['label'].title()} {inst_id} {field_meta['desc']}",
                                accessor=f"device_xml:{section}/{inst_id}/{field}",
                                data_access=data_access_id,
                                data_source_name="device_xml",
                                data_source_path=f"{section}/{inst_id}/{field}",
                                data_source_node=node_path,
                                units=field_meta.get("unit"),
                                description=(
                                    f"{field_meta['desc']} of {meta['label']} {inst_id}"
                                ),
                                discovery_source="xml_extraction",
                            )

            # Batch create DataNodes
            if data_nodes:
                gc.create_nodes("DataNode", data_nodes, id_field="path", batch_size=100)
                stats["data_nodes"] += len(data_nodes)

                # Create INTRODUCED_IN relationships (DataNode → StructuralEpoch)
                intro_records = [
                    {"path": dn["path"], "epoch_id": epoch_id} for dn in data_nodes
                ]
                gc.query(
                    """
                    UNWIND $records AS rec
                    MATCH (dn:DataNode {path: rec.path})
                    MATCH (se:StructuralEpoch {id: rec.epoch_id})
                    MERGE (dn)-[:INTRODUCED_IN]->(se)
                    """,
                    records=intro_records,
                )

        # 4. Create limiter DataNodes
        limiter_versions = config.get("limiter_versions", [])
        limiter_nodes: list[dict] = []
        for lv in limiter_versions:
            name = lv.get("name", "")
            limiter_data = parsed_limiters.get(name, {})
            if "error" in limiter_data:
                continue

            # Select contour segments: contour_sections specifies which
            # segments (0-indexed) to concatenate. Default: segment 0 only.
            contour_sections = lv.get("contour_sections")
            segments = limiter_data.get("segments", [])
            if contour_sections and segments:
                r_vals: list[float] = []
                z_vals: list[float] = []
                for idx in contour_sections:
                    if idx < len(segments):
                        r_vals.extend(segments[idx]["r"])
                        z_vals.extend(segments[idx]["z"])
                n_points = len(r_vals)
            else:
                r_vals = limiter_data.get("r", [])
                z_vals = limiter_data.get("z", [])
                n_points = limiter_data.get("n_points", len(r_vals))

            if not r_vals:
                continue

            node_path = f"{facility}:device_xml:limiter:{name}"

            dn: dict[str, Any] = {
                "path": node_path,
                "data_source_name": "device_xml",
                "facility_id": facility,
                "node_type": DataNodeType.NUMERIC.value,
                "source": DataNodeSource.introspection.value,
                "description": (f"First wall contour '{name}': {n_points} R,Z points"),
                "first_shot": lv.get("first_shot"),
                "last_shot": lv.get("last_shot"),
                "r_contour": r_vals,
                "z_contour": z_vals,
                "n_points": n_points,
            }

            # Provenance: track where the file was read from
            if limiter_data.get("file_source"):
                dn["file_source"] = limiter_data["file_source"]
            if limiter_data.get("file_path"):
                dn["file_path"] = limiter_data["file_path"]

            limiter_nodes.append(dn)

            # Create limiter signal
            sig_id = f"{facility}:magnetic_field_diagnostics/limiter_{name.lower()}"
            if sig_id not in all_signals:
                all_signals[sig_id] = FacilitySignal(
                    id=sig_id,
                    facility_id=facility,
                    status=FacilitySignalStatus.discovered,
                    physics_domain="magnetic_field_diagnostics",
                    name=f"Limiter {name}",
                    accessor=f"device_xml:limiter/{name}",
                    data_access=data_access_id,
                    data_source_name="device_xml",
                    data_source_path=f"limiter/{name}",
                    data_source_node=node_path,
                    description=f"First wall R,Z contour for {name} limiter configuration",
                    discovery_source="xml_extraction",
                )

        if limiter_nodes:
            gc.create_nodes("DataNode", limiter_nodes, id_field="path", batch_size=50)
            stats["limiter_nodes"] = len(limiter_nodes)

        # 4b. Create USES_LIMITER relationships (StructuralEpoch → limiter DataNode)
        # Each config version specifies its limiter via:
        #   1. uses_limiter: explicit limiter version name (pre-EFIT++ epochs)
        #   2. limiter: file path → matched to limiter_version by basename
        limiter_file_to_name: dict[str, str] = {}
        for lv in limiter_versions:
            if lv.get("file"):
                basename = lv["file"].rsplit("/", 1)[-1]
                limiter_file_to_name[basename] = lv["name"]

        uses_limiter_records = []
        for vc in versions_config:
            # Direct name reference takes priority
            lim_name = vc.get("uses_limiter")
            if not lim_name:
                # Fall back to file-based matching
                limiter_file = vc.get("limiter", "")
                if not limiter_file:
                    continue
                basename = limiter_file.rsplit("/", 1)[-1]
                lim_name = limiter_file_to_name.get(basename)
                # Try stripping _cc suffix (git vs filesystem naming convention)
                if not lim_name and basename.endswith("_cc"):
                    lim_name = limiter_file_to_name.get(basename[:-3])
            if lim_name:
                uses_limiter_records.append(
                    {
                        "epoch_id": f"{facility}:device_xml:{vc['version']}",
                        "limiter_path": f"{facility}:device_xml:limiter:{lim_name}",
                    }
                )

        if uses_limiter_records:
            gc.query(
                """
                UNWIND $records AS rec
                MATCH (se:StructuralEpoch {id: rec.epoch_id})
                MATCH (dn:DataNode {path: rec.limiter_path})
                MERGE (se)-[:USES_LIMITER]->(dn)
                """,
                records=uses_limiter_records,
            )

        # 5. Persist all signals
        if all_signals:
            signal_dicts = [
                s.model_dump(exclude_none=True) for s in all_signals.values()
            ]
            gc.create_nodes(
                "FacilitySignal", signal_dicts, id_field="id", batch_size=100
            )
            stats["signals"] = len(signal_dicts)

        # 6. Persist DataAccess node
        da = _build_data_access(facility, config)
        gc.create_nodes("DataAccess", [da.model_dump(exclude_none=True)], id_field="id")

    return stats


# =========================================================================
# JEC2020 Static Source Ingestion
# =========================================================================

# Physics domain → system code mapping for JEC2020 data
JEC2020_SYSTEM_MAP = {
    "magnetics_probe": "MP",
    "magnetics_flux": "FL",
    "pf_coils": "PF",
    "pf_circuits": "CI",
    "iron_core": "FE",
    "limiter": "LIM",
}


def _build_jec2020_data_access(facility: str, base_dir: str) -> DataAccess:
    """Build DataAccess node for JEC2020 filesystem XML access."""
    return DataAccess(
        id=f"{facility}:jec2020:xml",
        facility_id=facility,
        name="JEC2020 EFIT++ Geometry (XML)",
        method_type="static_xml",
        library="xml.etree.ElementTree",
        access_type="local",
        data_source="jec2020_geometry",
        imports_template="import xml.etree.ElementTree as ET",
        connection_template=(
            f"with open('{base_dir}/magnetics.xml', 'rb') as f:\n"
            "    root = ET.fromstring(f.read())"
        ),
        data_template=(
            "probes = root.iter('magneticProbe')\n"
            "for p in probes:\n"
            "    geom = p.find('geometry')\n"
            "    r = float(geom.get('rCentre'))\n"
            "    z = float(geom.get('zCentre'))"
        ),
        full_example=(
            "import xml.etree.ElementTree as ET\n\n"
            f"with open('{base_dir}/magnetics.xml', 'rb') as f:\n"
            "    root = ET.fromstring(f.read())\n\n"
            "for probe in root.iter('magneticProbe'):\n"
            "    geom = probe.find('geometry')\n"
            "    tt = probe.find('timeTrace')\n"
            "    print(f'Probe {probe.get(\"id\")}: '\n"
            '          f\'R={geom.get("rCentre")}, Z={geom.get("zCentre")}\'\n'
            "          f' PPF={tt.get(\"signalName\")}'\n"
            "          f' JPF={tt.get(\"signalName2\")}')"
        ),
    )


def _persist_jec2020_nodes(
    facility: str,
    source_config: dict[str, Any],
    parsed: dict[str, dict],
) -> dict[str, int]:
    """Persist JEC2020 data into the graph.

    Creates DataSource, DataNode, and FacilitySignal nodes for magnetics,
    PF systems, iron core boundaries, and high-res limiter contour.
    """
    source_name = source_config.get("name", "jec2020_geometry")
    base_dir = source_config.get("base_dir", "")
    reference_shot = source_config.get("reference_shot", 0)
    data_access_id = f"{facility}:jec2020:xml"

    stats: dict[str, int] = {
        "probes": 0,
        "flux_loops": 0,
        "pf_coils": 0,
        "pf_circuits": 0,
        "iron_segments": 0,
        "limiter_points": 0,
    }
    all_signals: dict[str, FacilitySignal] = {}

    with GraphClient() as gc:
        # 1. Create DataSource
        gc.query(
            """
            MERGE (ds:DataSource {name: $name})
            ON CREATE SET
                ds.facility_id = $facility,
                ds.source_type = $source_type,
                ds.source_format = $source_format,
                ds.description = $description,
                ds.shot_dependent = false
            WITH ds
            MATCH (f:Facility {id: $facility})
            MERGE (ds)-[:AT_FACILITY]->(f)
            """,
            name=source_name,
            facility=facility,
            source_type=DataSourceType.xml.value,
            source_format="jec2020_xml",
            description=(
                "JEC2020 EFIT++ geometry: magnetics probes, flux loops, "
                "PF coils/circuits, iron core boundary, and high-resolution "
                f"ILW limiter contour. Files at {base_dir}."
            ),
        )

        # 2. Magnetics probes
        magnetics = parsed.get("magnetics", {})
        probes = magnetics.get("probes", [])
        if probes:
            probe_nodes: list[dict] = []
            for p in probes:
                pid = p.get("id", "")
                node_path = f"{facility}:jec2020:probe:{pid}"
                desc_parts = [f"Magnetic probe {pid}"]
                if p.get("description"):
                    desc_parts.append(p["description"])

                dn: dict[str, Any] = {
                    "path": node_path,
                    "data_source_name": source_name,
                    "facility_id": facility,
                    "node_type": DataNodeType.NUMERIC.value,
                    "source": DataNodeSource.introspection.value,
                    "system": JEC2020_SYSTEM_MAP["magnetics_probe"],
                    "first_shot": reference_shot,
                    "description": ", ".join(desc_parts),
                }
                if p.get("rCentre") is not None:
                    dn["r"] = p["rCentre"]
                if p.get("zCentre") is not None:
                    dn["z"] = p["zCentre"]
                if p.get("poloidalOrientation") is not None:
                    dn["poloidal_orientation"] = p["poloidalOrientation"]
                if p.get("ppf_signal"):
                    dn["ppf_signal"] = p["ppf_signal"]
                if p.get("jpf_signal"):
                    dn["jpf_signal"] = p["jpf_signal"]
                if p.get("ppf_data_source"):
                    dn["ppf_data_source"] = p["ppf_data_source"]
                if p.get("jpf_data_source"):
                    dn["jpf_data_source"] = p["jpf_data_source"]
                if p.get("rel_error") is not None:
                    dn["rel_error"] = p["rel_error"]
                if p.get("abs_error") is not None:
                    dn["abs_error"] = p["abs_error"]

                probe_nodes.append(dn)

                # FacilitySignal for each probe
                sig_id = f"{facility}:magnetic_field_diagnostics/jec2020_probe_{pid}"
                if sig_id not in all_signals:
                    all_signals[sig_id] = FacilitySignal(
                        id=sig_id,
                        facility_id=facility,
                        status=FacilitySignalStatus.discovered,
                        physics_domain="magnetic_field_diagnostics",
                        name=f"JEC2020 Probe {pid} ({p.get('description', '')})",
                        accessor=f"jec2020:probe/{pid}",
                        data_access=data_access_id,
                        data_source_name=source_name,
                        data_source_path=f"probe/{pid}",
                        data_source_node=node_path,
                        description=(
                            f"Magnetic probe {pid}: "
                            f"PPF={p.get('ppf_signal', '')}, "
                            f"JPF={p.get('jpf_signal', '')}"
                        ),
                        discovery_source="jec2020_xml",
                    )

            gc.create_nodes("DataNode", probe_nodes, id_field="path", batch_size=100)
            stats["probes"] = len(probe_nodes)

        # 3. Flux loops
        flux_loops = magnetics.get("flux_loops", [])
        if flux_loops:
            loop_nodes: list[dict] = []
            for fl in flux_loops:
                fid = fl.get("id", "")
                node_path = f"{facility}:jec2020:flux_loop:{fid}"

                dn = {
                    "path": node_path,
                    "data_source_name": source_name,
                    "facility_id": facility,
                    "node_type": DataNodeType.NUMERIC.value,
                    "source": DataNodeSource.introspection.value,
                    "system": JEC2020_SYSTEM_MAP["magnetics_flux"],
                    "first_shot": reference_shot,
                    "description": f"Flux loop {fid}: {fl.get('description', '')}",
                }
                if fl.get("rCentre") is not None:
                    dn["r"] = fl["rCentre"]
                if fl.get("zCentre") is not None:
                    dn["z"] = fl["zCentre"]
                if fl.get("ppf_signal"):
                    dn["ppf_signal"] = fl["ppf_signal"]
                if fl.get("jpf_signal"):
                    dn["jpf_signal"] = fl["jpf_signal"]

                loop_nodes.append(dn)

                sig_id = (
                    f"{facility}:magnetic_field_diagnostics/jec2020_flux_loop_{fid}"
                )
                if sig_id not in all_signals:
                    all_signals[sig_id] = FacilitySignal(
                        id=sig_id,
                        facility_id=facility,
                        status=FacilitySignalStatus.discovered,
                        physics_domain="magnetic_field_diagnostics",
                        name=f"JEC2020 Flux Loop {fid}",
                        accessor=f"jec2020:flux_loop/{fid}",
                        data_access=data_access_id,
                        data_source_name=source_name,
                        data_source_path=f"flux_loop/{fid}",
                        data_source_node=node_path,
                        description=f"Flux loop {fid}: {fl.get('description', '')}",
                        discovery_source="jec2020_xml",
                    )

            gc.create_nodes("DataNode", loop_nodes, id_field="path", batch_size=100)
            stats["flux_loops"] = len(loop_nodes)

        # 4. PF coils
        pf_data = parsed.get("pf_coils", {})
        pf_coils = pf_data.get("coils", [])
        if pf_coils:
            coil_nodes: list[dict] = []
            for coil in pf_coils:
                cid = coil.get("id", "")
                cname = coil.get("name", "")
                node_path = f"{facility}:jec2020:pf_coil:{cid}"

                dn = {
                    "path": node_path,
                    "data_source_name": source_name,
                    "facility_id": facility,
                    "node_type": DataNodeType.NUMERIC.value,
                    "source": DataNodeSource.introspection.value,
                    "system": JEC2020_SYSTEM_MAP["pf_coils"],
                    "first_shot": reference_shot,
                    "description": f"PF coil {cid} ({cname})",
                }
                # Single-value geometry
                for attr in (
                    "rCentre",
                    "zCentre",
                    "dR",
                    "dZ",
                    "angle1",
                    "angle2",
                    "turnCount",
                ):
                    val = coil.get(attr)
                    if val is not None:
                        if isinstance(val, list):
                            # Multi-element coil — store as JSON array
                            dn[f"{attr}_array"] = val
                            dn[attr] = val[0]  # Store first element as scalar
                        else:
                            dn[attr] = val

                coil_nodes.append(dn)

                sig_id = f"{facility}:magnetic_field_diagnostics/jec2020_pf_coil_{cid}"
                if sig_id not in all_signals:
                    all_signals[sig_id] = FacilitySignal(
                        id=sig_id,
                        facility_id=facility,
                        status=FacilitySignalStatus.discovered,
                        physics_domain="magnetic_field_diagnostics",
                        name=f"JEC2020 PF Coil {cid} ({cname})",
                        accessor=f"jec2020:pf_coil/{cid}",
                        data_access=data_access_id,
                        data_source_name=source_name,
                        data_source_path=f"pf_coil/{cid}",
                        data_source_node=node_path,
                        description=f"PF coil {cid} ({cname})",
                        discovery_source="jec2020_xml",
                    )

            gc.create_nodes("DataNode", coil_nodes, id_field="path", batch_size=100)
            stats["pf_coils"] = len(coil_nodes)

        # 5. PF circuits
        pf_circuits = pf_data.get("circuits", [])
        if pf_circuits:
            circuit_nodes: list[dict] = []
            for circ in pf_circuits:
                cid = circ.get("id", "")
                cname = circ.get("name", "")
                node_path = f"{facility}:jec2020:pf_circuit:{cid}"

                dn = {
                    "path": node_path,
                    "data_source_name": source_name,
                    "facility_id": facility,
                    "node_type": DataNodeType.NUMERIC.value,
                    "source": DataNodeSource.introspection.value,
                    "system": JEC2020_SYSTEM_MAP["pf_circuits"],
                    "first_shot": reference_shot,
                    "description": f"PF circuit {cid} ({cname})",
                }
                if circ.get("coil_ids"):
                    dn["coil_ids"] = circ["coil_ids"]

                circuit_nodes.append(dn)

            gc.create_nodes("DataNode", circuit_nodes, id_field="path", batch_size=50)
            stats["pf_circuits"] = len(circuit_nodes)

            # Create COIL_IN_CIRCUIT relationships
            coil_circuit_records = []
            for circ in pf_circuits:
                cid = circ.get("id", "")
                for coil_id in circ.get("coil_ids", []):
                    coil_circuit_records.append(
                        {
                            "circuit_path": f"{facility}:jec2020:pf_circuit:{cid}",
                            "coil_path": f"{facility}:jec2020:pf_coil:{coil_id}",
                        }
                    )

            if coil_circuit_records:
                gc.query(
                    """
                    UNWIND $records AS rec
                    MATCH (circuit:DataNode {path: rec.circuit_path})
                    MATCH (coil:DataNode {path: rec.coil_path})
                    MERGE (coil)-[:IN_CIRCUIT]->(circuit)
                    """,
                    records=coil_circuit_records,
                )

        # 6. Iron core boundary
        iron_data = parsed.get("iron_core", {})
        if iron_data and "error" not in iron_data:
            r_vals = iron_data.get("r", [])
            z_vals = iron_data.get("z", [])
            if r_vals and z_vals:
                iron_node = {
                    "path": f"{facility}:jec2020:iron_boundary",
                    "data_source_name": source_name,
                    "facility_id": facility,
                    "node_type": DataNodeType.NUMERIC.value,
                    "source": DataNodeSource.introspection.value,
                    "system": JEC2020_SYSTEM_MAP["iron_core"],
                    "first_shot": reference_shot,
                    "description": (
                        f"Iron core boundary: {len(r_vals)} segments, "
                        f"length={iron_data.get('boundary_length', '?')}m"
                    ),
                    "r_contour": r_vals,
                    "z_contour": z_vals,
                    "n_points": len(r_vals),
                }
                if iron_data.get("permeabilities"):
                    iron_node["permeabilities"] = iron_data["permeabilities"]
                if iron_data.get("segment_lengths"):
                    iron_node["segment_lengths"] = iron_data["segment_lengths"]
                if iron_data.get("boundary_length"):
                    iron_node["boundary_length"] = iron_data["boundary_length"]

                gc.create_nodes("DataNode", [iron_node], id_field="path")
                stats["iron_segments"] = len(r_vals)

                sig_id = f"{facility}:magnetic_field_diagnostics/jec2020_iron_boundary"
                all_signals[sig_id] = FacilitySignal(
                    id=sig_id,
                    facility_id=facility,
                    status=FacilitySignalStatus.discovered,
                    physics_domain="magnetic_field_diagnostics",
                    name="JEC2020 Iron Core Boundary",
                    accessor="jec2020:iron_boundary",
                    data_access=data_access_id,
                    data_source_name=source_name,
                    data_source_path="iron_boundary",
                    data_source_node=f"{facility}:jec2020:iron_boundary",
                    description=(
                        f"Iron core boundary: {len(r_vals)} segments "
                        f"with permeabilities"
                    ),
                    discovery_source="jec2020_xml",
                )

        # 7. High-res limiter
        limiter_data = parsed.get("limiter", {})
        if limiter_data and "error" not in limiter_data:
            r_vals = limiter_data.get("r", [])
            z_vals = limiter_data.get("z", [])
            if r_vals and z_vals:
                limiter_node = {
                    "path": f"{facility}:jec2020:limiter",
                    "data_source_name": source_name,
                    "facility_id": facility,
                    "node_type": DataNodeType.NUMERIC.value,
                    "source": DataNodeSource.introspection.value,
                    "system": JEC2020_SYSTEM_MAP["limiter"],
                    "first_shot": reference_shot,
                    "description": (
                        f"JEC2020 ILW first wall contour at T=200°C: "
                        f"{len(r_vals)} R,Z points"
                    ),
                    "r_contour": r_vals,
                    "z_contour": z_vals,
                    "n_points": len(r_vals),
                }
                gc.create_nodes("DataNode", [limiter_node], id_field="path")
                stats["limiter_points"] = len(r_vals)

                sig_id = f"{facility}:magnetic_field_diagnostics/jec2020_limiter"
                all_signals[sig_id] = FacilitySignal(
                    id=sig_id,
                    facility_id=facility,
                    status=FacilitySignalStatus.discovered,
                    physics_domain="magnetic_field_diagnostics",
                    name="JEC2020 ILW Limiter Contour",
                    accessor="jec2020:limiter",
                    data_access=data_access_id,
                    data_source_name=source_name,
                    data_source_path="limiter",
                    data_source_node=f"{facility}:jec2020:limiter",
                    description=(
                        f"High-resolution ILW first wall contour at T=200°C: "
                        f"{len(r_vals)} R,Z points"
                    ),
                    discovery_source="jec2020_xml",
                )

                # Create SAME_GEOMETRY link to device_xml limiter if it exists
                gc.query(
                    """
                    MATCH (jec:DataNode {path: $jec_path})
                    MATCH (dx:DataNode {path: $dx_path})
                    MERGE (jec)-[:SAME_GEOMETRY]->(dx)
                    """,
                    jec_path=f"{facility}:jec2020:limiter",
                    dx_path=f"{facility}:device_xml:limiter:Mk2ILW",
                )

        # 8. Persist signals
        if all_signals:
            signal_dicts = [
                s.model_dump(exclude_none=True) for s in all_signals.values()
            ]
            gc.create_nodes(
                "FacilitySignal", signal_dicts, id_field="id", batch_size=100
            )

        # 9. Persist DataAccess
        da = _build_jec2020_data_access(facility, base_dir)
        gc.create_nodes("DataAccess", [da.model_dump(exclude_none=True)], id_field="id")

    stats["signals"] = len(all_signals)
    return stats


# =========================================================================
# MCFG Sensor Calibration Ingestion
# =========================================================================


def _persist_mcfg_nodes(
    facility: str,
    source_config: dict[str, Any],
    parsed: dict[str, dict],
) -> dict[str, int]:
    """Persist MCFG sensor positions and calibration data.

    Creates DataSource, DataNode for each sensor, and cross-references
    to JEC2020 probe nodes via SAME_SENSOR relationships.
    """
    source_name = source_config.get("name", "sensor_calibration")
    base_dir = source_config.get("base_dir", "")

    stats: dict[str, int] = {
        "coil_sensors": 0,
        "hall_probes": 0,
        "calibration_epochs": 0,
    }

    with GraphClient() as gc:
        # 1. Create DataSource
        gc.query(
            """
            MERGE (ds:DataSource {name: $name})
            ON CREATE SET
                ds.facility_id = $facility,
                ds.source_type = $source_type,
                ds.source_format = $source_format,
                ds.description = $description,
                ds.shot_dependent = false
            WITH ds
            MATCH (f:Facility {id: $facility})
            MERGE (ds)-[:AT_FACILITY]->(f)
            """,
            name=source_name,
            facility=facility,
            source_type=DataSourceType.xml.value,
            source_format="text",
            description=(
                "MCFG magnetics sensor positions from CATIA CAD reference and "
                f"calibration epochs. Files at {base_dir}."
            ),
        )

        # 2. Coil sensors
        sensors = parsed.get("sensors", {})
        coils = sensors.get("coils", [])
        if coils:
            coil_nodes: list[dict] = []
            for sensor in coils:
                jpf_name = sensor.get("jpf_name", "")
                node_path = f"{facility}:mcfg:sensor:{jpf_name}"

                dn: dict[str, Any] = {
                    "path": node_path,
                    "data_source_name": source_name,
                    "facility_id": facility,
                    "node_type": DataNodeType.NUMERIC.value,
                    "source": DataNodeSource.introspection.value,
                    "system": "MP",
                    "r": sensor["r"],
                    "z": sensor["z"],
                    "angle": sensor["angle"],
                    "gain": sensor["gain"],
                    "rel_error": sensor["rel_error"],
                    "abs_error": sensor["abs_error"],
                    "jpf_name": jpf_name,
                    "description": (
                        f"MCFG sensor {jpf_name}: "
                        f"R={sensor['r']}m, Z={sensor['z']}m, "
                        f"angle={sensor['angle']}°"
                    ),
                }
                if sensor.get("description"):
                    dn["sensor_description"] = sensor["description"]

                coil_nodes.append(dn)

            gc.create_nodes("DataNode", coil_nodes, id_field="path", batch_size=100)
            stats["coil_sensors"] = len(coil_nodes)

        # 3. Hall probes
        hall_probes = sensors.get("hall_probes", [])
        if hall_probes:
            hall_nodes: list[dict] = []
            for sensor in hall_probes:
                jpf_name = sensor.get("jpf_name", "")
                node_path = f"{facility}:mcfg:hall:{jpf_name}"

                dn = {
                    "path": node_path,
                    "data_source_name": source_name,
                    "facility_id": facility,
                    "node_type": DataNodeType.NUMERIC.value,
                    "source": DataNodeSource.introspection.value,
                    "system": "MP",
                    "r": sensor["r"],
                    "z": sensor["z"],
                    "gain": sensor["gain"],
                    "rel_error": sensor["rel_error"],
                    "abs_error": sensor["abs_error"],
                    "jpf_name": jpf_name,
                    "description": (
                        f"MCFG hall probe {jpf_name}: "
                        f"R={sensor['r']}m, Z={sensor['z']}m"
                    ),
                }
                hall_nodes.append(dn)

            gc.create_nodes("DataNode", hall_nodes, id_field="path", batch_size=50)
            stats["hall_probes"] = len(hall_nodes)

        # 4. Cross-reference MCFG ↔ JEC2020 sensors by R,Z proximity
        if coils:
            gc.query(
                """
                MATCH (mcfg:DataNode)
                WHERE mcfg.data_source_name = $mcfg_source AND mcfg.facility_id = $facility
                MATCH (jec:DataNode)
                WHERE jec.data_source_name = $jec_source AND jec.facility_id = $facility
                  AND jec.system = 'MP'
                  AND abs(mcfg.r - jec.r) < 0.001
                  AND abs(mcfg.z - jec.z) < 0.001
                MERGE (mcfg)-[:SAME_SENSOR]->(jec)
                """,
                mcfg_source=source_name,
                jec_source="jec2020_geometry",
                facility=facility,
            )

        # 5. Calibration epochs (store as properties on DataSource)
        cal_data = parsed.get("calibration_index", {})
        cal_epochs = cal_data.get("epochs", [])
        if cal_epochs:
            # Store epoch count on DataSource
            gc.query(
                """
                MATCH (ds:DataSource {name: $name})
                SET ds.calibration_epoch_count = $count,
                    ds.first_calibration_shot = $first_shot,
                    ds.last_calibration_shot = $last_shot
                """,
                name=source_name,
                count=len(cal_epochs),
                first_shot=cal_epochs[0]["first_shot"],
                last_shot=cal_epochs[-1]["first_shot"],
            )
            stats["calibration_epochs"] = len(cal_epochs)

    return stats


# =========================================================================
# PPF Static Geometry Signal Ingestion
# =========================================================================

# Map PPF static signals to their corresponding device_xml DataNode paths.
# EFIT/RLIM and EFIT/ZLIM both reference the Mk2ILW limiter contour.
PPF_GEOMETRY_CROSSREFS: dict[str, str] = {
    "EFIT/RLIM": "jet:device_xml:limiter:Mk2ILW",
    "EFIT/ZLIM": "jet:device_xml:limiter:Mk2ILW",
}


def _persist_ppf_static_nodes(
    facility: str,
    ppf_config: dict[str, Any],
) -> dict[str, int]:
    """Persist PPF static geometry DataAccess nodes and cross-references.

    Creates DataAccess nodes for each static PPF signal, a ppf_static
    DataSource, and ACCESSES_GEOMETRY relationships to device_xml DataNodes
    where the same geometry data exists in file-based form.
    """
    static_signals = ppf_config.get("static_signals", [])
    reference_pulse = ppf_config.get("reference_pulse", 0)
    setup_commands = ppf_config.get("setup_commands", [])

    stats: dict[str, int] = {
        "data_access_nodes": 0,
        "cross_references": 0,
    }

    # Filter to only truly static signals
    static_only = [s for s in static_signals if s.get("static", False)]
    if not static_only:
        return stats

    with GraphClient() as gc:
        # 1. Create ppf_static DataSource
        gc.query(
            """
            MERGE (ds:DataSource {name: $name})
            ON CREATE SET
                ds.facility_id = $facility,
                ds.source_type = $source_type,
                ds.source_format = 'ppf',
                ds.description = $description,
                ds.shot_dependent = false
            WITH ds
            MATCH (f:Facility {id: $facility})
            MERGE (ds)-[:AT_FACILITY]->(f)
            """,
            name="ppf_static",
            facility=facility,
            source_type=DataSourceType.ppf.value,
            description=(
                "PPF signals containing static machine description data. "
                "Accessed via MDSplus thin-client ppf() calls."
            ),
        )

        # 2. Create DataAccess nodes for each static signal
        da_nodes: list[dict] = []
        for sig in static_only:
            name = sig["name"]
            dda, dtype = name.split("/", 1)
            da_id = f"{facility}:ppf:{name}"

            da = DataAccess(
                id=da_id,
                facility_id=facility,
                name=f"PPF {name}",
                method_type="ppf",
                library="MDSplus",
                access_type="remote",
                data_source="ppf_static",
                imports_template="import MDSplus",
                connection_template=("conn = MDSplus.Connection('mdsplus.jet.uk')"),
                data_template=f"data = conn.get('ppf(\"{name}\", $)')",
                full_example=(
                    "import MDSplus\n\n"
                    "conn = MDSplus.Connection('mdsplus.jet.uk')\n"
                    f"data = conn.get('ppf(\"{name}\", {reference_pulse})')\n"
                    f"print(f'{name}: shape={{data.shape}}')"
                ),
                setup_commands=setup_commands,
            )
            da_nodes.append(da.model_dump())

        gc.create_nodes("DataAccess", da_nodes, id_field="id", batch_size=50)
        stats["data_access_nodes"] = len(da_nodes)

        # 3. Create ACCESSES_GEOMETRY cross-references to device_xml DataNodes
        for sig in static_only:
            name = sig["name"]
            da_id = f"{facility}:ppf:{name}"
            dx_path = PPF_GEOMETRY_CROSSREFS.get(name)
            if dx_path:
                gc.query(
                    """
                    MATCH (da:DataAccess {id: $da_id})
                    MATCH (dn:DataNode {path: $dx_path})
                    MERGE (da)-[:ACCESSES_GEOMETRY]->(dn)
                    """,
                    da_id=da_id,
                    dx_path=dx_path,
                )
                stats["cross_references"] += 1

    return stats


class DeviceXMLScanner:
    """Scanner for EFIT-format device XML geometry files in git.

    Reads device XML files from a bare git repo via SSH, parses geometry
    for PF coils, passive structures, magnetic probes, flux loops, and
    limiter contours. Also processes JEC2020 static XML sources for
    next-generation EFIT++ geometry. Creates graph nodes directly.

    Config (data_systems.device_xml):
        git_repo: str - Path to bare git repo
        input_prefix: str - Tree path prefix in git repo
        versions: list - Device geometry versions with pulse ranges
        limiter_versions: list - First-wall contour versions
        systems: list - Named subsystems (informational)

    Also processes data_systems.static_sources entries with name
    'jec2020_geometry' for JEC2020 XML ingestion.
    """

    scanner_type: str = "device_xml"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover geometry signals from device XML files.

        Runs parse_device_xml.py remotely to extract geometry from git,
        then persists DataSource, StructuralEpoch, DataNode, and
        FacilitySignal nodes to the graph.
        """
        git_repo = config.get("git_repo")
        input_prefix = config.get("input_prefix", "")
        versions = config.get("versions", [])
        limiter_versions = config.get("limiter_versions", [])

        if not git_repo:
            logger.error("device_xml scanner: no git_repo configured for %s", facility)
            return ScanResult(stats={"error": "no git_repo"})

        # Deduplicate device XML files to avoid re-parsing identical files
        seen_xmls: set[str] = set()
        parse_versions = []
        for vc in versions:
            xml_file = vc.get("device_xml", "")
            if not xml_file:
                continue  # Limiter-only epochs have no device XML to parse
            version = vc["version"]
            entry = {
                "version": version,
                "device_xml": xml_file,
            }
            snap_file = vc.get("snap_file")
            if snap_file:
                entry["snap_file"] = snap_file

            # Only parse XML if we haven't seen this exact file before,
            # but always include snap files (they differ per epoch)
            if xml_file not in seen_xmls:
                seen_xmls.add(xml_file)
            parse_versions.append(entry)

        # Build limiter file list
        limiter_files = []
        for lv in limiter_versions:
            if lv.get("file"):
                entry: dict[str, Any] = {
                    "name": lv["name"],
                    "file": lv["file"],
                }
                if lv.get("source_dir"):
                    entry["source_dir"] = lv["source_dir"]
                limiter_files.append(entry)

        # Run remote parse script
        script_input = {
            "git_repo": git_repo,
            "input_prefix": input_prefix,
            "versions": parse_versions,
            "limiter_files": limiter_files,
        }

        logger.info(
            "device_xml scanner: parsing %d versions, %d limiters for %s",
            len(parse_versions),
            len(limiter_files),
            facility,
        )

        try:
            output = run_python_script(
                "parse_device_xml.py",
                script_input,
                ssh_host=ssh_host,
                timeout=120,
            )

            # Parse last line as JSON (skip any logging/debug output)
            result = json.loads(output.strip().split("\n")[-1])
        except Exception as e:
            logger.error("device_xml remote parse failed for %s: %s", facility, e)
            return ScanResult(stats={"error": str(e)})

        parsed_versions = result.get("versions", {})
        parsed_limiters = result.get("limiters", {})

        # Check for parse errors
        errors = {
            v: data.get("error")
            for v, data in parsed_versions.items()
            if "error" in data
        }
        if errors:
            logger.warning("device_xml parse errors: %s", errors)

        # Persist to graph
        stats = _persist_graph_nodes(facility, config, parsed_versions, parsed_limiters)
        stats["parse_errors"] = errors

        logger.info(
            "device_xml scanner %s: %d epochs, %d nodes, %d signals, %d limiters",
            facility,
            stats["epochs"],
            stats["data_nodes"],
            stats["signals"],
            stats["limiter_nodes"],
        )

        # Build DataAccess for return
        data_access = _build_data_access(facility, config)

        # Process JEC2020 static sources if configured
        jec2020_stats = await self._scan_jec2020(facility, ssh_host, config)
        if jec2020_stats:
            stats["jec2020"] = jec2020_stats

        # Process MCFG sensor calibration if configured
        mcfg_stats = await self._scan_mcfg(facility, ssh_host, config)
        if mcfg_stats:
            stats["mcfg"] = mcfg_stats

        # Process PPF static geometry signals if configured
        ppf_stats = await self._scan_ppf_static(facility, config)
        if ppf_stats:
            stats["ppf_static"] = ppf_stats

        return ScanResult(
            signals=[],  # Signals written directly to graph
            data_access=data_access,
            metadata={
                "git_repo": git_repo,
                "distinct_xmls": len(seen_xmls),
                "versions_parsed": len(parsed_versions),
            },
            stats=stats,
        )

    async def _scan_jec2020(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
    ) -> dict[str, int] | None:
        """Scan JEC2020 static sources if configured.

        Looks for a static_sources entry named 'jec2020_geometry' in the
        facility config and processes its XML files.
        """
        from imas_codex.discovery.base.facility import get_facility

        facility_config = get_facility(facility)
        static_sources = facility_config.get("data_systems", {}).get(
            "static_sources", []
        )

        # Find JEC2020 source config
        jec2020_config = None
        for source in static_sources:
            if isinstance(source, dict) and source.get("name") == "jec2020_geometry":
                jec2020_config = source
                break

        if not jec2020_config:
            return None

        base_dir = jec2020_config.get("base_dir", "")
        files = jec2020_config.get("files", [])
        if not base_dir or not files:
            return None

        logger.info(
            "device_xml scanner: processing JEC2020 files from %s (%d files)",
            base_dir,
            len(files),
        )

        # Build script input
        script_input = {
            "base_dir": base_dir,
            "files": [{"path": f["path"], "role": f["role"]} for f in files],
        }

        try:
            output = run_python_script(
                "parse_jec2020.py",
                script_input,
                ssh_host=ssh_host,
                timeout=120,
            )
            parsed = json.loads(output.strip().split("\n")[-1])
        except Exception as e:
            logger.error("JEC2020 parse failed for %s: %s", facility, e)
            return {"error": str(e)}

        # Persist to graph
        jec_stats = _persist_jec2020_nodes(facility, jec2020_config, parsed)

        logger.info(
            "JEC2020 scanner %s: %d probes, %d flux loops, %d PF coils, "
            "%d iron segments, %d limiter points",
            facility,
            jec_stats.get("probes", 0),
            jec_stats.get("flux_loops", 0),
            jec_stats.get("pf_coils", 0),
            jec_stats.get("iron_segments", 0),
            jec_stats.get("limiter_points", 0),
        )

        return jec_stats

    async def _scan_mcfg(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
    ) -> dict[str, int] | None:
        """Scan MCFG sensor calibration static sources if configured.

        Looks for a static_sources entry named 'sensor_calibration' in the
        facility config and processes sensor position and calibration files.
        """
        from imas_codex.discovery.base.facility import get_facility

        facility_config = get_facility(facility)
        static_sources = facility_config.get("data_systems", {}).get(
            "static_sources", []
        )

        mcfg_config = None
        for source in static_sources:
            if isinstance(source, dict) and source.get("name") == "sensor_calibration":
                mcfg_config = source
                break

        if not mcfg_config:
            return None

        base_dir = mcfg_config.get("base_dir", "")
        files = mcfg_config.get("files", [])
        if not base_dir or not files:
            return None

        logger.info(
            "device_xml scanner: processing MCFG sensor files from %s (%d files)",
            base_dir,
            len(files),
        )

        script_input = {
            "base_dir": base_dir,
            "files": [{"path": f["path"], "role": f["role"]} for f in files],
        }

        try:
            output = run_python_script(
                "parse_mcfg_sensors.py",
                script_input,
                ssh_host=ssh_host,
                timeout=120,
            )
            parsed = json.loads(output.strip().split("\n")[-1])
        except Exception as e:
            logger.error("MCFG parse failed for %s: %s", facility, e)
            return {"error": str(e)}

        mcfg_stats = _persist_mcfg_nodes(facility, mcfg_config, parsed)

        logger.info(
            "MCFG scanner %s: %d coil sensors, %d hall probes, %d calibration epochs",
            facility,
            mcfg_stats.get("coil_sensors", 0),
            mcfg_stats.get("hall_probes", 0),
            mcfg_stats.get("calibration_epochs", 0),
        )

        return mcfg_stats

    async def _scan_ppf_static(
        self,
        facility: str,
        config: dict[str, Any],
    ) -> dict[str, int] | None:
        """Scan PPF static geometry signals if configured.

        Reads data_systems.ppf.static_signals[] from the facility config
        and creates DataAccess nodes with ACCESSES_GEOMETRY cross-references
        to device_xml DataNodes.
        """
        from imas_codex.discovery.base.facility import get_facility

        facility_config = get_facility(facility)
        ppf_config = facility_config.get("data_systems", {}).get("ppf", {})

        static_signals = ppf_config.get("static_signals", [])
        if not static_signals:
            return None

        # Only process if there are static signals
        has_static = any(s.get("static", False) for s in static_signals)
        if not has_static:
            return None

        logger.info(
            "device_xml scanner: processing %d PPF static signals for %s",
            sum(1 for s in static_signals if s.get("static")),
            facility,
        )

        ppf_stats = _persist_ppf_static_nodes(facility, ppf_config)

        logger.info(
            "PPF static scanner %s: %d DataAccess nodes, %d cross-references",
            facility,
            ppf_stats.get("data_access_nodes", 0),
            ppf_stats.get("cross_references", 0),
        )

        return ppf_stats

    async def check(
        self,
        facility: str,
        ssh_host: str,
        signals: list[FacilitySignal],
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> list[dict[str, Any]]:
        """Validate device XML signals.

        For static geometry data, validation checks that the DataNode
        has stored values (r, z, etc.) rather than querying live data.
        """
        results = []
        with GraphClient() as gc:
            for signal in signals:
                dn_path = signal.data_source_node
                if not dn_path:
                    results.append(
                        {
                            "signal_id": signal.id,
                            "valid": False,
                            "error": "no data_source_node",
                        }
                    )
                    continue

                # Check DataNode exists and has geometry values
                rows = gc.query(
                    """
                    MATCH (dn:DataNode {path: $path})
                    RETURN dn.r AS r, dn.z AS z, dn.path AS path
                    """,
                    path=dn_path,
                )
                if rows:
                    results.append(
                        {
                            "signal_id": signal.id,
                            "valid": True,
                            "shape": "[1]",
                            "dtype": "float64",
                        }
                    )
                else:
                    results.append(
                        {
                            "signal_id": signal.id,
                            "valid": False,
                            "error": f"DataNode not found: {dn_path}",
                        }
                    )

        return results


# Auto-register on import
register_scanner(DeviceXMLScanner())
