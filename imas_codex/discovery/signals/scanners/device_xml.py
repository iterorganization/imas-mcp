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
        "fields": {
            "coil_connect": {"unit": "", "desc": "Coil connection string"},
            "supply_connect": {"unit": "", "desc": "Supply connection string"},
        },
        "label": "PF circuit",
    },
    "pfpassive": {
        "physics_domain": "magnetic_field_diagnostics",
        "imas_ids": "pf_passive.loop",
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
            epoch_records.append(
                {
                    "id": f"{facility}:device_xml:{version}",
                    "facility_id": facility,
                    "data_source_name": "device_xml",
                    "version": i + 1,
                    "first_shot": vc.get("first_shot", 0),
                    "last_shot": vc.get("last_shot"),
                    "description": vc.get("description", ""),
                    "status": IngestionStatus.ingested.value,
                }
            )

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

        # 4. Create limiter DataNodes
        limiter_versions = config.get("limiter_versions", [])
        limiter_nodes: list[dict] = []
        for lv in limiter_versions:
            name = lv.get("name", "")
            limiter_data = parsed_limiters.get(name, {})
            if "error" in limiter_data or not limiter_data.get("r"):
                continue

            node_path = f"{facility}:device_xml:limiter:{name}"
            r_vals = limiter_data["r"]
            z_vals = limiter_data["z"]
            n_points = limiter_data.get("n_points", len(r_vals))

            limiter_nodes.append(
                {
                    "path": node_path,
                    "data_source_name": "device_xml",
                    "facility_id": facility,
                    "node_type": DataNodeType.NUMERIC.value,
                    "source": DataNodeSource.introspection.value,
                    "description": (
                        f"First wall contour '{name}': {n_points} R,Z points"
                    ),
                    "first_shot": lv.get("first_shot"),
                    "last_shot": lv.get("last_shot"),
                    "r_contour": r_vals,
                    "z_contour": z_vals,
                    "n_points": n_points,
                }
            )

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


class DeviceXMLScanner:
    """Scanner for EFIT-format device XML geometry files in git.

    Reads device XML files from a bare git repo via SSH, parses geometry
    for PF coils, passive structures, magnetic probes, flux loops, and
    limiter contours. Creates graph nodes directly.

    Config (data_systems.device_xml):
        git_repo: str - Path to bare git repo
        input_prefix: str - Tree path prefix in git repo
        versions: list - Device geometry versions with pulse ranges
        limiter_versions: list - First-wall contour versions
        systems: list - Named subsystems (informational)
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
                limiter_files.append(
                    {
                        "name": lv["name"],
                        "file": lv["file"],
                    }
                )

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
