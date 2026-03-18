"""Device XML scanner plugin for EFIT-format geometry files.

Parses device XML files from a git bare repo to discover machine description
geometry: PF coils, passive structures, magnetic probes, flux loops, circuits,
and limiter contours. Creates DataSource, SignalEpoch, SignalNode, and
FacilitySignal graph nodes.

Config key: data_systems.device_xml
Facility: JET (may be extended to other EFIT-based facilities)
"""

from __future__ import annotations

import abc
import asyncio
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
    DataSourceType,
    FacilitySignal,
    FacilitySignalStatus,
    IngestionStatus,
    SignalNodeSource,
    SignalNodeType,
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


# =========================================================================
# Static Source Handler Base Class
# =========================================================================


class StaticSourceHandler(abc.ABC):
    """Base class for static data source handlers within DeviceXMLScanner.

    Provides shared scaffold for DataSource creation, SignalNode batch
    persistence, FacilitySignal deduplication, and SignalEpoch creation.
    Each handler overrides ``persist()`` to implement handler-specific
    graph persistence logic.
    """

    source_name: str
    """Config key (name) used to look up this source in static_sources."""

    config_key: str
    """Key in the facility config to match by name."""

    remote_script: str | None = None
    """Remote parse script filename (e.g., 'parse_jec2020.py'). None for local-only."""

    timeout: int = 120
    """SSH script timeout in seconds."""

    needs_ssh: bool = True
    """Whether this handler requires SSH execution."""

    # -- Shared persistence helpers --

    @staticmethod
    def ensure_data_source(
        gc: GraphClient,
        facility: str,
        name: str,
        source_type: str,
        source_format: str,
        description: str,
    ) -> None:
        """Create or update a DataSource node with AT_FACILITY relationship."""
        gc.query(
            """
            MERGE (ds:DataSource {id: $facility + ':' + $name})
            ON CREATE SET
                ds.name = $name,
                ds.facility_id = $facility,
                ds.source_type = $source_type,
                ds.source_format = $source_format,
                ds.description = $description,
                ds.shot_dependent = false
            WITH ds
            MATCH (f:Facility {id: $facility})
            MERGE (ds)-[:AT_FACILITY]->(f)
            """,
            name=name,
            facility=facility,
            source_type=source_type,
            source_format=source_format,
            description=description,
        )

    @staticmethod
    def persist_signal_nodes(
        gc: GraphClient,
        nodes: list[dict],
        batch_size: int = 100,
    ) -> int:
        """Batch create SignalNode records, assigning id=path."""
        if not nodes:
            return 0
        for dn in nodes:
            dn["id"] = dn["path"]
        gc.create_nodes("SignalNode", nodes, batch_size=batch_size)
        return len(nodes)

    @staticmethod
    def persist_facility_signals(
        gc: GraphClient,
        signals: dict[str, FacilitySignal],
        batch_size: int = 100,
    ) -> int:
        """Batch create FacilitySignal from dedup dict."""
        if not signals:
            return 0
        signal_dicts = [s.model_dump(exclude_none=True) for s in signals.values()]
        gc.create_nodes("FacilitySignal", signal_dicts, batch_size=batch_size)
        return len(signal_dicts)

    @staticmethod
    def persist_epochs(
        gc: GraphClient,
        epoch_records: list[dict],
    ) -> int:
        """Batch create SignalEpoch with AT_FACILITY + IN_DATA_SOURCE."""
        if not epoch_records:
            return 0
        gc.query(
            """
            UNWIND $records AS rec
            MERGE (se:SignalEpoch {id: rec.id})
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
            MERGE (ds:DataSource {id: rec.facility_id + ':' + rec.data_source_name})
            MERGE (se)-[:IN_DATA_SOURCE]->(ds)
            """,
            records=epoch_records,
        )
        return len(epoch_records)

    @staticmethod
    def persist_introduced_in(
        gc: GraphClient,
        data_nodes: list[dict],
        epoch_id: str,
    ) -> None:
        """Create INTRODUCED_IN relationships (SignalNode → SignalEpoch)."""
        if not data_nodes:
            return
        intro_records = [{"id": dn["id"], "epoch_id": epoch_id} for dn in data_nodes]
        gc.query(
            """
            UNWIND $records AS rec
            MATCH (dn:SignalNode {id: rec.id})
            MATCH (se:SignalEpoch {id: rec.epoch_id})
            MERGE (dn)-[:INTRODUCED_IN]->(se)
            """,
            records=intro_records,
        )

    def lookup_config(self, facility: str) -> dict[str, Any] | None:
        """Look up this handler's config from facility static_sources."""
        from imas_codex.discovery.base.facility import get_facility

        facility_config = get_facility(facility)
        static_sources = facility_config.get("data_systems", {}).get(
            "static_sources", []
        )
        for source in static_sources:
            if isinstance(source, dict) and source.get("name") == self.config_key:
                return source
        return None

    async def run(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Template: lookup config → run remote → persist → return stats."""
        source_config = self.lookup_config(facility)
        if not source_config:
            return None

        parsed = await self.fetch(facility, ssh_host, source_config)
        if parsed is None:
            return None

        try:
            stats = await asyncio.to_thread(
                self.persist, facility, source_config, parsed
            )
        except Exception as e:
            logger.error(
                "%s graph persist failed for %s: %s",
                self.source_name,
                facility,
                e,
            )
            return {"error": str(e)}

        self.log_stats(facility, stats)
        return stats

    async def fetch(
        self,
        facility: str,
        ssh_host: str,
        source_config: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Run remote parse script and return parsed JSON. Override for custom fetch."""
        if not self.remote_script:
            return {}

        script_input = self.build_script_input(source_config)
        if not script_input:
            return None

        logger.info(
            "%s: processing for %s",
            self.source_name,
            facility,
        )

        try:
            output = await asyncio.to_thread(
                run_python_script,
                self.remote_script,
                script_input,
                ssh_host=ssh_host,
                timeout=self.timeout,
            )
            return json.loads(output.strip().split("\n")[-1])
        except Exception as e:
            logger.error(
                "%s parse failed for %s: %s",
                self.source_name,
                facility,
                e,
            )
            return None

    def build_script_input(
        self, source_config: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build input dict for remote script. Override for custom inputs."""
        base_dir = source_config.get("base_dir", "")
        if not base_dir:
            return None
        return {"base_dir": base_dir}

    @abc.abstractmethod
    def persist(
        self,
        facility: str,
        source_config: dict[str, Any],
        parsed: dict[str, Any],
    ) -> dict[str, Any]:
        """Handler-specific graph persistence. Must be overridden."""
        ...

    def log_stats(self, facility: str, stats: dict[str, Any]) -> None:
        """Log handler results. Override for custom logging."""
        logger.info(
            "%s scanner %s: %s",
            self.source_name,
            facility,
            stats,
        )


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
    """Persist DataSource, SignalEpoch, SignalNode, and FacilitySignal to graph."""
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
            MERGE (ds:DataSource {id: $facility + ':' + $name})
            ON CREATE SET
                ds.name = $name,
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

        # 2. Create SignalEpoch nodes
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

            # Provenance: record source file paths
            if vc.get("device_xml"):
                rec["device_xml_path"] = vc["device_xml"]
            if vc.get("snap_file"):
                rec["snap_file_path"] = vc["snap_file"]

            # PF configuration tracking
            if vc.get("pf_configuration"):
                rec["pf_configuration"] = vc["pf_configuration"]

            # Probe enable/disable from snap file parsing
            parsed = parsed_versions.get(version)
            if parsed and "error" not in parsed:
                enabled = parsed.get("enabled_probes", [])
                disabled = parsed.get("disabled_probes", [])
                if enabled:
                    rec["probes_enabled"] = enabled
                if disabled:
                    rec["probes_disabled"] = disabled

            epoch_records.append(rec)

        if epoch_records:
            gc.query(
                """
                UNWIND $records AS rec
                MERGE (se:SignalEpoch {id: rec.id})
                SET se.facility_id = rec.facility_id,
                    se.data_source_name = rec.data_source_name,
                    se.version = rec.version,
                    se.first_shot = rec.first_shot,
                    se.last_shot = rec.last_shot,
                    se.description = rec.description,
                    se.status = rec.status
                SET se.wall_configuration = rec.wall_configuration
                SET se.uses_limiter = rec.uses_limiter
                SET se.device_xml_path = rec.device_xml_path
                SET se.snap_file_path = rec.snap_file_path
                SET se.pf_configuration = rec.pf_configuration
                SET se.probes_enabled = rec.probes_enabled
                SET se.probes_disabled = rec.probes_disabled
                WITH se, rec
                MATCH (f:Facility {id: rec.facility_id})
                MERGE (se)-[:AT_FACILITY]->(f)
                WITH se, rec
                MERGE (ds:DataSource {id: rec.facility_id + ':' + rec.data_source_name})
                MERGE (se)-[:IN_DATA_SOURCE]->(ds)
                """,
                records=epoch_records,
            )
            stats["epochs"] = len(epoch_records)

        # 3. Create SignalNode and FacilitySignal nodes per version
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
                        "node_type": SignalNodeType.NUMERIC.value,
                        "source": SignalNodeSource.introspection.value,
                        "description": ", ".join(desc_parts),
                        "introduced_version": epoch_id,
                        "system": meta.get("system", ""),
                        "first_shot": vc.get("first_shot"),
                        "file_source": "git",
                        "file_path": xml_file,
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
                                unit=field_meta.get("unit"),
                                description=(
                                    f"{field_meta['desc']} of {meta['label']} {inst_id}"
                                ),
                                discovery_source="device_xml",
                                is_static=True,
                                hardware_section=section,
                            )

            # Batch create DataNodes
            if data_nodes:
                for dn in data_nodes:
                    dn["id"] = dn["path"]
                gc.create_nodes("SignalNode", data_nodes, batch_size=100)
                stats["data_nodes"] += len(data_nodes)

                # Create INTRODUCED_IN relationships (SignalNode → SignalEpoch)
                intro_records = [
                    {"id": dn["id"], "epoch_id": epoch_id} for dn in data_nodes
                ]
                gc.query(
                    """
                    UNWIND $records AS rec
                    MATCH (dn:SignalNode {id: rec.id})
                    MATCH (se:SignalEpoch {id: rec.epoch_id})
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
                "node_type": SignalNodeType.NUMERIC.value,
                "source": SignalNodeSource.introspection.value,
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
                    discovery_source="device_xml",
                    is_static=True,
                    hardware_section="limiter",
                )

        if limiter_nodes:
            for dn in limiter_nodes:
                dn["id"] = dn["path"]
            gc.create_nodes("SignalNode", limiter_nodes, batch_size=50)
            stats["limiter_nodes"] = len(limiter_nodes)

        # 4b. Create USES_LIMITER relationships (SignalEpoch → limiter SignalNode)
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
                MATCH (se:SignalEpoch {id: rec.epoch_id})
                MATCH (dn:SignalNode {id: rec.limiter_path})
                MERGE (se)-[:USES_LIMITER]->(dn)
                """,
                records=uses_limiter_records,
            )

        # 5. Persist all signals
        if all_signals:
            signal_dicts = [
                s.model_dump(exclude_none=True) for s in all_signals.values()
            ]
            gc.create_nodes("FacilitySignal", signal_dicts, batch_size=100)
            stats["signals"] = len(signal_dicts)

        # 6. Persist DataAccess node
        da = _build_data_access(facility, config)
        gc.create_nodes("DataAccess", [da.model_dump(exclude_none=True)])

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


class JEC2020Handler(StaticSourceHandler):
    """Handler for JEC2020 EFIT++ geometry data."""

    source_name = "jec2020_geometry"
    config_key = "jec2020_geometry"
    remote_script = "parse_jec2020.py"
    timeout = 120

    def build_script_input(
        self, source_config: dict[str, Any]
    ) -> dict[str, Any] | None:
        base_dir = source_config.get("base_dir", "")
        files = source_config.get("files", [])
        if not base_dir or not files:
            return None
        return {
            "base_dir": base_dir,
            "files": [{"path": f["path"], "role": f["role"]} for f in files],
        }

    def persist(
        self,
        facility: str,
        source_config: dict[str, Any],
        parsed: dict[str, Any],
    ) -> dict[str, int]:
        return _persist_jec2020_nodes(facility, source_config, parsed)

    def log_stats(self, facility: str, stats: dict[str, Any]) -> None:
        logger.info(
            "JEC2020 scanner %s: %d probes, %d flux loops, %d PF coils, "
            "%d iron segments, %d limiter points",
            facility,
            stats.get("probes", 0),
            stats.get("flux_loops", 0),
            stats.get("pf_coils", 0),
            stats.get("iron_segments", 0),
            stats.get("limiter_points", 0),
        )


def _persist_jec2020_nodes(
    facility: str,
    source_config: dict[str, Any],
    parsed: dict[str, dict],
) -> dict[str, int]:
    """Persist JEC2020 data into the graph.

    Creates DataSource, SignalNode, and FacilitySignal nodes for magnetics,
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
            MERGE (ds:DataSource {id: $facility + ':' + $name})
            ON CREATE SET
                ds.name = $name,
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
                    "node_type": SignalNodeType.NUMERIC.value,
                    "source": SignalNodeSource.introspection.value,
                    "system": JEC2020_SYSTEM_MAP["magnetics_probe"],
                    "first_shot": reference_shot,
                    "description": ", ".join(desc_parts),
                    "file_source": "filesystem",
                    "file_path": f"{base_dir}/magnetics.xml",
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
                        is_static=True,
                        hardware_section="magprobes",
                    )

            for dn in probe_nodes:
                dn["id"] = dn["path"]
            gc.create_nodes("SignalNode", probe_nodes, batch_size=100)
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
                    "node_type": SignalNodeType.NUMERIC.value,
                    "source": SignalNodeSource.introspection.value,
                    "system": JEC2020_SYSTEM_MAP["magnetics_flux"],
                    "first_shot": reference_shot,
                    "description": f"Flux loop {fid}: {fl.get('description', '')}",
                    "file_source": "filesystem",
                    "file_path": f"{base_dir}/magnetics.xml",
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
                        is_static=True,
                        hardware_section="flux",
                    )

            for dn in loop_nodes:
                dn["id"] = dn["path"]
            gc.create_nodes("SignalNode", loop_nodes, batch_size=100)
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
                    "node_type": SignalNodeType.NUMERIC.value,
                    "source": SignalNodeSource.introspection.value,
                    "system": JEC2020_SYSTEM_MAP["pf_coils"],
                    "first_shot": reference_shot,
                    "description": f"PF coil {cid} ({cname})",
                    "file_source": "filesystem",
                    "file_path": f"{base_dir}/pfSystems.xml",
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
                        is_static=True,
                        hardware_section="pfcoils",
                    )

            for dn in coil_nodes:
                dn["id"] = dn["path"]
            gc.create_nodes("SignalNode", coil_nodes, batch_size=100)
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
                    "node_type": SignalNodeType.NUMERIC.value,
                    "source": SignalNodeSource.introspection.value,
                    "system": JEC2020_SYSTEM_MAP["pf_circuits"],
                    "first_shot": reference_shot,
                    "description": f"PF circuit {cid} ({cname})",
                    "file_source": "filesystem",
                    "file_path": f"{base_dir}/pfSystems.xml",
                }
                if circ.get("coil_ids"):
                    dn["coil_ids"] = circ["coil_ids"]

                circuit_nodes.append(dn)

            for dn in circuit_nodes:
                dn["id"] = dn["path"]
            gc.create_nodes("SignalNode", circuit_nodes, batch_size=50)
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
                    MATCH (circuit:SignalNode {id: rec.circuit_path})
                    MATCH (coil:SignalNode {id: rec.coil_path})
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
                    "node_type": SignalNodeType.NUMERIC.value,
                    "source": SignalNodeSource.introspection.value,
                    "system": JEC2020_SYSTEM_MAP["iron_core"],
                    "first_shot": reference_shot,
                    "file_source": "filesystem",
                    "file_path": f"{base_dir}/ironBoundaries3.xml",
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

                iron_node["id"] = iron_node["path"]
                gc.create_nodes("SignalNode", [iron_node])
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
                    is_static=True,
                    hardware_section="iron_core",
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
                    "node_type": SignalNodeType.NUMERIC.value,
                    "source": SignalNodeSource.introspection.value,
                    "system": JEC2020_SYSTEM_MAP["limiter"],
                    "first_shot": reference_shot,
                    "file_source": "filesystem",
                    "file_path": f"{base_dir}/limiter.xml",
                    "description": (
                        f"JEC2020 ILW first wall contour at T=200°C: "
                        f"{len(r_vals)} R,Z points"
                    ),
                    "r_contour": r_vals,
                    "z_contour": z_vals,
                    "n_points": len(r_vals),
                }
                limiter_node["id"] = limiter_node["path"]
                gc.create_nodes("SignalNode", [limiter_node])
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
                    is_static=True,
                    hardware_section="limiter",
                )

                # Create SAME_GEOMETRY link to device_xml limiter if it exists
                gc.query(
                    """
                    MATCH (jec:SignalNode {id: $jec_path})
                    MATCH (dx:SignalNode {id: $dx_path})
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
            gc.create_nodes("FacilitySignal", signal_dicts, batch_size=100)

        # 9. Persist DataAccess
        da = _build_jec2020_data_access(facility, base_dir)
        gc.create_nodes("DataAccess", [da.model_dump(exclude_none=True)])

    stats["signals"] = len(all_signals)
    return stats


# =========================================================================
# MCFG Sensor Calibration Ingestion
# =========================================================================


class MCFGHandler(StaticSourceHandler):
    """Handler for MCFG sensor calibration data."""

    source_name = "sensor_calibration"
    config_key = "sensor_calibration"
    remote_script = "parse_mcfg_sensors.py"
    timeout = 120

    def build_script_input(
        self, source_config: dict[str, Any]
    ) -> dict[str, Any] | None:
        base_dir = source_config.get("base_dir", "")
        files = source_config.get("files", [])
        if not base_dir or not files:
            return None
        script_input: dict[str, Any] = {
            "base_dir": base_dir,
            "files": [{"path": f["path"], "role": f["role"]} for f in files],
        }
        # Forward versioned sensor files for historical tracking
        sensor_versions = source_config.get("sensor_versions", [])
        if sensor_versions:
            script_input["sensor_versions"] = [
                {"file": sv["file"], "path": sv["path"], "date": sv["date"]}
                for sv in sensor_versions
            ]
        return script_input

    def persist(
        self,
        facility: str,
        source_config: dict[str, Any],
        parsed: dict[str, Any],
    ) -> dict[str, int]:
        return _persist_mcfg_nodes(facility, source_config, parsed)

    def log_stats(self, facility: str, stats: dict[str, Any]) -> None:
        logger.info(
            "MCFG scanner %s: %d coil sensors, %d hall probes, "
            "%d calibration epochs, %d sensor versions",
            facility,
            stats.get("coil_sensors", 0),
            stats.get("hall_probes", 0),
            stats.get("calibration_epochs", 0),
            stats.get("sensor_versions", 0),
        )


def _persist_mcfg_nodes(
    facility: str,
    source_config: dict[str, Any],
    parsed: dict[str, dict],
) -> dict[str, int]:
    """Persist MCFG sensor positions and calibration data.

    Creates DataSource, SignalNode for each sensor, and cross-references
    to JEC2020 probe nodes via MATCHES_SENSOR relationships.
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
            MERGE (ds:DataSource {id: $facility + ':' + $name})
            ON CREATE SET
                ds.name = $name,
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
                    "node_type": SignalNodeType.NUMERIC.value,
                    "source": SignalNodeSource.introspection.value,
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

            for dn in coil_nodes:
                dn["id"] = dn["path"]
            gc.create_nodes("SignalNode", coil_nodes, batch_size=100)
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
                    "node_type": SignalNodeType.NUMERIC.value,
                    "source": SignalNodeSource.introspection.value,
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

            for dn in hall_nodes:
                dn["id"] = dn["path"]
            gc.create_nodes("SignalNode", hall_nodes, batch_size=50)
            stats["hall_probes"] = len(hall_nodes)

        # 4. Cross-reference MCFG ↔ JEC2020 sensors by R,Z proximity
        if coils:
            gc.query(
                """
                MATCH (mcfg:SignalNode)
                WHERE mcfg.data_source_name = $mcfg_source AND mcfg.facility_id = $facility
                MATCH (jec:SignalNode)
                WHERE jec.data_source_name = $jec_source AND jec.facility_id = $facility
                  AND jec.system = 'MP'
                  AND abs(mcfg.r - jec.r) < 0.001
                  AND abs(mcfg.z - jec.z) < 0.001
                MERGE (mcfg)-[:MATCHES_SENSOR]->(jec)
                """,
                mcfg_source=source_name,
                jec_source="jec2020_geometry",
                facility=facility,
            )

        # 5. Calibration epochs (store as properties on DataSource)
        cal_data = parsed.get("calibration_index", {})
        cal_epochs = cal_data.get("epochs", [])
        if cal_epochs:
            # Store aggregate epoch count on DataSource
            gc.query(
                """
                MATCH (ds:DataSource {id: $facility + ':' + $name})
                SET ds.calibration_epoch_count = $count,
                    ds.first_calibration_shot = $first_shot,
                    ds.last_calibration_shot = $last_shot
                """,
                name=source_name,
                facility=facility,
                count=len(cal_epochs),
                first_shot=cal_epochs[0]["first_shot"],
                last_shot=cal_epochs[-1]["first_shot"],
            )

            # Persist individual CalibrationEpoch nodes
            epoch_nodes = []
            for epoch in cal_epochs:
                config_id = epoch.get("config_id", "")
                epoch_id = f"{facility}:{source_name}:{config_id}"
                epoch_nodes.append(
                    {
                        "id": epoch_id,
                        "facility_id": facility,
                        "data_source_name": f"{facility}:{source_name}",
                        "date": epoch.get("date", ""),
                        "first_shot": epoch["first_shot"],
                        "config_id": config_id,
                        "config_type": epoch.get("config_type", ""),
                        "user": epoch.get("user", ""),
                        "description": epoch.get("description", ""),
                    }
                )

            gc.query(
                """
                UNWIND $records AS rec
                MERGE (ce:CalibrationEpoch {id: rec.id})
                SET ce.facility_id = rec.facility_id,
                    ce.data_source_name = rec.data_source_name,
                    ce.date = rec.date,
                    ce.first_shot = rec.first_shot,
                    ce.config_id = rec.config_id,
                    ce.config_type = rec.config_type,
                    ce.user = rec.user,
                    ce.description = rec.description
                WITH ce, rec
                MATCH (f:Facility {id: rec.facility_id})
                MERGE (ce)-[:AT_FACILITY]->(f)
                WITH ce, rec
                MATCH (ds:DataSource {id: rec.data_source_name})
                MERGE (ce)-[:IN_DATA_SOURCE]->(ds)
                """,
                records=epoch_nodes,
            )
            stats["calibration_epochs"] = len(cal_epochs)

        # 6. Versioned sensor files — persist each version's sensors
        # with version_date tags and create SUPERSEDES chains
        sensor_versions = parsed.get("sensor_versions", [])
        version_data_source_ids: list[str] = []
        if sensor_versions:
            for sv in sensor_versions:
                date = sv.get("date", "")
                filename = sv.get("file", "")
                sensors_data = sv.get("sensors")
                if sv.get("error") or not sensors_data:
                    continue

                version_ds_name = f"{source_name}:v{date}"
                version_ds_id = f"{facility}:{version_ds_name}"
                version_data_source_ids.append(version_ds_id)

                # Create a DataSource for this sensor version
                gc.query(
                    """
                    MERGE (ds:DataSource {id: $ds_id})
                    ON CREATE SET
                        ds.name = $ds_name,
                        ds.facility_id = $facility,
                        ds.source_type = $source_type,
                        ds.source_format = $source_format,
                        ds.description = $description,
                        ds.shot_dependent = false
                    WITH ds
                    MATCH (f:Facility {id: $facility})
                    MERGE (ds)-[:AT_FACILITY]->(f)
                    """,
                    ds_id=version_ds_id,
                    ds_name=version_ds_name,
                    facility=facility,
                    source_type=DataSourceType.config_file.value,
                    source_format="text",
                    description=(
                        f"MCFG sensor positions version {date} from file {filename}"
                    ),
                )

                # Persist coil sensors for this version
                v_coils = sensors_data.get("coils", [])
                if v_coils:
                    v_coil_nodes: list[dict] = []
                    for sensor in v_coils:
                        jpf_name = sensor.get("jpf_name", "")
                        node_path = f"{facility}:mcfg:sensor:{jpf_name}:v{date}"
                        dn = {
                            "path": node_path,
                            "id": node_path,
                            "data_source_name": version_ds_name,
                            "facility_id": facility,
                            "node_type": SignalNodeType.NUMERIC.value,
                            "source": SignalNodeSource.introspection.value,
                            "system": "MP",
                            "r": sensor["r"],
                            "z": sensor["z"],
                            "angle": sensor.get("angle", 0.0),
                            "gain": sensor.get("gain", 1.0),
                            "jpf_name": jpf_name,
                            "version_date": date,
                            "description": (
                                f"MCFG sensor {jpf_name} (v{date}): "
                                f"R={sensor['r']}m, Z={sensor['z']}m"
                            ),
                        }
                        v_coil_nodes.append(dn)

                    gc.create_nodes("SignalNode", v_coil_nodes, batch_size=100)

                # Persist hall probes for this version
                v_halls = sensors_data.get("hall_probes", [])
                if v_halls:
                    v_hall_nodes: list[dict] = []
                    for sensor in v_halls:
                        jpf_name = sensor.get("jpf_name", "")
                        node_path = f"{facility}:mcfg:hall:{jpf_name}:v{date}"
                        dn = {
                            "path": node_path,
                            "id": node_path,
                            "data_source_name": version_ds_name,
                            "facility_id": facility,
                            "node_type": SignalNodeType.NUMERIC.value,
                            "source": SignalNodeSource.introspection.value,
                            "system": "MP",
                            "r": sensor["r"],
                            "z": sensor["z"],
                            "gain": sensor.get("gain", 1.0),
                            "jpf_name": jpf_name,
                            "version_date": date,
                            "description": (
                                f"MCFG hall probe {jpf_name} (v{date}): "
                                f"R={sensor['r']}m, Z={sensor['z']}m"
                            ),
                        }
                        v_hall_nodes.append(dn)

                    gc.create_nodes("SignalNode", v_hall_nodes, batch_size=50)

            stats["sensor_versions"] = len(
                [sv for sv in sensor_versions if not sv.get("error")]
            )

        # 7. Create SUPERSEDES chain between versioned sensor DataSources
        # (newer version supersedes older version, sorted by date)
        if len(version_data_source_ids) >= 2:
            supersedes_records = []
            for i in range(1, len(version_data_source_ids)):
                supersedes_records.append(
                    {
                        "newer_id": version_data_source_ids[i],
                        "older_id": version_data_source_ids[i - 1],
                    }
                )
            gc.query(
                """
                UNWIND $records AS rec
                MATCH (newer:DataSource {id: rec.newer_id})
                MATCH (older:DataSource {id: rec.older_id})
                MERGE (newer)-[:SUPERSEDES]->(older)
                """,
                records=supersedes_records,
            )

    return stats


# =========================================================================
# =========================================================================
# Magnetics PPF Sensor Configuration Ingestion
# =========================================================================

# Map magnetics sensor types to IMAS IDS paths
MAGNETICS_SENSOR_IDS_MAP: dict[str, str] = {
    "BPOL": "magnetics.bpol_probe",
    "FLUX": "magnetics.flux_loop",
    "SADX": "magnetics.flux_loop",
    "BSAD": "magnetics.flux_loop",
    "TPC": "magnetics.bpol_probe",
    "TNC": "magnetics.bpol_probe",
    "PC": "magnetics.bpol_probe",
    "TS": "magnetics.flux_loop",
    "TSFL": "magnetics.flux_loop",
    "TSRR": "magnetics.flux_loop",
    "IC": "magnetics.bpol_probe",
    "UPC": "magnetics.bpol_probe",
    "UNC": "magnetics.bpol_probe",
    "DVC": "magnetics.diamagnetic_flux",
    "ITOR": "magnetics.ip",
    "IPLA": "magnetics.ip",
    "XTOR": "magnetics.bpol_probe",
    "XPOL": "magnetics.bpol_probe",
    "XNOR": "magnetics.bpol_probe",
    "VL": "magnetics.flux_loop",
    "IEXR": "magnetics.ip",
}

# System code for magnetics config sensors
MAGNETICS_SYSTEM_MAP: dict[str, str] = {
    "BPOL": "MP",
    "FLUX": "FL",
    "SADX": "FL",
    "BSAD": "FL",
    "TPC": "MP",
    "TNC": "MP",
    "PC": "MP",
    "TS": "FL",
    "TSFL": "FL",
    "TSRR": "FL",
    "IC": "MP",
    "UPC": "MP",
    "UNC": "MP",
    "DVC": "DV",
    "ITOR": "IT",
    "IPLA": "IP",
    "XTOR": "MP",
    "XPOL": "MP",
    "XNOR": "MP",
    "VL": "FL",
    "IEXR": "IP",
}


def _build_magnetics_config_data_access(facility: str, config_dir: str) -> DataAccess:
    """Build DataAccess node for magnetics config file access."""
    return DataAccess(
        id=f"{facility}:magnetics_config:file",
        facility_id=facility,
        name="JET Magnetics Sensor Configuration",
        method_type="magnetics_config",
        library="text",
        access_type="local",
        data_source=config_dir,
        imports_template="# Plain text config files — no library needed",
        connection_template=f"# Config files at {config_dir}/",
        data_template=(
            "# Parse indexr for shot→config mapping\n"
            "# Each config file defines sensors with JPF, PPF, R, Z, angle"
        ),
        full_example=(
            "# Magnetics sensor config file format:\n"
            "# Line 1: 'JPF_ADDRESS' 'PPF_SIGNAL INDEX'  idx cal1 cal2 R Z angle\n"
            "# Line 2: gain1 gain2 gain3 weight gain4 flag1 flag2 gain5 flag3\n"
            "#\n"
            f"# Index: {config_dir}/indexr\n"
            "# Maps shot ranges to config files (e.g., shots 1-27968 → limves)\n"
            "#\n"
            "# Each config file defines the complete set of magnetic sensors\n"
            "# available for a range of JET pulses."
        ),
    )


class MagneticsConfigHandler(StaticSourceHandler):
    """Handler for magnetics PPF sensor configuration data."""

    source_name = "magnetics_config"
    config_key = "magnetics_config"
    remote_script = "parse_magnetics_config.py"
    timeout = 120

    def build_script_input(
        self, source_config: dict[str, Any]
    ) -> dict[str, Any] | None:
        base_dir = source_config.get("base_dir", "")
        if not base_dir:
            return None
        return {
            "config_dir": base_dir,
            "index_file": source_config.get("index_file", ""),
        }

    def persist(
        self,
        facility: str,
        source_config: dict[str, Any],
        parsed: dict[str, Any],
    ) -> dict[str, int]:
        return _persist_magnetics_config_nodes(facility, source_config, parsed)

    def log_stats(self, facility: str, stats: dict[str, Any]) -> None:
        logger.info(
            "Magnetics config scanner %s: %d epochs, %d data nodes, %d signals",
            facility,
            stats.get("epochs", 0),
            stats.get("data_nodes", 0),
            stats.get("signals", 0),
        )


def _persist_magnetics_config_nodes(
    facility: str,
    source_config: dict[str, Any],
    parsed: dict[str, Any],
) -> dict[str, int]:
    """Persist magnetics config sensor data into the graph.

    Creates DataSource, SignalEpoch (per config epoch), SignalNode
    (per sensor per epoch), and FacilitySignal (deduplicated across epochs).
    Shot-range boundaries come from the indexr file.
    """
    source_name = source_config.get("name", "magnetics_config")
    base_dir = source_config.get("base_dir", "")
    data_access_id = f"{facility}:magnetics_config:file"

    stats: dict[str, int] = {
        "epochs": 0,
        "data_nodes": 0,
        "signals": 0,
    }

    index_entries = parsed.get("index", [])
    configs = parsed.get("configs", {})

    if not index_entries:
        logger.warning("magnetics_config: no index entries parsed")
        return stats

    # Deduplicate signals across all config epochs
    all_signals: dict[str, FacilitySignal] = {}

    with GraphClient() as gc:
        # 1. Create DataSource
        gc.query(
            """
            MERGE (ds:DataSource {id: $facility + ':' + $name})
            ON CREATE SET
                ds.name = $name,
                ds.facility_id = $facility,
                ds.source_type = $source_type,
                ds.source_format = 'text',
                ds.description = $description,
                ds.shot_dependent = false
            WITH ds
            MATCH (f:Facility {id: $facility})
            MERGE (ds)-[:AT_FACILITY]->(f)
            """,
            name=source_name,
            facility=facility,
            source_type=DataSourceType.config_file.value,
            description=(
                "JET magnetics PPF sensor configuration: shot-range to sensor "
                "set mapping with JPF addresses, PPF signals, geometry (R, Z, "
                f"angle), and calibration. Files at {base_dir}."
            ),
        )

        # 2. Create SignalEpoch nodes for each config epoch
        epoch_records = []
        for i, entry in enumerate(index_entries):
            config_name = entry["config_file"]
            epoch_id = f"{facility}:magnetics_config:{config_name}"

            config_data = configs.get(config_name, {})
            total_sensors = config_data.get("total_sensors", 0)
            sensor_types = list(config_data.get("sensor_counts", {}).keys())

            rec = {
                "id": epoch_id,
                "facility_id": facility,
                "data_source_name": source_name,
                "version": i + 1,
                "first_shot": entry["first_shot"],
                "last_shot": entry["last_shot"],
                "description": (
                    f"Magnetics config '{config_name}': "
                    f"{total_sensors} sensors ({', '.join(sensor_types[:5])}...), "
                    f"shots {entry['first_shot']}–{entry['last_shot']}"
                ),
                "status": IngestionStatus.ingested.value,
            }
            epoch_records.append(rec)

        if epoch_records:
            gc.query(
                """
                UNWIND $records AS rec
                MERGE (se:SignalEpoch {id: rec.id})
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
                MERGE (ds:DataSource {id: rec.facility_id + ':' + rec.data_source_name})
                MERGE (se)-[:IN_DATA_SOURCE]->(ds)
                """,
                records=epoch_records,
            )
            stats["epochs"] = len(epoch_records)

        # 3. Create SignalNode per sensor per config epoch
        for entry in index_entries:
            config_name = entry["config_file"]
            config_data = configs.get(config_name, {})
            if "error" in config_data or not config_data.get("sensors"):
                continue

            epoch_id = f"{facility}:magnetics_config:{config_name}"
            file_path = config_data.get("file_path", f"{base_dir}/{config_name}")
            data_nodes: list[dict] = []

            for sensor in config_data["sensors"]:
                sensor_type = sensor.get("sensor_type", "")
                ppf_signal = sensor.get("ppf_signal", "")
                ppf_index = sensor.get("ppf_index", 0)
                jpf_address = sensor.get("jpf_address", "")

                # Node path: unique per sensor per config epoch
                node_path = (
                    f"{facility}:magnetics_config:{config_name}:"
                    f"{sensor_type}:{ppf_index}"
                )

                system = MAGNETICS_SYSTEM_MAP.get(sensor_type, "MP")

                desc_parts = [
                    f"{sensor_type} sensor #{ppf_index}",
                    f"JPF={jpf_address}",
                    f"PPF={ppf_signal}/{ppf_index}",
                ]
                if sensor.get("r"):
                    desc_parts.append(f"R={sensor['r']:.4f}m")
                if sensor.get("z"):
                    desc_parts.append(f"Z={sensor['z']:.4f}m")

                dn: dict[str, Any] = {
                    "path": node_path,
                    "data_source_name": source_name,
                    "facility_id": facility,
                    "node_type": SignalNodeType.NUMERIC.value,
                    "source": SignalNodeSource.introspection.value,
                    "system": system,
                    "first_shot": entry["first_shot"],
                    "last_shot": entry["last_shot"],
                    "description": ", ".join(desc_parts),
                    "introduced_version": epoch_id,
                    "file_source": "filesystem",
                    "file_path": file_path,
                    "sensor_type": sensor_type,
                    "jpf_address": jpf_address,
                    "ppf_signal": ppf_signal,
                    "ppf_index": ppf_index,
                }

                # Geometry
                for field in ("r", "z", "angle"):
                    val = sensor.get(field)
                    if val is not None:
                        dn[field] = val

                # Calibration
                for field in ("cal1", "cal2"):
                    val = sensor.get(field)
                    if val is not None:
                        dn[field] = val

                # Gains
                for field in ("gain1", "gain2", "gain3", "gain4", "gain5", "weight"):
                    val = sensor.get(field)
                    if val is not None:
                        dn[field] = val

                data_nodes.append(dn)

                # Create FacilitySignal (deduplicated across epochs by sensor identity)
                sig_id = (
                    f"{facility}:magnetic_field_diagnostics/"
                    f"magnetics_config_{sensor_type}_{ppf_index}"
                )
                if sig_id not in all_signals:
                    all_signals[sig_id] = FacilitySignal(
                        id=sig_id,
                        facility_id=facility,
                        status=FacilitySignalStatus.discovered,
                        physics_domain="magnetic_field_diagnostics",
                        name=f"Magnetics {sensor_type} #{ppf_index} ({jpf_address})",
                        accessor=f"magnetics_config:{sensor_type}/{ppf_index}",
                        data_access=data_access_id,
                        data_source_name=source_name,
                        data_source_path=f"{sensor_type}/{ppf_index}",
                        data_source_node=node_path,
                        unit="T" if sensor_type in ("BPOL", "TPC", "TNC") else "Wb",
                        description=(
                            f"Magnetics sensor {sensor_type} #{ppf_index}: "
                            f"JPF={jpf_address}, PPF={ppf_signal}/{ppf_index}"
                        ),
                        discovery_source="magnetics_config",
                        is_static=True,
                        hardware_section=sensor_type.lower(),
                    )

            # Batch create DataNodes
            if data_nodes:
                for dn in data_nodes:
                    dn["id"] = dn["path"]
                gc.create_nodes("SignalNode", data_nodes, batch_size=200)
                stats["data_nodes"] += len(data_nodes)

                # Create INTRODUCED_IN relationships
                intro_records = [
                    {"id": dn["id"], "epoch_id": epoch_id} for dn in data_nodes
                ]
                gc.query(
                    """
                    UNWIND $records AS rec
                    MATCH (dn:SignalNode {id: rec.id})
                    MATCH (se:SignalEpoch {id: rec.epoch_id})
                    MERGE (dn)-[:INTRODUCED_IN]->(se)
                    """,
                    records=intro_records,
                )

        # 4. Persist signals
        if all_signals:
            signal_dicts = [
                s.model_dump(exclude_none=True) for s in all_signals.values()
            ]
            gc.create_nodes("FacilitySignal", signal_dicts, batch_size=100)
            stats["signals"] = len(signal_dicts)

        # 5. Persist DataAccess
        da = _build_magnetics_config_data_access(facility, base_dir)
        gc.create_nodes("DataAccess", [da.model_dump(exclude_none=True)])

        # 6. Cross-reference magnetics_config sensors ↔ JEC2020 probes by R,Z
        # Only for the latest config epoch (2002_01) which overlaps with JEC2020
        gc.query(
            """
            MATCH (mc:SignalNode)
            WHERE mc.data_source_name = $mc_source
              AND mc.facility_id = $facility
              AND mc.sensor_type = 'BPOL'
            MATCH (jec:SignalNode)
            WHERE jec.data_source_name = $jec_source
              AND jec.facility_id = $facility
              AND jec.system = 'MP'
              AND abs(mc.r - jec.r) < 0.01
              AND abs(mc.z - jec.z) < 0.01
            MERGE (mc)-[:MATCHES_SENSOR]->(jec)
            """,
            mc_source=source_name,
            jec_source="jec2020_geometry",
            facility=facility,
        )

        # 7. Cross-reference magnetics_config ↔ device_xml probes by R,Z
        gc.query(
            """
            MATCH (mc:SignalNode)
            WHERE mc.data_source_name = $mc_source
              AND mc.facility_id = $facility
              AND mc.sensor_type = 'BPOL'
            MATCH (dx:SignalNode)
            WHERE dx.data_source_name = 'device_xml'
              AND dx.facility_id = $facility
              AND dx.system = 'MP'
              AND abs(mc.r - dx.r) < 0.01
              AND abs(mc.z - dx.z) < 0.01
            MERGE (mc)-[:MATCHES_SENSOR]->(dx)
            """,
            mc_source=source_name,
            facility=facility,
        )

    return stats


# =========================================================================
# PF Coil Circuit Turns Ingestion
# =========================================================================


class PFCoilTurnsHandler(StaticSourceHandler):
    """Handler for PF coil circuit turns data."""

    source_name = "pf_coil_turns"
    config_key = "pf_coil_turns"
    remote_script = "parse_pf_coil_turns.py"
    timeout = 60

    def persist(
        self,
        facility: str,
        source_config: dict[str, Any],
        parsed: dict[str, Any],
    ) -> dict[str, int]:
        return _persist_pf_coil_turns_nodes(facility, source_config, parsed)

    def log_stats(self, facility: str, stats: dict[str, Any]) -> None:
        logger.info(
            "PF coil turns scanner %s: %d coil entries",
            facility,
            stats.get("coil_entries", 0),
        )


def _persist_pf_coil_turns_nodes(
    facility: str,
    source_config: dict[str, Any],
    parsed: dict[str, Any],
) -> dict[str, int]:
    """Persist PF coil circuit turns data into the graph.

    Creates DataSource and SignalNode records for PF coil turns ratios,
    and links them to existing device_xml PF coil/circuit DataNodes.
    """
    source_name = source_config.get("name", "pf_coil_turns")
    base_dir = source_config.get("base_dir", "")

    stats: dict[str, int] = {
        "coil_entries": 0,
        "cross_references": 0,
    }

    coils = parsed.get("coils", [])
    if not coils:
        return stats

    with GraphClient() as gc:
        # 1. Create DataSource
        gc.query(
            """
            MERGE (ds:DataSource {id: $facility + ':' + $name})
            ON CREATE SET
                ds.name = $name,
                ds.facility_id = $facility,
                ds.source_type = $source_type,
                ds.source_format = 'text',
                ds.description = $description,
                ds.shot_dependent = false
            WITH ds
            MATCH (f:Facility {id: $facility})
            MERGE (ds)-[:AT_FACILITY]->(f)
            """,
            name=source_name,
            facility=facility,
            source_type=DataSourceType.config_file.value,
            description=(
                "PF coil circuit turns ratios — maps PF coil names to turns "
                f"counts and circuit assignments. File at {base_dir}/cturns."
            ),
        )

        # 2. Create SignalNode per coil entry
        data_nodes: list[dict] = []
        for coil in coils:
            coil_name = coil.get("name", "")
            node_path = f"{facility}:pf_coil_turns:{coil_name}"

            dn: dict[str, Any] = {
                "path": node_path,
                "data_source_name": source_name,
                "facility_id": facility,
                "node_type": SignalNodeType.NUMERIC.value,
                "source": SignalNodeSource.introspection.value,
                "system": "PF",
                "file_source": "filesystem",
                "file_path": f"{base_dir}/cturns",
                "description": (f"PF coil {coil_name}: {coil.get('turns', '?')} turns"),
            }
            for field in ("turns", "circuit", "polarity", "resistance"):
                val = coil.get(field)
                if val is not None:
                    dn[field] = val

            data_nodes.append(dn)

        if data_nodes:
            for dn in data_nodes:
                dn["id"] = dn["path"]
            gc.create_nodes("SignalNode", data_nodes, batch_size=50)
            stats["coil_entries"] = len(data_nodes)

        # 3. Cross-reference to JEC2020 PF circuits by name prefix.
        # cturns names encode circuit identity: P2* → circuit 2 (PFX),
        # P3* → circuit 3 (SHAPE), PF4* → circuit 5 (P4).
        circuit_map = [
            {"prefix": "P2", "circuit_id": f"{facility}:jec2020:pf_circuit:2"},
            {"prefix": "P3", "circuit_id": f"{facility}:jec2020:pf_circuit:3"},
            {"prefix": "PF4", "circuit_id": f"{facility}:jec2020:pf_circuit:5"},
        ]
        result = gc.query(
            """
            UNWIND $mappings AS m
            MATCH (ct:SignalNode)
            WHERE ct.data_source_name = $ct_source
              AND ct.facility_id = $facility
            WITH ct, m, split(ct.path, ':')[-1] AS coil_name
            WHERE coil_name STARTS WITH m.prefix
            MATCH (circ:SignalNode {id: m.circuit_id})
            MERGE (ct)-[:SAME_COMPONENT]->(circ)
            RETURN count(*) AS refs_created
            """,
            mappings=circuit_map,
            ct_source=source_name,
            facility=facility,
        )
        if result:
            stats["cross_references"] = result[0].get("refs_created", 0)

    return stats


# =========================================================================
# Greens Table Version Mapping Ingestion
# =========================================================================


class GreensTableHandler(StaticSourceHandler):
    """Handler for Green's table version-to-shot mapping."""

    source_name = "greens_table"
    config_key = "greens_table"
    remote_script = "parse_greens_table.py"
    timeout = 60

    def persist(
        self,
        facility: str,
        source_config: dict[str, Any],
        parsed: dict[str, Any],
    ) -> dict[str, int]:
        return _persist_greens_table_nodes(facility, source_config, parsed)

    def log_stats(self, facility: str, stats: dict[str, Any]) -> None:
        logger.info(
            "Greens table scanner %s: %d versions",
            facility,
            stats.get("versions", 0),
        )


def _persist_greens_table_nodes(
    facility: str,
    source_config: dict[str, Any],
    parsed: dict[str, Any],
) -> dict[str, int]:
    """Persist Greens table version-to-shot mapping.

    Creates DataSource and SignalNode records for each Green's table version,
    linking shot ranges to specific pre-computed Green's function tables
    used by EFIT equilibrium reconstruction.
    """
    source_name = source_config.get("name", "greens_table")
    base_dir = source_config.get("base_dir", "")

    stats: dict[str, int] = {
        "versions": 0,
    }

    entries = parsed.get("entries", [])
    if not entries:
        return stats

    with GraphClient() as gc:
        # 1. Create DataSource
        gc.query(
            """
            MERGE (ds:DataSource {id: $facility + ':' + $name})
            ON CREATE SET
                ds.name = $name,
                ds.facility_id = $facility,
                ds.source_type = $source_type,
                ds.source_format = 'text',
                ds.description = $description,
                ds.shot_dependent = false
            WITH ds
            MATCH (f:Facility {id: $facility})
            MERGE (ds)-[:AT_FACILITY]->(f)
            """,
            name=source_name,
            facility=facility,
            source_type=DataSourceType.config_file.value,
            description=(
                "Green's function table version-to-shot mapping for EFIT. "
                "Each version is a precomputed table of Green's functions "
                "for the magnetic equilibrium solve. "
                f"Index at {base_dir}/green_list."
            ),
        )

        # 2. Create SignalNode per Greens table version
        data_nodes: list[dict] = []
        for entry in entries:
            version_name = entry.get("version", "")
            node_path = f"{facility}:greens_table:{version_name}"

            dn: dict[str, Any] = {
                "path": node_path,
                "data_source_name": source_name,
                "facility_id": facility,
                "node_type": SignalNodeType.NUMERIC.value,
                "source": SignalNodeSource.introspection.value,
                "system": "GR",
                "first_shot": entry.get("first_shot"),
                "last_shot": entry.get("last_shot"),
                "file_source": "filesystem",
                "file_path": f"{base_dir}/green_list",
            }
            if entry.get("greens_dir"):
                dn["greens_dir"] = entry["greens_dir"]
                dn["description"] = (
                    f"Greens table '{version_name}': shots "
                    f"{entry.get('first_shot', '?')}–{entry.get('last_shot', '?')}, "
                    f"dir={entry['greens_dir']}"
                )
            else:
                dn["description"] = (
                    f"Greens table '{version_name}': shots "
                    f"{entry.get('first_shot', '?')}–{entry.get('last_shot', '?')}"
                )

            data_nodes.append(dn)

        if data_nodes:
            for dn in data_nodes:
                dn["id"] = dn["path"]
            gc.create_nodes("SignalNode", data_nodes, batch_size=50)
            stats["versions"] = len(data_nodes)

        # 3. Link Greens table versions to SignalEpoch nodes by shot overlap
        gc.query(
            """
            MATCH (gt:SignalNode)
            WHERE gt.data_source_name = $gt_source AND gt.facility_id = $facility
            MATCH (se:SignalEpoch)
            WHERE se.facility_id = $facility
              AND se.data_source_name = 'device_xml'
              AND se.first_shot >= gt.first_shot
              AND (gt.last_shot IS NULL OR se.first_shot <= gt.last_shot)
            MERGE (se)-[:USES_GREENS]->(gt)
            """,
            gt_source=source_name,
            facility=facility,
        )

    return stats


# =========================================================================
# PPF Static Geometry Signal Ingestion
# =========================================================================

# Map PPF static signals to their corresponding device_xml SignalNode paths.
# EFIT/RLIM and EFIT/ZLIM both reference the Mk2ILW limiter contour.
PPF_GEOMETRY_CROSSREFS: dict[str, str] = {
    "EFIT/RLIM": "jet:device_xml:limiter:Mk2ILW",
    "EFIT/ZLIM": "jet:device_xml:limiter:Mk2ILW",
}


class PPFStaticHandler(StaticSourceHandler):
    """Handler for PPF static geometry signals."""

    source_name = "ppf_static"
    config_key = "ppf_static"  # Not used — custom lookup
    remote_script = None
    needs_ssh = False

    def lookup_config(self, facility: str) -> dict[str, Any] | None:
        from imas_codex.discovery.base.facility import get_facility

        facility_config = get_facility(facility)
        ppf_config = facility_config.get("data_systems", {}).get("ppf", {})
        static_signals = ppf_config.get("static_signals", [])
        if not static_signals:
            return None
        has_static = any(s.get("static", False) for s in static_signals)
        if not has_static:
            return None
        return ppf_config

    async def run(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        ppf_config = self.lookup_config(facility)
        if not ppf_config:
            return None

        logger.info(
            "device_xml scanner: processing %d PPF static signals for %s",
            sum(1 for s in ppf_config.get("static_signals", []) if s.get("static")),
            facility,
        )

        try:
            stats = await asyncio.to_thread(
                _persist_ppf_static_nodes, facility, ppf_config
            )
        except Exception as e:
            logger.error("PPF static graph persist failed for %s: %s", facility, e)
            return {"error": str(e)}

        self.log_stats(facility, stats)
        return stats

    def persist(
        self,
        facility: str,
        source_config: dict[str, Any],
        parsed: dict[str, Any],
    ) -> dict[str, int]:
        # Not used — overridden run() handles everything
        return _persist_ppf_static_nodes(facility, source_config)

    def log_stats(self, facility: str, stats: dict[str, Any]) -> None:
        logger.info(
            "PPF static scanner %s: %d DataAccess nodes, %d cross-references",
            facility,
            stats.get("data_access_nodes", 0),
            stats.get("cross_references", 0),
        )


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
            MERGE (ds:DataSource {id: $facility + ':' + $name})
            ON CREATE SET
                ds.name = $name,
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

        gc.create_nodes("DataAccess", da_nodes, batch_size=50)
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
                    MATCH (dn:SignalNode {id: $dx_path})
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
    limiter contours. Also processes static source handlers for
    next-generation geometry data. Creates graph nodes directly.

    Config (data_systems.device_xml):
        git_repo: str - Path to bare git repo
        input_prefix: str - Tree path prefix in git repo
        versions: list - Device geometry versions with pulse ranges
        limiter_versions: list - First-wall contour versions
        systems: list - Named subsystems (informational)
    """

    scanner_type: str = "device_xml"

    # Registered static source handlers
    handlers: list[StaticSourceHandler] = [
        JEC2020Handler(),
        MCFGHandler(),
        PPFStaticHandler(),
        MagneticsConfigHandler(),
        PFCoilTurnsHandler(),
        GreensTableHandler(),
    ]

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover geometry signals from device XML files.

        Runs parse_device_xml.py remotely to extract geometry from git,
        then persists DataSource, SignalEpoch, SignalNode, and
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
        default_limiter_dir = config.get("limiter_dir")
        limiter_files = []
        for lv in limiter_versions:
            if lv.get("file"):
                entry: dict[str, Any] = {
                    "name": lv["name"],
                    "file": lv["file"],
                }
                source_dir = lv.get("source_dir") or default_limiter_dir
                if source_dir:
                    entry["source_dir"] = source_dir
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
            output = await asyncio.to_thread(
                run_python_script,
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

        # Validate n_points from config against parsed limiter data
        for lv in limiter_versions:
            name = lv.get("name", "")
            expected_n = lv.get("n_points")
            if expected_n and name in parsed_limiters:
                parsed_data = parsed_limiters[name]
                actual_n = parsed_data.get("n_points", 0)
                if actual_n and actual_n != expected_n:
                    logger.warning(
                        "Limiter '%s': expected %d points, got %d",
                        name,
                        expected_n,
                        actual_n,
                    )

        # Check for parse errors
        errors = {
            v: data.get("error")
            for v, data in parsed_versions.items()
            if "error" in data
        }
        if errors:
            logger.warning("device_xml parse errors: %s", errors)

        # Persist to graph (non-fatal — static source scans should still run)
        try:
            stats = await asyncio.to_thread(
                _persist_graph_nodes,
                facility,
                config,
                parsed_versions,
                parsed_limiters,
            )
            stats["parse_errors"] = errors

            logger.info(
                "device_xml scanner %s: %d epochs, %d nodes, %d signals, %d limiters",
                facility,
                stats["epochs"],
                stats["data_nodes"],
                stats["signals"],
                stats["limiter_nodes"],
            )
        except Exception as e:
            logger.error("device_xml graph persist failed for %s: %s", facility, e)
            stats = {"error": str(e), "parse_errors": errors}

        # Build DataAccess for return
        data_access = _build_data_access(facility, config)

        # Process static source handlers
        for handler in self.handlers:
            handler_stats = await handler.run(facility, ssh_host, config)
            if handler_stats:
                stats[handler.source_name] = handler_stats

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

        For static geometry data, validation checks that the SignalNode
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

                # Check SignalNode exists and has geometry values
                rows = gc.query(
                    """
                    MATCH (dn:SignalNode {id: $path})
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
                            "error": f"SignalNode not found: {dn_path}",
                        }
                    )

        return results


# Auto-register on import
register_scanner(DeviceXMLScanner())
