"""Graph operations for IDS mapping and assembly.

Provides queries to load IMASMapping nodes and their linked SignalGroups
from the graph, and functions to seed mapping definitions.

Architecture:
    IMASMapping (orchestration) -[:USES_SIGNAL_GROUP]-> SignalGroup
    SignalGroup -[:MAPS_TO_IMAS]-> IMASNode (field-level transform)
    IMASMapping -[:POPULATES]-> IMASNode (struct-array root)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


@dataclass
class FieldMapping:
    """A resolved field mapping from a MAPS_TO_IMAS relationship."""

    signal_group_id: str
    source_property: str
    target_imas_path: str
    transform_code: str = "value"
    units_in: str | None = None
    units_out: str | None = None
    cocos_source: int | None = None
    cocos_target: int | None = None
    driver: str = "device_xml"


@dataclass
class Mapping:
    """A resolved IMASMapping from graph data."""

    id: str
    facility_id: str
    ids_name: str
    dd_version: str
    provider: str | None = None
    static_config: dict[str, Any] = field(default_factory=dict)
    sections: list[dict[str, Any]] = field(default_factory=list)
    field_mappings: list[FieldMapping] = field(default_factory=list)


def load_mapping(facility: str, ids_name: str, gc: GraphClient) -> Mapping | None:
    """Load an IMASMapping and its linked SignalGroups from the graph.

    Args:
        facility: Facility ID.
        ids_name: IDS name (e.g., 'pf_active').
        gc: Graph client instance.

    Returns:
        Mapping with populated field_mappings and sections, or None if not found.
    """
    rows = list(
        gc.query(
            """
            MATCH (m:IMASMapping {facility_id: $facility, ids_name: $ids_name})
            WHERE m.status = 'active'
            RETURN m.id AS id, m.facility_id AS facility_id,
                   m.ids_name AS ids_name, m.dd_version AS dd_version,
                   m.provider AS provider, m.static_config AS static_config
            LIMIT 1
            """,
            facility=facility,
            ids_name=ids_name,
        )
    )
    if not rows:
        return None

    row = rows[0]
    config_str = row.get("static_config") or "{}"
    mapping = Mapping(
        id=row["id"],
        facility_id=row["facility_id"],
        ids_name=row["ids_name"],
        dd_version=row["dd_version"],
        provider=row.get("provider"),
        static_config=json.loads(config_str),
    )

    mapping.sections = load_sections(mapping.id, gc)
    mapping.field_mappings = load_field_mappings(mapping.id, gc)
    return mapping


def load_sections(mapping_id: str, gc: GraphClient) -> list[dict[str, Any]]:
    """Load POPULATES relationships from an IMASMapping to struct-array roots.

    Args:
        mapping_id: IMASMapping node ID.
        gc: Graph client instance.

    Returns:
        List of section dicts with root_path and assembly config properties.
    """
    return list(
        gc.query(
            """
            MATCH (m:IMASMapping {id: $mapping_id})
                  -[t:POPULATES]->(root:IMASNode)
            RETURN root.id AS root_path,
                   t.structure AS structure,
                   t.init_arrays AS init_arrays,
                   t.elements_config AS elements_config,
                   t.nested_path AS nested_path,
                   t.parent_size AS parent_size,
                   t.source_system AS source_system,
                   t.source_data_source AS source_data_source,
                   t.source_epoch_field AS source_epoch_field,
                   t.source_select_via AS source_select_via,
                   t.enrichment AS enrichment
            """,
            mapping_id=mapping_id,
        )
    )


def load_field_mappings(mapping_id: str, gc: GraphClient) -> list[FieldMapping]:
    """Load field mappings via USES_SIGNAL_GROUP → MAPS_TO_IMAS traversal.

    Args:
        mapping_id: IMASMapping node ID.
        gc: Graph client instance.

    Returns:
        List of resolved field mappings.
    """
    rows = list(
        gc.query(
            """
            MATCH (m:IMASMapping {id: $mapping_id})
                  -[:USES_SIGNAL_GROUP]->(sg:SignalGroup)
                  -[map:MAPS_TO_IMAS]->(ip:IMASNode)
            RETURN sg.id AS signal_group_id,
                   map.source_property AS source_property,
                   ip.id AS target_imas_path,
                   map.transform_code AS transform_code,
                   map.units_in AS units_in,
                   map.units_out AS units_out,
                   map.cocos_source AS cocos_source,
                   map.cocos_target AS cocos_target,
                   map.driver AS driver,
                   ip.cocos_label_transformation AS cocos_label
            """,
            mapping_id=mapping_id,
        )
    )
    mappings = []
    for row in rows:
        fm = FieldMapping(
            signal_group_id=row["signal_group_id"],
            source_property=row.get("source_property") or "value",
            target_imas_path=row["target_imas_path"],
            transform_code=row.get("transform_code") or "value",
            units_in=row.get("units_in"),
            units_out=row.get("units_out"),
            cocos_source=row.get("cocos_source"),
            cocos_target=row.get("cocos_target"),
            driver=row.get("driver") or "device_xml",
        )
        cocos_label = row.get("cocos_label")
        if cocos_label and cocos_label != "one_like" and not fm.cocos_source:
            logger.warning(
                "COCOS-sensitive path %s (label=%s) has no cocos_source on mapping %s",
                fm.target_imas_path,
                cocos_label,
                fm.signal_group_id,
            )
        mappings.append(fm)
    return mappings


def select_nodes(
    facility: str,
    section_config: dict[str, Any],
    epoch_id: str,
    gc: GraphClient,
) -> list[dict[str, Any]]:
    """Query DataNodes for a structural section of the assembly.

    Uses the section_config's source definition to find matching
    DataNodes by system, data_source, and epoch.

    Args:
        facility: Facility ID.
        section_config: Section configuration from assembly_config.
        epoch_id: Full epoch ID (e.g., 'jet:device_xml:p68613').
        gc: Graph client instance.

    Returns:
        List of SignalNode property dicts.
    """
    system = section_config.get("source_system")
    data_source = section_config.get("source_data_source", "device_xml")
    epoch_field = section_config.get("source_epoch_field", "introduced_version")
    select_via = section_config.get("source_select_via")

    if select_via:
        # Relationship-based selection (e.g., USES_LIMITER from epoch)
        return list(
            gc.query(
                f"""
                MATCH (se:StructuralEpoch {{id: $epoch_id}})
                      -[:{select_via}]->(d:SignalNode)
                RETURN d
                ORDER BY d.sort_key, d.id
                """,
                epoch_id=epoch_id,
            )
        )

    return list(
        gc.query(
            f"""
            MATCH (d:SignalNode {{
                facility_id: $facility,
                data_source_name: $data_source,
                system: $system
            }})
            WHERE d.{epoch_field} = $epoch_id
            RETURN d
            ORDER BY d.sort_key, d.id
            """,
            facility=facility,
            data_source=data_source,
            system=system,
            epoch_id=epoch_id,
        )
    )


def select_enrichment_nodes(
    facility: str,
    enrichment_config: dict[str, Any],
    gc: GraphClient,
) -> dict[int, dict[str, Any]]:
    """Query enrichment DataNodes and index by entry position.

    Args:
        facility: Facility ID.
        enrichment_config: Enrichment section of assembly_config.
        gc: Graph client instance.

    Returns:
        Dict mapping entry index to enrichment node properties.
    """
    data_source = enrichment_config.get("data_source", "jec2020_geometry")
    system = enrichment_config.get("system")

    rows = list(
        gc.query(
            """
            MATCH (d:SignalNode {
                facility_id: $facility,
                data_source_name: $data_source,
                system: $system
            })
            RETURN d
            ORDER BY d.sort_key, d.id
            """,
            facility=facility,
            data_source=data_source,
            system=system,
        )
    )

    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        node = row.get("d", row)
        path = node.get("path", node.get("id", ""))
        idx = _index_from_path(path)
        result[idx] = dict(node)
    return result


def _index_from_path(path: str) -> int:
    """Extract the numeric index from a SignalNode path suffix."""
    return int(path.rsplit(":", 1)[-1])


# ------------------------------------------------------------------
# Mapping definitions for each IDS
# ------------------------------------------------------------------

# Each spec: (source_property, target_imas_path, transform_code, units_in, units_out)
MappingSpec = tuple[str, str, str, str | None, str | None]

PF_ACTIVE_COIL_MAPPINGS: list[MappingSpec] = [
    ("r", "pf_active/coil/element/geometry/rectangle/r", "value", "m", "m"),
    ("z", "pf_active/coil/element/geometry/rectangle/z", "value", "m", "m"),
    ("dr", "pf_active/coil/element/geometry/rectangle/width", "value", "m", "m"),
    ("dz", "pf_active/coil/element/geometry/rectangle/height", "value", "m", "m"),
    (
        "turnsperelement",
        "pf_active/coil/element/turns_with_sign",
        "value",
        None,
        None,
    ),
    ("description", "pf_active/coil/name", "str(value)", None, None),
]

PF_ACTIVE_CIRCUIT_MAPPINGS: list[MappingSpec] = [
    ("description", "pf_active/circuit/name", "str(value)", None, None),
]

MAGNETICS_BPOL_MAPPINGS: list[MappingSpec] = [
    (
        "r",
        "magnetics/b_field_pol_probe/position/r",
        "value",
        "m",
        "m",
    ),
    (
        "z",
        "magnetics/b_field_pol_probe/position/z",
        "value",
        "m",
        "m",
    ),
    (
        "angle",
        "magnetics/b_field_pol_probe/poloidal_angle",
        "math.radians(value)",
        "deg",
        "rad",
    ),
    (
        "description",
        "magnetics/b_field_pol_probe/name",
        "str(value)",
        None,
        None,
    ),
]

MAGNETICS_FLUX_LOOP_MAPPINGS: list[MappingSpec] = [
    ("r", "magnetics/flux_loop/position/r", "value", "m", "m"),
    ("z", "magnetics/flux_loop/position/z", "value", "m", "m"),
    (
        "dphi",
        "magnetics/flux_loop/position/phi",
        "math.radians(value)",
        "deg",
        "rad",
    ),
    ("description", "magnetics/flux_loop/name", "str(value)", None, None),
]

PF_PASSIVE_LOOP_MAPPINGS: list[MappingSpec] = [
    ("r", "pf_passive/loop/element/geometry/rectangle/r", "value", "m", "m"),
    ("z", "pf_passive/loop/element/geometry/rectangle/z", "value", "m", "m"),
    ("dr", "pf_passive/loop/element/geometry/rectangle/width", "value", "m", "m"),
    ("dz", "pf_passive/loop/element/geometry/rectangle/height", "value", "m", "m"),
    ("resistance", "pf_passive/loop/resistance", "value", "ohm", "ohm"),
    ("description", "pf_passive/loop/name", "str(value)", None, None),
]

WALL_LIMITER_MAPPINGS: list[MappingSpec] = [
    ("r_contour", "wall/description_2d/limiter/unit/outline/r", "value", "m", "m"),
    ("z_contour", "wall/description_2d/limiter/unit/outline/z", "value", "m", "m"),
    ("description", "wall/description_2d/limiter/unit/name", "str(value)", None, None),
]


def create_signal_group(
    facility: str,
    ids_name: str,
    section: str,
    system: str,
    mapping_specs: list[MappingSpec],
    gc: GraphClient,
) -> str:
    """Create a SignalGroup and MAPS_TO_IMAS relationships for field mappings.

    Args:
        facility: Facility ID (e.g., 'jet').
        ids_name: IDS name (e.g., 'pf_active').
        section: Section name (e.g., 'coil', 'b_field_pol_probe').
        system: System code (e.g., 'PF', 'MP').
        mapping_specs: List of (source_property, target_path, transform_code,
            units_in, units_out) tuples.
        gc: Graph client instance.

    Returns:
        SignalGroup ID.
    """
    group_id = f"{facility}:ids:{ids_name}:{system}"
    group_key = f"{ids_name}/{section}"

    # Create SignalGroup node
    gc.query(
        """
        MERGE (sg:SignalGroup {id: $group_id})
        SET sg.facility_id = $facility,
            sg.group_key = $group_key,
            sg.status = 'discovered'
        WITH sg
        MATCH (f:Facility {id: $facility})
        MERGE (sg)-[:AT_FACILITY]->(f)
        """,
        group_id=group_id,
        facility=facility,
        group_key=group_key,
    )

    # Create MAPS_TO_IMAS relationships
    maps = [
        {
            "source_property": source_prop,
            "target_path": target_path,
            "transform_code": transform,
            "units_in": units_in,
            "units_out": units_out,
            "driver": "device_xml",
            "status": "validated",
            "confidence": 1.0,
        }
        for source_prop, target_path, transform, units_in, units_out in mapping_specs
    ]

    gc.query(
        """
        UNWIND $maps AS m
        MATCH (sg:SignalGroup {id: $group_id})
        MATCH (ip:IMASNode {id: m.target_path})
        MERGE (sg)-[rel:MAPS_TO_IMAS]->(ip)
        SET rel.source_property = m.source_property,
            rel.transform_code = m.transform_code,
            rel.units_in = m.units_in,
            rel.units_out = m.units_out,
            rel.driver = m.driver,
            rel.status = m.status,
            rel.confidence = m.confidence
        """,
        group_id=group_id,
        maps=maps,
    )

    logger.info(
        "Created SignalGroup %s with %d MAPS_TO_IMAS relationships",
        group_id,
        len(maps),
    )
    return group_id


def create_imas_mapping(
    facility: str,
    ids_name: str,
    dd_version: str,
    assembly_config: dict[str, Any],
    signal_group_ids: list[str],
    gc: GraphClient,
    *,
    provider: str = "imas-codex",
) -> str:
    """Create an IMASMapping node with USES_SIGNAL_GROUP and POPULATES.

    Args:
        facility: Facility ID.
        ids_name: IDS name.
        dd_version: DD version string.
        assembly_config: Structural assembly configuration dict.
        signal_group_ids: IDs of SignalGroup nodes to link.
        gc: Graph client instance.
        provider: Provider string for ids_properties.

    Returns:
        IMASMapping node ID.
    """
    mapping_id = f"{facility}:{ids_name}"
    static_config = json.dumps(assembly_config.get("static", {}))

    gc.query(
        """
        MERGE (m:IMASMapping {id: $mapping_id})
        SET m.facility_id = $facility,
            m.ids_name = $ids_name,
            m.dd_version = $dd_version,
            m.static_config = $static_config,
            m.provider = $provider,
            m.status = 'active'
        WITH m
        MATCH (f:Facility {id: $facility})
        MERGE (m)-[:AT_FACILITY]->(f)
        """,
        mapping_id=mapping_id,
        facility=facility,
        ids_name=ids_name,
        dd_version=dd_version,
        static_config=static_config,
        provider=provider,
    )

    # Link to SignalGroups
    gc.query(
        """
        UNWIND $group_ids AS gid
        MATCH (m:IMASMapping {id: $mapping_id})
        MATCH (sg:SignalGroup {id: gid})
        MERGE (m)-[:USES_SIGNAL_GROUP]->(sg)
        """,
        mapping_id=mapping_id,
        group_ids=signal_group_ids,
    )

    # Create POPULATES relationships with assembly config per section
    sections = []
    for section_name, section_config in assembly_config.items():
        if section_name == "static":
            continue
        source = section_config.get("source", {})
        # Derive the IMAS struct-array root path
        root_path = f"{ids_name}/{section_name}"
        sections.append(
            {
                "root_path": root_path,
                "structure": section_config.get("structure", "array_per_node"),
                "init_arrays": json.dumps(section_config.get("init_arrays", {})),
                "elements_config": json.dumps(section_config.get("elements", {})),
                "nested_path": section_config.get("nested_path"),
                "parent_size": section_config.get("parent_size"),
                "source_system": source.get("system"),
                "source_data_source": source.get("data_source", "device_xml"),
                "source_epoch_field": source.get("epoch_field", "introduced_version"),
                "source_select_via": source.get("select_via"),
                "enrichment": json.dumps(section_config.get("enrichment", [])),
            }
        )

    if sections:
        gc.query(
            """
            UNWIND $sections AS s
            MATCH (m:IMASMapping {id: $mapping_id})
            MATCH (root:IMASNode {id: s.root_path})
            MERGE (m)-[t:POPULATES]->(root)
            SET t.structure = s.structure,
                t.init_arrays = s.init_arrays,
                t.elements_config = s.elements_config,
                t.nested_path = s.nested_path,
                t.parent_size = s.parent_size,
                t.source_system = s.source_system,
                t.source_data_source = s.source_data_source,
                t.source_epoch_field = s.source_epoch_field,
                t.source_select_via = s.source_select_via,
                t.enrichment = s.enrichment
            """,
            mapping_id=mapping_id,
            sections=sections,
        )

    logger.info(
        "Created IMASMapping %s with %d signal groups, %d sections",
        mapping_id,
        len(signal_group_ids),
        len(sections),
    )
    return mapping_id


# ------------------------------------------------------------------
# Assembly config templates for each IDS
# ------------------------------------------------------------------

PF_ACTIVE_ASSEMBLY_CONFIG: dict[str, Any] = {
    "static": {
        "ids_properties.homogeneous_time": 0,
        "ids_properties.comment": (
            "JET PF active coil geometry from device descriptions. "
            "Assembled from device_xml DataNodes by imas-codex."
        ),
    },
    "coil": {
        "source": {
            "system": "PF",
            "data_source": "device_xml",
            "epoch_field": "introduced_version",
        },
        "structure": "array_per_node",
        "elements": {"geometry_type": 2},
        "enrichment": [
            {
                "data_source": "jec2020_geometry",
                "system": "PF",
                "match_by": "coil_index",
            }
        ],
    },
    "circuit": {
        "source": {
            "system": "CI",
            "data_source": "device_xml",
            "epoch_field": "introduced_version",
        },
        "structure": "array_per_node",
    },
}

MAGNETICS_ASSEMBLY_CONFIG: dict[str, Any] = {
    "static": {
        "ids_properties.homogeneous_time": 0,
        "ids_properties.comment": (
            "JET magnetics diagnostic geometry from device descriptions. "
            "Assembled from device_xml DataNodes by imas-codex."
        ),
    },
    "b_field_pol_probe": {
        "source": {
            "system": "MP",
            "data_source": "device_xml",
            "epoch_field": "introduced_version",
        },
        "structure": "array_per_node",
    },
    "flux_loop": {
        "source": {
            "system": "FL",
            "data_source": "device_xml",
            "epoch_field": "introduced_version",
        },
        "structure": "array_per_node",
        "init_arrays": {"position": 1},
    },
}

PF_PASSIVE_ASSEMBLY_CONFIG: dict[str, Any] = {
    "static": {
        "ids_properties.homogeneous_time": 0,
        "ids_properties.comment": (
            "JET PF passive structure geometry from device descriptions. "
            "Assembled from device_xml DataNodes by imas-codex."
        ),
    },
    "loop": {
        "source": {
            "system": "PS",
            "data_source": "device_xml",
            "epoch_field": "introduced_version",
        },
        "structure": "array_per_node",
        "elements": {"geometry_type": 2},
    },
}

WALL_ASSEMBLY_CONFIG: dict[str, Any] = {
    "static": {
        "ids_properties.homogeneous_time": 0,
        "ids_properties.comment": (
            "JET first wall limiter contours from device descriptions. "
            "Assembled from device_xml DataNodes by imas-codex."
        ),
    },
    "description_2d": {
        "source": {
            "data_source": "device_xml",
            "select_via": "USES_LIMITER",
        },
        "structure": "nested_array",
        "nested_path": "limiter.unit",
        "parent_size": 1,
    },
}


def seed_ids_mappings(
    facility: str,
    ids_name: str,
    dd_version: str,
    gc: GraphClient,
) -> str:
    """Seed SignalGroups, MAPS_TO_IMAS relationships, and an IMASMapping.

    Creates SignalGroup nodes with field-level MAPS_TO_IMAS relationships,
    then an IMASMapping orchestration node linking them via USES_SIGNAL_GROUP
    and POPULATES.

    Args:
        facility: Facility ID.
        ids_name: IDS name ('pf_active', 'magnetics', 'pf_passive', 'wall').
        dd_version: DD version string.
        gc: Graph client instance.

    Returns:
        IMASMapping node ID.

    Raises:
        ValueError: If ids_name is not supported.
    """
    configs = {
        "pf_active": (
            PF_ACTIVE_ASSEMBLY_CONFIG,
            [
                ("coil", "PF", PF_ACTIVE_COIL_MAPPINGS),
                ("circuit", "CI", PF_ACTIVE_CIRCUIT_MAPPINGS),
            ],
        ),
        "magnetics": (
            MAGNETICS_ASSEMBLY_CONFIG,
            [
                ("b_field_pol_probe", "MP", MAGNETICS_BPOL_MAPPINGS),
                ("flux_loop", "FL", MAGNETICS_FLUX_LOOP_MAPPINGS),
            ],
        ),
        "pf_passive": (
            PF_PASSIVE_ASSEMBLY_CONFIG,
            [
                ("loop", "PS", PF_PASSIVE_LOOP_MAPPINGS),
            ],
        ),
        "wall": (
            WALL_ASSEMBLY_CONFIG,
            [
                ("description_2d", "LIM", WALL_LIMITER_MAPPINGS),
            ],
        ),
    }

    if ids_name not in configs:
        msg = (
            f"No mapping definitions for IDS '{ids_name}'. Supported: {sorted(configs)}"
        )
        raise ValueError(msg)

    assembly_config, sections = configs[ids_name]

    signal_group_ids: list[str] = []
    for section, system, specs in sections:
        group_id = create_signal_group(facility, ids_name, section, system, specs, gc)
        signal_group_ids.append(group_id)

    return create_imas_mapping(
        facility, ids_name, dd_version, assembly_config, signal_group_ids, gc
    )
