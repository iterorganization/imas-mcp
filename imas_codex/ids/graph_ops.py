"""Graph operations for IDS mapping and assembly.

Provides queries to load IMASMapping nodes and their linked SignalSources
from the graph, and functions to seed mapping definitions.

Architecture:
    IMASMapping (orchestration) -[:USES_SIGNAL_SOURCE]-> SignalSource
    SignalSource -[:MAPS_TO_IMAS]-> IMASNode (field-level transform)
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
class SignalMapping:
    """A resolved signal mapping from a MAPS_TO_IMAS relationship."""

    source_id: str
    source_property: str
    target_id: str
    transform_expression: str = "value"
    source_units: str | None = None
    target_units: str | None = None
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
    bindings: list[SignalMapping] = field(default_factory=list)


def load_mapping(facility: str, ids_name: str, gc: GraphClient) -> Mapping | None:
    """Load an IMASMapping and its linked SignalSources from the graph.

    Args:
        facility: Facility ID.
        ids_name: IDS name (e.g., 'pf_active').
        gc: Graph client instance.

    Returns:
        Mapping with populated bindings and sections, or None if not found.
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
    mapping.bindings = load_signal_mappings(mapping.id, gc)
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


def load_signal_mappings(mapping_id: str, gc: GraphClient) -> list[SignalMapping]:
    """Load signal mappings via USES_SIGNAL_SOURCE → MAPS_TO_IMAS traversal.

    Args:
        mapping_id: IMASMapping node ID.
        gc: Graph client instance.

    Returns:
        List of resolved signal mappings.
    """
    rows = list(
        gc.query(
            """
            MATCH (m:IMASMapping {id: $mapping_id})
                  -[:USES_SIGNAL_SOURCE]->(sg:SignalSource)
                  -[map:MAPS_TO_IMAS]->(ip:IMASNode)
            RETURN sg.id AS source_id,
                   map.source_property AS source_property,
                   ip.id AS target_id,
                   map.transform_expression AS transform_expression,
                   map.source_units AS source_units,
                   map.target_units AS target_units,
                   map.cocos_source AS cocos_source,
                   map.cocos_target AS cocos_target,
                   map.driver AS driver,
                   ip.cocos_transformation_type AS cocos_label
            """,
            mapping_id=mapping_id,
        )
    )
    mappings = []
    for row in rows:
        fm = SignalMapping(
            source_id=row["source_id"],
            source_property=row.get("source_property") or "value",
            target_id=row["target_id"],
            transform_expression=row.get("transform_expression") or "value",
            source_units=row.get("source_units"),
            target_units=row.get("target_units"),
            cocos_source=row.get("cocos_source"),
            cocos_target=row.get("cocos_target"),
            driver=row.get("driver") or "device_xml",
        )
        cocos_label = row.get("cocos_label")
        if cocos_label and cocos_label != "one_like" and not fm.cocos_source:
            logger.warning(
                "COCOS-sensitive path %s (label=%s) has no cocos_source on mapping %s",
                fm.target_id,
                cocos_label,
                fm.source_id,
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
                MATCH (se:SignalEpoch {{id: $epoch_id}})
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


# Each spec: (source_property, target_id, transform_expression, source_units, target_units)
MappingSpec = tuple[str, str, str, str | None, str | None]


def create_signal_source(
    facility: str,
    ids_name: str,
    section: str,
    system: str,
    mapping_specs: list[MappingSpec],
    gc: GraphClient,
) -> str:
    """Create a SignalSource and MAPS_TO_IMAS relationships for field mappings.

    Args:
        facility: Facility ID (e.g., 'jet').
        ids_name: IDS name (e.g., 'pf_active').
        section: Section name (e.g., 'coil', 'b_field_pol_probe').
        system: System code (e.g., 'PF', 'MP').
        mapping_specs: List of (source_property, target_id, transform_expression,
            source_units, target_units) tuples.
        gc: Graph client instance.

    Returns:
        SignalSource ID.
    """
    group_id = f"{facility}:ids:{ids_name}:{system}"
    group_key = f"{ids_name}/{section}"

    # Create SignalSource node
    gc.query(
        """
        MERGE (sg:SignalSource {id: $group_id})
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
            "transform_expression": transform,
            "source_units": source_units,
            "target_units": target_units,
            "driver": "device_xml",
            "status": "validated",
            "confidence": 1.0,
        }
        for source_prop, target_path, transform, source_units, target_units in mapping_specs
    ]

    gc.query(
        """
        UNWIND $maps AS m
        MATCH (sg:SignalSource {id: $group_id})
        MATCH (ip:IMASNode {id: m.target_path})
        MERGE (sg)-[rel:MAPS_TO_IMAS]->(ip)
        SET rel.source_property = m.source_property,
            rel.transform_expression = m.transform_expression,
            rel.source_units = m.source_units,
            rel.target_units = m.target_units,
            rel.driver = m.driver,
            rel.status = m.status,
            rel.confidence = m.confidence
        """,
        group_id=group_id,
        maps=maps,
    )

    logger.info(
        "Created SignalSource %s with %d MAPS_TO_IMAS relationships",
        group_id,
        len(maps),
    )
    return group_id


def create_imas_mapping(
    facility: str,
    ids_name: str,
    dd_version: str,
    assembly_config: dict[str, Any],
    signal_source_ids: list[str],
    gc: GraphClient,
    *,
    provider: str = "imas-codex",
) -> str:
    """Create an IMASMapping node with USES_SIGNAL_SOURCE and POPULATES.

    Args:
        facility: Facility ID.
        ids_name: IDS name.
        dd_version: DD version string.
        assembly_config: Structural assembly configuration dict.
        signal_source_ids: IDs of SignalSource nodes to link.
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

    # Link to SignalSources
    gc.query(
        """
        UNWIND $group_ids AS gid
        MATCH (m:IMASMapping {id: $mapping_id})
        MATCH (sg:SignalSource {id: gid})
        MERGE (m)-[:USES_SIGNAL_SOURCE]->(sg)
        """,
        mapping_id=mapping_id,
        group_ids=signal_source_ids,
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
        "Created IMASMapping %s with %d signal sources, %d sections",
        mapping_id,
        len(signal_source_ids),
        len(sections),
    )
    return mapping_id
