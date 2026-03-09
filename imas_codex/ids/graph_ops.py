"""Graph operations for IDS mapping and recipe management.

Provides queries to load IMASMapping and IDSRecipe nodes from the graph,
and functions to create mapping nodes for new field transformations.
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
    """A resolved field mapping from graph data."""

    id: str
    source_property: str
    target_imas_path: str
    transform_code: str = "value"
    units_in: str | None = None
    units_out: str | None = None
    cocos_source: int | None = None
    cocos_target: int | None = None
    driver: str = "device_xml"


@dataclass
class Recipe:
    """A resolved IDSRecipe from graph data."""

    id: str
    facility_id: str
    ids_name: str
    dd_version: str
    provider: str | None = None
    assembly_config: dict[str, Any] = field(default_factory=dict)
    mappings: list[FieldMapping] = field(default_factory=list)


def load_recipe(facility: str, ids_name: str, gc: GraphClient) -> Recipe | None:
    """Load an IDSRecipe and its linked IMASMappings from the graph.

    Args:
        facility: Facility ID.
        ids_name: IDS name (e.g., 'pf_active').
        gc: Graph client instance.

    Returns:
        Recipe with populated mappings, or None if not found.
    """
    rows = list(
        gc.query(
            """
            MATCH (r:IDSRecipe {facility_id: $facility, ids_name: $ids_name})
            WHERE r.status = 'active'
            RETURN r.id AS id, r.facility_id AS facility_id,
                   r.ids_name AS ids_name, r.dd_version AS dd_version,
                   r.provider AS provider, r.assembly_config AS assembly_config
            LIMIT 1
            """,
            facility=facility,
            ids_name=ids_name,
        )
    )
    if not rows:
        return None

    row = rows[0]
    config_str = row.get("assembly_config") or "{}"
    recipe = Recipe(
        id=row["id"],
        facility_id=row["facility_id"],
        ids_name=row["ids_name"],
        dd_version=row["dd_version"],
        provider=row.get("provider"),
        assembly_config=json.loads(config_str),
    )

    recipe.mappings = load_mappings(recipe.id, gc)
    return recipe


def load_mappings(recipe_id: str, gc: GraphClient) -> list[FieldMapping]:
    """Load IMASMappings linked to a recipe via INCLUDES_MAPPING.

    Args:
        recipe_id: IDSRecipe node ID.
        gc: Graph client instance.

    Returns:
        List of resolved field mappings.
    """
    rows = list(
        gc.query(
            """
            MATCH (r:IDSRecipe {id: $recipe_id})
                  -[:INCLUDES_MAPPING]->(m:IMASMapping)
                  -[:TARGET_PATH]->(ip:IMASPath)
            RETURN m.id AS id,
                   m.source_property AS source_property,
                   ip.id AS target_imas_path,
                   m.transform_code AS transform_code,
                   m.units_in AS units_in,
                   m.units_out AS units_out,
                   m.cocos_source AS cocos_source,
                   m.cocos_target AS cocos_target,
                   m.driver AS driver,
                   ip.cocos_label_transformation AS cocos_label
            """,
            recipe_id=recipe_id,
        )
    )
    mappings = []
    for row in rows:
        mapping = FieldMapping(
            id=row["id"],
            source_property=row.get("source_property") or "value",
            target_imas_path=row["target_imas_path"],
            transform_code=row.get("transform_code") or "value",
            units_in=row.get("units_in"),
            units_out=row.get("units_out"),
            cocos_source=row.get("cocos_source"),
            cocos_target=row.get("cocos_target"),
            driver=row.get("driver") or "device_xml",
        )
        # Warn if IMAS path is COCOS-sensitive but mapping has no source COCOS
        cocos_label = row.get("cocos_label")
        if cocos_label and cocos_label != "one_like" and not mapping.cocos_source:
            logger.warning(
                "COCOS-sensitive path %s (label=%s) has no cocos_source on mapping %s",
                mapping.target_imas_path,
                cocos_label,
                mapping.id,
            )
        mappings.append(mapping)
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
        List of DataNode property dicts.
    """
    source = section_config.get("source", {})
    system = source.get("system")
    data_source = source.get("data_source", "device_xml")
    epoch_field = source.get("epoch_field", "introduced_version")

    return list(
        gc.query(
            f"""
            MATCH (d:DataNode {{
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
            MATCH (d:DataNode {
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
    """Extract the numeric index from a DataNode path suffix."""
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


def create_mappings(
    facility: str,
    ids_name: str,
    section: str,
    system: str,
    mapping_specs: list[MappingSpec],
    gc: GraphClient,
) -> list[str]:
    """Create IMASMapping nodes for a set of field mappings.

    Args:
        facility: Facility ID (e.g., 'jet').
        ids_name: IDS name (e.g., 'pf_active').
        section: Section name (e.g., 'coil', 'b_field_pol_probe').
        system: System code (e.g., 'PF', 'MP').
        mapping_specs: List of (source_property, target_path, transform_code,
            units_in, units_out) tuples.
        gc: Graph client instance.

    Returns:
        List of created mapping IDs.
    """
    mappings = []
    for source_prop, target_path, transform, units_in, units_out in mapping_specs:
        mapping_id = f"{facility}:{system}:{source_prop}→{target_path}"
        mappings.append(
            {
                "id": mapping_id,
                "facility_id": facility,
                "source_property": source_prop,
                "target_path": target_path,
                "transform_code": transform,
                "units_in": units_in,
                "units_out": units_out,
                "driver": "device_xml",
                "status": "validated",
                "confidence": 1.0,
            }
        )

    gc.query(
        """
        UNWIND $mappings AS m
        MERGE (mapping:IMASMapping {id: m.id})
        SET mapping += m
        WITH mapping, m
        MATCH (f:Facility {id: m.facility_id})
        MERGE (mapping)-[:AT_FACILITY]->(f)
        WITH mapping, m
        MATCH (ip:IMASPath {id: m.target_path})
        MERGE (mapping)-[:TARGET_PATH]->(ip)
        """,
        mappings=mappings,
    )

    logger.info(
        "Created %d IMASMapping nodes for %s.%s (%s)",
        len(mappings),
        ids_name,
        section,
        facility,
    )
    return [m["id"] for m in mappings]


def create_recipe(
    facility: str,
    ids_name: str,
    dd_version: str,
    assembly_config: dict[str, Any],
    mapping_ids: list[str],
    gc: GraphClient,
    *,
    provider: str = "imas-codex",
) -> str:
    """Create an IDSRecipe node and link it to IMASMappings.

    Args:
        facility: Facility ID.
        ids_name: IDS name.
        dd_version: DD version string.
        assembly_config: Structural assembly configuration dict.
        mapping_ids: IDs of IMASMapping nodes to link.
        gc: Graph client instance.
        provider: Provider string for ids_properties.

    Returns:
        Recipe node ID.
    """
    recipe_id = f"{facility}:{ids_name}"
    config_json = json.dumps(assembly_config)

    gc.query(
        """
        MERGE (r:IDSRecipe {id: $recipe_id})
        SET r.facility_id = $facility,
            r.ids_name = $ids_name,
            r.dd_version = $dd_version,
            r.assembly_config = $config_json,
            r.provider = $provider,
            r.status = 'active'
        WITH r
        MATCH (f:Facility {id: $facility})
        MERGE (r)-[:AT_FACILITY]->(f)
        """,
        recipe_id=recipe_id,
        facility=facility,
        ids_name=ids_name,
        dd_version=dd_version,
        config_json=config_json,
        provider=provider,
    )

    # Link to mappings
    gc.query(
        """
        UNWIND $mapping_ids AS mid
        MATCH (r:IDSRecipe {id: $recipe_id})
        MATCH (m:IMASMapping {id: mid})
        MERGE (r)-[:INCLUDES_MAPPING]->(m)
        """,
        recipe_id=recipe_id,
        mapping_ids=mapping_ids,
    )

    logger.info("Created IDSRecipe %s with %d mappings", recipe_id, len(mapping_ids))
    return recipe_id


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


def seed_ids_mappings(
    facility: str,
    ids_name: str,
    dd_version: str,
    gc: GraphClient,
) -> str:
    """Seed IMASMapping and IDSRecipe nodes for a given IDS.

    Creates all field mapping nodes and a structural recipe for the
    specified IDS, using the canonical mapping definitions.

    Args:
        facility: Facility ID.
        ids_name: IDS name ('pf_active', 'magnetics', 'pf_passive').
        dd_version: DD version string.
        gc: Graph client instance.

    Returns:
        Recipe node ID.

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
    }

    if ids_name not in configs:
        msg = (
            f"No mapping definitions for IDS '{ids_name}'. Supported: {sorted(configs)}"
        )
        raise ValueError(msg)

    assembly_config, sections = configs[ids_name]

    all_mapping_ids: list[str] = []
    for section, system, specs in sections:
        mapping_ids = create_mappings(facility, ids_name, section, system, specs, gc)
        all_mapping_ids.extend(mapping_ids)

    return create_recipe(
        facility, ids_name, dd_version, assembly_config, all_mapping_ids, gc
    )
