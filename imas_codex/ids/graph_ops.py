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
