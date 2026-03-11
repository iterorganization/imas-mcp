"""IDS assembly engine.

Assembles complete IMAS IDS instances from facility graph data using two
modes:

1. **Graph-driven** (preferred): Reads IMASMapping and SignalSource nodes from
   the knowledge graph. The IMASMapping's POPULATES relationships define
   structural patterns (how DataNodes group into array-of-structures entries),
   while MAPS_TO_IMAS relationships on SignalSources define field-level
   transformations with executable transform_expression.

2. **YAML recipe** (fallback): Reads a YAML recipe file with embedded Cypher
   queries and field mappings. Used when no graph recipe exists.

Architecture:
    IMASMapping -[:POPULATES]-> IMASNode (struct-array assembly config)
    IMASMapping -[:USES_SIGNAL_SOURCE]-> SignalSource -[:MAPS_TO_IMAS]-> IMASNode
    IDSAssembler → reads mapping, queries graph, populates IDS (generic)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from imas_codex.graph.client import GraphClient
from imas_codex.ids.graph_ops import (
    Mapping,
    load_mapping,
    select_enrichment_nodes,
    select_nodes,
)
from imas_codex.ids.transforms import convert_units, execute_transform, set_nested

logger = logging.getLogger(__name__)

RECIPES_DIR = Path(__file__).parent / "recipes"


def _resolve_epoch_id(facility: str, epoch: str) -> str:
    """Convert a short epoch name to the full epoch ID used in the graph."""
    if epoch.startswith(f"{facility}:"):
        return epoch
    return f"{facility}:device_xml:{epoch}"


def _coil_index_from_path(path: str) -> int:
    """Extract the numeric coil/element index from a SignalNode path suffix."""
    return int(path.rsplit(":", 1)[-1])


def _flatten_node(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten a graph query result that may wrap the node in a key.

    Graph queries returning ``RETURN d`` wrap properties in ``{"d": {...}}``.
    This extracts the inner dict if present.
    """
    if len(row) == 1:
        key = next(iter(row))
        val = row[key]
        if isinstance(val, dict):
            return val
    return dict(row)


class IDSAssembler:
    """Assembles IMAS IDS instances from graph data.

    Supports two modes:
    - Graph-driven: Loads IMASMapping + SignalSources from the knowledge graph.
    - YAML fallback: Loads a YAML recipe file with embedded queries.

    Graph mode is used when an active IMASMapping exists in the graph.
    YAML mode is used as fallback when no graph recipe is found.

    Args:
        facility: Facility ID (e.g., 'jet').
        ids_name: IDS name (e.g., 'pf_active').
        recipe_path: Optional explicit path to YAML recipe. If not given,
            tries graph first, then recipes/{facility}/{ids_name}.yaml.
    """

    def __init__(
        self,
        facility: str,
        ids_name: str,
        recipe_path: Path | None = None,
    ):
        self.facility = facility
        self.ids_name = ids_name
        self._graph_mapping: Mapping | None = None
        self._yaml_recipe: dict[str, Any] | None = None

        if recipe_path is not None:
            # Explicit YAML path: use YAML mode
            self._load_yaml_recipe(recipe_path)
        else:
            # Try graph first, fall back to YAML
            try:
                with GraphClient() as gc:
                    self._graph_mapping = load_mapping(facility, ids_name, gc)
            except Exception:
                logger.debug("Graph mapping lookup failed, trying YAML", exc_info=True)

            if self._graph_mapping is None:
                yaml_path = RECIPES_DIR / facility / f"{ids_name}.yaml"
                if yaml_path.exists():
                    self._load_yaml_recipe(yaml_path)
                else:
                    msg = (
                        f"No recipe found for {facility}/{ids_name} "
                        f"(checked graph and {yaml_path})"
                    )
                    raise FileNotFoundError(msg)

    def _load_yaml_recipe(self, path: Path) -> None:
        if not path.exists():
            msg = f"No recipe found at {path}"
            raise FileNotFoundError(msg)
        self._yaml_recipe = yaml.safe_load(path.read_text())
        self._validate_yaml_recipe()

    @property
    def dd_version(self) -> str:
        if self._graph_mapping:
            return self._graph_mapping.dd_version
        return self._yaml_recipe["dd_version"]

    @property
    def recipe(self) -> dict[str, Any]:
        """Access the raw YAML recipe dict (for backward compatibility)."""
        if self._yaml_recipe is not None:
            return self._yaml_recipe
        # Synthesize a dict from graph mapping for compatibility
        m = self._graph_mapping
        return {
            "ids_name": m.ids_name,
            "facility_id": m.facility_id,
            "dd_version": m.dd_version,
            "provider": m.provider,
        }

    def _validate_yaml_recipe(self) -> None:
        """Basic validation of YAML recipe structure."""
        required = {"ids_name", "facility_id", "dd_version"}
        missing = required - set(self._yaml_recipe)
        if missing:
            msg = f"Recipe missing required fields: {missing}"
            raise ValueError(msg)

    def assemble(self, epoch: str) -> Any:
        """Build an IDS instance for the given epoch.

        Dispatches to graph-driven or YAML-driven assembly based on
        which recipe source is available.

        Args:
            epoch: Epoch version string (e.g., 'p68613' or full
                'jet:device_xml:p68613').

        Returns:
            Populated imas-python IDSToplevel object.
        """
        if self._graph_mapping is not None:
            return self._assemble_from_graph(epoch)
        return self._assemble_from_yaml(epoch)

    # ------------------------------------------------------------------
    # Graph-driven assembly
    # ------------------------------------------------------------------

    def _assemble_from_graph(self, epoch: str) -> Any:
        """Assemble IDS using IMASMapping and IMASMappings from graph."""
        import imas

        mapping = self._graph_mapping
        epoch_id = _resolve_epoch_id(self.facility, epoch)

        factory = imas.IDSFactory(mapping.dd_version)
        ids = factory.new(self.ids_name)

        # Set static properties
        for key, value in mapping.static_config.items():
            set_nested(ids, key, value)

        if mapping.provider:
            ids.ids_properties.provider = str(mapping.provider)

        # Build each structural section
        with GraphClient() as gc:
            for section in mapping.sections:
                self._build_graph_section(ids, section, epoch_id, mapping, gc)

        return ids

    def _build_graph_section(
        self,
        ids: Any,
        section: dict[str, Any],
        epoch_id: str,
        mapping: Mapping,
        gc: GraphClient,
    ) -> None:
        """Build one struct-array section from graph data."""
        root_path = section["root_path"]
        section_name = root_path.rsplit("/", 1)[-1]

        nodes = select_nodes(self.facility, section, epoch_id, gc)
        if not nodes:
            logger.warning(
                "No data for %s.%s epoch=%s", self.ids_name, section_name, epoch_id
            )
            return

        # Flatten returned nodes (handle {d: {...}} wrapping)
        flat_nodes = [_flatten_node(n) for n in nodes]

        # Load enrichment if configured
        enrichment: dict[int, dict[str, Any]] = {}
        enrichment_raw = section.get("enrichment") or "[]"
        enrichment_list = (
            json.loads(enrichment_raw)
            if isinstance(enrichment_raw, str)
            else enrichment_raw
        )
        for enrich_def in enrichment_list:
            enrichment.update(select_enrichment_nodes(self.facility, enrich_def, gc))

        # Get mappings relevant to this section's target IDS path
        section_mappings = [
            m
            for m in mapping.bindings
            if m.target_id.startswith(f"{self.ids_name}/{section_name}")
        ]

        struct_array = getattr(ids, section_name)
        structure = section.get("structure", "array_per_node")

        # Parse JSON fields from POPULATES relationship
        init_arrays_raw = section.get("init_arrays") or "{}"
        init_arrays = (
            json.loads(init_arrays_raw)
            if isinstance(init_arrays_raw, str)
            else init_arrays_raw
        )
        elements_raw = section.get("elements_config") or "{}"
        elements_config = (
            json.loads(elements_raw) if isinstance(elements_raw, str) else elements_raw
        )

        # Build a config dict for methods that expect it
        section_config = {
            "structure": structure,
            "init_arrays": init_arrays,
            "elements": elements_config or None,
            "nested_path": section.get("nested_path"),
            "parent_size": section.get("parent_size"),
        }

        if structure == "nested_array":
            self._build_nested_array(
                struct_array,
                flat_nodes,
                section_config,
                section_mappings,
                section_name,
            )
        elif structure == "array_per_node":
            struct_array.resize(len(flat_nodes))
            for i, node_data in enumerate(flat_nodes):
                entry = struct_array[i]
                idx = _coil_index_from_path(
                    node_data.get("path", node_data.get("id", ""))
                )
                enriched = enrichment.get(idx, {})
                merged = {**node_data, **enriched}

                # Initialize sub-arrays if configured (e.g., flux_loop.position)
                for sub_array_path, size in init_arrays.items():
                    sub_array = getattr(entry, sub_array_path)
                    sub_array.resize(size)

                self._apply_mappings(
                    entry, merged, section_mappings, section_name, section_config
                )

                # Build sub-arrays (elements) if configured
                if elements_config:
                    self._build_graph_elements(
                        entry,
                        node_data,
                        enriched,
                        elements_config,
                        section_mappings,
                        section_name,
                    )

    def _build_nested_array(
        self,
        parent_array: Any,
        flat_nodes: list[dict[str, Any]],
        section_config: dict[str, Any],
        section_mappings: list,
        section_name: str,
    ) -> None:
        """Build a nested struct-array (e.g., wall.description_2d[0].limiter.unit).

        Used for IDS structures where DataNodes populate a nested array
        within a parent container. The parent container is resized to
        parent_size (typically 1), and the nested array is resized to
        the number of DataNodes.
        """
        parent_size = section_config.get("parent_size", 1)
        nested_path = section_config.get("nested_path", "")

        parent_array.resize(parent_size)
        container = parent_array[0]

        # Navigate to the nested array
        nested_array = container
        for part in nested_path.split("."):
            nested_array = getattr(nested_array, part)

        nested_array.resize(len(flat_nodes))

        for i, node_data in enumerate(flat_nodes):
            entry = nested_array[i]
            self._apply_mappings(
                entry, node_data, section_mappings, section_name, section_config
            )

    def _apply_mappings(
        self,
        entry: Any,
        data: dict[str, Any],
        mappings: list,
        section_name: str,
        section_config: dict[str, Any] | None = None,
    ) -> None:
        """Apply field-level mappings to a struct entry.

        Handles transform_expression execution and automatic unit conversion
        when source_units != target_units.
        """
        init_arrays = (section_config or {}).get("init_arrays", {})
        nested_path = (section_config or {}).get("nested_path")
        for mapping in mappings:
            # Strip the IDS prefix to get the path relative to this entry
            # e.g., "pf_active/coil/name" -> "name"
            rel_path = mapping.target_id
            prefix = f"{self.ids_name}/{section_name}/"
            if rel_path.startswith(prefix):
                rel_path = rel_path[len(prefix) :]
            else:
                continue

            # Strip nested path prefix for nested_array structures
            # e.g., "limiter/unit/outline/r" -> "outline/r"
            if nested_path:
                nested_prefix = nested_path.replace(".", "/") + "/"
                if rel_path.startswith(nested_prefix):
                    rel_path = rel_path[len(nested_prefix) :]
                else:
                    continue

            # Convert IMAS path separators to Python attribute notation
            rel_path = rel_path.replace("/", ".")

            # Rewrite paths that traverse initialized arrays
            # e.g., with init_arrays={"position": 1}, "position.r" -> "position[0].r"
            for arr_name in init_arrays:
                if rel_path.startswith(f"{arr_name}."):
                    rel_path = f"{arr_name}[0].{rel_path[len(arr_name) + 1 :]}"
                    break

            # Skip element-level mappings (handled by _build_graph_elements)
            if "element." in rel_path or rel_path.startswith("element"):
                continue

            value = data.get(mapping.source_property)
            if value is not None:
                value = execute_transform(value, mapping.transform_expression)
                # Auto-convert units if specified and different
                if (
                    mapping.source_units
                    and mapping.target_units
                    and mapping.source_units != mapping.target_units
                    and isinstance(value, int | float)
                ):
                    value = convert_units(value, mapping.source_units, mapping.target_units)
                try:
                    set_nested(entry, rel_path, value)
                except (AttributeError, TypeError):
                    logger.debug("Cannot set %s on %s", rel_path, type(entry).__name__)

    def _build_graph_elements(
        self,
        coil: Any,
        node_data: dict[str, Any],
        enriched: dict[str, Any],
        elements_config: dict[str, Any],
        section_mappings: list,
        section_name: str,
    ) -> None:
        """Build element sub-arrays from array properties."""
        geometry_type = elements_config.get("geometry_type", 2)

        # Find element-level mappings
        element_prefix = f"{self.ids_name}/{section_name}/element/"
        element_mappings = [
            m for m in section_mappings if m.target_id.startswith(element_prefix)
        ]

        # Determine element count from array properties
        n_elements = 0
        merged = {**node_data, **enriched}
        for m in element_mappings:
            val = merged.get(m.source_property)
            if isinstance(val, list):
                n_elements = max(n_elements, len(val))

        if n_elements == 0:
            n_elements = 1

        coil.element.resize(n_elements)

        for j in range(n_elements):
            elem = coil.element[j]
            elem.geometry.geometry_type = geometry_type

            for m in element_mappings:
                rel_path = m.target_id[len(element_prefix) :].replace("/", ".")
                val = merged.get(m.source_property)
                if val is None:
                    continue
                if isinstance(val, list):
                    element_val = val[j]
                else:
                    element_val = val
                element_val = execute_transform(element_val, m.transform_expression)
                # Auto-convert units if specified and different
                if (
                    m.source_units
                    and m.target_units
                    and m.source_units != m.target_units
                    and isinstance(element_val, int | float)
                ):
                    element_val = convert_units(element_val, m.source_units, m.target_units)
                try:
                    set_nested(elem, rel_path, float(element_val))
                except (AttributeError, TypeError, ValueError):
                    logger.debug("Cannot set %s on element", rel_path)

    # ------------------------------------------------------------------
    # YAML-driven assembly (backward compatibility)
    # ------------------------------------------------------------------

    def _assemble_from_yaml(self, epoch: str) -> Any:
        """Build an IDS instance from YAML recipe (original implementation)."""
        import imas

        epoch_id = _resolve_epoch_id(self.facility, epoch)
        dd_version = self.recipe["dd_version"]

        factory = imas.IDSFactory(dd_version)
        ids = factory.new(self.ids_name)

        # Set static properties
        for dotted_path, value in self.recipe.get("static", {}).items():
            set_nested(ids, dotted_path, value)

        # Set provider
        if provider := self.recipe.get("provider"):
            ids.ids_properties.provider = str(provider)

        # Build struct arrays
        for array_name, array_def in self.recipe.get("arrays", {}).items():
            self._build_array(ids, array_name, array_def, epoch_id)

        return ids

    def _build_array(
        self,
        ids: Any,
        array_name: str,
        array_def: dict[str, Any],
        epoch_id: str,
    ) -> None:
        """Build one top-level struct-array (e.g., coil, circuit)."""
        source_def = array_def.get("source", {})
        query = source_def.get("query", "")
        if not query:
            logger.warning("No source query for array %s", array_name)
            return

        # Query graph for primary data
        with GraphClient() as gc:
            rows = list(
                gc.query(
                    query,
                    facility=self.facility,
                    epoch_id=epoch_id,
                )
            )

        if not rows:
            logger.warning(
                "No data found for %s.%s epoch=%s", self.ids_name, array_name, epoch_id
            )
            return

        # Query enrichment data if defined
        enrichment_data: dict[int, dict] = {}
        enrich_def = array_def.get("enrichment")
        if enrich_def:
            enrichment_data = self._query_enrichment(enrich_def)

        # Resize the struct array
        struct_array = getattr(ids, array_name)
        struct_array.resize(len(rows))

        logger.info(
            "Building %s.%s: %d entries from graph",
            self.ids_name,
            array_name,
            len(rows),
        )

        # Populate each entry
        for i, row in enumerate(rows):
            entry = struct_array[i]
            coil_index = _coil_index_from_path(row.get("path", ""))

            # Apply enrichment if available
            enriched = enrichment_data.get(coil_index, {})

            # Set simple fields
            for ids_field, source_field in array_def.get("fields", {}).items():
                value = enriched.get(source_field) or row.get(source_field)
                if value is not None:
                    set_nested(entry, ids_field, str(value))

            # Build elements if defined
            elements_def = array_def.get("elements")
            if elements_def:
                self._build_elements(entry, row, enriched, elements_def)

    def _build_elements(
        self,
        coil: Any,
        row: dict[str, Any],
        enriched: dict[str, Any],
        elements_def: dict[str, Any],
    ) -> None:
        """Build element sub-arrays within a coil from array properties."""
        geometry_type = elements_def.get("geometry_type", 2)
        field_mappings = elements_def.get("fields", {})

        # Determine number of elements from the first array-valued field
        n_elements = 0
        for source_field in field_mappings.values():
            val = row.get(source_field)
            if isinstance(val, list):
                n_elements = len(val)
                break

        if n_elements == 0:
            # Single element from scalar values
            n_elements = 1

        coil.element.resize(n_elements)

        for j in range(n_elements):
            elem = coil.element[j]
            elem.geometry.geometry_type = geometry_type

            for ids_path, source_field in field_mappings.items():
                val = row.get(source_field)
                if val is None:
                    continue

                # Array property → take element j; scalar → apply to all
                if isinstance(val, list):
                    element_val = float(val[j])
                else:
                    element_val = float(val)

                set_nested(elem, ids_path, element_val)

    def _query_enrichment(self, enrich_def: dict[str, Any]) -> dict[int, dict]:
        """Query enrichment data and index by coil index."""
        query = enrich_def.get("query", "")
        if not query:
            return {}

        with GraphClient() as gc:
            rows = list(gc.query(query, facility=self.facility))

        result: dict[int, dict] = {}
        for row in rows:
            path = row.get("path", "")
            idx = _coil_index_from_path(path)
            result[idx] = dict(row)
        return result

    def export(
        self,
        output_path: Path,
        epoch: str,
        *,
        backend: str = "hdf5",
    ) -> Path:
        """Assemble and write IDS to file.

        Args:
            output_path: Path for the output file (without extension).
            epoch: Epoch version string.
            backend: Storage backend ('hdf5' or 'netcdf').

        Returns:
            Path to the created file.
        """
        import imas

        ids = self.assemble(epoch)

        uri = f"imas:{backend}?path={output_path}"
        entry = imas.DBEntry(uri, "x")
        try:
            entry.put(ids)
        finally:
            entry.close()

        logger.info("Exported %s to %s", self.ids_name, output_path)
        return output_path

    def list_epochs(self) -> list[dict[str, Any]]:
        """List available epochs for this facility from the graph."""
        with GraphClient() as gc:
            return list(
                gc.query(
                    """
                    MATCH (se:SignalEpoch {
                        facility_id: $facility,
                        data_source_name: 'device_xml'
                    })
                    RETURN se.id AS id,
                           se.first_shot AS first_shot,
                           se.last_shot AS last_shot,
                           se.description AS description
                    ORDER BY se.first_shot
                    """,
                    facility=self.facility,
                )
            )

    def summary(self, epoch: str) -> dict[str, Any]:
        """Get assembly summary without building the full IDS."""
        epoch_id = _resolve_epoch_id(self.facility, epoch)
        stats: dict[str, Any] = {
            "ids_name": self.ids_name,
            "facility": self.facility,
            "epoch": epoch_id,
            "dd_version": self.dd_version,
            "arrays": {},
        }

        if self._graph_mapping is not None:
            return self._summary_from_graph(stats, epoch_id)
        return self._summary_from_yaml(stats, epoch_id)

    def _summary_from_graph(
        self, stats: dict[str, Any], epoch_id: str
    ) -> dict[str, Any]:
        """Build summary from graph-driven mapping."""
        mapping = self._graph_mapping
        with GraphClient() as gc:
            for section in mapping.sections:
                root_path = section["root_path"]
                section_name = root_path.rsplit("/", 1)[-1]

                nodes = select_nodes(self.facility, section, epoch_id, gc)
                array_stats: dict[str, Any] = {"count": len(nodes)}

                # Count elements if configured
                elements_raw = section.get("elements_config") or "{}"
                elements_config = (
                    json.loads(elements_raw)
                    if isinstance(elements_raw, str)
                    else elements_raw
                )
                if elements_config and nodes:
                    flat = [_flatten_node(n) for n in nodes]
                    total_elements = 0
                    section_mappings = [
                        m
                        for m in mapping.bindings
                        if m.target_id.startswith(
                            f"{self.ids_name}/{section_name}/element/"
                        )
                    ]
                    for node_data in flat:
                        for m in section_mappings:
                            val = node_data.get(m.source_property)
                            if isinstance(val, list):
                                total_elements += len(val)
                                break
                            else:
                                total_elements += 1
                                break
                    array_stats["total_elements"] = total_elements

                stats["arrays"][section_name] = array_stats
        return stats

    def _summary_from_yaml(
        self, stats: dict[str, Any], epoch_id: str
    ) -> dict[str, Any]:
        """Build summary from YAML recipe."""
        for array_name, array_def in self.recipe.get("arrays", {}).items():
            source_def = array_def.get("source", {})
            query = source_def.get("query", "")
            if not query:
                continue

            with GraphClient() as gc:
                rows = list(gc.query(query, facility=self.facility, epoch_id=epoch_id))

            array_stats: dict[str, Any] = {"count": len(rows)}

            # Count total elements if array has element definitions
            if array_def.get("elements") and rows:
                total_elements = 0
                for row in rows:
                    for source_field in (
                        array_def["elements"].get("fields", {}).values()
                    ):
                        val = row.get(source_field)
                        if isinstance(val, list):
                            total_elements += len(val)
                            break
                        else:
                            total_elements += 1
                            break
                array_stats["total_elements"] = total_elements

            stats["arrays"][array_name] = array_stats

        return stats


def list_recipes(facility: str | None = None) -> list[dict[str, str]]:
    """List available IDS recipes from graph and YAML files.

    Args:
        facility: Optional facility filter.

    Returns:
        List of dicts with 'facility', 'ids_name', 'dd_version', 'source' keys.
    """
    recipes: dict[str, dict[str, str]] = {}

    # Graph recipes
    try:
        with GraphClient() as gc:
            query = """
                MATCH (r:IMASMapping)
                WHERE r.status = 'active'
            """
            params: dict[str, str] = {}
            if facility:
                query += " AND r.facility_id = $facility"
                params["facility"] = facility
            query += """
                RETURN r.facility_id AS facility,
                       r.ids_name AS ids_name,
                       r.dd_version AS dd_version
                ORDER BY r.facility_id, r.ids_name
            """
            for row in gc.query(query, **params):
                key = f"{row['facility']}:{row['ids_name']}"
                recipes[key] = {
                    "facility": row["facility"],
                    "ids_name": row["ids_name"],
                    "dd_version": row.get("dd_version", ""),
                    "source": "graph",
                }
    except Exception:
        logger.debug("Graph recipe query failed", exc_info=True)

    # YAML recipes (only add if not already found in graph)
    search_dirs = [RECIPES_DIR / facility] if facility else RECIPES_DIR.iterdir()

    for facility_dir in search_dirs:
        if not facility_dir.is_dir():
            continue
        for recipe_file in sorted(facility_dir.glob("*.yaml")):
            recipe_data = yaml.safe_load(recipe_file.read_text())
            fac = recipe_data.get("facility_id", facility_dir.name)
            name = recipe_data.get("ids_name", recipe_file.stem)
            key = f"{fac}:{name}"
            if key not in recipes:
                recipes[key] = {
                    "facility": fac,
                    "ids_name": name,
                    "dd_version": recipe_data.get("dd_version", ""),
                    "source": "yaml",
                }
    return list(recipes.values())
