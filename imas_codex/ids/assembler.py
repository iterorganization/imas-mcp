"""IDS assembly engine.

Assembles complete IMAS IDS instances from facility graph data by reading
YAML recipes that define source queries and field mappings, then querying
the knowledge graph and populating imas-python IDS objects.

Architecture:
    Recipe YAML → defines queries + field mappings (declarative)
    Assembler   → reads recipe, queries graph, populates IDS (generic)
    Builders    → per-IDS logic for complex transformations (specific)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

RECIPES_DIR = Path(__file__).parent / "recipes"


def _resolve_epoch_id(facility: str, epoch: str) -> str:
    """Convert a short epoch name to the full epoch ID used in the graph."""
    if epoch.startswith(f"{facility}:"):
        return epoch
    return f"{facility}:device_xml:{epoch}"


def _coil_index_from_path(path: str) -> int:
    """Extract the numeric coil/element index from a DataNode path suffix."""
    return int(path.rsplit(":", 1)[-1])


def _set_nested(obj: Any, dotted_path: str, value: Any) -> None:
    """Set a value on an imas-python object using a dotted path.

    Handles nested structures like 'geometry.rectangle.r' by traversing
    getattr chain and setting the final attribute.
    """
    parts = dotted_path.split(".")
    current = obj
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


class IDSAssembler:
    """Assembles IMAS IDS instances from graph data using YAML recipes.

    Args:
        facility: Facility ID (e.g., 'jet').
        ids_name: IDS name (e.g., 'pf_active').
        recipe_path: Optional explicit path to recipe YAML. If not given,
            looks in recipes/{facility}/{ids_name}.yaml.
    """

    def __init__(
        self,
        facility: str,
        ids_name: str,
        recipe_path: Path | None = None,
    ):
        self.facility = facility
        self.ids_name = ids_name

        if recipe_path is None:
            recipe_path = RECIPES_DIR / facility / f"{ids_name}.yaml"
        if not recipe_path.exists():
            msg = f"No recipe found at {recipe_path}"
            raise FileNotFoundError(msg)

        self.recipe: dict[str, Any] = yaml.safe_load(recipe_path.read_text())
        self._validate_recipe()

    def _validate_recipe(self) -> None:
        """Basic validation of recipe structure."""
        required = {"ids_name", "facility_id", "dd_version"}
        missing = required - set(self.recipe)
        if missing:
            msg = f"Recipe missing required fields: {missing}"
            raise ValueError(msg)

    def assemble(self, epoch: str) -> Any:
        """Build an IDS instance for the given epoch.

        Args:
            epoch: Epoch version string (e.g., 'p68613' or full
                'jet:device_xml:p68613').

        Returns:
            Populated imas-python IDSToplevel object.
        """
        import imas

        epoch_id = _resolve_epoch_id(self.facility, epoch)
        dd_version = self.recipe["dd_version"]

        factory = imas.IDSFactory(dd_version)
        ids = factory.new(self.ids_name)

        # Set static properties
        for dotted_path, value in self.recipe.get("static", {}).items():
            _set_nested(ids, dotted_path, value)

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
                    _set_nested(entry, ids_field, str(value))

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

                _set_nested(elem, ids_path, element_val)

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
                    MATCH (se:StructuralEpoch {
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
            "dd_version": self.recipe["dd_version"],
            "arrays": {},
        }

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
    """List available IDS recipes.

    Args:
        facility: Optional facility filter.

    Returns:
        List of dicts with 'facility', 'ids_name', 'path' keys.
    """
    recipes = []
    search_dirs = [RECIPES_DIR / facility] if facility else RECIPES_DIR.iterdir()

    for facility_dir in search_dirs:
        if not facility_dir.is_dir():
            continue
        for recipe_file in sorted(facility_dir.glob("*.yaml")):
            recipe_data = yaml.safe_load(recipe_file.read_text())
            recipes.append(
                {
                    "facility": recipe_data.get("facility_id", facility_dir.name),
                    "ids_name": recipe_data.get("ids_name", recipe_file.stem),
                    "dd_version": recipe_data.get("dd_version", ""),
                    "path": str(recipe_file),
                }
            )
    return recipes
