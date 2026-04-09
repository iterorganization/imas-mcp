#!/usr/bin/env python3
"""Generate schema_context_data.py from LinkML schemas.

Reads facility.yaml, imas_dd.yaml, common.yaml, and task_groups.yaml to
produce a Python module with all schema context data needed by schema_for().

Usage:
    uv run python scripts/gen_schema_context.py
    uv run python scripts/gen_schema_context.py --force

Output is gitignored and regenerated during ``uv sync``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def generate_schema_context(
    output_path: Path | None = None, force: bool = False
) -> Path:
    """Generate schema_context_data.py from LinkML schemas.

    Args:
        output_path: Override output file location (for testing).
        force: Skip freshness check.

    Returns:
        Path to generated file.
    """
    project_root = get_project_root()
    schemas_dir = project_root / "imas_codex" / "schemas"

    if output_path is None:
        output_path = project_root / "imas_codex" / "graph" / "schema_context_data.py"

    # Source files
    schema_files = [
        schemas_dir / "facility.yaml",
        schemas_dir / "common.yaml",
        schemas_dir / "imas_dd.yaml",
        schemas_dir / "standard_name.yaml",
    ]
    task_groups_file = schemas_dir / "task_groups.yaml"

    existing = [f for f in schema_files if f.exists()]
    if not existing:
        print("[gen-schema-context] No schema files found, skipping")
        return output_path

    # Freshness check
    all_sources = [*existing, task_groups_file]
    if output_path.exists() and not force:
        output_mtime = output_path.stat().st_mtime
        if all(f.stat().st_mtime <= output_mtime for f in all_sources if f.exists()):
            print(f"[gen-schema-context] Up to date: {output_path}")
            return output_path

    # Add project root to path for imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from imas_codex.graph.schema import GraphSchema

    facility_schema = GraphSchema(schemas_dir / "facility.yaml")
    dd_schema = GraphSchema(schemas_dir / "imas_dd.yaml")

    # Load task groups
    with open(task_groups_file) as f:
        task_groups_raw = yaml.safe_load(f)

    # Merge all labels
    all_labels = sorted(set(facility_schema.node_labels + dd_schema.node_labels))

    # Validate task groups against schemas
    for group_name, group in task_groups_raw.items():
        for label in group["labels"]:
            if label not in all_labels:
                msg = (
                    f"Task group '{group_name}' references label '{label}' "
                    f"which is not in any LinkML schema"
                )
                raise ValueError(msg)

    # Helper to pick the right schema for a given label
    def schema_for_label(label: str) -> GraphSchema:
        if label in facility_schema.node_labels:
            return facility_schema
        return dd_schema

    # ---- Build NODE_LABEL_PROPS ----
    node_label_props: dict[str, dict[str, str]] = {}
    for label in all_labels:
        schema = schema_for_label(label)
        slots = schema.get_all_slots(label)
        props: dict[str, str] = {}
        for slot_name, info in slots.items():
            # Skip relationship slots and embedding vectors
            if info.get("relationship"):
                continue
            if slot_name.endswith("embedding"):
                continue

            # Build compact type string
            slot_type = info.get("type", "string")
            flags = []
            if info.get("identifier"):
                flags.append("ID")
            if info.get("required"):
                flags.append("required")
            if info.get("multivalued"):
                flags.append("list")

            type_str = slot_type
            if flags:
                type_str = f"{slot_type} ({', '.join(flags)})"
            props[slot_name] = type_str
        node_label_props[label] = props

    # ---- Build ENUM_VALUES ----
    enum_values: dict[str, list[str]] = {}
    enum_values.update(dd_schema.get_enums())
    enum_values.update(facility_schema.get_enums())

    # ---- Build RELATIONSHIPS ----
    relationships: list[tuple[str, str, str, str]] = []
    seen_rels: set[tuple[str, str, str]] = set()

    for schema in [facility_schema, dd_schema]:
        for rel in schema.relationships:
            key = (rel.from_class, rel.cypher_type, rel.to_class)
            if key not in seen_rels:
                seen_rels.add(key)
                cardinality = "many" if rel.multivalued else "one"
                relationships.append(
                    (rel.from_class, rel.cypher_type, rel.to_class, cardinality)
                )

    relationships.sort()

    # ---- Build VECTOR_INDEXES ----
    vector_indexes: dict[str, tuple[str, str]] = {}
    for schema in [facility_schema, dd_schema]:
        for idx_name, label, prop in schema.vector_indexes:
            if idx_name not in vector_indexes:
                vector_indexes[idx_name] = (label, prop)

    # ---- Build TASK_GROUPS (just the label lists) ----
    task_groups: dict[str, list[str]] = {}
    for group_name, group in task_groups_raw.items():
        task_groups[group_name] = group["labels"]

    # ---- Build FULLTEXT_INDEXES ----
    fulltext_indexes: dict[str, tuple[str, list[str]]] = {}
    for schema in [facility_schema, dd_schema]:
        for idx_name, label, props in schema.fulltext_indexes:
            if idx_name not in fulltext_indexes:
                fulltext_indexes[idx_name] = (label, props)

    # ---- Generate Python module ----
    lines = [
        '"""Auto-generated schema context data from LinkML schemas.',
        "",
        "DO NOT EDIT — regenerated by: uv run build-models --force",
        "Source: imas_codex/schemas/facility.yaml, imas_dd.yaml, common.yaml",
        '"""',
        "",
        "# All node labels with their key properties (excludes embeddings, relationships)",
        f"NODE_LABEL_PROPS: dict[str, dict[str, str]] = {_format_dict(node_label_props)}",
        "",
        "# Enum values (complete, from LinkML)",
        f"ENUM_VALUES: dict[str, list[str]] = {_format_dict(enum_values)}",
        "",
        "# Relationship types with directionality",
        "# (from_label, rel_type, to_label, cardinality)",
        f"RELATIONSHIPS: list[tuple[str, str, str, str]] = {_format_list(relationships)}",
        "",
        "# Vector indexes: index_name -> (label, property)",
        f"VECTOR_INDEXES: dict[str, tuple[str, str]] = {_format_dict(vector_indexes)}",
        "",
        "# Fulltext indexes: index_name -> (label, [properties])",
        f"FULLTEXT_INDEXES: dict[str, tuple[str, list[str]]] = {_format_dict(fulltext_indexes)}",
        "",
        "# Task groups: task_name -> list of relevant labels",
        f"TASK_GROUPS: dict[str, list[str]] = {_format_dict(task_groups)}",
        "",
    ]

    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(
        f"[gen-schema-context] Generated {output_path} "
        f"({len(all_labels)} labels, {len(vector_indexes)} vector indexes, "
        f"{len(fulltext_indexes)} fulltext indexes, "
        f"{len(task_groups)} task groups)"
    )
    return output_path


def _format_dict(d: dict) -> str:
    """Format a dict as a readable Python literal."""
    import pprint

    return pprint.pformat(d, width=100, sort_dicts=True)


def _format_list(lst: list) -> str:
    """Format a list as a readable Python literal."""
    import pprint

    return pprint.pformat(lst, width=100)


if __name__ == "__main__":
    force = "--force" in sys.argv
    generate_schema_context(force=force)
