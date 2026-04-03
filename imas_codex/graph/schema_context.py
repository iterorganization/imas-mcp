"""Runtime schema context for agents — provides task-specific schema slices.

Uses auto-generated data from ``schema_context_data.py`` (produced by
``scripts/gen_schema_context.py`` during ``uv sync``) to give agents
compact, relevant schema context for Cypher query generation.

Example:
    >>> from imas_codex.graph.schema_context import schema_for
    >>> print(schema_for(task="signals"))  # Only signal-related schema
    >>> print(schema_for("Facility", "DataSource"))  # Specific labels
"""

from __future__ import annotations

from imas_codex.graph.schema_context_data import (
    ENUM_VALUES,
    NODE_LABEL_PROPS,
    RELATIONSHIPS,
    TASK_GROUPS,
    VECTOR_INDEXES,
)

# Hand-maintained example Cypher patterns for common tasks.
# These encode domain-specific query recipes that agents commonly need.
_EXAMPLE_PATTERNS: dict[str, list[str]] = {
    "signals": [
        (
            "# Find signals with data access patterns\n"
            "MATCH (s:FacilitySignal)-[:DATA_ACCESS]->(da:DataAccess)\n"
            "WHERE s.facility_id = $facility\n"
            "RETURN s.id, s.name, s.description, da.template_python\n"
            "ORDER BY s.name LIMIT 20"
        ),
        (
            "# Semantic search for signals\n"
            "CALL () {\n"
            "  SEARCH signal:FacilitySignal\n"
            "  USING VECTOR INDEX facility_signal_desc_embedding\n"
            "  WHERE signal.facility_id = $facility\n"
            "  WITH signal, vector.similarity.cosine(signal.embedding, $embedding) AS score\n"
            "  ORDER BY score DESC\n"
            "  LIMIT $k\n"
            "}\n"
            "MATCH (signal)-[:DATA_ACCESS]->(da:DataAccess)\n"
            "RETURN signal.id, signal.description, da.template_python, score\n"
            "ORDER BY score DESC"
        ),
    ],
    "wiki": [
        (
            "# Search wiki chunks with page context\n"
            "CALL () {\n"
            "  SEARCH node:WikiChunk\n"
            "  USING VECTOR INDEX wiki_chunk_embedding\n"
            "  WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score\n"
            "  ORDER BY score DESC\n"
            "  LIMIT $k\n"
            "}\n"
            "MATCH (p:WikiPage)-[:HAS_CHUNK]->(node)\n"
            "RETURN node.text, p.title, p.url, score\n"
            "ORDER BY score DESC"
        ),
    ],
    "imas": [
        (
            "# Search IMAS paths\n"
            "CALL () {\n"
            "  SEARCH node:IMASNode\n"
            "  USING VECTOR INDEX imas_node_embedding\n"
            "  WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score\n"
            "  ORDER BY score DESC\n"
            "  LIMIT $k\n"
            "}\n"
            "WHERE NOT (node)-[:DEPRECATED_IN]->(:DDVersion)\n"
            "RETURN node.id, node.documentation, node.units, score\n"
            "ORDER BY score DESC"
        ),
        (
            "# Get a semantic cluster and its members\n"
            "MATCH (c:IMASSemanticCluster)<-[:IN_CLUSTER]-(p:IMASNode)\n"
            "WHERE c.label = $label\n"
            "RETURN c.label, c.description, collect(p.id) AS paths"
        ),
    ],
    "code": [
        (
            "# Search code chunks\n"
            "CALL () {\n"
            "  SEARCH node:CodeChunk\n"
            "  USING VECTOR INDEX code_chunk_embedding\n"
            "  WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score\n"
            "  ORDER BY score DESC\n"
            "  LIMIT $k\n"
            "}\n"
            "MATCH (cf:CodeFile)-[:HAS_CHUNK]->(node)\n"
            "RETURN node.text, cf.path, cf.facility_id, score\n"
            "ORDER BY score DESC"
        ),
    ],
    "trees": [
        (
            "# List tree nodes for a specific tree\n"
            "MATCH (n:SignalNode {data_source_name: $tree})-[:AT_FACILITY]->(f:Facility {id: $facility})\n"
            "RETURN n.path, n.description, n.unit, n.physics_domain\n"
            "ORDER BY n.path LIMIT $limit"
        ),
    ],
    "facility": [
        (
            "# Facility overview with counts\n"
            "MATCH (f:Facility {id: $facility})\n"
            "OPTIONAL MATCH (d:Diagnostic)-[:AT_FACILITY]->(f)\n"
            "OPTIONAL MATCH (t:DataSource)-[:AT_FACILITY]->(f)\n"
            "RETURN f.id, f.name, count(DISTINCT d) AS diagnostics, count(DISTINCT t) AS trees"
        ),
    ],
}


def schema_for(
    *labels: str,
    task: str | None = None,
    include_relationships: bool = True,
    include_examples: bool = True,
) -> str:
    """Get compact, task-relevant schema context for Cypher query generation.

    Uses auto-generated data from LinkML schemas — always in sync with
    the graph structure.

    Args:
        *labels: Specific node labels to include. Mutually exclusive with task.
        task: Task group name ("signals", "wiki", "imas", "code", "facility",
              "trees") or "overview" for a compact summary of all labels.
        include_relationships: Include relevant relationships in output.
        include_examples: Include example Cypher patterns for the task.

    Returns:
        Compact text schema context suitable for LLM consumption.

    Raises:
        ValueError: If task name or label is unknown.

    Examples:
        >>> schema_for(task="signals")     # Signal-related schema only
        >>> schema_for(task="overview")    # Compact summary of everything
        >>> schema_for("Facility", "DataSource")  # Specific labels
    """
    if task and labels:
        msg = "Specify either task or labels, not both"
        raise ValueError(msg)

    if not task and not labels:
        task = "overview"

    if task == "overview":
        return _build_overview()

    if task:
        if task not in TASK_GROUPS:
            msg = f"Unknown task: '{task}'. Valid: {sorted(TASK_GROUPS.keys())}"
            raise ValueError(msg)
        selected_labels = TASK_GROUPS[task]
    else:
        # Validate explicit labels
        for label in labels:
            if label not in NODE_LABEL_PROPS:
                msg = f"Unknown label: '{label}'. Valid labels: {sorted(NODE_LABEL_PROPS.keys())}"
                raise ValueError(msg)
        selected_labels = list(labels)

    return _build_slice(
        selected_labels,
        task=task,
        include_relationships=include_relationships,
        include_examples=include_examples,
    )


def _build_overview() -> str:
    """Build a compact overview of all labels with key info."""
    lines = ["# Graph Schema Overview", ""]

    # Label summary with identifiers
    lines.append(f"## Node Labels ({len(NODE_LABEL_PROPS)} types)")
    lines.append("")
    for label, props in sorted(NODE_LABEL_PROPS.items()):
        id_prop = next(
            (name for name, typ in props.items() if "ID" in typ),
            None,
        )
        prop_count = len(props)
        id_str = f" (id: {id_prop})" if id_prop else ""
        lines.append(f"- **{label}**{id_str} — {prop_count} properties")
    lines.append("")

    # Vector indexes
    lines.append(f"## Vector Indexes ({len(VECTOR_INDEXES)})")
    lines.append("")
    for idx_name, (label, prop) in sorted(VECTOR_INDEXES.items()):
        lines.append(f"- `{idx_name}` → {label}.{prop}")
    lines.append("")

    # Task groups
    lines.append("## Task Groups (use schema_for(task=...) for details)")
    lines.append("")
    for group_name, group_labels in sorted(TASK_GROUPS.items()):
        lines.append(f"- **{group_name}**: {', '.join(group_labels)}")
    lines.append("")

    # Relationship types summary
    rel_types = sorted({rel[1] for rel in RELATIONSHIPS})
    lines.append(f"## Relationship Types ({len(rel_types)})")
    lines.append("")
    lines.append(", ".join(f"`{r}`" for r in rel_types))
    lines.append("")

    return "\n".join(lines)


def _build_slice(
    selected_labels: list[str],
    task: str | None = None,
    include_relationships: bool = True,
    include_examples: bool = True,
) -> str:
    """Build detailed schema context for a set of labels."""
    lines = []
    task_desc = f" (task: {task})" if task else ""
    lines.append(f"# Schema Context{task_desc}")
    lines.append("")

    # Node details
    for label in sorted(selected_labels):
        if label not in NODE_LABEL_PROPS:
            continue
        props = NODE_LABEL_PROPS[label]
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Property | Type |")
        lines.append("|----------|------|")
        for prop_name, prop_type in sorted(props.items()):
            lines.append(f"| {prop_name} | {prop_type} |")
        lines.append("")

    # Relevant relationships
    if include_relationships:
        label_set = set(selected_labels)
        relevant_rels = [
            rel for rel in RELATIONSHIPS if rel[0] in label_set or rel[2] in label_set
        ]
        if relevant_rels:
            lines.append("## Relationships")
            lines.append("")
            for from_label, rel_type, to_label, cardinality in relevant_rels:
                multi = " (many)" if cardinality == "many" else ""
                lines.append(f"- ({from_label})-[:{rel_type}]->({to_label}){multi}")
            lines.append("")

    # Relevant vector indexes
    label_set = set(selected_labels)
    relevant_indexes = {
        name: (label, prop)
        for name, (label, prop) in VECTOR_INDEXES.items()
        if label in label_set
    }
    if relevant_indexes:
        lines.append("## Vector Indexes")
        lines.append("")
        for idx_name, (label, prop) in sorted(relevant_indexes.items()):
            lines.append(f"- `{idx_name}` → {label}.{prop}")
        lines.append("")

    # Relevant enums
    relevant_enums = _find_relevant_enums(selected_labels)
    if relevant_enums:
        lines.append("## Enums")
        lines.append("")
        for enum_name, values in sorted(relevant_enums.items()):
            lines.append(f"- **{enum_name}**: {', '.join(f'`{v}`' for v in values)}")
        lines.append("")

    # Example Cypher patterns
    if include_examples and task and task in _EXAMPLE_PATTERNS:
        lines.append("## Example Cypher")
        lines.append("")
        for pattern in _EXAMPLE_PATTERNS[task]:
            lines.append("```cypher")
            lines.append(pattern)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def _find_relevant_enums(labels: list[str]) -> dict[str, list[str]]:
    """Find enum types used by the given labels' properties."""
    # Build a set of type names used by the labels
    type_names: set[str] = set()
    for label in labels:
        if label in NODE_LABEL_PROPS:
            for prop_type in NODE_LABEL_PROPS[label].values():
                # Extract base type (before parenthetical flags)
                base_type = prop_type.split(" (")[0].strip()
                type_names.add(base_type)

    # Match against known enums
    return {name: values for name, values in ENUM_VALUES.items() if name in type_names}
