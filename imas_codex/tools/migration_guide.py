"""DD migration guide generator.

Assembles structured migration reports between DD versions using
graph-stored change data, COCOS transformations, and breaking change
classifications.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


def _resolve_version_range(gc: GraphClient, from_ver: str, to_ver: str) -> list[str]:
    """Get ordered list of all versions between from_ver and to_ver (exclusive of from, inclusive of to)."""
    result = gc.query(
        """
        MATCH (v:DDVersion)
        WHERE v.id > $from_ver AND v.id <= $to_ver
        RETURN v.id AS version
        ORDER BY v.id
        """,
        from_ver=from_ver,
        to_ver=to_ver,
    )
    return [r["version"] for r in result]


def _get_change_summary(gc: GraphClient, version_range: list[str]) -> list[dict]:
    """Get aggregated change counts by type and breaking level."""
    return gc.query(
        """
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions
        WITH c.change_type AS type,
             coalesce(c.breaking_level, 'informational') AS level,
             count(c) AS cnt
        RETURN type, level, cnt
        ORDER BY cnt DESC
        """,
        versions=version_range,
    )


def _get_cocos_table(
    gc: GraphClient,
    to_versions: list[str],
    ids_filter: str | None = None,
) -> list[dict]:
    """Get COCOS-labeled paths, merging from/to version data."""
    ids_clause = "AND p.ids_name = $ids_filter" if ids_filter else ""

    result = gc.query(
        f"""
        MATCH (p:IMASNode)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $to_versions
          AND (p.cocos_label_transformation IS NOT NULL
               OR p.cocos_label IS NOT NULL
               OR p.cocos_transformation_expression IS NOT NULL)
          AND p.node_category = 'data'
          {ids_clause}
        RETURN p.ids_name AS ids, p.path AS path,
               coalesce(p.cocos_label, p.cocos_label_transformation) AS label,
               p.cocos_transformation_expression AS expr,
               coalesce(p.cocos_label_source, 'xml') AS source
        ORDER BY p.ids_name, p.path
        """,
        to_versions=to_versions,
        ids_filter=ids_filter,
    )

    # For paths with expression but no label, infer from expression
    import re

    for row in result:
        if not row.get("label") and row.get("expr"):
            m = re.search(r"\{(\w+_like)\}", row["expr"])
            if m:
                row["label"] = m.group(1)
            elif "fact_psi" in (row["expr"] or ""):
                row["label"] = "psi_like"
            elif "sigma_ip" in (row["expr"] or ""):
                row["label"] = "ip_like"
            elif "sigma_b0" in (row["expr"] or ""):
                row["label"] = "b0_like"

    return [r for r in result if r.get("label")]


def _get_renames(gc: GraphClient, version_range: list[str]) -> list[dict]:
    """Get path renames in the version range."""
    return gc.query(
        """
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions AND c.change_type = 'path_renamed'
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        RETURN p.ids AS ids, c.old_value AS old_path, c.new_value AS new_path
        ORDER BY p.ids, c.old_value
        """,
        versions=version_range,
    )


def _get_unit_changes(gc: GraphClient, version_range: list[str]) -> list[dict]:
    """Get unit changes (non-cosmetic) in the version range."""
    return gc.query(
        """
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions AND c.change_type = 'units'
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        RETURN p.ids AS ids, p.id AS path, c.old_value AS old_unit,
               c.new_value AS new_unit,
               coalesce(c.unit_change_subtype, 'unknown') AS subtype,
               coalesce(c.breaking_level, 'informational') AS level
        ORDER BY c.breaking_level DESC, p.ids, p.id
        """,
        versions=version_range,
    )


def _get_type_changes(gc: GraphClient, version_range: list[str]) -> list[dict]:
    """Get data type changes in the version range."""
    return gc.query(
        """
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions AND c.change_type = 'data_type'
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        RETURN p.ids AS ids, p.id AS path, c.old_value AS old_type,
               c.new_value AS new_type
        ORDER BY p.ids, p.id
        """,
        versions=version_range,
    )


def _get_removals(gc: GraphClient, version_range: list[str]) -> list[dict]:
    """Get removed paths with potential replacements."""
    return gc.query(
        """
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions AND c.change_type = 'path_removed'
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        OPTIONAL MATCH (p)-[:RENAMED_TO]->(replacement:IMASNode)
        RETURN p.ids AS ids, p.id AS path,
               replacement.id AS replacement
        ORDER BY p.ids, p.id
        """,
        versions=version_range,
    )


def _get_additions(gc: GraphClient, version_range: list[str]) -> list[dict]:
    """Get added paths in the version range."""
    return gc.query(
        """
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions AND c.change_type = 'path_added'
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        RETURN p.ids AS ids, p.id AS path
        ORDER BY p.ids, p.id
        """,
        versions=version_range,
    )


def _compute_cocos_factors(
    labeled_paths: list[dict], from_cocos: int | None, to_cocos: int | None
) -> list[dict]:
    """Compute sign/scale factors for COCOS-labeled paths."""
    if not labeled_paths:
        return []

    can_compute = (
        from_cocos is not None and to_cocos is not None and from_cocos != to_cocos
    )

    results = []
    for p in labeled_paths:
        if can_compute:
            try:
                from imas_codex.ids.transforms import cocos_sign

                factor = cocos_sign(p["label"], cocos_in=from_cocos, cocos_out=to_cocos)
            except Exception:
                factor = None
        else:
            factor = None

        action = (
            "Unknown (COCOS IDs not set)"
            if factor is None
            else ("No change needed" if factor == 1 else f"Multiply by {factor}")
        )
        results.append({**p, "factor": factor, "action": action})

    return results


def _get_version_cocos(gc: GraphClient, version: str) -> int | None:
    """Get COCOS convention for a specific version."""
    result = gc.query(
        "MATCH (v:DDVersion {id: $ver}) RETURN v.cocos_id AS cocos",
        ver=version,
    )
    if result and result[0].get("cocos"):
        return result[0]["cocos"]
    # Fallback: DD 3.x uses COCOS 11, DD 4.x uses COCOS 17
    if version.startswith("3."):
        return 11
    if version.startswith("4."):
        return 17
    return None


def _render_guide(
    from_ver: str,
    to_ver: str,
    version_range: list[str],
    summary: list[dict],
    cocos_table: list[dict],
    renames: list[dict],
    unit_changes: list[dict],
    type_changes: list[dict],
    removals: list[dict],
    additions: list[dict],
    from_cocos: int | None,
    to_cocos: int | None,
    ids_filter: str | None,
    include_recipes: bool,
) -> str:
    """Render the migration guide as structured markdown."""
    lines = [f"# DD Migration Guide: {from_ver} → {to_ver}\n"]

    if ids_filter:
        cocos_table = [r for r in cocos_table if r["ids"] == ids_filter]
        renames = [r for r in renames if r["ids"] == ids_filter]
        unit_changes = [r for r in unit_changes if r["ids"] == ids_filter]
        type_changes = [r for r in type_changes if r["ids"] == ids_filter]
        removals = [r for r in removals if r["ids"] == ids_filter]
        additions = [r for r in additions if r["ids"] == ids_filter]

    total_breaking = sum(r["cnt"] for r in summary if r["level"] == "breaking")
    total_advisory = sum(r["cnt"] for r in summary if r["level"] == "advisory")
    total_info = sum(r["cnt"] for r in summary if r["level"] == "informational")

    # Count COCOS sign flips as breaking changes
    sign_flip_count = sum(
        1 for p in cocos_table if p.get("factor") is not None and p["factor"] != 1
    )
    total_breaking += sign_flip_count
    total = total_breaking + total_advisory + total_info

    lines.append("## Summary\n")
    if from_cocos and to_cocos and from_cocos != to_cocos:
        lines.append(f"- **COCOS convention change:** {from_cocos} → {to_cocos}")
    lines.append(f"- **Versions spanned:** {len(version_range)}")
    lines.append(f"- **Total changes:** {total}")
    lines.append(f"- **Breaking changes:** {total_breaking}")
    if sign_flip_count:
        lines.append(
            f"- **{sign_flip_count} COCOS sign flips** (breaking — codes must multiply affected quantities)"
        )
    lines.append(f"- **Advisory changes:** {total_advisory}")
    lines.append(f"- **Informational:** {total_info}")
    lines.append(f"- **Path renames:** {len(renames)}")
    lines.append(f"- **Paths removed:** {len(removals)}")
    lines.append(f"- **Paths added:** {len(additions)}")

    affected_ids: dict[str, int] = {}
    for section in [renames, unit_changes, type_changes, removals]:
        for r in section:
            ids_name = r.get("ids", "unknown")
            affected_ids[ids_name] = affected_ids.get(ids_name, 0) + 1
    if affected_ids:
        top_ids = sorted(affected_ids.items(), key=lambda x: -x[1])[:10]
        ids_str = ", ".join(f"{name} ({cnt})" for name, cnt in top_ids)
        lines.append(f"- **IDS most affected:** {ids_str}")
    lines.append("")

    if cocos_table:
        if from_cocos and to_cocos and from_cocos != to_cocos:
            lines.append(
                f"\n## COCOS Sign-Flip Table (COCOS {from_cocos} → {to_cocos})\n"
            )
        else:
            lines.append("\n## COCOS-Sensitive Paths\n")
            lines.append(
                "*COCOS convention IDs not set on DDVersion nodes — factors not computed.*\n"
            )
        lines.append("| IDS | Path | Label | Factor | Action |")
        lines.append("|-----|------|-------|--------|--------|")
        for r in cocos_table:
            factor = r.get("factor") if r.get("factor") is not None else "?"
            action = r.get("action", "Check manually")
            path_short = r["path"].split("/", 1)[-1] if "/" in r["path"] else r["path"]
            lines.append(
                f"| {r['ids']} | {path_short} | {r['label']} | {factor} | {action} |"
            )
        lines.append("")

    if removals:
        lines.append(f"## Removed Paths ({len(removals)})\n")
        lines.append("| IDS | Path | Replacement |")
        lines.append("|-----|------|-------------|")
        for r in removals:
            replacement = r.get("replacement") or "—"
            lines.append(f"| {r['ids']} | {r['path']} | {replacement} |")
        lines.append("")

    if renames:
        lines.append(f"## Path Renames ({len(renames)})\n")
        lines.append("| IDS | Old Path | New Path |")
        lines.append("|-----|----------|----------|")
        for r in renames:
            lines.append(f"| {r['ids']} | {r['old_path']} | {r['new_path']} |")
        lines.append("")

    breaking_units = [r for r in unit_changes if r["level"] == "breaking"]
    advisory_units = [r for r in unit_changes if r["level"] == "advisory"]
    if breaking_units or advisory_units:
        lines.append(
            f"## Unit Changes ({len(breaking_units)} breaking, {len(advisory_units)} advisory)\n"
        )
        lines.append("| IDS | Path | Old Unit | New Unit | Severity |")
        lines.append("|-----|------|----------|----------|----------|")
        for r in breaking_units + advisory_units:
            lines.append(
                f"| {r['ids']} | {r['path']} | {r['old_unit']} | {r['new_unit']} | **{r['level']}** |"
            )
        lines.append("")

    if type_changes:
        lines.append(f"## Type Changes ({len(type_changes)})\n")
        lines.append("| IDS | Path | Old Type | New Type |")
        lines.append("|-----|------|----------|----------|")
        for r in type_changes:
            lines.append(
                f"| {r['ids']} | {r['path']} | {r['old_type']} | {r['new_type']} |"
            )
        lines.append("")

    if include_recipes and (renames or cocos_table):
        lines.append("## Code Update Recipes\n")

        if cocos_table:
            lines.append("### COCOS Sign Flip\n")
            lines.append("```python")
            lines.append("from imas_codex.ids.transforms import cocos_sign")
            lines.append("")
            example = cocos_table[0] if cocos_table else None
            if example and from_cocos and to_cocos:
                lines.append(
                    f'factor = cocos_sign("{example["label"]}", cocos_in={from_cocos}, cocos_out={to_cocos})'
                )
                lines.append(f"# factor = {example.get('factor', '...')}")
                lines.append(
                    f"# Apply: ids.{example['path'].replace('/', '.')} *= factor"
                )
            elif example:
                lines.append(
                    f'factor = cocos_sign("{example["label"]}", cocos_in=<FROM>, cocos_out=<TO>)'
                )
                lines.append(
                    f"# Apply: ids.{example['path'].replace('/', '.')} *= factor"
                )
            lines.append("```\n")

        if renames:
            lines.append("### Path Renames\n")
            lines.append("```python")
            for r in renames[:5]:
                lines.append(f"# {r['old_path']}  →  {r['new_path']}")
            if len(renames) > 5:
                lines.append(f"# ... and {len(renames) - 5} more renames")
            lines.append("```\n")

    return "\n".join(lines)


def generate_migration_guide(
    gc: GraphClient,
    from_version: str,
    to_version: str,
    ids_filter: str | None = None,
    include_recipes: bool = True,
) -> str:
    """Generate a DD migration guide between two versions.

    Args:
        gc: Graph client
        from_version: Source DD version (e.g. "3.39.0")
        to_version: Target DD version (e.g. "4.0.0")
        ids_filter: Optional IDS name to restrict output
        include_recipes: Whether to include code update snippets

    Returns:
        Structured markdown migration guide
    """
    for ver in (from_version, to_version):
        check = gc.query("MATCH (v:DDVersion {id: $v}) RETURN v.id", v=ver)
        if not check:
            return f"Error: DD version '{ver}' not found in graph."

    version_range = _resolve_version_range(gc, from_version, to_version)
    if not version_range:
        return f"Error: No versions found between {from_version} and {to_version}."

    summary = _get_change_summary(gc, version_range)
    cocos_paths = _get_cocos_table(gc, version_range, ids_filter)
    renames = _get_renames(gc, version_range)
    unit_changes = _get_unit_changes(gc, version_range)
    type_changes = _get_type_changes(gc, version_range)
    removals = _get_removals(gc, version_range)
    additions = _get_additions(gc, version_range)

    from_cocos = _get_version_cocos(gc, from_version)
    to_cocos = _get_version_cocos(gc, to_version)

    cocos_table = _compute_cocos_factors(cocos_paths, from_cocos, to_cocos)

    return _render_guide(
        from_ver=from_version,
        to_ver=to_version,
        version_range=version_range,
        summary=summary,
        cocos_table=cocos_table,
        renames=renames,
        unit_changes=unit_changes,
        type_changes=type_changes,
        removals=removals,
        additions=additions,
        from_cocos=from_cocos,
        to_cocos=to_cocos,
        ids_filter=ids_filter,
        include_recipes=include_recipes,
    )
