"""DD migration guide generator.

Assembles structured code-migration guides between DD versions using
graph-stored change data, COCOS transformations, and breaking change
classifications. Produces language-agnostic advice with search patterns.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from imas_codex.models.migration_models import (
    CocosMigrationAdvice,
    CodeMigrationGuide,
    CodeUpdateAction,
    PathUpdateAdvice,
    TypeUpdateAdvice,
    generate_search_patterns,
)

if TYPE_CHECKING:
    from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph query helpers
# ---------------------------------------------------------------------------


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


def _get_change_summary(
    gc: GraphClient, version_range: list[str], ids_filter: str | None = None
) -> list[dict]:
    """Get aggregated change counts by type and breaking level."""
    ids_clause = ""
    params: dict = {"versions": version_range}
    if ids_filter:
        ids_clause = (
            "MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode) WHERE p.ids = $ids_filter WITH c"
        )
        params["ids_filter"] = ids_filter

    return gc.query(
        f"""
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions
        {ids_clause}
        WITH c.change_type AS type,
             coalesce(c.breaking_level, 'informational') AS level,
             count(c) AS cnt
        RETURN type, level, cnt
        ORDER BY cnt DESC
        """,
        **params,
    )


def _get_cocos_table(
    gc: GraphClient,
    to_versions: list[str],
    ids_filter: str | None = None,
) -> list[dict]:
    """Get COCOS-labeled paths from the graph.

    IMASNode stores COCOS data directly as properties — no version
    relationship needed (labels don't change between versions).
    """
    ids_clause = "AND p.ids = $ids_filter" if ids_filter else ""

    result = gc.query(
        f"""
        MATCH (p:IMASNode)
        WHERE (p.cocos_label_transformation IS NOT NULL
               OR p.cocos_transformation_expression IS NOT NULL)
          AND p.node_category = 'data'
          {ids_clause}
        RETURN p.ids AS ids, p.id AS path,
               p.cocos_label_transformation AS label,
               p.cocos_transformation_expression AS expr,
               coalesce(p.cocos_label_source, 'xml') AS source
        ORDER BY p.ids, p.id
        """,
        to_versions=to_versions,
        ids_filter=ids_filter,
    )

    # For paths with expression but no label, infer from expression
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


def _get_renames(
    gc: GraphClient, version_range: list[str], ids_filter: str | None = None
) -> list[dict]:
    """Get path renames in the version range."""
    ids_clause = ""
    params: dict = {"versions": version_range}
    if ids_filter:
        ids_clause = "AND p.ids = $ids_filter"
        params["ids_filter"] = ids_filter
    return gc.query(
        f"""
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions AND c.change_type = 'path_renamed'
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        WHERE true {ids_clause}
        RETURN p.ids AS ids, c.old_value AS old_path, c.new_value AS new_path
        ORDER BY p.ids, c.old_value
        """,
        **params,
    )


def _get_unit_changes(
    gc: GraphClient, version_range: list[str], ids_filter: str | None = None
) -> list[dict]:
    """Get unit changes (non-cosmetic) in the version range."""
    ids_clause = ""
    params: dict = {"versions": version_range}
    if ids_filter:
        ids_clause = "AND p.ids = $ids_filter"
        params["ids_filter"] = ids_filter
    return gc.query(
        f"""
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions AND c.change_type = 'units'
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        WHERE true {ids_clause}
        RETURN p.ids AS ids, p.id AS path, c.old_value AS old_unit,
               c.new_value AS new_unit,
               coalesce(c.unit_change_subtype, 'unknown') AS subtype,
               coalesce(c.breaking_level, 'informational') AS level
        ORDER BY c.breaking_level DESC, p.ids, p.id
        """,
        **params,
    )


def _get_type_changes(
    gc: GraphClient, version_range: list[str], ids_filter: str | None = None
) -> list[dict]:
    """Get data type changes in the version range."""
    ids_clause = ""
    params: dict = {"versions": version_range}
    if ids_filter:
        ids_clause = "AND p.ids = $ids_filter"
        params["ids_filter"] = ids_filter
    return gc.query(
        f"""
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions AND c.change_type = 'data_type'
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        WHERE true {ids_clause}
        RETURN p.ids AS ids, p.id AS path, c.old_value AS old_type,
               c.new_value AS new_type
        ORDER BY p.ids, p.id
        """,
        **params,
    )


def _get_removals(
    gc: GraphClient, version_range: list[str], ids_filter: str | None = None
) -> list[dict]:
    """Get removed paths with potential replacements."""
    ids_clause = ""
    params: dict = {"versions": version_range}
    if ids_filter:
        ids_clause = "AND p.ids = $ids_filter"
        params["ids_filter"] = ids_filter
    return gc.query(
        f"""
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions AND c.change_type = 'path_removed'
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        WHERE true {ids_clause}
        OPTIONAL MATCH (p)-[:RENAMED_TO]->(replacement:IMASNode)
        RETURN p.ids AS ids, p.id AS path,
               replacement.id AS replacement
        ORDER BY p.ids, p.id
        """,
        **params,
    )


def _get_additions(
    gc: GraphClient, version_range: list[str], ids_filter: str | None = None
) -> list[dict]:
    """Get added paths in the version range."""
    ids_clause = ""
    params: dict = {"versions": version_range}
    if ids_filter:
        ids_clause = "AND p.ids = $ids_filter"
        params["ids_filter"] = ids_filter
    return gc.query(
        f"""
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions AND c.change_type = 'path_added'
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        WHERE true {ids_clause}
        RETURN p.ids AS ids, p.id AS path
        ORDER BY p.ids, p.id
        """,
        **params,
    )


def _get_semantic_doc_changes(
    gc: GraphClient, version_range: list[str], ids_filter: str | None = None
) -> list[dict]:
    """Get documentation changes with physics significance (sign/coordinate/format conventions)."""
    ids_clause = ""
    params: dict = {"versions": version_range}
    if ids_filter:
        ids_clause = "AND p.ids = $ids_filter"
        params["ids_filter"] = ids_filter
    return gc.query(
        f"""
        MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
        WHERE v.id IN $versions
          AND c.change_type = 'documentation'
          AND c.semantic_type IN ['sign_convention', 'coordinate_convention', 'data_format_change']
        MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
        WHERE true {ids_clause}
        RETURN p.ids AS ids, p.id AS path,
               c.semantic_type AS semantic_type,
               c.old_value AS old_doc, c.new_value AS new_doc,
               coalesce(c.breaking_level, 'informational') AS level,
               c.keywords_detected AS keywords,
               v.id AS version
        ORDER BY c.breaking_level DESC, c.semantic_type, p.ids, p.id
        """,
        **params,
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
        "MATCH (v:DDVersion {id: $ver}) RETURN v.cocos AS cocos",
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


# ---------------------------------------------------------------------------
# Legacy rendering (kept for backward compatibility with existing tests)
# ---------------------------------------------------------------------------


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
    """Render the migration guide as structured markdown (legacy format)."""
    lines = [f"# DD Migration Guide: {from_ver} \u2192 {to_ver}\n"]

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
        lines.append(f"- **COCOS convention change:** {from_cocos} \u2192 {to_cocos}")
    lines.append(f"- **Versions spanned:** {len(version_range)}")
    lines.append(f"- **Total changes:** {total}")
    lines.append(f"- **Breaking changes:** {total_breaking}")
    if sign_flip_count:
        lines.append(
            f"- **{sign_flip_count} COCOS sign flips** (breaking \u2014 codes must multiply affected quantities)"
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
                f"\n## COCOS Sign-Flip Table (COCOS {from_cocos} \u2192 {to_cocos})\n"
            )
        else:
            lines.append("\n## COCOS-Sensitive Paths\n")
            lines.append(
                "*COCOS convention IDs not set on DDVersion nodes \u2014 factors not computed.*\n"
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
            replacement = r.get("replacement") or "\u2014"
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
                lines.append(f"# {r['old_path']}  \u2192  {r['new_path']}")
            if len(renames) > 5:
                lines.append(f"# ... and {len(renames) - 5} more renames")
            lines.append("```\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# New structured migration guide
# ---------------------------------------------------------------------------


def build_migration_guide(
    gc: GraphClient,
    from_version: str,
    to_version: str,
    ids_filter: str | None = None,
    include_recipes: bool = True,
) -> CodeMigrationGuide:
    """Build a structured code migration guide.

    Queries graph for COCOS changes, path renames, unit changes, type
    changes, removals and additions, then classifies each as required
    or optional and attaches language-agnostic search patterns.
    """
    version_range = _resolve_version_range(gc, from_version, to_version)

    from_cocos = _get_version_cocos(gc, from_version)
    to_cocos = _get_version_cocos(gc, to_version)
    cocos_change = None
    if from_cocos and to_cocos and from_cocos != to_cocos:
        cocos_change = f"{from_cocos} \u2192 {to_cocos}"

    required_actions: list[CodeUpdateAction] = []
    optional_actions: list[CodeUpdateAction] = []
    ids_affected: set[str] = set()
    search_patterns: dict[str, list[str]] = {}

    # --- COCOS sign flip actions ---
    cocos_advice = None
    if cocos_change:
        cocos_paths = _get_cocos_table(gc, version_range, ids_filter)
        cocos_table = _compute_cocos_factors(cocos_paths, from_cocos, to_cocos)
        sign_flips: list[dict] = []
        no_change: list[dict] = []

        for entry in cocos_table:
            path = entry.get("path", "")
            factor = entry.get("factor", 1)
            ids_name = entry.get("ids", path.split("/")[0] if "/" in path else "")
            ids_affected.add(ids_name)

            patterns = generate_search_patterns(path, "cocos_sign_flip")
            search_patterns.setdefault(ids_name, []).extend(patterns)

            action = CodeUpdateAction(
                path=path,
                ids=ids_name,
                change_type="cocos_sign_flip",
                severity="required"
                if factor is not None and factor != 1
                else "optional",
                search_patterns=patterns,
                path_fragments=path.split("/")[1:],
                description=f"COCOS {from_cocos}\u2192{to_cocos}: multiply by {factor}",
                before=f"Value in COCOS {from_cocos}",
                after=f"Value in COCOS {to_cocos}, factor={factor}",
                cocos_label=entry.get("label", ""),
                cocos_factor=float(factor) if factor is not None else None,
            )

            if factor is not None and factor != 1:
                required_actions.append(action)
                sign_flips.append(
                    {
                        "path": path,
                        "factor": factor,
                        "label": entry.get("label", ""),
                    }
                )
            else:
                optional_actions.append(action)
                no_change.append(
                    {
                        "path": path,
                        "factor": factor,
                        "label": entry.get("label", ""),
                    }
                )

        cocos_advice = CocosMigrationAdvice(
            from_cocos=from_cocos,
            to_cocos=to_cocos,
            sign_flips=sign_flips,
            no_change=no_change,
        )

    # --- Path renames ---
    renames = _get_renames(gc, version_range, ids_filter)
    renamed_list: list[dict] = []
    removed_list: list[dict] = []
    new_list: list[dict] = []

    for r in renames:
        old_path = r.get("old_path", "")
        new_path_val = r.get("new_path", "")
        ids_name = r.get("ids", old_path.split("/")[0] if "/" in old_path else "")
        ids_affected.add(ids_name)
        patterns = generate_search_patterns(old_path, "path_rename")
        search_patterns.setdefault(ids_name, []).extend(patterns)

        required_actions.append(
            CodeUpdateAction(
                path=old_path,
                ids=ids_name,
                change_type="path_rename",
                severity="required",
                search_patterns=patterns,
                path_fragments=old_path.split("/")[1:],
                description=f"Path renamed: {old_path} \u2192 {new_path_val}",
                before=f"Access {old_path}",
                after=f"Access {new_path_val}",
                old_path=old_path,
                new_path=new_path_val,
            )
        )
        renamed_list.append({"old_path": old_path, "new_path": new_path_val})

    # --- Removals ---
    removals = _get_removals(gc, version_range, ids_filter)
    for rm in removals:
        path = rm.get("path", "")
        ids_name = rm.get("ids", path.split("/")[0] if "/" in path else "")
        ids_affected.add(ids_name)
        patterns = generate_search_patterns(path, "path_removed")
        search_patterns.setdefault(ids_name, []).extend(patterns)

        replacement = rm.get("replacement")
        desc = f"Path removed: {path}"
        if replacement:
            desc += f" (replacement: {replacement})"

        required_actions.append(
            CodeUpdateAction(
                path=path,
                ids=ids_name,
                change_type="path_removed",
                severity="required",
                search_patterns=patterns,
                path_fragments=path.split("/")[1:],
                description=desc,
                before=f"Access {path}",
                after=f"Use {replacement}" if replacement else "Path no longer exists",
                old_path=path,
                new_path=replacement,
            )
        )
        removed_list.append({"path": path, "replacement": replacement})

    # --- Additions ---
    additions = _get_additions(gc, version_range, ids_filter)
    for add in additions:
        path = add.get("path", "")
        ids_name = add.get("ids", path.split("/")[0] if "/" in path else "")
        ids_affected.add(ids_name)

        optional_actions.append(
            CodeUpdateAction(
                path=path,
                ids=ids_name,
                change_type="new_path",
                severity="optional",
                description=f"New path available: {path}",
                after=f"New field {path} is now available",
            )
        )
        new_list.append({"path": path})

    path_update_advice = None
    if renamed_list or removed_list or new_list:
        path_update_advice = PathUpdateAdvice(
            renamed_paths=renamed_list,
            removed_paths=removed_list,
            new_paths=new_list,
        )

    # --- Unit changes ---
    unit_changes = _get_unit_changes(gc, version_range, ids_filter)
    for uc in unit_changes:
        path = uc.get("path", "")
        ids_name = uc.get("ids", path.split("/")[0] if "/" in path else "")
        ids_affected.add(ids_name)
        patterns = generate_search_patterns(path, "unit_change")
        search_patterns.setdefault(ids_name, []).extend(patterns)
        level = uc.get("level", "informational")

        action = CodeUpdateAction(
            path=path,
            ids=ids_name,
            change_type="unit_change",
            severity="required" if level == "breaking" else "optional",
            search_patterns=patterns,
            path_fragments=path.split("/")[1:],
            description=f"Unit change: {uc.get('old_unit', '?')} \u2192 {uc.get('new_unit', '?')}",
            before=f"Units: {uc.get('old_unit', '?')}",
            after=f"Units: {uc.get('new_unit', '?')}",
            old_units=uc.get("old_unit"),
            new_units=uc.get("new_unit"),
        )
        if level == "breaking":
            required_actions.append(action)
        else:
            optional_actions.append(action)

    # --- Type changes ---
    type_changes = _get_type_changes(gc, version_range, ids_filter)
    type_advice = None
    if type_changes:
        tc_list: list[dict] = []
        for tc in type_changes:
            path = tc.get("path", "")
            ids_name = tc.get("ids", path.split("/")[0] if "/" in path else "")
            ids_affected.add(ids_name)
            patterns = generate_search_patterns(path, "type_change")
            search_patterns.setdefault(ids_name, []).extend(patterns)

            required_actions.append(
                CodeUpdateAction(
                    path=path,
                    ids=ids_name,
                    change_type="type_change",
                    severity="required",
                    search_patterns=patterns,
                    path_fragments=path.split("/")[1:],
                    description=f"Type change: {tc.get('old_type', '?')} \u2192 {tc.get('new_type', '?')}",
                    before=f"Type: {tc.get('old_type', '?')}",
                    after=f"Type: {tc.get('new_type', '?')}",
                    old_type=tc.get("old_type"),
                    new_type=tc.get("new_type"),
                )
            )
            tc_list.append(
                {
                    "path": path,
                    "old_type": tc.get("old_type"),
                    "new_type": tc.get("new_type"),
                }
            )
        type_advice = TypeUpdateAdvice(type_changes=tc_list)

    # --- Semantic documentation changes (sign/coordinate conventions) ---
    semantic_doc_changes = _get_semantic_doc_changes(gc, version_range, ids_filter)
    for sdc in semantic_doc_changes:
        path = sdc.get("path", "")
        ids_name = sdc.get("ids", path.split("/")[0] if "/" in path else "")
        ids_affected.add(ids_name)
        patterns = generate_search_patterns(path, "definition_change")
        search_patterns.setdefault(ids_name, []).extend(patterns)
        level = sdc.get("level", "informational")
        semantic = sdc.get("semantic_type", "")

        desc = f"Convention change ({semantic}): {path}"
        old_excerpt = (sdc.get("old_doc") or "")[:200]
        new_excerpt = (sdc.get("new_doc") or "")[:200]

        action = CodeUpdateAction(
            path=path,
            ids=ids_name,
            change_type="definition_change",
            severity="required" if level == "breaking" else "optional",
            search_patterns=patterns,
            path_fragments=path.split("/")[1:],
            description=desc,
            before=old_excerpt,
            after=new_excerpt,
        )
        if level == "breaking":
            required_actions.append(action)
        else:
            optional_actions.append(action)

    # Deduplicate search patterns per IDS
    for ids_name in search_patterns:
        search_patterns[ids_name] = list(dict.fromkeys(search_patterns[ids_name]))

    return CodeMigrationGuide(
        from_version=from_version,
        to_version=to_version,
        cocos_change=cocos_change,
        required_actions=required_actions,
        optional_actions=optional_actions,
        total_actions=len(required_actions) + len(optional_actions),
        required_count=len(required_actions),
        optional_count=len(optional_actions),
        ids_affected=sorted(ids_affected - {""}),
        global_search_patterns=search_patterns,
        cocos_advice=cocos_advice,
        path_update_advice=path_update_advice,
        type_update_advice=type_advice,
    )


def format_migration_guide(guide: CodeMigrationGuide) -> str:
    """Render a *CodeMigrationGuide* to markdown.

    Sections:
      1. Executive summary (versions, COCOS, action counts)
      2. Required updates (grouped by IDS)
      3. COCOS migration table
      4. Search strategy with grep patterns
      5. Optional improvements
      6. Verification checklist
    """
    lines: list[str] = []

    # -- Executive summary --
    lines.append(
        f"# Code Migration Guide: DD {guide.from_version} \u2192 {guide.to_version}"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- **Total actions:** {guide.total_actions}")
    lines.append(f"- **Required (breaking):** {guide.required_count}")
    lines.append(f"- **Optional (best practice):** {guide.optional_count}")
    if guide.cocos_change:
        lines.append(f"- **COCOS change:** {guide.cocos_change}")
    if guide.ids_affected:
        lines.append(f"- **IDS affected:** {', '.join(guide.ids_affected)}")
    lines.append("")

    # -- Required updates --
    if guide.required_actions:
        lines.append("## Required Updates")
        lines.append("")
        lines.append(
            "These changes **must** be applied or code will produce incorrect results."
        )
        lines.append("")

        # Group by IDS
        by_ids: dict[str, list[CodeUpdateAction]] = {}
        for action in guide.required_actions:
            by_ids.setdefault(action.ids, []).append(action)

        for ids_name in sorted(by_ids):
            lines.append(f"### {ids_name}")
            lines.append("")
            for action in by_ids[ids_name]:
                lines.append(f"**{action.change_type}:** `{action.path}`")
                lines.append(f"  - {action.description}")
                if action.search_patterns:
                    patterns_str = ", ".join(
                        f"`{p}`" for p in action.search_patterns[:3]
                    )
                    lines.append(f"  - Search for: {patterns_str}")
                lines.append("")

    # -- COCOS section --
    if guide.cocos_advice:
        ca = guide.cocos_advice
        lines.append(f"## COCOS Migration ({ca.from_cocos} \u2192 {ca.to_cocos})")
        lines.append("")
        if ca.sign_flips:
            lines.append("### Sign Flips Required")
            lines.append("")
            lines.append("| Path | Label | Factor |")
            lines.append("|------|-------|--------|")
            for sf in ca.sign_flips:
                lines.append(
                    f"| `{sf['path']}` | {sf.get('label', '')} | {sf['factor']} |"
                )
            lines.append("")
        if ca.no_change:
            lines.append(f"### No Change Required ({len(ca.no_change)} paths)")
            lines.append("")
            lines.append(
                "These COCOS-dependent paths have factor=1 (no sign change needed)."
            )
            lines.append("")

    # -- Convention changes --
    convention_actions = [
        a
        for a in guide.required_actions + guide.optional_actions
        if a.change_type == "definition_change"
    ]
    if convention_actions:
        lines.append("## Convention Changes")
        lines.append("")
        lines.append(
            "These changes affect data interpretation without changing path names or types."
        )
        lines.append(
            "Code that reads these fields may produce **silently incorrect results**"
        )
        lines.append("if not updated.")
        lines.append("")
        for action in convention_actions:
            severity_badge = (
                "**BREAKING**" if action.severity == "required" else "advisory"
            )
            lines.append(f"### `{action.path}` ({severity_badge})")
            lines.append("")
            lines.append(f"  {action.description}")
            if action.before:
                lines.append(f"  - **Before:** {action.before}")
            if action.after:
                lines.append(f"  - **After:** {action.after}")
            if action.search_patterns:
                patterns_str = ", ".join(f"`{p}`" for p in action.search_patterns[:3])
                lines.append(f"  - **Search for:** {patterns_str}")
            lines.append("")

    # -- Search strategy --
    if guide.global_search_patterns:
        lines.append("## Search Strategy")
        lines.append("")
        lines.append("Use these patterns to find affected code in your codebase:")
        lines.append("")
        for ids_name in sorted(guide.global_search_patterns):
            patterns = guide.global_search_patterns[ids_name]
            if patterns:
                lines.append(f"### {ids_name}")
                lines.append("```")
                for p in patterns[:10]:
                    lines.append(f"grep -r '{p}' /path/to/code/")
                lines.append("```")
                lines.append("")

    # -- Optional updates --
    if guide.optional_actions:
        lines.append("## Optional Updates")
        lines.append("")
        lines.append("Best-practice changes for full DD compliance:")
        lines.append("")
        for action in guide.optional_actions[:20]:
            lines.append(f"- `{action.path}`: {action.description}")
        if len(guide.optional_actions) > 20:
            lines.append(f"- ... and {len(guide.optional_actions) - 20} more")
        lines.append("")

    # -- Verification checklist --
    lines.append("## Verification Checklist")
    lines.append("")
    lines.append("After applying changes:")
    lines.append("- [ ] Run your test suite to verify no regressions")
    if guide.cocos_advice:
        lines.append(
            f"- [ ] Verify sign conventions match COCOS {guide.cocos_advice.to_cocos}"
        )
    lines.append("- [ ] Update DD version reference in your code configuration")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_migration_guide(
    gc: GraphClient,
    from_version: str,
    to_version: str,
    ids_filter: str | None = None,
    include_recipes: bool = True,
    summary_only: bool = False,
) -> str:
    """Generate a DD migration guide between two versions.

    Args:
        gc: Graph client
        from_version: Source DD version (e.g. "3.39.0")
        to_version: Target DD version (e.g. "4.0.0")
        ids_filter: Optional IDS name to restrict output
        include_recipes: Whether to include code update snippets
        summary_only: If True, return only aggregate statistics without per-path details

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

    if summary_only:
        change_summary = _get_change_summary(gc, version_range, ids_filter)
        from_cocos = _get_version_cocos(gc, from_version)
        to_cocos = _get_version_cocos(gc, to_version)

        # Aggregate change statistics
        by_type: dict[str, dict] = {}
        total_changes = 0
        breaking_total = 0
        for row in change_summary:
            ctype = row["type"]
            level = row["level"]
            cnt = row["cnt"]
            total_changes += cnt
            if ctype not in by_type:
                by_type[ctype] = {"count": 0, "breaking": 0}
            by_type[ctype]["count"] += cnt
            if level in ("required", "breaking"):
                by_type[ctype]["breaking"] += cnt
                breaking_total += cnt

        # Get affected IDS list
        ids_query = gc.query(
            """
            MATCH (c:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion)
            WHERE v.id IN $versions
            MATCH (c)-[:FOR_IMAS_PATH]->(p:IMASNode)
            RETURN DISTINCT p.ids AS ids
            ORDER BY ids
            """,
            versions=version_range,
        )
        ids_affected = [r["ids"] for r in ids_query]

        cocos_change = None
        if from_cocos and to_cocos and from_cocos != to_cocos:
            cocos_change = f"{from_cocos} → {to_cocos}"

        # Format as readable summary
        lines = [
            f"# Migration Summary: {from_version} → {to_version}",
            "",
            f"**COCOS change:** {cocos_change or 'None'}",
            f"**Total changes:** {total_changes:,}",
            f"**Breaking changes:** {breaking_total:,}",
            f"**IDS affected:** {len(ids_affected)}",
            "",
            "## Changes by Type",
            "",
            "| Change Type | Count | Breaking |",
            "|------------|------:|--------:|",
        ]
        for ctype in sorted(by_type, key=lambda t: by_type[t]["count"], reverse=True):
            info = by_type[ctype]
            lines.append(f"| {ctype} | {info['count']:,} | {info['breaking']:,} |")

        if ids_affected:
            lines.extend(["", "## IDS Affected", ""])
            lines.append(", ".join(ids_affected))

        if breaking_total > 1000:
            lines.extend(
                [
                    "",
                    f"**Recommendation:** Major migration — {breaking_total:,} breaking changes. "
                    f"Use full guide with `ids_filter` for per-IDS migration.",
                ]
            )

        return "\n".join(lines)

    guide = build_migration_guide(
        gc, from_version, to_version, ids_filter, include_recipes
    )
    return format_migration_guide(guide)
