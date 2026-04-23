"""Harvest VocabGap nodes from the graph and format PR-ready YAML.

The harvester reads :class:`~imas_codex.graph.models.VocabGap` nodes,
groups by grammar segment, and emits a structured YAML document suitable
for filing as an ISN grammar PR.

Typical use via the CLI::

    imas-codex sn gaps --format yaml
    imas-codex sn gaps --format yaml --output vocab-gaps.yaml

Or programmatically::

    from imas_codex.standard_names.gap_harvest import harvest_vocab_gaps, format_pr_yaml
    records = harvest_vocab_gaps(gc)
    print(format_pr_yaml(records))
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

# Segments that have a fixed, closed vocabulary — a missing token is a real gap.
# Populated at runtime from the ISN package; falls back to known closed segments.
_KNOWN_CLOSED_SEGMENTS: frozenset[str] = frozenset(
    {
        "coordinate",
        "process",
        "position",
        "component",
        "subject",
        "region",
        "geometry",
        "geometric_base",
        "device",
        "object",
    }
)

# Maximum example_dd_paths / example_reasons to include per token entry.
_MAX_EXAMPLES = 5
_MAX_REASONS = 5


def _closed_segments() -> frozenset[str]:
    """Return the set of closed-vocabulary grammar segments.

    Tries the installed ISN package first; falls back to our compile-time set.
    """
    try:
        from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

        return frozenset(seg for seg, tokens in SEGMENT_TOKEN_MAP.items() if tokens)
    except Exception:  # ImportError or any parsing error
        return _KNOWN_CLOSED_SEGMENTS


def _isn_version() -> str | None:
    """Return the active ISN grammar version from the graph, if available."""
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            rows = list(
                gc.query(
                    "MATCH (gv:ISNGrammarVersion {active: true}) RETURN gv.version LIMIT 1"
                )
            )
            if rows:
                return rows[0]["gv.version"]
    except Exception:
        pass
    return None


def _dd_version() -> str | None:
    """Return the current DD version from the graph, if available."""
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            rows = list(
                gc.query("MATCH (d:DDVersion {current: true}) RETURN d.version LIMIT 1")
            )
            if rows:
                return rows[0]["d.version"]
    except Exception:
        pass
    return None


def harvest_vocab_gaps(
    gc: GraphClient,
    *,
    segment_filter: str | None = None,
    include_open: bool = False,
) -> list[dict[str, Any]]:
    """Query VocabGap nodes and collect per-source evidence.

    Returns a flat list of enriched gap records.  Each record has::

        {
            "segment": str,
            "needed_token": str,
            "occurrences": int,          # source_count from graph
            "example_count": int | None, # stored on node (may differ)
            "source_types": list[str],   # "IMASNode", "FacilitySignal", …
            "example_dd_paths": list[str],   # up to _MAX_EXAMPLES IMASNode ids
            "example_reasons": list[str],    # up to _MAX_REASONS distinct reasons
            "first_seen": str | None,
            "last_seen": str | None,
        }

    The list is sorted by segment then occurrences (descending).

    Args:
        gc: Open :class:`~imas_codex.graph.client.GraphClient` instance.
        segment_filter: When set, restrict results to this segment.
        include_open: When ``False`` (default) open/pseudo segments are excluded.
    """
    from imas_codex.standard_names.segments import PSEUDO_SEGMENTS, open_segments

    params: dict[str, Any] = {}
    where_clauses: list[str] = []

    if segment_filter:
        where_clauses.append("vg.segment = $segment")
        params["segment"] = segment_filter

    if not include_open:
        excluded = sorted(open_segments() | PSEUDO_SEGMENTS)
        if excluded:
            where_clauses.append("NOT (vg.segment IN $excluded_segments)")
            params["excluded_segments"] = excluded

    where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # Collect source IDs and reasons from HAS_STANDARD_NAME_VOCAB_GAP rels
    rows = list(
        gc.query(
            f"""
            MATCH (vg:VocabGap)
            {where_clause}
            OPTIONAL MATCH (src)-[rel:HAS_STANDARD_NAME_VOCAB_GAP]->(vg)
            WITH vg,
                 count(src) AS source_count,
                 collect(DISTINCT labels(src)[0]) AS source_types,
                 collect(DISTINCT CASE
                     WHEN 'IMASNode' IN labels(src) THEN src.id ELSE null
                 END)[..{_MAX_EXAMPLES}] AS example_dd_paths,
                 collect(DISTINCT CASE
                     WHEN rel.reason IS NOT NULL AND rel.reason <> '' THEN rel.reason ELSE null
                 END)[..{_MAX_REASONS}] AS example_reasons
            RETURN vg.segment AS segment,
                   vg.needed_token AS needed_token,
                   vg.example_count AS example_count,
                   source_count,
                   source_types,
                   example_dd_paths,
                   example_reasons,
                   vg.first_seen_at AS first_seen,
                   vg.last_seen_at AS last_seen
            ORDER BY vg.segment, source_count DESC
            """,
            **params,
        )
    )

    records: list[dict[str, Any]] = []
    for r in rows:
        # Remove None values from list fields
        dd_paths = [p for p in (r.get("example_dd_paths") or []) if p]
        reasons = [p for p in (r.get("example_reasons") or []) if p]
        records.append(
            {
                "segment": r["segment"],
                "needed_token": r["needed_token"],
                "occurrences": r["source_count"] or 0,
                "example_count": r.get("example_count"),
                "source_types": sorted({t for t in (r.get("source_types") or []) if t}),
                "example_dd_paths": dd_paths,
                "example_reasons": reasons,
                "first_seen": (
                    str(r["first_seen"])[:19] if r.get("first_seen") else None
                ),
                "last_seen": (str(r["last_seen"])[:19] if r.get("last_seen") else None),
            }
        )
    return records


def format_pr_yaml(
    records: list[dict[str, Any]],
    *,
    isn_version: str | None = None,
    dd_version: str | None = None,
) -> str:
    """Format gap records into a PR-ready YAML document.

    The document structure::

        metadata:
          generated_at: "2025-07-01T12:00:00Z"
          isn_version: "0.7.0rc23"
          dd_version: "3.41.0"
          total_gaps: 42
          distinct_tokens: 17

        gaps_by_segment:
          transformation:
            segment_type: closed
            tokens:
              - needed_token: derivative_of
                occurrences: 5
                example_dd_paths: [...]
                example_reasons: [...]
          physical_base:
            segment_type: open
            tokens: [...]

    Args:
        records: Flat list from :func:`harvest_vocab_gaps`.
        isn_version: Override ISN grammar version string.
        dd_version: Override DD version string.
    """
    import yaml

    closed = _closed_segments()

    # Group by segment
    by_segment: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        seg = r["segment"]
        by_segment.setdefault(seg, []).append(r)

    # Build segment entries
    gaps_by_segment: dict[str, Any] = {}
    for seg in sorted(by_segment):
        seg_records = by_segment[seg]
        seg_type = "closed" if seg in closed else "open"
        tokens: list[dict[str, Any]] = []
        for r in sorted(seg_records, key=lambda x: x["occurrences"], reverse=True):
            entry: dict[str, Any] = {
                "needed_token": r["needed_token"],
                "occurrences": r["occurrences"],
            }
            if r.get("example_dd_paths"):
                entry["example_dd_paths"] = r["example_dd_paths"]
            if r.get("example_reasons"):
                entry["example_reasons"] = r["example_reasons"]
            if r.get("first_seen"):
                entry["first_seen"] = r["first_seen"]
            tokens.append(entry)
        gaps_by_segment[seg] = {
            "segment_type": seg_type,
            "tokens": tokens,
        }

    distinct_tokens = len({r["needed_token"] for r in records})

    doc: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "isn_version": isn_version or "unknown",
            "dd_version": dd_version or "unknown",
            "total_gaps": len(records),
            "distinct_tokens": distinct_tokens,
        },
        "gaps_by_segment": gaps_by_segment,
    }

    header = (
        "# ISN Vocabulary Gap Report\n"
        "# Generated by: imas-codex sn gaps --format yaml\n"
        "# File as an ISN grammar PR at: "
        "https://github.com/iterorganization/imas-standard-names\n"
        "#\n"
    )
    return header + yaml.dump(doc, default_flow_style=False, sort_keys=False)


def format_pr_markdown(
    records: list[dict[str, Any]],
    *,
    isn_version: str | None = None,
    dd_version: str | None = None,
    top_n: int = 20,
) -> str:
    """Format gap records as a PR-ready Markdown summary / report.

    Includes:
    - Header metadata
    - Per-segment counts table
    - Top-N tokens table
    - Suggested PR title and body template

    Args:
        records: Flat list from :func:`harvest_vocab_gaps`.
        isn_version: Override ISN grammar version string.
        dd_version: Override DD version string.
        top_n: Number of top tokens to show in the table.
    """
    closed = _closed_segments()
    total = len(records)
    distinct = len({r["needed_token"] for r in records})
    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    isn_v = isn_version or "unknown"
    dd_v = dd_version or "unknown"

    lines: list[str] = []
    lines.append("# ISN Vocabulary Gap Report")
    lines.append("")
    lines.append(f"Generated: `{generated_at}` · ISN: `{isn_v}` · DD: `{dd_v}`")
    lines.append("")
    lines.append(f"**{total} gap records** across **{distinct} distinct tokens**")
    lines.append("")

    # Per-segment table
    lines.append("## Gaps by Grammar Segment")
    lines.append("")
    lines.append("| Segment | Type | Distinct Tokens | Total Occurrences |")
    lines.append("|---------|------|----------------|-------------------|")
    by_segment: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_segment.setdefault(r["segment"], []).append(r)
    for seg in sorted(by_segment):
        seg_records = by_segment[seg]
        seg_type = "closed" if seg in closed else "open"
        dt = len({r["needed_token"] for r in seg_records})
        total_occ = sum(r["occurrences"] for r in seg_records)
        lines.append(f"| `{seg}` | {seg_type} | {dt} | {total_occ} |")
    lines.append("")

    # Top-N tokens
    sorted_records = sorted(records, key=lambda x: x["occurrences"], reverse=True)
    top = sorted_records[:top_n]
    lines.append(f"## Top {min(top_n, len(top))} Tokens by Occurrence")
    lines.append("")
    lines.append("| Rank | Token | Segment | Occurrences |")
    lines.append("|------|-------|---------|-------------|")
    for i, r in enumerate(top, 1):
        lines.append(
            f"| {i} | `{r['needed_token']}` | `{r['segment']}` | {r['occurrences']} |"
        )
    lines.append("")

    # ISN PR template
    lines.append("## ISN Grammar PR Template")
    lines.append("")
    lines.append("### Suggested PR Title")
    lines.append("")
    lines.append(
        f"```\ngrammar: add {distinct} missing vocabulary tokens (ISN {isn_v})\n```"
    )
    lines.append("")
    lines.append("### PR Body Template")
    lines.append("")
    lines.append("```markdown")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"This PR adds {distinct} missing grammar tokens across "
        f"{len(by_segment)} segment(s), identified by imas-codex during "
        "standard-name composition over IMAS DD paths and facility signals."
    )
    lines.append("")
    lines.append("## Motivation")
    lines.append("")
    lines.append(
        "During automated composition, the LLM requested grammar tokens that "
        "do not exist in the current ISN vocabulary.  Without these tokens, "
        "composition falls back to `vocab_gap` status, blocking standard-name "
        "generation for the affected paths."
    )
    lines.append("")
    lines.append("## Changes")
    lines.append("")
    for seg in sorted(by_segment):
        tokens = sorted({r["needed_token"] for r in by_segment[seg]})
        seg_type = "closed" if seg in closed else "open"
        lines.append(
            f"- **`{seg}`** ({seg_type}): {', '.join(f'`{t}`' for t in tokens)}"
        )
    lines.append("")
    lines.append("## Acceptance Criteria")
    lines.append("")
    lines.append("- [ ] All listed tokens are added to the correct segment vocabulary")
    lines.append("- [ ] Grammar tests pass (`pytest tests/`)")
    lines.append("- [ ] `imas-codex sn validate-isn` reports no new warnings")
    lines.append("- [ ] `imas-codex sn gaps` shows reduced total after sync")
    lines.append("```")
    lines.append("")

    if not records:
        lines.append(
            "> No VocabGap nodes found in the graph. "
            "Run `imas-codex sn run` to populate gaps, then re-harvest."
        )

    return "\n".join(lines) + "\n"


__all__ = [
    "format_pr_markdown",
    "format_pr_yaml",
    "harvest_vocab_gaps",
]
