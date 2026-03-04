"""Result formatters for graph query output.

Provides ``as_table()``, ``as_summary()``, and ``pick()`` — lightweight
helpers that turn lists of dicts (as returned by domain queries and
``graph_search()``) into token-efficient text for agent consumption.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def pick(results: list[dict], *fields: str) -> list[dict]:
    """Project results to only the specified fields.

    Missing fields are returned as ``None``.

    Args:
        results: List of dicts to project.
        *fields: Field names to keep.

    Returns:
        List of dicts containing only the requested fields.
    """
    return [{f: row.get(f) for f in fields} for row in results]


def as_table(
    results: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> str:
    """Format results as a compact markdown table.

    Args:
        results: List of dicts to format.
        columns: Column names to include. Defaults to all keys from first row.

    Returns:
        Markdown table string, or ``""`` if results is empty.
    """
    if not results:
        return ""

    cols = columns or list(results[0].keys())

    # Header
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"

    # Rows
    rows = []
    for row in results:
        cells = [_cell(row.get(c, "")) for c in cols]
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator, *rows])


def as_summary(
    results: list[dict[str, Any]],
    group_by: str | None = None,
) -> str:
    """Summarise results with counts, optionally grouped.

    Args:
        results: List of dicts to summarise.
        group_by: Optional key to group and count by.

    Returns:
        Human-readable summary string.
    """
    total = len(results)
    parts = [f"{total} results"]

    if group_by and results:
        counts: Counter[str] = Counter()
        for row in results:
            val = row.get(group_by, "<none>")
            counts[str(val)] += 1
        group_lines = [f"  {k}: {v}" for k, v in counts.most_common()]
        parts.append(f"by {group_by}:")
        parts.extend(group_lines)

    return "\n".join(parts)


def _cell(value: Any) -> str:
    """Convert a value to a table cell string, truncating long values."""
    if value is None:
        return ""
    s = str(value)
    if len(s) > 80:
        return s[:77] + "..."
    return s
