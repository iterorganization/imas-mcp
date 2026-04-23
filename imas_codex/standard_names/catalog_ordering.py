"""Deterministic topological ordering for per-domain catalog entries.

Implements Kahn's topological sort over the ordering-parent relation
derived from ``HAS_ARGUMENT`` and ``HAS_ERROR`` graph edges, with
alphabetic tie-break and clean-root / orphan queue separation.

Pure function of ``(entry-ids, in-domain-edges, cross-domain-edge
presence)`` — does not touch the graph.

See plan 40 §2 for algorithm specification.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class OrderingError(ValueError):
    """Raised when topological sort cannot emit all entries (cycle detected)."""


def order_entries_by_hierarchy(
    entries: list[dict],
    edges: list[tuple[str, str, str]],
    *,
    cross_domain_parent_ids: set[str] | None = None,
) -> list[dict]:
    """Order entries by deterministic topological traversal.

    Parameters
    ----------
    entries:
        List of entry dicts, each with at least an ``"name"`` key
        (or ``"id"``).
    edges:
        List of ``(src_id, tgt_id, edge_type)`` tuples where
        *edge_type* ∈ ``{"HAS_ARGUMENT", "HAS_ERROR"}``.
        All edges are **in-domain** (both endpoints present in
        *entries*).
    cross_domain_parent_ids:
        Set of entry IDs (names) in *entries* whose full-graph
        ordering-parent lives **outside** this domain.  These are
        placed in the orphan queue instead of the clean-roots queue
        when their in-domain in-degree is zero.

    Returns
    -------
    Entries re-ordered so that every entry appears after all its
    in-domain ordering-parents, with alphabetic tie-break.

    Raises
    ------
    OrderingError
        If queues drain with unemitted entries (cycle in the DAG).
    """
    if cross_domain_parent_ids is None:
        cross_domain_parent_ids = set()

    # Build entry lookup by name
    entry_by_name: dict[str, dict] = {}
    for e in entries:
        name = e.get("name") or e.get("id", "")
        entry_by_name[name] = e

    all_names = set(entry_by_name.keys())

    # ── Build ordering-parent → child adjacency ────────────────────
    # Unified ordering-parent relation:
    #   HAS_ARGUMENT: src -[:HAS_ARGUMENT]-> tgt  ⇒  tgt is parent of src
    #   HAS_ERROR:    src -[:HAS_ERROR]-> tgt     ⇒  src is parent of tgt
    children: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = dict.fromkeys(all_names, 0)

    for src, tgt, edge_type in edges:
        if src not in all_names or tgt not in all_names:
            continue  # skip edges with endpoints outside this domain
        if edge_type == "HAS_ARGUMENT":
            # tgt is ordering-parent of src
            children[tgt].append(src)
            in_degree[src] += 1
        elif edge_type == "HAS_ERROR":
            # src is ordering-parent of tgt
            children[src].append(tgt)
            in_degree[tgt] += 1

    # ── Seed queues ────────────────────────────────────────────────
    clean_roots: list[str] = []
    orphan_queue: list[str] = []

    for name in sorted(all_names):
        if in_degree[name] == 0:
            if name in cross_domain_parent_ids:
                orphan_queue.append(name)
            else:
                clean_roots.append(name)

    # Queues are maintained sorted (alphabetic tie-break)
    clean_roots.sort()
    orphan_queue.sort()

    # ── Kahn's drain ───────────────────────────────────────────────
    result: list[dict] = []
    emitted: set[str] = set()

    while clean_roots or orphan_queue:
        # Pop from clean-roots first, else orphan
        if clean_roots:
            current = clean_roots.pop(0)
        else:
            current = orphan_queue.pop(0)

        if current in emitted:
            continue
        emitted.add(current)
        result.append(entry_by_name[current])

        # Decrement children; newly-ready children go to clean-roots
        for child in sorted(children.get(current, [])):
            in_degree[child] -= 1
            if in_degree[child] == 0 and child not in emitted:
                # Children always inherit clean-root queue status
                _insort(clean_roots, child)

    # ── Cycle detection ────────────────────────────────────────────
    if len(emitted) != len(all_names):
        stuck = sorted(all_names - emitted)
        raise OrderingError(
            f"Topological sort stuck with {len(stuck)} unemitted "
            f"entries (cycle?): {stuck}"
        )

    return result


def _insort(sorted_list: list[str], value: str) -> None:
    """Insert *value* into *sorted_list* maintaining sorted order."""
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    sorted_list.insert(lo, value)
