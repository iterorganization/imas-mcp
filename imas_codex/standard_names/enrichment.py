"""DD enrichment layer for standard name generation.

Navigates DD graph query results to build rich context for standard name
generation.  Implements primary cluster selection (resolving many-to-many
cluster memberships to exactly one per path) and groups paths **globally**
by (cluster × unit) for unit-safe batching.

Data flow::

    sources/dd.py (graph query, multi-cluster rows)
        → enrich_paths()              (classify, deduplicate, select primary cluster)
        → group_by_concept_and_unit() (global grouping, batch splitting)
        → list[ExtractionBatch]       (ready for compose worker)
"""

from __future__ import annotations

import logging
from collections import defaultdict

from imas_codex.standard_names.classifier import classify_path
from imas_codex.standard_names.sources.base import ExtractionBatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain reclassification for magnetics IDS paths.
#
# Paths from the ``magnetics`` IDS are assigned ``physics_domain`` by the DD
# enrichment layer, which often routes them into ``equilibrium`` (via the
# constraint subtree) or ``general``.  For standard-name generation these
# paths should live in ``magnetic_field_diagnostics``.
# ---------------------------------------------------------------------------

#: Magnetics IDS subtrees that should be reclassified to
#: ``magnetic_field_diagnostics``.
MAGNETICS_DIAGNOSTICS_SUBTREES: tuple[str, ...] = (
    "magnetics/bpol_probe",
    "magnetics/flux_loop",
    "magnetics/rogowski_coil",
    "magnetics/diamagnetic_flux",
    "magnetics/b_field_",
    "magnetics/ip",
    "magnetics/method",
    "magnetics/time",
    "magnetics/shunt",
)


def reclassify_magnetics_domain(row: dict) -> None:
    """Override ``physics_domain`` for magnetics-IDS paths.

    Only modifies rows whose ``ids_name`` is ``magnetics``.  Any path under
    the ``magnetics`` IDS that currently has a domain other than
    ``magnetic_field_diagnostics`` is reclassified.

    Args:
        row: Enriched path dict (mutated in place).
    """
    ids_name = row.get("ids_name") or ""
    if ids_name != "magnetics":
        return

    current_domain = row.get("physics_domain") or ""
    if current_domain == "magnetic_field_diagnostics":
        return  # already correct

    row["physics_domain"] = "magnetic_field_diagnostics"


# ---------------------------------------------------------------------------
# Scope priority for primary cluster selection (most specific first).
# ---------------------------------------------------------------------------

_SCOPE_PRIORITY: dict[str, int] = {
    "ids": 0,
    "domain": 1,
    "global": 2,
}
_DEFAULT_SCOPE_RANK = 3  # missing or unrecognised scope


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_primary_cluster(clusters: list[dict]) -> dict | None:
    """Choose ONE primary cluster per path from its many-to-many memberships.

    Resolution order (most specific first — best for per-item context):

    1. IDS-scope cluster (most specific) — check ``scope`` field
    2. Domain-scope cluster
    3. Global-scope cluster

    Within same scope: highest ``similarity_score`` (if available), else
    first by ``cluster_label`` (deterministic tie-break).

    Args:
        clusters: List of dicts with keys ``cluster_id``, ``cluster_label``,
            ``cluster_description``, ``scope`` (ids/domain/global),
            ``similarity_score`` (optional).

    Returns:
        The selected primary cluster dict, or ``None`` if *clusters* is empty.
    """
    if not clusters:
        return None
    if len(clusters) == 1:
        return clusters[0]

    def _sort_key(c: dict) -> tuple[int, float, str]:
        scope_rank = _SCOPE_PRIORITY.get((c.get("scope") or ""), _DEFAULT_SCOPE_RANK)
        # Negate similarity so higher scores sort first.
        sim = -(c.get("similarity_score") or 0.0)
        label = c.get("cluster_label") or ""
        return (scope_rank, sim, label)

    return min(clusters, key=_sort_key)


# ---------------------------------------------------------------------------
# Reversed scope priority for grouping (widest first).
# ---------------------------------------------------------------------------

_GROUPING_SCOPE_PRIORITY: dict[str, int] = {
    "global": 0,
    "domain": 1,
    "ids": 2,
}


def select_grouping_cluster(clusters: list[dict]) -> dict | None:
    """Choose the best cluster for batch grouping.

    Uses REVERSED priority: global → domain → IDS (widest first) to ensure
    equivalent paths across different IDSs land in the same batch.

    Returns:
        The selected grouping cluster dict, or ``None`` if *clusters* is empty.
    """
    if not clusters:
        return None
    if len(clusters) == 1:
        return clusters[0]

    def _sort_key(c: dict) -> tuple[int, float, str]:
        scope_rank = _GROUPING_SCOPE_PRIORITY.get(
            (c.get("scope") or ""), _DEFAULT_SCOPE_RANK
        )
        sim = -(c.get("similarity_score") or 0.0)
        label = c.get("cluster_label") or ""
        return (scope_rank, sim, label)

    return min(clusters, key=_sort_key)


def enrich_paths(paths: list[dict]) -> list[dict]:
    """Enrich DD paths with classification and primary cluster selection.

    The enriched graph query in ``sources/dd.py`` may return **multiple rows
    per path** (one per cluster membership).  This function:

    1. Deduplicates rows to one entry per unique path.
    2. Collects all cluster memberships for each path.
    3. Classifies each path via :func:`classify_path` (quantity / skip).
    4. Selects primary cluster from multi-cluster memberships.
    5. Attaches enrichment metadata.

    Returns:
        Only ``"quantity"`` paths, each with ``primary_cluster_id``,
        ``primary_cluster_label``, ``primary_cluster_description``, and
        ``all_clusters`` attached.
    """
    if not paths:
        return []

    # --- Step 1+2: deduplicate rows, collect clusters -----------------------
    path_base: dict[str, dict] = {}  # path → first row (node attributes)
    path_clusters: dict[str, list[dict]] = defaultdict(list)

    for row in paths:
        p = row.get("path", "")
        if not p:
            continue

        # First row seen becomes the canonical base row.
        if p not in path_base:
            path_base[p] = dict(row)

        # Collect cluster membership if present.
        cid = row.get("cluster_id")
        if cid:
            # Avoid adding the same cluster twice (defensive).
            existing_ids = {c["cluster_id"] for c in path_clusters[p]}
            if cid not in existing_ids:
                path_clusters[p].append(
                    {
                        "cluster_id": cid,
                        "cluster_label": row.get("cluster_label") or "",
                        "cluster_description": row.get("cluster_description") or "",
                        "scope": row.get("cluster_scope") or "",
                        "similarity_score": row.get("similarity_score"),
                    }
                )

    # --- Step 3+4+5: classify, select primary cluster, attach enrichment ----
    enriched: list[dict] = []
    skip_count = 0

    for path, base_row in path_base.items():
        scope = classify_path(base_row)

        if scope == "skip":
            skip_count += 1
            continue

        # Reclassify magnetics-IDS paths to magnetic_field_diagnostics
        reclassify_magnetics_domain(base_row)

        # Select primary cluster (IDS-local, for per-item context)
        clusters = path_clusters.get(path, [])
        primary = select_primary_cluster(clusters)

        base_row["primary_cluster_id"] = primary["cluster_id"] if primary else None
        base_row["primary_cluster_label"] = (
            primary["cluster_label"] if primary else None
        )
        base_row["primary_cluster_description"] = (
            primary["cluster_description"] if primary else None
        )
        base_row["all_clusters"] = clusters

        # Select grouping cluster (global/domain preferred, for batch formation)
        grouping = select_grouping_cluster(clusters)
        base_row["grouping_cluster_id"] = grouping["cluster_id"] if grouping else None
        base_row["grouping_cluster_label"] = (
            grouping["cluster_label"] if grouping else None
        )

        enriched.append(base_row)

    logger.info(
        "Enriched %d quantity paths (skipped %d) from %d raw rows",
        len(enriched),
        skip_count,
        len(paths),
    )
    return enriched


def build_batch_context(
    items: list[dict], group_key: str, cocos_version: int | None = None
) -> str:
    """Build rich context summary for a batch.

    Includes cluster label, authoritative unit, path count, cross-IDS
    summary, concept description, and cluster sibling preview.
    """
    parts: list[str] = []

    # Derive cluster label from items (preferred) or group_key (fallback)
    if group_key.startswith("unclustered/"):
        parts.append("Unclustered paths")
        inner = group_key[len("unclustered/") :]
        last_slash = inner.rfind("/")
        if last_slash > 0:
            parent = inner[:last_slash]
            parts.append(f"Parent structure: {parent}")
    else:
        cluster_label = items[0].get("grouping_cluster_label") or items[0].get(
            "primary_cluster_label"
        )
        if cluster_label:
            parts.append(f"Cluster: {cluster_label}")
        else:
            parts.append(f"Group: {group_key}")

    # Authoritative unit.
    unit = items[0].get("unit") or "dimensionless"
    parts.append(f"Authoritative unit: {unit}")

    # Path count.
    parts.append(f"{len(items)} paths sharing this concept")

    # Cross-IDS summary.
    ids_names = sorted({item.get("ids_name", "unknown") for item in items})
    if len(ids_names) > 1:
        parts.append(f"Cross-IDS: {', '.join(ids_names)}")
    elif ids_names:
        parts.append(f"IDS: {ids_names[0]}")

    # Cluster description.
    desc = items[0].get("primary_cluster_description")
    if desc:
        parts.append(f"Concept: {desc}")

    # Sibling preview.
    siblings = items[0].get("cluster_siblings", [])
    if siblings:
        sib_strs = [f"  {s['path']} ({s.get('unit', '?')})" for s in siblings[:5]]
        parts.append("Cross-IDS siblings:\n" + "\n".join(sib_strs))

    # COCOS context
    cocos_labels = {
        item.get("cocos_label") for item in items if item.get("cocos_label")
    }
    if cocos_labels:
        labels_str = ", ".join(sorted(cocos_labels))
        parts.append(f"COCOS transformation types: {labels_str}")
        if cocos_version:
            parts.append(f"COCOS convention: {cocos_version}")

    return "\n".join(parts)


def group_by_concept_and_unit(
    items: list[dict],
    max_batch_size: int = 25,
    existing_names: set[str] | None = None,
    max_tokens: int | None = None,
) -> list[ExtractionBatch]:
    """Group enriched paths by (primary_cluster × unit) **globally**.

    Critical design decisions:

    * **Global grouping** — same concept across IDSs → same batch → same
      name.
    * **Primary cluster** — each path appears in exactly ONE batch (no
      Cartesian product from multi-cluster membership).
    * **Mixed-unit clusters** split into separate batches.
    * **Oversized groups** split into chunks of *max_batch_size*.
    * **Unclustered paths** sub-grouped by ``parent_path``.

    Args:
        items: Enriched path dicts (output of :func:`enrich_paths`).
        max_batch_size: Maximum concepts per batch (token budget guard).
        existing_names: Known standard names for dedup awareness.
        max_tokens: When set, apply a pre-flight token check that
            binary-splits any batch exceeding this estimated token count.

    Returns:
        List of :class:`ExtractionBatch` objects ready for the compose
        worker.
    """
    if existing_names is None:
        existing_names = set()

    if not items:
        return []

    # --- Build groups: (grouping_cluster_id / unit) -------------------------
    # Uses grouping cluster (global/domain preferred) and cluster ID (not label)
    # to ensure cross-IDS paths sharing the same concept land in one batch.
    groups: dict[str, list[dict]] = defaultdict(list)

    for item in items:
        cluster_id = item.get("grouping_cluster_id")
        unit = item.get("unit") or "dimensionless"

        if cluster_id:
            group_key = f"{cluster_id}/{unit}"
        else:
            # Unclustered: sub-group by IDS + parent for coherent batches.
            ids_name = item.get("ids_name") or "unknown"
            parent = item.get("parent_path") or "root"
            group_key = f"unclustered/{ids_name}/{parent}/{unit}"

        groups[group_key].append(item)

    # --- Split oversized groups and build batches ---------------------------
    batches: list[ExtractionBatch] = []

    for group_key in sorted(groups):
        group_items = groups[group_key]

        # Chunk into max_batch_size slices.
        chunks = [
            group_items[i : i + max_batch_size]
            for i in range(0, len(group_items), max_batch_size)
        ]

        for chunk_idx, chunk in enumerate(chunks):
            batch_key = group_key if len(chunks) == 1 else f"{group_key}#{chunk_idx}"
            context = build_batch_context(chunk, group_key)

            batches.append(
                ExtractionBatch(
                    source="dd",
                    group_key=batch_key,
                    items=chunk,
                    context=context,
                    existing_names=existing_names,
                )
            )

    logger.info(
        "Grouped %d paths into %d batches (max_batch_size=%d)",
        len(items),
        len(batches),
        max_batch_size,
    )

    # Pre-flight token check: split batches that exceed token budget
    if max_tokens is not None:
        from imas_codex.standard_names.batching import pre_flight_token_check

        batches = pre_flight_token_check(batches, max_tokens=max_tokens)

    return batches


# ---------------------------------------------------------------------------
# Name-only grouping (Workstream 2a) — coarser batches, broader concept bins.
# ---------------------------------------------------------------------------


def build_name_only_context(items: list[dict], group_key: str) -> str:
    """Build compact context for a ``name_only`` batch.

    Name-only batches use ``(physics_domain × unit)`` as the grouping
    key and may contain items from many clusters and IDSs.  The prompt
    asks the LLM to identify natural sub-groups within the batch, so
    the context summarises the domain-level shape rather than a single
    cluster.
    """
    parts: list[str] = []

    domain = items[0].get("physics_domain") or "unspecified"
    unit = items[0].get("unit") or "dimensionless"

    parts.append(f"Physics domain: {domain}")
    parts.append(f"Authoritative unit: {unit}")
    parts.append(f"{len(items)} paths sharing (domain, unit)")

    ids_names = sorted({item.get("ids_name") or "unknown" for item in items})
    if len(ids_names) > 1:
        parts.append(f"Cross-IDS: {', '.join(ids_names)}")
    elif ids_names:
        parts.append(f"IDS: {ids_names[0]}")

    # Surface the cluster diversity so the LLM knows to look for
    # natural sub-groups when composing names.
    cluster_labels = {
        (item.get("grouping_cluster_label") or item.get("primary_cluster_label"))
        for item in items
        if item.get("grouping_cluster_label") or item.get("primary_cluster_label")
    }
    cluster_labels.discard(None)
    if cluster_labels:
        labels_preview = sorted(cluster_labels)[:8]
        suffix = (
            f" (+{len(cluster_labels) - len(labels_preview)} more)"
            if len(cluster_labels) > len(labels_preview)
            else ""
        )
        parts.append("Represented clusters: " + "; ".join(labels_preview) + suffix)

    return "\n".join(parts)


def group_for_name_only(
    items: list[dict],
    batch_size: int = 50,
    existing_names: set[str] | None = None,
    max_tokens: int | None = None,
) -> list[ExtractionBatch]:
    """Group enriched paths by ``(physics_domain × unit)`` for name-only mode.

    This is the Workstream 2a batching strategy: coarser grouping than
    :func:`group_by_concept_and_unit` to amortise the system-prompt
    cost across many more items per LLM call.  Empirically this
    collapses the ~35 %% singleton tail of (cluster × unit) batching
    into dense bins (mean ≈ 13 items at ``batch_size=50``) while
    preserving unit safety.

    Args:
        items: Enriched path dicts (output of :func:`enrich_paths`).
        batch_size: Maximum items per batch.  Items beyond this cap
            are split into chunks that keep the same group key but
            append a ``#<idx>`` suffix for traceability.
        existing_names: Known standard names for dedup awareness.
        max_tokens: When set, apply a pre-flight token check that
            binary-splits any batch exceeding this estimated token count.

    Returns:
        List of :class:`ExtractionBatch` objects with ``mode="names"``.
    """
    if existing_names is None:
        existing_names = set()

    if not items:
        return []

    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        domain = item.get("physics_domain") or "unspecified"
        unit = item.get("unit") or "dimensionless"
        # `ids_name` is NOT part of the key — we intentionally merge
        # across IDSs to encourage cross-IDS name reuse.
        group_key = f"name_only/{domain}/{unit}"
        groups[group_key].append(item)

    batches: list[ExtractionBatch] = []
    for group_key in sorted(groups):
        group_items = groups[group_key]
        chunks = [
            group_items[i : i + batch_size]
            for i in range(0, len(group_items), batch_size)
        ]
        for chunk_idx, chunk in enumerate(chunks):
            batch_key = group_key if len(chunks) == 1 else f"{group_key}#{chunk_idx}"
            context = build_name_only_context(chunk, group_key)
            batches.append(
                ExtractionBatch(
                    source="dd",
                    group_key=batch_key,
                    items=chunk,
                    context=context,
                    existing_names=existing_names,
                    mode="names",
                )
            )

    logger.info(
        "Name-only grouping: %d paths → %d batches (batch_size=%d, mean=%.1f)",
        len(items),
        len(batches),
        batch_size,
        len(items) / len(batches) if batches else 0.0,
    )

    # Pre-flight token check: split batches that exceed token budget
    if max_tokens is not None:
        from imas_codex.standard_names.batching import pre_flight_token_check

        batches = pre_flight_token_check(batches, max_tokens=max_tokens)

    return batches
