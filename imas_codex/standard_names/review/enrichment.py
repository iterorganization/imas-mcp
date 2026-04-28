"""Cluster reconstruction, batching, and neighborhood context for review.

Data flow::

    graph (StandardName → IMASNode → SemanticCluster)
        → reconstruct_clusters_batch()        (batch cluster lookup)
        → group_into_review_batches()          (cluster × unit grouping)
        → build_neighborhood_context()         (semantic search for context)
        → enriched review batches              (ready for LLM review worker)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from imas_codex.standard_names.domain_priority import domain_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cluster reconstruction (single + batch)
# ---------------------------------------------------------------------------


def reconstruct_dominant_cluster(name_id: str, gc: Any) -> dict | None:
    """Query graph for the dominant cluster of a StandardName.

    Follows the path ``StandardName ← HAS_STANDARD_NAME ← IMASNode
    → MEMBER_OF → SemanticCluster`` and selects the cluster with the
    most source IMASNodes, breaking ties by scope (ids > domain > global)
    then by label.

    Args:
        name_id: StandardName id to look up.
        gc: An open :class:`~imas_codex.graph.client.GraphClient`.

    Returns:
        Cluster dict (cluster_id, cluster_label, cluster_description,
        scope, source_count) or ``None`` if no cluster found.
    """
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $name_id})<-[:HAS_STANDARD_NAME]-(node:IMASNode)
              -[:MEMBER_OF]->(cluster:SemanticCluster)
        WITH cluster, count(DISTINCT node) AS source_count
        OPTIONAL MATCH (cluster)<-[:MEMBER_OF]-(other:IMASNode)
        WITH cluster, source_count, avg(1.0) AS sim
        RETURN cluster.id AS cluster_id,
               cluster.label AS cluster_label,
               cluster.description AS cluster_description,
               cluster.scope AS scope,
               source_count
        ORDER BY source_count DESC,
                 CASE cluster.scope WHEN 'ids' THEN 0 WHEN 'domain' THEN 1 ELSE 2 END,
                 cluster.label
        LIMIT 1
        """,
        name_id=name_id,
    )
    if not rows:
        return None

    row = rows[0]
    return {
        "cluster_id": row["cluster_id"],
        "cluster_label": row["cluster_label"],
        "cluster_description": row["cluster_description"],
        "scope": row["scope"],
        "source_count": row["source_count"],
    }


def reconstruct_clusters_batch(names: list[dict], gc: Any) -> dict[str, dict | None]:
    """Batch cluster reconstruction for multiple StandardNames.

    Single graph query returning all cluster info, grouped by name_id.
    For each name, selects the dominant cluster using priority:
    most source nodes → scope (ids > domain > global) → label tiebreak.

    Args:
        names: List of dicts with at least an ``id`` key.
        gc: An open :class:`~imas_codex.graph.client.GraphClient`.

    Returns:
        ``{name_id: cluster_dict_or_None}`` for every name in *names*.
    """
    name_ids = [n["id"] for n in names if n.get("id")]
    if not name_ids:
        return {}

    rows = gc.query(
        """
        UNWIND $name_ids AS nid
        MATCH (sn:StandardName {id: nid})<-[:HAS_STANDARD_NAME]-(node:IMASNode)
              -[:MEMBER_OF]->(cluster:SemanticCluster)
        WITH nid, cluster, count(DISTINCT node) AS source_count
        RETURN nid AS name_id,
               cluster.id AS cluster_id,
               cluster.label AS cluster_label,
               cluster.description AS cluster_description,
               cluster.scope AS scope,
               source_count
        ORDER BY nid, source_count DESC
        """,
        name_ids=name_ids,
    )

    # Group rows by name_id
    per_name: dict[str, list[dict]] = defaultdict(list)
    for row in rows or []:
        per_name[row["name_id"]].append(
            {
                "cluster_id": row["cluster_id"],
                "cluster_label": row["cluster_label"],
                "cluster_description": row["cluster_description"],
                "scope": row["scope"],
                "source_count": row["source_count"],
            }
        )

    # Select dominant cluster per name using the enrichment selector
    from imas_codex.standard_names.enrichment import select_primary_cluster

    result: dict[str, dict | None] = {}
    for nid in name_ids:
        candidates = per_name.get(nid, [])
        if not candidates:
            result[nid] = None
        elif len(candidates) == 1:
            result[nid] = candidates[0]
        else:
            # select_primary_cluster expects similarity_score key
            for c in candidates:
                c["similarity_score"] = c.get("source_count", 0)
            result[nid] = select_primary_cluster(candidates)

    return result


# ---------------------------------------------------------------------------
# Token estimation + batching
# ---------------------------------------------------------------------------


def estimate_name_tokens(name: dict) -> int:
    """Rough token estimate for a single name in an LLM prompt.

    Uses character-length heuristic (1 token ≈ 4 chars) plus 80 tokens
    of per-item scaffolding (JSON keys, formatting, etc.).
    """
    desc_len = len(name.get("description", "") or "")
    doc_len = len(name.get("documentation", "") or "")
    return (desc_len + doc_len) // 4 + 80


def group_into_review_batches(
    names: list[dict],
    clusters: dict[str, dict | None],
    *,
    max_batch_size: int = 25,
    token_budget: int = 8000,
) -> list[dict]:
    """Group names by (dominant_cluster_id × unit) for review batches.

    Within each group, fills batches respecting *token_budget* and
    *max_batch_size*.

    Args:
        names: StandardName dicts to batch.
        clusters: ``{name_id: cluster_dict_or_None}`` from
            :func:`reconstruct_clusters_batch`.
        max_batch_size: Hard cap on names per batch.
        token_budget: Soft cap on estimated tokens per batch.

    Returns:
        List of batch dicts, each with keys ``group_key``, ``names``,
        ``cluster``, ``estimated_tokens``.
    """
    if not names:
        return []

    # --- Build groups: (cluster_id / unit) ---------------------------------
    groups: dict[str, list[dict]] = defaultdict(list)

    for name in names:
        nid = name.get("id", "")
        cluster = clusters.get(nid)
        unit = name.get("unit") or "dimensionless"

        if cluster:
            group_key = f"{cluster['cluster_id']}/{unit}"
        else:
            domain = domain_key(name.get("physics_domain"))
            group_key = f"unclustered/{domain}/{unit}"

        groups[group_key].append(name)

    # --- Fill batches by token budget / max_batch_size ---------------------
    batches: list[dict] = []

    for group_key in sorted(groups):
        group_items = groups[group_key]
        # Determine the cluster for this group (from any member)
        sample_id = group_items[0].get("id", "")
        group_cluster = clusters.get(sample_id)

        current_batch: list[dict] = []
        current_tokens = 0

        for name in group_items:
            name_tokens = estimate_name_tokens(name)

            # Start new batch when adding this name would exceed limits
            if current_batch and (
                current_tokens + name_tokens > token_budget
                or len(current_batch) >= max_batch_size
            ):
                batches.append(
                    {
                        "group_key": group_key,
                        "names": current_batch,
                        "cluster": group_cluster,
                        "estimated_tokens": current_tokens,
                    }
                )
                current_batch = []
                current_tokens = 0

            current_batch.append(name)
            current_tokens += name_tokens

        # Flush remainder
        if current_batch:
            batches.append(
                {
                    "group_key": group_key,
                    "names": current_batch,
                    "cluster": group_cluster,
                    "estimated_tokens": current_tokens,
                }
            )

    logger.info(
        "Grouped %d names into %d review batches (max_batch_size=%d, token_budget=%d)",
        len(names),
        len(batches),
        max_batch_size,
        token_budget,
    )
    return batches


# ---------------------------------------------------------------------------
# Neighborhood context (semantic search)
# ---------------------------------------------------------------------------


def build_neighborhood_context(
    batch: dict,
    all_names: list[dict],
    k: int = 10,
) -> list[dict]:
    """Build semantic neighborhood context for a review batch.

    Searches for existing StandardNames near the batch's cluster label /
    descriptions to provide the reviewer with cross-catalog awareness.

    Args:
        batch: Batch dict from :func:`group_into_review_batches`.
        all_names: Full StandardName catalog (unused in current impl,
            reserved for future dedup-aware enrichment).
        k: Maximum number of neighbor results.

    Returns:
        List of neighbor dicts with keys ``id``, ``description``, ``kind``,
        ``unit``, ``review_tier``.  Empty list on search failure.
    """
    try:
        from imas_codex.standard_names.search import search_similar_names
    except Exception:
        logger.debug("search_similar_names unavailable", exc_info=True)
        return []

    batch_names = batch.get("names", [])
    batch_ids = {n.get("id", "") for n in batch_names}

    # Build representative query: cluster label + first 3 descriptions
    parts: list[str] = []
    cluster = batch.get("cluster")
    if cluster and cluster.get("cluster_label"):
        parts.append(cluster["cluster_label"])
    for name in batch_names[:3]:
        desc = name.get("description", "")
        if desc:
            parts.append(desc)

    query = " ".join(parts).strip()
    if not query:
        return []

    try:
        raw_results = search_similar_names(query, k=k + 5)
    except Exception:
        logger.debug("Neighborhood search failed", exc_info=True)
        raw_results = []

    # Filter out names that ARE in the current batch
    filtered = [r for r in raw_results if r.get("id", "") not in batch_ids]

    # Per-name fallback for unclustered batches with sparse results
    if len(filtered) < 3:
        unclustered_names = [
            n for n in batch_names if not batch.get("cluster") and n.get("description")
        ]
        for name in unclustered_names[:3]:
            try:
                per_name = search_similar_names(name["description"], k=3)
                for r in per_name:
                    if r.get("id", "") not in batch_ids:
                        filtered.append(r)
            except Exception:
                continue

    # Deduplicate by id
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in filtered:
        rid = r.get("id", "")
        if rid and rid not in seen:
            seen.add(rid)
            deduped.append(r)
        if len(deduped) >= k:
            break

    # Return summary-only dicts
    return [
        {
            "id": r.get("id", ""),
            "description": r.get("description", ""),
            "kind": r.get("kind", ""),
            "unit": r.get("unit", ""),
            "review_tier": r.get("review_tier", ""),
        }
        for r in deduped
    ]
