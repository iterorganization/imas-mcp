"""Helpers for walking REFINED_FROM and DOCS_REVISION_OF chains.

These functions serialise graph chain history into prompt-renderable dicts
for the refine_name and refine_docs LLM workers (P2.3 / P4.3 will wire them
in).  Results are ordered oldest → newest, capped at *limit* entries.
"""

from __future__ import annotations

import json

from imas_codex.graph.client import GraphClient

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def name_chain_history(sn_id: str, *, limit: int = 5) -> list[dict]:
    """Return REFINED_FROM ancestors of *sn_id*, oldest first.

    Walks ``(:StandardName {id: $sn_id})-[:REFINED_FROM*1..]->(ancestor)``
    and returns up to *limit* entries ordered by ``chain_length ASC``
    (i.e. oldest / shortest chain first).

    Each entry dict contains:

    * ``name``                      – ancestor ``id`` (= the standard name string)
    * ``model``                     – LLM model that produced the attempt
    * ``reviewer_score``            – numeric score (0–1) from the name reviewer
    * ``reviewer_verdict``          – accept / revise / reject string
    * ``reviewer_comments_per_dim`` – per-dimension comments as ``{dim: comment}``
    * ``generated_at``              – ISO-8601 datetime string or ``None``
    """
    cypher = """
        MATCH (sn:StandardName {id: $sn_id})-[:REFINED_FROM*1..]->(ancestor:StandardName)
        WITH ancestor
        ORDER BY ancestor.chain_length ASC
        LIMIT $limit
        RETURN
          ancestor.id                               AS name,
          ancestor.model                            AS model,
          coalesce(ancestor.reviewer_score_name,
                   ancestor.reviewer_score)         AS reviewer_score,
          ancestor.reviewer_verdict                 AS reviewer_verdict,
          ancestor.reviewer_comments_per_dim_name   AS reviewer_comments_per_dim,
          ancestor.generated_at                     AS generated_at
    """
    with GraphClient() as gc:
        rows = gc.query(cypher, sn_id=sn_id, limit=limit)

    out: list[dict] = []
    for row in rows:
        out.append(
            {
                "name": row["name"],
                "model": row.get("model") or "unknown",
                "reviewer_score": float(row.get("reviewer_score") or 0.0),
                "reviewer_verdict": row.get("reviewer_verdict") or "unknown",
                "reviewer_comments_per_dim": _parse_comments(
                    row.get("reviewer_comments_per_dim")
                ),
                "generated_at": row.get("generated_at"),
            }
        )
    return out


def docs_chain_history(sn_id: str, *, limit: int = 5) -> list[dict]:
    """Return ``DOCS_REVISION_OF`` snapshots for *sn_id*, oldest first.

    Walks ``(:StandardName {id: $sn_id})-[:DOCS_REVISION_OF]->(rev:DocsRevision)``
    and returns up to *limit* entries ordered by ``created_at ASC``.

    Each entry dict contains:

    * ``documentation``             – prior documentation text
    * ``model``                     – LLM model that produced the snapshot
    * ``reviewer_score``            – numeric docs score (0–1)
    * ``reviewer_comments_per_dim`` – per-dimension comments as ``{dim: comment}``
    * ``created_at``                – ISO-8601 datetime string or ``None``
    """
    cypher = """
        MATCH (sn:StandardName {id: $sn_id})-[:DOCS_REVISION_OF]->(rev:DocsRevision)
        WITH rev
        ORDER BY rev.created_at ASC
        LIMIT $limit
        RETURN
          rev.documentation                   AS documentation,
          rev.model                           AS model,
          rev.reviewer_score_docs             AS reviewer_score,
          rev.reviewer_comments_per_dim_docs  AS reviewer_comments_per_dim,
          rev.created_at                      AS created_at
    """
    with GraphClient() as gc:
        rows = gc.query(cypher, sn_id=sn_id, limit=limit)

    out: list[dict] = []
    for row in rows:
        out.append(
            {
                "documentation": row.get("documentation") or "",
                "model": row.get("model") or "unknown",
                "reviewer_score": float(row.get("reviewer_score") or 0.0),
                "reviewer_comments_per_dim": _parse_comments(
                    row.get("reviewer_comments_per_dim")
                ),
                "created_at": row.get("created_at"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_comments(raw: object) -> dict[str, str]:
    """Normalise reviewer comments to a ``{dimension: comment}`` dict.

    Comments may be stored in the graph as a JSON string (legacy) or already
    decoded as a Python dict.  Returns an empty dict for ``None`` or
    unparseable values.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
