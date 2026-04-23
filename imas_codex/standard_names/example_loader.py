"""Dynamic scored-example loader for standard-name prompt injection.

Queries reviewed StandardName nodes from the graph at fixed absolute
score targets and returns them as dicts ready for Jinja template rendering.

Two public functions share the same underlying Cypher logic:

- ``load_compose_examples`` — injected into compose system prompts.
- ``load_review_examples`` — injected into review/review_names/review_docs prompts.

Zero-opp at cold start: returns ``[]`` when no reviewed names exist.
Naturally rich as the graph fills with reviewed entries.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

# Default target scores span the full quality range:
#   1.00 = outstanding, 0.80 = good, 0.65 = publish threshold, 0.40 = poor
_DEFAULT_TARGETS: tuple[float, ...] = (1.00, 0.80, 0.65, 0.40)
_DEFAULT_TOLERANCE: float = 0.05
_DEFAULT_PER_BUCKET: int = 1

# Axis literal shared across public loaders. Callers must be explicit about
# which axis of review their examples target so that axis-split storage
# (see AGENTS.md "Standard Names / StandardName Lifecycle") is respected.
Axis = Literal["name", "docs", "full"]


def _parse_json_field(value: Any) -> dict:
    """Parse a JSON string field into a dict, tolerating None and bad JSON."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _axis_projection(axis: Axis) -> dict[str, str]:
    """Cypher projection fragments for each axis.

    Returns a dict with keys ``score``, ``verdict``, ``scores``,
    ``comments_per_dim``, ``comments``. Each value is a Cypher expression
    that reads from the axis-specific column with fallback to the shared
    slot — this lets pre-migration rows (where only shared slots are
    populated) still surface as examples.
    """
    if axis == "name":
        return {
            "score": "coalesce(sn.reviewer_score_name, sn.reviewer_score)",
            "verdict": "coalesce(sn.reviewer_verdict_name, sn.reviewer_verdict)",
            "scores": "coalesce(sn.reviewer_scores_name, sn.reviewer_scores)",
            "comments_per_dim": (
                "coalesce(sn.reviewer_comments_per_dim_name, "
                "sn.reviewer_comments_per_dim)"
            ),
            "comments": "coalesce(sn.reviewer_comments_name, sn.reviewer_comments)",
        }
    if axis == "docs":
        return {
            "score": "sn.reviewer_score_docs",
            "verdict": "sn.reviewer_verdict_docs",
            "scores": "sn.reviewer_scores_docs",
            "comments_per_dim": "sn.reviewer_comments_per_dim_docs",
            "comments": "sn.reviewer_comments_docs",
        }
    # full
    return {
        "score": "sn.reviewer_score",
        "verdict": "sn.reviewer_verdict",
        "scores": "sn.reviewer_scores",
        "comments_per_dim": "sn.reviewer_comments_per_dim",
        "comments": "sn.reviewer_comments",
    }


def _query_examples_for_target(
    gc: GraphClient,
    target: float,
    tolerance: float,
    per_bucket: int,
    physics_domains: list[str],
    axis: Axis,
) -> list[dict[str, Any]]:
    """Query reviewed StandardName nodes closest to a single target score.

    Tries domain-scoped query first; falls back to all domains when
    the scoped query returns zero rows.
    """
    proj = _axis_projection(axis)
    score_expr = proj["score"]
    verdict_expr = proj["verdict"]
    scores_expr = proj["scores"]
    cpd_expr = proj["comments_per_dim"]
    comments_expr = proj["comments"]

    # --- Domain-scoped query ---
    if physics_domains:
        rows = gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE {score_expr} IS NOT NULL
              AND {verdict_expr} IS NOT NULL
              AND sn.physics_domain IN $domains
              AND abs({score_expr} - $target) <= $tolerance
            RETURN sn.id AS id,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   sn.unit AS unit,
                   {score_expr} AS reviewer_score,
                   {verdict_expr} AS reviewer_verdict,
                   {scores_expr} AS reviewer_scores_json,
                   {cpd_expr} AS reviewer_comments_per_dim_json,
                   {comments_expr} AS reviewer_comments,
                   sn.physics_domain AS physics_domain
            ORDER BY abs({score_expr} - $target) ASC, sn.id ASC
            LIMIT $per_bucket
            """,
            domains=physics_domains,
            target=target,
            tolerance=tolerance,
            per_bucket=per_bucket,
        )
        if rows:
            return [dict(r) for r in rows]

    # --- Fallback: all domains ---
    rows = gc.query(
        f"""
        MATCH (sn:StandardName)
        WHERE {score_expr} IS NOT NULL
          AND {verdict_expr} IS NOT NULL
          AND abs({score_expr} - $target) <= $tolerance
        RETURN sn.id AS id,
               sn.description AS description,
               sn.documentation AS documentation,
               sn.kind AS kind,
               sn.unit AS unit,
               {score_expr} AS reviewer_score,
               {verdict_expr} AS reviewer_verdict,
               {scores_expr} AS reviewer_scores_json,
               {cpd_expr} AS reviewer_comments_per_dim_json,
               {comments_expr} AS reviewer_comments,
               sn.physics_domain AS physics_domain
        ORDER BY abs({score_expr} - $target) ASC, sn.id ASC
        LIMIT $per_bucket
        """,
        target=target,
        tolerance=tolerance,
        per_bucket=per_bucket,
    )
    return [dict(r) for r in rows] if rows else []


def _project_example(row: dict[str, Any], target: float) -> dict[str, Any]:
    """Project a raw graph row into the dict shape expected by templates.

    Template keys:
      - id, description, documentation, kind, unit
      - reviewer_score (float), reviewer_verdict (str)
      - scores (dict — parsed from reviewer_scores_json)
      - dimension_comments (dict — parsed from reviewer_comments_per_dim_json)
      - reviewer_comments (str)
      - physics_domain (str)
      - target_score (float — the bucket this example was selected for)

    Template aliases (backwards-compat with existing template variables):
      - score → reviewer_score
      - tier → reviewer_verdict
      - domain → physics_domain
      - issues → [] (not tracked at row level)
      - verdict → reviewer_verdict
    """
    scores = _parse_json_field(row.get("reviewer_scores_json"))
    comments = _parse_json_field(row.get("reviewer_comments_per_dim_json"))

    return {
        # Core fields
        "id": row.get("id", ""),
        "description": row.get("description", ""),
        "documentation": row.get("documentation", ""),
        "kind": row.get("kind", ""),
        "unit": row.get("unit", ""),
        # Review fields
        "reviewer_score": row.get("reviewer_score"),
        "reviewer_verdict": row.get("reviewer_verdict", ""),
        "scores": scores,
        "dimension_comments": comments,
        "reviewer_comments": row.get("reviewer_comments", ""),
        "physics_domain": row.get("physics_domain", ""),
        "target_score": target,
        # Template aliases
        "score": row.get("reviewer_score"),
        "tier": row.get("reviewer_verdict", ""),
        "domain": row.get("physics_domain", ""),
        "issues": [],
        "verdict": row.get("reviewer_verdict", ""),
    }


def _load_examples(
    gc: GraphClient,
    physics_domains: list[str],
    *,
    axis: Axis,
    target_scores: tuple[float, ...] = _DEFAULT_TARGETS,
    tolerance: float = _DEFAULT_TOLERANCE,
    per_bucket: int = _DEFAULT_PER_BUCKET,
) -> list[dict[str, Any]]:
    """Shared loader used by both compose and review example functions.

    Iterates target_scores in descending order, querying per-bucket
    examples for each target. Returns deterministic ordering:
    (target_score DESC, name.id ASC).
    """
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for target in sorted(target_scores, reverse=True):
        rows = _query_examples_for_target(
            gc, target, tolerance, per_bucket, physics_domains, axis
        )
        for row in rows:
            name_id = row.get("id", "")
            if name_id and name_id not in seen_ids:
                seen_ids.add(name_id)
                results.append(_project_example(row, target))

    if results:
        logger.debug(
            "Loaded %d scored examples (axis=%s, domains=%s, targets=%s)",
            len(results),
            axis,
            physics_domains or "all",
            target_scores,
        )

    return results


def load_compose_examples(
    gc: GraphClient,
    physics_domains: list[str],
    *,
    axis: Axis,
    target_scores: tuple[float, ...] = _DEFAULT_TARGETS,
    tolerance: float = _DEFAULT_TOLERANCE,
    per_bucket: int = _DEFAULT_PER_BUCKET,
) -> list[dict[str, Any]]:
    """Query reviewed StandardName nodes closest to each target score.

    For each target in *target_scores*, find up to *per_bucket* names whose
    ``reviewer_score`` is closest (``|score - target| <= tolerance``),
    preferring names whose ``physics_domain`` is in *physics_domains*.
    Fall back to all domains when the domain-scoped query returns nothing
    for a given target.

    ``axis`` is required. The compose worker regenerates name+grammar, so
    it should pass ``axis="name"``.

    Ordering: deterministic by (target_score DESC, name.id ASC) so that once
    the graph stabilises, the example set becomes static and cacheable.

    Returns list of dicts projecting only fields the prompt needs::

        {id, description, documentation, reviewer_score, reviewer_verdict,
         scores (parsed dict), dimension_comments (parsed dict),
         physics_domain, ...}
    """
    return _load_examples(
        gc,
        physics_domains,
        axis=axis,
        target_scores=target_scores,
        tolerance=tolerance,
        per_bucket=per_bucket,
    )


def load_review_examples(
    gc: GraphClient,
    physics_domains: list[str],
    *,
    axis: Axis,
    target_scores: tuple[float, ...] = _DEFAULT_TARGETS,
    tolerance: float = _DEFAULT_TOLERANCE,
    per_bucket: int = _DEFAULT_PER_BUCKET,
) -> list[dict[str, Any]]:
    """Query reviewed StandardName nodes for review prompt calibration.

    Same logic as :func:`load_compose_examples` — the distinction exists
    so callers can evolve projection or target defaults independently.

    ``axis`` must match the reviewer mode calling this function:
    ``axis="name"`` for name-only review, ``axis="docs"`` for docs review,
    ``axis="full"`` for the aggregate 6-dim rubric.
    """
    return _load_examples(
        gc,
        physics_domains,
        axis=axis,
        target_scores=target_scores,
        tolerance=tolerance,
        per_bucket=per_bucket,
    )
