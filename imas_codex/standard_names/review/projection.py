"""Canonical-review projection helper.

Given a StandardName ID and review axis (``"names"`` or ``"docs"``),
derives the single authoritative :class:`CanonicalReview` record from the
``StandardNameReview`` nodes attached to that SN in the graph — regardless of how many
review cycles ran or how many review groups exist.

This module **defines** the canonical projection semantics that the rest of
the W40 migration converges toward.  It is a read-only helper: it never
modifies the graph.  Future consumers (e.g. ``sn status --deep``, reviewer-
disagreement audits) should call :func:`project_canonical_review` rather
than querying StandardNameReview nodes directly.

Precedence (from most-authoritative to least)
---------------------------------------------
1. The most-recent ``review_group_id`` for this SN+axis is selected first.
   Multiple groups may exist when an SN has been re-reviewed; the latest
   group is canonical.
2. **Escalator branch** — ``cycle_index = 2 AND
   resolution_method = 'authoritative_escalation'`` → ``source="escalator"``.
   A cycle-2 node carrying ``max_cycles_reached`` is *not* an escalator and
   falls through.
3. **Quorum-mean branch** — cycles {0, 1} both present in the group AND
   neither carries ``resolution_method ∈ {'retry_item', 'max_cycles_reached'}``
   AND at least one has ``resolution_method = 'quorum_consensus'`` →
   mean of cycles 0 and 1, ``source="quorum_mean"``.
4. **Single branch** — cycle 0 exists → return it as-is,
   ``source="single"``.  Covers ``single_review``, ``retry_item`` on cycle 0
   only, and any mid-cycle aborted run.
5. **None** — no cycle 0 found → return ``None``, ``source="none"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from imas_codex.graph.client import GraphClient

# resolution_method values that disqualify a cycle from the quorum-mean branch
_DISQUALIFYING_METHODS = {"retry_item", "max_cycles_reached"}


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class CanonicalReview:
    """Canonical review result for a StandardName on one axis.

    Attributes
    ----------
    score:
        Aggregate normalised score on [0, 1], or ``None`` when unavailable.
    scores_json:
        JSON string of per-dimension raw scores (0–20 each), or ``None``.
    model:
        LLM model slug used for the authoritative review cycle.
    comments:
        Free-text reviewer comments (possibly concatenated for quorum-mean).
    comments_per_dim:
        JSON string of per-dimension comment strings, or ``None``.
    tier:
        Qualitative tier label (``"outstanding"``, ``"good"``,
        ``"inadequate"``, ``"poor"``).
    source:
        Which branch was taken:

        * ``"escalator"`` — cycle-2 authoritative escalation
        * ``"quorum_mean"`` — mean of cycle-0 and cycle-1 quorum
        * ``"single"`` — only cycle-0 available
        * ``"none"`` — no StandardNameReview nodes found
    """

    score: float | None
    scores_json: str | None
    model: str | None
    comments: str | None
    comments_per_dim: str | None
    tier: str | None
    source: Literal["escalator", "quorum_mean", "single", "none"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mean_scores_json(s0: str | None, s1: str | None) -> str | None:
    """Return a JSON string of per-dimension means from two scores_json values.

    Gracefully handles ``None``, empty strings, and malformed JSON by
    treating them as empty dicts.  Returns ``None`` when both inputs are
    empty/invalid.
    """
    try:
        d0: dict[str, float] = json.loads(s0 or "{}")
    except (json.JSONDecodeError, TypeError):
        d0 = {}
    try:
        d1: dict[str, float] = json.loads(s1 or "{}")
    except (json.JSONDecodeError, TypeError):
        d1 = {}

    if not d0 and not d1:
        return None

    all_dims = set(d0) | set(d1)
    mean: dict[str, float] = {}
    for dim in all_dims:
        v0 = float(d0.get(dim, d1.get(dim, 0.0)))
        v1 = float(d1.get(dim, d0.get(dim, 0.0)))
        mean[dim] = (v0 + v1) / 2.0
    return json.dumps(mean)


def _mean_score(s0: float | None, s1: float | None) -> float | None:
    """Return the mean of two aggregate scores, or whichever is non-None."""
    if s0 is None and s1 is None:
        return None
    if s0 is None:
        return float(s1)  # type: ignore[arg-type]
    if s1 is None:
        return float(s0)
    return (float(s0) + float(s1)) / 2.0


def _row_to_canonical(
    row: dict, *, source: Literal["escalator", "quorum_mean", "single", "none"]
) -> CanonicalReview:
    """Build a :class:`CanonicalReview` from a raw graph query row."""
    return CanonicalReview(
        score=row.get("score"),
        scores_json=row.get("scores_json"),
        model=row.get("model"),
        comments=row.get("comments"),
        comments_per_dim=row.get("comments_per_dim"),
        tier=row.get("tier"),
        source=source,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CYPHER = """
MATCH (sn:StandardName {id: $sn_id})-[:HAS_REVIEW]->(r:StandardNameReview {review_axis: $axis})
RETURN r.review_group_id       AS review_group_id,
       r.cycle_index            AS cycle_index,
       r.resolution_method      AS resolution_method,
       r.score                  AS score,
       r.scores_json            AS scores_json,
       r.reviewer_model         AS model,
       r.comments               AS comments,
       r.comments_per_dim_json  AS comments_per_dim,
       r.tier                   AS tier
ORDER BY r.review_group_id DESC, r.cycle_index ASC
"""


def project_canonical_review(
    sn_id: str,
    axis: Literal["names", "docs"],
    gc: GraphClient,
) -> CanonicalReview | None:
    """Project the canonical :class:`CanonicalReview` for *sn_id* on *axis*.

    Parameters
    ----------
    sn_id:
        The ``StandardName.id`` to project.
    axis:
        Review axis — ``"names"`` or ``"docs"``.
    gc:
        An open :class:`~imas_codex.graph.client.GraphClient` instance.

    Returns
    -------
    CanonicalReview
        The projected canonical review result.
    None
        When no ``StandardNameReview`` nodes exist for this SN+axis combination.
    """
    rows = list(gc.query(_CYPHER, sn_id=sn_id, axis=axis))

    if not rows:
        return None

    # --- Step 1: restrict to the most-recent review_group_id ---------------
    latest_group = rows[0]["review_group_id"]
    group_rows = [r for r in rows if r["review_group_id"] == latest_group]

    # Index rows by cycle_index for O(1) lookup
    by_cycle: dict[int, dict] = {}
    for row in group_rows:
        ci = row.get("cycle_index")
        if ci is not None:
            by_cycle[int(ci)] = row

    # --- Step 2: escalator branch -------------------------------------------
    cycle2 = by_cycle.get(2)
    if (
        cycle2 is not None
        and cycle2.get("resolution_method") == "authoritative_escalation"
    ):
        return _row_to_canonical(cycle2, source="escalator")

    # --- Step 3: quorum-mean branch -----------------------------------------
    cycle0 = by_cycle.get(0)
    cycle1 = by_cycle.get(1)
    if (
        cycle0 is not None
        and cycle1 is not None
        and cycle0.get("resolution_method") not in _DISQUALIFYING_METHODS
        and cycle1.get("resolution_method") not in _DISQUALIFYING_METHODS
        and (
            cycle0.get("resolution_method") == "quorum_consensus"
            or cycle1.get("resolution_method") == "quorum_consensus"
        )
    ):
        mean_score_val = _mean_score(cycle0.get("score"), cycle1.get("score"))
        mean_scores = _mean_scores_json(
            cycle0.get("scores_json"), cycle1.get("scores_json")
        )
        return CanonicalReview(
            score=mean_score_val,
            scores_json=mean_scores,
            model=cycle0.get("model"),
            comments=cycle0.get("comments"),
            comments_per_dim=cycle0.get("comments_per_dim"),
            tier=cycle0.get("tier"),
            source="quorum_mean",
        )

    # --- Step 4: single branch ----------------------------------------------
    if cycle0 is not None:
        return _row_to_canonical(cycle0, source="single")

    # --- Step 5: none -------------------------------------------------------
    return None
