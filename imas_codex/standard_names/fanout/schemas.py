"""Discriminated-union schemas for structured fan-out (plan 39 §3).

The proposer LLM emits a JSON :class:`FanoutPlan` whose ``queries`` are
parsed via a Pydantic discriminated union on ``fn_id``.  Bounds
(``max_fan_degree``, per-call ``k``/``max_results``) are enforced **at
parse time** — there is no agentic loop, no runtime arg validation, and
no ``dict[str, Any]`` payload.

Caller-injected scope (``physics_domain``, ``ids_filter``,
``dd_version``) is held in :class:`FanoutScope` and passed to runners
out-of-band; the proposer never sees those fields.

Module structure:
    - :class:`_SearchExistingNames`, :class:`_SearchDDPaths`,
      :class:`_FindRelatedDDPaths`, :class:`_SearchDDClusters` — typed
      catalog calls (one per ``fn_id``).
    - :data:`FanoutCall` — discriminated union over the four variants.
    - :class:`FanoutPlan` — proposer output (``queries`` list, capped at
      :data:`MAX_FAN_DEGREE`).
    - :class:`FanoutScope` — runtime context injected by the caller.
    - :class:`CandidateContext` — refine-site candidate metadata.
    - :class:`FanoutHit` — single executor hit (kind, id, label, score).
    - :class:`FanoutResult` — per-runner outcome (ok, hits, error,
      elapsed).
    - :class:`FanoutOutcome` — high-level run outcome literal (used by
      the telemetry node).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# =====================================================================
# Hard-coded architectural bounds (plan 39 §7.1)
# =====================================================================

MAX_FAN_DEGREE: int = 3
"""Maximum number of fan-out queries per Stage A plan (parse-time)."""


# =====================================================================
# Catalog call variants — one Pydantic model per fn_id
# =====================================================================


class _SearchExistingNames(BaseModel):
    """Find existing StandardName nodes similar to a description string."""

    fn_id: Literal["search_existing_names"]
    query: str = Field(..., min_length=2, max_length=200)
    k: int = Field(default=5, ge=1, le=10)


class _SearchDDPaths(BaseModel):
    """Hybrid search over DD paths.

    ``ids_filter`` / ``physics_domain`` / ``dd_version`` are supplied by
    the caller via :class:`FanoutScope` (plan 39 §3.3), not by the LLM.
    """

    fn_id: Literal["search_dd_paths"]
    query: str = Field(..., min_length=2, max_length=200)
    k: int = Field(default=8, ge=1, le=15)


class _FindRelatedDDPaths(BaseModel):
    """Cluster / coordinate / unit siblings of a known DD path."""

    fn_id: Literal["find_related_dd_paths"]
    path: str = Field(..., min_length=3, max_length=300)
    max_results: int = Field(default=12, ge=1, le=20)


class _SearchDDClusters(BaseModel):
    """Concept-level semantic-cluster discovery."""

    fn_id: Literal["search_dd_clusters"]
    query: str = Field(..., min_length=2, max_length=200)
    k: int = Field(default=8, ge=1, le=15)


FanoutCall = Annotated[
    _SearchExistingNames | _SearchDDPaths | _FindRelatedDDPaths | _SearchDDClusters,
    Field(discriminator="fn_id"),
]


# =====================================================================
# Plan, scope, candidate context
# =====================================================================


class FanoutPlan(BaseModel):
    """Stage A output — the proposer's chosen catalog calls."""

    queries: list[FanoutCall] = Field(
        default_factory=list,
        max_length=MAX_FAN_DEGREE,
        description="Catalog calls to execute in parallel; <= MAX_FAN_DEGREE.",
    )
    notes: str = Field(
        default="",
        max_length=200,
        description="Free-form proposer commentary; logged only.",
    )


class FanoutScope(BaseModel):
    """Caller-injected scope.  Never LLM-supplied (plan 39 §3.3)."""

    physics_domain: str | None = None
    ids_filter: str | None = None
    dd_version: int | None = None


class CandidateContext(BaseModel):
    """Refine-site candidate metadata passed to fan-out.

    Carries everything Stage A's user prompt needs about the candidate
    being refined.  The ``chain_length`` field drives escalation gating
    upstream of :func:`run_fanout` (plan 39 §7.3); fan-out itself only
    reads it to populate the proposer prompt context.
    """

    sn_id: str
    name: str
    path: str
    description: str = ""
    physics_domain: str = ""
    chain_length: int = 0


# =====================================================================
# Executor outputs
# =====================================================================


class FanoutHit(BaseModel):
    """One hit returned by a catalog runner."""

    kind: Literal["standard_name", "dd_path", "cluster"]
    id: str
    label: str
    score: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class FanoutResult(BaseModel):
    """Per-runner result.

    ``ok=True`` means the runner completed without raising or timing
    out; ``hits`` may still be empty (an empty-but-ok result).
    ``ok=False`` captures both timeouts and exceptions; ``error``
    carries a short diagnostic string.
    """

    fn_id: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    hits: list[FanoutHit] = Field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0


# =====================================================================
# Outcome literal (plan 39 §8.2 — Fanout node `outcome` property)
# =====================================================================


FanoutOutcome = Literal[
    "ok",
    "planner_schema_fail",
    "planner_all_invalid",
    "executor_partial_fail",
    "executor_all_empty",
    "no_budget",
    "off_arm",
]
"""Outcome label written onto the Fanout telemetry node.

``feature_disabled`` is *not* a valid outcome — when fan-out is disabled
no Fanout node is written (S5).
"""


__all__ = [
    "MAX_FAN_DEGREE",
    "FanoutCall",
    "FanoutPlan",
    "FanoutScope",
    "CandidateContext",
    "FanoutHit",
    "FanoutResult",
    "FanoutOutcome",
    "_SearchExistingNames",
    "_SearchDDPaths",
    "_FindRelatedDDPaths",
    "_SearchDDClusters",
]
