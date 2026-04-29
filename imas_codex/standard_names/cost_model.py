"""Pipeline-aware cost estimator for the 6-pool SN pipeline.

Pure-function module — no I/O, no graph access.  All state is passed
in via frozen dataclasses; the caller (display.py) is responsible for
querying graph buckets and pool statistics.

Projection formula follows the disjoint-bucket pipeline-flow model
described in plan.md §F5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ═══════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PipelineBuckets:
    """Disjoint pipeline buckets counted in name-units.

    Each bucket is mutually exclusive by construction (see Cypher
    queries in ``graph_ops.query_pipeline_buckets``).
    """

    a_sources: int  # pre-draft StandardNameSource items (status='extracted')
    b_drafted_unreviewed: int  # name drafted, no reviewer_score_name yet
    c_refine_pending: (
        int  # definite name refines (reviewed, below threshold, under cap)
    )
    d_accepted_no_docs: int  # name accepted, docs not yet drafted
    e_docs_unreviewed: int  # docs drafted, no reviewer_score_docs yet
    f_refine_docs_pending: int  # definite docs refines


@dataclass(frozen=True)
class CycleEstimates:
    """Smoothed cycle-count estimates for the pipeline.

    ``r_n_effective`` and ``r_d_effective`` are prior-smoothed refine
    ratios; ``eta`` is the source→name yield; ``p_accept`` is the
    name-acceptance probability.
    """

    r_n_effective: float  # name-refine ratio (smoothed)
    r_d_effective: float  # docs-refine ratio (smoothed)
    eta: float  # source→name yield (default 1.0)
    p_accept: float  # name acceptance probability (default 0.85)
    n_obs: int  # observation count used for shrinkage weighting


@dataclass(frozen=True)
class PoolCpi:
    """Cost-per-item for a single pool with provenance tracking."""

    value: float
    source: Literal["observed", "historical", "fallback"]


# ═══════════════════════════════════════════════════════════════════════
# Cycle estimation with prior smoothing
# ═══════════════════════════════════════════════════════════════════════


def compute_cycle_estimates(
    refine_name_done: int,
    name_review_first_pass_done: int,
    refine_docs_done: int,
    docs_review_first_pass_done: int,
    accepted_count: int,
    total_completed_name_stage: int,
    sources_attempted: int,
    names_drafted: int,
    r_n_prior: float = 0.3,
    r_d_prior: float = 0.3,
    shrinkage_n: int = 5,
) -> CycleEstimates:
    """Compute smoothed cycle estimates from observed pipeline counters.

    Uses Bayesian-style shrinkage: the weight toward the observed ratio
    increases linearly with ``n_obs`` until ``shrinkage_n`` observations,
    after which the observed ratio dominates.

    Args:
        refine_name_done: Number of completed name refine cycles this run.
        name_review_first_pass_done: Number of names that completed at
            least one name-review pass.
        refine_docs_done: Number of completed docs refine cycles this run.
        docs_review_first_pass_done: Number of names that completed at
            least one docs-review pass.
        accepted_count: Number of names that reached ``accepted`` stage.
        total_completed_name_stage: Total names that completed name-stage
            (reviewed or accepted — denominator for P_accept).
        sources_attempted: Sources that entered the generate_name pool.
        names_drafted: Names successfully drafted from those sources.
        r_n_prior: Prior name-refine ratio (default 0.3).
        r_d_prior: Prior docs-refine ratio (default 0.3).
        shrinkage_n: Number of observations for full convergence.

    Returns:
        ``CycleEstimates`` with smoothed ratios.
    """
    # --- Name refine ratio ---
    n_obs_name = name_review_first_pass_done
    if n_obs_name > 0:
        r_n_observed = refine_name_done / n_obs_name
    else:
        r_n_observed = r_n_prior
    weight_n = min(n_obs_name / shrinkage_n, 1.0) if shrinkage_n > 0 else 1.0
    r_n_eff = weight_n * r_n_observed + (1.0 - weight_n) * r_n_prior

    # --- Docs refine ratio ---
    n_obs_docs = docs_review_first_pass_done
    if n_obs_docs > 0:
        r_d_observed = refine_docs_done / n_obs_docs
    else:
        r_d_observed = r_d_prior
    weight_d = min(n_obs_docs / shrinkage_n, 1.0) if shrinkage_n > 0 else 1.0
    r_d_eff = weight_d * r_d_observed + (1.0 - weight_d) * r_d_prior

    # --- Source → name yield (η) ---
    if sources_attempted > 0 and names_drafted > 0:
        eta = names_drafted / sources_attempted
    else:
        eta = 1.0

    # --- P_accept (acceptance probability) ---
    n_obs_accept = total_completed_name_stage
    if n_obs_accept >= shrinkage_n and n_obs_accept > 0:
        p_accept = accepted_count / n_obs_accept
    elif n_obs_accept > 0:
        p_obs = accepted_count / n_obs_accept
        w = min(n_obs_accept / shrinkage_n, 1.0) if shrinkage_n > 0 else 1.0
        p_accept = w * p_obs + (1.0 - w) * 0.85
    else:
        p_accept = 0.85

    return CycleEstimates(
        r_n_effective=r_n_eff,
        r_d_effective=r_d_eff,
        eta=eta,
        p_accept=p_accept,
        n_obs=n_obs_name,
    )


# ═══════════════════════════════════════════════════════════════════════
# CPI resolution (priority chain)
# ═══════════════════════════════════════════════════════════════════════


def resolve_pool_cpi(
    pool: str,
    observed_cost: float,
    observed_completed: int,
    historical: dict[str, float] | None,
    sibling_cpi: float | None,
    min_obs: int = 3,
) -> PoolCpi:
    """Resolve cost-per-item for a pool using the priority chain.

    Priority: this-run observed > historical (graph) > sibling fallback.

    Args:
        pool: Pool name (e.g. ``"generate_name"``).
        observed_cost: Total cost observed this run for the pool.
        observed_completed: Number of items completed this run.
        historical: Dict of pool→historical CPI from graph (or None).
        sibling_cpi: Fallback CPI from a related pool (or None).
        min_obs: Minimum completed items to trust observed CPI.

    Returns:
        ``PoolCpi`` with resolved value and source provenance.
    """
    # 1. This-run observed (requires min_obs items)
    if observed_completed >= min_obs and observed_cost > 0:
        return PoolCpi(
            value=observed_cost / observed_completed,
            source="observed",
        )

    # 2. Historical from graph
    if historical and pool in historical and historical[pool] > 0:
        return PoolCpi(value=historical[pool], source="historical")

    # 3. Sibling fallback
    if sibling_cpi is not None and sibling_cpi > 0:
        return PoolCpi(value=sibling_cpi, source="fallback")

    # Last resort: zero (pool contributes nothing to projection)
    return PoolCpi(value=0.0, source="fallback")


# ═══════════════════════════════════════════════════════════════════════
# Pipeline ETC projection
# ═══════════════════════════════════════════════════════════════════════


def compute_pipeline_etc(
    buckets: PipelineBuckets,
    cycles: CycleEstimates,
    cpis: dict[str, PoolCpi],
    accumulated_cost: float,
) -> float | None:
    """Compute projected total cost using the pipeline-flow model.

    Each bucket maps to remaining work across the 6 cost-incurring pools
    (generate_name, review_name, refine_name, generate_docs, review_docs,
    refine_docs).  The formula ensures no double-counting: buckets are
    disjoint by construction.

    Args:
        buckets: Disjoint pipeline bucket counts.
        cycles: Smoothed cycle estimates (R_n, R_d, η, P_accept).
        cpis: Cost-per-item per pool.
        accumulated_cost: Current accumulated cost (this run + historical).

    Returns:
        Projected total cost (accumulated + remaining), or ``None`` if
        all buckets are empty (no remaining work).
    """
    A = buckets.a_sources
    B = buckets.b_drafted_unreviewed
    C = buckets.c_refine_pending
    D = buckets.d_accepted_no_docs
    E = buckets.e_docs_unreviewed
    F = buckets.f_refine_docs_pending

    if A + B + C + D + E + F == 0:
        return None

    eta = cycles.eta
    r_n = cycles.r_n_effective
    r_d = cycles.r_d_effective
    p_accept = cycles.p_accept

    # --- Remaining items per pool ---
    gen_name_remaining = A * eta
    review_name_remaining = (
        A * eta * (1.0 + r_n)  # new sources: 1st review + speculative re-reviews
        + B * (1.0 + r_n)  # drafted-unreviewed: 1st review + speculative re-reviews
        + C * 1.0  # definite: re-review after each refine
    )
    refine_name_remaining = (
        A * eta * r_n  # speculative for new sources
        + B * r_n  # speculative for unreviewed drafts
        + C * 1.0  # definite refine (1×)
    )
    gen_docs_remaining = (
        (A * eta + B) * p_accept  # upstream items that will be accepted
        + D  # already accepted, docs pending
    )
    review_docs_remaining = (
        gen_docs_remaining * (1.0 + r_d)
        + E * (1.0 + r_d)
        + F * 1.0  # re-review after definite docs refine
    )
    refine_docs_remaining = (
        gen_docs_remaining * r_d + E * r_d + F * 1.0  # definite docs refine
    )

    # --- Per-pool CPI ---
    def _cpi(pool: str) -> float:
        return cpis[pool].value if pool in cpis else 0.0

    projected_cost = (
        gen_name_remaining * _cpi("generate_name")
        + review_name_remaining * _cpi("review_name")
        + refine_name_remaining * _cpi("refine_name")
        + gen_docs_remaining * _cpi("generate_docs")
        + review_docs_remaining * _cpi("review_docs")
        + refine_docs_remaining * _cpi("refine_docs")
    )

    return accumulated_cost + projected_cost


# ═══════════════════════════════════════════════════════════════════════
# Stall detection
# ═══════════════════════════════════════════════════════════════════════


def detect_stall(
    pool_pending: dict[str, int],
    pool_last_completion_at: dict[str, float | None],
    now_ts: float,
    timeout_s: float = 300.0,
) -> bool:
    """Detect whether any pool with pending work has stalled.

    A pool is stalled if it has pending items **and** the last
    completion was more than ``timeout_s`` seconds ago (or has never
    completed).

    Args:
        pool_pending: Pending item count per pool.
        pool_last_completion_at: Timestamp of last completion per pool
            (``None`` = never completed).
        now_ts: Current timestamp (``time.time()``).
        timeout_s: Stall timeout in seconds (default 300s / 5 minutes).

    Returns:
        ``True`` if any pool is stalled, ``False`` otherwise.
    """
    for pool, pending in pool_pending.items():
        if pending <= 0:
            continue
        last_at = pool_last_completion_at.get(pool)
        if last_at is None:
            # Never completed — only stall if pool has been running
            # long enough (use timeout_s as grace period).
            continue
        if now_ts - last_at > timeout_s:
            return True
    return False
