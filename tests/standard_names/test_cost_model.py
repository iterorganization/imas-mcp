"""Tests for the pipeline-aware cost estimator (cost_model.py).

All tests exercise pure functions — no graph, no mocks, no I/O.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.cost_model import (
    CycleEstimates,
    PipelineBuckets,
    PoolCpi,
    compute_cycle_estimates,
    compute_pipeline_etc,
    detect_stall,
    resolve_pool_cpi,
)

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

_EMPTY_BUCKETS = PipelineBuckets(
    a_sources=0,
    b_drafted_unreviewed=0,
    c_refine_pending=0,
    d_accepted_no_docs=0,
    e_docs_unreviewed=0,
    f_refine_docs_pending=0,
)

_DEFAULT_CYCLES = CycleEstimates(
    r_n_effective=0.3,
    r_d_effective=0.3,
    eta=1.0,
    p_accept=0.85,
    n_obs=0,
)

_UNIT_CPI = {
    pool: PoolCpi(value=1.0, source="fallback")
    for pool in [
        "generate_name",
        "review_name",
        "refine_name",
        "generate_docs",
        "review_docs",
        "refine_docs",
    ]
}


# ═══════════════════════════════════════════════════════════════════════
# 1. Empty pipeline → None
# ═══════════════════════════════════════════════════════════════════════


def test_empty_pipeline_returns_none():
    """All buckets zero → ETC should be None (no remaining work)."""
    result = compute_pipeline_etc(
        buckets=_EMPTY_BUCKETS,
        cycles=_DEFAULT_CYCLES,
        cpis=_UNIT_CPI,
        accumulated_cost=5.0,
    )
    assert result is None


# ═══════════════════════════════════════════════════════════════════════
# 2. Cold start uses priors
# ═══════════════════════════════════════════════════════════════════════


def test_cold_start_uses_priors():
    """Only A_sources > 0, no observations → uses priors (R_n=0.3, η=1.0)."""
    cycles = compute_cycle_estimates(
        refine_name_done=0,
        name_review_first_pass_done=0,
        refine_docs_done=0,
        docs_review_first_pass_done=0,
        accepted_count=0,
        total_completed_name_stage=0,
        sources_attempted=0,
        names_drafted=0,
    )
    assert cycles.r_n_effective == pytest.approx(0.3)
    assert cycles.r_d_effective == pytest.approx(0.3)
    assert cycles.eta == pytest.approx(1.0)
    assert cycles.p_accept == pytest.approx(0.85)

    buckets = PipelineBuckets(
        a_sources=100,
        b_drafted_unreviewed=0,
        c_refine_pending=0,
        d_accepted_no_docs=0,
        e_docs_unreviewed=0,
        f_refine_docs_pending=0,
    )
    result = compute_pipeline_etc(
        buckets=buckets,
        cycles=cycles,
        cpis=_UNIT_CPI,
        accumulated_cost=0.0,
    )
    assert result is not None
    # With unit CPI ($1 each) and 100 sources, the projection should
    # be positive and sane (not astronomically large).
    assert result > 0
    # Rough check: 100 sources × η=1 through ~6 pools with refine cycles
    # should be somewhere in hundreds, not millions.
    assert result < 10000


# ═══════════════════════════════════════════════════════════════════════
# 3. Observed cycles converge (n_obs >= 5)
# ═══════════════════════════════════════════════════════════════════════


def test_observed_cycles_converge():
    """With n_obs >= 5, R_n_effective equals the observed ratio exactly."""
    cycles = compute_cycle_estimates(
        refine_name_done=4,
        name_review_first_pass_done=10,
        refine_docs_done=2,
        docs_review_first_pass_done=10,
        accepted_count=8,
        total_completed_name_stage=10,
        sources_attempted=20,
        names_drafted=18,
    )
    # n_obs_name = 10 ≥ 5 → weight = 1.0 → R_n_eff = 4/10 = 0.4
    assert cycles.r_n_effective == pytest.approx(0.4)
    # n_obs_docs = 10 ≥ 5 → weight = 1.0 → R_d_eff = 2/10 = 0.2
    assert cycles.r_d_effective == pytest.approx(0.2)
    # η = 18/20 = 0.9
    assert cycles.eta == pytest.approx(0.9)
    # P_accept = 8/10 = 0.8 (n_obs ≥ 5 → fully observed)
    assert cycles.p_accept == pytest.approx(0.8)


# ═══════════════════════════════════════════════════════════════════════
# 4. Partial observation smooths
# ═══════════════════════════════════════════════════════════════════════


def test_partial_observation_smooths():
    """n_obs = 2 → weight = 0.4, blends prior and observed."""
    cycles = compute_cycle_estimates(
        refine_name_done=1,
        name_review_first_pass_done=2,
        refine_docs_done=0,
        docs_review_first_pass_done=2,
        accepted_count=1,
        total_completed_name_stage=2,
        sources_attempted=5,
        names_drafted=3,
    )
    # n_obs_name = 2, weight = 2/5 = 0.4
    # R_n_observed = 1/2 = 0.5
    # R_n_eff = 0.4*0.5 + 0.6*0.3 = 0.2 + 0.18 = 0.38
    assert cycles.r_n_effective == pytest.approx(0.38)

    # n_obs_docs = 2, weight = 0.4
    # R_d_observed = 0/2 = 0.0
    # R_d_eff = 0.4*0.0 + 0.6*0.3 = 0.18
    assert cycles.r_d_effective == pytest.approx(0.18)

    # η = 3/5 = 0.6
    assert cycles.eta == pytest.approx(0.6)


# ═══════════════════════════════════════════════════════════════════════
# 5. Disjoint buckets — no double counting
# ═══════════════════════════════════════════════════════════════════════


def test_disjoint_buckets_no_double_count():
    """A=10, B=5, C=3 — verify gen_name sees only A, review_name sees
    contributions from A+B+C correctly per the pipeline formula."""
    buckets = PipelineBuckets(
        a_sources=10,
        b_drafted_unreviewed=5,
        c_refine_pending=3,
        d_accepted_no_docs=0,
        e_docs_unreviewed=0,
        f_refine_docs_pending=0,
    )
    cycles = CycleEstimates(
        r_n_effective=0.3,
        r_d_effective=0.3,
        eta=1.0,
        p_accept=0.85,
        n_obs=10,
    )

    # Use distinct CPIs per pool to verify each pool's contribution.
    cpis = {
        "generate_name": PoolCpi(value=2.0, source="observed"),
        "review_name": PoolCpi(value=1.0, source="observed"),
        "refine_name": PoolCpi(value=1.5, source="observed"),
        "generate_docs": PoolCpi(value=0.0, source="fallback"),
        "review_docs": PoolCpi(value=0.0, source="fallback"),
        "refine_docs": PoolCpi(value=0.0, source="fallback"),
    }

    result = compute_pipeline_etc(
        buckets=buckets, cycles=cycles, cpis=cpis, accumulated_cost=0.0
    )
    assert result is not None

    # Manual calculation:
    # gen_name_remaining = 10 * 1.0 = 10
    gen_name = 10 * 1.0
    # review_name_remaining = 10*1*(1+0.3) + 5*(1+0.3) + 3*1 = 13 + 6.5 + 3 = 22.5
    review_name = 10 * 1.0 * 1.3 + 5 * 1.3 + 3 * 1.0
    # refine_name_remaining = 10*1*0.3 + 5*0.3 + 3*1 = 3 + 1.5 + 3 = 7.5
    refine_name = 10 * 1.0 * 0.3 + 5 * 0.3 + 3 * 1.0

    expected = gen_name * 2.0 + review_name * 1.0 + refine_name * 1.5
    assert result == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════
# 6. P_accept smoothing
# ═══════════════════════════════════════════════════════════════════════


def test_p_accept_smoothing():
    """Fewer than 5 observations uses prior (0.85); 5+ uses observed."""
    # Under threshold: 3 completed, 2 accepted → blend
    cycles_few = compute_cycle_estimates(
        refine_name_done=0,
        name_review_first_pass_done=3,
        refine_docs_done=0,
        docs_review_first_pass_done=0,
        accepted_count=2,
        total_completed_name_stage=3,
        sources_attempted=5,
        names_drafted=5,
    )
    # weight = 3/5 = 0.6, p_obs = 2/3, p_accept = 0.6*(2/3) + 0.4*0.85
    expected_few = 0.6 * (2 / 3) + 0.4 * 0.85
    assert cycles_few.p_accept == pytest.approx(expected_few)

    # Over threshold: 10 completed, 6 accepted → fully observed
    cycles_many = compute_cycle_estimates(
        refine_name_done=0,
        name_review_first_pass_done=10,
        refine_docs_done=0,
        docs_review_first_pass_done=0,
        accepted_count=6,
        total_completed_name_stage=10,
        sources_attempted=10,
        names_drafted=10,
    )
    assert cycles_many.p_accept == pytest.approx(0.6)


# ═══════════════════════════════════════════════════════════════════════
# 7. CPI priority chain
# ═══════════════════════════════════════════════════════════════════════


def test_resolve_pool_cpi_priority():
    """observed > historical > sibling fallback; below min_obs falls through."""
    # Observed wins (>= min_obs items)
    cpi = resolve_pool_cpi(
        pool="review_name",
        observed_cost=3.0,
        observed_completed=5,
        historical={"review_name": 0.5},
        sibling_cpi=0.8,
        min_obs=3,
    )
    assert cpi.value == pytest.approx(0.6)  # 3.0 / 5
    assert cpi.source == "observed"

    # Below min_obs → falls to historical
    cpi_hist = resolve_pool_cpi(
        pool="review_name",
        observed_cost=1.0,
        observed_completed=2,
        historical={"review_name": 0.5},
        sibling_cpi=0.8,
        min_obs=3,
    )
    assert cpi_hist.value == pytest.approx(0.5)
    assert cpi_hist.source == "historical"

    # No historical → sibling fallback
    cpi_sib = resolve_pool_cpi(
        pool="refine_name",
        observed_cost=0.0,
        observed_completed=0,
        historical={"review_name": 0.5},  # no refine_name key
        sibling_cpi=0.5,
        min_obs=3,
    )
    assert cpi_sib.value == pytest.approx(0.5)
    assert cpi_sib.source == "fallback"

    # No sibling → zero fallback
    cpi_zero = resolve_pool_cpi(
        pool="refine_name",
        observed_cost=0.0,
        observed_completed=0,
        historical=None,
        sibling_cpi=None,
        min_obs=3,
    )
    assert cpi_zero.value == pytest.approx(0.0)
    assert cpi_zero.source == "fallback"


# ═══════════════════════════════════════════════════════════════════════
# 8. Stall detection
# ═══════════════════════════════════════════════════════════════════════


def test_stall_detection():
    """pending > 0 + stale last_completion_at → stalled."""
    now = 1000.0
    result = detect_stall(
        pool_pending={"review_name": 5, "generate_name": 0},
        pool_last_completion_at={"review_name": 600.0, "generate_name": None},
        now_ts=now,
        timeout_s=300.0,
    )
    # review_name: pending=5, last_at=600, now-600=400 > 300 → stalled
    assert result is True


# ═══════════════════════════════════════════════════════════════════════
# 9. No stall when no pending
# ═══════════════════════════════════════════════════════════════════════


def test_no_stall_when_no_pending():
    """pending = 0 returns False even if last_completion_at is old."""
    now = 1000.0
    result = detect_stall(
        pool_pending={"review_name": 0, "generate_name": 0},
        pool_last_completion_at={"review_name": 100.0, "generate_name": 100.0},
        now_ts=now,
        timeout_s=300.0,
    )
    assert result is False


# ═══════════════════════════════════════════════════════════════════════
# Additional edge-case tests
# ═══════════════════════════════════════════════════════════════════════


def test_full_pipeline_projection():
    """End-to-end projection with all buckets populated."""
    buckets = PipelineBuckets(
        a_sources=50,
        b_drafted_unreviewed=20,
        c_refine_pending=5,
        d_accepted_no_docs=15,
        e_docs_unreviewed=10,
        f_refine_docs_pending=3,
    )
    cycles = CycleEstimates(
        r_n_effective=0.25,
        r_d_effective=0.20,
        eta=0.9,
        p_accept=0.80,
        n_obs=20,
    )
    cpis = {
        "generate_name": PoolCpi(value=0.01, source="observed"),
        "review_name": PoolCpi(value=0.005, source="observed"),
        "refine_name": PoolCpi(value=0.008, source="historical"),
        "generate_docs": PoolCpi(value=0.02, source="observed"),
        "review_docs": PoolCpi(value=0.006, source="historical"),
        "refine_docs": PoolCpi(value=0.006, source="fallback"),
    }
    result = compute_pipeline_etc(
        buckets=buckets,
        cycles=cycles,
        cpis=cpis,
        accumulated_cost=2.50,
    )
    assert result is not None
    # Accumulated cost is included in total.
    assert result > 2.50
    # Total should be reasonable for ~100 items through the pipeline.
    assert result < 100.0


def test_only_docs_buckets():
    """Pipeline with only docs-phase work remaining (D, E, F)."""
    buckets = PipelineBuckets(
        a_sources=0,
        b_drafted_unreviewed=0,
        c_refine_pending=0,
        d_accepted_no_docs=30,
        e_docs_unreviewed=10,
        f_refine_docs_pending=2,
    )
    result = compute_pipeline_etc(
        buckets=buckets,
        cycles=_DEFAULT_CYCLES,
        cpis=_UNIT_CPI,
        accumulated_cost=0.0,
    )
    assert result is not None
    # gen_name_remaining = 0 (no A sources)
    # review_name_remaining = 0 (no A, B, C for name)
    # gen_docs_remaining = 0 + 30 = 30
    # review_docs_remaining = 30*(1+0.3) + 10*(1+0.3) + 2*1 = 39+13+2 = 54
    # refine_docs_remaining = 30*0.3 + 10*0.3 + 2*1 = 9+3+2 = 14
    expected = 0 + 0 + 0 + 30 + 54 + 14  # = 98 with unit CPI
    assert result == pytest.approx(expected)


def test_stall_not_triggered_for_never_completed():
    """Pool that has never completed doesn't trigger stall (grace period)."""
    now = 1000.0
    result = detect_stall(
        pool_pending={"review_name": 5},
        pool_last_completion_at={"review_name": None},
        now_ts=now,
        timeout_s=300.0,
    )
    # Never completed = no stall (might be starting up).
    assert result is False
