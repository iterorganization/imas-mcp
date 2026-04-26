"""Regression tests for _adaptive_review_budget share allocation.

Covers the W23B fix: when both generate and enrich phases are skipped
(--only review / --only review_names), review_names must receive 85%
of cost_limit rather than the default 45%.

This ensures that ``--only review --cost-limit 1.50`` allocates ~$1.275
to review_names, clearing the opus-only 15-name batch reservation floor
(~$1.125).  Before the fix, the allocation was $0.675 — below the floor —
causing zero reviews to run.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.turn import PhaseResult, _adaptive_review_budget


def _skipped(name: str) -> PhaseResult:
    return PhaseResult(name=name, skipped=True)


def _spent(name: str, cost: float) -> PhaseResult:
    return PhaseResult(name=name, cost=cost)


# ── Full-pipeline behaviour (unchanged) ──────────────────────────────────────


def test_full_pipeline_review_names_45pct():
    """Full pipeline: review_names gets 45% of remaining (default behaviour)."""
    # generate and enrich both ran (cost=0 simulates cheap/zero-cost runs)
    prior = [_spent("generate", 0.0), _spent("enrich", 0.0)]
    budget = _adaptive_review_budget(1.50, prior, "review_names")
    assert abs(budget - 1.50 * 0.45) < 1e-9, (
        f"Expected {1.50 * 0.45:.4f}, got {budget:.4f}"
    )


def test_full_pipeline_with_generate_spend():
    """Full pipeline: review_names share of remaining after generate spent some."""
    prior = [_spent("generate", 0.50), _spent("enrich", 0.20)]
    # remaining = 1.50 - 0.70 = 0.80
    budget = _adaptive_review_budget(1.50, prior, "review_names")
    assert abs(budget - 0.80 * 0.45) < 1e-9, (
        f"Expected {0.80 * 0.45:.4f}, got {budget:.4f}"
    )


# ── --only review regression (W23B fix) ─────────────────────────────────────


def test_only_review_review_names_85pct():
    """--only review: review_names must receive 85% of cost_limit (W23B fix)."""
    prior = [_skipped("generate"), _skipped("enrich")]
    budget = _adaptive_review_budget(1.50, prior, "review_names")
    assert budget > 0.80, (
        f"review_names budget {budget:.4f} too low for --only review "
        f"(expected >$0.80, should be ~$1.275)"
    )
    assert abs(budget - 1.50 * 0.85) < 1e-9, (
        f"Expected {1.50 * 0.85:.4f} (85% of 1.50), got {budget:.4f}"
    )


def test_only_review_names_85pct():
    """--only review_names: also gets 85% when both generate phases are skipped."""
    prior = [_skipped("generate"), _skipped("enrich")]
    budget = _adaptive_review_budget(2.00, prior, "review_names")
    assert abs(budget - 2.00 * 0.85) < 1e-9, (
        f"Expected {2.00 * 0.85:.4f}, got {budget:.4f}"
    )


def test_only_review_clears_floor_at_1_50():
    """$1.50 cost_limit with --only review must allocate >$1.125 to review_names.

    The opus-only reviewer needs ~$1.125 to reserve a 15-name batch.
    Before W23B the allocation was $0.675 (below floor), blocking all reviews.
    """
    prior = [_skipped("generate"), _skipped("enrich")]
    budget = _adaptive_review_budget(1.50, prior, "review_names")
    assert budget > 1.125, (
        f"review_names budget {budget:.4f} is below the $1.125 batch "
        f"reservation floor — --only review --cost-limit 1.50 would run 0 reviews"
    )


# ── Only generate skipped (partial skip, not review-only) ───────────────────


def test_only_generate_skipped_uses_default_share():
    """Only generate skipped (enrich ran): uses the default 0.45 share."""
    prior = [_skipped("generate"), _spent("enrich", 0.10)]
    budget = _adaptive_review_budget(1.50, prior, "review_names")
    remaining = 1.50 - 0.10
    assert abs(budget - remaining * 0.45) < 1e-9, (
        f"Expected {remaining * 0.45:.4f}, got {budget:.4f}"
    )


def test_only_enrich_skipped_uses_default_share():
    """Only enrich skipped (generate ran): uses the default 0.45 share."""
    prior = [_spent("generate", 0.30), _skipped("enrich")]
    budget = _adaptive_review_budget(1.50, prior, "review_names")
    remaining = 1.50 - 0.30
    assert abs(budget - remaining * 0.45) < 1e-9, (
        f"Expected {remaining * 0.45:.4f}, got {budget:.4f}"
    )


# ── review_docs and regen shares are unaffected ──────────────────────────────


def test_review_docs_share_unchanged_in_review_only_mode():
    """review_docs share (0.35) is not changed by the W23B fix."""
    prior = [_skipped("generate"), _skipped("enrich")]
    budget = _adaptive_review_budget(1.50, prior, "review_docs")
    assert abs(budget - 1.50 * 0.35) < 1e-9, (
        f"review_docs budget should be 35% = {1.50 * 0.35:.4f}, got {budget:.4f}"
    )


def test_regen_share_unchanged_in_review_only_mode():
    """regen gets 100% of remaining regardless of phase configuration."""
    prior = [_skipped("generate"), _skipped("enrich")]
    budget = _adaptive_review_budget(1.50, prior, "regen")
    assert abs(budget - 1.50) < 1e-9, (
        f"regen budget should be 100% = 1.50, got {budget:.4f}"
    )
