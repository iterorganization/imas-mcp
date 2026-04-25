"""Tests for Phase-B budget-split rebalancing and per-phase hard caps.

Covers:
  - TURN_SPLIT_LEGACY and TURN_SPLIT_LEAN constant values
  - TurnConfig.compose_lean flag activates the correct split
  - BudgetManager per-phase cap rejects over-budget reservations
  - Per-phase caps are independent (compose cap does not block review)
  - Both split tuples sum exactly to 1.0
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.turn import (
    TURN_SPLIT_LEAN,
    TURN_SPLIT_LEGACY,
    TurnConfig,
)

# ── Split constants ──────────────────────────────────────────────────────────


def test_split_sum_invariant():
    """Both legacy and lean splits must sum to exactly 1.0."""
    assert abs(sum(TURN_SPLIT_LEGACY) - 1.0) < 1e-9, (
        f"TURN_SPLIT_LEGACY sums to {sum(TURN_SPLIT_LEGACY)}, expected 1.0"
    )
    assert abs(sum(TURN_SPLIT_LEAN) - 1.0) < 1e-9, (
        f"TURN_SPLIT_LEAN sums to {sum(TURN_SPLIT_LEAN)}, expected 1.0"
    )


# ── TurnConfig.compose_lean flag ─────────────────────────────────────────────


def test_legacy_split_unchanged():
    """compose_lean=False (default) preserves the legacy 30/25/15/15/15 split."""
    cfg = TurnConfig(domain="magnetics", compose_lean=False)
    assert cfg.split == TURN_SPLIT_LEGACY
    assert cfg.split == (0.30, 0.25, 0.15, 0.15, 0.15)


def test_lean_split_review_60pct():
    """compose_lean=True activates the lean split where review phases get 60%."""
    cfg = TurnConfig(domain="magnetics", compose_lean=True)
    assert cfg.split == TURN_SPLIT_LEAN
    # review_names (index 2) + review_docs (index 3) must total 60%
    names_plus_docs = cfg.split[2] + cfg.split[3]
    assert abs(names_plus_docs - 0.60) < 1e-9, (
        f"review_names + review_docs = {names_plus_docs:.4f}, expected 0.60"
    )


def test_default_is_legacy():
    """TurnConfig default (no compose_lean kwarg) must use the legacy split."""
    cfg = TurnConfig(domain="equilibrium")
    assert cfg.split == TURN_SPLIT_LEGACY


# ── BudgetManager per-phase caps ─────────────────────────────────────────────


def test_phase_cap_enforces_at_1_5x():
    """BudgetManager with cap=1.0 accepts 1.4 but rejects 1.6 (limit = 1.5)."""
    # 1.4 ≤ 1.0 * 1.5 = 1.5 → should succeed
    mgr_ok = BudgetManager(total_budget=10.0, phase_caps={"compose": 1.0})
    lease = mgr_ok.reserve(1.4, phase="compose")
    assert lease is not None, "reservation of 1.4 should succeed (< cap*1.5=1.5)"
    lease.release_unused()

    # 1.6 > 1.0 * 1.5 = 1.5 → should be rejected
    mgr_bad = BudgetManager(total_budget=10.0, phase_caps={"compose": 1.0})
    lease2 = mgr_bad.reserve(1.6, phase="compose")
    assert lease2 is None, "reservation of 1.6 should be rejected (> cap*1.5=1.5)"


def test_phase_cap_per_phase():
    """Compose phase at-cap does not block an independent review reservation."""
    mgr = BudgetManager(
        total_budget=10.0,
        phase_caps={"compose": 1.0, "review_names": 3.0},
    )

    # Fill compose to its hard cap (1.0 * 1.5 = 1.5)
    lease_compose = mgr.reserve(1.5, phase="compose")
    assert lease_compose is not None, "initial compose reservation should succeed"

    # Compose is now at-cap — next compose reservation must be rejected
    lease_compose2 = mgr.reserve(0.1, phase="compose")
    assert lease_compose2 is None, "compose should be rejected once at-cap"

    # review_names has its own cap — must not be blocked by compose
    lease_review = mgr.reserve(1.0, phase="review_names")
    assert lease_review is not None, (
        "review_names reservation must succeed independently"
    )
