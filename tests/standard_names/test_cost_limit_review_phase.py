"""Tests proving --cost-limit gates review phase via shared BudgetManager.

Covers the W33A fix: before this change, each phase created its own
independent BudgetManager so review could claim a fresh $N budget even
after compose had already spent the user-specified cost_limit.  After the
fix, all phases draw from a single shared pool so total spend ≤ cost_limit.

All tests are mock-based — no Neo4j or real LLM required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.standard_names.budget import BudgetManager, LLMCostEvent
from imas_codex.standard_names.loop import run_sn_loop
from imas_codex.standard_names.turn import PhaseResult


def _ce(lease, amount, phase=None):
    """Simulate LLM spend (replaces charge_soft in tests)."""
    evt_phase = phase or lease.phase or "test"
    return lease.charge_event(
        amount,
        LLMCostEvent(model="test-model", tokens_in=0, tokens_out=0, phase=evt_phase),
    )


# ═══════════════════════════════════════════════════════════════════════
# Shared BudgetManager passed through TurnConfig
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_shared_budget_passed_to_turn_config():
    """run_sn_loop creates one BudgetManager and passes it to every TurnConfig."""
    captured_shared: list = []

    async def fake_turn(cfg):
        captured_shared.append(cfg.shared_budget)
        return [PhaseResult(name="generate", count=1, cost=1.0)]

    domain_calls = [{"domain": "equilibrium", "remaining": 5}, None]
    with (
        patch(
            "imas_codex.standard_names.loop.select_next_domain",
            side_effect=domain_calls,
        ),
        patch("imas_codex.standard_names.turn.run_turn", side_effect=fake_turn),
        patch("imas_codex.standard_names.graph_ops.finalize_sn_run"),
        patch("imas_codex.standard_names.graph_ops.create_sn_run_open"),
    ):
        await run_sn_loop(cost_limit=5.0, dry_run=False)

    assert len(captured_shared) == 1
    assert isinstance(captured_shared[0], BudgetManager)
    # The shared manager should reflect the same budget ceiling
    assert captured_shared[0].total_budget == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_same_shared_budget_across_multiple_turns():
    """All turns within one run receive the SAME BudgetManager instance."""
    captured_shared: list = []
    call_count = 0

    async def fake_turn(cfg):
        nonlocal call_count
        call_count += 1
        captured_shared.append(cfg.shared_budget)
        return [PhaseResult(name="generate", count=1, cost=0.5)]

    def rotating_selector(**_kw):
        calls = ["equilibrium", "magnetics", "transport", None]
        idx = len(captured_shared)
        if idx >= len(calls) or calls[idx] is None:
            return None
        return {"domain": calls[idx], "remaining": 5}

    with (
        patch(
            "imas_codex.standard_names.loop.select_next_domain",
            side_effect=rotating_selector,
        ),
        patch("imas_codex.standard_names.turn.run_turn", side_effect=fake_turn),
        patch("imas_codex.standard_names.graph_ops.finalize_sn_run"),
        patch("imas_codex.standard_names.graph_ops.create_sn_run_open"),
    ):
        await run_sn_loop(cost_limit=5.0, dry_run=False)

    assert len(captured_shared) == 3
    # Every turn got the exact same object
    assert captured_shared[0] is captured_shared[1]
    assert captured_shared[1] is captured_shared[2]


# ═══════════════════════════════════════════════════════════════════════
# Total spend bounded by cost_limit
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_total_spend_bounded_by_cost_limit():
    """cost_spent never exceeds cost_limit even when phases report high costs."""

    # Simulate a turn where compose + review each try to claim $4 from a $5 pool.
    async def greedy_turn(cfg):
        mgr: BudgetManager = cfg.shared_budget
        compose_cost = 4.0
        review_cost = 4.0
        if mgr is not None:
            g = mgr.reserve(compose_cost, phase="generate")
            if g:
                # charge_soft always records; in production LLM already paid
                actual = min(compose_cost, mgr.remaining + compose_cost)
                _ce(g, actual)
                g.__exit__(None, None, None)
            r = mgr.reserve(review_cost, phase="review_names")
            if r:
                g2 = min(review_cost, mgr.remaining + review_cost)
                _ce(r, g2)
                r.__exit__(None, None, None)
            # Review reservation returns None when pool exhausted — no charge
        return [
            PhaseResult(name="generate", count=3, cost=compose_cost),
            PhaseResult(name="review_names", count=0, cost=0.0),
        ]

    with (
        patch(
            "imas_codex.standard_names.loop.select_next_domain",
            side_effect=[{"domain": "equilibrium", "remaining": 10}, None],
        ),
        patch("imas_codex.standard_names.turn.run_turn", side_effect=greedy_turn),
        patch("imas_codex.standard_names.graph_ops.finalize_sn_run"),
        patch("imas_codex.standard_names.graph_ops.create_sn_run_open"),
    ):
        summary = await run_sn_loop(cost_limit=5.0, dry_run=False)

    # The shared BudgetManager pool was $5; review reservation should have been
    # blocked after compose claimed $4, leaving only $1 for review.
    assert summary.cost_spent <= 5.0 + 0.01  # allow floating-point epsilon


@pytest.mark.asyncio
async def test_review_blocked_when_compose_exhausts_pool():
    """When compose consumes the full budget, review reservation returns None."""
    reservation_outcomes: list[str] = []

    async def spend_all_turn(cfg):
        mgr: BudgetManager = cfg.shared_budget
        if mgr is not None:
            # Consume entire pool in compose
            g = mgr.reserve(5.0, phase="generate")
            if g:
                _ce(g, 5.0)
                g.__exit__(None, None, None)
            # Review reservation should now fail
            r = mgr.reserve(1.0, phase="review_names")
            reservation_outcomes.append("blocked" if r is None else "granted")
        return [
            PhaseResult(name="generate", count=5, cost=5.0),
            PhaseResult(name="review_names", count=0, cost=0.0),
        ]

    with (
        patch(
            "imas_codex.standard_names.loop.select_next_domain",
            side_effect=[{"domain": "equilibrium", "remaining": 10}, None],
        ),
        patch("imas_codex.standard_names.turn.run_turn", side_effect=spend_all_turn),
        patch("imas_codex.standard_names.graph_ops.finalize_sn_run"),
        patch("imas_codex.standard_names.graph_ops.create_sn_run_open"),
    ):
        await run_sn_loop(cost_limit=5.0, dry_run=False)

    assert reservation_outcomes == ["blocked"]


# ═══════════════════════════════════════════════════════════════════════
# L7 cost tracking
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_l7_cost_returned_from_revise_candidate():
    """_opus_revise_candidate returns (revised_name | None, cost_float)."""
    from imas_codex.standard_names.workers import _opus_revise_candidate

    candidate = {
        "id": "electron_temperature",
        "description": "Temperature of electrons",
        "reason": "low confidence",
    }

    # Mock acall_fn to return a successful revision with a known cost
    class FakeRevision:
        revised_name = "electron_kinetic_temperature"
        explanation = "more precise"

    async def fake_acall(*, model, messages, response_model, service):
        return FakeRevision(), 0.042, {"prompt_tokens": 100, "completion_tokens": 50}

    revised, cost, _ti, _to = await _opus_revise_candidate(
        candidate,
        domain_vocabulary="",
        reviewer_themes=[],
        acall_fn=fake_acall,
    )

    assert revised == "electron_kinetic_temperature"
    assert cost == pytest.approx(0.042)


@pytest.mark.asyncio
async def test_l7_cost_returned_even_when_revision_rejected():
    """Cost is always returned from _opus_revise_candidate."""
    from imas_codex.standard_names.workers import _opus_revise_candidate

    candidate = {
        "id": "electron_temperature",
        "description": "Temperature of electrons",
        "reason": "moderate",
    }

    class LowConfidenceRevision:
        revised_name = "electron_temperature_v2"
        explanation = "worse"

    async def fake_acall(*, model, messages, response_model, service):
        return LowConfidenceRevision(), 0.025, {}

    revised, cost, _ti, _to = await _opus_revise_candidate(
        candidate,
        domain_vocabulary="",
        reviewer_themes=[],
        acall_fn=fake_acall,
    )

    assert revised == "electron_temperature_v2"  # accepted (no confidence gate)
    assert cost == pytest.approx(0.025)


@pytest.mark.asyncio
async def test_l7_cost_returned_on_exception():
    """Even on LLM exception, cost is 0.0 (nothing charged)."""
    from imas_codex.standard_names.workers import _opus_revise_candidate

    candidate = {"id": "bad_name", "description": "", "reason": ""}

    async def broken_acall(**_kwargs):
        raise RuntimeError("LLM timeout")

    revised, cost, _ti, _to = await _opus_revise_candidate(
        candidate,
        domain_vocabulary="",
        reviewer_themes=[],
        acall_fn=broken_acall,
    )

    assert revised is None
    assert cost == 0.0
