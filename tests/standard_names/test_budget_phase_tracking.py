"""Tests for per-phase spend tracking in BudgetManager.

Covers:
- phase_spent property populated by _record_spend
- phase tag stored per lease and released on _release
- summary dict includes phase_spent key
- concurrent leases from different phases stay isolated
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from imas_codex.standard_names.budget import BudgetManager, LLMCostEvent

# ── Test helper ───────────────────────────────────────────────────────


def _ce(lease, amount, phase=None):
    """Simulate LLM spend via charge_event (legacy charge_soft replacement)."""
    evt_phase = phase or lease.phase or "test"
    return lease.charge_event(
        amount,
        LLMCostEvent(model="test-model", tokens_in=0, tokens_out=0, phase=evt_phase),
    )


class TestPhaseSpentTracking:
    """Unit tests for the phase_spent property on BudgetManager."""

    def test_phase_spent_empty_at_start(self):
        mgr = BudgetManager(10.0)
        assert mgr.phase_spent == {}

    def test_phase_spent_populated_on_charge(self):
        mgr = BudgetManager(10.0)
        lease = mgr.reserve(2.0, phase="generate")
        assert lease is not None
        _ce(lease, 1.5)
        lease.__exit__(None, None, None)
        assert mgr.phase_spent.get("generate", 0.0) == pytest.approx(1.5)

    def test_phase_spent_accumulates_across_batches(self):
        mgr = BudgetManager(10.0)
        for _ in range(3):
            lease = mgr.reserve(1.0, phase="review_names")
            assert lease is not None
            _ce(lease, 0.4)
            lease.__exit__(None, None, None)
        assert mgr.phase_spent.get("review_names", 0.0) == pytest.approx(1.2)

    def test_multiple_phases_tracked_independently(self):
        mgr = BudgetManager(10.0)

        g = mgr.reserve(3.0, phase="generate")
        assert g is not None
        _ce(g, 2.0)
        g.__exit__(None, None, None)

        r = mgr.reserve(3.0, phase="review_names")
        assert r is not None
        _ce(r, 1.5)
        r.__exit__(None, None, None)

        spent = mgr.phase_spent
        assert spent["generate"] == pytest.approx(2.0)
        assert spent["review_names"] == pytest.approx(1.5)

    def test_summary_includes_phase_spent(self):
        mgr = BudgetManager(5.0)
        lease = mgr.reserve(1.0, phase="regen")
        assert lease is not None
        _ce(lease, 0.7)
        lease.__exit__(None, None, None)
        s = mgr.summary
        assert "phase_spent" in s
        assert s["phase_spent"].get("regen", 0.0) == pytest.approx(0.7)

    def test_lease_phases_cleaned_up_on_release(self):
        """After release, the lease_id should not remain in _lease_phases."""
        mgr = BudgetManager(5.0)
        lease = mgr.reserve(1.0, phase="generate")
        assert lease is not None
        lease_id = lease._lease_id
        lease.__exit__(None, None, None)
        # Internal cleanup: no key left for this lease
        with mgr._lock:
            assert lease_id not in mgr._lease_phases

    def test_phase_spent_thread_safe(self):
        """Concurrent charges from multiple threads must not corrupt totals."""
        mgr = BudgetManager(100.0)
        errors: list[Exception] = []

        def worker(phase: str, amount: float, n: int) -> None:
            try:
                for _ in range(n):
                    lease = mgr.reserve(amount, phase=phase)
                    if lease is None:
                        return
                    _ce(lease, amount * 0.5)
                    lease.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=("generate", 0.1, 10)),
            threading.Thread(target=worker, args=("review_names", 0.1, 10)),
            threading.Thread(target=worker, args=("review_docs", 0.1, 10)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        spent = mgr.phase_spent
        # Each thread charged 10 × 0.05 = 0.50
        for phase in ("generate", "review_names", "review_docs"):
            assert spent[phase] == pytest.approx(0.50, abs=1e-9)


class TestPhaseSpentInLoop:
    """Integration-level: shared BudgetManager phase accounting flows through loop."""

    @pytest.mark.asyncio
    async def test_shared_budget_phase_split_in_summary(self):
        """summary.compose_cost / review_cost reflect phase_spent from shared mgr."""
        from unittest.mock import AsyncMock

        from imas_codex.standard_names.budget import BudgetManager
        from imas_codex.standard_names.loop import run_sn_loop
        from imas_codex.standard_names.turn import PhaseResult

        # We inject a custom side_effect that records actual spend on the
        # shared BudgetManager that the loop creates internally.
        captured_mgr: list[BudgetManager] = []

        async def fake_turn(cfg):
            # Capture the shared manager so we can charge it.
            mgr: BudgetManager = cfg.shared_budget
            if mgr is not None:
                captured_mgr.append(mgr)
                # Simulate generate spend
                g = mgr.reserve(0.5, phase="generate")
                if g:
                    _ce(g, 0.3)
                    g.__exit__(None, None, None)
                # Simulate review_names spend
                r = mgr.reserve(0.5, phase="review_names")
                if r:
                    _ce(r, 0.2)
                    r.__exit__(None, None, None)
            return [
                PhaseResult(name="generate", count=2, cost=0.3),
                PhaseResult(name="review_names", count=2, cost=0.2),
            ]

        with (
            patch(
                "imas_codex.standard_names.loop.select_next_domain",
                side_effect=[{"domain": "equilibrium", "remaining": 5}, None],
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                side_effect=fake_turn,
            ),
            patch("imas_codex.standard_names.graph_ops.finalize_sn_run"),
            patch("imas_codex.standard_names.graph_ops.create_sn_run_open"),
        ):
            summary = await run_sn_loop(cost_limit=5.0, dry_run=False)

        assert summary.compose_cost == pytest.approx(0.3)
        assert summary.review_cost == pytest.approx(0.2)
        assert summary.cost_spent == pytest.approx(0.5)
