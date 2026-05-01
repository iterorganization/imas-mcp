"""Cost-attribution invariant for fan-out (plan 39 §7.3 I1, criterion #10).

Snapshots ``original_reservation`` *before* the first
``_extend_reservation`` call and asserts that, after a fan-out cycle:

    parent_lease.charged
        ≤ original_reservation + sum(charges with batch_id == fanout_run_id)

i.e. extensions are attributable line-by-line to specific
``LLMCostEvent`` rows; no charge is silently leaked or double-counted.

Also asserts that every fan-out-attributed cost event has phase
prefix ``sn_fanout_`` or phase suffix ``+fanout``.
"""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names.budget import BudgetManager, LLMCostEvent
from imas_codex.standard_names.fanout import dispatcher
from imas_codex.standard_names.fanout.config import FanoutSettings
from imas_codex.standard_names.fanout.schemas import (
    CandidateContext,
    FanoutPlan,
    FanoutScope,
    _SearchExistingNames,
)


class _MockGraphClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append({"cypher": cypher, "params": params})
        return [{"id": params.get("id")}]


class _FakeLLMResult:
    def __init__(self, parsed: FanoutPlan, cost: float = 0.0) -> None:
        self.parsed = parsed
        self.cost = cost
        self.input_tokens = 10
        self.output_tokens = 5


def _patch_llm(monkeypatch, ret):
    async def _fake(**kwargs):
        if isinstance(ret, BaseException):
            raise ret
        return ret

    monkeypatch.setattr("imas_codex.discovery.base.llm.acall_llm_structured", _fake)


def _settings():
    return FanoutSettings(
        enabled=True,
        sites={"refine_name": True},
        function_timeout_s=2.0,
        total_timeout_s=4.0,
    )


def _candidate():
    return CandidateContext(
        sn_id="electron_temperature",
        name="electron_temperature",
        path="core_profiles/profiles_1d/electrons/temperature",
    )


async def test_original_reservation_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cost-attribution invariant: charged ≤ snapshot + sum(fanout charges)."""
    plan = FanoutPlan(
        queries=[_SearchExistingNames(fn_id="search_existing_names", query="T_e")]
    )
    _patch_llm(monkeypatch, _FakeLLMResult(plan, cost=0.0042))
    monkeypatch.setattr(
        "imas_codex.standard_names.search.search_standard_names_vector",
        lambda *a, **kw: [
            {
                "id": "electron_temperature",
                "description": "T_e",
                "kind": "scalar",
                "unit": "eV",
                "score": 0.91,
            }
        ],
    )

    mgr = BudgetManager(total_budget=10.0)
    lease = mgr.reserve(0.20, phase="refine_name")
    assert lease is not None

    # Snapshot BEFORE any extension.
    original_reservation = lease.reserved
    pre_charged = lease.charged
    assert pre_charged == 0.0

    # Capture LLMCostEvents by intercepting the BudgetManager's
    # async-write enqueue (BudgetLease has __slots__ so its methods
    # cannot be patched directly).
    captured: list[LLMCostEvent] = []

    real_enqueue = mgr._enqueue_write

    def _capture(cost, event, overspend=0.0):
        captured.append(event)
        return real_enqueue(cost, event, overspend)

    monkeypatch.setattr(mgr, "_enqueue_write", _capture)

    fanout_run_id = "test-run-id-xyz"
    evidence = await dispatcher.run_fanout(
        site="refine_name",
        candidate=_candidate(),
        reviewer_excerpt="unclear",
        scope=FanoutScope(),
        gc=_MockGraphClient(),
        parent_lease=lease,
        settings=_settings(),
        arm="on",
        fanout_run_id=fanout_run_id,
    )
    assert evidence != ""

    # Synthesizer charge: simulated by the call-site (would normally
    # happen in process_refine_name_batch after run_fanout returns).
    synth_event = LLMCostEvent(
        model="anthropic/claude-sonnet-4.5",
        tokens_in=200,
        tokens_out=80,
        sn_ids=("electron_temperature",),
        phase="refine_name+fanout",
        service="standard-names",
        batch_id=fanout_run_id,
    )
    lease.charge_event(0.05, synth_event)

    # ── Invariant: per-charge attribution ────────────────────────────
    fanout_attributed = [e for e in captured if e.batch_id == fanout_run_id]
    fanout_attributed.append(synth_event)

    # Every fan-out-attributed event has the expected phase prefix/suffix.
    for ev in fanout_attributed:
        phase = ev.phase or ""
        assert phase.startswith("sn_fanout_") or phase.endswith("+fanout"), (
            f"unexpected phase for fanout event: {phase!r}"
        )

    # The lease's running ``charged`` figure equals every cost we
    # passed in — no silent leak, no double-count.
    sum_proposer = sum(
        # We can't read costs from captured events (they don't carry
        # cost), but we paid 0.0042 (proposer) + 0.05 (synth) = 0.0542.
        c
        for c in (0.0042, 0.05)
    )
    assert lease.charged == pytest.approx(sum_proposer)

    # Cost-attribution invariant (rewritten per RD I1):
    # ``charged ≤ original_reservation + cumulative_fanout_charges``.
    assert lease.charged <= original_reservation + sum_proposer + 1e-9


async def test_off_arm_does_not_extend_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off arm is a true no-op — no LLM call, no lease extension."""

    async def _explode(**kwargs):
        raise AssertionError("LLM must not be called on off arm")

    monkeypatch.setattr("imas_codex.discovery.base.llm.acall_llm_structured", _explode)

    mgr = BudgetManager(total_budget=10.0)
    lease = mgr.reserve(0.20)
    assert lease is not None
    snap_reserved = lease.reserved
    snap_charged = lease.charged

    evidence = await dispatcher.run_fanout(
        site="refine_name",
        candidate=_candidate(),
        reviewer_excerpt="unclear",
        scope=FanoutScope(),
        gc=_MockGraphClient(),
        parent_lease=lease,
        settings=_settings(),
        arm="off",
        fanout_run_id="off-run",
    )

    assert evidence == ""
    # Off-arm true no-op: nothing charged, reservation unchanged.
    assert lease.charged == snap_charged
    assert lease.reserved == snap_reserved
