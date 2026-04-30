"""Dispatcher tests — propose / execute / run_fanout (plan 39 §4, §12.2)."""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.fanout import dispatcher
from imas_codex.standard_names.fanout.config import FanoutSettings
from imas_codex.standard_names.fanout.schemas import (
    CandidateContext,
    FanoutHit,
    FanoutPlan,
    FanoutResult,
    FanoutScope,
    _SearchDDClusters,
    _SearchDDPaths,
    _SearchExistingNames,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


class _MockGraphClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append({"cypher": cypher, "params": params})
        return [{"id": params.get("id")}]


def _settings(**overrides: Any) -> FanoutSettings:
    base: dict[str, Any] = {
        "enabled": True,
        "sites": {"refine_name": True},
        "function_timeout_s": 2.0,
        "total_timeout_s": 4.0,
    }
    base.update(overrides)
    return FanoutSettings(**base)


def _candidate() -> CandidateContext:
    return CandidateContext(
        sn_id="electron_temperature",
        name="electron_temperature",
        path="core_profiles/profiles_1d/electrons/temperature",
        description="T_e",
        physics_domain="kinetics",
    )


def _scope() -> FanoutScope:
    return FanoutScope()


def _new_lease():
    mgr = BudgetManager(total_budget=10.0)  # run_id=None -> no graph writes
    lease = mgr.reserve(1.0)
    assert lease is not None
    return lease


class _FakeLLMResult:
    def __init__(self, parsed: FanoutPlan, cost: float = 0.0) -> None:
        self.parsed = parsed
        self.cost = cost
        self.input_tokens = 100
        self.output_tokens = 50


def _patch_llm(monkeypatch: pytest.MonkeyPatch, ret: Any) -> list[dict[str, Any]]:
    """Patch ``acall_llm_structured`` and return a list of recorded calls."""
    recorded: list[dict[str, Any]] = []

    async def _fake(
        *,
        model: str,
        messages: list[dict[str, str]],
        response_model: type,
        temperature: float = 0.0,
        service: str = "",
        **kwargs: Any,
    ):
        recorded.append(
            {
                "model": model,
                "messages": messages,
                "response_model": response_model,
                "temperature": temperature,
                "service": service,
            }
        )
        if isinstance(ret, BaseException):
            raise ret
        return ret

    monkeypatch.setattr("imas_codex.discovery.base.llm.acall_llm_structured", _fake)
    return recorded


# ---------------------------------------------------------------------
# Outcome classifier
# ---------------------------------------------------------------------


class TestClassifyOutcome:
    def test_ok(self) -> None:
        rs = [
            FanoutResult(
                fn_id="x", ok=True, hits=[FanoutHit(kind="cluster", id="a", label="a")]
            )
        ]
        assert dispatcher._classify_outcome(rs) == "ok"

    def test_executor_all_empty(self) -> None:
        rs = [FanoutResult(fn_id="x", ok=True, hits=[])]
        assert dispatcher._classify_outcome(rs) == "executor_all_empty"

    def test_partial_fail(self) -> None:
        rs = [
            FanoutResult(
                fn_id="x", ok=True, hits=[FanoutHit(kind="cluster", id="a", label="a")]
            ),
            FanoutResult(fn_id="y", ok=False, error="timeout"),
        ]
        assert dispatcher._classify_outcome(rs) == "executor_partial_fail"

    def test_all_failed(self) -> None:
        rs = [
            FanoutResult(fn_id="x", ok=False, error="timeout"),
            FanoutResult(fn_id="y", ok=False, error="boom"),
        ]
        assert dispatcher._classify_outcome(rs) == "executor_partial_fail"


# ---------------------------------------------------------------------
# propose() — Stage A
# ---------------------------------------------------------------------


class TestPropose:
    async def test_planner_schema_fail_returns_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_llm(monkeypatch, ValueError("schema parse failed"))
        plan, outcome = await dispatcher.propose(
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=_scope(),
            settings=_settings(),
            parent_lease=_new_lease(),
            fanout_run_id="run-1",
        )
        assert plan is None
        assert outcome == "planner_schema_fail"

    async def test_planner_all_invalid_when_empty_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty = FanoutPlan(queries=[])
        _patch_llm(monkeypatch, _FakeLLMResult(empty))
        plan, outcome = await dispatcher.propose(
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=_scope(),
            settings=_settings(),
            parent_lease=_new_lease(),
            fanout_run_id="run-1",
        )
        assert plan is None
        assert outcome == "planner_all_invalid"

    async def test_dedup_collapses_duplicate_queries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two queries that normalise to the same key.
        plan = FanoutPlan(
            queries=[
                _SearchExistingNames(fn_id="search_existing_names", query="T_e"),
                _SearchExistingNames(fn_id="search_existing_names", query="  t_e  "),
                _SearchDDPaths(fn_id="search_dd_paths", query="T_e"),
            ]
        )
        _patch_llm(monkeypatch, _FakeLLMResult(plan))
        out_plan, outcome = await dispatcher.propose(
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=_scope(),
            settings=_settings(),
            parent_lease=_new_lease(),
            fanout_run_id="run-1",
        )
        assert outcome is None
        assert out_plan is not None
        # Duplicate search_existing_names dropped; search_dd_paths kept.
        fn_ids = [q.fn_id for q in out_plan.queries]
        assert fn_ids == ["search_existing_names", "search_dd_paths"]

    async def test_proposer_cost_charged_to_parent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = FanoutPlan(
            queries=[_SearchExistingNames(fn_id="search_existing_names", query="T_e")]
        )
        _patch_llm(monkeypatch, _FakeLLMResult(plan, cost=0.0042))
        lease = _new_lease()
        before = lease.charged
        out_plan, outcome = await dispatcher.propose(
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=_scope(),
            settings=_settings(),
            parent_lease=lease,
            fanout_run_id="run-XYZ",
        )
        assert outcome is None
        assert out_plan is not None
        assert lease.charged == pytest.approx(before + 0.0042)

    async def test_scope_not_in_llm_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Caller-injected scope must never appear in LLM messages (plan §3.3)."""
        plan = FanoutPlan(
            queries=[_SearchExistingNames(fn_id="search_existing_names", query="T_e")]
        )
        recorded = _patch_llm(monkeypatch, _FakeLLMResult(plan))
        scope = FanoutScope(
            ids_filter="core_profiles", physics_domain="kinetics", dd_version=4
        )
        await dispatcher.propose(
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=scope,
            settings=_settings(),
            parent_lease=_new_lease(),
            fanout_run_id="run-1",
        )
        # scope appears in user-prompt context (informational), but NOT
        # in any structured arg list.  The point of the assertion is
        # that the schema sent to the LLM does not have ids_filter /
        # physics_domain / dd_version fields the LLM can drive.
        assert recorded[0]["response_model"] is FanoutPlan
        # FanoutPlan.queries' variants must not declare scope fields.
        from imas_codex.standard_names.fanout.schemas import (
            _FindRelatedDDPaths as _FRDD,
            _SearchDDClusters as _SDDC,
            _SearchDDPaths as _SDDP,
            _SearchExistingNames as _SEN,
        )

        for cls in (_SEN, _SDDP, _FRDD, _SDDC):
            for forbidden in ("ids_filter", "physics_domain", "dd_version"):
                assert forbidden not in cls.model_fields


# ---------------------------------------------------------------------
# run_fanout() — top-level outcomes
# ---------------------------------------------------------------------


class TestRunFanout:
    async def test_off_arm_writes_node_and_returns_empty(self) -> None:
        gc = _MockGraphClient()
        evidence = await dispatcher.run_fanout(
            site="refine_name",
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=_scope(),
            gc=gc,
            parent_lease=_new_lease(),
            settings=_settings(),
            arm="off",
            fanout_run_id="run-off",
        )
        assert evidence == ""
        assert len(gc.calls) == 1
        assert gc.calls[0]["params"]["outcome"] == "off_arm"
        assert gc.calls[0]["params"]["arm"] == "off"

    async def test_no_budget_writes_node(self) -> None:
        gc = _MockGraphClient()
        s = _settings(
            fanout_max_charge_per_cycle_baseline=0.0,
            fanout_max_charge_per_cycle_escalation=0.0,
        )
        evidence = await dispatcher.run_fanout(
            site="refine_name",
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=_scope(),
            gc=gc,
            parent_lease=_new_lease(),
            settings=s,
            fanout_run_id="run-nb",
        )
        assert evidence == ""
        assert gc.calls[0]["params"]["outcome"] == "no_budget"

    async def test_planner_schema_fail_writes_node_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gc = _MockGraphClient()
        _patch_llm(monkeypatch, ValueError("bad json"))
        evidence = await dispatcher.run_fanout(
            site="refine_name",
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=_scope(),
            gc=gc,
            parent_lease=_new_lease(),
            settings=_settings(),
            fanout_run_id="run-schema-fail",
        )
        assert evidence == ""
        assert gc.calls[0]["params"]["outcome"] == "planner_schema_fail"

    async def test_executor_all_empty_writes_node_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gc = _MockGraphClient()
        plan = FanoutPlan(
            queries=[_SearchExistingNames(fn_id="search_existing_names", query="T_e")]
        )
        _patch_llm(monkeypatch, _FakeLLMResult(plan))
        # Stub the runner to return ok=True with empty hits.
        monkeypatch.setattr(
            "imas_codex.standard_names.search.search_standard_names_vector",
            lambda *a, **kw: [],
        )
        evidence = await dispatcher.run_fanout(
            site="refine_name",
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=_scope(),
            gc=gc,
            parent_lease=_new_lease(),
            settings=_settings(),
            fanout_run_id="run-empty",
        )
        assert evidence == ""
        # First call is the Fanout node write.
        outcomes = [c["params"]["outcome"] for c in gc.calls]
        assert "executor_all_empty" in outcomes

    async def test_ok_path_returns_evidence_and_writes_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gc = _MockGraphClient()
        plan = FanoutPlan(
            queries=[_SearchExistingNames(fn_id="search_existing_names", query="T_e")]
        )
        _patch_llm(monkeypatch, _FakeLLMResult(plan))
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
        evidence = await dispatcher.run_fanout(
            site="refine_name",
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=_scope(),
            gc=gc,
            parent_lease=_new_lease(),
            settings=_settings(),
            fanout_run_id="run-ok",
        )
        assert evidence != ""
        assert "electron_temperature" in evidence
        outcomes = [c["params"]["outcome"] for c in gc.calls]
        assert "ok" in outcomes
