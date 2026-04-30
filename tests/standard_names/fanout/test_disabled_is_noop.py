"""Disabled = true no-op invariants (plan 39 §7.2 S5)."""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.fanout.config import FanoutSettings
from imas_codex.standard_names.fanout.dispatcher import run_fanout
from imas_codex.standard_names.fanout.schemas import CandidateContext, FanoutScope


class _MockGraphClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append({"cypher": cypher, "params": params})
        return [{"id": params.get("id")}]


def _new_lease():
    mgr = BudgetManager(total_budget=10.0)
    lease = mgr.reserve(1.0)
    assert lease is not None
    return lease


def _candidate() -> CandidateContext:
    return CandidateContext(
        sn_id="electron_temperature",
        name="electron_temperature",
        path="core_profiles/profiles_1d/electrons/temperature",
    )


class TestDisabledIsNoop:
    async def test_master_switch_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Make the LLM call fatal — should never be reached.
        async def _explode(**kwargs: Any):
            raise AssertionError("LLM must not be called when disabled")

        monkeypatch.setattr(
            "imas_codex.discovery.base.llm.acall_llm_structured", _explode
        )
        gc = _MockGraphClient()
        evidence = await run_fanout(
            site="refine_name",
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=FanoutScope(),
            gc=gc,
            parent_lease=_new_lease(),
            settings=FanoutSettings(enabled=False),
        )
        assert evidence == ""
        # No Fanout node written.
        assert gc.calls == []

    async def test_per_site_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _explode(**kwargs: Any):
            raise AssertionError("LLM must not be called when site disabled")

        monkeypatch.setattr(
            "imas_codex.discovery.base.llm.acall_llm_structured", _explode
        )
        gc = _MockGraphClient()
        evidence = await run_fanout(
            site="refine_name",
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=FanoutScope(),
            gc=gc,
            parent_lease=_new_lease(),
            settings=FanoutSettings(enabled=True, sites={"refine_name": False}),
        )
        assert evidence == ""
        # No Fanout node written when the site is off (true no-op).
        assert gc.calls == []

    async def test_unknown_site_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _explode(**kwargs: Any):
            raise AssertionError("LLM must not be called for unknown site")

        monkeypatch.setattr(
            "imas_codex.discovery.base.llm.acall_llm_structured", _explode
        )
        gc = _MockGraphClient()
        evidence = await run_fanout(
            site="some_future_site",
            candidate=_candidate(),
            reviewer_excerpt="unclear",
            scope=FanoutScope(),
            gc=gc,
            parent_lease=_new_lease(),
            settings=FanoutSettings(enabled=True, sites={"refine_name": True}),
        )
        assert evidence == ""
        assert gc.calls == []
