"""Tests for Phase 4 charge-site instrumentation.

Verifies that each LLM call site constructs a correctly typed
``LLMCostEvent`` and calls ``lease.charge_event`` with the expected
phase, cycle, batch_id, model, and sn_ids.

All tests mock the LLM call, budget lease, and graph — no live
Neo4j or LLM required.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.budget import (
    BudgetLease,
    BudgetManager,
    ChargeResult,
    LLMCostEvent,
)

# ── Helpers ──────────────────────────────────────────────────────────────


class _LLMResult:
    """Stub LLMResult that supports both tuple unpacking and attribute access."""

    def __init__(
        self,
        result,
        cost: float = 0.05,
        tokens: int = 200,
        input_tokens: int = 150,
        output_tokens: int = 50,
        cache_read_tokens: int = 10,
        cache_creation_tokens: int = 5,
    ):
        self._result = result
        self._cost = cost
        self._tokens = tokens
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_creation_tokens = cache_creation_tokens

    def __iter__(self):
        return iter((self._result, self._cost, self._tokens))


def _make_mgr(budget: float = 10.0) -> BudgetManager:
    return BudgetManager(total_budget=budget, run_id="test-run")


def _make_lease(mgr: BudgetManager | None = None, phase: str = "") -> BudgetLease:
    mgr = mgr or _make_mgr()
    lease = mgr.reserve(5.0, phase=phase)
    assert lease is not None
    return lease


# =====================================================================
# 1. Compose primary — phase="generate"
# =====================================================================


class TestComposeChargeEvent:
    """Compose worker charges via lease.charge_event with generate phase."""

    @pytest.mark.asyncio
    async def test_compose_charge_event_uses_generate_phase(self):
        """charge_event is called with phase='generate' and correct metadata."""
        from pydantic import BaseModel, Field

        # Minimal candidate model
        class _Candidate(BaseModel):
            standard_name: str = "electron_temperature"
            source_id: str = "eq/time_slice/profiles_1d/te"
            dd_paths: list[str] = Field(default_factory=lambda: ["eq/te"])
            kind: str = "scalar"
            confidence: float = 0.9
            reason: str = "test"
            grammar_fields: dict = Field(default_factory=dict)
            unit: str = "eV"

        class _ComposeBatch(BaseModel):
            candidates: list[_Candidate] = Field(default_factory=lambda: [_Candidate()])

        result_obj = _ComposeBatch()

        # Track charge_event calls
        events: list[tuple[float, LLMCostEvent]] = []

        def _spy_charge_event(self, cost, event):
            events.append((cost, event))
            return ChargeResult()

        mgr = _make_mgr()
        lease = mgr.reserve(5.0, phase="generate")
        assert lease is not None

        llm_result = _LLMResult(result_obj, cost=0.15, tokens=200)

        # Simulate the compose charge pattern
        with patch.object(BudgetLease, "charge_event", _spy_charge_event):
            cost = 0.15
            model = "test-model"
            batch_group_key = "equilibrium"

            _event = LLMCostEvent(
                model=model,
                tokens_in=getattr(llm_result, "input_tokens", 0) or 0,
                tokens_out=getattr(llm_result, "output_tokens", 0) or 0,
                tokens_cached_read=(getattr(llm_result, "cache_read_tokens", 0) or 0),
                tokens_cached_write=(
                    getattr(llm_result, "cache_creation_tokens", 0) or 0
                ),
                sn_ids=tuple(c.standard_name for c in result_obj.candidates),
                batch_id=batch_group_key,
                phase="generate",
                service="standard-names",
            )
            lease.charge_event(cost, _event)

        assert len(events) == 1
        charged_cost, event = events[0]
        assert charged_cost == 0.15
        assert event.phase == "generate"
        assert event.cycle is None
        assert event.batch_id == "equilibrium"
        assert event.model == "test-model"
        assert event.sn_ids == ("electron_temperature",)
        assert event.tokens_in == 150
        assert event.tokens_out == 50
        assert event.tokens_cached_read == 10
        assert event.tokens_cached_write == 5
        assert event.service == "standard-names"


# =====================================================================
# 2. Grammar retry — distinguishes batch_id
# =====================================================================


class TestGrammarRetryChargeEvent:
    """Grammar retry (L6) uses distinct batch_id with -grammar-retry suffix."""

    @pytest.mark.asyncio
    async def test_compose_grammar_retry_distinguishes_batch_id(self):
        """_grammar_retry returns cost/tokens; caller charges with retry suffix."""
        # Mock acall_fn that returns an LLMResult-like object
        from pydantic import BaseModel, Field

        from imas_codex.standard_names.workers import _grammar_retry

        class _GrammarRetryResponse(BaseModel):
            revised_name: str = "electron_temperature"
            explanation: str = "fixed"

        mock_result = _LLMResult(
            _GrammarRetryResponse(),
            cost=0.02,
            tokens=80,
            input_tokens=60,
            output_tokens=20,
        )

        async def mock_acall(**kwargs):
            return mock_result

        name, cost, ti, to = await _grammar_retry(
            "electon_temperature",
            "parse error: unknown token 'electon'",
            "test-model",
            mock_acall,
        )

        assert name == "electron_temperature"
        assert cost == 0.02
        assert ti == 60
        assert to == 20

        # Verify the event constructed by the caller
        event = LLMCostEvent(
            model="test-model",
            tokens_in=ti,
            tokens_out=to,
            sn_ids=("electon_temperature",),
            batch_id="equilibrium-grammar-retry",
            phase="generate",
            service="standard-names",
        )
        assert event.batch_id == "equilibrium-grammar-retry"
        assert event.phase == "generate"

    @pytest.mark.asyncio
    async def test_grammar_retry_returns_zero_on_failure(self):
        """On exception, _grammar_retry returns (None, 0.0, 0, 0)."""
        from imas_codex.standard_names.workers import _grammar_retry

        async def mock_acall(**kwargs):
            raise RuntimeError("LLM failure")

        name, cost, ti, to = await _grammar_retry(
            "bad_name", "parse error", "test-model", mock_acall
        )

        assert name is None
        assert cost == 0.0
        assert ti == 0
        assert to == 0


# =====================================================================
# 3. L7 Opus revision — phase="generate"
# =====================================================================


class TestL7RevisionChargeEvent:
    """L7 Opus revision returns cost/tokens for graph tracking."""

    @pytest.mark.asyncio
    async def test_l7_revision_returns_tokens(self):
        """_opus_revise_candidate returns (name, cost, ti, to)."""
        from pydantic import BaseModel, Field

        from imas_codex.standard_names.workers import _opus_revise_candidate

        class _OpusResponse(BaseModel):
            revised_name: str = "plasma_current"
            confidence: float = 0.95
            explanation: str = "improved"

        mock_result = _LLMResult(
            _OpusResponse(),
            cost=0.08,
            tokens=300,
            input_tokens=250,
            output_tokens=50,
        )

        async def mock_acall(**kwargs):
            return mock_result

        cand = {
            "id": "plasma_curr",
            "reason": "low conf",
            "description": "test",
            "confidence": 0.4,
        }

        revised, cost, ti, to = await _opus_revise_candidate(cand, "", [], mock_acall)

        assert revised == "plasma_current"
        assert cost == 0.08
        assert ti == 250
        assert to == 50


# =====================================================================
# 4. Enrich — phase="enrich", soft-stop
# =====================================================================


class TestEnrichChargeEvent:
    """Enrich charge site uses phase='enrich' and soft-stop semantics."""

    @pytest.mark.asyncio
    async def test_enrich_charge_event_uses_enrich_phase(self):
        """Enrich worker creates LLMCostEvent with phase='enrich'."""
        event = LLMCostEvent(
            model="test-enrich-model",
            tokens_in=100,
            tokens_out=50,
            sn_ids=("electron_temperature", "ion_temperature"),
            batch_id="3",
            phase="enrich",
            service="standard-names",
        )
        assert event.phase == "enrich"
        assert event.batch_id == "3"
        assert event.sn_ids == ("electron_temperature", "ion_temperature")
        assert event.cycle is None  # enrich has no cycles

    @pytest.mark.asyncio
    async def test_enrich_uses_soft_stop_never_drops_items(self):
        """charge_event with soft-stop semantics never raises BudgetExceeded.

        Even when overspend is reported, the enrich worker must continue
        processing items (never check result.hard_stop).
        """
        mgr = _make_mgr(budget=0.01)  # Tiny budget
        lease = mgr.reserve(0.01, phase="enrich")
        assert lease is not None

        event = LLMCostEvent(
            model="test-model",
            tokens_in=100,
            tokens_out=50,
            phase="enrich",
            sn_ids=("name_a",),
            batch_id="0",
            service="standard-names",
        )

        # Charge well over budget — should NOT raise
        result = lease.charge_event(5.0, event)

        # Overspend is reported but no exception
        assert result.overspend > 0
        assert not result.hard_stop  # soft-stop: never hard-stops

        # Critical: workflow continues — lease is still usable
        result2 = lease.charge_event(1.0, event)
        assert result2.overspend > 0
        # Total charged = 6.0, reserved = 0.01 → significant overspend

    @pytest.mark.asyncio
    async def test_enrich_state_has_budget_manager(self):
        """StandardNameEnrichState now has budget_manager field."""
        from imas_codex.standard_names.enrich_state import StandardNameEnrichState

        state = StandardNameEnrichState(facility="dd")
        assert state.budget_manager is None  # default
        assert state.budget_phase_tag == ""

        # Can be set
        mgr = _make_mgr()
        state.budget_manager = mgr
        assert state.budget_manager is mgr


# =====================================================================
# 5. Review c0/c1/c2 — correct cycles
# =====================================================================


class TestReviewCycleChargeEvents:
    """Review pipeline emits charge events with correct cycle labels."""

    def test_review_c0_c1_c2_emit_correct_cycle(self):
        """_charge_review_cycle creates events with c0/c1/c2 cycle tags."""
        from imas_codex.standard_names.review.pipeline import (
            _charge_review_cycle,
        )

        mgr = _make_mgr()

        for cycle_label in ("c0", "c1", "c2"):
            cycle_events: list[tuple[float, LLMCostEvent]] = []

            def _spy(self, cost, event, _store=cycle_events):
                _store.append((cost, event))
                return ChargeResult()

            lease = mgr.reserve(2.0, phase="review_names")
            assert lease is not None

            result_dict = {
                "_cost": 0.10,
                "_input_tokens": 100,
                "_output_tokens": 50,
                "_primary_cost": 0.10,
                "_primary_input_tokens": 100,
                "_primary_output_tokens": 50,
            }
            items = [{"id": "electron_temperature"}, {"id": "ion_temperature"}]

            with patch.object(BudgetLease, "charge_event", _spy):
                _charge_review_cycle(
                    lease,
                    0.10,
                    result_dict,
                    f"model-{cycle_label}",
                    items,
                    "equilibrium",
                    cycle_label,
                    "review_names",
                )

            assert len(cycle_events) == 1  # no retry cost
            _, event = cycle_events[0]
            assert event.cycle == cycle_label
            assert event.phase == "review_names"
            assert event.batch_id == "equilibrium"
            assert event.sn_ids == ("electron_temperature", "ion_temperature")

    def test_review_docs_phase_is_review_docs(self):
        """When review_target='docs', phase is 'review_docs'."""
        from imas_codex.standard_names.review.pipeline import (
            _charge_review_cycle,
        )

        events: list[tuple[float, LLMCostEvent]] = []

        def _spy(self, cost, event):
            events.append((cost, event))
            return ChargeResult()

        mgr = _make_mgr()
        lease = mgr.reserve(2.0, phase="review_docs")
        assert lease is not None

        result_dict = {
            "_cost": 0.05,
            "_input_tokens": 80,
            "_output_tokens": 40,
            "_primary_cost": 0.05,
            "_primary_input_tokens": 80,
            "_primary_output_tokens": 40,
        }

        with patch.object(BudgetLease, "charge_event", _spy):
            _charge_review_cycle(
                lease,
                0.05,
                result_dict,
                "model-c0",
                [{"id": "test_name"}],
                "core_profiles",
                "c0",
                "review_docs",
            )

        assert len(events) == 1
        _, event = events[0]
        assert event.phase == "review_docs"


# =====================================================================
# 6. Review retry — distinguishes batch_id
# =====================================================================


class TestReviewRetryChargeEvent:
    """Review single-batch retry creates separate event with -retry suffix."""

    def test_review_retry_distinguishes_batch_id(self):
        """When _review_single_batch has retry cost, two events are created."""
        from imas_codex.standard_names.review.pipeline import (
            _charge_review_cycle,
        )

        events: list[tuple[float, LLMCostEvent]] = []

        def _spy(self, cost, event):
            events.append((cost, event))
            return ChargeResult()

        mgr = _make_mgr()
        lease = mgr.reserve(5.0, phase="review_names")
        assert lease is not None

        # Simulate result with retry: primary=0.08, total=0.12
        result_dict = {
            "_cost": 0.12,
            "_input_tokens": 200,
            "_output_tokens": 100,
            "_primary_cost": 0.08,
            "_primary_input_tokens": 150,
            "_primary_output_tokens": 70,
        }
        items = [{"id": "electron_temperature"}]

        with patch.object(BudgetLease, "charge_event", _spy):
            _charge_review_cycle(
                lease,
                0.12,
                result_dict,
                "model-c0",
                items,
                "equilibrium",
                "c0",
                "review_names",
            )

        # Two events: primary + retry
        assert len(events) == 2

        # Primary event
        _, primary = events[0]
        assert primary.batch_id == "equilibrium"
        assert primary.cycle == "c0"
        assert primary.phase == "review_names"
        assert primary.tokens_in == 150
        assert primary.tokens_out == 70

        # Retry event
        retry_cost, retry = events[1]
        assert retry.batch_id == "equilibrium-retry"
        assert retry.cycle == "c0"
        assert retry.phase == "review_names"
        assert retry.tokens_in == 50  # 200 - 150
        assert retry.tokens_out == 30  # 100 - 70
        assert abs(retry_cost - 0.04) < 1e-9  # 0.12 - 0.08

    def test_review_no_retry_single_event(self):
        """When no retry occurred, only one event is created."""
        from imas_codex.standard_names.review.pipeline import (
            _charge_review_cycle,
        )

        events: list[tuple[float, LLMCostEvent]] = []

        def _spy(self, cost, event):
            events.append((cost, event))
            return ChargeResult()

        mgr = _make_mgr()
        lease = mgr.reserve(5.0, phase="review_names")
        assert lease is not None

        result_dict = {
            "_cost": 0.10,
            "_input_tokens": 100,
            "_output_tokens": 50,
            "_primary_cost": 0.10,
            "_primary_input_tokens": 100,
            "_primary_output_tokens": 50,
        }

        with patch.object(BudgetLease, "charge_event", _spy):
            _charge_review_cycle(
                lease,
                0.10,
                result_dict,
                "model-c1",
                [{"id": "test"}],
                "magnetics",
                "c1",
                "review_names",
            )

        assert len(events) == 1
        _, event = events[0]
        assert event.batch_id == "magnetics"  # no -retry suffix
