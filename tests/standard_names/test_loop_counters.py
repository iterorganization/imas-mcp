"""Regression tests for run_sn_loop phase-name counter matching.

Plan 39, phase 5c: verify that the summary counters correctly accumulate
counts from the phase names actually emitted by turn.py.

Previously, ``names_reviewed`` was always zero because the loop matched
``phase.name == "review"`` while turn.py emits ``"review_names"`` and
``"review_docs"``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.standard_names.loop import RunSummary, run_sn_loop
from imas_codex.standard_names.turn import PhaseResult

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _make_phases(**counts: int) -> list[PhaseResult]:
    """Build a list of PhaseResult objects from name → count pairs."""
    return [
        PhaseResult(name=name, count=count, cost=0.0, skipped=False, error=None)
        for name, count in counts.items()
    ]


def _make_summary() -> RunSummary:
    return RunSummary(
        run_id="test-run",
        turn_number=1,
        started_at=datetime.now(UTC),
        cost_limit=100.0,
    )


# ─────────────────────────────────────────────────────────────────────
# Counter matching tests (unit-level, no graph/LLM calls)
# ─────────────────────────────────────────────────────────────────────


class TestReviewCounterMatching:
    """Verify that review_names and review_docs both increment names_reviewed."""

    def test_review_names_increments_counter(self, monkeypatch):
        """``review_names`` phase count accumulates into names_reviewed."""
        summary = _make_summary()
        phases = _make_phases(review_names=7)
        for phase in phases:
            if phase.name in ("review_names", "review_docs"):
                summary.names_reviewed += phase.count
        assert summary.names_reviewed == 7

    def test_review_docs_increments_counter(self, monkeypatch):
        """``review_docs`` phase count accumulates into names_reviewed."""
        summary = _make_summary()
        phases = _make_phases(review_docs=3)
        for phase in phases:
            if phase.name in ("review_names", "review_docs"):
                summary.names_reviewed += phase.count
        assert summary.names_reviewed == 3

    def test_both_review_phases_sum(self):
        """Both review phases are summed together."""
        summary = _make_summary()
        phases = _make_phases(review_names=5, review_docs=4)
        for phase in phases:
            if phase.name in ("review_names", "review_docs"):
                summary.names_reviewed += phase.count
        assert summary.names_reviewed == 9

    def test_old_review_name_not_matched(self):
        """Old stale phase name ``review`` does NOT increment the counter."""
        summary = _make_summary()
        phases = _make_phases(review=10)  # stale / wrong name
        for phase in phases:
            if phase.name in ("review_names", "review_docs"):
                summary.names_reviewed += phase.count
        assert summary.names_reviewed == 0


class TestAllPhaseCountersViaLoop:
    """Integration test: run_sn_loop aggregates all phase counters correctly."""

    @pytest.mark.asyncio
    async def test_counter_accumulation_across_phases(self):
        """All six counter types are accumulated from a single fake turn."""
        phases = _make_phases(
            generate=3,
            generate_docs=2,
            review_names=5,
            review_docs=4,
            refine_name=1,
            reconcile=6,
            link=8,
        )

        one_domain = [{"domain": "magnetics", "remaining": 10}]

        with (
            patch(
                "imas_codex.standard_names.loop._count_eligible_domains",
                return_value=one_domain,
            ),
            patch(
                "imas_codex.standard_names.loop._existing_domain_targets",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.finalize_sn_run",
            ),
            patch(
                "imas_codex.standard_names.graph_ops.create_sn_run_open",
            ),
            patch(
                "imas_codex.standard_names.turn.run_turn",
                new=AsyncMock(return_value=phases),
            ),
        ):
            summary = await run_sn_loop(cost_limit=100.0)

        assert summary.names_composed == 3, "generate → names_composed"
        assert summary.names_enriched == 2, "generate_docs → names_enriched"
        assert summary.names_reviewed == 9, "review_names+review_docs → names_reviewed"
        assert summary.names_regenerated == 1, "refine_name → names_regenerated"
        assert summary.sources_reconciled == 6, "reconcile → sources_reconciled"
        assert summary.links_resolved == 8, "link → links_resolved"
