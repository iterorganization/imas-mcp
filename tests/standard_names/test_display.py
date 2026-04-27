"""Tests for StandardNameProgressDisplay resources section.

Verifies that after removing the broken ``_build_resources_section``
override, the parent ``DataDrivenProgressDisplay`` correctly renders
both ETA and cost in the resources section.

Regression guard for: the SN-specific override that passed
``accumulated_cost=0.0`` (showing bold $0.00) and omitted ``eta``
(showing "--" forever).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.supervision import PipelinePhase
from imas_codex.standard_names.progress import StandardNameProgressDisplay

# ---------------------------------------------------------------------------
# Minimal engine-state stub (mirrors StandardNameBuildState fields used
# by DataDrivenProgressDisplay._build_resources_section and tick())
# ---------------------------------------------------------------------------


def _make_state(
    *,
    compose_processed: int = 5,
    compose_total: int = 10,
    compose_cost: float = 0.45,
    extract_processed: int = 10,
    extract_total: int = 10,
) -> SimpleNamespace:
    """Return a minimal engine-state namespace the display can read."""
    extract_stats = WorkerStats()
    extract_stats.processed = extract_processed
    extract_stats.total = extract_total
    extract_stats.freeze_rate()

    compose_stats = WorkerStats()
    compose_stats.processed = compose_processed
    compose_stats.total = compose_total
    compose_stats.cost = compose_cost
    # Give it a plausible rate so ETA can be computed
    compose_stats._ema_rate = 0.2  # 0.2 items/s → ETA ≈ remaining / 0.2

    finalize_stats = WorkerStats()
    finalize_stats.processed = 0
    finalize_stats.total = 3

    return SimpleNamespace(
        extract_stats=extract_stats,
        compose_stats=compose_stats,
        finalize_stats=finalize_stats,
        extract_phase=PipelinePhase("extract"),
        compose_phase=PipelinePhase("compose"),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def display():
    """Headless display with a 5-item compose backlog and $0.45 cost."""
    d = StandardNameProgressDisplay(cost_limit=2.0)
    d.set_engine_state(_make_state())
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResourcesSectionAfterFix:
    """Resources section must show real cost and ETA, not $0.00 / '--'."""

    def test_cost_row_shows_real_session_cost(self, display):
        """COST row must display the session cost (not $0.00)."""
        section = display._build_resources_section()
        text = section.plain
        assert "COST" in text, "COST row missing entirely"
        assert "$0.45" in text, f"Session cost $0.45 not in output: {text!r}"

    def test_cost_row_does_not_show_zero_as_primary(self, display):
        """$0.00 must not appear as the primary cost figure."""
        section = display._build_resources_section()
        text = section.plain
        # $0.45 is present; the erroneous $0.00 (from missing accumulated_cost)
        # should not appear now that the broken override is gone.
        assert "$0.00" not in text, (
            "Stale $0.00 still present — broken override may have been restored"
        )

    def test_eta_appears_when_compose_has_remaining_work(self, display):
        """ETA must be shown when compose has remaining items and a rate."""
        section = display._build_resources_section()
        text = section.plain
        assert "ETA" in text, (
            f"ETA missing from resources section: {text!r}\n"
            "Likely cause: _build_resources_section override was not removed"
        )

    def test_eta_value_is_plausible(self, display):
        """ETA value must be a recognisable time string, not '--'."""
        section = display._build_resources_section()
        text = section.plain
        # With 5 remaining items at 0.2 items/s → ETA ≈ 25s
        # Accept any digit-containing time format (e.g. "25s", "1m", "2m")
        import re

        time_pattern = re.compile(r"\d+[smh]")
        assert time_pattern.search(text), (
            f"No time value found after 'ETA' in: {text!r}"
        )

    def test_etc_projection_when_cost_and_remaining(self, display):
        """ETC appears when there are remaining items with a cost rate."""
        section = display._build_resources_section()
        text = section.plain
        # ETC = accumulated ($0.45) + remaining (5) * cost_per_item ($0.09)
        # ≈ $0.90 — check 'ETC' label is present
        assert "ETC" in text, f"ETC projection missing: {text!r}"

    def test_no_cost_row_when_no_spend(self):
        """COST row must be absent when compose has incurred no cost yet."""
        d = StandardNameProgressDisplay(cost_limit=2.0)
        state = _make_state(compose_cost=0.0)
        d.set_engine_state(state)
        section = d._build_resources_section()
        assert "COST" not in section.plain

    def test_no_eta_when_extract_only_complete(self):
        """No ETA when all stages are fully processed (nothing remaining)."""
        d = StandardNameProgressDisplay(cost_limit=2.0)
        state = _make_state(
            compose_processed=10,
            compose_total=10,
            compose_cost=0.45,
        )
        d.set_engine_state(state)
        section = d._build_resources_section()
        text = section.plain
        # All done — ETA row should show 'complete' or no ETA marker
        assert "ETA" not in text or "complete" in text, (
            f"Unexpected ETA after completion: {text!r}"
        )

    def test_uses_parent_implementation(self):
        """StandardNameProgressDisplay must NOT override _build_resources_section."""
        # Direct class dict check — override would hide the parent method
        assert "_build_resources_section" not in StandardNameProgressDisplay.__dict__, (
            "Broken _build_resources_section override is back — remove it from "
            "imas_codex/standard_names/progress.py"
        )
