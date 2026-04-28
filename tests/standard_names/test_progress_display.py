"""Tests for Phase 8 per-subpool health display with wedge detection.

Covers:
- Dual-subpool status_text formatting (REVIEW, GENERATE rows).
- Single-subpool simple form (ENRICH row).
- Wedge threshold logic (is_wedged timing).
- Red Rich markup for wedged subpool names.
- Idle-is-fine (pending=0 → never wedged regardless of time).
- Both-subpools-wedged multi-label rendering.
"""

from __future__ import annotations

import time

import pytest

from imas_codex.standard_names.pools import PoolHealth
from imas_codex.standard_names.progress import (
    _ENRICH_SUBPOOLS,
    _GENERATE_SUBPOOLS,
    _REVIEW_SUBPOOLS,
    WEDGE_THRESHOLD,
    SNPoolState,
    build_sn_pool_stages,
    format_pool_health_text,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_health(pool: str, *, pending: int = 0, ago: float = 0.0) -> PoolHealth:
    """Create a PoolHealth with last_progress_at = now - ago."""
    ph = PoolHealth(pool=pool)
    ph.pending_count = pending
    ph.last_progress_at = time.time() - ago
    return ph


def _make_pool_state(
    *,
    compose_pending: int = 0,
    compose_ago: float = 0.0,
    regen_pending: int = 0,
    regen_ago: float = 0.0,
    enrich_pending: int = 0,
    enrich_ago: float = 0.0,
    names_pending: int = 0,
    names_ago: float = 0.0,
    docs_pending: int = 0,
    docs_ago: float = 0.0,
) -> SNPoolState:
    """Create an SNPoolState with populated PoolHealth references."""
    state = SNPoolState()
    now = time.time()
    for name, pending, ago in [
        ("generate", compose_pending, compose_ago),
        ("regen", regen_pending, regen_ago),
        ("enrich", enrich_pending, enrich_ago),
        ("review_names", names_pending, names_ago),
        ("review_docs", docs_pending, docs_ago),
    ]:
        ph = PoolHealth(pool=name)
        ph.pending_count = pending
        ph.last_progress_at = now - ago
        state.set_pool_health(name, ph)
    return state


# ═══════════════════════════════════════════════════════════════════════
# 1. Dual-subpool row text (REVIEW row: names + docs)
# ═══════════════════════════════════════════════════════════════════════


class TestReviewRowSubpoolCounts:
    def test_review_row_shows_names_and_docs_pending(self):
        """Both subpools non-zero → status_text contains names=42 and docs=8."""
        state = _make_pool_state(names_pending=42, docs_pending=8)
        state.refresh_pool_health()

        assert "names=42" in state.review_stats.status_text
        assert "docs=8" in state.review_stats.status_text


# ═══════════════════════════════════════════════════════════════════════
# 2. Dual-subpool row text (GENERATE row: compose + regen)
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateRowSubpoolCounts:
    def test_generate_row_shows_compose_and_regen_pending(self):
        """Both subpools non-zero → status_text contains compose=15 and regen=3."""
        state = _make_pool_state(compose_pending=15, regen_pending=3)
        state.refresh_pool_health()

        assert "compose=15" in state.generate_stats.status_text
        assert "regen=3" in state.generate_stats.status_text


# ═══════════════════════════════════════════════════════════════════════
# 3. Single-subpool simple form (ENRICH row)
# ═══════════════════════════════════════════════════════════════════════


class TestEnrichRowSimpleForm:
    def test_enrich_row_simple_form(self):
        """Single subpool row → simple pending=N form."""
        state = _make_pool_state(enrich_pending=27)
        state.refresh_pool_health()

        assert "pending=27" in state.enrich_stats.status_text
        # Should NOT have dual-subpool notation.
        assert "enrich=" not in state.enrich_stats.status_text


# ═══════════════════════════════════════════════════════════════════════
# 4. Wedge threshold detection
# ═══════════════════════════════════════════════════════════════════════


class TestWedgeThreshold:
    def test_wedged_when_stale_and_pending(self):
        """last_progress_at=7s ago, pending=5 → is_wedged=True."""
        state = _make_pool_state(names_pending=5, names_ago=7.0)
        assert state.is_wedged("review_names") is True

    def test_not_wedged_when_recent(self):
        """last_progress_at=4s ago, pending=5 → is_wedged=False."""
        state = _make_pool_state(names_pending=5, names_ago=4.0)
        assert state.is_wedged("review_names") is False


# ═══════════════════════════════════════════════════════════════════════
# 5. Wedge renders red subpool name
# ═══════════════════════════════════════════════════════════════════════


class TestWedgeRendersRedMarkup:
    def test_wedge_renders_red_subpool_name(self):
        """When names subpool is wedged, status_text contains [red]names[/red]."""
        state = _make_pool_state(
            names_pending=42,
            names_ago=10.0,  # well past WEDGE_THRESHOLD
            docs_pending=8,
            docs_ago=1.0,  # recent — not wedged
        )
        state.refresh_pool_health()

        text = state.review_stats.status_text
        assert "[red]names[/red]" in text
        assert "docs=8" in text
        # docs should NOT be red.
        assert "[red]docs[/red]" not in text
        # Markup flag should be set.
        assert state.review_stats.status_markup is True


# ═══════════════════════════════════════════════════════════════════════
# 6. No wedge when pending is zero (idle is fine)
# ═══════════════════════════════════════════════════════════════════════


class TestNoWedgeWhenPendingZero:
    def test_no_wedge_when_pending_zero(self):
        """last_progress_at=10m ago BUT pending_count=0 → NOT wedged."""
        state = _make_pool_state(
            names_pending=0,
            names_ago=600.0,  # 10 minutes
        )
        assert state.is_wedged("review_names") is False

    def test_no_wedge_status_text_empty_when_pending_zero(self):
        """Empty pending → no status_text at all (idle)."""
        state = _make_pool_state(
            names_pending=0,
            names_ago=600.0,
        )
        state.refresh_pool_health()
        assert state.review_stats.status_text == ""


# ═══════════════════════════════════════════════════════════════════════
# 7. Both subpools wedged → both names red
# ═══════════════════════════════════════════════════════════════════════


class TestBothSubpoolsWedged:
    def test_both_subpools_wedged_lists_both(self):
        """names AND docs both wedged → both appear red in message."""
        state = _make_pool_state(
            names_pending=42,
            names_ago=30.0,
            docs_pending=8,
            docs_ago=20.0,
        )
        state.refresh_pool_health()

        text = state.review_stats.status_text
        assert "[red]names[/red]" in text
        assert "[red]docs[/red]" in text
        assert "wedged" in text
        assert state.review_stats.status_markup is True


# ═══════════════════════════════════════════════════════════════════════
# Supplementary: format_pool_health_text unit tests
# ═══════════════════════════════════════════════════════════════════════


class TestFormatPoolHealthText:
    def test_empty_health_map(self):
        assert format_pool_health_text(_REVIEW_SUBPOOLS, {}) == ""

    def test_single_subpool_no_pending(self):
        ph = PoolHealth(pool="enrich")
        ph.pending_count = 0
        assert format_pool_health_text(_ENRICH_SUBPOOLS, {"enrich": ph}) == ""

    def test_multi_subpool_one_zero(self):
        """When one subpool has pending=0, only the other appears."""
        now = time.time()
        names_h = PoolHealth(pool="review_names")
        names_h.pending_count = 10
        names_h.last_progress_at = now
        docs_h = PoolHealth(pool="review_docs")
        docs_h.pending_count = 0
        docs_h.last_progress_at = now
        text = format_pool_health_text(
            _REVIEW_SUBPOOLS,
            {"review_names": names_h, "review_docs": docs_h},
            now=now,
        )
        assert "names=10" in text
        assert "docs" not in text


# ═══════════════════════════════════════════════════════════════════════
# Supplementary: build_sn_pool_stages structure
# ═══════════════════════════════════════════════════════════════════════


class TestBuildSNPoolStages:
    def test_three_stages_by_default(self):
        stages = build_sn_pool_stages()
        assert len(stages) == 3
        assert [s.name for s in stages] == ["GENERATE", "ENRICH", "REVIEW"]

    def test_skip_flags(self):
        stages = build_sn_pool_stages(
            skip_generate=True,
            skip_enrich=True,
            skip_review=True,
        )
        assert all(s.disabled for s in stages)

    def test_stages_match_pool_state(self):
        """Stage stats_attr values exist on SNPoolState."""
        state = SNPoolState()
        stages = build_sn_pool_stages()
        for stage in stages:
            assert hasattr(state, stage.stats_attr), (
                f"SNPoolState missing {stage.stats_attr}"
            )
