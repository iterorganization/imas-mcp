"""Tests for wiki progress display.

Covers ProgressState properties, WikiProgressDisplay builder methods,
and display items.
"""

from __future__ import annotations

import time

import pytest

from imas_codex.discovery.wiki.progress import (
    ArtifactItem,
    ImageItem,
    IngestItem,
    ProgressState,
    ScoreItem,
    WikiProgressDisplay,
)

# =============================================================================
# ProgressState properties
# =============================================================================


class TestProgressStateProperties:
    """Tests for ProgressState computed properties."""

    def _state(self, **kwargs) -> ProgressState:
        defaults = {"facility": "tcv", "cost_limit": 1.0}
        defaults.update(kwargs)
        return ProgressState(**defaults)

    def test_elapsed(self):
        """Elapsed time should be positive."""
        state = self._state()
        time.sleep(0.01)
        assert state.elapsed > 0

    def test_run_cost_combines_all_sources(self):
        """Run cost should sum all cost types across sites."""
        state = self._state(
            _run_score_cost=0.10,
            _run_ingest_cost=0.05,
            _run_artifact_score_cost=0.03,
            _run_image_score_cost=0.02,
            _offset_score_cost=0.01,
            _offset_ingest_cost=0.01,
            _offset_artifact_score_cost=0.01,
            _offset_image_score_cost=0.01,
        )
        expected = 0.10 + 0.05 + 0.03 + 0.02 + 0.01 + 0.01 + 0.01 + 0.01
        assert abs(state.run_cost - expected) < 1e-10

    def test_cost_fraction_within_bounds(self):
        """Cost fraction should be [0, 1]."""
        state = self._state(cost_limit=1.0, _run_score_cost=0.5)
        assert 0.0 <= state.cost_fraction <= 1.0

    def test_cost_fraction_zero_limit(self):
        """Zero cost limit should give 0 fraction."""
        state = self._state(cost_limit=0.0)
        assert state.cost_fraction == 0.0

    def test_cost_fraction_over_limit(self):
        """Over-limit cost should clamp to 1.0."""
        state = self._state(cost_limit=0.10, _run_score_cost=0.50)
        assert state.cost_fraction == 1.0

    def test_cost_limit_reached(self):
        """Should detect when cost exceeds limit."""
        state = self._state(cost_limit=0.10, _run_score_cost=0.15)
        assert state.cost_limit_reached is True

    def test_cost_limit_not_reached(self):
        """Should return False when cost is under limit."""
        state = self._state(cost_limit=1.0, _run_score_cost=0.01)
        assert state.cost_limit_reached is False

    def test_cost_limit_zero(self):
        """Zero cost limit should not trigger limit."""
        state = self._state(cost_limit=0.0, _run_score_cost=10.0)
        assert state.cost_limit_reached is False

    def test_page_limit_reached(self):
        """Should detect when page limit is hit."""
        state = self._state(
            cost_limit=1.0,
            page_limit=100,
            run_scored=100,
        )
        assert state.page_limit_reached is True

    def test_page_limit_not_reached(self):
        state = self._state(
            cost_limit=1.0,
            page_limit=100,
            run_scored=50,
        )
        assert state.page_limit_reached is False

    def test_page_limit_none(self):
        """No page limit should never be reached."""
        state = self._state(cost_limit=1.0, page_limit=None, run_scored=1000)
        assert state.page_limit_reached is False

    def test_limit_reason_cost(self):
        state = self._state(cost_limit=0.01, _run_score_cost=0.02)
        assert state.limit_reason == "cost"

    def test_limit_reason_page(self):
        state = self._state(
            cost_limit=10.0, page_limit=50, run_scored=50
        )
        assert state.limit_reason == "page"

    def test_limit_reason_none(self):
        state = self._state(cost_limit=10.0, run_scored=0)
        assert state.limit_reason is None

    def test_cost_per_page(self):
        """Cost per page should be average scoring cost."""
        state = self._state(cost_limit=1.0, run_scored=10, _run_score_cost=0.50)
        assert state.cost_per_page == pytest.approx(0.05)

    def test_cost_per_page_no_scored(self):
        """No scored pages should return None."""
        state = self._state(cost_limit=1.0, run_scored=0)
        assert state.cost_per_page is None

    def test_cost_per_artifact_score(self):
        state = self._state(
            cost_limit=1.0,
            run_artifacts_scored=5,
            _run_artifact_score_cost=0.25,
        )
        assert state.cost_per_artifact_score == pytest.approx(0.05)

    def test_cost_per_image_score(self):
        state = self._state(
            cost_limit=1.0,
            run_images_scored=10,
            _run_image_score_cost=0.10,
        )
        assert state.cost_per_image_score == pytest.approx(0.01)

    def test_total_run_scored_with_offsets(self):
        """Should sum offsets from previous sites."""
        state = self._state(
            cost_limit=1.0,
            run_scored=20,
            _offset_scored=30,
        )
        assert state.total_run_scored == 50

    def test_total_run_ingested_with_offsets(self):
        state = self._state(
            cost_limit=1.0,
            run_ingested=10,
            _offset_ingested=15,
        )
        assert state.total_run_ingested == 25

    def test_total_run_artifacts_with_offsets(self):
        state = self._state(
            cost_limit=1.0,
            run_artifacts=5,
            _offset_artifacts=10,
        )
        assert state.total_run_artifacts == 15

    def test_total_run_images_with_offsets(self):
        state = self._state(
            cost_limit=1.0,
            run_images_scored=3,
            _offset_images_scored=7,
        )
        assert state.total_run_images_scored == 10


# =============================================================================
# ProgressState ETA calculation
# =============================================================================


class TestProgressStateEta:
    """Tests for ProgressState.eta_seconds computation."""

    def test_eta_from_cost_limit(self):
        """Cost-limited run should estimate time to exhaust budget."""
        state = ProgressState(
            facility="tcv",
            cost_limit=1.0,
            _run_score_cost=0.50,
        )
        # Force elapsed to 60 seconds
        state.start_time = time.time() - 60
        eta = state.eta_seconds
        assert eta is not None
        assert eta > 0  # Should have remaining budget time

    def test_eta_from_page_limit(self):
        """Page-limited run should estimate time to reach limit."""
        state = ProgressState(
            facility="tcv",
            cost_limit=0.0,  # No cost limit
            page_limit=100,
            run_scored=50,
        )
        state.start_time = time.time() - 100
        eta = state.eta_seconds
        assert eta is not None
        assert eta > 0

    def test_eta_from_pending_work(self):
        """Work-based ETA should use slowest worker group."""
        state = ProgressState(
            facility="tcv",
            cost_limit=0.0,
            pending_score=100,
            score_rate=10.0,  # 10/sec → 10s for score
            pending_ingest=200,
            ingest_rate=5.0,  # 5/sec → 40s for ingest
        )
        state.start_time = time.time() - 60
        eta = state.eta_seconds
        assert eta is not None
        # Should pick the slower worker (ingest: 200/5 = 40s)
        assert eta == pytest.approx(40.0)

    def test_eta_none_when_no_data(self):
        """No rate data should return None."""
        state = ProgressState(
            facility="tcv",
            cost_limit=0.0,
        )
        assert state.eta_seconds is None


# =============================================================================
# Display items
# =============================================================================


class TestDisplayItems:
    """Tests for display item dataclasses."""

    def test_score_item(self):
        item = ScoreItem(
            title="Thomson_Scattering",
            score=0.85,
            physics_domain="diagnostics",
            description="Thomson scattering diagnostic page",
            is_physics=True,
        )
        assert item.title == "Thomson_Scattering"
        assert item.score == 0.85
        assert item.skipped is False

    def test_score_item_skipped(self):
        item = ScoreItem(
            title="Meeting_Notes",
            skipped=True,
            skip_reason="too short",
        )
        assert item.skipped is True
        assert item.skip_reason == "too short"

    def test_ingest_item(self):
        item = IngestItem(
            title="LIUQE Documentation",
            score=0.90,
            chunk_count=12,
        )
        assert item.chunk_count == 12

    def test_artifact_item(self):
        item = ArtifactItem(
            filename="report.pdf",
            artifact_type="pdf",
            score=0.75,
            chunk_count=8,
        )
        assert item.filename == "report.pdf"
        assert item.artifact_type == "pdf"

    def test_image_item(self):
        item = ImageItem(
            image_id="tcv:abc123",
            description="Plasma cross-section",
            score=0.60,
        )
        assert item.image_id == "tcv:abc123"
        assert item.description == "Plasma cross-section"


# =============================================================================
# WikiProgressDisplay construction
# =============================================================================


class TestWikiProgressDisplayConstruction:
    """Tests for WikiProgressDisplay initialization."""

    def test_default_construction(self):
        display = WikiProgressDisplay(facility="tcv", cost_limit=1.0)
        assert display.state.facility == "tcv"
        assert display.state.cost_limit == 1.0
        assert display.state.scan_only is False
        assert display.state.score_only is False

    def test_scan_only_mode(self):
        display = WikiProgressDisplay(
            facility="iter", cost_limit=0.0, scan_only=True
        )
        assert display.state.scan_only is True

    def test_score_only_mode(self):
        display = WikiProgressDisplay(
            facility="jet", cost_limit=0.5, score_only=True
        )
        assert display.state.score_only is True

    def test_with_focus(self):
        display = WikiProgressDisplay(
            facility="tcv", cost_limit=1.0, focus="diagnostics"
        )
        assert display.state.focus == "diagnostics"

    def test_with_page_limit(self):
        display = WikiProgressDisplay(
            facility="tcv", cost_limit=1.0, page_limit=500
        )
        assert display.state.page_limit == 500

    def test_width_calculation(self):
        """Width should be at least MIN_WIDTH."""
        from imas_codex.discovery.base.progress import MIN_WIDTH

        display = WikiProgressDisplay(facility="tcv", cost_limit=1.0)
        assert display.width >= MIN_WIDTH


# =============================================================================
# WikiProgressDisplay header/title
# =============================================================================


class TestWikiProgressDisplayHeader:
    """Tests for header rendering."""

    def test_basic_header(self):
        display = WikiProgressDisplay(facility="tcv", cost_limit=1.0)
        header = display._build_header()
        assert "TCV" in header.plain
        assert "Wiki Discovery" in header.plain

    def test_scan_only_header(self):
        display = WikiProgressDisplay(
            facility="tcv", cost_limit=0.0, scan_only=True
        )
        header = display._build_header()
        assert "SCAN ONLY" in header.plain

    def test_score_only_header(self):
        display = WikiProgressDisplay(
            facility="tcv", cost_limit=1.0, score_only=True
        )
        header = display._build_header()
        assert "SCORE ONLY" in header.plain

    def test_focus_in_header(self):
        display = WikiProgressDisplay(
            facility="tcv", cost_limit=1.0, focus="diagnostics"
        )
        header = display._build_header()
        assert "diagnostics" in header.plain

    def test_multi_site_header(self):
        display = WikiProgressDisplay(facility="jt60sa", cost_limit=1.0)
        display.state.total_sites = 3
        display.state.current_site_name = "https://jt60sa.org/twiki"
        header = display._build_header()
        assert "jt60sa.org/twiki" in header.plain


# =============================================================================
# WikiProgressDisplay._clip_title
# =============================================================================


class TestClipTitle:
    """Tests for title clipping utility."""

    def test_short_title_unchanged(self):
        display = WikiProgressDisplay(facility="tcv", cost_limit=1.0)
        assert display._clip_title("Short Title", max_len=70) == "Short Title"

    def test_long_title_clipped(self):
        display = WikiProgressDisplay(facility="tcv", cost_limit=1.0)
        long = "A" * 100
        clipped = display._clip_title(long, max_len=30)
        assert len(clipped) == 30
        assert clipped.endswith("...")

    def test_exact_length_unchanged(self):
        display = WikiProgressDisplay(facility="tcv", cost_limit=1.0)
        title = "A" * 70
        assert display._clip_title(title, max_len=70) == title


# =============================================================================
# StreamQueue
# =============================================================================


class TestStreamQueue:
    """Tests for StreamQueue rate-limiting behavior."""

    def test_initially_empty(self):
        from imas_codex.discovery.base.progress import StreamQueue

        q = StreamQueue(rate=1.0)
        assert q.is_empty()

    def test_add_and_not_empty(self):
        from imas_codex.discovery.base.progress import StreamQueue

        q = StreamQueue(rate=1.0, max_rate=10.0, min_display_time=0.0)
        q.add(["item1"])
        assert not q.is_empty()

    def test_len(self):
        from imas_codex.discovery.base.progress import StreamQueue

        q = StreamQueue(rate=1.0, max_rate=10.0, min_display_time=0.0)
        q.add(["a", "b"])
        assert len(q) == 2
