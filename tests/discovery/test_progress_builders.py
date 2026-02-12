"""Tests for unified progress section builders.

Tests the composable building blocks in ``imas_codex.discovery.base.progress``
that all discovery CLIs share:
    - ``build_progress_section`` (pipeline progress bars)
    - ``build_activity_section`` (current activity rows)
    - ``build_resource_section`` (TIME, COST, TOTAL, STATS)
"""

from __future__ import annotations

import pytest
from rich.text import Text

from imas_codex.discovery.base.progress import (
    ActivityRowConfig,
    ProgressRowConfig,
    ResourceConfig,
    build_activity_section,
    build_progress_section,
    build_resource_section,
)

# =============================================================================
# build_progress_section
# =============================================================================


class TestBuildProgressSection:
    """Tests for build_progress_section."""

    def test_single_row(self):
        """Single row renders label, bar, count, and percentage."""
        rows = [
            ProgressRowConfig(name="SCAN", style="bold blue", completed=50, total=100),
        ]
        result = build_progress_section(rows, bar_width=20)
        assert isinstance(result, Text)
        text = result.plain
        assert "SCAN" in text
        assert "50" in text
        assert "50%" in text

    def test_multiple_rows_separated_by_newlines(self):
        """Multiple rows are joined by newlines."""
        rows = [
            ProgressRowConfig(name="SCAN", style="bold blue", completed=10, total=100),
            ProgressRowConfig(name="SCORE", style="bold green", completed=5, total=10),
        ]
        result = build_progress_section(rows, bar_width=20)
        lines = result.plain.split("\n")
        assert len(lines) == 2
        assert "SCAN" in lines[0]
        assert "SCORE" in lines[1]

    def test_rate_displayed_when_present(self):
        """Rate is shown when provided."""
        rows = [
            ProgressRowConfig(
                name="SCAN",
                style="bold blue",
                completed=50,
                total=100,
                rate=12.3,
            ),
        ]
        result = build_progress_section(rows, bar_width=20)
        assert "12.3/s" in result.plain

    def test_rate_hidden_when_none(self):
        """No rate text when rate is None."""
        rows = [
            ProgressRowConfig(
                name="SCAN",
                style="bold blue",
                completed=50,
                total=100,
                rate=None,
            ),
        ]
        result = build_progress_section(rows, bar_width=20)
        assert "/s" not in result.plain

    def test_disabled_row(self):
        """Disabled rows show disabled message instead of bar."""
        rows = [
            ProgressRowConfig(
                name="SCORE",
                style="bold green",
                disabled=True,
                disabled_msg="disabled",
            ),
        ]
        result = build_progress_section(rows, bar_width=20)
        text = result.plain
        assert "SCORE" in text
        assert "disabled" in text
        # No percentage or rate
        assert "%" not in text

    def test_zero_total_no_division_error(self):
        """Zero total defaults to 1 to avoid division by zero."""
        rows = [
            ProgressRowConfig(name="SCAN", style="bold blue", completed=0, total=0),
        ]
        # Should not raise
        result = build_progress_section(rows, bar_width=20)
        assert "SCAN" in result.plain

    def test_show_pct_false_hides_percentage(self):
        """show_pct=False hides the percentage column."""
        rows = [
            ProgressRowConfig(
                name="SCAN",
                style="bold blue",
                completed=50,
                total=100,
                show_pct=False,
            ),
        ]
        result = build_progress_section(rows, bar_width=20)
        assert "%" not in result.plain

    def test_completed_exceeds_total_clamps_to_100(self):
        """Progress ratio clamps at 1.0 when completed > total."""
        rows = [
            ProgressRowConfig(name="SCAN", style="bold blue", completed=150, total=100),
        ]
        result = build_progress_section(rows, bar_width=20)
        assert "100%" in result.plain


# =============================================================================
# build_activity_section
# =============================================================================


class TestBuildActivitySection:
    """Tests for build_activity_section."""

    def test_idle_state(self):
        """Default state shows 'idle'."""
        rows = [ActivityRowConfig(name="SCAN", style="bold blue")]
        result = build_activity_section(rows, content_width=80)
        text = result.plain
        assert "SCAN" in text
        assert "idle" in text

    def test_content_displays_primary_and_detail(self):
        """When primary_text is set, it appears on line 1 with details on line 2."""
        rows = [
            ActivityRowConfig(
                name="SCORE",
                style="bold green",
                primary_text="/home/user/code/analysis.py",
                detail_parts=[("0.85", "green"), ("  equilibrium", "cyan")],
            ),
        ]
        result = build_activity_section(rows, content_width=80)
        text = result.plain
        assert "/home/user/code/analysis.py" in text
        assert "0.85" in text
        assert "equilibrium" in text

    def test_processing_state(self):
        """Processing state shows processing label."""
        rows = [
            ActivityRowConfig(
                name="SCORE",
                style="bold green",
                is_processing=True,
                processing_label="scoring batch...",
            ),
        ]
        result = build_activity_section(rows, content_width=80)
        assert "scoring batch..." in result.plain

    def test_processing_paused(self):
        """Processing + paused shows 'paused' instead of processing label."""
        rows = [
            ActivityRowConfig(
                name="SCORE",
                style="bold green",
                is_processing=True,
                is_paused=True,
            ),
        ]
        result = build_activity_section(rows, content_width=80)
        assert "paused" in result.plain

    def test_queue_streaming(self):
        """Queue with items shows streaming message."""
        rows = [
            ActivityRowConfig(
                name="SCAN",
                style="bold blue",
                queue_size=42,
            ),
        ]
        result = build_activity_section(rows, content_width=80)
        assert "streaming 42 items" in result.plain

    def test_complete_state(self):
        """Complete state shows complete label."""
        rows = [
            ActivityRowConfig(
                name="SCORE",
                style="bold green",
                is_complete=True,
                complete_label="cost limit",
            ),
        ]
        result = build_activity_section(rows, content_width=80)
        assert "cost limit" in result.plain

    def test_complete_default_label(self):
        """Complete state with default label shows 'complete'."""
        rows = [
            ActivityRowConfig(
                name="SCAN",
                style="bold blue",
                is_complete=True,
            ),
        ]
        result = build_activity_section(rows, content_width=80)
        assert "complete" in result.plain

    def test_paused_state(self):
        """Paused state without processing shows 'paused'."""
        rows = [
            ActivityRowConfig(
                name="SCAN",
                style="bold blue",
                is_paused=True,
            ),
        ]
        result = build_activity_section(rows, content_width=80)
        assert "paused" in result.plain

    def test_disabled_row_skipped(self):
        """Disabled rows produce no output."""
        rows = [
            ActivityRowConfig(
                name="SCAN",
                style="bold blue",
                disabled=True,
            ),
        ]
        result = build_activity_section(rows, content_width=80)
        assert "SCAN" not in result.plain

    def test_multiple_rows(self):
        """Multiple rows each get their label rendered."""
        rows = [
            ActivityRowConfig(name="SCAN", style="bold blue", primary_text="/path/a"),
            ActivityRowConfig(
                name="SCORE",
                style="bold green",
                primary_text="/path/b",
            ),
        ]
        result = build_activity_section(rows, content_width=80)
        text = result.plain
        assert "SCAN" in text
        assert "SCORE" in text
        assert "/path/a" in text
        assert "/path/b" in text

    def test_long_primary_text_clipped(self):
        """Primary text longer than content_width is truncated with ellipsis."""
        long_text = "x" * 200
        rows = [
            ActivityRowConfig(
                name="SCAN",
                style="bold blue",
                primary_text=long_text,
            ),
        ]
        result = build_activity_section(rows, content_width=60)
        text = result.plain
        assert "..." in text
        assert len(long_text) > 60  # Original was longer
        # The displayed text should be shorter than original
        # Can't check exact length due to label prefix

    def test_has_content_property(self):
        """has_content is True when primary_text is set."""
        row = ActivityRowConfig(
            name="SCAN", style="bold blue", primary_text="some text"
        )
        assert row.has_content is True

        row_empty = ActivityRowConfig(name="SCAN", style="bold blue")
        assert row_empty.has_content is False


# =============================================================================
# build_resource_section (ResourceConfig)
# =============================================================================


class TestBuildResourceSection:
    """Tests for build_resource_section with ResourceConfig."""

    def test_time_row_always_present(self):
        """TIME row is always rendered."""
        config = ResourceConfig(elapsed=3661.0)
        result = build_resource_section(config, gauge_width=20)
        text = result.plain
        assert "TIME" in text
        assert "1h 01m" in text

    def test_eta_displayed(self):
        """ETA is shown when provided."""
        config = ResourceConfig(elapsed=60.0, eta=120.0)
        result = build_resource_section(config, gauge_width=20)
        assert "ETA" in result.plain

    def test_eta_zero_shows_complete(self):
        """ETA of 0 shows 'complete'."""
        config = ResourceConfig(elapsed=60.0, eta=0.0)
        result = build_resource_section(config, gauge_width=20)
        assert "complete" in result.plain

    def test_eta_zero_with_limit_reason(self):
        """ETA of 0 with limit_reason shows the limit message."""
        config = ResourceConfig(elapsed=60.0, eta=0.0, limit_reason="cost")
        result = build_resource_section(config, gauge_width=20)
        assert "cost limit reached" in result.plain

    def test_cost_row_displayed(self):
        """COST row appears when run_cost and cost_limit are set."""
        config = ResourceConfig(elapsed=60.0, run_cost=2.50, cost_limit=10.0)
        result = build_resource_section(config, gauge_width=20)
        text = result.plain
        assert "COST" in text
        assert "$2.50" in text
        assert "$10.00" in text

    def test_cost_hidden_in_scan_only(self):
        """COST row is hidden when scan_only=True."""
        config = ResourceConfig(
            elapsed=60.0, run_cost=2.50, cost_limit=10.0, scan_only=True
        )
        result = build_resource_section(config, gauge_width=20)
        assert "COST" not in result.plain

    def test_total_row_with_accumulated(self):
        """TOTAL row shows accumulated + run cost."""
        config = ResourceConfig(
            elapsed=60.0,
            run_cost=2.50,
            cost_limit=10.0,
            accumulated_cost=5.0,
        )
        result = build_resource_section(config, gauge_width=20)
        text = result.plain
        assert "TOTAL" in text
        assert "$7.50" in text  # 5.0 + 2.5

    def test_total_row_with_etc(self):
        """TOTAL row shows ETC when projection exceeds current total."""
        config = ResourceConfig(
            elapsed=60.0,
            run_cost=1.0,
            cost_limit=10.0,
            accumulated_cost=0.0,
            etc=5.0,
        )
        result = build_resource_section(config, gauge_width=20)
        text = result.plain
        assert "ETC" in text
        assert "$5.00" in text

    def test_stats_row(self):
        """STATS row renders label=value pairs."""
        config = ResourceConfig(
            elapsed=60.0,
            stats=[
                ("scored", "100", "blue"),
                ("ingested", "50", "green"),
            ],
        )
        result = build_resource_section(config, gauge_width=20)
        text = result.plain
        assert "STATS" in text
        assert "scored=100" in text
        assert "ingested=50" in text

    def test_pending_in_stats(self):
        """Pending counts appear after stats."""
        config = ResourceConfig(
            elapsed=60.0,
            stats=[("scored", "100", "blue")],
            pending=[("score", 42), ("ingest", 10)],
        )
        result = build_resource_section(config, gauge_width=20)
        text = result.plain
        assert "pending=" in text
        assert "score:42" in text
        assert "ingest:10" in text

    def test_pending_zero_hidden(self):
        """Pending items with count=0 are not shown."""
        config = ResourceConfig(
            elapsed=60.0,
            stats=[("scored", "100", "blue")],
            pending=[("score", 0), ("ingest", 0)],
        )
        result = build_resource_section(config, gauge_width=20)
        text = result.plain
        assert "pending" not in text

    def test_no_cost_row_when_none(self):
        """No COST row when run_cost or cost_limit is None."""
        config = ResourceConfig(elapsed=60.0)
        result = build_resource_section(config, gauge_width=20)
        assert "COST" not in result.plain

    def test_eta_hidden_in_scan_only(self):
        """ETA is suppressed in scan_only mode."""
        config = ResourceConfig(elapsed=60.0, eta=120.0, scan_only=True)
        result = build_resource_section(config, gauge_width=20)
        # Should not have ETA text
        assert "ETA" not in result.plain


# =============================================================================
# Integration: display classes can render without errors
# =============================================================================


class TestDisplayClassIntegration:
    """Smoke tests verifying each display class renders a Panel without errors."""

    def test_wiki_display_renders(self):
        """WikiProgressDisplay._build_display() produces a Panel."""
        from rich.panel import Panel

        from imas_codex.discovery.wiki.progress import WikiProgressDisplay

        display = WikiProgressDisplay(facility="test", cost_limit=1.0, console=None)
        # Seed some state
        display.state.total_pages = 100
        display.state.pages_scored = 10
        display.state.pages_ingested = 5
        panel = display._build_display()
        assert isinstance(panel, Panel)

    def test_data_display_renders(self):
        """DataProgressDisplay._build_display() produces a Panel."""
        from rich.panel import Panel

        from imas_codex.discovery.data.progress import DataProgressDisplay

        display = DataProgressDisplay(facility="test", cost_limit=1.0, console=None)
        display.state.total_signals = 50
        display.state.signals_enriched = 5
        panel = display._build_display()
        assert isinstance(panel, Panel)

    def test_paths_display_renders(self):
        """ParallelProgressDisplay._build_display() produces a Panel."""
        from rich.panel import Panel

        from imas_codex.discovery.paths.progress import ParallelProgressDisplay

        display = ParallelProgressDisplay(facility="test", cost_limit=1.0, console=None)
        display.state.total = 100
        display.state.scored = 20
        panel = display._build_display()
        assert isinstance(panel, Panel)

    def test_wiki_display_scan_only_renders(self):
        """Wiki display in scan_only mode skips SCORE/PAGE sections."""
        from imas_codex.discovery.wiki.progress import WikiProgressDisplay

        display = WikiProgressDisplay(facility="test", cost_limit=1.0, scan_only=True)
        panel = display._build_display()
        text = panel.renderable.plain
        # scan_only should show "SCAN ONLY" in header
        assert "SCAN ONLY" in text
