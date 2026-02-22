"""Tests for unified progress section builders.

Tests the composable building blocks in ``imas_codex.discovery.base.progress``
that all discovery CLIs share:
    - ``build_pipeline_section`` (unified pipeline: progress bars + activity)
    - ``build_resource_section`` (TIME, COST, TOTAL, STATS)
"""

from __future__ import annotations

import pytest
from rich.text import Text

from imas_codex.discovery.base.progress import (
    PipelineRowConfig,
    ResourceConfig,
    build_pipeline_section,
    build_resource_section,
)

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
        """COST row appears when run_cost is set."""
        config = ResourceConfig(elapsed=60.0, run_cost=2.50)
        result = build_resource_section(config, gauge_width=20)
        text = result.plain
        assert "COST" in text
        assert "$2.50" in text

    def test_cost_hidden_in_scan_only(self):
        """COST row is hidden when scan_only=True."""
        config = ResourceConfig(elapsed=60.0, run_cost=2.50, scan_only=True)
        result = build_resource_section(config, gauge_width=20)
        assert "COST" not in result.plain

    def test_cost_with_accumulated(self):
        """COST row shows accumulated total from graph as primary cost."""
        config = ResourceConfig(
            elapsed=60.0,
            run_cost=2.50,
            accumulated_cost=5.0,
        )
        result = build_resource_section(config, gauge_width=20)
        text = result.plain
        assert "COST" in text
        assert "$5.00" in text  # accumulated from graph is primary
        assert "session $2.50" in text

    def test_cost_with_etc(self):
        """COST row shows ETC when projection exceeds current total."""
        config = ResourceConfig(
            elapsed=60.0,
            run_cost=1.0,
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
        """No COST row when run_cost is None."""
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
# build_pipeline_section (unified progress + activity)
# =============================================================================


class TestBuildPipelineSection:
    """Tests for the unified pipeline section builder."""

    def test_single_row_renders_three_lines(self):
        """A single pipeline row renders 3 lines: bar, activity, detail."""
        rows = [
            PipelineRowConfig(
                name="TRIAGE",
                style="bold blue",
                completed=50,
                total=100,
                primary_text="Thomson_Scattering",
                detail_parts=[("0.85  ", "green"), ("diagnostics", "cyan")],
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        text = result.plain
        assert "TRIAGE" in text
        assert "50" in text
        assert "Thomson_Scattering" in text
        assert "0.85" in text
        assert "diagnostics" in text
        # Should have 3 lines
        lines = text.split("\n")
        assert len(lines) == 3

    def test_cost_displayed_when_present(self):
        """Per-stage cost is shown in the progress bar line."""
        rows = [
            PipelineRowConfig(
                name="TRIAGE",
                style="bold blue",
                completed=10,
                total=100,
                cost=2.50,
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        assert "$2.50" in result.plain

    def test_cost_hidden_when_none(self):
        """No cost text when cost is None."""
        rows = [
            PipelineRowConfig(
                name="PAGES",
                style="bold magenta",
                completed=10,
                total=100,
                cost=None,
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        assert "$" not in result.plain

    def test_worker_count_annotation(self):
        """Worker count shows xN in label."""
        rows = [
            PipelineRowConfig(
                name="TRIAGE",
                style="bold blue",
                completed=10,
                total=100,
                worker_count=4,
                primary_text="some page",
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        assert "x4" in result.plain

    def test_worker_annotation_with_backoff(self):
        """Worker annotation adds extra info after xN."""
        rows = [
            PipelineRowConfig(
                name="DOCS",
                style="bold yellow",
                completed=5,
                total=50,
                worker_count=2,
                worker_annotation="(1 backoff)",
                primary_text="report.pdf",
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        assert "x2" in result.plain
        assert "1 backoff" in result.plain

    def test_idle_state(self):
        """No content shows 'idle'."""
        rows = [
            PipelineRowConfig(name="IMAGES", style="bold green", completed=0, total=1),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        assert "idle" in result.plain

    def test_processing_state(self):
        """Processing without content shows processing label."""
        rows = [
            PipelineRowConfig(
                name="TRIAGE",
                style="bold blue",
                completed=5,
                total=100,
                is_processing=True,
                processing_label="scoring...",
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        assert "scoring..." in result.plain

    def test_paused_state(self):
        """Paused processing shows 'paused'."""
        rows = [
            PipelineRowConfig(
                name="TRIAGE",
                style="bold blue",
                completed=5,
                total=100,
                is_processing=True,
                is_paused=True,
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        assert "paused" in result.plain

    def test_complete_state(self):
        """Complete state shows complete label."""
        rows = [
            PipelineRowConfig(
                name="DOCS",
                style="bold yellow",
                completed=50,
                total=50,
                is_complete=True,
                complete_label="cost limit",
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        assert "cost limit" in result.plain

    def test_queue_streaming(self):
        """Queue with items shows streaming message."""
        rows = [
            PipelineRowConfig(
                name="TRIAGE",
                style="bold blue",
                completed=5,
                total=100,
                queue_size=42,
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        assert "streaming 42 items" in result.plain

    def test_disabled_row_skipped(self):
        """Disabled rows are not rendered."""
        rows = [
            PipelineRowConfig(
                name="TRIAGE",
                style="bold blue",
                disabled=True,
            ),
            PipelineRowConfig(
                name="PAGES",
                style="bold magenta",
                completed=10,
                total=100,
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        text = result.plain
        assert "TRIAGE" not in text
        assert "PAGES" in text

    def test_multiple_rows_separated(self):
        """Multiple rows are separated by newlines."""
        rows = [
            PipelineRowConfig(
                name="TRIAGE",
                style="bold blue",
                completed=50,
                total=100,
                primary_text="page A",
            ),
            PipelineRowConfig(
                name="PAGES",
                style="bold magenta",
                completed=25,
                total=50,
                primary_text="page B",
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        text = result.plain
        assert "TRIAGE" in text
        assert "PAGES" in text
        assert "page A" in text
        assert "page B" in text

    def test_has_content_property(self):
        """has_content reflects primary_text presence."""
        row = PipelineRowConfig(name="TRIAGE", style="bold blue", primary_text="text")
        assert row.has_content is True

        row_empty = PipelineRowConfig(name="TRIAGE", style="bold blue")
        assert row_empty.has_content is False

    def test_rate_displayed(self):
        """Rate appears as N.NN/s (2 decimal places)."""
        rows = [
            PipelineRowConfig(
                name="TRIAGE",
                style="bold blue",
                completed=50,
                total=100,
                rate=3.7,
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        assert "3.70/s" in result.plain

    def test_percentage_displayed(self):
        """Percentage is shown when show_pct=True (default)."""
        rows = [
            PipelineRowConfig(
                name="PAGES",
                style="bold magenta",
                completed=25,
                total=100,
            ),
        ]
        result = build_pipeline_section(rows, bar_width=20)
        assert "25%" in result.plain


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

        from imas_codex.discovery.signals.progress import DataProgressDisplay

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
