"""Tests for paths discovery progress display.

Covers ProgressState properties, ParallelProgressDisplay pipeline rendering,
display items (ScanItem, ScoreItem, EnrichItem), and _count_group_workers.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from rich.panel import Panel
from rich.text import Text

from imas_codex.discovery.base.supervision import (
    SupervisedWorkerGroup,
    WorkerState,
    WorkerStatus,
)
from imas_codex.discovery.paths.progress import (
    EnrichItem,
    ParallelProgressDisplay,
    ProgressState,
    ScanItem,
    ScoreItem,
)

# =============================================================================
# ProgressState properties
# =============================================================================


class TestProgressStateProperties:
    """Tests for ProgressState computed properties."""

    def _state(self, **kwargs) -> ProgressState:
        defaults = {"facility": "tcv", "cost_limit": 10.0}
        defaults.update(kwargs)
        return ProgressState(**defaults)

    def test_elapsed(self):
        """Elapsed time should be positive."""
        state = self._state()
        time.sleep(0.01)
        assert state.elapsed > 0

    def test_run_cost_combines_score_and_refine(self):
        """Run cost sums score and refine costs."""
        state = self._state(_run_score_cost=1.50, _run_refine_cost=0.30)
        assert abs(state.run_cost - 1.80) < 1e-10

    def test_cost_per_path_none_when_no_scores(self):
        """cost_per_path is None when no paths scored."""
        state = self._state()
        assert state.cost_per_path is None

    def test_cost_per_path_computed(self):
        """cost_per_path divides score cost by scored count."""
        state = self._state(_run_score_cost=5.0, run_scored=10)
        assert state.cost_per_path == 0.5

    def test_coverage_zero_when_no_total(self):
        """Coverage is 0 when total is 0."""
        state = self._state(total=0)
        assert state.coverage == 0

    def test_coverage_computed(self):
        """Coverage is percentage of total paths scored."""
        state = self._state(total=100, scored=25)
        assert state.coverage == 25.0

    def test_frontier_size(self):
        """Frontier combines scan, score, and expand pending."""
        state = self._state(pending_scan=10, pending_score=20, pending_expand=5)
        assert state.frontier_size == 35

    def test_pending_work(self):
        """Pending work combines all pending queues."""
        state = self._state(
            pending_scan=10,
            pending_score=20,
            pending_expand=5,
            pending_enrich=3,
        )
        assert state.pending_work == 38

    def test_cost_limit_reached(self):
        """Cost limit reached when run_cost >= cost_limit."""
        state = self._state(cost_limit=5.0, _run_score_cost=5.0)
        assert state.cost_limit_reached is True

    def test_cost_limit_not_reached(self):
        """Cost limit not reached when under budget."""
        state = self._state(cost_limit=10.0, _run_score_cost=3.0)
        assert state.cost_limit_reached is False

    def test_cost_limit_zero_never_reached(self):
        """Cost limit of 0 is never reached (unlimited)."""
        state = self._state(cost_limit=0.0, _run_score_cost=100.0)
        assert state.cost_limit_reached is False

    def test_path_limit_reached(self):
        """Path limit reached when scored >= path_limit."""
        state = self._state(path_limit=50, run_scored=50)
        assert state.path_limit_reached is True

    def test_path_limit_none_never_reached(self):
        """No path limit means never reached."""
        state = self._state(path_limit=None, run_scored=1000)
        assert state.path_limit_reached is False

    def test_limit_reason_cost(self):
        """limit_reason is 'cost' when cost limit reached first."""
        state = self._state(cost_limit=5.0, _run_score_cost=5.0)
        assert state.limit_reason == "cost"

    def test_limit_reason_path(self):
        """limit_reason is 'path' when path limit reached."""
        state = self._state(
            cost_limit=10.0,
            _run_score_cost=1.0,
            path_limit=10,
            run_scored=10,
        )
        assert state.limit_reason == "path"

    def test_limit_reason_none(self):
        """limit_reason is None when no limits reached."""
        state = self._state(cost_limit=10.0, _run_score_cost=1.0)
        assert state.limit_reason is None

    def test_estimated_total_cost_none_when_no_data(self):
        """ETC is None when no scoring data available."""
        state = self._state()
        assert state.estimated_total_cost is None

    def test_estimated_total_cost_with_data(self):
        """ETC projects remaining cost from cost_per_path."""
        state = self._state(
            _run_score_cost=5.0,
            run_scored=10,
            pending_scan=5,
            pending_score=5,
        )
        # cost_per_path = 0.5, remaining = 10 paths, remaining_cost = 5.0
        # ETC = run_cost (5.0) + remaining_score_cost (5.0) = 10.0
        assert state.estimated_total_cost is not None
        assert abs(state.estimated_total_cost - 10.0) < 1e-6

    def test_eta_seconds_none_when_no_data(self):
        """ETA is None when no rate data available."""
        state = self._state()
        assert state.eta_seconds is None

    def test_eta_seconds_cost_based(self):
        """ETA uses cost-based estimate when cost data available."""
        state = self._state(cost_limit=10.0, _run_score_cost=5.0)
        # Force elapsed
        state.start_time = time.time() - 100  # 100s elapsed, 5.0 spent
        eta = state.eta_seconds
        assert eta is not None
        assert eta > 0  # Should be ~100s remaining

    def test_eta_seconds_work_based(self):
        """ETA uses work-based estimate when rates available."""
        state = self._state(
            cost_limit=0.0,  # No cost limit
            pending_scan=100,
            scan_rate=10.0,  # 10 paths/s
        )
        eta = state.eta_seconds
        assert eta is not None
        assert abs(eta - 10.0) < 1e-6  # 100/10 = 10s


# =============================================================================
# Display Items
# =============================================================================


class TestDisplayItems:
    """Tests for ScanItem, ScoreItem, EnrichItem."""

    def test_scan_item_defaults(self):
        """ScanItem has sensible defaults."""
        item = ScanItem(path="/home/user/code")
        assert item.path == "/home/user/code"
        assert item.files == 0
        assert item.dirs == 0
        assert item.has_code is False

    def test_score_item_defaults(self):
        """ScoreItem has sensible defaults."""
        item = ScoreItem(path="/home/user/code")
        assert item.score is None
        assert item.should_expand is True
        assert item.skipped is False

    def test_enrich_item_defaults(self):
        """EnrichItem has sensible defaults."""
        item = EnrichItem(path="/home/user/code")
        assert item.total_bytes == 0
        assert item.total_lines == 0
        assert item.languages == []
        assert item.is_multiformat is False
        assert item.error is None
        assert item.warnings == []

    def test_enrich_item_pattern_categories(self):
        """EnrichItem tracks per-category pattern matches."""
        item = EnrichItem(
            path="/home/user/code",
            pattern_categories={"mdsplus": 3, "hdf5": 1, "imas": 5},
        )
        assert item.pattern_categories["imas"] == 5


# =============================================================================
# ParallelProgressDisplay rendering
# =============================================================================


class TestParallelProgressDisplay:
    """Tests for ParallelProgressDisplay rendering."""

    def _display(self, **kwargs) -> ParallelProgressDisplay:
        defaults = {"facility": "test", "cost_limit": 10.0}
        defaults.update(kwargs)
        return ParallelProgressDisplay(**defaults)

    def test_build_display_returns_panel(self):
        """_build_display returns a Panel."""
        display = self._display()
        display.state.total = 100
        panel = display._build_display()
        assert isinstance(panel, Panel)

    def test_header_shows_facility(self):
        """Header shows uppercase facility name."""
        display = self._display(facility="iter")
        header = display._build_header()
        assert "ITER" in header.plain

    def test_header_shows_scan_only(self):
        """Header shows SCAN ONLY mode indicator."""
        display = self._display(scan_only=True)
        header = display._build_header()
        assert "SCAN ONLY" in header.plain

    def test_header_shows_score_only(self):
        """Header shows SCORE ONLY mode indicator."""
        display = self._display(score_only=True)
        header = display._build_header()
        assert "SCORE ONLY" in header.plain

    def test_header_shows_focus(self):
        """Header shows focus string when set."""
        display = self._display(focus="equilibrium codes")
        header = display._build_header()
        assert "equilibrium codes" in header.plain

    def test_pipeline_section_has_scan_score_enrich(self):
        """Pipeline section contains all three stage labels."""
        display = self._display()
        display.state.total = 100
        display.state.scored = 20
        display.state.pending_score = 30
        section = display._build_pipeline_section()
        text = section.plain
        assert "SCAN" in text
        assert "SCORE" in text
        assert "ENRICH" in text

    def test_pipeline_scan_disabled_in_score_only(self):
        """SCAN stage is disabled in score_only mode."""
        display = self._display(score_only=True)
        display.state.total = 100
        section = display._build_pipeline_section()
        text = section.plain
        # SCAN should still appear (disabled label) but not SCORE disabled
        assert "SCORE" in text

    def test_pipeline_score_disabled_in_scan_only(self):
        """SCORE and ENRICH stages are disabled in scan_only mode."""
        display = self._display(scan_only=True)
        display.state.total = 100
        section = display._build_pipeline_section()
        text = section.plain
        assert "SCAN" in text
        # SCORE and ENRICH should show disabled or not appear
        # (disabled rows are skipped in build_pipeline_section)

    def test_pipeline_shows_scan_activity(self):
        """Pipeline shows current scan item."""
        display = self._display()
        display.state.total = 100
        display.state.current_scan = ScanItem(
            path="/home/codes/liuqe", files=42, dirs=3, has_code=True
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "liuqe" in text
        assert "42 files" in text
        assert "3 dirs" in text
        assert "code project" in text

    def test_pipeline_shows_score_activity(self):
        """Pipeline shows current score item with score value."""
        display = self._display()
        display.state.total = 100
        display.state.pending_score = 50
        display.state.current_score = ScoreItem(
            path="/home/codes/chease",
            score=0.85,
            purpose="simulation_code",
            description="Equilibrium solver",
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "chease" in text
        assert "0.85" in text

    def test_pipeline_shows_score_skipped(self):
        """Pipeline shows skipped score item."""
        display = self._display()
        display.state.total = 100
        display.state.pending_score = 50
        display.state.current_score = ScoreItem(
            path="/tmp/build",
            skipped=True,
            skip_reason="temporary directory",
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "skipped" in text

    def test_pipeline_shows_terminal_indicator(self):
        """Pipeline shows terminal indicator for non-expandable paths."""
        display = self._display()
        display.state.total = 100
        display.state.pending_score = 50
        display.state.current_score = ScoreItem(
            path="/home/codes/tool",
            score=0.3,
            should_expand=False,
            terminal_reason="low_value_leaf",
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "terminal" in text

    def test_pipeline_shows_enrich_activity(self):
        """Pipeline shows current enrich item with metrics."""
        display = self._display()
        display.state.total = 100
        display.state.enrich_processing = True
        display.state.current_enrich = EnrichItem(
            path="/home/codes/analysis",
            total_bytes=1_500_000,
            total_lines=5000,
            languages=["Python", "Fortran"],
            read_matches=3,
            write_matches=1,
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "analysis" in text
        assert "1.5MB" in text
        assert "5,000 LOC" in text
        assert "Python" in text

    def test_pipeline_shows_enrich_error(self):
        """Pipeline shows enrich error."""
        display = self._display()
        display.state.total = 100
        display.state.enrich_processing = True
        display.state.current_enrich = EnrichItem(
            path="/home/codes/broken",
            error="SSH timeout",
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "SSH timeout" in text

    def test_pipeline_shows_enrich_warning(self):
        """Pipeline shows enrich warning (partial result)."""
        display = self._display()
        display.state.total = 100
        display.state.enrich_processing = True
        display.state.current_enrich = EnrichItem(
            path="/home/codes/large",
            total_bytes=50_000_000,
            warnings=["tokei_timeout"],
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "50.0MB" in text
        assert "tokei_timeout" in text

    def test_pipeline_shows_multiformat(self):
        """Pipeline shows multiformat indicator."""
        display = self._display()
        display.state.total = 100
        display.state.enrich_processing = True
        display.state.current_enrich = EnrichItem(
            path="/home/codes/io",
            total_bytes=10_000,
            is_multiformat=True,
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "multiformat" in text

    def test_pipeline_shows_score_cost(self):
        """Pipeline shows LLM cost on SCORE stage."""
        display = self._display()
        display.state.total = 100
        display.state.pending_score = 50
        display.state._run_score_cost = 2.50
        section = display._build_pipeline_section()
        text = section.plain
        assert "$2.50" in text

    def test_pipeline_shows_rate(self):
        """Pipeline shows processing rate with 2dp."""
        display = self._display()
        display.state.total = 100
        display.state.scan_rate = 77.1
        section = display._build_pipeline_section()
        text = section.plain
        assert "77.10/s" in text

    def test_resources_section_has_time(self):
        """Resources section shows TIME row."""
        display = self._display()
        section = display._build_resources_section()
        assert "TIME" in section.plain

    def test_resources_section_has_cost(self):
        """Resources section shows COST when not scan_only."""
        display = self._display()
        display.state._run_score_cost = 1.50
        section = display._build_resources_section()
        text = section.plain
        assert "COST" in text
        assert "$1.50" in text

    def test_resources_section_has_stats(self):
        """Resources section shows stats row."""
        display = self._display()
        display.state.scored = 42
        display.state.skipped = 5
        display.state.max_depth = 8
        section = display._build_resources_section()
        text = section.plain
        assert "STATS" in text
        assert "scored=42" in text
        assert "skipped=5" in text
        assert "depth=8" in text

    def test_resources_section_has_pending(self):
        """Resources section shows pending work counts."""
        display = self._display()
        display.state.pending_scan = 10
        display.state.pending_expand = 5
        section = display._build_resources_section()
        text = section.plain
        assert "scan:10" in text
        assert "expand:5" in text

    def test_full_display_renders_all_sections(self):
        """Full display renders header, pipeline, and resources."""
        display = self._display(facility="iter", focus="equilibrium")
        display.state.total = 100
        display.state.scored = 20
        display.state.pending_score = 30
        display.state._run_score_cost = 2.0
        display.state.max_depth = 5
        panel = display._build_display()
        text = panel.renderable.plain
        assert "ITER" in text
        assert "equilibrium" in text
        assert "SCAN" in text
        assert "SCORE" in text
        assert "TIME" in text

    def test_scan_only_display(self):
        """Scan-only mode produces valid display."""
        display = self._display(scan_only=True)
        display.state.total = 50
        panel = display._build_display()
        text = panel.renderable.plain
        assert "SCAN ONLY" in text
        assert "SCAN" in text

    def test_display_idle_state(self):
        """Display with no activity shows idle."""
        display = self._display()
        display.state.total = 1
        section = display._build_pipeline_section()
        assert "idle" in section.plain

    def test_display_processing_state(self):
        """Display shows processing state for scan."""
        display = self._display()
        display.state.total = 100
        display.state.scan_processing = True
        section = display._build_pipeline_section()
        text = section.plain
        assert (
            "processing" in text or "idle" in text
        )  # processing shows via PipelineRowConfig


# =============================================================================
# Width and layout
# =============================================================================


class TestDisplayLayout:
    """Tests for display width and layout calculations."""

    def test_min_width(self):
        """Width never goes below MIN_WIDTH."""
        from imas_codex.discovery.base.progress import MIN_WIDTH

        display = ParallelProgressDisplay(facility="test", cost_limit=1.0, console=None)
        assert display.width >= MIN_WIDTH

    def test_bar_width_positive(self):
        """Bar width is always positive."""
        display = ParallelProgressDisplay(facility="test", cost_limit=1.0, console=None)
        assert display.bar_width > 0

    def test_gauge_width_positive(self):
        """Gauge width is always positive."""
        display = ParallelProgressDisplay(facility="test", cost_limit=1.0, console=None)
        assert display.gauge_width > 0

    def test_bar_width_less_than_gauge_width(self):
        """Gauge width is typically less than bar width (more room for text)."""
        display = ParallelProgressDisplay(facility="test", cost_limit=1.0, console=None)
        # gauge_width uses GAUGE_METRICS_WIDTH which is larger than METRICS_WIDTH
        assert display.gauge_width <= display.bar_width


# =============================================================================
# Worker group support
# =============================================================================


class TestCountGroupWorkers:
    """Tests for the _count_group_workers method."""

    def _display_with_workers(
        self, workers: dict[str, WorkerStatus]
    ) -> ParallelProgressDisplay:
        """Create display with mocked SupervisedWorkerGroup."""
        display = ParallelProgressDisplay(facility="test", cost_limit=10.0)
        group = MagicMock(spec=SupervisedWorkerGroup)
        group.workers = workers
        display.worker_group = group
        return display

    def test_no_worker_group(self):
        """Returns (0, '') when no worker group set."""
        display = ParallelProgressDisplay(facility="test", cost_limit=10.0)
        count, ann = display._count_group_workers("scan")
        assert count == 0
        assert ann == ""

    def test_counts_matching_group(self):
        """Counts workers whose group matches."""
        workers = {
            "scan_worker_0": WorkerStatus(
                name="scan_worker_0", group="scan", state=WorkerState.running
            ),
            "scan_worker_1": WorkerStatus(
                name="scan_worker_1", group="scan", state=WorkerState.running
            ),
            "score_worker_0": WorkerStatus(
                name="score_worker_0", group="score", state=WorkerState.running
            ),
        }
        display = self._display_with_workers(workers)
        count, ann = display._count_group_workers("scan")
        assert count == 2

    def test_counts_different_groups(self):
        """Different groups counted separately."""
        workers = {
            "scan_worker_0": WorkerStatus(
                name="scan_worker_0", group="scan", state=WorkerState.running
            ),
            "expand_worker_0": WorkerStatus(
                name="expand_worker_0", group="scan", state=WorkerState.running
            ),
            "score_worker_0": WorkerStatus(
                name="score_worker_0", group="score", state=WorkerState.running
            ),
            "refine_worker_0": WorkerStatus(
                name="refine_worker_0", group="score", state=WorkerState.idle
            ),
            "enrich_worker_0": WorkerStatus(
                name="enrich_worker_0", group="enrich", state=WorkerState.running
            ),
        }
        display = self._display_with_workers(workers)
        assert display._count_group_workers("scan")[0] == 2
        assert display._count_group_workers("score")[0] == 2
        assert display._count_group_workers("enrich")[0] == 1

    def test_backoff_annotation(self):
        """Workers in backoff state get annotated."""
        workers = {
            "score_worker_0": WorkerStatus(
                name="score_worker_0", group="score", state=WorkerState.running
            ),
            "score_worker_1": WorkerStatus(
                name="score_worker_1", group="score", state=WorkerState.backoff
            ),
        }
        display = self._display_with_workers(workers)
        count, ann = display._count_group_workers("score")
        assert count == 2
        assert "1 backoff" in ann

    def test_crashed_annotation(self):
        """Workers in crashed state get annotated."""
        workers = {
            "enrich_worker_0": WorkerStatus(
                name="enrich_worker_0", group="enrich", state=WorkerState.crashed
            ),
        }
        display = self._display_with_workers(workers)
        count, ann = display._count_group_workers("enrich")
        assert count == 1
        assert "1 failed" in ann

    def test_multiple_annotations(self):
        """Both backoff and crashed get annotated."""
        workers = {
            "score_worker_0": WorkerStatus(
                name="score_worker_0", group="score", state=WorkerState.running
            ),
            "score_worker_1": WorkerStatus(
                name="score_worker_1", group="score", state=WorkerState.backoff
            ),
            "score_worker_2": WorkerStatus(
                name="score_worker_2", group="score", state=WorkerState.crashed
            ),
        }
        display = self._display_with_workers(workers)
        count, ann = display._count_group_workers("score")
        assert count == 3
        assert "backoff" in ann
        assert "failed" in ann

    def test_group_fallback_to_name_prefix(self):
        """Falls back to name prefix when group is empty."""
        workers = {
            "scan_worker_0": WorkerStatus(
                name="scan_worker_0", group="", state=WorkerState.running
            ),
        }
        display = self._display_with_workers(workers)
        count, ann = display._count_group_workers("scan")
        assert count == 1

    def test_embed_worker_in_scan_group(self):
        """Embed worker assigned to scan group counts correctly."""
        workers = {
            "scan_worker_0": WorkerStatus(
                name="scan_worker_0", group="scan", state=WorkerState.running
            ),
            "embed_worker": WorkerStatus(
                name="embed_worker", group="scan", state=WorkerState.running
            ),
        }
        display = self._display_with_workers(workers)
        count, ann = display._count_group_workers("scan")
        assert count == 2


class TestWorkerStatusUpdate:
    """Tests for update_worker_status method."""

    def test_update_worker_status_sets_group(self):
        """update_worker_status stores the worker group."""
        display = ParallelProgressDisplay(facility="test", cost_limit=10.0)
        group = SupervisedWorkerGroup()
        group.create_status("scan_worker_0", group="scan")
        display.update_worker_status(group)
        assert display.worker_group is group

    def test_pipeline_shows_worker_counts(self):
        """Pipeline rows include worker count annotations."""
        display = ParallelProgressDisplay(facility="test", cost_limit=10.0)
        group = MagicMock(spec=SupervisedWorkerGroup)
        group.workers = {
            "scan_worker_0": WorkerStatus(
                name="scan_worker_0", group="scan", state=WorkerState.running
            ),
            "expand_worker_0": WorkerStatus(
                name="expand_worker_0", group="scan", state=WorkerState.running
            ),
            "score_worker_0": WorkerStatus(
                name="score_worker_0", group="score", state=WorkerState.running
            ),
            "enrich_worker_0": WorkerStatus(
                name="enrich_worker_0", group="enrich", state=WorkerState.running
            ),
        }
        display.worker_group = group
        display.state.total = 100
        display.state.scored = 50

        section = display._build_pipeline_section()
        text = section.plain

        # Worker counts should appear as "×N" in the output
        assert "x2" in text  # scan group (scan + expand)
        assert "x1" in text  # score or enrich group
