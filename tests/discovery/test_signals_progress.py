"""Tests for signals discovery progress display.

Covers DataProgressState properties, DataProgressDisplay pipeline rendering,
display items (ScanItem, EnrichItem, CheckItem), and _count_group_workers.
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
from imas_codex.discovery.signals.progress import (
    CheckItem,
    DataProgressDisplay,
    DataProgressState,
    EnrichItem,
    ScanItem,
)

# =============================================================================
# DataProgressState properties
# =============================================================================


class TestDataProgressStateProperties:
    """Tests for DataProgressState computed properties."""

    def _state(self, **kwargs) -> DataProgressState:
        defaults = {"facility": "tcv", "cost_limit": 10.0}
        defaults.update(kwargs)
        return DataProgressState(**defaults)

    def test_elapsed(self):
        """Elapsed time should be positive."""
        state = self._state()
        time.sleep(0.01)
        assert state.elapsed > 0

    def test_run_cost_from_enrich(self):
        """Run cost equals enrich cost."""
        state = self._state(_run_enrich_cost=2.50)
        assert abs(state.run_cost - 2.50) < 1e-10

    def test_cost_fraction_zero_when_no_limit(self):
        """Cost fraction is 0 when cost_limit is 0."""
        state = self._state(cost_limit=0.0, _run_enrich_cost=5.0)
        assert state.cost_fraction == 0.0

    def test_cost_fraction_computed(self):
        """Cost fraction is current/limit ratio."""
        state = self._state(cost_limit=10.0, _run_enrich_cost=3.0)
        assert abs(state.cost_fraction - 0.3) < 1e-10

    def test_cost_fraction_capped_at_one(self):
        """Cost fraction never exceeds 1.0."""
        state = self._state(cost_limit=5.0, _run_enrich_cost=10.0)
        assert state.cost_fraction == 1.0

    def test_cost_limit_reached(self):
        """Cost limit reached when run_cost >= cost_limit."""
        state = self._state(cost_limit=5.0, _run_enrich_cost=5.0)
        assert state.cost_limit_reached is True

    def test_cost_limit_not_reached(self):
        """Cost limit not reached when under budget."""
        state = self._state(cost_limit=10.0, _run_enrich_cost=3.0)
        assert state.cost_limit_reached is False

    def test_cost_limit_zero_never_reached(self):
        """Cost limit of 0 is never reached."""
        state = self._state(cost_limit=0.0, _run_enrich_cost=100.0)
        assert state.cost_limit_reached is False

    def test_signal_limit_reached(self):
        """Signal limit reached when enriched >= signal_limit."""
        state = self._state(signal_limit=50, run_enriched=50)
        assert state.signal_limit_reached is True

    def test_signal_limit_not_reached(self):
        """Signal limit not reached when under."""
        state = self._state(signal_limit=50, run_enriched=10)
        assert state.signal_limit_reached is False

    def test_signal_limit_none_never_reached(self):
        """No signal limit means never reached."""
        state = self._state(signal_limit=None, run_enriched=1000)
        assert state.signal_limit_reached is False

    def test_signal_limit_zero_never_reached(self):
        """Signal limit of 0 is never reached."""
        state = self._state(signal_limit=0, run_enriched=1000)
        assert state.signal_limit_reached is False

    def test_eta_seconds_none_when_no_data(self):
        """ETA is None when no rate data available."""
        state = self._state()
        assert state.eta_seconds is None

    def test_eta_seconds_cost_based(self):
        """ETA uses cost-based estimate when cost data available."""
        state = self._state(cost_limit=10.0, _run_enrich_cost=5.0)
        state.start_time = time.time() - 100  # 100s elapsed
        eta = state.eta_seconds
        assert eta is not None
        assert eta > 0  # Should be ~100s remaining

    def test_eta_seconds_enrich_rate(self):
        """ETA uses enrich rate when available."""
        state = self._state(
            cost_limit=0.0,  # No cost limit
            pending_enrich=100,
            enrich_rate=10.0,  # 10 signals/s
        )
        eta = state.eta_seconds
        assert eta is not None
        assert abs(eta - 10.0) < 1e-6

    def test_eta_seconds_check_rate(self):
        """ETA uses check rate when available."""
        state = self._state(
            cost_limit=0.0,
            pending_check=50,
            check_rate=5.0,  # 5 signals/s
        )
        eta = state.eta_seconds
        assert eta is not None
        assert abs(eta - 10.0) < 1e-6

    def test_eta_seconds_max_of_pipelines(self):
        """ETA returns maximum across all pipelines."""
        state = self._state(
            cost_limit=0.0,
            pending_enrich=100,
            enrich_rate=10.0,  # 10s ETA
            pending_check=200,
            check_rate=5.0,  # 40s ETA
        )
        eta = state.eta_seconds
        assert eta is not None
        assert abs(eta - 40.0) < 1e-6  # Slowest pipeline wins

    def test_eta_seconds_signal_limit(self):
        """ETA considers signal limit."""
        state = self._state(
            cost_limit=0.0,
            signal_limit=100,
            run_enriched=50,
        )
        state.start_time = time.time() - 50  # 50s for 50 signals = 1/s
        eta = state.eta_seconds
        assert eta is not None
        assert abs(eta - 50.0) < 0.01  # 50 remaining at 1/s


# =============================================================================
# Display Items
# =============================================================================


class TestDisplayItems:
    """Tests for ScanItem, EnrichItem, CheckItem."""

    def test_scan_item_defaults(self):
        """ScanItem has sensible defaults."""
        item = ScanItem(signal_id="tcv:magnetics/bpol_probe/flux")
        assert item.signal_id == "tcv:magnetics/bpol_probe/flux"
        assert item.tree_name is None
        assert item.node_path is None
        assert item.signals_in_tree == 0
        assert item.epoch_phase is None

    def test_scan_item_with_epoch(self):
        """ScanItem tracks epoch detection progress."""
        item = ScanItem(
            signal_id="tcv:magnetics/ip",
            tree_name="magnetics",
            epoch_phase="coarse",
            epoch_shots_scanned=50,
            epoch_total_shots=200,
            epoch_boundaries_found=2,
        )
        assert item.epoch_phase == "coarse"
        assert item.epoch_shots_scanned == 50
        assert item.epoch_boundaries_found == 2

    def test_enrich_item_defaults(self):
        """EnrichItem has sensible defaults."""
        item = EnrichItem(signal_id="tcv:equilibrium/plasma_current")
        assert item.physics_domain is None
        assert item.description == ""

    def test_enrich_item_with_domain(self):
        """EnrichItem tracks physics domain."""
        item = EnrichItem(
            signal_id="tcv:equilibrium/ip",
            physics_domain="equilibrium",
            description="Plasma current from LIUQE",
        )
        assert item.physics_domain == "equilibrium"
        assert "LIUQE" in item.description

    def test_check_item_defaults(self):
        """CheckItem has sensible defaults."""
        item = CheckItem(signal_id="tcv:magnetics/bpol_probe/flux")
        assert item.shot is None
        assert item.success is None
        assert item.error is None

    def test_check_item_success(self):
        """CheckItem tracks successful validation."""
        item = CheckItem(
            signal_id="tcv:magnetics/ip",
            shot=85000,
            success=True,
        )
        assert item.success is True
        assert item.shot == 85000

    def test_check_item_failure(self):
        """CheckItem tracks validation failure."""
        item = CheckItem(
            signal_id="tcv:broken/signal",
            shot=85000,
            success=False,
            error="TreeNNF: node not found",
        )
        assert item.success is False
        assert "TreeNNF" in item.error


# =============================================================================
# _count_group_workers
# =============================================================================


class TestCountGroupWorkers:
    """Tests for the _count_group_workers method."""

    def _display_with_workers(
        self, workers: dict[str, WorkerStatus]
    ) -> DataProgressDisplay:
        """Create display with mocked SupervisedWorkerGroup."""
        display = DataProgressDisplay(facility="test", cost_limit=10.0)
        group = MagicMock(spec=SupervisedWorkerGroup)
        group.workers = workers
        display.worker_group = group
        return display

    def test_no_worker_group(self):
        """Returns (0, '') when no worker group set."""
        display = DataProgressDisplay(facility="test", cost_limit=10.0)
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
            "enrich_worker_0": WorkerStatus(
                name="enrich_worker_0", group="enrich", state=WorkerState.running
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
            "enrich_worker_0": WorkerStatus(
                name="enrich_worker_0", group="enrich", state=WorkerState.running
            ),
            "enrich_worker_1": WorkerStatus(
                name="enrich_worker_1", group="enrich", state=WorkerState.idle
            ),
            "check_worker_0": WorkerStatus(
                name="check_worker_0", group="check", state=WorkerState.running
            ),
        }
        display = self._display_with_workers(workers)
        assert display._count_group_workers("scan")[0] == 1
        assert display._count_group_workers("enrich")[0] == 2
        assert display._count_group_workers("check")[0] == 1

    def test_backoff_annotation(self):
        """Workers in backoff state get annotated."""
        workers = {
            "enrich_worker_0": WorkerStatus(
                name="enrich_worker_0", group="enrich", state=WorkerState.running
            ),
            "enrich_worker_1": WorkerStatus(
                name="enrich_worker_1", group="enrich", state=WorkerState.backoff
            ),
        }
        display = self._display_with_workers(workers)
        count, ann = display._count_group_workers("enrich")
        assert count == 2
        assert "1 backoff" in ann

    def test_crashed_annotation(self):
        """Workers in crashed state get annotated."""
        workers = {
            "check_worker_0": WorkerStatus(
                name="check_worker_0", group="check", state=WorkerState.crashed
            ),
        }
        display = self._display_with_workers(workers)
        count, ann = display._count_group_workers("check")
        assert count == 1
        assert "1 failed" in ann

    def test_multiple_annotations(self):
        """Both backoff and crashed get annotated."""
        workers = {
            "enrich_worker_0": WorkerStatus(
                name="enrich_worker_0", group="enrich", state=WorkerState.running
            ),
            "enrich_worker_1": WorkerStatus(
                name="enrich_worker_1", group="enrich", state=WorkerState.backoff
            ),
            "enrich_worker_2": WorkerStatus(
                name="enrich_worker_2", group="enrich", state=WorkerState.crashed
            ),
        }
        display = self._display_with_workers(workers)
        count, ann = display._count_group_workers("enrich")
        assert count == 3
        assert "1 backoff" in ann
        assert "1 failed" in ann

    def test_no_annotation_when_all_healthy(self):
        """No annotation when all workers running/idle."""
        workers = {
            "scan_worker_0": WorkerStatus(
                name="scan_worker_0", group="scan", state=WorkerState.running
            ),
            "scan_worker_1": WorkerStatus(
                name="scan_worker_1", group="scan", state=WorkerState.idle
            ),
        }
        display = self._display_with_workers(workers)
        count, ann = display._count_group_workers("scan")
        assert count == 2
        assert ann == ""

    def test_group_fallback_from_name(self):
        """Group inferred from worker name when group field is empty."""
        workers = {
            "scan_worker_0": WorkerStatus(
                name="scan_worker_0", group="", state=WorkerState.running
            ),
        }
        display = self._display_with_workers(workers)
        count, ann = display._count_group_workers("scan")
        assert count == 1

    def test_nonexistent_group(self):
        """Unknown group returns (0, '')."""
        workers = {
            "scan_worker_0": WorkerStatus(
                name="scan_worker_0", group="scan", state=WorkerState.running
            ),
        }
        display = self._display_with_workers(workers)
        count, ann = display._count_group_workers("nonexistent")
        assert count == 0
        assert ann == ""


# =============================================================================
# DataProgressDisplay rendering
# =============================================================================


class TestDataProgressDisplay:
    """Tests for DataProgressDisplay rendering."""

    def _display(self, **kwargs) -> DataProgressDisplay:
        defaults = {"facility": "test", "cost_limit": 10.0}
        defaults.update(kwargs)
        return DataProgressDisplay(**defaults)

    def test_build_display_returns_panel(self):
        """_build_display returns a Panel."""
        display = self._display()
        display.state.total_signals = 100
        panel = display._build_display()
        assert isinstance(panel, Panel)

    def test_header_shows_facility_upper(self):
        """Header shows uppercase facility name."""
        display = self._display(facility="iter")
        header = display._build_header()
        assert "ITER" in header.plain

    def test_header_shows_signal_discovery(self):
        """Header contains 'Signal Discovery'."""
        display = self._display()
        header = display._build_header()
        assert "Signal Discovery" in header.plain

    def test_header_shows_scan_only(self):
        """Header shows SCAN ONLY mode."""
        display = self._display(discover_only=True)
        header = display._build_header()
        assert "SCAN ONLY" in header.plain

    def test_header_shows_enrich_only(self):
        """Header shows ENRICH ONLY mode."""
        display = self._display(enrich_only=True)
        header = display._build_header()
        assert "ENRICH ONLY" in header.plain

    def test_header_shows_focus(self):
        """Header shows focus string when set."""
        display = self._display(focus="magnetics")
        header = display._build_header()
        assert "magnetics" in header.plain

    def test_pipeline_section_has_all_stages(self):
        """Pipeline section includes SCAN, ENRICH, CHECK."""
        display = self._display()
        display.state.total_signals = 100
        display.state.signals_enriched = 20
        section = display._build_pipeline_section()
        text = section.plain
        assert "SCAN" in text
        assert "ENRICH" in text
        assert "CHECK" in text

    def test_pipeline_enrich_disabled_in_discover_only(self):
        """ENRICH disabled in discover_only mode."""
        display = self._display(discover_only=True)
        display.state.total_signals = 100
        section = display._build_pipeline_section()
        text = section.plain
        assert "SCAN" in text
        # ENRICH and CHECK disabled (not rendered by build_pipeline_section)

    def test_pipeline_check_disabled_in_enrich_only(self):
        """CHECK disabled in enrich_only mode."""
        display = self._display(enrich_only=True)
        display.state.total_signals = 100
        section = display._build_pipeline_section()
        text = section.plain
        assert "SCAN" in text
        assert "ENRICH" in text

    def test_pipeline_shows_scan_activity(self):
        """Pipeline shows current scan with node path."""
        display = self._display()
        display.state.total_signals = 100
        display.state.current_scan = ScanItem(
            signal_id="tcv:magnetics/ip",
            tree_name="magnetics",
            node_path="\\MAGNETICS::IP",
            signals_in_tree=150,
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "MAGNETICS" in text
        assert "tree=magnetics" in text
        assert "150" in text

    def test_pipeline_shows_epoch_scan_coarse(self):
        """Pipeline shows epoch coarse scan progress."""
        display = self._display()
        display.state.total_signals = 100
        display.state.current_scan = ScanItem(
            signal_id="tcv:magnetics/ip",
            tree_name="magnetics",
            epoch_phase="coarse",
            epoch_shots_scanned=50,
            epoch_total_shots=200,
            epoch_current_shot=45000,
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "coarse scan" in text
        assert "25%" in text
        assert "50/200" in text

    def test_pipeline_shows_epoch_scan_refine(self):
        """Pipeline shows epoch refinement progress."""
        display = self._display()
        display.state.total_signals = 100
        display.state.current_scan = ScanItem(
            signal_id="tcv:magnetics/ip",
            epoch_phase="refine",
            epoch_boundaries_found=3,
            epoch_boundaries_refined=1,
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "refining boundary" in text
        assert "2/3" in text  # refined+1 / found

    def test_pipeline_shows_epoch_build(self):
        """Pipeline shows epoch build phase."""
        display = self._display()
        display.state.total_signals = 100
        display.state.current_scan = ScanItem(
            signal_id="tcv:magnetics/ip",
            epoch_phase="build",
            epoch_boundaries_found=3,
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "building" in text
        assert "3 epochs" in text

    def test_pipeline_shows_enrich_activity(self):
        """Pipeline shows current enrich with domain."""
        display = self._display()
        display.state.total_signals = 100
        display.state.current_enrich = EnrichItem(
            signal_id="tcv:equilibrium/plasma_current",
            physics_domain="equilibrium",
            description="Main plasma current from LIUQE",
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "equilibrium" in text
        assert "plasma_current" in text

    def test_pipeline_shows_check_testing(self):
        """Pipeline shows check item in testing state."""
        display = self._display()
        display.state.total_signals = 100
        display.state.signals_enriched = 50
        display.state.current_check = CheckItem(
            signal_id="tcv:magnetics/bpol",
            shot=85000,
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "shot=85000" in text
        assert "testing" in text

    def test_pipeline_shows_check_success(self):
        """Pipeline shows successful check."""
        display = self._display()
        display.state.total_signals = 100
        display.state.signals_enriched = 50
        display.state.current_check = CheckItem(
            signal_id="tcv:magnetics/bpol",
            shot=85000,
            success=True,
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "success" in text

    def test_pipeline_shows_check_failure(self):
        """Pipeline shows check failure with error."""
        display = self._display()
        display.state.total_signals = 100
        display.state.signals_enriched = 50
        display.state.current_check = CheckItem(
            signal_id="tcv:broken/signal",
            shot=85000,
            success=False,
            error="TreeNNF: node not found",
        )
        section = display._build_pipeline_section()
        text = section.plain
        assert "TreeNNF" in text

    def test_pipeline_shows_enrich_cost(self):
        """Pipeline shows LLM cost on ENRICH stage."""
        display = self._display()
        display.state.total_signals = 100
        display.state._run_enrich_cost = 1.50
        section = display._build_pipeline_section()
        text = section.plain
        assert "$1.50" in text

    def test_pipeline_shows_rate(self):
        """Pipeline shows discover rate with 2dp."""
        display = self._display()
        display.state.total_signals = 100
        display.state.discover_rate = 83.4
        section = display._build_pipeline_section()
        text = section.plain
        assert "83.40/s" in text

    def test_pipeline_worker_count_annotations(self):
        """Pipeline shows worker count from SupervisedWorkerGroup."""
        display = self._display()
        display.state.total_signals = 100

        # Mock the worker group
        group = MagicMock(spec=SupervisedWorkerGroup)
        group.workers = {
            "scan_worker_0": WorkerStatus(
                name="scan_worker_0", group="scan", state=WorkerState.running
            ),
            "enrich_worker_0": WorkerStatus(
                name="enrich_worker_0", group="enrich", state=WorkerState.running
            ),
            "enrich_worker_1": WorkerStatus(
                name="enrich_worker_1", group="enrich", state=WorkerState.running
            ),
        }
        display.worker_group = group

        section = display._build_pipeline_section()
        text = section.plain
        assert "x1" in text  # scan workers
        assert "x2" in text  # enrich workers

    def test_resources_section_has_time(self):
        """Resources section shows TIME."""
        display = self._display()
        section = display._build_resources_section()
        assert "TIME" in section.plain

    def test_resources_section_has_cost(self):
        """Resources section shows COST."""
        display = self._display()
        display.state._run_enrich_cost = 2.00
        section = display._build_resources_section()
        text = section.plain
        assert "COST" in text
        assert "$2.00" in text

    def test_resources_section_has_stats(self):
        """Resources section shows stats."""
        display = self._display()
        display.state.total_signals = 100
        display.state.signals_enriched = 25
        display.state.signals_checked = 10
        section = display._build_resources_section()
        text = section.plain
        assert "STATS" in text
        assert "discovered=100" in text
        assert "enriched=35" in text  # enriched + checked
        assert "checked=10" in text

    def test_resources_section_shows_failed(self):
        """Resources section shows failure count."""
        display = self._display()
        display.state.signals_failed = 3
        section = display._build_resources_section()
        text = section.plain
        assert "failed=3" in text

    def test_resources_section_shows_pending(self):
        """Resources section shows pending work."""
        display = self._display()
        display.state.pending_enrich = 50
        display.state.pending_check = 10
        section = display._build_resources_section()
        text = section.plain
        assert "enrich:50" in text
        assert "check:10" in text

    def test_full_display_all_sections(self):
        """Full display renders header, pipeline, and resources."""
        display = self._display(facility="iter", focus="magnetics")
        display.state.total_signals = 500
        display.state.signals_enriched = 100
        display.state._run_enrich_cost = 3.0
        panel = display._build_display()
        text = panel.renderable.plain
        assert "ITER" in text
        assert "magnetics" in text
        assert "SCAN" in text
        assert "ENRICH" in text
        assert "TIME" in text

    def test_discover_only_display(self):
        """Discover-only mode produces valid display."""
        display = self._display(discover_only=True)
        display.state.total_signals = 200
        panel = display._build_display()
        text = panel.renderable.plain
        assert "SCAN ONLY" in text

    def test_display_idle_state(self):
        """Display with no activity shows idle."""
        display = self._display()
        display.state.total_signals = 1
        section = display._build_pipeline_section()
        assert "idle" in section.plain

    def test_processing_label_scanning(self):
        """Display shows processing label for scan."""
        display = self._display()
        display.state.total_signals = 100
        display.state.scan_processing = True
        display.state.current_tree = "magnetics"
        section = display._build_pipeline_section()
        text = section.plain
        assert "tree=magnetics" in text


# =============================================================================
# Width and layout
# =============================================================================


class TestDisplayLayout:
    """Tests for display width and layout calculations."""

    def test_min_width(self):
        """Width never goes below MIN_WIDTH."""
        from imas_codex.discovery.base.progress import MIN_WIDTH

        display = DataProgressDisplay(facility="test", cost_limit=1.0, console=None)
        assert display.width >= MIN_WIDTH

    def test_bar_width_positive(self):
        """Bar width always positive."""
        display = DataProgressDisplay(facility="test", cost_limit=1.0, console=None)
        assert display.bar_width > 0

    def test_gauge_width_positive(self):
        """Gauge width always positive."""
        display = DataProgressDisplay(facility="test", cost_limit=1.0, console=None)
        assert display.gauge_width > 0


# =============================================================================
# Summary
# =============================================================================


class TestSummary:
    """Tests for print_summary output."""

    def test_build_summary_has_stages(self):
        """Summary includes all stage labels."""
        display = DataProgressDisplay(facility="test", cost_limit=10.0)
        display.state.total_signals = 100
        display.state.signals_enriched = 50
        display.state.signals_checked = 30
        display.state.signals_skipped = 5
        display.state._run_enrich_cost = 2.50
        summary = display._build_summary()
        text = summary.plain
        assert "SCAN" in text
        assert "ENRICH" in text
        assert "CHECK" in text
        assert "USAGE" in text

    def test_build_summary_shows_cost(self):
        """Summary shows cost information."""
        display = DataProgressDisplay(facility="test", cost_limit=10.0)
        display.state._run_enrich_cost = 1.23
        summary = display._build_summary()
        text = summary.plain
        assert "$1.23" in text

    def test_build_summary_shows_accumulated_cost(self):
        """Summary shows accumulated cost when present."""
        display = DataProgressDisplay(facility="test", cost_limit=10.0)
        display.state._run_enrich_cost = 1.00
        display.state.accumulated_cost = 5.00
        summary = display._build_summary()
        text = summary.plain
        assert "cost=$5.00" in text
        assert "session=$1.00" in text
