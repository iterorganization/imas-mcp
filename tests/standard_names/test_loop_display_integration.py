"""Integration tests for SN loop DataDrivenProgressDisplay wiring.

Verifies that SNLoopState exposes WorkerStats, that workers can update
them, and that the DataDrivenProgressDisplay reads them correctly.
Does NOT test Rich rendering (covered by discover display tests).
"""

from __future__ import annotations

from rich.console import Console

from imas_codex.discovery.base.progress import (
    DataDrivenProgressDisplay,
    WorkerStats,
)
from imas_codex.standard_names.progress import (
    SNLoopState,
    build_sn_loop_stages,
)


class TestSNLoopStateHasWorkerStats:
    """SNLoopState must expose WorkerStats for each phase."""

    def test_all_five_stats_fields(self):
        state = SNLoopState()
        for attr in (
            "generate_stats",
            "regen_stats",
            "enrich_stats",
            "review_names_stats",
            "review_docs_stats",
        ):
            val = getattr(state, attr)
            assert isinstance(val, WorkerStats), f"{attr} is not WorkerStats"

    def test_domain_tracking_defaults(self):
        state = SNLoopState()
        assert state.total_domains == 0
        assert state.done_domains == 0
        assert state.current_domain == ""


class TestWorkerStatsUpdates:
    """Workers update processed/cost/stream_queue directly."""

    def test_processed_increment(self):
        state = SNLoopState()
        state.generate_stats.processed += 1
        state.generate_stats.processed += 1
        assert state.generate_stats.processed == 2

    def test_cost_accumulation(self):
        state = SNLoopState()
        state.enrich_stats.cost += 0.10
        state.enrich_stats.cost += 0.15
        assert abs(state.enrich_stats.cost - 0.25) < 1e-6

    def test_stream_queue_push(self):
        state = SNLoopState()
        state.review_names_stats.stream_queue.add(
            [
                {
                    "primary_text": "sn=electron_temperature",
                    "primary_text_style": "white",
                    "description": "batch-0",
                }
            ]
        )
        # Bypass rate limiting for test
        state.review_names_stats.stream_queue.last_pop = 0
        item = state.review_names_stats.stream_queue.pop()
        assert item is not None
        assert item["primary_text"] == "sn=electron_temperature"


class TestDisplayReadsFromState:
    """DataDrivenProgressDisplay reads WorkerStats from SNLoopState."""

    def test_display_tick_drains_queue(self):
        state = SNLoopState()
        stages = build_sn_loop_stages()
        display = DataDrivenProgressDisplay(
            facility="sn",
            cost_limit=5.0,
            stages=stages,
            title_suffix="Standard Name Loop",
        )
        display.set_engine_state(state)

        # Push an item to the generate stream queue
        state.generate_stats.stream_queue.add(
            [
                {
                    "primary_text": "sn=plasma_current",
                    "primary_text_style": "white",
                    "description": "test",
                }
            ]
        )
        # Bypass rate limiting for test
        state.generate_stats.stream_queue.last_pop = 0

        # tick() should drain the queue
        display.tick()
        assert state.generate_stats._current_stream_item is not None
        assert (
            state.generate_stats._current_stream_item["primary_text"]
            == "sn=plasma_current"
        )

    def test_display_renders_without_crash(self):
        """Display must render even with zero-progress state."""
        state = SNLoopState()
        console = Console(record=True, width=100, force_terminal=True)
        stages = build_sn_loop_stages()
        display = DataDrivenProgressDisplay(
            facility="sn",
            cost_limit=5.0,
            stages=stages,
            console=console,
            title_suffix="Standard Name Loop",
        )
        display.set_engine_state(state)

        # Should not raise
        panel = display._build_display()
        console.print(panel)
        text = console.export_text()
        assert "Standard Name Loop" in text

    def test_display_shows_progress(self):
        """After updating stats, display should reflect them."""
        state = SNLoopState()
        state.generate_stats.processed = 42
        state.generate_stats.total = 100
        state.generate_stats.cost = 0.50

        console = Console(record=True, width=100, force_terminal=True)
        stages = build_sn_loop_stages()
        display = DataDrivenProgressDisplay(
            facility="sn",
            cost_limit=5.0,
            stages=stages,
            console=console,
            title_suffix="Standard Name Loop",
        )
        display.set_engine_state(state)

        panel = display._build_display()
        console.print(panel)
        text = console.export_text()
        assert "42" in text
        assert "42%" in text


class TestBuildSNLoopStages:
    """Stage spec builder respects skip flags."""

    def test_default_all_enabled(self):
        stages = build_sn_loop_stages()
        assert len(stages) == 5
        assert all(not s.disabled for s in stages)

    def test_skip_flags(self):
        stages = build_sn_loop_stages(
            skip_generate=True,
            skip_enrich=True,
            skip_review=True,
            skip_regen=True,
        )
        for s in stages:
            assert s.disabled, f"{s.name} should be disabled"

    def test_stage_attrs_match_state(self):
        """stats_attr names must match SNLoopState field names."""
        state = SNLoopState()
        stages = build_sn_loop_stages()
        for stage in stages:
            assert hasattr(state, stage.stats_attr), (
                f"SNLoopState missing {stage.stats_attr}"
            )


class TestSNLoopProgressDisplayDeleted:
    """The old SNLoopProgressDisplay must no longer be importable."""

    def test_import_raises(self):
        import importlib

        mod = importlib.import_module("imas_codex.standard_names.progress")
        assert not hasattr(mod, "SNLoopProgressDisplay")

    def test_old_helpers_removed(self):
        import importlib

        mod = importlib.import_module("imas_codex.standard_names.progress")
        assert not hasattr(mod, "LoopEvent")
        assert not hasattr(mod, "PhaseState")
        assert not hasattr(mod, "_EVENT_RING_SIZE")


class TestDomainTrackingViaStatsCallback:
    """Domain tracking is surfaced through the stats_fn callback."""

    def test_stats_fn_returns_domain_info(self):
        state = SNLoopState()
        state.total_domains = 10
        state.done_domains = 3
        state.current_domain = "equilibrium"

        def stats_fn():
            return [
                ("done", str(state.done_domains), "green"),
                ("current", state.current_domain or "—", "cyan"),
                (
                    "pending",
                    str(max(0, state.total_domains - state.done_domains)),
                    "dim",
                ),
            ]

        result = stats_fn()
        assert result[0] == ("done", "3", "green")
        assert result[1] == ("current", "equilibrium", "cyan")
        assert result[2] == ("pending", "7", "dim")
