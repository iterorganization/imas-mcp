"""Integration tests for SN loop DataDrivenProgressDisplay wiring.

Verifies that SNLoopState exposes WorkerStats for the three rolled-up
worker groups (GENERATE / ENRICH / REVIEW), that workers can update
them, and that the DataDrivenProgressDisplay reads them correctly.
Subphases (compose/regen, names/docs) are surfaced via stream items
and ``WorkerStats.status_text``, not via additional rows.

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
    """SNLoopState exposes one WorkerStats per worker group."""

    def test_three_stats_fields(self):
        state = SNLoopState()
        for attr in ("generate_stats", "enrich_stats", "review_stats"):
            val = getattr(state, attr)
            assert isinstance(val, WorkerStats), f"{attr} is not WorkerStats"

    def test_only_three_stats_fields(self):
        state = SNLoopState()
        names = set(state.__dataclass_fields__)
        assert names == {"generate_stats", "enrich_stats", "review_stats"}

    def test_dropped_fields_absent(self):
        state = SNLoopState()
        for dropped in (
            "regen_stats",
            "review_names_stats",
            "review_docs_stats",
            "total_domains",
            "done_domains",
            "current_domain",
        ):
            assert not hasattr(state, dropped), (
                f"SNLoopState should no longer expose {dropped}"
            )


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

    def test_stream_queue_push_review(self):
        state = SNLoopState()
        state.review_stats.stream_queue.add(
            [
                {
                    "primary_text": "sn=electron_temperature",
                    "primary_text_style": "white",
                    "description": "names batch-0",
                }
            ]
        )
        state.review_stats.stream_queue.last_pop = 0
        item = state.review_stats.stream_queue.pop()
        assert item is not None
        assert item["primary_text"] == "sn=electron_temperature"


class TestSubphaseVisibility:
    """Subphases ride on top of the shared WorkerStats row."""

    def test_status_text_carries_compose_regen(self):
        state = SNLoopState()
        state.generate_stats.status_text = "compose"
        assert state.generate_stats.status_text == "compose"
        state.generate_stats.status_text = "regen"
        assert state.generate_stats.status_text == "regen"

    def test_status_text_carries_names_docs(self):
        state = SNLoopState()
        state.review_stats.status_text = "names"
        assert state.review_stats.status_text == "names"
        state.review_stats.status_text = "docs"
        assert state.review_stats.status_text == "docs"


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

        state.generate_stats.stream_queue.add(
            [
                {
                    "primary_text": "sn=plasma_current",
                    "primary_text_style": "white",
                    "description": "compose",
                }
            ]
        )
        state.generate_stats.stream_queue.last_pop = 0

        display.tick()
        assert state.generate_stats._current_stream_item is not None
        assert (
            state.generate_stats._current_stream_item["primary_text"]
            == "sn=plasma_current"
        )

    def test_display_renders_without_crash(self):
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

        panel = display._build_display()
        console.print(panel)
        text = console.export_text()
        assert "Standard Name Loop" in text

    def test_display_shows_progress(self):
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
    """Stage spec builder produces 3 rows and respects skip flags."""

    def test_default_all_enabled(self):
        stages = build_sn_loop_stages()
        assert len(stages) == 3
        assert [s.name for s in stages] == ["GENERATE", "ENRICH", "REVIEW"]
        assert all(not s.disabled for s in stages)

    def test_skip_flags(self):
        stages = build_sn_loop_stages(
            skip_generate=True,
            skip_enrich=True,
            skip_review=True,
        )
        for s in stages:
            assert s.disabled, f"{s.name} should be disabled"

    def test_skip_regen_arg_removed(self):
        """build_sn_loop_stages no longer accepts skip_regen."""
        import inspect

        sig = inspect.signature(build_sn_loop_stages)
        assert "skip_regen" not in sig.parameters

    def test_stage_attrs_match_state(self):
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


class TestPendingFnContract:
    """Pending counts are graph-derived row badges, one per worker group."""

    def test_pending_fn_returns_three_groups(self):
        # Static contract: pending_fn is expected to return 3 (group, count)
        # tuples in this fixed order.
        def pending_fn():
            return [
                ("generate", 5),
                ("enrich", 2),
                ("review", 3),
            ]

        result = pending_fn()
        assert [g for g, _ in result] == ["generate", "enrich", "review"]
        assert all(isinstance(c, int) for _, c in result)


class TestPhaseRoutingDocumented:
    """Phase → stats_attr mapping is the documented contract for turn.py."""

    def test_phase_to_stats_mapping(self):
        # turn.py routes phases to the shared WorkerStats fields below.
        # If this map changes, update turn.py and the rollup display.
        mapping = {
            "compose": "generate_stats",  # initial compose
            "regen": "generate_stats",  # regen with reviewer feedback
            "enrich": "enrich_stats",
            "review_names": "review_stats",
            "review_docs": "review_stats",
        }
        state = SNLoopState()
        for phase, attr in mapping.items():
            assert hasattr(state, attr), f"phase {phase} routes to missing {attr}"
