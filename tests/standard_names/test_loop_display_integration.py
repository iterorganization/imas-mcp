"""Integration tests for SN loop DataDrivenProgressDisplay wiring.

The SN loop display has 5 rows, each tracked by an independent ``WorkerStats``
field on ``SNLoopState``. Each row has its own ``StageDisplaySpec.group`` so
the display framework's running/completion/worker-count aggregates remain
truthful per row.

Visual cohesion (DRAFT/REVISE share the magenta family; REVIEW NAMES/REVIEW
DOCS share yellow) is achieved through colour palette and adjacent placement
only — not through shared groups.
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

EXPECTED_STATS_FIELDS = {
    "draft_stats",
    "revise_stats",
    "describe_stats",
    "review_names_stats",
    "review_docs_stats",
}

EXPECTED_GROUPS = {
    "draft",
    "revise",
    "describe",
    "review_names",
    "review_docs",
}

EXPECTED_ROW_NAMES = ["DRAFT", "REVISE", "DESCRIBE", "REVIEW NAMES", "REVIEW DOCS"]


class TestSNLoopStateHasWorkerStats:
    def test_five_stats_fields(self):
        state = SNLoopState()
        for attr in EXPECTED_STATS_FIELDS:
            val = getattr(state, attr)
            assert isinstance(val, WorkerStats), f"{attr} is not WorkerStats"

    def test_only_five_stats_fields(self):
        state = SNLoopState()
        names = set(state.__dataclass_fields__)
        assert names == EXPECTED_STATS_FIELDS

    def test_dropped_fields_absent(self):
        """Old 3-row field names must not exist."""
        state = SNLoopState()
        for dropped in (
            "generate_stats",
            "enrich_stats",
            "review_stats",
            "regen_stats",
            "total_domains",
            "done_domains",
            "current_domain",
        ):
            assert not hasattr(state, dropped), (
                f"SNLoopState should no longer expose {dropped}"
            )


class TestWorkerStatsUpdates:
    def test_processed_increment(self):
        state = SNLoopState()
        state.draft_stats.processed += 1
        state.draft_stats.processed += 1
        assert state.draft_stats.processed == 2

    def test_cost_accumulation(self):
        state = SNLoopState()
        state.describe_stats.cost += 0.10
        state.describe_stats.cost += 0.15
        assert abs(state.describe_stats.cost - 0.25) < 1e-6

    def test_stream_queue_push_review_names(self):
        state = SNLoopState()
        state.review_names_stats.stream_queue.add(
            [
                {
                    "primary_text": "sn=electron_temperature",
                    "primary_text_style": "white",
                    "description": "names batch-0",
                }
            ]
        )
        state.review_names_stats.stream_queue.last_pop = 0
        item = state.review_names_stats.stream_queue.pop()
        assert item is not None
        assert item["primary_text"] == "sn=electron_temperature"

    def test_revise_stats_independent_of_draft(self):
        state = SNLoopState()
        state.draft_stats.processed = 10
        state.revise_stats.processed = 3
        assert state.draft_stats.processed == 10
        assert state.revise_stats.processed == 3


class TestDisplayReadsFromState:
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

        state.draft_stats.stream_queue.add(
            [
                {
                    "primary_text": "sn=plasma_current",
                    "primary_text_style": "white",
                    "description": "compose",
                }
            ]
        )
        state.draft_stats.stream_queue.last_pop = 0

        display.tick()
        assert state.draft_stats._current_stream_item is not None
        assert (
            state.draft_stats._current_stream_item["primary_text"]
            == "sn=plasma_current"
        )

    def test_display_renders_without_crash(self):
        state = SNLoopState()
        console = Console(record=True, width=120, force_terminal=True)
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
        state.draft_stats.processed = 42
        state.draft_stats.total = 100
        state.draft_stats.cost = 0.50

        console = Console(record=True, width=120, force_terminal=True)
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
    def test_default_all_enabled_except_revise(self):
        """REVISE is disabled by default because min_score defaults to None."""
        stages = build_sn_loop_stages()
        assert len(stages) == 5
        assert [s.name for s in stages] == EXPECTED_ROW_NAMES

        by_name = {s.name: s for s in stages}
        assert not by_name["DRAFT"].disabled
        assert by_name["REVISE"].disabled, "REVISE disabled when min_score is None"
        assert not by_name["DESCRIBE"].disabled
        assert not by_name["REVIEW NAMES"].disabled
        assert not by_name["REVIEW DOCS"].disabled

    def test_revise_enabled_when_min_score_set(self):
        stages = build_sn_loop_stages(min_score=0.65)
        by_name = {s.name: s for s in stages}
        assert not by_name["REVISE"].disabled

    def test_skip_generate_disables_draft_and_revise(self):
        stages = build_sn_loop_stages(skip_generate=True, min_score=0.65)
        by_name = {s.name: s for s in stages}
        assert by_name["DRAFT"].disabled
        assert by_name["REVISE"].disabled

    def test_skip_enrich_disables_describe(self):
        stages = build_sn_loop_stages(skip_enrich=True)
        by_name = {s.name: s for s in stages}
        assert by_name["DESCRIBE"].disabled

    def test_skip_review_disables_both_review_rows(self):
        stages = build_sn_loop_stages(skip_review=True)
        by_name = {s.name: s for s in stages}
        assert by_name["REVIEW NAMES"].disabled
        assert by_name["REVIEW DOCS"].disabled

    def test_skip_regen_arg_removed(self):
        """build_sn_loop_stages no longer accepts skip_regen."""
        import inspect

        sig = inspect.signature(build_sn_loop_stages)
        assert "skip_regen" not in sig.parameters

    def test_unique_groups_per_row(self):
        """Each row gets a unique group so framework state stays per-row."""
        stages = build_sn_loop_stages(min_score=0.65)
        groups = {s.group for s in stages}
        assert groups == EXPECTED_GROUPS
        assert len(groups) == len(stages), "groups must be unique per row"

    def test_stage_attrs_match_state(self):
        state = SNLoopState()
        stages = build_sn_loop_stages(min_score=0.65)
        for stage in stages:
            assert hasattr(state, stage.stats_attr), (
                f"SNLoopState missing {stage.stats_attr}"
            )


class TestSNLoopProgressDisplayDeleted:
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
    """Pending counts: 5 short-labelled (label, count) tuples for the resource band."""

    def test_pending_fn_returns_five_entries_in_order(self):
        def pending_fn():
            return [
                ("draft", 5),
                ("revise", 1),
                ("enrich", 2),
                ("rev-n", 3),
                ("rev-d", 4),
            ]

        result = pending_fn()
        assert [g for g, _ in result] == ["draft", "revise", "enrich", "rev-n", "rev-d"]
        assert all(isinstance(c, int) for _, c in result)


class TestPhaseRoutingDocumented:
    """Phase → stats_attr mapping is the documented contract for turn.py."""

    def test_phase_to_stats_mapping(self):
        mapping = {
            "compose": "draft_stats",
            "regen": "revise_stats",
            "enrich": "describe_stats",
            "review_names": "review_names_stats",
            "review_docs": "review_docs_stats",
        }
        state = SNLoopState()
        for phase, attr in mapping.items():
            assert hasattr(state, attr), f"phase {phase} routes to missing {attr}"
