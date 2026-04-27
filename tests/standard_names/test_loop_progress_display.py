"""Tests for SNLoopProgressDisplay (loop-mode Rich progress).

These tests verify the display rendering without Neo4j — all graph
interactions are stubbed via mock ``accumulated_cost_fn`` callbacks.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from rich.console import Console

from imas_codex.standard_names.progress import (
    _EVENT_RING_SIZE,
    SNLoopProgressDisplay,
)


def _make_display(
    *,
    cost_fn: callable | None = None,
    cost_limit: float = 5.0,
) -> tuple[SNLoopProgressDisplay, Console]:
    """Create a display with a recording console for test assertions."""
    console = Console(record=True, width=100, force_terminal=True)
    display = SNLoopProgressDisplay(
        run_id="abc12def-1234-5678-9abc-def012345678",
        mode="loop",
        target="full",
        cost_limit=cost_limit,
        accumulated_cost_fn=cost_fn,
        console=console,
    )
    return display, console


def _render(display: SNLoopProgressDisplay, console: Console) -> str:
    """Render the display once and return the exported text."""
    panel = display._build_display()
    console.print(panel)
    return console.export_text()


class TestDisplayRendersInitialLayout:
    """Instantiate, call start_run, start_turn, render once, check output."""

    def test_initial_render_before_start(self):
        display, console = _make_display()
        text = _render(display, console)
        # Should show the title
        assert "Standard Name Loop" in text

    def test_render_after_start_run(self):
        display, console = _make_display()
        display.start_run(total_domains=20)
        text = _render(display, console)
        assert "pending=20" in text
        assert "done=0" in text

    def test_render_after_start_turn(self):
        display, console = _make_display()
        display.start_run(total_domains=5)
        display.start_turn(
            domain="equilibrium",
            phase_plan=["reconcile", "generate", "enrich"],
        )
        text = _render(display, console)
        assert "equilibrium" in text
        assert "generate" in text
        assert "enrich" in text
        assert "waiting" in text  # phases should show as waiting initially

    def test_header_shows_target(self):
        display, console = _make_display()
        text = _render(display, console)
        assert "target=full" in text


class TestPhaseLifecycle:
    """Start/update/end a phase and verify the rendered output."""

    def test_phase_active_shows_bar(self):
        display, console = _make_display()
        display.start_run(total_domains=1)
        display.start_turn(
            domain="magnetics",
            phase_plan=["generate", "enrich"],
        )
        display.start_phase("generate", total=100, model="claude-opus-4.6")
        display.update_phase("generate", completed=50, cost=0.25)
        text = _render(display, console)
        # Should show count
        assert "50" in text
        assert "100" in text
        # Should show cost
        assert "$0.25" in text

    def test_phase_completed_shows_checkmark(self):
        display, console = _make_display()
        display.start_run(total_domains=1)
        display.start_turn(
            domain="magnetics",
            phase_plan=["generate"],
        )
        display.start_phase("generate", total=10)
        display.update_phase("generate", completed=10, cost=0.5)
        display.end_phase("generate", status="completed")
        text = _render(display, console)
        assert "✓" in text

    def test_phase_skipped(self):
        display, console = _make_display()
        display.start_run(total_domains=1)
        display.start_turn(
            domain="magnetics",
            phase_plan=["reconcile", "generate"],
        )
        display.end_phase("reconcile", status="skipped")
        text = _render(display, console)
        assert "skipped" in text


class TestEventRingBufferBounded:
    """Push many events and verify only the last N are retained."""

    def test_ring_buffer_size(self):
        display, _console = _make_display()
        for i in range(20):
            display.push_event(
                phase="generate",
                label=f"sn=item_{i}",
                cost=0.01,
            )
        # Ring buffer should contain at most _EVENT_RING_SIZE items
        assert len(display._events) == _EVENT_RING_SIZE
        # The oldest should be the (20 - _EVENT_RING_SIZE)-th item
        oldest = display._events[0]
        assert f"item_{20 - _EVENT_RING_SIZE}" in oldest.label

    def test_events_render_in_display(self):
        display, console = _make_display()
        display.start_run(total_domains=1)
        display.start_turn(
            domain="equilibrium",
            phase_plan=["generate"],
        )
        display.push_event(
            phase="generate",
            label="sn=plasma_density",
            cost=0.0042,
        )
        text = _render(display, console)
        assert "plasma_density" in text
        assert "generate" in text
        assert "$0.0042" in text


class TestCostGaugeCallsCallback:
    """Verify the accumulated_cost_fn is called during rendering."""

    def test_callback_called_on_render(self):
        mock_fn = MagicMock(return_value=1.25)
        display, console = _make_display(cost_fn=mock_fn)
        _render(display, console)
        mock_fn.assert_called()

    def test_callback_value_appears_in_output(self):
        mock_fn = MagicMock(return_value=2.50)
        display, console = _make_display(cost_fn=mock_fn, cost_limit=10.0)
        display.start_run(total_domains=1)
        text = _render(display, console)
        assert "$2.50" in text

    def test_fallback_when_no_callback(self):
        """Without a cost callback, local cost tracking is used."""
        display, console = _make_display(cost_fn=None)
        display.push_event(phase="generate", label="test", cost=0.10)
        display.push_event(phase="generate", label="test2", cost=0.15)
        # Local cost should sum to 0.25
        assert abs(display._local_cost - 0.25) < 1e-6

    def test_callback_exception_falls_back_gracefully(self):
        """If callback raises, the display should not crash."""

        def bad_fn() -> float:
            raise RuntimeError("Graph unavailable")

        display, console = _make_display(cost_fn=bad_fn)
        display.push_event(phase="generate", label="test", cost=0.10)
        # Should not raise
        text = _render(display, console)
        # Should fall back to local cost
        assert "Standard Name Loop" in text


class TestDomainTracking:
    """Verify domain counter updates."""

    def test_end_turn_increments_done(self):
        display, console = _make_display()
        display.start_run(total_domains=3)
        display.start_turn(domain="equilibrium", phase_plan=["generate"])
        display.end_turn(domain="equilibrium")
        assert display._done_domains == 1
        assert display._pending_domains == 2

        display.start_turn(domain="magnetics", phase_plan=["generate"])
        display.end_turn(domain="magnetics")
        assert display._done_domains == 2
        assert display._pending_domains == 1

    def test_domains_appear_in_render(self):
        display, console = _make_display()
        display.start_run(total_domains=10)
        display.start_turn(domain="transport", phase_plan=["generate"])
        display.end_turn(domain="transport")
        display.start_turn(domain="equilibrium", phase_plan=["generate"])
        text = _render(display, console)
        assert "done=1" in text
        assert "current=equilibrium" in text
        assert "pending=9" in text


class TestTickRefresh:
    """Verify tick() and refresh_from_graph() don't crash."""

    def test_tick_refreshes(self):
        display, console = _make_display()
        display.start_run(total_domains=1)
        # tick should not raise
        display.tick()

    def test_refresh_from_graph(self):
        display, console = _make_display()
        # Should not raise even with no live display
        display.refresh_from_graph("sn")


class TestContextManager:
    """Verify __enter__/__exit__ lifecycle."""

    def test_enter_exit_no_crash(self):
        display, console = _make_display()
        with display:
            display.start_run(total_domains=1)
            display.tick()


class TestPrintSummary:
    """Verify print_summary output."""

    def test_summary_shows_run_id(self):
        mock_fn = MagicMock(return_value=1.50)
        display, console = _make_display(cost_fn=mock_fn, cost_limit=5.0)
        display.start_run(total_domains=2)
        display.start_turn(domain="eq", phase_plan=["generate"])
        display.end_turn(domain="eq")
        display.start_turn(domain="mag", phase_plan=["generate"])
        display.end_turn(domain="mag")
        display.print_summary()
        text = console.export_text()
        assert "abc12def" in text
        assert "2 processed" in text
