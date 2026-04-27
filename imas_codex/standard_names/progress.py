"""Progress display for standard name build pipeline.

Contains two display classes:

- :class:`StandardNameProgressDisplay` — 3-stage display for the single-pass
  ``--paths`` pipeline (extract → compose → finalize).
- :class:`SNLoopProgressDisplay` — 7-phase display for the loop-mode
  ``sn run`` pipeline (reconcile → generate → enrich → link → review_names →
  review_docs → regen), wired through the ``run_discovery()`` harness.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.text import Text

from imas_codex.discovery.base.progress import (
    BaseProgressDisplay,
    DataDrivenProgressDisplay,
    ResourceConfig,
    StageDisplaySpec,
    build_resource_section,
    format_time,
    make_bar,
)
from imas_codex.discovery.base.supervision import SupervisedWorkerGroup

logger = logging.getLogger(__name__)

# Phase display metadata: (label, style)
_LOOP_PHASES: list[tuple[str, str]] = [
    ("reconcile", "bold blue"),
    ("generate", "bold magenta"),
    ("enrich", "bold cyan"),
    ("link", "bold green"),
    ("review_names", "bold yellow"),
    ("review_docs", "bold yellow"),
    ("regen", "bold red"),
]

# How many recent events to keep in the ring buffer
_EVENT_RING_SIZE = 10


def build_sn_stages() -> list[StageDisplaySpec]:
    """Build the stage specs for the SN progress display.

    3 rows: EXTRACT → COMPOSE → FINALIZE
    """
    return [
        StageDisplaySpec(
            name="EXTRACT",
            style="bold blue",
            group="extract",
            stats_attr="extract_stats",
            phase_attr="extract_phase",
        ),
        StageDisplaySpec(
            name="COMPOSE",
            style="bold magenta",
            group="compose",
            stats_attr="compose_stats",
            phase_attr="compose_phase",
        ),
        StageDisplaySpec(
            name="FINALIZE",
            style="bold green",
            group="finalize",
            stats_attr="finalize_stats",
        ),
    ]


class StandardNameProgressDisplay(DataDrivenProgressDisplay):
    """Rich progress display for the SN build pipeline.

    Shows 3 phases: Extract → Compose → Finalize
    where Finalize groups validate + consolidate + persist.
    """

    def __init__(
        self,
        source: str = "dd",
        *,
        console: Any | None = None,
        cost_limit: float = 5.0,
        mode_label: str | None = None,
    ):
        stages = build_sn_stages()
        super().__init__(
            facility="sn",
            cost_limit=cost_limit,
            console=console,
            stages=stages,
            title_suffix="Standard Name Build",
            mode_label=mode_label,
        )
        self.source = source

    def on_worker_status(self, group: SupervisedWorkerGroup) -> None:
        """Callback for worker status updates."""
        self.update_worker_status(group)

    def refresh_from_graph(self, facility: str) -> None:
        """Refresh display data from graph (no-op for SN build)."""
        self._refresh()

    def print_summary(self) -> None:
        """Print a brief summary after build completes."""
        if self._engine_state is None:
            return

        lines: list[str] = []
        for label, attr in [
            ("EXTRACT", "extract_stats"),
            ("COMPOSE", "compose_stats"),
            ("FINALIZE", "finalize_stats"),
        ]:
            stats = getattr(self._engine_state, attr, None)
            if stats and stats.processed > 0:
                parts = [f"{label}: {stats.processed:,}"]
                if stats.cost > 0:
                    parts.append(f"${stats.cost:.2f}")
                lines.append("  ".join(parts))

        if lines:
            self.console.print()
            for line in lines:
                self.console.print(f"  {line}")


# ═══════════════════════════════════════════════════════════════════════
# Loop-mode progress display (sn run without --paths)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class LoopEvent:
    """One atomic event in the live stream ring buffer."""

    timestamp: float
    phase: str
    label: str
    cost: float | None = None


@dataclass
class PhaseState:
    """Mutable state for one phase within the current turn."""

    name: str
    total: int = 0
    completed: int = 0
    cost: float = 0.0
    model: str | None = None
    status: str = "pending"  # pending | active | completed | skipped


class SNLoopProgressDisplay(BaseProgressDisplay):
    """Rich progress display for the SN loop pipeline.

    Shows 7 phases per turn: RECONCILE → GENERATE → ENRICH → LINK →
    REVIEW_NAMES → REVIEW_DOCS → REGEN.  Designed for ``sn run`` without
    ``--paths`` — the multi-domain completion loop.

    Layout::

        ┌─ SN Standard Name Loop · target=full · cost-cap=$5.00 ──────┐
        │ Cost:  $0.42 / $5.00  ████░░░░  8% │ Tokens: …             │
        │ Run:   abc12def · elapsed 00:02:15 · domain=equilibrium     │
        │                                                              │
        │ ┌─ Phases ──────────────────────────────────────────────────┐│
        │ │ generate     ████████░  72%  142/197  $0.18               ││
        │ │ enrich       ██░░░░░░  18%   35/192  $0.12               ││
        │ │ review_names waiting                                      ││
        │ └──────────────────────────────────────────────────────────┘│
        │                                                              │
        │ Latest: [14:34:01] enrich · sn=plasma_pressure · $0.004    │
        │                                                              │
        │ Domains: done=2  current=equilibrium  pending=18            │
        └─────────────────────────────────────────────────────────────┘

    Workers push events via :meth:`push_event` and phase progress via
    :meth:`start_phase` / :meth:`update_phase` / :meth:`end_phase`.
    The display is driven by the ``run_discovery()`` ticker which calls
    :meth:`tick` periodically.
    """

    def __init__(
        self,
        run_id: str,
        mode: str,
        target: str,
        cost_limit: float,
        *,
        accumulated_cost_fn: Callable[[], float] | None = None,
        console: Any | None = None,
    ) -> None:
        super().__init__(
            facility="sn",
            cost_limit=cost_limit,
            console=console,
            title_suffix="Standard Name Loop",
        )
        self.run_id = run_id
        self.mode = mode
        self.target = target
        self._accumulated_cost_fn = accumulated_cost_fn

        # Domain tracking
        self._total_domains: int = 0
        self._done_domains: int = 0
        self._current_domain: str | None = None
        self._pending_domains: int = 0

        # Phase tracking (reset each turn)
        self._phases: dict[str, PhaseState] = {}
        self._phase_plan: list[str] = []

        # Event ring buffer
        self._events: deque[LoopEvent] = deque(maxlen=_EVENT_RING_SIZE)

        # Cumulative cost from push_event calls (local tracking)
        self._local_cost: float = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────

    def start_run(self, *, total_domains: int) -> None:
        """Signal the start of the loop run."""
        self._total_domains = total_domains
        self._pending_domains = total_domains
        self._refresh()

    def start_turn(self, *, domain: str, phase_plan: list[str]) -> None:
        """Signal the start of a new turn on *domain*."""
        self._current_domain = domain
        self._phase_plan = phase_plan
        self._phases = {name: PhaseState(name=name) for name in phase_plan}
        self._refresh()

    def end_turn(self, *, domain: str) -> None:
        """Signal the end of a turn — advance domain counters."""
        self._done_domains += 1
        self._pending_domains = max(0, self._total_domains - self._done_domains)
        self._refresh()

    # ── Phase tracking ─────────────────────────────────────────────

    def start_phase(
        self, phase: str, *, total: int = 0, model: str | None = None
    ) -> None:
        """Mark a phase as active."""
        ps = self._phases.get(phase)
        if ps is None:
            ps = PhaseState(name=phase)
            self._phases[phase] = ps
        ps.status = "active"
        ps.total = total
        ps.completed = 0
        ps.model = model
        self._refresh()

    def update_phase(
        self, phase: str, *, completed: int, cost: float | None = None
    ) -> None:
        """Update progress counters for a phase."""
        ps = self._phases.get(phase)
        if ps is None:
            return
        ps.completed = completed
        if cost is not None:
            ps.cost = cost
        # no _refresh() here — let the ticker handle it for smoothness

    def end_phase(self, phase: str, *, status: str = "completed") -> None:
        """Mark a phase as completed."""
        ps = self._phases.get(phase)
        if ps is None:
            return
        ps.status = status
        self._refresh()

    # ── Live event stream ──────────────────────────────────────────

    def push_event(self, *, phase: str, label: str, cost: float | None = None) -> None:
        """Push an atomic event into the ring buffer.

        Called by workers after each LLM call or significant action.
        """
        self._events.append(
            LoopEvent(
                timestamp=time.time(),
                phase=phase,
                label=label,
                cost=cost,
            )
        )
        if cost is not None:
            self._local_cost += cost

    # ── Display rendering ──────────────────────────────────────────

    def _header_mode_label(self) -> str | None:
        return f"target={self.target}"

    def _get_cost(self) -> float:
        """Get current accumulated cost (graph-backed or local fallback)."""
        if self._accumulated_cost_fn is not None:
            try:
                return self._accumulated_cost_fn()
            except Exception:
                pass
        return self._local_cost

    def _build_pipeline_section(self) -> Text:
        """Build phase rows for the current turn."""
        section = Text()

        if not self._phases:
            section.append("  No active turn", style="dim italic")
            return section

        # Build a row for each phase in the plan
        for i, phase_name in enumerate(self._phase_plan):
            ps = self._phases.get(phase_name)
            if ps is None:
                continue
            if i > 0:
                section.append("\n")

            label = f"  {phase_name:<15}"

            if ps.status == "pending":
                section.append(label, style="dim")
                section.append("waiting", style="dim italic")
            elif ps.status == "skipped":
                section.append(label, style="dim")
                section.append("skipped", style="dim italic")
            elif ps.status in ("active", "completed"):
                style = dict(_LOOP_PHASES).get(phase_name, "bold white")
                section.append(label, style=style)

                # Progress bar
                total = max(ps.total, 1)
                ratio = ps.completed / total if ps.total > 0 else 0.0
                bar_w = min(self.bar_width, 24)
                section.append(make_bar(ratio, bar_w))
                section.append("  ")

                # Count and percentage
                if ps.total > 0:
                    pct = int(ratio * 100)
                    section.append(
                        f"{ps.completed:>4}/{ps.total:<4}  {pct:>3}%",
                        style="bold" if ps.status == "active" else "dim",
                    )
                elif ps.status == "completed":
                    section.append(
                        f"{ps.completed:>4}",
                        style="dim",
                    )

                # Cost
                if ps.cost > 0:
                    section.append(f"  ${ps.cost:.4f}", style="yellow")

                # Model
                if ps.model and ps.status == "active":
                    section.append(f"  {ps.model}", style="dim")

                # Completion marker
                if ps.status == "completed":
                    section.append("  ✓", style="green")

        return section

    def _build_events_section(self) -> Text:
        """Build the latest events section from the ring buffer."""
        section = Text()
        if not self._events:
            return section

        section.append("  LATEST", style="bold white")
        section.append("\n")

        # Show most recent events (newest first)
        events = list(self._events)
        for evt in reversed(events[-5:]):
            ts = time.strftime("%H:%M:%S", time.localtime(evt.timestamp))
            section.append(f"    [{ts}] ", style="dim")
            section.append(f"{evt.phase}", style="cyan")
            section.append(f" · {evt.label}", style="white")
            if evt.cost is not None:
                section.append(f" · ${evt.cost:.4f}", style="yellow")
            section.append("\n")

        return section

    def _build_domains_section(self) -> Text:
        """Build the domains tracker row."""
        section = Text()
        section.append("  DOMAINS", style="bold green")
        section.append(f"  done={self._done_domains}", style="green")
        if self._current_domain:
            section.append(f"  current={self._current_domain}", style="cyan bold")
        section.append(f"  pending={self._pending_domains}", style="dim")
        return section

    def _build_resources_section(self) -> Text:
        """Build resource section with cost gauge and timing."""
        cost = self._get_cost()

        config = ResourceConfig(
            elapsed=self.elapsed,
            run_cost=cost if cost > 0 else None,
            cost_limit=self.cost_limit if self.cost_limit > 0 else None,
            accumulated_cost=cost,
            stats=[
                ("run", self.run_id[:8], "dim"),
            ],
        )
        return build_resource_section(config, self.gauge_width)

    def _build_display(self):
        """Assemble the full panel layout."""
        from rich.panel import Panel

        sections: list[Text] = [self._build_header()]

        # Servers section (optional)
        servers = self._build_servers_section()
        if servers is not None:
            sections.append(Text("─" * (self.width - 4), style="dim"))
            sections.append(servers)

        # Pipeline (phases) section
        sections.append(Text("─" * (self.width - 4), style="dim"))
        sections.append(self._build_pipeline_section())

        # Events section
        events = self._build_events_section()
        if events.plain:
            sections.append(Text("─" * (self.width - 4), style="dim"))
            sections.append(events)

        # Domains tracker
        sections.append(Text("─" * (self.width - 4), style="dim"))
        sections.append(self._build_domains_section())

        # Resources (time, cost)
        sections.append(Text("─" * (self.width - 4), style="dim"))
        sections.append(self._build_resources_section())

        # Shutdown section
        if self._shutting_down:
            sections.append(Text("─" * (self.width - 4), style="dim"))
            sections.append(self._build_shutdown_section())

        content = Text()
        for i, section in enumerate(sections):
            if i > 0:
                content.append("\n")
            content.append_text(section)

        return Panel(
            content,
            border_style="yellow" if self._shutting_down else "cyan",
            width=self.width,
            padding=(0, 1),
        )

    def tick(self) -> None:
        """Periodic refresh driven by the harness ticker."""
        self._refresh()

    def refresh_from_graph(self, facility: str) -> None:
        """Called by the harness graph-refresh task."""
        self._refresh()

    def print_summary(self) -> None:
        """Print summary after the loop completes."""
        cost = self._get_cost()
        self.console.print()
        self.console.print(
            f"  Run {self.run_id[:8]}… completed in {format_time(self.elapsed)}"
        )
        if cost > 0:
            self.console.print(f"  Cost: ${cost:.2f} / ${self.cost_limit:.2f}")
        self.console.print(f"  Domains: {self._done_domains} processed")
