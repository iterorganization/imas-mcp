"""
Progress display for parallel data signal discovery.

Design principles (matching wiki and paths progress displays):
- Clean hierarchy: Target → Progress → Activity → Resources
- Gradient progress bars with percentage
- Resource gauges for time and cost budgets
- Minimal visual clutter (no emojis, thin progress bars)

Display layout: WORKERS → PROGRESS → ACTIVITY → RESOURCES
- SCAN: MDSplus tree traversal, TDI introspection
- ENRICH: LLM classification of physics domain, description
- VALIDATE: Test data access, verify units/sign

Uses common progress infrastructure from base.progress module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from imas_codex.discovery.base.progress import (
    StreamQueue,
    clean_text,
    clip_text,
    format_time,
    make_bar,
    make_resource_gauge,
)
from imas_codex.discovery.base.supervision import (
    SupervisedWorkerGroup,
    WorkerState,
)

if TYPE_CHECKING:
    from imas_codex.discovery.base.progress import WorkerStats


# =============================================================================
# Display Items
# =============================================================================


@dataclass
class ScanItem:
    """Current scan activity."""

    signal_id: str
    tree_name: str | None = None
    node_path: str | None = None
    signals_in_tree: int = 0  # Count of signals discovered in current tree
    # Epoch detection progress
    epoch_phase: str | None = None  # "coarse", "refine", "build"
    epoch_current_shot: int | None = None
    epoch_shots_scanned: int = 0
    epoch_total_shots: int = 0
    epoch_boundaries_found: int = 0
    epoch_boundaries_refined: int = 0


@dataclass
class EnrichItem:
    """Current enrich activity."""

    signal_id: str
    physics_domain: str | None = None
    description: str = ""


@dataclass
class ValidateItem:
    """Current validate activity."""

    signal_id: str
    shot: int | None = None
    success: bool | None = None
    error: str | None = None
    physics_domain: str | None = None  # For display on second line


# =============================================================================
# Progress State
# =============================================================================


@dataclass
class DataProgressState:
    """All state for the data progress display."""

    facility: str
    cost_limit: float
    signal_limit: int | None = None
    focus: str = ""

    # Mode flags
    discover_only: bool = False
    enrich_only: bool = False

    # Worker group for status tracking
    worker_group: SupervisedWorkerGroup | None = None

    # Counts from graph
    total_signals: int = 0  # All FacilitySignal nodes for this facility
    signals_discovered: int = 0
    signals_enriched: int = 0
    signals_validated: int = 0
    signals_skipped: int = 0
    signals_failed: int = 0

    # Pending work counts
    pending_enrich: int = 0
    pending_validate: int = 0

    # This run stats
    run_discovered: int = 0
    run_enriched: int = 0
    run_validated: int = 0
    _run_enrich_cost: float = 0.0
    discover_rate: float | None = None
    enrich_rate: float | None = None
    validate_rate: float | None = None

    # Accumulated facility cost (from graph)
    accumulated_cost: float = 0.0

    # Current items
    current_scan: ScanItem | None = None
    current_enrich: EnrichItem | None = None
    current_validate: ValidateItem | None = None
    scan_processing: bool = False
    current_tree: str | None = None  # Currently scanning tree
    enrich_processing: bool = False
    validate_processing: bool = False

    # Streaming queues
    scan_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=2.0, max_rate=5.0, min_display_time=0.3
        )
    )
    enrich_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.0, min_display_time=0.4
        )
    )
    validate_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.0, min_display_time=0.4
        )
    )

    # Tracking
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def run_cost(self) -> float:
        return self._run_enrich_cost

    @property
    def cost_fraction(self) -> float:
        if self.cost_limit <= 0:
            return 0.0
        return min(1.0, self.run_cost / self.cost_limit)

    @property
    def cost_limit_reached(self) -> bool:
        return self.run_cost >= self.cost_limit if self.cost_limit > 0 else False

    @property
    def signal_limit_reached(self) -> bool:
        if self.signal_limit is None or self.signal_limit <= 0:
            return False
        return self.run_enriched >= self.signal_limit

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time to termination."""
        if self.run_cost > 0 and self.cost_limit > 0:
            cost_rate = self.run_cost / self.elapsed if self.elapsed > 0 else 0
            if cost_rate > 0:
                remaining_budget = self.cost_limit - self.run_cost
                return max(0, remaining_budget / cost_rate)

        if self.signal_limit is not None and self.signal_limit > 0:
            if self.run_enriched > 0 and self.elapsed > 0:
                rate = self.run_enriched / self.elapsed
                remaining = self.signal_limit - self.run_enriched
                return max(0, remaining / rate) if rate > 0 else None

        if not self.enrich_rate or self.enrich_rate <= 0:
            return None
        remaining = self.pending_enrich
        return remaining / self.enrich_rate if remaining > 0 else 0


# =============================================================================
# Main Display Class
# =============================================================================


class DataProgressDisplay:
    """Clean progress display for parallel signal discovery.

    Layout (100 chars wide):
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                               TCV Signal Discovery                                               │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  WORKERS  scan:1 (running)  enrich:2 (1 active)  validate:1                                      │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  SCAN   ━━━━━━━━━━━━━━━━━━━━━━━━━━━────────────────────────────   2944        83.4/s             │
    │  ENRICH ━━━━━━━━━━━━━━━━━━━━━━━━──────────────────────────────       5   0%    0.0/s             │
    │  VALIDATE━━━━━━━━─────────────────────────────────────────────       0   0%    0.0/s             │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  SCAN   \\HYBRID::PID_I                                                                          │
    │          tree=hybrid  2944 signals discovered                                                    │
    │  ENRICH tcv:equilibrium/plasma_current                                                           │
    │          equilibrium  Main plasma current from LIUQE equilibrium code                            │
    │  VALIDATE shot=85000 testing...                                                                  │
    │          tcv:equilibrium/elongation                                                              │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  TIME   │━━━━━━━━━━━━━━━━━━━━━━│  6m 50s                                                         │
    │  COST   │━━━━━━━━│             │  $0.00 / $0.20                                                   │
    │  STATS  discovered=2944  enriched=5  validated=0  pending=[enrich:2944 validate:5]               │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
    """

    WIDTH = 100
    BAR_WIDTH = 48
    GAUGE_WIDTH = 24

    def __init__(
        self,
        facility: str,
        cost_limit: float,
        signal_limit: int | None = None,
        focus: str = "",
        console: Console | None = None,
        discover_only: bool = False,
        enrich_only: bool = False,
    ) -> None:
        self.console = console or Console()
        self.state = DataProgressState(
            facility=facility,
            cost_limit=cost_limit,
            signal_limit=signal_limit,
            focus=focus,
            discover_only=discover_only,
            enrich_only=enrich_only,
        )
        self._live: Live | None = None

    def _build_header(self) -> Text:
        """Build centered header with facility and focus."""
        header = Text()

        title = f"{self.state.facility.upper()} Signal Discovery"
        if self.state.discover_only:
            title += " (SCAN ONLY)"
        elif self.state.enrich_only:
            title += " (ENRICH ONLY)"
        header.append(title.center(self.WIDTH - 4), style="bold cyan")

        if self.state.focus:
            header.append("\n")
            focus_line = f"Focus: {self.state.focus}"
            header.append(focus_line.center(self.WIDTH - 4), style="italic dim")

        return header

    def _build_worker_section(self) -> Text:
        """Build worker status section."""
        section = Text()
        section.append("  WORKERS", style="bold green")

        wg = self.state.worker_group
        if not wg:
            section.append("  no status available", style="dim italic")
            return section

        task_groups: dict[str, list[tuple[str, WorkerState]]] = {
            "scan": [],
            "enrich": [],
            "validate": [],
        }

        for name, status in wg.workers.items():
            if "discover" in name or "scan" in name:
                task_groups["scan"].append((name, status.state))
            elif "enrich" in name:
                task_groups["enrich"].append((name, status.state))
            elif "validate" in name:
                task_groups["validate"].append((name, status.state))

        for task, workers in task_groups.items():
            if not workers:
                section.append(f"  {task}:0", style="dim")
                continue

            count = len(workers)
            state_counts: dict[WorkerState, int] = {}
            for _, state in workers:
                state_counts[state] = state_counts.get(state, 0) + 1

            if state_counts.get(WorkerState.crashed, 0) > 0:
                style = "red"
            elif state_counts.get(WorkerState.backoff, 0) > 0:
                style = "yellow"
            elif state_counts.get(WorkerState.running, 0) > 0:
                style = "green"
            else:
                style = "dim"

            section.append(f"  {task}:{count}", style=style)

            running = state_counts.get(WorkerState.running, 0)
            backing_off = state_counts.get(WorkerState.backoff, 0)
            failed = state_counts.get(WorkerState.crashed, 0)

            if backing_off > 0 or failed > 0:
                parts = []
                if running > 0:
                    parts.append(f"{running} active")
                if backing_off > 0:
                    parts.append(f"{backing_off} backoff")
                if failed > 0:
                    parts.append(f"{failed} failed")
                section.append(f" ({', '.join(parts)})", style="dim")

        return section

    def _build_progress_section(self) -> Text:
        """Build the main progress bars.

        Progress semantics:
        - SCAN: Shows total signals discovered. Progress bar fills as scan completes.
        - ENRICH: Shows enriched count relative to total signals discovered.
        - VALIDATE: Shows validated count relative to enriched signals.

        Note: total_signals is the high water mark (all signals ever discovered),
        not the current count in 'discovered' status which decreases as signals
        are enriched.
        """
        section = Text()
        bar_width = self.BAR_WIDTH

        # Total signals in graph is our TEC (total eventually consistent)
        total = max(self.state.total_signals, 1)
        enriched = self.state.signals_enriched + self.state.signals_validated
        validated = self.state.signals_validated

        # SCAN row - shows total signals discovered
        # Progress = (enriched + validated) / total when scan is done,
        # or scan progress during active scanning
        section.append("  SCAN    ", style="bold blue")
        if self.state.scan_processing or self.state.signals_discovered > 0:
            # Active scan or scan pending - show actual discovered signals
            ratio = 1.0 if self.state.total_signals > 0 else 0
        else:
            # Scan complete - show full bar
            ratio = 1.0
        section.append(make_bar(ratio, bar_width), style="blue")
        section.append(f" {total:>6,}", style="bold")
        section.append("     ", style="dim")  # No percentage for scanning
        if self.state.discover_rate and self.state.discover_rate > 0:
            section.append(f" {self.state.discover_rate:>5.1f}/s", style="dim")
        section.append("\n")

        # ENRICH row - shows enriched progress relative to total discovered
        enrich_pct = enriched / total * 100 if total > 0 else 0

        if self.state.discover_only:
            section.append("  ENRICH  ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("   disabled", style="dim italic")
        else:
            section.append("  ENRICH  ", style="bold green")
            ratio = min(enriched / total, 1.0) if total > 0 else 0
            section.append(make_bar(ratio, bar_width), style="green")
            section.append(f" {enriched:>6,}", style="bold")
            section.append(f" {enrich_pct:>3.0f}%", style="cyan")
            if self.state.enrich_rate and self.state.enrich_rate > 0:
                section.append(f" {self.state.enrich_rate:>5.1f}/s", style="dim")
        section.append("\n")

        # VALIDATE row - shows validated relative to enriched
        # Denominator is (enriched + validated) = signals that passed enrichment
        validate_denom = max(enriched, 1)
        validate_pct = validated / validate_denom * 100 if validate_denom > 0 else 0

        if self.state.discover_only or self.state.enrich_only:
            section.append("  VALIDATE", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append(" disabled", style="dim italic")
        else:
            section.append("  VALIDATE", style="bold magenta")
            ratio = min(validated / validate_denom, 1.0) if validate_denom > 0 else 0
            section.append(make_bar(ratio, bar_width), style="magenta")
            section.append(f" {validated:>6,}", style="bold")
            section.append(f" {validate_pct:>3.0f}%", style="cyan")
            if self.state.validate_rate and self.state.validate_rate > 0:
                section.append(f" {self.state.validate_rate:>5.1f}/s", style="dim")

        return section

    def _build_activity_section(self) -> Text:
        """Build the current activity section with two lines per streamer."""
        section = Text()
        content_width = self.WIDTH - 6

        # SCAN section (2 lines)
        scan = self.state.current_scan
        section.append("  SCAN    ", style="bold blue")
        if scan:
            # First line: show epoch phase or path
            if scan.epoch_phase:
                # Show epoch detection progress
                if scan.epoch_phase == "coarse":
                    pct = (
                        int(100 * scan.epoch_shots_scanned / scan.epoch_total_shots)
                        if scan.epoch_total_shots > 0
                        else 0
                    )
                    status = f"coarse scan {pct}% ({scan.epoch_shots_scanned}/{scan.epoch_total_shots} shots)"
                    if scan.epoch_current_shot:
                        status += f" at shot {scan.epoch_current_shot}"
                elif scan.epoch_phase == "refine":
                    status = (
                        f"refining boundary {scan.epoch_boundaries_refined + 1}/"
                        f"{scan.epoch_boundaries_found}"
                    )
                else:
                    status = f"building {scan.epoch_boundaries_found} epochs"
                section.append(status, style="cyan")
            else:
                path = scan.node_path or scan.signal_id
                section.append(clip_text(path, content_width - 10), style="white")
            section.append("\n")
            # Second line: tree name, boundaries found, signal count
            section.append("           ", style="dim")  # 11 spaces align with content
            if scan.tree_name:
                section.append(f"tree={scan.tree_name}  ", style="cyan")
            if scan.epoch_phase and scan.epoch_boundaries_found > 0:
                section.append(
                    f"{scan.epoch_boundaries_found} epochs detected  ", style="yellow"
                )
            if scan.signals_in_tree > 0:
                section.append(
                    f"{scan.signals_in_tree:,} signals discovered", style="dim"
                )
        elif self.state.scan_processing:
            section.append("scanning...", style="cyan italic")
            section.append("\n")
            section.append("           ", style="dim")
            if self.state.current_tree:
                section.append(f"tree={self.state.current_tree}", style="cyan")
        else:
            section.append("idle", style="dim italic")
            section.append("\n")
            section.append("           ", style="dim")
        section.append("\n")

        # ENRICH section (2 lines)
        if not self.state.discover_only:
            enrich = self.state.current_enrich
            section.append("  ENRICH  ", style="bold green")
            if enrich:
                section.append(
                    clip_text(enrich.signal_id, content_width - 10), style="white"
                )
                section.append("\n")
                section.append(
                    "           ", style="dim"
                )  # 11 spaces align with content
                if enrich.physics_domain:
                    section.append(f"{enrich.physics_domain}  ", style="cyan")
                if enrich.description:
                    desc = clean_text(enrich.description)
                    used = 10 + (
                        len(enrich.physics_domain) + 2 if enrich.physics_domain else 0
                    )
                    section.append(
                        clip_text(desc, content_width - used), style="italic dim"
                    )
            elif self.state.enrich_processing:
                section.append("classifying...", style="cyan italic")
                section.append("\n")
                section.append("           ", style="dim")
            else:
                section.append("idle", style="dim italic")
                section.append("\n")
                section.append("           ", style="dim")
            section.append("\n")

        # VALIDATE section (2 lines)
        if not self.state.discover_only and not self.state.enrich_only:
            validate = self.state.current_validate
            section.append("  VALIDATE ", style="bold magenta")
            if validate:
                shot_str = f"shot={validate.shot}" if validate.shot else ""
                if validate.success is True:
                    section.append(f"{shot_str} success", style="green")
                elif validate.success is False:
                    err = validate.error[:40] if validate.error else "failed"
                    section.append(f"{shot_str} {err}", style="red")
                else:
                    section.append(f"{shot_str} testing...", style="cyan italic")
                section.append("\n")
                # Second line: signal ID
                section.append("           ", style="dim")  # 11 spaces for VALIDATE
                section.append(
                    clip_text(validate.signal_id, content_width - 11), style="dim"
                )
            elif self.state.validate_processing:
                section.append("testing...", style="cyan italic")
                section.append("\n")
                section.append("           ", style="dim")
            else:
                section.append("idle", style="dim italic")
                section.append("\n")
                section.append("           ", style="dim")

        return section

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges."""
        section = Text()

        # TIME row
        section.append("  TIME  ", style="bold cyan")
        eta = None if self.state.discover_only else self.state.eta_seconds
        if eta is not None and eta > 0:
            total_est = self.state.elapsed + eta
            section.append_text(
                make_resource_gauge(self.state.elapsed, total_est, self.GAUGE_WIDTH)
            )
        else:
            section.append("│", style="dim")
            section.append("━" * self.GAUGE_WIDTH, style="cyan")
            section.append("│", style="dim")
        section.append(f"  {format_time(self.state.elapsed)}", style="bold")
        if eta is not None and eta > 0:
            section.append(f"  ETA {format_time(eta)}", style="dim")
        section.append("\n")

        # COST row
        if not self.state.discover_only:
            section.append("  COST  ", style="bold yellow")
            section.append_text(
                make_resource_gauge(
                    self.state.run_cost, self.state.cost_limit, self.GAUGE_WIDTH
                )
            )
            section.append(f"  ${self.state.run_cost:.2f}", style="bold")
            section.append(f" / ${self.state.cost_limit:.2f}", style="dim")
            section.append("\n")

            # TOTAL row - progress toward estimated total cost (ETC)
            # Cost accumulates from enrichment, so estimate based on signals remaining
            total_cost = self.state.accumulated_cost + self.state.run_cost
            signals_enriched = self.state.run_enriched
            signals_remaining = self.state.pending_enrich + self.state.pending_validate

            # Compute ETC (Estimated Total Cost)
            etc = total_cost
            if signals_enriched > 0 and signals_remaining > 0:
                cost_per_signal = self.state.run_cost / signals_enriched
                etc = total_cost + (cost_per_signal * signals_remaining)

            if total_cost > 0 or signals_remaining > 0:
                section.append("  TOTAL ", style="bold white")
                # Progress bar shows current cost toward ETC
                if etc > 0:
                    section.append_text(
                        make_resource_gauge(total_cost, etc, self.GAUGE_WIDTH)
                    )
                else:
                    section.append("│", style="dim")
                    section.append("━" * self.GAUGE_WIDTH, style="white")
                    section.append("│", style="dim")

                section.append(f"  ${total_cost:.2f}", style="bold")
                # Show ETC (dynamic estimate)
                if etc > total_cost:
                    section.append(f"  ETC ${etc:.2f}", style="dim")
                section.append("\n")

        # STATS row - show counts and pending work
        total = self.state.total_signals
        enriched = self.state.signals_enriched + self.state.signals_validated
        validated = self.state.signals_validated

        section.append("  STATS ", style="bold magenta")
        section.append(f"discovered={total}", style="blue")
        section.append(f"  enriched={enriched}", style="green")
        section.append(f"  validated={validated}", style="magenta")

        # Show pending work counts
        pending_parts = []
        if self.state.pending_enrich > 0:
            pending_parts.append(f"enrich:{self.state.pending_enrich}")
        if self.state.pending_validate > 0:
            pending_parts.append(f"validate:{self.state.pending_validate}")
        if pending_parts:
            section.append(f"  pending=[{' '.join(pending_parts)}]", style="cyan dim")

        # Failed/skipped last
        if self.state.signals_failed > 0:
            section.append(f"  failed={self.state.signals_failed}", style="red")
        if self.state.signals_skipped > 0:
            section.append(f"  skipped={self.state.signals_skipped}", style="dim")

        return section

    def _build_display(self) -> Panel:
        """Build the complete display."""
        sections = [
            self._build_header(),
            Text("─" * (self.WIDTH - 4), style="dim"),
            self._build_worker_section(),
            Text("─" * (self.WIDTH - 4), style="dim"),
            self._build_progress_section(),
            Text("─" * (self.WIDTH - 4), style="dim"),
            self._build_activity_section(),
            Text("─" * (self.WIDTH - 4), style="dim"),
            self._build_resources_section(),
        ]

        content = Text()
        for i, section in enumerate(sections):
            if i > 0:
                content.append("\n")
            content.append_text(section)

        return Panel(
            content,
            border_style="cyan",
            width=self.WIDTH,
            padding=(0, 1),
        )

    # ========================================================================
    # Public API
    # ========================================================================

    def __enter__(self) -> DataProgressDisplay:
        """Start live display."""
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
            vertical_overflow="visible",
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        """Stop live display."""
        if self._live:
            self._live.__exit__(*args)

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_display())

    def tick(self) -> None:
        """Drain streaming queues for smooth display."""
        if item := self.state.scan_queue.pop():
            # Extract epoch progress if present
            epoch_progress = item.get("epoch_progress", {})
            self.state.current_scan = ScanItem(
                signal_id=item.get("signal_id", ""),
                tree_name=item.get("tree_name"),
                node_path=item.get("node_path"),
                signals_in_tree=item.get("signals_in_tree", 0),
                epoch_phase=epoch_progress.get("phase"),
                epoch_current_shot=epoch_progress.get("current_shot"),
                epoch_shots_scanned=epoch_progress.get("shots_scanned", 0),
                epoch_total_shots=epoch_progress.get("total_shots", 0),
                epoch_boundaries_found=epoch_progress.get("boundaries_found", 0),
                epoch_boundaries_refined=epoch_progress.get("boundaries_refined", 0),
            )
            # Track current tree for idle display
            if item.get("tree_name"):
                self.state.current_tree = item.get("tree_name")

        if item := self.state.enrich_queue.pop():
            self.state.current_enrich = EnrichItem(
                signal_id=item.get("signal_id", ""),
                physics_domain=item.get("physics_domain"),
                description=item.get("description", ""),
            )

        if item := self.state.validate_queue.pop():
            self.state.current_validate = ValidateItem(
                signal_id=item.get("signal_id", ""),
                shot=item.get("shot"),
                success=item.get("success"),
                error=item.get("error"),
                physics_domain=item.get("physics_domain"),
            )

        self._refresh()

    def update_scan(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
        current_tree: str | None = None,
    ) -> None:
        """Update scan worker state."""
        self.state.run_discovered = stats.processed
        self.state.discover_rate = stats.rate

        if current_tree:
            self.state.current_tree = current_tree

        if "scanning" in message.lower() or "epoch" in message.lower():
            self.state.scan_processing = True
        else:
            self.state.scan_processing = False

        if results:
            # Get signal count for this tree from results
            tree_counts: dict[str, int] = {}
            for r in results:
                tree = r.get("tree_name")
                if tree:
                    tree_counts[tree] = tree_counts.get(tree, 0) + 1

            items = [
                {
                    "signal_id": r.get("id", ""),
                    "tree_name": r.get("tree_name"),
                    "node_path": r.get("node_path"),
                    "signals_in_tree": tree_counts.get(r.get("tree_name", ""), 0),
                    "epoch_progress": r.get("epoch_progress"),  # Include epoch progress
                }
                for r in results
            ]
            max_rate = 5.0
            display_rate = min(stats.rate, max_rate) if stats.rate else 2.0
            self.state.scan_queue.add(items, display_rate)

        self._refresh()

    # Backward compatibility alias
    def update_discover(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Alias for update_scan for backward compatibility."""
        self.update_scan(message, stats, results)

    def update_enrich(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update enrich worker state."""
        self.state.run_enriched = stats.processed
        self.state.enrich_rate = stats.rate
        self.state._run_enrich_cost = stats.cost

        if "classifying" in message.lower() or "enriching" in message.lower():
            self.state.enrich_processing = True
        else:
            self.state.enrich_processing = False

        if results:
            items = [
                {
                    "signal_id": r.get("id", ""),
                    "physics_domain": r.get("physics_domain"),
                    "description": r.get("description", ""),
                }
                for r in results
            ]
            max_rate = 2.0
            display_rate = min(stats.rate, max_rate) if stats.rate else 0.5
            self.state.enrich_queue.add(items, display_rate)

        self._refresh()

    def update_validate(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update validate worker state."""
        self.state.run_validated = stats.processed
        self.state.validate_rate = stats.rate

        if "testing" in message.lower() or "validating" in message.lower():
            self.state.validate_processing = True
        else:
            self.state.validate_processing = False

        if results:
            items = [
                {
                    "signal_id": r.get("id", ""),
                    "shot": r.get("shot"),
                    "success": r.get("success"),
                    "error": r.get("error"),
                    "physics_domain": r.get("physics_domain"),
                }
                for r in results
            ]
            max_rate = 2.0
            display_rate = min(stats.rate, max_rate) if stats.rate else 0.5
            self.state.validate_queue.add(items, display_rate)

        self._refresh()

    def update_from_graph(
        self,
        total_signals: int = 0,
        signals_discovered: int = 0,
        signals_enriched: int = 0,
        signals_validated: int = 0,
        signals_skipped: int = 0,
        signals_failed: int = 0,
        pending_enrich: int = 0,
        pending_validate: int = 0,
        accumulated_cost: float = 0.0,
        **kwargs,
    ) -> None:
        """Update state from graph statistics."""
        self.state.total_signals = total_signals
        self.state.signals_discovered = signals_discovered
        self.state.signals_enriched = signals_enriched
        self.state.signals_validated = signals_validated
        self.state.signals_skipped = signals_skipped
        self.state.signals_failed = signals_failed
        self.state.pending_enrich = pending_enrich
        self.state.pending_validate = pending_validate
        self.state.accumulated_cost = accumulated_cost
        self._refresh()

    def update_worker_status(self, worker_group: SupervisedWorkerGroup) -> None:
        """Update worker status from supervised worker group."""
        self.state.worker_group = worker_group
        self._refresh()

    def print_summary(self) -> None:
        """Print final discovery summary."""
        self.console.print()
        self.console.print(
            Panel(
                self._build_summary(),
                title=f"{self.state.facility.upper()} Signal Discovery Complete",
                border_style="green",
                width=self.WIDTH,
            )
        )

    def _build_summary(self) -> Text:
        """Build final summary text."""
        summary = Text()

        total = self.state.total_signals
        enriched = self.state.signals_enriched + self.state.signals_validated
        validated = self.state.signals_validated

        # SCAN stats
        summary.append("  SCAN   ", style="bold blue")
        summary.append(f"discovered={total:,}", style="blue")
        if self.state.discover_rate:
            summary.append(f"  {self.state.discover_rate:.1f}/s", style="dim")
        summary.append("\n")

        # ENRICH stats
        summary.append("  ENRICH ", style="bold green")
        summary.append(f"enriched={enriched:,}", style="green")
        summary.append(f"  skipped={self.state.signals_skipped:,}", style="yellow")
        summary.append(f"  cost=${self.state.run_cost:.3f}", style="yellow")
        if self.state.enrich_rate:
            summary.append(f"  {self.state.enrich_rate:.1f}/s", style="dim")
        summary.append("\n")

        # VALIDATE stats
        summary.append("  VALIDATE", style="bold magenta")
        summary.append(f" validated={validated:,}", style="magenta")
        summary.append(f"  failed={self.state.signals_failed:,}", style="red")
        if self.state.validate_rate:
            summary.append(f"  {self.state.validate_rate:.1f}/s", style="dim")
        summary.append("\n")

        # USAGE stats
        total_cost = self.state.accumulated_cost + self.state.run_cost
        summary.append("  USAGE ", style="bold cyan")
        summary.append(f"time={format_time(self.state.elapsed)}", style="white")
        summary.append(f"  cost=${self.state.run_cost:.2f}", style="yellow")
        if self.state.accumulated_cost > 0:
            summary.append(f"  total=${total_cost:.2f}", style="dim")

        return summary
