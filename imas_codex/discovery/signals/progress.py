"""
Progress display for parallel data signal discovery.

Design principles (matching wiki and paths progress displays):
- Clean hierarchy: Target → Pipeline → Resources
- Unified per-stage blocks: progress bar + current activity + detail
- Per-stage cost and worker count annotations
- Gradient progress bars with percentage
- Resource gauges for time and cost budgets
- Minimal visual clutter (no emojis, thin progress bars)

Display layout: PIPELINE → RESOURCES
- SCAN: MDSplus tree traversal, TDI introspection
- ENRICH: LLM classification of physics domain, description
- CHECK: Test data access, verify units/sign

Uses common pipeline infrastructure from base.progress module.
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
    GAUGE_METRICS_WIDTH,
    LABEL_WIDTH,
    METRICS_WIDTH,
    MIN_WIDTH,
    PipelineRowConfig,
    ResourceConfig,
    StreamQueue,
    build_pipeline_section,
    build_resource_section,
    clean_text,
    clip_text,
    compute_bar_width,
    compute_gauge_width,
    format_time,
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
class CheckItem:
    """Current check activity."""

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
    signals_checked: int = 0
    signals_skipped: int = 0
    signals_failed: int = 0

    # Pending work counts
    pending_enrich: int = 0
    pending_check: int = 0

    # This run stats
    run_discovered: int = 0
    run_enriched: int = 0
    run_checked: int = 0
    _run_enrich_cost: float = 0.0
    discover_rate: float | None = None
    enrich_rate: float | None = None
    check_rate: float | None = None

    # Accumulated facility cost (from graph)
    accumulated_cost: float = 0.0

    # Current items
    current_scan: ScanItem | None = None
    current_enrich: EnrichItem | None = None
    current_check: CheckItem | None = None
    scan_processing: bool = False
    current_tree: str | None = None  # Currently scanning tree
    enrich_processing: bool = False
    check_processing: bool = False

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
    check_queue: StreamQueue = field(
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
        """Estimated time to termination.

        Returns the maximum ETA across all active worker pipelines:
        - Enrich ETA: based on pending_enrich / enrich_rate
        - Check ETA: based on pending_check / check_rate
        - Cost ETA: based on remaining budget / cost rate

        The actual completion time is bounded by the slowest pipeline.
        """
        etas = []

        # Cost-based ETA (if we have cost tracking)
        if self.run_cost > 0 and self.cost_limit > 0 and self.elapsed > 0:
            cost_rate = self.run_cost / self.elapsed
            if cost_rate > 0:
                remaining_budget = self.cost_limit - self.run_cost
                if remaining_budget > 0:
                    etas.append(remaining_budget / cost_rate)

        # Signal limit ETA
        if self.signal_limit is not None and self.signal_limit > 0:
            if self.run_enriched > 0 and self.elapsed > 0:
                rate = self.run_enriched / self.elapsed
                remaining = self.signal_limit - self.run_enriched
                if rate > 0 and remaining > 0:
                    etas.append(remaining / rate)

        # Enrich ETA (pipeline bottleneck)
        if self.enrich_rate and self.enrich_rate > 0 and self.pending_enrich > 0:
            etas.append(self.pending_enrich / self.enrich_rate)

        # Check ETA (pipeline bottleneck)
        if self.check_rate and self.check_rate > 0 and self.pending_check > 0:
            etas.append(self.pending_check / self.check_rate)

        # Return the maximum (slowest pipeline determines completion)
        return max(etas) if etas else None


# =============================================================================
# Main Display Class
# =============================================================================


class DataProgressDisplay:
    """Clean progress display for parallel signal discovery.

    Layout (100 chars wide):
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                               TCV Signal Discovery                                               │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  WORKERS  scan:1 (running)  enrich:2 (1 active)  check:1                                        │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  SCAN   ━━━━━━━━━━━━━━━━━━━━━━━━━━━────────────────────────────   2944        83.4/s             │
    │  ENRICH ━━━━━━━━━━━━━━━━━━━━━━━━──────────────────────────────       5   0%    0.0/s             │
    │  CHECK━━━━━━━━─────────────────────────────────────────────       0   0%    0.0/s             │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  SCAN   \\HYBRID::PID_I                                                                          │
    │          tree=hybrid  2944 signals discovered                                                    │
    │  ENRICH tcv:equilibrium/plasma_current                                                           │
    │          equilibrium  Main plasma current from LIUQE equilibrium code                            │
    │  CHECK shot=85000 testing...                                                                  │
    │          tcv:equilibrium/elongation                                                              │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  TIME    ━━━━━━━━━━━━━━━━━━━━━━  6m 50s                                                          │
    │  COST    ━━━━━━━━━━━━━━━━━━━━━━  $0.00 / $0.20                                                   │
    │  STATS  discovered=2944  enriched=5  checked=0  pending=[enrich:2944 check:5]                    │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
    """

    # Layout constants imported from base.progress
    LABEL_WIDTH = LABEL_WIDTH
    MIN_WIDTH = MIN_WIDTH
    METRICS_WIDTH = METRICS_WIDTH
    GAUGE_METRICS_WIDTH = GAUGE_METRICS_WIDTH

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

    @property
    def width(self) -> int:
        """Get display width based on terminal size (fills terminal)."""
        term_width = self.console.width or 100
        return max(self.MIN_WIDTH, term_width)

    @property
    def bar_width(self) -> int:
        """Calculate progress bar width to fill available space."""
        return compute_bar_width(self.width)

    @property
    def gauge_width(self) -> int:
        """Calculate resource gauge width (shorter than bar to fit metrics)."""
        return compute_gauge_width(self.width)

    def _build_header(self) -> Text:
        """Build centered header with facility and focus."""
        header = Text()

        title = f"{self.state.facility.upper()} Signal Discovery"
        if self.state.discover_only:
            title += " (SCAN ONLY)"
        elif self.state.enrich_only:
            title += " (ENRICH ONLY)"
        header.append(title.center(self.width - 4), style="bold cyan")

        if self.state.focus:
            header.append("\n")
            focus_line = f"Focus: {self.state.focus}"
            header.append(focus_line.center(self.width - 4), style="italic dim")

        return header

    def _count_group_workers(self, group: str) -> tuple[int, str]:
        """Count workers in a group and build annotation string.

        Returns (count, annotation) where annotation describes
        unhealthy workers (e.g., "(1 backoff)").
        """
        wg = self.state.worker_group
        if not wg:
            return 0, ""

        count = 0
        running = 0
        backoff = 0
        crashed = 0
        for _name, status in wg.workers.items():
            grp = status.group or _name.split("_worker")[0]
            if grp == group:
                count += 1
                if status.state == WorkerState.running:
                    running += 1
                elif status.state == WorkerState.backoff:
                    backoff += 1
                elif status.state == WorkerState.crashed:
                    crashed += 1

        ann_parts: list[str] = []
        if backoff > 0:
            ann_parts.append(f"{backoff} backoff")
        if crashed > 0:
            ann_parts.append(f"{crashed} failed")
        annotation = f"({', '.join(ann_parts)})" if ann_parts else ""
        return count, annotation

    def _build_pipeline_section(self) -> Text:
        """Build the unified pipeline section (progress + activity merged).

        Each pipeline stage gets a 3-line block:
          Line 1: SCAN   ━━━━━━━━━━━━━━━━━━    2,944       83.4/s
          Line 2:        \\HYBRID::PID_I                      ×1
          Line 3:        tree=hybrid  2944 signals discovered

        Stages: SCAN → ENRICH → CHECK
        """
        content_width = self.width - 6

        # --- Compute progress data ---

        total = max(self.state.total_signals, 1)
        enriched = self.state.signals_enriched + self.state.signals_checked
        checked = self.state.signals_checked
        check_denom = max(enriched, 1)

        # Worker counts per group
        scan_count, scan_ann = self._count_group_workers("scan")
        enrich_count, enrich_ann = self._count_group_workers("enrich")
        check_count, check_ann = self._count_group_workers("check")

        # Enrich cost
        enrich_cost = (
            self.state._run_enrich_cost
            if self.state._run_enrich_cost > 0
            else None
        )

        # --- Build activity data ---

        # SCAN activity
        scan = self.state.current_scan
        scan_text = ""
        scan_detail: list[tuple[str, str]] | None = None
        if scan:
            if scan.epoch_phase:
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
                scan_text = status
            else:
                path = scan.node_path or scan.signal_id
                scan_text = clip_text(path, content_width - 10)

            parts: list[tuple[str, str]] = []
            if scan.tree_name:
                parts.append((f"tree={scan.tree_name}  ", "cyan"))
            if scan.epoch_phase and scan.epoch_boundaries_found > 0:
                parts.append(
                    (f"{scan.epoch_boundaries_found} epochs detected", "yellow")
                )
            elif scan.signals_in_tree > 0:
                parts.append((f"{scan.signals_in_tree:,} nodes scanned", "dim"))
            scan_detail = parts or None
        elif self.state.scan_processing and self.state.current_tree:
            scan_text = ""  # Will show processing label
            scan_detail = [(f"tree={self.state.current_tree}", "cyan")]

        # ENRICH activity
        enrich = self.state.current_enrich
        enrich_text = ""
        enrich_detail: list[tuple[str, str]] | None = None
        if enrich:
            enrich_text = clip_text(enrich.signal_id, content_width - 10)
            parts = []
            if enrich.physics_domain:
                parts.append((f"{enrich.physics_domain}  ", "cyan"))
            if enrich.description:
                desc = clean_text(enrich.description)
                used = 10 + (
                    len(enrich.physics_domain) + 2 if enrich.physics_domain else 0
                )
                parts.append((clip_text(desc, content_width - used), "italic dim"))
            enrich_detail = parts or None

        # CHECK activity
        validate = self.state.current_check
        check_text = ""
        check_detail: list[tuple[str, str]] | None = None
        if validate:
            shot_str = f"shot={validate.shot}" if validate.shot else ""
            if validate.success is True:
                check_text = f"{shot_str} success"
            elif validate.success is False:
                err = validate.error[:40] if validate.error else "failed"
                check_text = f"{shot_str} {err}"
            else:
                check_text = f"{shot_str} testing..."
            check_detail = [
                (clip_text(validate.signal_id, content_width - 10), "dim")
            ]

        # --- Build pipeline rows ---

        rows = [
            PipelineRowConfig(
                name="SCAN",
                style="bold blue",
                completed=total,
                total=total,
                rate=self.state.discover_rate,
                show_pct=False,
                worker_count=scan_count,
                worker_annotation=scan_ann,
                primary_text=scan_text,
                detail_parts=scan_detail,
                is_processing=self.state.scan_processing,
                processing_label="scanning...",
            ),
            PipelineRowConfig(
                name="ENRICH",
                style="bold green",
                completed=enriched,
                total=total,
                rate=self.state.enrich_rate,
                cost=enrich_cost,
                disabled=self.state.discover_only,
                worker_count=enrich_count,
                worker_annotation=enrich_ann,
                primary_text=enrich_text,
                detail_parts=enrich_detail,
                is_processing=self.state.enrich_processing,
                processing_label="classifying...",
            ),
            PipelineRowConfig(
                name="CHECK",
                style="bold magenta",
                completed=checked,
                total=check_denom,
                rate=self.state.check_rate,
                disabled=self.state.discover_only or self.state.enrich_only,
                worker_count=check_count,
                worker_annotation=check_ann,
                primary_text=check_text,
                detail_parts=check_detail,
                is_processing=self.state.check_processing,
                processing_label="testing...",
            ),
        ]
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges using unified builder."""
        total = self.state.total_signals
        enriched = self.state.signals_enriched + self.state.signals_checked
        checked = self.state.signals_checked

        # Compute ETC
        total_cost = self.state.accumulated_cost + self.state.run_cost
        etc = total_cost
        signals_enriched = self.state.run_enriched
        signals_remaining = self.state.pending_enrich + self.state.pending_check
        if signals_enriched > 0 and signals_remaining > 0:
            cost_per_signal = self.state.run_cost / signals_enriched
            etc = total_cost + (cost_per_signal * signals_remaining)

        # Build stats
        stats: list[tuple[str, str, str]] = [
            ("discovered", str(total), "blue"),
            ("enriched", str(enriched), "green"),
            ("checked", str(checked), "magenta"),
        ]
        if self.state.signals_failed > 0:
            stats.append(("failed", str(self.state.signals_failed), "red"))
        if self.state.signals_skipped > 0:
            stats.append(("skipped", str(self.state.signals_skipped), "dim"))

        config = ResourceConfig(
            elapsed=self.state.elapsed,
            eta=None if self.state.discover_only else self.state.eta_seconds,
            run_cost=self.state.run_cost,
            cost_limit=self.state.cost_limit,
            accumulated_cost=self.state.accumulated_cost,
            etc=etc if etc > total_cost else None,
            scan_only=self.state.discover_only,
            stats=stats,
            pending=[
                ("enrich", self.state.pending_enrich),
                ("check", self.state.pending_check),
            ],
        )
        return build_resource_section(config, self.gauge_width)

    def _build_display(self) -> Panel:
        """Build the complete display."""
        sections = [
            self._build_header(),
            Text("─" * (self.width - 4), style="dim"),
            self._build_pipeline_section(),
            Text("─" * (self.width - 4), style="dim"),
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
            width=self.width,
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

        if item := self.state.check_queue.pop():
            self.state.current_check = CheckItem(
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

    def update_check(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update validate worker state."""
        self.state.run_checked = stats.processed
        self.state.check_rate = stats.rate

        if "testing" in message.lower() or "validating" in message.lower():
            self.state.check_processing = True
        else:
            self.state.check_processing = False

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
            self.state.check_queue.add(items, display_rate)

        self._refresh()

    def update_from_graph(
        self,
        total_signals: int = 0,
        signals_discovered: int = 0,
        signals_enriched: int = 0,
        signals_checked: int = 0,
        signals_skipped: int = 0,
        signals_failed: int = 0,
        pending_enrich: int = 0,
        pending_check: int = 0,
        accumulated_cost: float = 0.0,
        **kwargs,
    ) -> None:
        """Update state from graph statistics."""
        self.state.total_signals = total_signals
        self.state.signals_discovered = signals_discovered
        self.state.signals_enriched = signals_enriched
        self.state.signals_checked = signals_checked
        self.state.signals_skipped = signals_skipped
        self.state.signals_failed = signals_failed
        self.state.pending_enrich = pending_enrich
        self.state.pending_check = pending_check
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
                width=self.width,
            )
        )

    def _build_summary(self) -> Text:
        """Build final summary text."""
        summary = Text()

        total = self.state.total_signals
        enriched = self.state.signals_enriched + self.state.signals_checked
        checked = self.state.signals_checked

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

        # CHECK stats
        summary.append("  CHECK  ", style="bold magenta")
        summary.append(f"checked={checked:,}", style="magenta")
        summary.append(f"  failed={self.state.signals_failed:,}", style="red")
        if self.state.check_rate:
            summary.append(f"  {self.state.check_rate:.1f}/s", style="dim")
        summary.append("\n")

        # USAGE stats
        total_cost = self.state.accumulated_cost + self.state.run_cost
        summary.append("  USAGE ", style="bold cyan")
        summary.append(f"time={format_time(self.state.elapsed)}", style="white")
        summary.append(f"  cost=${self.state.run_cost:.2f}", style="yellow")
        if self.state.accumulated_cost > 0:
            summary.append(f"  total=${total_cost:.2f}", style="dim")

        return summary
