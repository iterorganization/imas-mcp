"""
Progress display for parallel wiki discovery.

Design principles (matching paths parallel_progress.py):
- Minimal visual clutter (no emojis, no stopwatch icons)
- Clear hierarchy: Target → Progress → Activity → Resources
- Gradient progress bars with percentage
- Resource gauges for time and cost budgets
- Compact current activity with relevant details only

Uses common progress infrastructure from progress_common module.
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
    format_time,
    make_bar,
    make_resource_gauge,
)

if TYPE_CHECKING:
    from imas_codex.discovery.base.progress import WorkerStats


# =============================================================================
# Display Items
# =============================================================================


@dataclass
class ScanItem:
    """Current scan activity."""

    title: str
    out_links: int = 0
    depth: int = 0


@dataclass
class ScoreItem:
    """Current score activity."""

    title: str
    score: float | None = None
    is_physics: bool = False
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class IngestItem:
    """Current ingest activity."""

    title: str
    chunk_count: int = 0


# =============================================================================
# Progress State
# =============================================================================


@dataclass
class ProgressState:
    """All state for the wiki progress display."""

    facility: str
    cost_limit: float
    page_limit: int | None = None
    focus: str = ""

    # Mode flags
    scan_only: bool = False
    score_only: bool = False

    # Counts from graph
    total_pages: int = 0
    pages_scanned: int = 0
    pages_prefetched: int = 0
    pages_scored: int = 0
    pages_ingested: int = 0
    pages_skipped: int = 0
    artifacts_found: int = 0

    # Pending work counts (for progress bars)
    pending_scan: int = 0
    pending_prefetch: int = 0
    pending_score: int = 0
    pending_ingest: int = 0

    # This run stats
    run_scanned: int = 0
    run_scored: int = 0
    run_ingested: int = 0
    _run_score_cost: float = 0.0
    _run_ingest_cost: float = 0.0
    scan_rate: float | None = None
    score_rate: float | None = None
    ingest_rate: float | None = None

    # Current items (and their processing state)
    current_scan: ScanItem | None = None
    current_score: ScoreItem | None = None
    current_ingest: IngestItem | None = None
    scan_processing: bool = False
    score_processing: bool = False
    ingest_processing: bool = False

    # Streaming queues
    scan_queue: StreamQueue = field(default_factory=StreamQueue)
    score_queue: StreamQueue = field(default_factory=StreamQueue)
    ingest_queue: StreamQueue = field(default_factory=StreamQueue)

    # Tracking
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def run_cost(self) -> float:
        """Total cost for this run (score + ingest)."""
        return self._run_score_cost + self._run_ingest_cost

    @property
    def cost_fraction(self) -> float:
        if self.cost_limit <= 0:
            return 0.0
        return min(1.0, self.run_cost / self.cost_limit)

    @property
    def cost_limit_reached(self) -> bool:
        return self.run_cost >= self.cost_limit if self.cost_limit > 0 else False

    @property
    def page_limit_reached(self) -> bool:
        if self.page_limit is None or self.page_limit <= 0:
            return False
        return self.run_scored >= self.page_limit

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time to termination based on limits."""
        # Try cost-based ETA first
        if self.run_cost > 0 and self.cost_limit > 0:
            cost_rate = self.run_cost / self.elapsed if self.elapsed > 0 else 0
            if cost_rate > 0:
                remaining_budget = self.cost_limit - self.run_cost
                return max(0, remaining_budget / cost_rate)

        # Try page-limit-based ETA
        if self.page_limit is not None and self.page_limit > 0:
            if self.run_scored > 0 and self.elapsed > 0:
                rate = self.run_scored / self.elapsed
                remaining = self.page_limit - self.run_scored
                return max(0, remaining / rate) if rate > 0 else None

        # Fall back to work-based ETA
        if not self.score_rate or self.score_rate <= 0:
            return None
        remaining = self.pending_scan + self.pending_score
        return remaining / self.score_rate if remaining > 0 else 0


# =============================================================================
# Main Display Class
# =============================================================================


class WikiProgressDisplay:
    """Clean progress display for parallel wiki discovery.

    Layout (88 chars wide) - matching paths parallel_progress.py:
    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │                         JT60SA Wiki Discovery                                      │
    ├────────────────────────────────────────────────────────────────────────────────────┤
    │  SCAN   ━━━━━━━━━━━━━━━━━━━━━━━━━━─────────────────────    234  42%  7.1/s        │
    │  SCORE  ━━━━━━━━━━━━━━━━━━━━━─────────────────────────      92  28%  0.8/s        │
    │  INGEST ━━━━━━━━━━━━━─────────────────────────────────      45  14%  0.3/s        │
    ├────────────────────────────────────────────────────────────────────────────────────┤
    │  SCAN  IMAS Data Dictionary Overview                                               │
    │    12 links, depth 2                                                               │
    │  SCORE 0.85 COCOS Convention Documentation                                         │
    ├────────────────────────────────────────────────────────────────────────────────────┤
    │  TIME   ▐████████░░░░░░░░░░░▌  2m 30s  ETA 8m                                      │
    │  COST   ▐████████░░░░░░░░░░░▌  $0.45 / $1.00                                       │
    │  STATS  scanned=234  scored=92  ingested=45  skipped=12                            │
    └────────────────────────────────────────────────────────────────────────────────────┘
    """

    WIDTH = 88
    BAR_WIDTH = 40
    GAUGE_WIDTH = 20

    def __init__(
        self,
        facility: str,
        cost_limit: float,
        page_limit: int | None = None,
        focus: str = "",
        console: Console | None = None,
        scan_only: bool = False,
        score_only: bool = False,
    ) -> None:
        self.console = console or Console()
        self.state = ProgressState(
            facility=facility,
            cost_limit=cost_limit,
            page_limit=page_limit,
            focus=focus,
            scan_only=scan_only,
            score_only=score_only,
        )
        self._live: Live | None = None

    def _build_header(self) -> Text:
        """Build centered header with facility and focus."""
        header = Text()

        # Facility name with mode indicator
        title = f"{self.state.facility.upper()} Wiki Discovery"
        if self.state.scan_only:
            title += " (SCAN ONLY)"
        elif self.state.score_only:
            title += " (SCORE ONLY)"
        header.append(title.center(self.WIDTH - 4), style="bold cyan")

        # Focus (if set and not scan_only)
        if self.state.focus and not self.state.scan_only:
            header.append("\n")
            focus_line = f"Focus: {self.state.focus}"
            header.append(focus_line.center(self.WIDTH - 4), style="italic dim")

        return header

    def _build_progress_section(self) -> Text:
        """Build the main progress bars for scan, score, ingest."""
        section = Text()
        bar_width = self.BAR_WIDTH

        # SCAN row
        scan_total = self.state.pages_scanned + self.state.pending_scan
        if scan_total <= 0:
            scan_total = 1
        scan_pct = self.state.pages_scanned / scan_total * 100

        if self.state.score_only:
            section.append("  SCAN  ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  SCAN  ", style="bold blue")
            scan_ratio = min(self.state.pages_scanned / scan_total, 1.0)
            section.append(make_bar(scan_ratio, bar_width), style="blue")
            section.append(f" {self.state.pages_scanned:>6,}", style="bold")
            section.append(f" {scan_pct:>3.0f}%", style="cyan")
            if self.state.scan_rate and self.state.scan_rate > 0:
                section.append(f" {self.state.scan_rate:>5.1f}/s", style="dim")
        section.append("\n")

        # SCORE row
        score_total = self.state.pages_scored + self.state.pending_score
        if score_total <= 0:
            score_total = 1
        score_pct = self.state.pages_scored / score_total * 100

        if self.state.scan_only:
            section.append("  SCORE ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  SCORE ", style="bold green")
            score_ratio = min(self.state.pages_scored / score_total, 1.0)
            section.append(make_bar(score_ratio, bar_width), style="green")
            section.append(f" {self.state.pages_scored:>6,}", style="bold")
            section.append(f" {score_pct:>3.0f}%", style="cyan")
            if self.state.score_rate and self.state.score_rate > 0:
                section.append(f" {self.state.score_rate:>5.1f}/s", style="dim")
        section.append("\n")

        # INGEST row
        ingest_total = self.state.pages_ingested + self.state.pending_ingest
        if ingest_total <= 0:
            ingest_total = 1
        ingest_pct = self.state.pages_ingested / ingest_total * 100

        if self.state.scan_only:
            section.append("  INGEST", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  INGEST", style="bold magenta")
            ingest_ratio = min(self.state.pages_ingested / ingest_total, 1.0)
            section.append(make_bar(ingest_ratio, bar_width), style="magenta")
            section.append(f" {self.state.pages_ingested:>6,}", style="bold")
            section.append(f" {ingest_pct:>3.0f}%", style="cyan")
            if self.state.ingest_rate and self.state.ingest_rate > 0:
                section.append(f" {self.state.ingest_rate:>5.1f}/s", style="dim")

        return section

    def _clip_title(self, title: str, max_len: int = 60) -> str:
        """Clip title to max length."""
        if len(title) <= max_len:
            return title
        return title[: max_len - 3] + "..."

    def _build_activity_section(self) -> Text:
        """Build the current activity section showing what's happening now."""
        section = Text()

        scan = self.state.current_scan
        score = self.state.current_score

        # Helper to determine if worker should show "idle"
        # Only show idle when worker is not processing AND queue is empty
        def should_show_idle(processing: bool, queue) -> bool:
            return not processing and queue.is_empty()

        # SCAN section - always 2 lines for consistent height
        section.append("  SCAN ", style="bold blue")
        if scan:
            section.append(self._clip_title(scan.title, self.WIDTH - 10), style="white")
            section.append("\n")
            section.append("    ", style="dim")
            section.append(f"{scan.out_links} links", style="cyan")
            section.append(", ", style="dim")
            section.append(f"depth {scan.depth}", style="cyan")
        elif self.state.scan_processing:
            section.append("processing batch...", style="cyan italic")
            section.append("\n    ", style="dim")
        elif should_show_idle(self.state.scan_processing, self.state.scan_queue):
            section.append("idle", style="dim italic")
            section.append("\n    ", style="dim")
        else:
            # Queue has items but nothing displayed yet (waiting for tick)
            section.append("...", style="dim italic")
            section.append("\n    ", style="dim")
        section.append("\n")

        # SCORE section - always 2 lines (skip in scan_only mode)
        if not self.state.scan_only:
            section.append("  SCORE ", style="bold green")
            if score:
                # Show score value and title
                if score.score is not None:
                    if score.score >= 0.7:
                        style = "bold green"
                    elif score.score >= 0.4:
                        style = "yellow"
                    else:
                        style = "red"
                    section.append(f"{score.score:.2f} ", style=style)
                section.append(
                    self._clip_title(score.title, self.WIDTH - 20), style="white"
                )
                section.append("\n")
                section.append("    ", style="dim")
                if score.is_physics:
                    section.append("[physics domain] ", style="cyan")
                elif score.skipped:
                    section.append(f"skipped: {score.skip_reason}", style="yellow dim")
            elif self.state.score_processing:
                section.append("processing batch...", style="cyan italic")
                section.append("\n    ", style="dim")
            elif should_show_idle(self.state.score_processing, self.state.score_queue):
                section.append("idle", style="dim italic")
                section.append("\n    ", style="dim")
            else:
                # Queue has items but nothing displayed yet (waiting for tick)
                section.append("...", style="dim italic")
                section.append("\n    ", style="dim")

        return section

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges."""
        section = Text()

        # TIME row
        section.append("  TIME  ", style="bold cyan")
        eta = None if self.state.scan_only else self.state.eta_seconds
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

        if eta is not None:
            if eta <= 0:
                if self.state.cost_limit_reached:
                    section.append("  cost limit reached", style="yellow dim")
                elif self.state.page_limit_reached:
                    section.append("  page limit reached", style="yellow dim")
                else:
                    section.append("  complete", style="green dim")
            else:
                section.append(f"  ETA {format_time(eta)}", style="dim")
        section.append("\n")

        # COST row (hidden in scan_only mode)
        if not self.state.scan_only:
            section.append("  COST  ", style="bold yellow")
            section.append_text(
                make_resource_gauge(
                    self.state.run_cost, self.state.cost_limit, self.GAUGE_WIDTH
                )
            )
            section.append(f"  ${self.state.run_cost:.2f}", style="bold")
            section.append(f" / ${self.state.cost_limit:.2f}", style="dim")
            section.append("\n")

        # STATS row
        section.append("  STATS ", style="bold magenta")
        section.append(f"scanned={self.state.pages_scanned}", style="blue")
        section.append(f"  scored={self.state.pages_scored}", style="green")
        section.append(f"  ingested={self.state.pages_ingested}", style="magenta")
        section.append(f"  skipped={self.state.pages_skipped}", style="yellow")

        if self.state.artifacts_found > 0:
            section.append(f"  artifacts={self.state.artifacts_found}", style="cyan")

        return section

    def _build_display(self) -> Panel:
        """Build the complete display."""
        sections = [
            self._build_header(),
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

    def __enter__(self) -> WikiProgressDisplay:
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
        # Pop from scan queue
        if item := self.state.scan_queue.pop():
            self.state.current_scan = ScanItem(
                title=item.get("title", ""),
                out_links=item.get("out_links", 0),
                depth=item.get("depth", 0),
            )

        # Pop from score queue
        if item := self.state.score_queue.pop():
            self.state.current_score = ScoreItem(
                title=item.get("title", ""),
                score=item.get("score"),
                is_physics=item.get("is_physics", False),
                skipped=item.get("skipped", False),
                skip_reason=item.get("skip_reason", ""),
            )

        # Pop from ingest queue
        if item := self.state.ingest_queue.pop():
            self.state.current_ingest = IngestItem(
                title=item.get("title", ""),
                chunk_count=item.get("chunk_count", 0),
            )

        self._refresh()

    def update_scan(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update scanner state."""
        self.state.run_scanned = stats.processed
        self.state.scan_rate = stats.rate

        # Track processing state
        # Don't clear current_scan when idle - let queue drain naturally
        if message == "idle":
            self.state.scan_processing = False
        elif "scanning" in message.lower():
            self.state.scan_processing = True
        else:
            self.state.scan_processing = False

        # Queue results for streaming with adaptive rate
        if results:
            # Calculate display rate to spread items over ~15 seconds
            target_duration = 15.0
            batch_size = len(results)
            display_rate = batch_size / target_duration if batch_size > 0 else 0.5
            self.state.scan_queue.add(results, display_rate)

        self._refresh()

    def update_score(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update scorer state."""
        self.state.run_scored = stats.processed
        self.state.score_rate = stats.rate
        self.state._run_score_cost = stats.cost

        # Track processing state
        # Don't clear current_score when waiting - let queue drain naturally
        if "waiting" in message.lower() or message == "idle":
            self.state.score_processing = False
        elif "scoring" in message.lower():
            self.state.score_processing = True
        else:
            self.state.score_processing = False

        # Queue results for streaming with adaptive rate
        if results:
            # Calculate display rate to spread items over ~15 seconds
            target_duration = 15.0
            batch_size = len(results)
            display_rate = batch_size / target_duration if batch_size > 0 else 0.5
            self.state.score_queue.add(results, display_rate)

        self._refresh()

    def update_ingest(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update ingester state."""
        self.state.run_ingested = stats.processed
        self.state.ingest_rate = stats.rate
        self.state._run_ingest_cost = stats.cost

        # Track processing state
        # Don't clear current_ingest when waiting - let queue drain naturally
        if "waiting" in message.lower() or message == "idle":
            self.state.ingest_processing = False
        elif "ingesting" in message.lower():
            self.state.ingest_processing = True
        else:
            self.state.ingest_processing = False

        # Queue results for streaming with adaptive rate
        if results:
            # Calculate display rate to spread items over ~15 seconds
            target_duration = 15.0
            batch_size = len(results)
            display_rate = batch_size / target_duration if batch_size > 0 else 0.5
            self.state.ingest_queue.add(results, display_rate)

        self._refresh()

    def update_prefetch(
        self,
        message: str,
        stats: WorkerStats,
    ) -> None:
        """Update prefetch worker state (no display, just tracking)."""
        # Prefetch doesn't have dedicated display, just affects pending counts
        pass

    def update_from_graph(
        self,
        total_pages: int = 0,
        pages_scanned: int = 0,
        pages_prefetched: int = 0,
        pages_scored: int = 0,
        pages_ingested: int = 0,
        pages_skipped: int = 0,
        pending_scan: int = 0,
        pending_prefetch: int = 0,
        pending_score: int = 0,
        pending_ingest: int = 0,
        artifacts_found: int = 0,
    ) -> None:
        """Update state from graph statistics."""
        self.state.total_pages = total_pages
        self.state.pages_scanned = pages_scanned
        self.state.pages_prefetched = pages_prefetched
        self.state.pages_scored = pages_scored
        self.state.pages_ingested = pages_ingested
        self.state.pages_skipped = pages_skipped
        self.state.pending_scan = pending_scan
        self.state.pending_prefetch = pending_prefetch
        self.state.pending_score = pending_score
        self.state.pending_ingest = pending_ingest
        self.state.artifacts_found = artifacts_found
        self._refresh()

    def print_summary(self) -> None:
        """Print final discovery summary."""
        self.console.print()
        self.console.print(
            Panel(
                self._build_summary(),
                title=f"{self.state.facility.upper()} Wiki Discovery Complete",
                border_style="green",
                width=self.WIDTH,
            )
        )

    def _build_summary(self) -> Text:
        """Build final summary text."""
        summary = Text()

        # SCAN stats
        summary.append("  SCAN  ", style="bold blue")
        summary.append(f"scanned={self.state.pages_scanned:,}", style="blue")
        summary.append(f"  prefetched={self.state.pages_prefetched:,}", style="cyan")
        if self.state.scan_rate:
            summary.append(f"  {self.state.scan_rate:.1f}/s", style="dim")
        summary.append("\n")

        # SCORE stats
        summary.append("  SCORE ", style="bold green")
        summary.append(f"scored={self.state.pages_scored:,}", style="green")
        summary.append(f"  skipped={self.state.pages_skipped:,}", style="yellow")
        summary.append(f"  cost=${self.state._run_score_cost:.3f}", style="yellow")
        if self.state.score_rate:
            summary.append(f"  {self.state.score_rate:.1f}/s", style="dim")
        summary.append("\n")

        # INGEST stats
        summary.append("  INGEST", style="bold magenta")
        summary.append(f"  ingested={self.state.pages_ingested:,}", style="magenta")
        if self.state.artifacts_found > 0:
            summary.append(f"  artifacts={self.state.artifacts_found:,}", style="cyan")
        summary.append("\n")

        # USAGE stats
        summary.append("  USAGE ", style="bold white")
        summary.append(f"time={format_time(self.state.elapsed)}", style="white")
        summary.append(f"  total_cost=${self.state.run_cost:.2f}", style="yellow")

        return summary
