"""
Clean progress display for parallel facility discovery.

Design principles:
- Minimal visual clutter (no emojis, no stopwatch icons)
- Clear hierarchy: Target → Progress → Activity → Resources
- Gradient progress bars with percentage
- Resource gauges for time and cost budgets
- Compact current activity with relevant details only
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from imas_codex.discovery.parallel import WorkerStats


# ============================================================================
# Utility Functions
# ============================================================================


def clip_path(path: str, max_len: int = 63) -> str:
    """Clip middle of path: /home/user/.../deep/dir"""
    if len(path) <= max_len:
        return path
    keep_start = max_len // 3
    keep_end = max_len - keep_start - 3
    return f"{path[:keep_start]}...{path[-keep_end:]}"


def format_time(seconds: float) -> str:
    """Format duration: 1h 23m, 5m 30s, 45s"""
    if seconds < 0:
        return "--"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs:02d}s" if secs else f"{mins}m"
    hours, rem = divmod(int(seconds), 3600)
    mins = rem // 60
    return f"{hours}h {mins:02d}m" if mins else f"{hours}h"


def make_bar(
    ratio: float, width: int, filled_char: str = "━", empty_char: str = "─"
) -> str:
    """Create a simple thin progress bar string."""
    ratio = max(0.0, min(1.0, ratio))
    filled = int(width * ratio)
    return filled_char * filled + empty_char * (width - filled)


def make_gradient_bar(ratio: float, width: int) -> Text:
    """Create a gradient progress bar (green → yellow → red as it fills)."""
    ratio = max(0.0, min(1.0, ratio))
    filled = int(width * ratio)

    bar = Text()
    for i in range(width):
        if i < filled:
            # Gradient based on position
            pos_ratio = i / width
            if pos_ratio < 0.5:
                bar.append("━", style="green")
            elif pos_ratio < 0.75:
                bar.append("━", style="yellow")
            else:
                bar.append("━", style="red")
        else:
            bar.append("─", style="dim")
    return bar


def make_resource_gauge(
    used: float, limit: float, width: int = 20, unit: str = ""
) -> Text:
    """Create a resource consumption gauge with color coding."""
    ratio = used / limit if limit > 0 else 0
    ratio = max(0.0, min(1.0, ratio))

    # Color based on consumption
    if ratio < 0.5:
        color = "green"
    elif ratio < 0.8:
        color = "yellow"
    else:
        color = "red"

    filled = int(width * ratio)

    gauge = Text()
    gauge.append("│", style="dim")
    gauge.append("━" * filled, style=color)
    gauge.append("─" * (width - filled), style="dim")
    gauge.append("│", style="dim")

    return gauge


# ============================================================================
# Display Items
# ============================================================================


@dataclass
class ScanItem:
    """Current scan activity."""

    path: str
    files: int = 0
    dirs: int = 0
    has_code: bool = False  # README, Makefile, git


@dataclass
class ScoreItem:
    """Current score activity."""

    path: str
    score: float | None = None
    purpose: str = ""
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class StreamQueue:
    """Rate-limited queue for smooth display updates."""

    items: deque = field(default_factory=deque)
    last_pop: float = field(default_factory=time.time)
    rate: float = 2.0  # items per second

    def add(self, items: list, rate: float | None = None) -> None:
        self.items.extend(items)
        if rate and rate > 0:
            self.rate = rate

    def pop(self) -> Any | None:
        if not self.items:
            return None
        interval = 1.0 / self.rate if self.rate > 0 else 0.5
        now = time.time()
        if now - self.last_pop >= interval:
            self.last_pop = now
            return self.items.popleft()
        return None

    def __len__(self) -> int:
        return len(self.items)


# ============================================================================
# Progress State
# ============================================================================


@dataclass
class ProgressState:
    """All state for the progress display."""

    facility: str
    cost_limit: float
    model: str = ""
    focus: str = ""

    # Mode flags
    scan_only: bool = False
    score_only: bool = False

    # Graph totals (aligned with new state machine)
    total: int = 0
    discovered: int = 0  # Awaiting scan
    listed: int = 0  # Awaiting score
    scored: int = 0  # Scored complete
    skipped: int = 0  # Low value or dead-end
    excluded: int = 0  # Matched exclusion pattern
    max_depth: int = 0  # Maximum tree depth

    # This run
    run_scanned: int = 0
    run_scored: int = 0
    run_cost: float = 0.0
    scan_rate: float | None = None
    score_rate: float | None = None

    # Current items
    current_scan: ScanItem | None = None
    current_score: ScoreItem | None = None

    # Streaming
    scan_queue: StreamQueue = field(default_factory=StreamQueue)
    score_queue: StreamQueue = field(default_factory=StreamQueue)

    # Tracking
    scored_paths: set[str] = field(default_factory=set)
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def frontier_size(self) -> int:
        """Total paths awaiting work (scan or score)."""
        return self.discovered + self.listed

    @property
    def cost_per_path(self) -> float | None:
        """Average cost per scored path."""
        if self.run_scored > 0:
            return self.run_cost / self.run_scored
        return None

    @property
    def estimated_total_cost(self) -> float | None:
        """Estimated total cost based on current rate."""
        cpp = self.cost_per_path
        if cpp is not None and self.total > 0:
            # Estimate: paths remaining * cost per path + current cost
            remaining = self.frontier_size
            return self.run_cost + (remaining * cpp)
        return None

    @property
    def coverage(self) -> float:
        """Percentage of total paths scored."""
        return (self.scored / self.total * 100) if self.total > 0 else 0

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time to completion based on score rate."""
        if not self.score_rate or self.score_rate <= 0:
            return None
        remaining = self.discovered + self.listed
        return remaining / self.score_rate if remaining > 0 else 0


# ============================================================================
# Main Display Class
# ============================================================================


class ParallelProgressDisplay:
    """Clean progress display for parallel discovery.

    Layout (88 chars wide):
    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                         ITER Discovery                                               │
    │                     Focus: equilibrium codes                                         │
    ├──────────────────────────────────────────────────────────────────────────────────────┤
    │  SCAN   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━─────────────────────────  1,234  42%  77.1/s   │
    │  SCORE  ━━━━━━━━━━━━━━━━━━━━━─────────────────────────────────    892  28%   3.2/s   │
    ├──────────────────────────────────────────────────────────────────────────────────────┤
    │  /gss/work/imas/codes/chease/src                                                     │
    │    Scan:  45 files, 3 dirs, code project                                             │
    │    Score: 0.85 simulation_code                                                       │
    ├──────────────────────────────────────────────────────────────────────────────────────┤
    │  COST   ▐████████░░░░░░░░░░░▌  $4.50 / $10.00                                        │
    │  TIME   ▐██████████████░░░░░▌  12m 30s  ETA 8m                                       │
    └──────────────────────────────────────────────────────────────────────────────────────┘
    """

    WIDTH = 88
    BAR_WIDTH = 50
    GAUGE_WIDTH = 20

    def __init__(
        self,
        facility: str,
        cost_limit: float,
        model: str = "",
        console: Console | None = None,
        focus: str = "",
        scan_only: bool = False,
        score_only: bool = False,
    ) -> None:
        self.console = console or Console()
        self.state = ProgressState(
            facility=facility,
            cost_limit=cost_limit,
            model=model,
            focus=focus,
            scan_only=scan_only,
            score_only=score_only,
        )
        self._live: Live | None = None

    def _build_header(self) -> Text:
        """Build centered header with facility and focus."""
        header = Text()

        # Facility name with mode indicator
        title = f"{self.state.facility.upper()} Discovery"
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
        """Build the main progress bars for scan and score."""
        section = Text()

        # Calculate totals and percentages
        # Scan progress: listed out of total known paths
        # "listed" means scan is complete, discovered means awaiting scan
        scanned_count = self.state.total - self.state.discovered
        scan_total = self.state.total if self.state.total > 0 else 1
        scan_pct = (scanned_count / scan_total * 100) if scan_total > 0 else 0

        # Score progress: scored out of listed paths
        score_total = max(scanned_count, self.state.scored) if scanned_count > 0 else 1
        score_pct = (self.state.scored / score_total * 100) if score_total > 0 else 0
        score_pct = min(score_pct, 100.0)  # Cap at 100%

        # Shorter bar to fit everything on one line
        bar_width = 40

        # SCAN row: "  SCAN  ━━━━────  1,234  42%  12.3/s" or disabled
        if self.state.score_only:
            # Disabled state - show as dim with "disabled" indicator
            section.append("  SCAN  ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  SCAN  ", style="bold blue")
            scan_ratio = min(scanned_count / scan_total, 1.0) if scan_total > 0 else 0
            section.append(make_bar(scan_ratio, bar_width), style="blue")
            section.append(f" {scanned_count:>6,}", style="bold")
            section.append(f" {scan_pct:>3.0f}%", style="cyan")
            if self.state.scan_rate:
                section.append(f" {self.state.scan_rate:>5.1f}/s", style="dim")
        section.append("\n")

        # SCORE row: "  SCORE ━━━━────    892  28%   4.2/s" or disabled
        if self.state.scan_only:
            # Disabled state - show as dim with "disabled" indicator
            section.append("  SCORE ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  SCORE ", style="bold green")
            score_ratio = (
                min(self.state.scored / score_total, 1.0) if score_total > 0 else 0
            )
            section.append(make_bar(score_ratio, bar_width), style="green")
            section.append(f" {self.state.scored:>6,}", style="bold")
            section.append(f" {score_pct:>3.0f}%", style="cyan")
            if self.state.score_rate:
                section.append(f" {self.state.score_rate:>5.1f}/s", style="dim")

        return section

    def _build_activity_section(self) -> Text:
        """Build the current activity section showing what's happening now.

        Format:
          SCAN <path>
            <stats>
          SCORE <path>  <score> <reason>
        """
        section = Text()

        scan = self.state.current_scan
        score = self.state.current_score

        # SCAN <path>
        if scan:
            section.append("  SCAN ", style="bold blue")
            section.append(clip_path(scan.path, self.WIDTH - 10), style="white")
            section.append("\n")
            # Stats indented below
            section.append("    ", style="dim")
            section.append(f"{scan.files} files", style="cyan")
            section.append(", ", style="dim")
            section.append(f"{scan.dirs} dirs", style="cyan")
            if scan.has_code:
                section.append("  ", style="dim")
                section.append("code project", style="green dim")
            section.append("\n")

        # SCORE <path>
        #   <score> <reason>
        if score:
            section.append("  SCORE ", style="bold green")
            section.append(clip_path(score.path, self.WIDTH - 10), style="white")
            section.append("\n")
            # Score details indented below (matching SCAN layout)
            section.append("    ", style="dim")

            if score.skipped:
                section.append("skipped", style="yellow")
                if score.skip_reason:
                    # Clip reason to fit display
                    reason = (
                        score.skip_reason[:40] + "..."
                        if len(score.skip_reason) > 40
                        else score.skip_reason
                    )
                    section.append(f" ({reason})", style="dim")
            elif score.score is not None:
                # Color code the score
                if score.score >= 0.7:
                    style = "bold green"
                elif score.score >= 0.4:
                    style = "yellow"
                else:
                    style = "red"
                section.append(f"{score.score:.2f}", style=style)
                if score.purpose:
                    section.append(f"  {score.purpose}", style="italic dim")

        # Fallback if nothing is happening
        if not scan and not score:
            section.append("  ", style="dim")
            section.append("Initializing...", style="italic dim")

        return section

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges."""
        section = Text()

        # Cost gauge with ETA (hidden in scan_only mode)
        if not self.state.scan_only:
            est_cost = self.state.estimated_total_cost
            cost_limit = est_cost if est_cost else self.state.cost_limit
            section.append("  COST  ", style="bold yellow")
            section.append_text(
                make_resource_gauge(self.state.run_cost, cost_limit, self.GAUGE_WIDTH)
            )
            section.append(f"  ${self.state.run_cost:.2f}", style="bold")
            section.append(f" / ${self.state.cost_limit:.2f}", style="dim")
            # Show estimated total cost
            if est_cost is not None and est_cost > self.state.run_cost:
                section.append(f"  Est ${est_cost:.2f}", style="dim")
            section.append("\n")

        # Time with ETA
        section.append("  TIME  ", style="bold cyan")

        # Estimate total time if we have an ETA (score_only mode only)
        eta = None if self.state.scan_only else self.state.eta_seconds
        if eta is not None and eta > 0:
            total_est = self.state.elapsed + eta
            self.state.elapsed / total_est
            section.append_text(
                make_resource_gauge(self.state.elapsed, total_est, self.GAUGE_WIDTH)
            )
        else:
            # Unknown total - show elapsed only with thin bar
            section.append("│", style="dim")
            section.append("━" * self.GAUGE_WIDTH, style="cyan")
            section.append("│", style="dim")

        section.append(f"  {format_time(self.state.elapsed)}", style="bold")

        if eta is not None:
            if eta <= 0:
                section.append("  done", style="green dim")
            else:
                section.append(f"  ETA {format_time(eta)}", style="dim")

        # Frontier and depth metrics
        section.append("\n")
        section.append("  STATS ", style="bold magenta")
        section.append(f"frontier={self.state.frontier_size}", style="cyan")
        section.append(f"  depth={self.state.max_depth}", style="cyan")
        section.append(f"  skipped={self.state.skipped}", style="yellow")
        section.append(f"  excluded={self.state.excluded}", style="dim")

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

    def __enter__(self) -> ParallelProgressDisplay:
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

    def update_scan(
        self,
        message: str,
        stats: WorkerStats,
        paths: list[str] | None = None,
        scan_results: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update scanner state."""
        self.state.run_scanned = stats.processed
        self.state.scan_rate = stats.rate

        # Queue scan results for streaming
        if scan_results:
            items = [
                ScanItem(
                    path=r.get("path", ""),
                    files=r.get("total_files", 0),
                    dirs=r.get("total_dirs", 0),
                    has_code=r.get("has_readme")
                    or r.get("has_makefile")
                    or r.get("has_git", False),
                )
                for r in scan_results
            ]
            self.state.scan_queue.add(items, stats.rate)

        # Pop next item
        next_item = self.state.scan_queue.pop()
        if next_item:
            self.state.current_scan = next_item

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
        self.state.run_cost = stats.cost

        # Queue score results for streaming
        if results:
            items = []
            for r in results:
                path = r.get("path", "")
                self.state.scored_paths.add(path)
                items.append(
                    ScoreItem(
                        path=path,
                        score=r.get("score"),
                        purpose=r.get("label", "") or r.get("path_purpose", ""),
                        skipped=bool(r.get("skip_reason")),
                        skip_reason=r.get("skip_reason", ""),
                    )
                )
            self.state.score_queue.add(items, stats.rate)

        # Pop next item
        next_item = self.state.score_queue.pop()
        if next_item:
            self.state.current_score = next_item

        self._refresh()

    def refresh_from_graph(self, facility: str) -> None:
        """Refresh totals from graph database."""
        from imas_codex.discovery.frontier import get_discovery_stats

        stats = get_discovery_stats(facility)
        self.state.total = stats["total"]
        self.state.discovered = stats["discovered"]
        self.state.listed = stats["listed"]
        self.state.scored = stats["scored"]
        self.state.skipped = stats["skipped"]
        self.state.excluded = stats["excluded"]
        self.state.max_depth = stats["max_depth"]

        self._refresh()

    def tick(self) -> None:
        """Drain streaming queues for smooth display."""
        updated = False

        next_scan = self.state.scan_queue.pop()
        if next_scan:
            self.state.current_scan = next_scan
            updated = True

        next_score = self.state.score_queue.pop()
        if next_score:
            self.state.current_score = next_score
            updated = True

        if updated:
            self._refresh()

    def get_paths_scored_this_run(self) -> set[str]:
        """Get paths scored during this run."""
        return self.state.scored_paths

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_display())
