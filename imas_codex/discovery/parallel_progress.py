"""
Dual-panel progress display for parallel discovery.

Shows scan and score workers side-by-side with:
- Single current path display with stats/details
- Full-width progress bars
- Summary statistics panel

Design: Each worker panel shows only the current item with rich detail,
not a scrolling history. Stats are shown inline for immediate feedback.

Streaming: Items are queued and released at the observed rate to give
the impression of continuous processing rather than batch updates.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from imas_codex.discovery.parallel import WorkerStats


def clip_path(path: str, max_len: int = 60) -> str:
    """Clip middle of path if too long: /home/user/.../deep/dir"""
    if len(path) <= max_len:
        return path
    keep_start = max_len // 3
    keep_end = max_len - keep_start - 3
    return f"{path[:keep_start]}...{path[-keep_end:]}"


def format_size(bytes_val: int | None) -> str:
    """Format bytes as human-readable size."""
    if bytes_val is None or bytes_val == 0:
        return "-"
    size = float(bytes_val)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


@dataclass
class ScanDisplayItem:
    """Current item being displayed in scanner panel."""

    path: str
    total_files: int = 0
    total_dirs: int = 0
    file_types: dict[str, int] = field(default_factory=dict)
    has_readme: bool = False
    has_makefile: bool = False
    has_git: bool = False


@dataclass
class ScoreDisplayItem:
    """Current item being displayed in scorer panel."""

    path: str
    score: float | None = None
    purpose: str = ""
    description: str = ""
    score_code: float = 0.0
    score_data: float = 0.0
    score_imas: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class StreamingQueue:
    """Queue that releases items at an approximate rate.

    When a batch arrives, items are queued and released one at a time
    based on the observed rate, giving impression of continuous streaming.
    """

    items: deque = field(default_factory=deque)
    last_release_time: float = field(default_factory=time.time)
    observed_rate: float = 1.0  # items per second
    min_interval: float = 0.1  # minimum seconds between releases

    def add_batch(self, items: list, rate: float | None = None) -> None:
        """Add a batch of items to the queue."""
        self.items.extend(items)
        if rate and rate > 0:
            self.observed_rate = rate

    def pop_ready(self) -> Any | None:
        """Pop an item if enough time has passed based on rate."""
        if not self.items:
            return None

        # Calculate time between releases based on rate
        interval = (
            max(self.min_interval, 1.0 / self.observed_rate)
            if self.observed_rate > 0
            else 0.5
        )
        now = time.time()

        if now - self.last_release_time >= interval:
            self.last_release_time = now
            return self.items.popleft()
        return None

    def peek(self) -> Any | None:
        """Peek at the next item without removing it."""
        return self.items[0] if self.items else None

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class ParallelProgressState:
    """State for dual-panel progress display."""

    facility: str
    cost_limit: float
    model: str = ""

    # Graph state
    total_paths: int = 0
    pending: int = 0
    scanned: int = 0
    scored: int = 0

    # Worker status
    scan_status: str = "starting"
    score_status: str = "starting"

    # Accumulated stats
    total_scanned: int = 0
    total_scored: int = 0
    total_cost: float = 0.0
    scan_rate: float | None = None
    score_rate: float | None = None

    # Current display items
    current_scan: ScanDisplayItem | None = None
    current_score: ScoreDisplayItem | None = None

    # Streaming queues for smooth display
    scan_queue: StreamingQueue = field(default_factory=StreamingQueue)
    score_queue: StreamingQueue = field(default_factory=StreamingQueue)

    # Track paths scored in this run for filtering high-value display
    scored_this_run: set[str] = field(default_factory=set)

    # Timing
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def elapsed_str(self) -> str:
        e = self.elapsed
        if e < 60:
            return f"{e:.0f}s"
        elif e < 3600:
            return f"{e / 60:.1f}m"
        else:
            return f"{e / 3600:.1f}h"


class ParallelProgressDisplay:
    """Dual-panel progress display for parallel scan/score workers.

    Each worker shows a single current item with full details rather
    than a scrolling ticker. This provides more useful information
    about what's happening.
    """

    # Layout constants - sized for 100-col terminal to give more space
    # Full outer width = 100, outer borders = 4, inner content = 96
    # Each panel = 48 (half of inner content)
    OUTER_WIDTH = 100
    PANEL_WIDTH = 48
    CONTENT_WIDTH = 44  # Inside each panel after borders/padding
    # Fixed number of lines per panel to prevent flicker
    PANEL_LINES = 8

    def __init__(
        self,
        facility: str,
        cost_limit: float,
        model: str = "",
        console: Console | None = None,
        cumulative_cost: float = 0.0,  # Previous discovery costs for this facility
    ) -> None:
        self.console = console or Console()
        self.state = ParallelProgressState(
            facility=facility,
            cost_limit=cost_limit,
            model=model,
        )
        self.cumulative_cost = cumulative_cost  # Total historical cost
        self._live: Live | None = None

    def _make_full_bar(
        self, completed: int, total: int, color: str, width: int
    ) -> Text:
        """Create a thin progress bar spanning specified width."""
        if total <= 0:
            return Text("─" * width, style="dim")

        pct = min(1.0, completed / total)
        filled = int(width * pct)
        partial = (width * pct) - filled
        empty = width - filled - (1 if partial > 0.5 else 0)

        bar = Text()
        bar.append("━" * filled, style=color)
        if partial > 0.5:
            bar.append("╸", style=color)
        bar.append("─" * empty, style="dim")
        return bar

    def _build_scanner_panel(self) -> Panel:
        """Build scan panel with current path and stats.

        Fixed height (PANEL_LINES) to prevent flicker.
        """
        # Status indicator
        status = self.state.scan_status
        if "idle" in status.lower() or "waiting" in status.lower():
            indicator = "[yellow]○[/yellow]"
        elif "error" in status.lower():
            indicator = "[red]✗[/red]"
        else:
            indicator = "[green]●[/green]"

        lines: list[Text] = []

        # Line 1: Progress on single line - fraction | bar | %
        total = self.state.scanned + self.state.pending
        pct = (self.state.scanned / total * 100) if total > 0 else 0
        rate_str = f" {self.state.scan_rate:.1f}/s" if self.state.scan_rate else ""
        progress_line = Text()
        progress_line.append(f"{self.state.scanned:,}", style="bold")
        progress_line.append(f"/{total:,}", style="dim")
        progress_line.append(rate_str, style="cyan")
        lines.append(progress_line)

        # Line 2: Progress bar with % at end
        bar_width = self.CONTENT_WIDTH - 6  # Leave room for " 100%"
        bar = self._make_full_bar(self.state.scanned, total, "blue", bar_width)
        bar.append(f" {pct:3.0f}%", style="bold cyan")
        lines.append(bar)

        # Line 3: Separator
        lines.append(Text("─" * self.CONTENT_WIDTH, style="dim"))

        # Lines 5+: Current scan item with stats
        item = self.state.current_scan
        if item:
            # Path (clipped)
            path_line = Text()
            path_line.append(clip_path(item.path, self.CONTENT_WIDTH), style="white")
            lines.append(path_line)

            # Stats line: files/dirs + markers
            stats_line = Text()
            stats_line.append(f"{item.total_files} files", style="cyan")
            stats_line.append(" · ", style="dim")
            stats_line.append(f"{item.total_dirs} dirs", style="cyan")
            markers = []
            if item.has_readme:
                markers.append("README")
            if item.has_makefile:
                markers.append("Make")
            if item.has_git:
                markers.append("git")
            if markers:
                stats_line.append(f"  [{', '.join(markers)}]", style="green dim")
            lines.append(stats_line)
        else:
            lines.append(Text("(waiting...)", style="dim"))
            lines.append(Text(""))

        # Pad to fixed height
        while len(lines) < self.PANEL_LINES:
            lines.append(Text(""))

        # Combine into content
        content = Text()
        for i, line in enumerate(lines[: self.PANEL_LINES]):
            if i > 0:
                content.append("\n")
            content.append_text(line)

        return Panel(
            content,
            title=f"{indicator} Scan",
            border_style="blue",
            width=self.PANEL_WIDTH,
            padding=(0, 1),
        )

    def _build_scorer_panel(self) -> Panel:
        """Build score panel with current path and score details.

        Fixed height (PANEL_LINES) to prevent flicker.
        """
        status = self.state.score_status
        if "idle" in status.lower() or "waiting" in status.lower():
            indicator = "[yellow]○[/yellow]"
        elif "error" in status.lower():
            indicator = "[red]✗[/red]"
        else:
            indicator = "[green]●[/green]"

        lines: list[Text] = []

        # Line 1: Progress on single line - fraction | rate
        scanned_unscored = max(0, self.state.scanned - self.state.scored)
        total = self.state.scored + scanned_unscored
        pct = (self.state.scored / total * 100) if total > 0 else 0
        rate_str = f" {self.state.score_rate:.1f}/s" if self.state.score_rate else ""
        progress_line = Text()
        progress_line.append(f"{self.state.scored:,}", style="bold")
        progress_line.append(f"/{total:,}", style="dim")
        progress_line.append(rate_str, style="cyan")
        lines.append(progress_line)

        # Line 2: Progress bar with % at end
        bar_width = self.CONTENT_WIDTH - 6  # Leave room for " 100%"
        bar = self._make_full_bar(self.state.scored, total, "green", bar_width)
        bar.append(f" {pct:3.0f}%", style="bold cyan")
        lines.append(bar)

        # Line 3: Separator
        lines.append(Text("─" * self.CONTENT_WIDTH, style="dim"))

        # Lines 5+: Current score item with details
        item = self.state.current_score
        if item:
            # Path (clipped)
            path_line = Text()
            path_line.append(clip_path(item.path, self.CONTENT_WIDTH), style="white")
            lines.append(path_line)

            if item.skipped:
                # Show skip reason
                skip_line = Text()
                skip_line.append("→ ", style="dim")
                skip_line.append("skipped", style="yellow")
                skip_line.append(f": {item.skip_reason}", style="dim")
                lines.append(skip_line)
            elif item.score is not None:
                # Score with color and purpose
                score_line = Text()
                score_line.append("→ ", style="dim")
                if item.score >= 0.7:
                    score_style = "green bold"
                elif item.score >= 0.4:
                    score_style = "yellow"
                else:
                    score_style = "red"
                score_line.append(f"{item.score:.2f}", style=score_style)
                if item.purpose:
                    score_line.append(f"  {item.purpose}", style="italic dim")
                lines.append(score_line)

                # Dimension scores on same line
                dim_line = Text()
                dim_line.append(
                    f"code={item.score_code:.1f} data={item.score_data:.1f} imas={item.score_imas:.1f}",
                    style="dim",
                )
                lines.append(dim_line)
        else:
            lines.append(Text("(waiting...)", style="dim"))
            lines.append(Text(""))

        # Pad to fixed height
        while len(lines) < self.PANEL_LINES:
            lines.append(Text(""))

        # Combine into content
        content = Text()
        for i, line in enumerate(lines[: self.PANEL_LINES]):
            if i > 0:
                content.append("\n")
            content.append_text(line)

        return Panel(
            content,
            title=f"{indicator} Score",
            border_style="green",
            width=self.PANEL_WIDTH,
            padding=(0, 1),
        )

    def _build_summary_row(self) -> Panel:
        """Build the bottom summary row with clean stats layout.

        Two lines:
        - Line 1: scored/total | progress bar | %
        - Line 2: Budget | Time | Model | Flow
        """
        summary_width = self.PANEL_WIDTH * 2

        # Line 1: scored/total | bar | %
        pct = (
            (self.state.scored / self.state.total_paths * 100)
            if self.state.total_paths > 0
            else 0
        )
        progress = Text()
        progress.append(f"{self.state.scored:,}", style="bold green")
        progress.append(f"/{self.state.total_paths:,}", style="dim")
        progress.append(" ", style="dim")

        # Bar takes remaining width: summary_width - 4 (padding) - 14 (scored/total) - 5 (%)
        bar_width = summary_width - 25
        bar = self._make_full_bar(
            self.state.scored, self.state.total_paths, "magenta", bar_width
        )
        progress.append_text(bar)
        progress.append(f" {pct:3.0f}%", style="bold magenta")

        # Line 2: Budget | Time | Model | Flow
        this_run_cost = self.state.total_cost
        stats = Text()
        stats.append(f"${this_run_cost:.2f}", style="yellow bold")
        stats.append(f"/${self.state.cost_limit:.2f}", style="dim")
        stats.append(" │ ", style="dim")
        stats.append(f"{self.state.elapsed_str}", style="cyan")
        if self.state.model:
            # Show model name clearly
            model_short = self.state.model.replace("anthropic/", "")
            stats.append(f" {model_short}", style="dim")
        stats.append(" │ ", style="dim")
        stats.append(f"{self.state.pending:,}", style="cyan")
        stats.append("→", style="dim")
        stats.append(f"{self.state.scanned:,}", style="blue")
        stats.append("→", style="dim")
        stats.append(f"{self.state.scored:,}", style="green")

        content = Text()
        content.append_text(progress)
        content.append("\n")
        content.append_text(stats)

        return Panel(content, border_style="dim", width=summary_width)

    def _build_display(self) -> Panel:
        """Build the complete dual-panel display."""
        scan_panel = self._build_scanner_panel()
        score_panel = self._build_scorer_panel()

        # Use Table.grid for side-by-side layout
        workers = Table.grid(expand=False)
        workers.add_column(width=self.PANEL_WIDTH)
        workers.add_column(width=self.PANEL_WIDTH)
        workers.add_row(scan_panel, score_panel)

        full_display = Group(workers, self._build_summary_row())

        # Uppercase facility name for title
        facility_upper = self.state.facility.upper()
        title = f"{facility_upper} Discovery"
        return Panel(
            full_display,
            title=title,
            border_style="cyan",
            width=self.OUTER_WIDTH,
        )

    def __enter__(self) -> ParallelProgressDisplay:
        """Start live display.

        Uses:
        - vertical_overflow="visible" with fixed-height panels to prevent truncation
        - transient=False to keep display stable
        - refresh_per_second=2 to reduce flicker
        """
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=2,
            vertical_overflow="visible",
            transient=False,
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
        """Update scanner status with current path and stats.

        Items are queued and released at the observed rate to simulate
        continuous streaming rather than batch updates.

        Args:
            message: Status message
            stats: Worker stats
            paths: List of paths being scanned (shows last one)
            scan_results: List of scan result dicts with file/dir counts
        """
        self.state.scan_status = message.split()[0] if message else "idle"
        self.state.total_scanned = stats.processed
        self.state.scan_rate = stats.rate

        # Add batch to streaming queue
        if scan_results:
            display_items = [
                ScanDisplayItem(
                    path=r.get("path", ""),
                    total_files=r.get("total_files", 0),
                    total_dirs=r.get("total_dirs", 0),
                    file_types=r.get("file_types", {}),
                    has_readme=r.get("has_readme", False),
                    has_makefile=r.get("has_makefile", False),
                    has_git=r.get("has_git", False),
                )
                for r in scan_results
            ]
            self.state.scan_queue.add_batch(display_items, stats.rate)

        # Pop next item from queue if ready
        next_item = self.state.scan_queue.pop_ready()
        if next_item:
            self.state.current_scan = next_item
        elif not self.state.current_scan and paths:
            # Fallback: just show path if no detailed results
            self.state.current_scan = ScanDisplayItem(path=paths[-1])

        self._refresh()

    def update_score(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update scorer status with current path and score details.

        Items are queued and released at the observed rate to simulate
        continuous streaming rather than batch updates.

        Args:
            message: Status message
            stats: Worker stats
            results: List of score result dicts
        """
        self.state.score_status = message.split()[0] if message else "waiting"
        self.state.total_scored = stats.processed
        self.state.score_rate = stats.rate
        self.state.total_cost = stats.cost

        # Add batch to streaming queue
        if results:
            display_items = []
            for r in results:
                path = r.get("path", "")
                self.state.scored_this_run.add(path)

                skip_reason = r.get("skip_reason", "")
                purpose = r.get("label", "") or r.get("path_purpose", "")

                is_skipped = bool(skip_reason)
                if purpose == "user_home" and r.get("total_files", 0) == 0:
                    is_skipped = True
                    skip_reason = "empty user home"

                display_items.append(
                    ScoreDisplayItem(
                        path=path,
                        score=r.get("score"),
                        purpose=purpose,
                        description=r.get("description", ""),
                        score_code=r.get("score_code", 0.0),
                        score_data=r.get("score_data", 0.0),
                        score_imas=r.get("score_imas", 0.0),
                        skipped=is_skipped,
                        skip_reason=skip_reason,
                    )
                )
            self.state.score_queue.add_batch(display_items, stats.rate)

        # Pop next item from queue if ready
        next_item = self.state.score_queue.pop_ready()
        if next_item:
            self.state.current_score = next_item

        self._refresh()

    def refresh_from_graph(self, facility: str) -> None:
        """Refresh graph state from database."""
        from imas_codex.discovery.frontier import get_discovery_stats

        stats = get_discovery_stats(facility)
        self.state.total_paths = stats["total"]
        self.state.pending = stats["pending"]
        self.state.scanned = stats["scanned"]
        self.state.scored = stats["scored"]

        self._refresh()

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_display())

    def tick(self) -> None:
        """Called periodically to drain streaming queues.

        This should be called in an async loop to provide smooth
        streaming of items even between batch updates.
        """
        updated = False

        # Try to pop from scan queue
        next_scan = self.state.scan_queue.pop_ready()
        if next_scan:
            self.state.current_scan = next_scan
            updated = True

        # Try to pop from score queue
        next_score = self.state.score_queue.pop_ready()
        if next_score:
            self.state.current_score = next_score
            updated = True

        if updated:
            self._refresh()

    def get_paths_scored_this_run(self) -> set[str]:
        """Get the set of paths that were scored during this run."""
        return self.state.scored_this_run
