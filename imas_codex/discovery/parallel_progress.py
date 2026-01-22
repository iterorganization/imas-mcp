"""
Dual-panel progress display for parallel discovery.

Shows scan and score workers side-by-side with:
- Thin progress bars spanning full width
- Streaming ticker with progressive fade (new at top, fading to bottom)
- Overall progress bar stretching full width

The streaming ticker unwraps batched operations at a steady rate to give
continuous visual feedback without flooding the display.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from imas_codex.discovery.parallel import WorkerStats


def clip_path(path: str, max_len: int = 45) -> str:
    """Clip middle of path if too long: /home/user/.../deep/dir"""
    if len(path) <= max_len:
        return path
    keep_start = max_len // 3
    keep_end = max_len - keep_start - 3
    return f"{path[:keep_start]}...{path[-keep_end:]}"


@dataclass
class StreamItem:
    """Single item in the streaming ticker."""

    path: str
    score: float | None = None
    label: str | None = None
    timestamp: float = field(default_factory=time.time)


class StreamingTicker:
    """Rate-limited streaming ticker for batch results.

    New items appear at the top and scroll down with progressive fading.
    """

    def __init__(self, max_lines: int = 5, max_rate: float = 4.0) -> None:
        """Initialize ticker.

        Args:
            max_lines: Number of lines to show in ticker
            max_rate: Maximum updates per second
        """
        self.max_lines = max_lines
        self.max_rate = max_rate
        self.min_interval = 1.0 / max_rate
        self._items: deque[StreamItem] = deque(maxlen=max_lines)
        self._last_emit: float = 0.0
        self._pending: deque[StreamItem] = deque(maxlen=100)

    def add_batch(self, items: list[StreamItem]) -> None:
        """Add a batch of items to be streamed."""
        self._pending.extend(items)

    def add(self, item: StreamItem) -> None:
        """Add a single item."""
        self._pending.append(item)

    def tick(self) -> None:
        """Emit pending items at the configured rate."""
        now = time.time()
        while self._pending and (now - self._last_emit) >= self.min_interval:
            item = self._pending.popleft()
            self._items.appendleft(item)  # New items at front (top)
            self._last_emit = now
            now = time.time()

    def get_lines(self, max_path_len: int = 45) -> list[Text]:
        """Get current ticker lines with progressive fading.

        Newest items (index 0) are brightest, older items fade.
        """
        self.tick()
        lines = []

        # Opacity levels: newest=bright, oldest=very dim
        opacity_styles = ["", "dim", "dim", "dim italic", "dim italic"]

        for i, item in enumerate(self._items):
            text = Text()
            clipped = clip_path(item.path, max_path_len)

            # Get opacity style based on position (0=newest, higher=older)
            opacity = opacity_styles[min(i, len(opacity_styles) - 1)]

            if item.score is not None:
                # Score result: path → score label
                if opacity:
                    text.append(clipped, style=opacity)
                else:
                    text.append(clipped)
                text.append(" → ", style="dim")

                # Color based on score
                if item.score >= 0.7:
                    score_style = "green"
                elif item.score >= 0.4:
                    score_style = "yellow"
                else:
                    score_style = "red"

                if opacity:
                    text.append(f"{item.score:.2f}", style=f"{score_style} {opacity}")
                else:
                    text.append(f"{item.score:.2f}", style=score_style)

                if item.label:
                    text.append(
                        f" {item.label}",
                        style=f"italic {opacity}" if opacity else "italic dim",
                    )
            else:
                # Scan result: just path
                text.append(clipped, style=opacity if opacity else "")

            lines.append(text)

        # Pad with empty lines if needed
        while len(lines) < self.max_lines:
            lines.append(Text("", style="dim"))

        return lines


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
    """Dual-panel progress display for parallel scan/score workers."""

    # Layout constants - sized for 80-col terminal
    # Outer panel uses 4 chars (2 border + 2 padding)
    # Each worker panel uses 4 chars (2 border + 2 padding)
    # Available for content = 80 - 4 (outer) = 76 → 38 per panel
    # Content inside each panel = 38 - 4 = 34
    PANEL_WIDTH = 38  # Each worker panel width including borders
    TICKER_LINES = 5
    CONTENT_WIDTH = 34  # Usable content area inside each panel

    def __init__(
        self,
        facility: str,
        cost_limit: float,
        model: str = "",
        console: Console | None = None,
    ) -> None:
        self.console = console or Console()
        self.state = ParallelProgressState(
            facility=facility,
            cost_limit=cost_limit,
            model=model,
        )
        self._live: Live | None = None

        # Streaming tickers for each worker (5 lines, scrolling at ~4/s)
        self._scan_ticker = StreamingTicker(max_lines=self.TICKER_LINES, max_rate=4.0)
        self._score_ticker = StreamingTicker(max_lines=self.TICKER_LINES, max_rate=4.0)

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

    def _build_worker_panel(
        self,
        title: str,
        status: str,
        processed: int,
        remaining: int,
        rate: float | None,
        ticker: StreamingTicker,
        color: str = "blue",
    ) -> Panel:
        """Build a worker status panel with full-width bar and streaming ticker."""
        # Status indicator
        if "idle" in status.lower() or "waiting" in status.lower():
            indicator = "[yellow]○[/yellow]"
            status_style = "yellow"
        elif "error" in status.lower():
            indicator = "[red]✗[/red]"
            status_style = "red"
        else:
            indicator = "[green]●[/green]"
            status_style = "green"

        lines = []

        # Line 1: Status and rate
        status_line = Text()
        status_line.append(status, style=status_style)
        if rate:
            status_line.append(f"  {rate:.1f}/s", style="cyan")
        lines.append(status_line)

        # Line 2: Progress fraction (done/total) right-aligned
        total = processed + remaining
        pct = (processed / total * 100) if total > 0 else 0
        progress_line = Text()
        progress_line.append("Progress: ", style="dim")
        progress_line.append(f"{processed:,}", style="bold")
        progress_line.append("/", style="dim")
        progress_line.append(f"{total:,}", style="dim")
        progress_line.append(f"  {pct:.0f}%", style="bold")
        lines.append(progress_line)

        # Line 3: Full-width progress bar
        bar = self._make_full_bar(processed, total, color, self.CONTENT_WIDTH)
        lines.append(bar)

        # Line 4: Separator
        lines.append(Text("─" * self.CONTENT_WIDTH, style="dim"))

        # Lines 5+: Streaming ticker (5 lines, newest at top, fading down)
        ticker_lines = ticker.get_lines(max_path_len=self.CONTENT_WIDTH)
        lines.extend(ticker_lines)

        # Combine into content
        content = Text()
        for i, line in enumerate(lines):
            if i > 0:
                content.append("\n")
            content.append_text(line)

        return Panel(
            content,
            title=f"{indicator} {title}",
            border_style=color,
            width=self.PANEL_WIDTH,
            padding=(0, 1),  # Vertical=0, horizontal=1
        )

    def _build_summary_row(self) -> Panel:
        """Build the bottom summary row with full-width overall progress."""
        # Summary panel must fit inside outer panel's content area
        # Outer panel width = (PANEL_WIDTH * 2) + 4 = 80
        # Outer content area = 80 - 4 (borders + padding) = 76
        summary_width = self.PANEL_WIDTH * 2  # 76 = fits in outer content
        # Content width inside summary = summary_width - 4 (borders + padding)
        full_bar_width = summary_width - 4  # 72

        lines = []

        # Line 1: Overall progress bar (full width)
        pct = (
            (self.state.scored / self.state.total_paths * 100)
            if self.state.total_paths > 0
            else 0
        )
        overall_line = Text()
        overall_line.append("Overall: ", style="bold")
        overall_line.append(f"{self.state.scored:,}", style="bold green")
        overall_line.append("/", style="dim")
        overall_line.append(f"{self.state.total_paths:,}", style="dim")
        overall_line.append(f"  {pct:.1f}%", style="bold magenta")
        lines.append(overall_line)

        # Full-width bar
        bar = self._make_full_bar(
            self.state.scored, self.state.total_paths, "magenta", full_bar_width
        )
        lines.append(bar)

        # Line 2: Graph flow
        flow_line = Text()
        flow_line.append("Flow: ", style="bold")
        flow_line.append(f"{self.state.pending:,}", style="cyan")
        flow_line.append(" pending → ", style="dim")
        flow_line.append(f"{self.state.scanned:,}", style="blue")
        flow_line.append(" scanned → ", style="dim")
        flow_line.append(f"{self.state.scored:,}", style="green")
        flow_line.append(" scored", style="dim")
        lines.append(flow_line)

        # Line 3: Cost, time, model
        cost_pct = (
            (self.state.total_cost / self.state.cost_limit * 100)
            if self.state.cost_limit > 0
            else 0
        )
        cost_line = Text()
        cost_line.append("Cost: ", style="bold")
        cost_line.append(f"${self.state.total_cost:.2f}", style="yellow")
        cost_line.append(f"/${self.state.cost_limit:.2f}", style="dim")
        cost_line.append(f" ({cost_pct:.0f}%)", style="dim")
        cost_line.append("  Elapsed: ", style="dim")
        cost_line.append(self.state.elapsed_str, style="cyan")
        if self.state.model:
            model_display = (
                self.state.model.split("/")[-1]
                if "/" in self.state.model
                else self.state.model
            )
            cost_line.append(f"  {model_display}", style="dim")
        lines.append(cost_line)

        # Combine
        content = Text()
        for i, line in enumerate(lines):
            if i > 0:
                content.append("\n")
            content.append_text(line)

        return Panel(content, border_style="dim", width=summary_width)

    def _build_display(self) -> Panel:
        """Build the complete dual-panel display."""
        # Worker panels side by side using Columns for proper width handling
        scan_panel = self._build_worker_panel(
            title="Scanner",
            status=self.state.scan_status,
            processed=self.state.scanned,
            remaining=self.state.pending,
            rate=self.state.scan_rate,
            ticker=self._scan_ticker,
            color="blue",
        )

        scanned_unscored = max(0, self.state.scanned - self.state.scored)
        score_panel = self._build_worker_panel(
            title="Scorer",
            status=self.state.score_status,
            processed=self.state.scored,
            remaining=scanned_unscored,
            rate=self.state.score_rate,
            ticker=self._score_ticker,
            color="green",
        )

        # Use Table.grid for side-by-side layout
        workers = Table.grid(expand=False)
        workers.add_column(width=self.PANEL_WIDTH)
        workers.add_column(width=self.PANEL_WIDTH)
        workers.add_row(scan_panel, score_panel)

        full_display = Group(workers, self._build_summary_row())

        title = f"Discovering {self.state.facility} filesystem"
        # Outer panel must fit both worker panels (2 x PANEL_WIDTH) + outer borders/padding
        outer_width = (self.PANEL_WIDTH * 2) + 4
        return Panel(
            full_display,
            title=title,
            border_style="cyan",
            width=outer_width,
        )

    def __enter__(self) -> ParallelProgressDisplay:
        """Start live display."""
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
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
    ) -> None:
        """Update scanner status."""
        self.state.scan_status = message.split()[0] if message else "idle"
        self.state.total_scanned = stats.processed
        self.state.scan_rate = stats.rate

        if paths:
            items = [StreamItem(path=p) for p in paths]
            self._scan_ticker.add_batch(items)

        self._refresh()

    def update_score(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update scorer status."""
        self.state.score_status = message.split()[0] if message else "waiting"
        self.state.total_scored = stats.processed
        self.state.score_rate = stats.rate
        self.state.total_cost = stats.cost

        if results:
            items = [
                StreamItem(
                    path=r.get("path", ""),
                    score=r.get("score"),
                    label=r.get("label", ""),
                )
                for r in results
            ]
            self._score_ticker.add_batch(items)

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
