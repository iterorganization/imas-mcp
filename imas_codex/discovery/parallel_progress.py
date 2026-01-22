"""
Dual-panel progress display for parallel discovery.

Shows scan and score workers side-by-side with:
- Thin progress bars based on graph state
- Streaming ticker showing recent paths processed
- Overall progress and cost tracking

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


def clip_path(path: str, max_len: int = 35) -> str:
    """Clip middle of path if too long: /home/user/.../deep/dir"""
    if len(path) <= max_len:
        return path
    # Keep first ~15 chars and last ~17 chars
    keep_start = max_len // 3
    keep_end = max_len - keep_start - 3  # Account for "..."
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

    Unwraps batched results at a steady rate to give continuous feedback.
    Skips items if the rate exceeds max_rate to keep display stable.
    """

    def __init__(self, max_lines: int = 3, max_rate: float = 4.0) -> None:
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
        self._pending: deque[StreamItem] = deque(maxlen=100)  # Buffer for batches

    def add_batch(self, items: list[StreamItem]) -> None:
        """Add a batch of items to be streamed.

        Items will be emitted at the configured rate.
        """
        self._pending.extend(items)

    def add(self, item: StreamItem) -> None:
        """Add a single item."""
        self._pending.append(item)

    def tick(self) -> None:
        """Emit pending items at the configured rate."""
        now = time.time()
        while self._pending and (now - self._last_emit) >= self.min_interval:
            item = self._pending.popleft()
            self._items.append(item)
            self._last_emit = now
            now = time.time()

    def get_lines(self) -> list[Text]:
        """Get current ticker lines as Rich Text objects."""
        self.tick()  # Process any pending items
        lines = []
        for item in self._items:
            text = Text()
            clipped = clip_path(item.path)
            text.append(clipped, style="dim")
            if item.score is not None:
                text.append(" → ", style="dim")
                # Color based on score
                if item.score >= 0.7:
                    score_style = "green"
                elif item.score >= 0.4:
                    score_style = "yellow"
                else:
                    score_style = "red"
                text.append(f"{item.score:.2f}", style=score_style)
                if item.label:
                    text.append(f" {item.label}", style="dim italic")
            lines.append(text)
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

    @property
    def scan_work_remaining(self) -> int:
        """Paths awaiting scan (pending)."""
        return self.pending

    @property
    def score_work_remaining(self) -> int:
        """Paths awaiting score (scanned but not scored)."""
        return max(0, self.scanned - self.scored)


class ParallelProgressDisplay:
    """Dual-panel progress display for parallel scan/score workers.

    Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │         Discovering <facility> filesystem                    │
    ├──────────────────────────┬──────────────────────────────────┤
    │  ● Scanner               │  ● Scorer                        │
    │  [━━━━━░░░░░] 45%        │  [━━━░░░░░░░] 32%                │
    │  Done: 523  Rem: 638     │  Done: 412  Rem: 111             │
    │  Rate: 15.3/s            │  Rate: 8.2/s                     │
    │  ─────────────────       │  ─────────────────               │
    │  /home/user/.../code     │  /work/imas → 0.85 code          │
    │  /tmp/scratch            │  /tmp/scratch → 0.12 skip        │
    ├──────────────────────────┴──────────────────────────────────┤
    │  Overall: [━━━━░░░░░░] 38%  (412/1,100 scored)              │
    │  Cost: $3.45 / $10.00 (34.5%)  Elapsed: 2m 15s              │
    └─────────────────────────────────────────────────────────────┘
    """

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

        # Streaming tickers for each worker
        self._scan_ticker = StreamingTicker(max_lines=2, max_rate=4.0)
        self._score_ticker = StreamingTicker(max_lines=2, max_rate=4.0)

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
        """Build a single worker status panel with progress bar and ticker."""
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

        # Build content with table grid
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="dim")
        table.add_column(justify="left")

        # Status row
        table.add_row("Status:", f"[{status_style}]{status}[/{status_style}]")

        # Thin progress bar
        total = processed + remaining
        bar = self._make_thin_bar(processed, total, color)
        table.add_row("Progress:", bar)

        # Compact stats row
        rate_str = f"{rate:.1f}/s" if rate else "—"
        stats_text = Text()
        stats_text.append(f"{processed:,}", style="bold")
        stats_text.append(" done  ", style="dim")
        stats_text.append(f"{remaining:,}", style="bold")
        stats_text.append(" left  ", style="dim")
        stats_text.append(rate_str, style="cyan")
        table.add_row("", stats_text)

        # Ticker separator
        table.add_row("", Text("─" * 32, style="dim"))

        # Streaming ticker lines
        ticker_lines = ticker.get_lines()
        if ticker_lines:
            for line in ticker_lines:
                table.add_row("", line)
        else:
            table.add_row("", Text("waiting...", style="dim italic"))

        return Panel(
            table,
            title=f"{indicator} {title}",
            border_style=color,
            width=40,
            height=10,  # Fixed height for stable layout
        )

    def _make_thin_bar(self, completed: int, total: int, color: str) -> Text:
        """Create a thin progress bar using Unicode characters."""
        width = 20
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
        bar.append(f" {pct * 100:.0f}%", style="dim")
        return bar

    def _build_summary_row(self) -> Table:
        """Build the bottom summary row with overall progress."""
        table = Table.grid(padding=(0, 2))
        table.add_column()

        # Overall progress bar
        overall_bar = self._make_thin_bar(
            self.state.scored, self.state.total_paths, "magenta"
        )
        progress_text = Text()
        progress_text.append("Overall: ", style="bold")
        progress_text.append_text(overall_bar)
        progress_text.append(
            f"  ({self.state.scored:,}/{self.state.total_paths:,} scored)", style="dim"
        )
        table.add_row(progress_text)

        # Graph flow
        graph_text = Text()
        graph_text.append("Flow: ", style="bold")
        graph_text.append(f"{self.state.pending:,}", style="cyan")
        graph_text.append(" pending → ", style="dim")
        graph_text.append(f"{self.state.scanned:,}", style="blue")
        graph_text.append(" scanned → ", style="dim")
        graph_text.append(f"{self.state.scored:,}", style="green")
        graph_text.append(" scored", style="dim")
        table.add_row(graph_text)

        # Cost and time
        cost_pct = (
            (self.state.total_cost / self.state.cost_limit * 100)
            if self.state.cost_limit > 0
            else 0
        )
        cost_text = Text()
        cost_text.append("Cost: ", style="bold")
        cost_text.append(f"${self.state.total_cost:.2f}", style="yellow")
        cost_text.append(f" / ${self.state.cost_limit:.2f}")
        cost_text.append(f" ({cost_pct:.1f}%)", style="dim")
        cost_text.append("  Elapsed: ", style="dim")
        cost_text.append(self.state.elapsed_str, style="cyan")
        if self.state.model:
            model_display = (
                self.state.model.split("/")[-1]
                if "/" in self.state.model
                else self.state.model
            )
            cost_text.append("  Model: ", style="dim")
            cost_text.append(model_display, style="dim")
        table.add_row(cost_text)

        return table

    def _build_display(self) -> Panel:
        """Build the complete dual-panel display."""
        # Worker panels side by side
        workers = Table.grid(padding=(0, 2))
        workers.add_column()
        workers.add_column()

        # Scanner: remaining = pending paths
        scan_panel = self._build_worker_panel(
            title="Scanner",
            status=self.state.scan_status,
            processed=self.state.scanned,
            remaining=self.state.pending,
            rate=self.state.scan_rate,
            ticker=self._scan_ticker,
            color="blue",
        )

        # Scorer: remaining = scanned but not scored
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

        workers.add_row(scan_panel, score_panel)

        # Combine with summary
        full_display = Group(
            workers,
            Panel(self._build_summary_row(), border_style="dim", width=86),
        )

        title = f"Discovering {self.state.facility} filesystem"
        return Panel(full_display, title=title, border_style="cyan", width=90)

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
        """Update scanner status.

        Args:
            message: Status message (e.g., "scanning 50 paths", "idle")
            stats: Worker statistics
            paths: Optional list of paths just scanned (for ticker)
        """
        self.state.scan_status = message.split()[0] if message else "idle"
        self.state.total_scanned = stats.processed
        self.state.scan_rate = stats.rate

        # Add paths to ticker
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
        """Update scorer status.

        Args:
            message: Status message (e.g., "scored 25 ($0.021)", "waiting")
            stats: Worker statistics
            results: Optional list of score results (for ticker)
        """
        self.state.score_status = message.split()[0] if message else "waiting"
        self.state.total_scored = stats.processed
        self.state.score_rate = stats.rate
        self.state.total_cost = stats.cost

        # Add results to ticker
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
        """Refresh graph state from database.

        Args:
            facility: Facility ID to query
        """
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
