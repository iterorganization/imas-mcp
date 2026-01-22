"""
Dual-panel progress display for parallel discovery.

Shows scan and score workers side-by-side with independent progress tracking.
Each panel displays:
- Worker status (scanning/scoring/idle/waiting)
- Count processed
- Rate (items/second)
- Last batch info

The graph state is shown in a summary row at the bottom.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from imas_codex.discovery.parallel import WorkerStats


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
    """Dual-panel progress display for parallel scan/score workers.

    Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │         Discovering <facility> filesystem                    │
    ├──────────────────────────┬──────────────────────────────────┤
    │  ● Scanner               │  ● Scorer                        │
    │  Status: scanning        │  Status: scoring                 │
    │  Processed: 523          │  Processed: 412                  │
    │  Rate: 15.3/s            │  Rate: 8.2/s                     │
    │  Last: scanned 50        │  Last: scored 25 ($0.021)        │
    ├──────────────────────────┴──────────────────────────────────┤
    │  Graph: 1523 total | 245 pending | 523 scanned | 412 scored │
    │  Cost: $3.45 / $10.00 (34.5%)  Elapsed: 2m 15s              │
    └─────────────────────────────────────────────────────────────┘

    Usage:
        with ParallelProgressDisplay("iter", 10.0) as display:
            def on_scan(msg, stats):
                display.update_scan(msg, stats)
            await run_parallel_discovery(..., on_scan_progress=on_scan)
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
        self._last_scan_msg: str = ""
        self._last_score_msg: str = ""

    def _build_worker_panel(
        self,
        title: str,
        status: str,
        processed: int,
        rate: float | None,
        last_msg: str,
        style: str = "blue",
    ) -> Panel:
        """Build a single worker status panel."""
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

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="dim")
        table.add_column(justify="left")

        table.add_row("Status:", f"[{status_style}]{status}[/{status_style}]")
        table.add_row("Processed:", f"{processed:,}")
        rate_str = f"{rate:.1f}/s" if rate else "—"
        table.add_row("Rate:", rate_str)
        if last_msg:
            # Truncate long messages
            if len(last_msg) > 30:
                last_msg = last_msg[:27] + "..."
            table.add_row("Last:", f"[dim]{last_msg}[/dim]")

        return Panel(
            table,
            title=f"{indicator} {title}",
            border_style=style,
            width=35,
        )

    def _build_summary_row(self) -> Table:
        """Build the bottom summary row."""
        table = Table.grid(padding=(0, 2))
        table.add_column()

        # Graph state
        graph_text = Text()
        graph_text.append("Graph: ", style="bold")
        graph_text.append(f"{self.state.total_paths:,} total", style="white")
        graph_text.append(" | ")
        graph_text.append(f"{self.state.pending:,} pending", style="cyan")
        graph_text.append(" | ")
        graph_text.append(f"{self.state.scanned:,} scanned", style="blue")
        graph_text.append(" | ")
        graph_text.append(f"{self.state.scored:,} scored", style="green")

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
        cost_text.append(f" ({cost_pct:.1f}%)")
        cost_text.append("  Elapsed: ")
        cost_text.append(self.state.elapsed_str, style="cyan")

        table.add_row(cost_text)

        # Model (abbreviated)
        if self.state.model:
            model_display = self.state.model
            if "/" in model_display:
                model_display = model_display.split("/")[-1]
            model_text = Text()
            model_text.append("Model: ", style="bold dim")
            model_text.append(model_display, style="dim")
            table.add_row(model_text)

        return table

    def _build_display(self) -> Panel:
        """Build the complete dual-panel display."""
        # Worker panels side by side
        workers = Table.grid(padding=(0, 2))
        workers.add_column()
        workers.add_column()

        scan_panel = self._build_worker_panel(
            title="Scanner",
            status=self.state.scan_status,
            processed=self.state.total_scanned,
            rate=self.state.scan_rate,
            last_msg=self._last_scan_msg,
            style="blue",
        )

        score_panel = self._build_worker_panel(
            title="Scorer",
            status=self.state.score_status,
            processed=self.state.total_scored,
            rate=self.state.score_rate,
            last_msg=self._last_score_msg,
            style="green",
        )

        workers.add_row(scan_panel, score_panel)

        # Combine with summary
        full_display = Group(
            workers,
            Panel(self._build_summary_row(), border_style="dim"),
        )

        title = f"Discovering {self.state.facility} filesystem"
        return Panel(full_display, title=title, border_style="cyan")

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

    def update_scan(self, message: str, stats: WorkerStats) -> None:
        """Update scanner status.

        Args:
            message: Status message (e.g., "scanning 50 paths", "idle")
            stats: Worker statistics
        """
        self.state.scan_status = message.split()[0] if message else "idle"
        self.state.total_scanned = stats.processed
        self.state.scan_rate = stats.rate
        self._last_scan_msg = message

        self._refresh()

    def update_score(self, message: str, stats: WorkerStats) -> None:
        """Update scorer status.

        Args:
            message: Status message (e.g., "scored 25 ($0.021)", "waiting")
            stats: Worker statistics
        """
        self.state.score_status = message.split()[0] if message else "waiting"
        self.state.total_scored = stats.processed
        self.state.score_rate = stats.rate
        self.state.total_cost = stats.cost
        self._last_score_msg = message

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

    def print_summary(self) -> None:
        """Print final summary after display closes."""
        self.console.print()
        self.console.print("[bold]Discovery Summary[/bold]")
        self.console.print(f"  Facility: {self.state.facility}")
        self.console.print(f"  Duration: {self.state.elapsed_str}")
        self.console.print(f"  Scanned: {self.state.total_scanned:,}")
        self.console.print(f"  Scored: {self.state.total_scored:,}")
        self.console.print(f"  Total cost: ${self.state.total_cost:.3f}")
        if self.state.scan_rate:
            self.console.print(f"  Scan rate: {self.state.scan_rate:.1f}/s")
        if self.state.score_rate:
            self.console.print(f"  Score rate: {self.state.score_rate:.1f}/s")
