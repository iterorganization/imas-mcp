"""
Rich progress display for discovery operations.

Provides a comprehensive progress UI showing:
- Overview panel with counts and coverage
- Progress bar for current operation
- Budget tracking (for score/discover)
- Frontier size indicator
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    pass


@dataclass
class DiscoveryStats:
    """Live statistics for discovery progress."""

    # Counts
    total_paths: int = 0
    pending: int = 0
    scanned: int = 0
    scored: int = 0
    skipped: int = 0

    # Current operation
    current_phase: str = "idle"  # scan, score, idle
    current_path: str = ""

    # Budget tracking (score phase)
    accumulated_cost: float = 0.0
    budget_limit: float | None = None
    model: str = ""

    # Cycle tracking (discover command)
    current_cycle: int = 0
    max_cycles: int = 0

    @property
    def frontier_size(self) -> int:
        """Paths awaiting scan."""
        return self.pending

    @property
    def completion_fraction(self) -> float:
        """Fraction of known paths that are scored."""
        if self.total_paths == 0:
            return 0.0
        return self.scored / self.total_paths

    @property
    def budget_fraction(self) -> float:
        """Fraction of budget used."""
        if not self.budget_limit:
            return 0.0
        return self.accumulated_cost / self.budget_limit


class DiscoveryProgressDisplay:
    """Rich progress display for discovery operations.

    Shows:
    - Overview panel with counts and coverage
    - Progress bar for current operation
    - Budget tracking (for score/discover)
    - Frontier size indicator

    Usage:
        with DiscoveryProgressDisplay() as display:
            display.update(description="Scanning...", advance=1)
            display.refresh_from_graph("iter")
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.stats = DiscoveryStats()
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_id = None

    def _build_overview_panel(self) -> Panel:
        """Build the overview statistics panel."""
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold")
        table.add_column(justify="left")
        table.add_column(justify="right", style="bold")
        table.add_column(justify="left")

        # Row 1: Counts
        table.add_row(
            "Total:",
            f"{self.stats.total_paths:,}",
            "Frontier:",
            f"[cyan]{self.stats.frontier_size:,}[/cyan]",
        )

        # Row 2: Status breakdown
        table.add_row(
            "Scanned:",
            f"{self.stats.scanned:,}",
            "Scored:",
            f"[green]{self.stats.scored:,}[/green]",
        )

        # Row 3: Coverage and skipped
        coverage_pct = self.stats.completion_fraction * 100
        table.add_row(
            "Skipped:",
            f"{self.stats.skipped:,}",
            "Coverage:",
            f"{coverage_pct:.1f}%",
        )

        # Row 4: Budget (if applicable)
        if self.stats.budget_limit:
            budget_pct = self.stats.budget_fraction * 100
            table.add_row(
                "Spent:",
                f"${self.stats.accumulated_cost:.2f}",
                "Budget:",
                f"${self.stats.budget_limit:.2f} ({budget_pct:.0f}%)",
            )
            table.add_row(
                "Model:",
                self.stats.model,
                "",
                "",
            )

        # Row 5: Cycle (if in discover mode)
        if self.stats.max_cycles > 0:
            table.add_row(
                "Cycle:",
                f"{self.stats.current_cycle}/{self.stats.max_cycles}",
                "Phase:",
                f"[yellow]{self.stats.current_phase}[/yellow]",
            )

        return Panel(table, title="Discovery Progress", border_style="blue")

    def _build_display(self) -> Group:
        """Build complete display."""
        return Group(self._build_overview_panel(), self._progress)

    def __enter__(self) -> DiscoveryProgressDisplay:
        """Start live display."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self._task_id = self._progress.add_task("Starting...", total=100)

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

    def update(
        self,
        description: str | None = None,
        advance: int = 0,
        total: int | None = None,
        **stats_updates,
    ) -> None:
        """Update progress and stats.

        Args:
            description: Progress bar description
            advance: Number of steps to advance
            total: New total for progress bar
            **stats_updates: Updates to stats fields (e.g., scanned=10)
        """
        # Update stats
        for key, value in stats_updates.items():
            if hasattr(self.stats, key):
                setattr(self.stats, key, value)

        # Update progress bar
        if self._progress and self._task_id is not None:
            if description:
                self._progress.update(self._task_id, description=description)
            if total is not None:
                self._progress.update(self._task_id, total=total)
            if advance:
                self._progress.advance(self._task_id, advance)

        # Refresh display
        if self._live:
            self._live.update(self._build_display())

    def refresh_from_graph(self, facility: str) -> None:
        """Refresh stats from graph state.

        Args:
            facility: Facility ID to query
        """
        from imas_codex.discovery.frontier import get_discovery_stats

        stats = get_discovery_stats(facility)
        self.stats.total_paths = stats["total"]
        self.stats.pending = stats["pending"]
        self.stats.scanned = stats["scanned"]
        self.stats.scored = stats["scored"]
        self.stats.skipped = stats["skipped"]

        # Refresh display
        if self._live:
            self._live.update(self._build_display())


def print_discovery_status(facility: str, console: Console | None = None) -> None:
    """Print a formatted discovery status report.

    Args:
        facility: Facility ID
        console: Optional Rich console
    """
    from imas_codex.discovery.frontier import get_discovery_stats, get_high_value_paths

    console = console or Console()
    stats = get_discovery_stats(facility)

    # Header
    console.print(f"\n[bold]Facility: {facility}[/bold]")
    console.print(f"Total paths: {stats['total']:,}")

    # Status breakdown
    total = stats["total"] or 1  # Avoid division by zero
    console.print(
        f"├─ Pending:   {stats['pending']:,} ({stats['pending'] / total * 100:.1f}%)"
    )
    console.print(
        f"├─ Scanned:  {stats['scanned']:,} ({stats['scanned'] / total * 100:.1f}%)"
    )
    console.print(
        f"├─ Scored:   {stats['scored']:,} ({stats['scored'] / total * 100:.1f}%)"
    )
    console.print(
        f"└─ Skipped:   {stats['skipped']:,} ({stats['skipped'] / total * 100:.1f}%)"
    )

    # Summary
    console.print(f"\nFrontier: {stats['pending']} paths awaiting scan")
    coverage = stats["scored"] / total * 100 if total > 0 else 0
    console.print(f"Coverage: {coverage:.1f}% scored")

    # High value paths
    high_value = get_high_value_paths(facility, min_score=0.7, limit=10)
    if high_value:
        console.print(f"High-value paths (score > 0.7): {len(high_value)}")
        for p in high_value[:5]:
            console.print(f"  [{p['score']:.2f}] {p['path']}")
        if len(high_value) > 5:
            console.print(f"  ... and {len(high_value) - 5} more")
