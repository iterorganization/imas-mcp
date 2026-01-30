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

    # Counts (aligned with PathStatus enum)
    total_paths: int = 0
    discovered: int = 0  # Awaiting scan
    listed: int = 0  # Awaiting score
    scored: int = 0
    skipped: int = 0
    excluded: int = 0
    max_depth: int = 0

    # Current operation
    current_phase: str = "idle"  # scan, score, idle
    current_path: str = ""
    facility: str = ""  # Facility being discovered

    # Budget tracking (score phase)
    accumulated_cost: float = 0.0
    budget_limit: float | None = None
    model: str = ""

    # Cycle tracking (discover command)
    current_cycle: int = 0
    max_cycles: int = 0

    # Rate tracking
    scan_count: int = 0
    score_count: int = 0
    scan_start_time: float | None = None
    score_start_time: float | None = None

    @property
    def frontier_size(self) -> int:
        """Paths awaiting work (scan or score)."""
        return self.discovered + self.listed

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

    @property
    def scan_rate(self) -> float | None:
        """Scans per second, or None if not tracking."""
        import time

        if self.scan_start_time is None or self.scan_count == 0:
            return None
        elapsed = time.time() - self.scan_start_time
        if elapsed <= 0:
            return None
        return self.scan_count / elapsed

    @property
    def score_rate(self) -> float | None:
        """Scores per second, or None if not tracking."""
        import time

        if self.score_start_time is None or self.score_count == 0:
            return None
        elapsed = time.time() - self.score_start_time
        if elapsed <= 0:
            return None
        return self.score_count / elapsed


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
            "Listed:",
            f"{self.stats.listed:,}",
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
                "Limit:",
                f"${self.stats.budget_limit:.2f} ({budget_pct:.0f}%)",
            )

        # Row 5: Model (if set)
        if self.stats.model:
            # Display abbreviated model name
            model_display = self.stats.model
            if model_display.startswith("anthropic/"):
                model_display = model_display[len("anthropic/") :]
            table.add_row(
                "Model:",
                model_display,
                "",
                "",
            )

        # Row 6: Cycle (if in discover mode)
        if self.stats.max_cycles > 0:
            table.add_row(
                "Cycle:",
                f"{self.stats.current_cycle}/{self.stats.max_cycles}",
                "Phase:",
                f"[yellow]{self.stats.current_phase}[/yellow]",
            )

        # Row 7: Rates (if tracking)
        scan_rate = self.stats.scan_rate
        score_rate = self.stats.score_rate
        if scan_rate is not None or score_rate is not None:
            scan_str = f"{scan_rate:.1f}/s" if scan_rate else "-"
            score_str = f"{score_rate:.1f}/s" if score_rate else "-"
            table.add_row(
                "Scan rate:",
                scan_str,
                "Score rate:",
                score_str,
            )

        # Build title with facility name
        title = "Discovery Progress"
        if self.stats.facility:
            title = f"Discovering {self.stats.facility} filesystem"

        return Panel(table, title=title, border_style="blue")

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
        self.stats.discovered = stats["discovered"]
        self.stats.listed = stats["listed"]
        self.stats.scored = stats["scored"]
        self.stats.skipped = stats["skipped"]
        self.stats.excluded = stats["excluded"]
        self.stats.max_depth = stats["max_depth"]

        # Refresh display
        if self._live:
            self._live.update(self._build_display())


def print_discovery_status(facility: str, console: Console | None = None) -> None:
    """Print a formatted discovery status report.

    Args:
        facility: Facility ID
        console: Optional Rich console
    """
    from imas_codex.discovery.frontier import (
        get_discovery_stats,
        get_high_value_paths,
        get_purpose_distribution,
    )

    console = console or Console()
    stats = get_discovery_stats(facility)

    # Header
    console.print(f"\n[bold]Facility: {facility}[/bold]")
    console.print(f"Total paths: {stats['total']:,}")

    # Status breakdown
    total = stats["total"] or 1  # Avoid division by zero
    discovered = stats.get("discovered", 0)
    listed = stats.get("listed", 0)
    scored = stats.get("scored", 0)
    skipped = stats.get("skipped", 0)
    excluded = stats.get("excluded", 0)
    max_depth = stats.get("max_depth", 0)

    console.print(f"├─ Discovered: {discovered:,} ({discovered / total * 100:.1f}%)")
    console.print(f"├─ Listed:     {listed:,} ({listed / total * 100:.1f}%)")
    console.print(f"├─ Scored:     {scored:,} ({scored / total * 100:.1f}%)")
    console.print(f"├─ Skipped:    {skipped:,} ({skipped / total * 100:.1f}%)")
    console.print(f"└─ Excluded:   {excluded:,} ({excluded / total * 100:.1f}%)")

    # Purpose distribution
    purpose_dist = get_purpose_distribution(facility)
    if purpose_dist:
        console.print("\n[bold]By Purpose:[/bold]")
        # Group into categories for cleaner display
        modeling = sum(
            purpose_dist.get(p, 0) for p in ["modeling_code", "modeling_data"]
        )
        analysis = sum(
            purpose_dist.get(p, 0)
            for p in ["analysis_code", "operations_code", "experimental_data"]
        )
        infrastructure = sum(
            purpose_dist.get(p, 0) for p in ["data_access", "workflow", "visualization"]
        )
        support = sum(
            purpose_dist.get(p, 0)
            for p in ["documentation", "configuration", "test_suite"]
        )
        structural = sum(
            purpose_dist.get(p, 0)
            for p in ["container", "archive", "build_artifact", "system"]
        )

        console.print(f"├─ [cyan]Modeling[/cyan]:       {modeling:,} (code + data)")
        console.print(
            f"├─ [green]Analysis[/green]:       {analysis:,} (code + ops + data)"
        )
        console.print(f"├─ [yellow]Infrastructure[/yellow]: {infrastructure:,}")
        console.print(f"├─ [blue]Support[/blue]:        {support:,}")
        console.print(f"└─ [dim]Structural[/dim]:     {structural:,}")

        # Detail breakdown if verbose
        console.print("\n  [dim]Detail:[/dim]")
        for purpose, count in purpose_dist.items():
            pct = count / sum(purpose_dist.values()) * 100 if purpose_dist else 0
            console.print(f"    {purpose}: {count} ({pct:.1f}%)")

    # Summary
    frontier = discovered + listed
    console.print(f"\nFrontier: {frontier} paths awaiting work")
    console.print(f"Max depth: {max_depth}")
    coverage = scored / total * 100 if total > 0 else 0
    console.print(f"Coverage: {coverage:.1f}% scored")

    # High value paths
    high_value = get_high_value_paths(facility, min_score=0.7, limit=10)
    if high_value:
        console.print(f"High-value paths (score > 0.7): {len(high_value)}")
        for p in high_value[:5]:
            console.print(f"  [{p['score']:.2f}] {p['path']}")
        if len(high_value) > 5:
            console.print(f"  ... and {len(high_value) - 5} more")
