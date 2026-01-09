"""Rich progress monitoring for wiki ingestion.

Provides a multi-stage progress display for CLI and MCP tool status
reporting. Tracks pages scraped, chunks created, and entities linked.

Example:
    monitor = WikiProgressMonitor()
    monitor.start(total_pages=50)
    monitor.update_scrape("Thomson", chunks=12, links=45)
    monitor.finish()
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WikiIngestionStats:
    """Statistics for a wiki ingestion run."""

    pages_total: int = 0
    pages_scraped: int = 0
    pages_failed: int = 0
    chunks_created: int = 0
    tree_nodes_linked: int = 0
    imas_paths_linked: int = 0
    conventions_found: int = 0
    units_found: int = 0
    started_at: float = field(default_factory=time.time)
    current_page: str = ""

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        return time.time() - self.started_at

    @property
    def pages_per_second(self) -> float:
        """Processing rate in pages per second."""
        if self.elapsed_seconds > 0 and self.pages_scraped > 0:
            return self.pages_scraped / self.elapsed_seconds
        return 0.0

    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining in seconds."""
        remaining = self.pages_total - self.pages_scraped
        if self.pages_per_second > 0:
            return remaining / self.pages_per_second
        return 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for MCP tool output."""
        return {
            "pages_total": self.pages_total,
            "pages_scraped": self.pages_scraped,
            "pages_failed": self.pages_failed,
            "chunks_created": self.chunks_created,
            "tree_nodes_linked": self.tree_nodes_linked,
            "imas_paths_linked": self.imas_paths_linked,
            "conventions_found": self.conventions_found,
            "units_found": self.units_found,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "pages_per_second": round(self.pages_per_second, 2),
            "eta_seconds": round(self.eta_seconds, 1),
            "current_page": self.current_page,
            "completion_pct": round(
                100 * self.pages_scraped / max(1, self.pages_total), 1
            ),
        }


class WikiProgressMonitor:
    """Progress monitor for wiki ingestion with Rich display support.

    Provides two modes:
    1. Rich console mode (for CLI): Live updating progress bars and panels
    2. Logging mode (for MCP/background): Standard logger output

    The monitor tracks multi-stage progress:
    - Scrape: Fetching HTML from wiki
    - Chunk: Splitting content and generating embeddings
    - Link: Creating graph relationships to TreeNodes/IMASPaths
    """

    def __init__(self, use_rich: bool = True, logger: logging.Logger | None = None):
        """Initialize progress monitor.

        Args:
            use_rich: If True, use Rich console display (requires terminal)
            logger: Optional logger for non-Rich mode
        """
        self.use_rich = use_rich
        self.logger = logger or logging.getLogger(__name__)
        self.stats = WikiIngestionStats()
        self._live = None
        self._progress = None
        self._task_scrape = None
        self._task_chunk = None
        self._task_link = None
        self._content_preview = ""
        self._mdsplus_preview: list[str] = []

    def start(self, total_pages: int) -> None:
        """Start progress tracking.

        Args:
            total_pages: Total number of pages to process
        """
        self.stats = WikiIngestionStats(pages_total=total_pages)

        if self.use_rich:
            try:
                from rich.console import Console
                from rich.live import Live
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                    TimeElapsedColumn,
                    TimeRemainingColumn,
                )

                self._console = Console()
                self._progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=30),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TextColumn("→"),
                    TimeRemainingColumn(),
                )
                self._task_scrape = self._progress.add_task(
                    "Scraping", total=total_pages
                )
                self._live = Live(
                    self._create_display(),
                    console=self._console,
                    refresh_per_second=4,
                )
                self._live.start()
            except ImportError:
                self.use_rich = False
                self.logger.info("Starting wiki ingestion of %d pages", total_pages)
        else:
            self.logger.info("Starting wiki ingestion of %d pages", total_pages)

    def update_scrape(
        self,
        page_name: str,
        chunks: int = 0,
        tree_nodes: int = 0,
        imas_paths: int = 0,
        conventions: int = 0,
        units: int = 0,
        failed: bool = False,
        content_preview: str = "",
        mdsplus_paths: list[str] | None = None,
    ) -> None:
        """Update progress after scraping a page.

        Args:
            page_name: Name of the page just processed
            chunks: Number of chunks created from this page
            tree_nodes: Number of TreeNode links created
            imas_paths: Number of IMASPath links created
            conventions: Number of conventions found
            units: Number of units found
            failed: Whether the page failed to process
            content_preview: Optional preview of extracted content (first ~100 chars)
            mdsplus_paths: Optional list of MDSplus paths found on page
        """
        self.stats.current_page = page_name
        self._content_preview = content_preview[:200] if content_preview else ""
        self._mdsplus_preview = (mdsplus_paths or [])[:5]

        if failed:
            self.stats.pages_failed += 1
        else:
            self.stats.pages_scraped += 1
            self.stats.chunks_created += chunks
            self.stats.tree_nodes_linked += tree_nodes
            self.stats.imas_paths_linked += imas_paths
            self.stats.conventions_found += conventions
            self.stats.units_found += units

        if self.use_rich and self._progress and self._task_scrape is not None:
            self._progress.update(
                self._task_scrape,
                completed=self.stats.pages_scraped + self.stats.pages_failed,
                description=f"Scraping: {page_name[:30]}",
            )
            if self._live:
                self._live.update(self._create_display())
        else:
            if failed:
                self.logger.warning("Failed: %s", page_name)
            else:
                self.logger.info(
                    "[%d/%d] %s: %d chunks, %d links",
                    self.stats.pages_scraped,
                    self.stats.pages_total,
                    page_name,
                    chunks,
                    tree_nodes + imas_paths,
                )

    def finish(self) -> WikiIngestionStats:
        """Finish progress tracking and return final statistics.

        Returns:
            Final WikiIngestionStats
        """
        if self.use_rich and self._live:
            self._live.stop()
            # Print final summary
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table

            console = Console()
            summary = Table(
                title="Wiki Ingestion Complete", show_header=False, box=None
            )
            summary.add_column(style="dim")
            summary.add_column(justify="right")
            summary.add_row(
                "Pages scraped:", f"[green]{self.stats.pages_scraped}[/green]"
            )
            summary.add_row("Pages failed:", f"[red]{self.stats.pages_failed}[/red]")
            summary.add_row("Chunks created:", str(self.stats.chunks_created))
            summary.add_row("TreeNodes linked:", str(self.stats.tree_nodes_linked))
            summary.add_row("IMAS paths linked:", str(self.stats.imas_paths_linked))
            summary.add_row("Conventions found:", str(self.stats.conventions_found))
            summary.add_row("Time:", f"{self.stats.elapsed_seconds:.1f}s")
            console.print(Panel(summary, border_style="green"))
        else:
            self.logger.info(
                "Wiki ingestion complete: %d pages, %d chunks, %d links",
                self.stats.pages_scraped,
                self.stats.chunks_created,
                self.stats.tree_nodes_linked + self.stats.imas_paths_linked,
            )

        return self.stats

    def _create_display(self):
        """Create Rich display for live updates."""
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        # Statistics panel
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column(style="dim")
        stats_table.add_column(justify="right")
        stats_table.add_row("Chunks:", str(self.stats.chunks_created))
        stats_table.add_row(
            "TreeNodes:", f"[cyan]{self.stats.tree_nodes_linked}[/cyan]"
        )
        stats_table.add_row(
            "IMAS paths:", f"[cyan]{self.stats.imas_paths_linked}[/cyan]"
        )
        stats_table.add_row("Conventions:", str(self.stats.conventions_found))
        stats_table.add_row("Rate:", f"{self.stats.pages_per_second:.2f} pages/sec")

        stats_panel = Panel(
            stats_table,
            title="[bold]Statistics[/bold]",
            border_style="blue",
            padding=(0, 1),
        )

        # Content preview panel (shows extracted text sample)
        content_parts = [self._progress, stats_panel]

        if self._content_preview or self._mdsplus_preview:
            preview_table = Table.grid(padding=(0, 1))
            preview_table.add_column(style="dim", width=12)
            preview_table.add_column()

            if self._content_preview:
                # Truncate and clean preview text
                preview = self._content_preview.replace("\n", " ").strip()
                if len(preview) > 150:
                    preview = preview[:147] + "..."
                preview_table.add_row(
                    "Content:", Text(preview, style="italic", overflow="ellipsis")
                )

            if self._mdsplus_preview:
                paths_text = ", ".join(self._mdsplus_preview)
                if len(self._mdsplus_preview) >= 5:
                    paths_text += "..."
                preview_table.add_row("MDSplus:", Text(paths_text, style="green"))

            preview_panel = Panel(
                preview_table,
                title=f"[bold]{self.stats.current_page}[/bold]",
                border_style="dim",
                padding=(0, 1),
            )
            content_parts.append(preview_panel)

        return Group(*content_parts)

    def get_status(self) -> dict:
        """Get current status as a dictionary (for MCP tools).

        Returns:
            Status dictionary suitable for JSON serialization
        """
        return self.stats.to_dict()


# Global monitor instance for MCP tool access
_current_monitor: WikiProgressMonitor | None = None


def get_current_monitor() -> WikiProgressMonitor | None:
    """Get the currently active progress monitor (if any).

    Used by MCP tools to report ingestion progress.
    """
    return _current_monitor


def set_current_monitor(monitor: WikiProgressMonitor | None) -> None:
    """Set the current progress monitor.

    Called by the pipeline when starting/stopping ingestion.
    """
    global _current_monitor
    _current_monitor = monitor


@dataclass
class CrawlStats:
    """Statistics for wiki crawl progress."""

    pages_crawled: int = 0
    pages_skipped: int = 0
    artifacts_found: int = 0
    links_discovered: int = 0
    frontier_size: int = 0
    max_depth: int = 0
    current_depth: int = 0
    current_page: str = ""
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    def elapsed_formatted(self) -> str:
        """Format elapsed time as human-readable string."""
        seconds = int(self.elapsed_seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, mins = divmod(minutes, 60)
        if hours < 24:
            return f"{hours}h {mins}m {secs}s"
        days, hrs = divmod(hours, 24)
        return f"{days}d {hrs}h {mins}m {secs}s"

    @property
    def pages_per_second(self) -> float:
        if self.elapsed_seconds > 0 and self.pages_crawled > 0:
            return self.pages_crawled / self.elapsed_seconds
        return 0.0

    @property
    def progress_pct(self) -> float:
        """Dynamic progress: crawled / (crawled + remaining frontier)."""
        total = self.pages_crawled + self.frontier_size
        if total > 0:
            return 100 * self.pages_crawled / total
        return 0.0


class CrawlProgressMonitor:
    """Progress monitor for wiki crawl with clean Rich display.

    Displays running totals integrated into the progress bar,
    avoiding verbose output corruption. Progress percentage is
    dynamic: crawled / (crawled + frontier).

    Example:
        with CrawlProgressMonitor() as monitor:
            for page, links in crawl_generator():
                monitor.update(page, links_found=len(links), depth=depth)
    """

    def __init__(self, facility: str = "wiki"):
        self.stats = CrawlStats()
        self.facility = facility
        self._live = None
        self._console = None

    def __enter__(self) -> "CrawlProgressMonitor":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.finish()

    def start(self) -> None:
        """Start the live display."""
        from rich.console import Console
        from rich.live import Live

        self._console = Console()
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
        )
        self._live.start()

    def update(
        self,
        page: str = "",
        links_found: int = 0,
        artifacts_found: int = 0,
        frontier_size: int = 0,
        depth: int = 0,
        skipped: bool = False,
    ) -> None:
        """Update crawl progress.

        Args:
            page: Current page being crawled
            links_found: New page links discovered from this page
            artifacts_found: New artifact links discovered from this page
            frontier_size: Current frontier queue size
            depth: Current link depth
            skipped: Whether this page was skipped
        """
        if page:
            self.stats.current_page = page
        if skipped:
            self.stats.pages_skipped += 1
        else:
            self.stats.pages_crawled += 1
        self.stats.links_discovered += links_found
        self.stats.artifacts_found += artifacts_found
        self.stats.frontier_size = frontier_size
        self.stats.current_depth = depth
        self.stats.max_depth = max(self.stats.max_depth, depth)

        if self._live:
            self._live.update(self._render())

    def _render(self):
        """Render the progress display."""
        from rich.console import Group
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
        from rich.table import Table

        # Dynamic total: crawled + frontier remaining
        total = self.stats.pages_crawled + self.stats.frontier_size
        pct = self.stats.progress_pct

        # Progress bar with percentage
        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]Crawling {self.facility} wiki"),
            BarColumn(bar_width=40),
            TextColumn(f"[green]{pct:5.1f}%[/]"),
            expand=False,
        )
        task = progress.add_task("", total=max(1, total))
        progress.update(task, completed=self.stats.pages_crawled)

        # Stats table - compact 2-column layout
        stats = Table.grid(padding=(0, 2))
        stats.add_column(style="dim", width=12)
        stats.add_column(justify="right", width=8)
        stats.add_column(style="dim", width=12)
        stats.add_column(justify="right", width=8)

        stats.add_row(
            "Crawled:",
            f"[green]{self.stats.pages_crawled}[/]",
            "Frontier:",
            f"[yellow]{self.stats.frontier_size}[/]",
        )
        stats.add_row(
            "Artifacts:",
            f"[blue]{self.stats.artifacts_found}[/]",
            "Elapsed:",
            f"[dim]{self.stats.elapsed_formatted()}[/]",
        )
        stats.add_row(
            "Depth:",
            f"[magenta]{self.stats.current_depth}[/]",
            "Max depth:",
            f"[magenta]{self.stats.max_depth}[/]",
        )
        if self.stats.pages_skipped > 0:
            stats.add_row(
                "Skipped:",
                f"[dim]{self.stats.pages_skipped}[/]",
                "Rate:",
                f"[dim]{self.stats.pages_per_second:.1f}/s[/]",
            )
        else:
            stats.add_row(
                "Rate:",
                f"[dim]{self.stats.pages_per_second:.1f}/s[/]",
                "",
                "",
            )

        # Current page
        current = ""
        if self.stats.current_page:
            page_display = self.stats.current_page
            if len(page_display) > 50:
                page_display = page_display[:47] + "..."
            current = f"[dim]→ {page_display}[/]"

        return Group(progress, stats, current) if current else Group(progress, stats)

    def finish(self) -> CrawlStats:
        """Stop the live display and return final stats."""
        if self._live:
            self._live.stop()
            self._live = None

        if self._console:
            elapsed = self.stats.elapsed_formatted()
            artifacts_msg = ""
            if self.stats.artifacts_found > 0:
                artifacts_msg = f", [blue]{self.stats.artifacts_found}[/] artifacts"
            self._console.print(
                f"\n[green]✓[/] Crawled [bold]{self.stats.pages_crawled}[/] pages"
                f"{artifacts_msg} in [bold]{elapsed}[/]"
            )

        return self.stats


@dataclass
class ScoreStats:
    """Statistics for wiki scoring progress."""

    total_pages: int = 0
    pages_scored: int = 0
    high_score_count: int = 0
    low_score_count: int = 0
    current_page: str = ""
    current_score: float = 0.0
    cost_spent_usd: float = 0.0
    cost_limit_usd: float = 20.0
    agent_iterations: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    def elapsed_formatted(self) -> str:
        """Format elapsed time as human-readable string."""
        seconds = int(self.elapsed_seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, mins = divmod(minutes, 60)
        if hours < 24:
            return f"{hours}h {mins}m {secs}s"
        days, hrs = divmod(hours, 24)
        return f"{days}d {hrs}h {mins}m {secs}s"

    @property
    def pages_per_second(self) -> float:
        if self.elapsed_seconds > 0 and self.pages_scored > 0:
            return self.pages_scored / self.elapsed_seconds
        return 0.0

    @property
    def progress_pct(self) -> float:
        """Progress percentage: scored / total."""
        if self.total_pages > 0:
            return 100 * self.pages_scored / self.total_pages
        return 0.0

    @property
    def cost_remaining(self) -> float:
        return max(0, self.cost_limit_usd - self.cost_spent_usd)

    @property
    def budget_exhausted(self) -> bool:
        return self.cost_spent_usd >= self.cost_limit_usd


class ScoreProgressMonitor:
    """Progress monitor for wiki scoring with Rich display.

    Displays scoring progress with cost tracking.

    Example:
        with ScoreProgressMonitor(total=3000, cost_limit=20.0) as monitor:
            for page, score in score_generator():
                monitor.update(page=page, score=score, cost=0.01)
    """

    def __init__(
        self, total: int = 0, cost_limit: float = 20.0, facility: str = "wiki"
    ):
        self.stats = ScoreStats(total_pages=total, cost_limit_usd=cost_limit)
        self.facility = facility
        self._live = None
        self._console = None

    def __enter__(self) -> "ScoreProgressMonitor":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.finish()

    def start(self) -> None:
        """Start the live display."""
        from rich.console import Console
        from rich.live import Live

        self._console = Console()
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
        )
        self._live.start()

    def update(
        self,
        page: str = "",
        score: float = 0.0,
        is_high: bool = False,
        is_low: bool = False,
        cost: float = 0.0,
        batch_size: int = 0,
    ) -> None:
        """Update scoring progress.

        Args:
            page: Current page being scored
            score: Score assigned to this page
            is_high: Whether this was a high score (>=0.7)
            is_low: Whether this was a low score (<0.3)
            cost: Cost incurred for this batch
            batch_size: Number of pages scored in this batch
        """
        if page:
            self.stats.current_page = page
            self.stats.current_score = score
        if batch_size > 0:
            self.stats.pages_scored += batch_size
        if is_high:
            self.stats.high_score_count += 1
        if is_low:
            self.stats.low_score_count += 1
        if cost > 0:
            self.stats.cost_spent_usd += cost

        if self._live:
            self._live.update(self._render())

    def add_batch(self, scored: int, high: int, low: int, cost: float = 0.0) -> None:
        """Add a batch of scored pages."""
        self.stats.pages_scored += scored
        self.stats.high_score_count += high
        self.stats.low_score_count += low
        self.stats.cost_spent_usd += cost
        self.stats.agent_iterations += 1

        if self._live:
            self._live.update(self._render())

    def set_current(self, page: str, score: float) -> None:
        """Set current page being displayed."""
        self.stats.current_page = page
        self.stats.current_score = score
        if self._live:
            self._live.update(self._render())

    def _render(self):
        """Render the progress display."""
        from rich.console import Group
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
        from rich.table import Table

        pct = self.stats.progress_pct

        # Progress bar with percentage
        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]Scoring {self.facility} wiki"),
            BarColumn(bar_width=40),
            TextColumn(f"[green]{pct:5.1f}%[/]"),
            expand=False,
        )
        task = progress.add_task("", total=max(1, self.stats.total_pages))
        progress.update(task, completed=self.stats.pages_scored)

        # Stats table - compact 2-column layout
        stats = Table.grid(padding=(0, 2))
        stats.add_column(style="dim", width=12)
        stats.add_column(justify="right", width=8)
        stats.add_column(style="dim", width=12)
        stats.add_column(justify="right", width=8)

        stats.add_row(
            "Scored:",
            f"[green]{self.stats.pages_scored}[/]/{self.stats.total_pages}",
            "Elapsed:",
            f"[dim]{self.stats.elapsed_formatted()}[/]",
        )
        stats.add_row(
            "High (≥0.7):",
            f"[cyan]{self.stats.high_score_count}[/]",
            "Low (<0.3):",
            f"[yellow]{self.stats.low_score_count}[/]",
        )
        stats.add_row(
            "Cost:",
            f"[magenta]${self.stats.cost_spent_usd:.2f}[/]",
            "Limit:",
            f"[dim]${self.stats.cost_limit_usd:.2f}[/]",
        )
        stats.add_row(
            "Rate:",
            f"[dim]{self.stats.pages_per_second:.1f}/s[/]",
            "Iterations:",
            f"[dim]{self.stats.agent_iterations}[/]",
        )

        # Current page
        current = ""
        if self.stats.current_page:
            page_display = self.stats.current_page
            if len(page_display) > 40:
                page_display = page_display[:37] + "..."
            score_color = (
                "green"
                if self.stats.current_score >= 0.7
                else ("yellow" if self.stats.current_score >= 0.3 else "red")
            )
            current = f"[dim]→ {page_display}[/] [{score_color}]{self.stats.current_score:.2f}[/]"

        return Group(progress, stats, current) if current else Group(progress, stats)

    def finish(self) -> ScoreStats:
        """Stop the live display and return final stats."""
        if self._live:
            self._live.stop()
            self._live = None

        if self._console:
            elapsed = self.stats.elapsed_formatted()
            self._console.print(
                f"\n[green]✓[/] Scored [bold]{self.stats.pages_scored}[/] pages "
                f"in [bold]{elapsed}[/] (${self.stats.cost_spent_usd:.2f})"
            )
            self._console.print(
                f"  High score (≥0.7): [cyan]{self.stats.high_score_count}[/]"
            )
            self._console.print(
                f"  Low score (<0.3): [yellow]{self.stats.low_score_count}[/]"
            )

        return self.stats
