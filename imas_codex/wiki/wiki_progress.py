"""
Wiki discovery progress display.

Extends BaseProgressDisplay from progress_common with wiki-specific
visualization of scan, prefetch, score, and ingest workers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Group
from rich.table import Table
from rich.text import Text

from imas_codex.discovery.progress_common import (
    BaseProgressDisplay,
    BaseProgressState,
    StreamQueue,
    format_time,
    make_bar,
    make_gradient_bar,
    make_resource_gauge,
)

if TYPE_CHECKING:
    from rich.console import Console, RenderableType


# =============================================================================
# Wiki Activity Items
# =============================================================================


@dataclass
class WikiScanItem:
    """A recently scanned page for activity display."""

    title: str
    out_links: int
    depth: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class WikiScoreItem:
    """A recently scored page for activity display."""

    title: str
    score: float
    is_physics: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class WikiIngestItem:
    """A recently ingested page for activity display."""

    title: str
    chunk_count: int
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# Wiki Progress State
# =============================================================================


@dataclass
class WikiProgressState(BaseProgressState):
    """State for wiki discovery progress tracking."""

    # Counts
    pages_scanned: int = 0
    pages_prefetched: int = 0
    pages_scored: int = 0
    pages_ingested: int = 0
    pages_skipped: int = 0
    artifacts_found: int = 0

    # Queue sizes (work remaining)
    pending_scan: int = 0
    pending_prefetch: int = 0
    pending_score: int = 0
    pending_ingest: int = 0

    # Worker status
    scan_worker_status: str = "idle"
    prefetch_worker_status: str = "idle"
    score_worker_status: str = "idle"
    ingest_worker_status: str = "idle"

    # LLM cost tracking
    score_cost: float = 0.0
    ingest_cost: float = 0.0
    cost_limit: float = 10.0

    # Page limit
    page_limit: int | None = None

    # High score threshold
    min_ingest_score: float = 0.5

    @property
    def total_cost(self) -> float:
        return self.score_cost + self.ingest_cost

    @property
    def cost_fraction(self) -> float:
        if self.cost_limit <= 0:
            return 0.0
        return min(1.0, self.total_cost / self.cost_limit)

    @property
    def scan_rate(self) -> float:
        if self.elapsed <= 0:
            return 0.0
        return self.pages_scanned / self.elapsed

    @property
    def score_rate(self) -> float:
        if self.elapsed <= 0:
            return 0.0
        return self.pages_scored / self.elapsed


# =============================================================================
# Wiki Progress Display
# =============================================================================


class WikiProgressDisplay(BaseProgressDisplay):
    """Rich live display for wiki discovery progress."""

    def __init__(
        self,
        console: Console | None = None,
        cost_limit: float = 10.0,
        page_limit: int | None = None,
    ) -> None:
        super().__init__(console)
        self.state = WikiProgressState(cost_limit=cost_limit, page_limit=page_limit)

        # Activity streams for smooth display
        self.scan_stream: StreamQueue[WikiScanItem] = StreamQueue(
            max_size=100, rate=2.0
        )
        self.score_stream: StreamQueue[WikiScoreItem] = StreamQueue(
            max_size=100, rate=1.5
        )
        self.ingest_stream: StreamQueue[WikiIngestItem] = StreamQueue(
            max_size=50, rate=0.5
        )

        # Recent activity for display
        self._recent_scans: list[WikiScanItem] = []
        self._recent_scores: list[WikiScoreItem] = []
        self._recent_ingests: list[WikiIngestItem] = []
        self._max_recent = 5

    def _build_display(self) -> RenderableType:
        """Build the complete progress display."""
        sections = []

        # Header with elapsed time and cost
        sections.append(self._build_header())

        # Progress bars section
        sections.append(self._build_progress_bars())

        # Worker status section
        sections.append(self._build_worker_status())

        # Queue depths
        sections.append(self._build_queue_depths())

        # Recent activity
        if self._recent_scans or self._recent_scores or self._recent_ingests:
            sections.append(self._build_activity())

        return Group(*sections)

    def _build_header(self) -> RenderableType:
        """Build header with time and cost."""
        elapsed_str = format_time(self.state.elapsed)

        # Cost gauge
        cost_pct = self.state.cost_fraction * 100
        cost_str = f"${self.state.total_cost:.2f}/${self.state.cost_limit:.2f}"

        header = Text()
        header.append("Wiki Discovery ", style="bold cyan")
        header.append(f"({elapsed_str})", style="dim")
        header.append("  ")
        header.append("Cost: ", style="dim")
        header.append(cost_str, style="yellow" if cost_pct > 80 else "green")

        if self.state.page_limit:
            header.append(
                f"  Limit: {self.state.pages_scored}/{self.state.page_limit}",
                style="dim",
            )

        return header

    def _build_progress_bars(self) -> RenderableType:
        """Build pipeline progress bars."""
        table = Table.grid(padding=(0, 2))
        table.add_column(width=12)  # Label
        table.add_column(width=30)  # Bar
        table.add_column(width=15)  # Stats

        # Scan progress
        scan_bar = make_bar(
            self.state.pages_scanned,
            self.state.pages_scanned + self.state.pending_scan,
            width=28,
            color="green",
        )
        table.add_row(
            Text("Scan", style="bold"),
            scan_bar,
            Text(
                f"{self.state.pages_scanned:,} pages ({self.state.scan_rate:.1f}/s)",
                style="dim",
            ),
        )

        # Score progress with cost gradient
        score_bar = make_gradient_bar(
            self.state.cost_fraction,
            width=28,
            colors=["green", "yellow", "red"],
        )
        table.add_row(
            Text("Score", style="bold"),
            score_bar,
            Text(
                f"{self.state.pages_scored:,} scored ({self.state.score_rate:.1f}/s)",
                style="dim",
            ),
        )

        # Ingest progress
        high_value = self.state.pages_ingested + self.state.pending_ingest
        if high_value > 0:
            ingest_bar = make_bar(
                self.state.pages_ingested,
                high_value,
                width=28,
                color="blue",
            )
        else:
            ingest_bar = Text("─" * 28, style="dim")
        table.add_row(
            Text("Ingest", style="bold"),
            ingest_bar,
            Text(f"{self.state.pages_ingested:,} chunks", style="dim"),
        )

        return table

    def _build_worker_status(self) -> RenderableType:
        """Build worker status indicators."""
        table = Table.grid(padding=(0, 3))

        statuses = [
            ("Scan", self.state.scan_worker_status),
            ("Prefetch", self.state.prefetch_worker_status),
            ("Score", self.state.score_worker_status),
            ("Ingest", self.state.ingest_worker_status),
        ]

        for _name, _status in statuses:
            table.add_column()

        row = []
        for name, status in statuses:
            if status == "idle":
                style = "dim"
                icon = "○"
            elif "scanning" in status or "scoring" in status or "ingesting" in status:
                style = "green bold"
                icon = "●"
            else:
                style = "yellow"
                icon = "◐"

            cell = Text()
            cell.append(f"{icon} {name}: ", style=style)
            cell.append(status[:20], style="dim")
            row.append(cell)

        table.add_row(*row)
        return table

    def _build_queue_depths(self) -> RenderableType:
        """Build queue depth gauges."""
        table = Table.grid(padding=(0, 2))

        # Show resource gauges for each queue
        gauges = [
            ("Pending", self.state.pending_scan, 1000),
            ("Scanned", self.state.pending_prefetch, 500),
            ("Prefetched", self.state.pending_score, 500),
            ("Scored", self.state.pending_ingest, 100),
        ]

        for _name, _current, _max_val in gauges:
            table.add_column()

        row = []
        for name, current, max_val in gauges:
            gauge = make_resource_gauge(current, max_val, width=6)
            cell = Text()
            cell.append(f"{name}: ", style="dim")
            cell.append_text(gauge)
            cell.append(f" ({current:,})", style="dim")
            row.append(cell)

        table.add_row(*row)
        return table

    def _build_activity(self) -> RenderableType:
        """Build recent activity list."""
        lines = []

        # Recent scans
        for item in self._recent_scans[-3:]:
            title = item.title[:40] + "..." if len(item.title) > 40 else item.title
            text = Text()
            text.append("○ ", style="green")
            text.append(title, style="bold")
            text.append(f" (+{item.out_links} links, depth {item.depth})", style="dim")
            lines.append(text)

        # Recent scores
        for item in self._recent_scores[-3:]:
            score_color = (
                "green"
                if item.score >= 0.7
                else "yellow"
                if item.score >= 0.5
                else "red"
            )
            title = item.title[:40] + "..." if len(item.title) > 40 else item.title
            text = Text()
            text.append("● ", style=score_color)
            text.append(title, style="bold")
            text.append(f" ({item.score:.0%})", style=score_color)
            if item.is_physics:
                text.append(" [physics]", style="cyan")
            lines.append(text)

        # Recent ingests
        for item in self._recent_ingests[-2:]:
            title = item.title[:40] + "..." if len(item.title) > 40 else item.title
            text = Text()
            text.append("★ ", style="blue")
            text.append(title, style="bold blue")
            text.append(f" ({item.chunk_count} chunks)", style="dim")
            lines.append(text)

        return Group(*lines) if lines else Text("")

    def tick(self) -> None:
        """Drain activity streams and update display."""
        # Drain streams into recent activity lists
        while (item := self.scan_stream.get_nowait()) is not None:
            self._recent_scans.append(item)
            if len(self._recent_scans) > self._max_recent:
                self._recent_scans.pop(0)

        while (item := self.score_stream.get_nowait()) is not None:
            self._recent_scores.append(item)
            if len(self._recent_scores) > self._max_recent:
                self._recent_scores.pop(0)

        while (item := self.ingest_stream.get_nowait()) is not None:
            self._recent_ingests.append(item)
            if len(self._recent_ingests) > self._max_recent:
                self._recent_ingests.pop(0)

    # =========================================================================
    # Update Methods (called by workers)
    # =========================================================================

    def update_scan_progress(
        self,
        status: str,
        pages_scanned: int,
        pending_scan: int,
        artifacts_found: int = 0,
        recent: list[WikiScanItem] | None = None,
    ) -> None:
        """Update scan worker progress."""
        self.state.scan_worker_status = status
        self.state.pages_scanned = pages_scanned
        self.state.pending_scan = pending_scan
        self.state.artifacts_found = artifacts_found

        if recent:
            for item in recent:
                self.scan_stream.put(item)

    def update_prefetch_progress(
        self,
        status: str,
        pages_prefetched: int,
        pending_prefetch: int,
    ) -> None:
        """Update prefetch worker progress."""
        self.state.prefetch_worker_status = status
        self.state.pages_prefetched = pages_prefetched
        self.state.pending_prefetch = pending_prefetch

    def update_score_progress(
        self,
        status: str,
        pages_scored: int,
        pending_score: int,
        pending_ingest: int,
        cost: float,
        recent: list[WikiScoreItem] | None = None,
    ) -> None:
        """Update score worker progress."""
        self.state.score_worker_status = status
        self.state.pages_scored = pages_scored
        self.state.pending_score = pending_score
        self.state.pending_ingest = pending_ingest
        self.state.score_cost = cost

        if recent:
            for item in recent:
                self.score_stream.put(item)

    def update_ingest_progress(
        self,
        status: str,
        pages_ingested: int,
        cost: float,
        recent: list[WikiIngestItem] | None = None,
    ) -> None:
        """Update ingest worker progress."""
        self.state.ingest_worker_status = status
        self.state.pages_ingested = pages_ingested
        self.state.ingest_cost = cost

        if recent:
            for item in recent:
                self.ingest_stream.put(item)

    def update_from_graph_stats(self, stats: dict[str, int]) -> None:
        """Update state from graph statistics query."""
        self.state.pages_scanned = (
            stats.get("scanned", 0)
            + stats.get("prefetched", 0)
            + stats.get("scored", 0)
            + stats.get("ingested", 0)
        )
        self.state.pages_scored = stats.get("scored", 0) + stats.get("ingested", 0)
        self.state.pages_ingested = stats.get("ingested", 0)
        self.state.pages_skipped = stats.get("skipped", 0)
        self.state.artifacts_found = stats.get("total_artifacts", 0)

        self.state.pending_scan = stats.get("pending", 0)
        self.state.pending_prefetch = stats.get("scanned", 0)
        self.state.pending_score = stats.get("prefetched", 0)
        self.state.pending_ingest = stats.get("scored", 0)
