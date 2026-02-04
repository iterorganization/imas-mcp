"""
Progress display for parallel wiki discovery.

Design principles (matching paths parallel_progress.py):
- Minimal visual clutter (no emojis, no stopwatch icons)
- Clear hierarchy: Target → Progress → Activity → Resources
- Gradient progress bars with percentage
- Resource gauges for time and cost budgets
- Compact current activity with relevant details only

Display layout: SCORE → INGEST (matching actual workers)
- SCORE: Content-aware LLM scoring (fetches content, scores with LLM)
- INGEST: Chunk and embed high-value pages (score >= 0.5)

Progress is tracked against total pages in graph, not just this session.
ETA/ETC metrics calculated like paths discovery.

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
    clean_text,
    clip_text,
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
class ScoreItem:
    """Current score activity."""

    title: str
    score: float | None = None
    physics_domain: str | None = None  # Physics domain if detected
    description: str = ""  # LLM description of page value
    is_physics: bool = False
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class IngestItem:
    """Current ingest activity."""

    title: str
    score: float | None = None  # Redisplay score on ingest
    description: str = ""  # Redisplay LLM description
    physics_domain: str | None = None  # Physics domain if detected
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

    # Counts from graph (total pages for progress denominator)
    total_pages: int = 0  # All wiki pages in graph for this facility
    pages_scanned: int = 0  # Status = scanned (awaiting score)
    pages_scored: int = 0  # Status = scored (awaiting ingest or skipped)
    pages_ingested: int = 0  # Status = ingested (final state)
    pages_skipped: int = 0  # Skipped (low score or skip_reason)

    # Pending work counts (for queue display)
    pending_score: int = 0  # scanned pages awaiting scoring
    pending_ingest: int = 0  # scored pages awaiting ingestion

    # This run stats
    run_scored: int = 0
    run_ingested: int = 0
    _run_score_cost: float = 0.0
    _run_ingest_cost: float = 0.0
    score_rate: float | None = None
    ingest_rate: float | None = None

    # Accumulated facility cost (from graph)
    accumulated_cost: float = 0.0

    # Current items (and their processing state)
    current_score: ScoreItem | None = None
    current_ingest: IngestItem | None = None
    score_processing: bool = False
    ingest_processing: bool = False

    # Streaming queues - adaptive rate based on worker speed
    score_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.5, min_display_time=0.4
        )
    )
    ingest_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.5, min_display_time=0.4
        )
    )

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
    def limit_reason(self) -> str | None:
        """Return which limit was reached, or None if no limit reached."""
        if self.cost_limit_reached:
            return "cost"
        if self.page_limit_reached:
            return "page"
        return None

    @property
    def cost_per_page(self) -> float | None:
        """Average cost per scored page."""
        if self.run_scored > 0:
            return self.run_cost / self.run_scored
        return None

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time to termination based on limits.

        Priority order:
        1. Cost limit: if budget set, estimate time to exhaust budget
        2. Page limit: if --max-pages set, estimate time to process that many
        3. Full work: estimate time to complete all remaining scoring
        """
        # Try cost-based ETA first (if we have cost data)
        if self.run_cost > 0 and self.cost_limit > 0:
            cost_rate = self.run_cost / self.elapsed if self.elapsed > 0 else 0
            if cost_rate > 0:
                remaining_budget = self.cost_limit - self.run_cost
                return max(0, remaining_budget / cost_rate)

        # Try page-limit-based ETA (if --max-pages set)
        if self.page_limit is not None and self.page_limit > 0:
            if self.run_scored > 0 and self.elapsed > 0:
                rate = self.run_scored / self.elapsed
                remaining = self.page_limit - self.run_scored
                return max(0, remaining / rate) if rate > 0 else None

        # Fall back to work-based ETA (all pending pages)
        if not self.score_rate or self.score_rate <= 0:
            return None
        remaining = self.pending_score
        return remaining / self.score_rate if remaining > 0 else 0


# =============================================================================
# Main Display Class
# =============================================================================


class WikiProgressDisplay:
    """Clean progress display for parallel wiki discovery.

    Layout (100 chars wide - matching paths summary panel):
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                               TCV Wiki Discovery                                                 │
    │                           Focus: diagnostics                                                     │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  SCORE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━────────────────────────────    388  5%   0.6/s              │
    │  INGEST ━━━━━━━━━━━━━━━━━━━━━━━━──────────────────────────────     136  35%  0.3/s              │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  SCORE 0.70 Microwave_lab/software                                                               │
    │    [physics domain] Power supply calibration for gyrotron system                                 │
    │  INGEST Service_Mécanique/Information_TCV/Connexion_dans_l                                       │
    │    0.65 [equilibrium] Vacuum vessel port documentation with coordinates                          │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  TIME   │━━━━━━━━━━━━━━━━━━━━━━│  7m     ETA 2h 15m                                              │
    │  COST   │━━━━━━━━━━━━━━━━━━━━━━│  $0.04 / $0.01  cost limit reached                              │
    │  TOTAL  │━━━━━━━━━━━━━━━━━━━━━━│  $0.04  ETC $1.50                                               │
    │  STATS  scored=388  ingested=136  skipped=0  pending=[score:7386 ingest:252]                     │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

    Workflow:
    - SCORE: Content-aware LLM scoring (scanned → scored)
    - INGEST: Chunk and embed high-value pages (scored → ingested)
    """

    WIDTH = 100
    BAR_WIDTH = 48
    GAUGE_WIDTH = 24

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
        """Build the main progress bars for SCORE and INGEST.

        Progress is measured against total pages in graph, not just this session.
        This shows true coverage of the wiki.
        """
        section = Text()
        bar_width = self.BAR_WIDTH

        # SCORE row - scoring progress against total pages
        # Total to score = total_pages (all wiki pages for this facility)
        # Completed = pages_scored + pages_ingested (scored includes ingest-ready)
        score_total = self.state.total_pages or 1
        scored_pages = self.state.pages_scored + self.state.pages_ingested
        score_pct = scored_pages / score_total * 100 if score_total > 0 else 0

        if self.state.scan_only:
            section.append("  SCORE ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  SCORE ", style="bold blue")
            score_ratio = min(scored_pages / score_total, 1.0) if score_total > 0 else 0
            section.append(make_bar(score_ratio, bar_width), style="blue")
            section.append(f" {scored_pages:>6,}", style="bold")
            section.append(f" {score_pct:>3.0f}%", style="cyan")
            if self.state.score_rate and self.state.score_rate > 0:
                section.append(f" {self.state.score_rate:>5.1f}/s", style="dim")
        section.append("\n")

        # INGEST row - ingestion progress against scored pages
        # Total to ingest = pages that qualify for ingestion (score >= 0.5)
        # For simplicity, use pages_scored + pages_ingested as denominator
        ingest_total = self.state.pages_scored + self.state.pages_ingested
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

    def _clip_title(self, title: str, max_len: int = 70) -> str:
        """Clip title to max length, preferring end truncation."""
        if len(title) <= max_len:
            return title
        return title[: max_len - 3] + "..."

    def _build_activity_section(self) -> Text:
        """Build the current activity section showing SCORE and INGEST.

        Shows physics domain and LLM description for both workers.
        All content is clipped to fit within the panel width.
        """
        section = Text()

        score = self.state.current_score
        ingest = self.state.current_ingest

        # Maximum content width for the panel (account for padding and border)
        content_width = self.WIDTH - 6  # Panel padding + border

        # Helper to determine if worker should show "idle"
        def should_show_idle(processing: bool, queue: StreamQueue) -> bool:
            return not processing and queue.is_empty()

        # SCORE section - always 2 lines for consistent height
        if not self.state.scan_only:
            section.append("  SCORE ", style="bold blue")
            if score:
                # Line 1: Page title (clipped to fit remaining space)
                title_width = content_width - 8  # 8 = "  SCORE "
                section.append(
                    self._clip_title(score.title, title_width), style="white"
                )
                section.append("\n")

                # Line 2: Score, physics domain, description
                section.append("    ", style="dim")
                score_str = ""
                if score.score is not None:
                    if score.score >= 0.7:
                        style = "bold green"
                    elif score.score >= 0.4:
                        style = "yellow"
                    else:
                        style = "red"
                    score_str = f"{score.score:.2f}"
                    section.append(f"{score_str}  ", style=style)

                # Physics domain without brackets (matching paths CLI)
                domain_str = ""
                if score.physics_domain:
                    domain_str = score.physics_domain
                    section.append(f"{domain_str}  ", style="cyan")
                elif score.is_physics:
                    domain_str = "physics"
                    section.append(f"{domain_str}  ", style="cyan")

                if score.description:
                    desc = clean_text(score.description)
                    # Calculate remaining width for description
                    # 4 indent + score (5) + space (2) + domain + space (2)
                    used = (
                        4
                        + (len(score_str) + 2 if score_str else 0)
                        + (len(domain_str) + 2 if domain_str else 0)
                    )
                    desc_width = content_width - used
                    section.append(
                        clip_text(desc, max(10, desc_width)), style="italic dim"
                    )
                elif score.skipped:
                    reason = score.skip_reason[:40] if score.skip_reason else ""
                    section.append(f"skipped: {reason}", style="yellow dim")
            elif self.state.score_processing:
                section.append("processing batch...", style="cyan italic")
                section.append("\n    ", style="dim")
            elif should_show_idle(self.state.score_processing, self.state.score_queue):
                section.append("idle", style="dim italic")
                section.append("\n    ", style="dim")
            else:
                section.append("...", style="dim italic")
                section.append("\n    ", style="dim")
            section.append("\n")

        # INGEST section - 2 lines: title, then score + domain + description
        if not self.state.scan_only:
            section.append("  INGEST", style="bold magenta")
            if ingest:
                # Line 1: Page title (clipped to fit remaining space)
                title_width = content_width - 9  # 9 = "  INGEST "
                section.append(
                    " " + self._clip_title(ingest.title, title_width), style="white"
                )
                section.append("\n")

                # Line 2: score + physics domain + description + chunk count
                section.append("    ", style="dim")
                score_str = ""
                if ingest.score is not None:
                    if ingest.score >= 0.7:
                        style = "bold green"
                    elif ingest.score >= 0.4:
                        style = "yellow"
                    else:
                        style = "dim"
                    score_str = f"{ingest.score:.2f}"
                    section.append(f"{score_str}  ", style=style)

                # Physics domain without brackets (matching paths CLI)
                domain_str = ""
                if ingest.physics_domain:
                    domain_str = ingest.physics_domain
                    section.append(f"{domain_str}  ", style="cyan")

                if ingest.description:
                    desc = clean_text(ingest.description)
                    # Calculate remaining width for description
                    # 4 indent + score + domain
                    used = (
                        4
                        + (len(score_str) + 2 if score_str else 0)
                        + (len(domain_str) + 2 if domain_str else 0)
                    )
                    desc_width = content_width - used
                    section.append(
                        clip_text(desc, max(10, desc_width)), style="italic dim"
                    )
            elif self.state.ingest_processing:
                section.append(" processing batch...", style="cyan italic")
                section.append("\n    ", style="dim")
            elif should_show_idle(
                self.state.ingest_processing, self.state.ingest_queue
            ):
                section.append(" idle", style="dim italic")
                section.append("\n    ", style="dim")
            else:
                section.append(" ...", style="dim italic")
                section.append("\n    ", style="dim")

        return section

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges with ETA/ETC like paths CLI."""
        section = Text()

        # TIME row with ETA
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
                # Show which limit was reached
                limit_reason = self.state.limit_reason
                if limit_reason == "cost":
                    section.append("  cost limit reached", style="yellow dim")
                elif limit_reason == "page":
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

            # TOTAL row - cumulative cost with ETC (Estimated Total Cost)
            total_facility_cost = self.state.accumulated_cost + self.state.run_cost
            if total_facility_cost > 0 or self.state.pending_score > 0:
                section.append("  TOTAL ", style="bold white")

                # Calculate ETC based on cost per page
                cpp = self.state.cost_per_page
                etc = total_facility_cost
                if cpp and cpp > 0 and self.state.pending_score > 0:
                    etc = total_facility_cost + (self.state.pending_score * cpp)

                if etc > 0:
                    section.append_text(
                        make_resource_gauge(total_facility_cost, etc, self.GAUGE_WIDTH)
                    )
                else:
                    section.append("│", style="dim")
                    section.append("━" * self.GAUGE_WIDTH, style="white")
                    section.append("│", style="dim")

                section.append(f"  ${total_facility_cost:.2f}", style="bold")
                if etc > total_facility_cost:
                    section.append(f"  ETC ${etc:.2f}", style="dim")
                section.append("\n")

        # STATS row - graph state with pending work
        section.append("  STATS ", style="bold magenta")
        section.append(
            f"scored={self.state.pages_scored + self.state.pages_ingested}",
            style="blue",
        )
        section.append(f"  ingested={self.state.pages_ingested}", style="magenta")
        section.append(f"  skipped={self.state.pages_skipped}", style="yellow")

        # Pending work by worker type
        pending_parts = []
        if self.state.pending_score > 0:
            pending_parts.append(f"score:{self.state.pending_score}")
        if self.state.pending_ingest > 0:
            pending_parts.append(f"ingest:{self.state.pending_ingest}")
        if pending_parts:
            section.append(f"  pending=[{' '.join(pending_parts)}]", style="cyan dim")

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
        # Pop from score queue
        if item := self.state.score_queue.pop():
            self.state.current_score = ScoreItem(
                title=item.get("title", ""),
                score=item.get("score"),
                physics_domain=item.get("physics_domain"),
                description=item.get("description", ""),
                is_physics=item.get("is_physics", False),
                skipped=item.get("skipped", False),
                skip_reason=item.get("skip_reason", ""),
            )

        # Pop from ingest queue
        if item := self.state.ingest_queue.pop():
            self.state.current_ingest = IngestItem(
                title=item.get("title", ""),
                score=item.get("score"),
                description=item.get("description", ""),
                physics_domain=item.get("physics_domain"),
                chunk_count=item.get("chunk_count", 0),
            )

        self._refresh()

    def update_scan(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update scanner state (currently unused since bulk discovery runs before display)."""
        # Bulk discovery now runs as a setup step before the display starts,
        # so scan messages only come from link-crawling mode (non-bulk)
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
        if "waiting" in message.lower() or message == "idle":
            self.state.score_processing = False
        elif "scoring" in message.lower() or "fetching" in message.lower():
            self.state.score_processing = True
        else:
            self.state.score_processing = False

        # Queue results for streaming with adaptive rate
        if results:
            items = []
            for r in results:
                # Use title if already extracted, otherwise extract from id
                title = r.get("title") or r.get("id", "?").split(":")[-1]
                items.append(
                    {
                        "title": title[:60],
                        "score": r.get("score"),
                        "physics_domain": r.get("physics_domain"),
                        "description": r.get("description", ""),
                        "is_physics": r.get("is_physics", False),
                        "skipped": r.get("skipped", False),
                        "skip_reason": r.get("skip_reason", ""),
                    }
                )
            # Use actual worker rate, capped for readability
            max_display_rate = 2.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 1.0
            self.state.score_queue.add(items, display_rate)

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
        if "waiting" in message.lower() or message == "idle":
            self.state.ingest_processing = False
        elif "ingesting" in message.lower():
            self.state.ingest_processing = True
        else:
            self.state.ingest_processing = False

        # Queue results for streaming with adaptive rate
        if results:
            items = []
            for r in results:
                # Use title if already extracted, otherwise extract from id
                title = r.get("title") or r.get("id", "?").split(":")[-1]
                items.append(
                    {
                        "title": title[:60],
                        "score": r.get("score"),
                        "description": r.get("description", ""),
                        "physics_domain": r.get("physics_domain"),
                        "chunk_count": r.get("chunk_count", 0),
                    }
                )
            max_display_rate = 2.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 1.0
            self.state.ingest_queue.add(items, display_rate)

        self._refresh()

    def update_from_graph(
        self,
        total_pages: int = 0,
        pages_scanned: int = 0,
        pages_scored: int = 0,
        pages_ingested: int = 0,
        pages_skipped: int = 0,
        pending_score: int = 0,
        pending_ingest: int = 0,
        accumulated_cost: float = 0.0,
        **kwargs,  # Ignore extra args for compatibility
    ) -> None:
        """Update state from graph statistics."""
        self.state.total_pages = total_pages
        self.state.pages_scanned = pages_scanned
        self.state.pages_scored = pages_scored
        self.state.pages_ingested = pages_ingested
        self.state.pages_skipped = pages_skipped
        self.state.pending_score = pending_score
        self.state.pending_ingest = pending_ingest
        self.state.accumulated_cost = accumulated_cost
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

        # SCORE stats
        total_scored = self.state.pages_scored + self.state.pages_ingested
        summary.append("  SCORE ", style="bold blue")
        summary.append(f"scored={total_scored:,}", style="blue")
        summary.append(f"  skipped={self.state.pages_skipped:,}", style="yellow")
        summary.append(f"  cost=${self.state._run_score_cost:.3f}", style="yellow")
        if self.state.score_rate:
            summary.append(f"  {self.state.score_rate:.1f}/s", style="dim")
        summary.append("\n")

        # INGEST stats
        summary.append("  INGEST", style="bold magenta")
        summary.append(f"  ingested={self.state.pages_ingested:,}", style="magenta")
        if self.state.ingest_rate:
            summary.append(f"  {self.state.ingest_rate:.1f}/s", style="dim")
        summary.append("\n")

        # USAGE stats
        summary.append("  USAGE ", style="bold white")
        summary.append(f"time={format_time(self.state.elapsed)}", style="white")
        summary.append(f"  total_cost=${self.state.run_cost:.2f}", style="yellow")

        # Show coverage percentage
        if self.state.total_pages > 0:
            coverage = total_scored / self.state.total_pages * 100
            summary.append(f"  coverage={coverage:.1f}%", style="cyan")

        return summary
