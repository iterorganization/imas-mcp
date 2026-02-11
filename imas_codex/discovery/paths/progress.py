"""
Progress display for parallel facility path discovery.

Design principles:
- Minimal visual clutter (no emojis, no stopwatch icons)
- Clear hierarchy: Target → Progress → Activity → Resources
- Gradient progress bars with percentage
- Resource gauges for time and cost budgets
- Compact current activity with relevant details only

Uses common progress infrastructure from progress_common module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Import common utilities
from imas_codex.discovery.base.progress import (
    StreamQueue,
    clean_text,
    clip_path,
    clip_text,
    format_time,
    make_bar,
    make_resource_gauge,
)

if TYPE_CHECKING:
    from imas_codex.discovery.paths.parallel import WorkerStats


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
    purpose: str = ""  # Category: experimental_data, modeling_code, etc.
    description: str = ""  # LLM reasoning about why this path is valuable
    skipped: bool = False
    skip_reason: str = ""
    should_expand: bool = True  # False = terminal (won't explore children)
    terminal_reason: str = ""  # TerminalReason enum value when terminal


@dataclass
class EnrichItem:
    """Current enrich activity."""

    path: str
    total_bytes: int = 0
    total_lines: int = 0
    read_matches: int = 0  # Format read pattern matches
    write_matches: int = 0  # Format write pattern matches
    languages: list[str] = field(default_factory=list)  # Top languages found
    is_multiformat: bool = False  # True if read+write patterns found
    pattern_categories: dict[str, int] = field(
        default_factory=dict
    )  # Per-category matches (mdsplus, hdf5, imas, etc.)
    error: str | None = None


# ============================================================================
# Progress State
# ============================================================================


@dataclass
class ProgressState:
    """All state for the progress display."""

    facility: str
    cost_limit: float
    path_limit: int | None = None  # Optional limit from -l flag
    model: str = ""
    focus: str = ""

    # Mode flags
    scan_only: bool = False
    score_only: bool = False

    # Graph totals (aligned with new state machine)
    total: int = 0
    discovered: int = 0  # Awaiting scan
    scanned: int = 0  # Awaiting score
    scored: int = 0  # Scored complete
    skipped: int = 0  # Low value or dead-end
    excluded: int = 0  # Matched exclusion pattern
    max_depth: int = 0  # Maximum tree depth

    # Pending work counts (for progress bars)
    pending_scan: int = 0  # discovered + scanning
    pending_score: int = 0  # scanned + scoring
    pending_expand: int = 0  # scored + should_expand + not expanded
    pending_enrich: int = 0  # scored + should_enrich + not enriched
    pending_rescore: int = 0  # enriched + not rescored

    # This run stats
    run_scanned: int = 0
    run_scored: int = 0
    run_expanded: int = 0
    run_enriched: int = 0
    run_rescored: int = 0
    # Track score and rescore costs separately to avoid double-counting
    _run_score_cost: float = 0.0
    _run_rescore_cost: float = 0.0
    scan_rate: float | None = None
    score_rate: float | None = None
    expand_rate: float | None = None
    enrich_rate: float | None = None
    rescore_rate: float | None = None

    # Accumulated facility cost (from graph)
    accumulated_cost: float = 0.0

    # Current items (and their processing state)
    current_scan: ScanItem | None = None
    current_score: ScoreItem | None = None
    current_enrich: EnrichItem | None = None
    scan_processing: bool = False  # True when awaiting SSH batch
    score_processing: bool = False  # True when awaiting LLM batch
    enrich_processing: bool = False  # True when awaiting SSH enrichment batch

    # Streaming queues - enrich adapts rate based on batch size to fill inter-batch gaps
    scan_queue: StreamQueue = field(default_factory=StreamQueue)
    score_queue: StreamQueue = field(default_factory=StreamQueue)
    enrich_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.5, min_display_time=0.4
        )
    )

    # Tracking
    scored_paths: set[str] = field(default_factory=set)
    start_time: float = field(default_factory=time.time)

    # Enrichment aggregates for summary display
    total_bytes_enriched: int = 0
    total_lines_enriched: int = 0
    total_read_matches: int = 0
    total_write_matches: int = 0
    multiformat_count: int = 0  # Count of paths with both read+write patterns
    pattern_category_totals: dict[str, int] = field(
        default_factory=dict
    )  # Aggregate per-category pattern counts (mdsplus, hdf5, imas, etc.)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def pending_work(self) -> int:
        """Total pending work: scan + score + expand + enrich queues."""
        return (
            self.pending_scan
            + self.pending_score
            + self.pending_expand
            + self.pending_enrich
        )

    @property
    def cost_per_path(self) -> float | None:
        """Average LLM cost per initially scored path (excludes rescore cost)."""
        if self.run_scored > 0:
            return self._run_score_cost / self.run_scored
        return None

    @property
    def estimated_total_cost(self) -> float | None:
        """Estimated total cost based on current rate.

        Predicts cost for all remaining LLM work:
        - pending_score paths at cost_per_path rate (initial scoring)
        - pending_rescore paths at rescore cost rate
        """
        cpp = self.cost_per_path
        if cpp is None:
            return None

        remaining_score_cost = (self.pending_scan + self.pending_score) * cpp

        # Rescore cost rate (separate from initial scoring)
        remaining_rescore_cost = 0.0
        if self.run_rescored > 0 and self._run_rescore_cost > 0:
            cost_per_rescore = self._run_rescore_cost / self.run_rescored
            remaining_rescore_cost = self.pending_rescore * cost_per_rescore

        return self.run_cost + remaining_score_cost + remaining_rescore_cost

    @property
    def run_cost(self) -> float:
        """Total cost for this run (score + rescore)."""
        return self._run_score_cost + self._run_rescore_cost

    @property
    def coverage(self) -> float:
        """Percentage of total paths scored."""
        return (self.scored / self.total * 100) if self.total > 0 else 0

    @property
    def frontier_size(self) -> int:
        """Total paths awaiting work (scan or score)."""
        return self.pending_scan + self.pending_score + self.pending_expand

    @property
    def cost_limit_reached(self) -> bool:
        """Check if cost limit has been reached."""
        return self.run_cost >= self.cost_limit if self.cost_limit > 0 else False

    @property
    def path_limit_reached(self) -> bool:
        """Check if path limit has been reached.

        Uses scored count (terminal state) rather than scan + score
        since scanning is just the start of the pipeline.
        """
        if self.path_limit is None or self.path_limit <= 0:
            return False
        return self.run_scored >= self.path_limit

    @property
    def limit_reason(self) -> str | None:
        """Return which limit was reached, or None if no limit reached."""
        if self.cost_limit_reached:
            return "cost"
        if self.path_limit_reached:
            return "path"
        return None

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time to termination based on limits.

        Priority order:
        1. Cost limit: if -c flag set, estimate time to exhaust budget
        2. Path limit: if -l flag set, estimate time to process that many paths
        3. Full work: max of all worker ETAs (parallel pipeline)
           Terminal time = slowest worker group since they run concurrently.
        """
        # Try cost-based ETA first (if we have cost data)
        if self.run_cost > 0 and self.cost_limit > 0:
            cost_rate = self.run_cost / self.elapsed if self.elapsed > 0 else 0
            if cost_rate > 0:
                remaining_budget = self.cost_limit - self.run_cost
                return max(0, remaining_budget / cost_rate)

        # Try path-limit-based ETA (if -l flag set)
        if self.path_limit is not None and self.path_limit > 0:
            # Use scored count (terminal state) for path limit ETA
            if self.run_scored > 0 and self.elapsed > 0:
                rate = self.run_scored / self.elapsed
                remaining = self.path_limit - self.run_scored
                return max(0, remaining / rate) if rate > 0 else None

        # Fall back to work-based ETA from slowest worker group.
        # Each worker group has its own remaining work and processing rate.
        # Terminal time = max of all worker ETAs (parallel pipeline).
        worker_etas: list[float] = []

        # Scan + expand pipeline: pending_scan paths at combined scan+expand rate
        combined_scan_rate = sum(r for r in [self.scan_rate, self.expand_rate] if r)
        if self.pending_scan > 0 and combined_scan_rate > 0:
            worker_etas.append(self.pending_scan / combined_scan_rate)

        # Score pipeline: pending_score paths at combined score+rescore rate
        combined_score_rate = sum(r for r in [self.score_rate, self.rescore_rate] if r)
        if self.pending_score > 0 and combined_score_rate > 0:
            worker_etas.append(self.pending_score / combined_score_rate)

        # Expand pipeline: pending_expand at expand rate
        if self.pending_expand > 0 and self.expand_rate and self.expand_rate > 0:
            worker_etas.append(self.pending_expand / self.expand_rate)

        # Enrich pipeline: pending_enrich at enrich rate
        if self.pending_enrich > 0 and self.enrich_rate and self.enrich_rate > 0:
            worker_etas.append(self.pending_enrich / self.enrich_rate)

        # Rescore pipeline: pending_rescore at rescore rate
        if self.pending_rescore > 0 and self.rescore_rate and self.rescore_rate > 0:
            worker_etas.append(self.pending_rescore / self.rescore_rate)

        if worker_etas:
            return max(worker_etas)

        # No rate data yet - can't estimate
        return None


# ============================================================================
# Main Display Class
# ============================================================================


class ParallelProgressDisplay:
    """Clean progress display for parallel discovery.

    Layout (100 chars wide - matches summary panel):
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                               ITER Discovery                                                     │
    │                           Focus: equilibrium codes                                               │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  SCAN   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━─────────────────────────  1,234  42%  77.1/s        │
    │  SCORE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━─────────────────────────────     892  28%   3.2/s        │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  /gss/work/imas/codes/chease/src                                                                 │
    │    Scan:  45 files, 3 dirs, code project                                                         │
    │    Score: 0.85 simulation_code                                                                   │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  COST   ▐████████████░░░░░░░░░░░░░░▌  $4.50 / $10.00                                             │
    │  TIME    ━━━━━━━━━━━━━━━━━━━━━━━━━━  12m 30s  ETA 8m                                            │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
    """

    # Layout constants - label widths are fixed, bars fill remaining space
    LABEL_WIDTH = 10  # "  SCAN    " etc
    MIN_WIDTH = 80
    METRICS_WIDTH = 22  # " {count:>6,} {pct:>3.0f}% {rate:>5.1f}/s"
    GAUGE_METRICS_WIDTH = 28  # "  {time}  ETA {eta}" or "  ${cost:.2f} / ${limit:.2f}"

    def __init__(
        self,
        facility: str,
        cost_limit: float,
        path_limit: int | None = None,
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
            path_limit=path_limit,
            model=model,
            focus=focus,
            scan_only=scan_only,
            score_only=score_only,
        )
        self._live: Live | None = None

    @property
    def width(self) -> int:
        """Get display width based on terminal size (fills terminal)."""
        term_width = self.console.width or 100
        return max(self.MIN_WIDTH, term_width)

    @property
    def bar_width(self) -> int:
        """Calculate progress bar width to fill available space."""
        return self.width - 4 - self.LABEL_WIDTH - self.METRICS_WIDTH

    @property
    def gauge_width(self) -> int:
        """Calculate resource gauge width to match progress bars."""
        return self.bar_width

    def _build_header(self) -> Text:
        """Build centered header with facility and focus."""
        header = Text()

        # Facility name with mode indicator
        title = f"{self.state.facility.upper()} Discovery"
        if self.state.scan_only:
            title += " (SCAN ONLY)"
        elif self.state.score_only:
            title += " (SCORE ONLY)"
        header.append(title.center(self.width - 4), style="bold cyan")

        # Focus (if set and not scan_only)
        if self.state.focus and not self.state.scan_only:
            header.append("\n")
            focus_line = f"Focus: {self.state.focus}"
            header.append(focus_line.center(self.width - 4), style="italic dim")

        return header

    def _build_progress_section(self) -> Text:
        """Build the main progress bars for scan and score.

        Pipeline: discovered → scanning → scanned → scoring → scored/skipped
                                                            (excluded at any point)

        SCAN shows: paths that completed scanning / all paths needing to be scanned
        SCORE shows: paths that completed scoring / all paths needing to scored

        The key insight: SCAN completion includes paths currently being scored
        (they have already been scanned), while SCORE completion only includes
        terminal states (scored + skipped).

        Note: `excluded` paths have status='skipped' AND terminal_reason='excluded'.
        They are counted in both `skipped` and `excluded` stats. We subtract
        `excluded` from both numerator and denominator to avoid double-counting.
        """
        section = Text()

        bar_width = self.bar_width

        # SCAN progress: paths that have finished scanning vs all that need scanning
        # Completed scan = scanned + scoring + scored + (skipped - excluded)
        #   - excluded paths were pre-filtered, never actually scanned
        #   - skipped includes excluded, so subtract to avoid double-counting
        # Total needing scan = total - excluded (everything except excluded needs scanning)
        scanned_paths = (
            self.state.pending_score
            + self.state.scored
            + self.state.skipped
            - self.state.excluded
        )
        scan_total = self.state.total - self.state.excluded
        if scan_total <= 0:
            scan_total = 1
        scan_pct = (scanned_paths / scan_total * 100) if scan_total > 0 else 0

        # SCORE progress: paths that completed LLM scoring vs all needing scoring
        # Excludes 'skipped' since those include bulk-skipped paths that never
        # entered the scorer queue (e.g., data container children)
        # Completed = scored only (paths that went through LLM)
        # Total = pending_score + scored
        scored_paths = self.state.scored
        score_total = self.state.pending_score + self.state.scored
        if score_total <= 0:
            score_total = 1
        score_pct = (scored_paths / score_total * 100) if score_total > 0 else 0

        # SCAN row: shows full exploration scanning progress
        if self.state.score_only:
            section.append("  SCAN    ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  SCAN    ", style="bold blue")
            scan_ratio = min(scanned_paths / scan_total, 1.0) if scan_total > 0 else 0
            section.append(make_bar(scan_ratio, bar_width), style="blue")
            section.append(f" {scanned_paths:>6,}", style="bold")
            section.append(f" {scan_pct:>3.0f}%", style="cyan")
            # Show combined rate (scan + expand)
            combined_rate = sum(
                r for r in [self.state.scan_rate, self.state.expand_rate] if r
            )
            if combined_rate > 0:
                section.append(f" {combined_rate:>5.1f}/s", style="dim")
        section.append("\n")

        # SCORE row: shows full exploration scoring progress
        if self.state.scan_only:
            section.append("  SCORE   ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  SCORE   ", style="bold green")
            score_ratio = min(scored_paths / score_total, 1.0) if score_total > 0 else 0
            section.append(make_bar(score_ratio, bar_width), style="green")
            section.append(f" {scored_paths:>6,}", style="bold")
            section.append(f" {score_pct:>3.0f}%", style="cyan")
            # Show combined rate (score + rescore)
            combined_rate = sum(
                r for r in [self.state.score_rate, self.state.rescore_rate] if r
            )
            if combined_rate > 0:
                section.append(f" {combined_rate:>5.1f}/s", style="dim")
        section.append("\n")

        # ENRICH row: shows deep analysis progress on high-value paths
        # Enrichment runs on paths where should_enrich=true (scored by LLM)
        if self.state.scan_only:
            section.append("  ENRICH  ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  ENRICH  ", style="bold magenta")
            # Completed enrichment = run_enriched
            # Total = pending_enrich + run_enriched
            enrich_total = self.state.pending_enrich + self.state.run_enriched
            if enrich_total <= 0:
                enrich_total = 1
            enrich_ratio = min(self.state.run_enriched / enrich_total, 1.0)
            enrich_pct = (
                self.state.run_enriched / enrich_total * 100 if enrich_total > 0 else 0
            )
            section.append(make_bar(enrich_ratio, bar_width), style="magenta")
            section.append(f" {self.state.run_enriched:>6,}", style="bold")
            section.append(f" {enrich_pct:>3.0f}%", style="cyan")
            if self.state.enrich_rate and self.state.enrich_rate > 0:
                section.append(f" {self.state.enrich_rate:>5.1f}/s", style="dim")

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

        # Helper to determine if worker should show "idle"
        # Only show idle when worker is not processing AND queue is empty
        def should_show_idle(processing: bool, queue: StreamQueue) -> bool:
            return not processing and queue.is_empty()

        # SCAN section - always 2 lines for consistent height
        section.append("  SCAN ", style="bold blue")
        if scan:
            section.append(clip_path(scan.path, self.width - 10), style="white")
            section.append("\n")
            # Stats indented below
            section.append("    ", style="dim")
            section.append(f"{scan.files} files", style="cyan")
            section.append(", ", style="dim")
            section.append(f"{scan.dirs} dirs", style="cyan")
            if scan.has_code:
                section.append("  ", style="dim")
                section.append("code project", style="green dim")
        elif self.state.scan_processing:
            section.append("processing batch...", style="cyan italic")
            section.append("\n    ", style="dim")  # Empty second line
        elif should_show_idle(self.state.scan_processing, self.state.scan_queue):
            section.append("idle", style="dim italic")
            section.append("\n    ", style="dim")  # Empty second line
        else:
            # Queue has items but nothing displayed yet (waiting for tick)
            section.append("...", style="dim italic")
            section.append("\n    ", style="dim")
        section.append("\n")

        # SCORE section - always 2 lines for consistent height (skip in scan_only mode)
        if not self.state.scan_only:
            section.append("  SCORE ", style="bold green")
            if score:
                # Show path with terminal indicator
                path_display = clip_path(score.path, self.width - 20)
                section.append(path_display, style="white")
                if not score.should_expand:
                    section.append(" terminal", style="magenta")
                section.append("\n")
                # Score details indented below (matching SCAN layout)
                section.append("    ", style="dim")

                # Always show score if available
                if score.score is not None:
                    # Color code the score
                    if score.score >= 0.7:
                        style = "bold green"
                    elif score.score >= 0.4:
                        style = "yellow"
                    else:
                        style = "red"
                    section.append(f"{score.score:.2f}", style=style)

                    # Calculate available width for description
                    # Layout: "    0.85  " = 10 chars, leave 2 for padding
                    desc_width = self.width - 12

                    # Build description for line 2
                    # Priority: description (LLM reasoning) > purpose (category)
                    # For terminal paths with terminal_reason (empty, access_denied), show that
                    if score.terminal_reason:
                        desc = score.terminal_reason.replace("_", " ")
                    elif score.description:
                        desc = clean_text(score.description)
                    elif score.purpose:
                        desc = clean_text(score.purpose)
                    else:
                        desc = ""

                    if desc:
                        # Use clip_text for end-clipping, cap at 65 to prevent wrapping
                        clipped_desc = clip_text(desc, min(desc_width, 65))
                        section.append(f"  {clipped_desc}", style="italic dim")
                elif score.skipped:
                    # No score available, just show skipped status
                    desc_width = self.width - 16  # "    skipped  " = ~12 chars
                    section.append("skipped", style="yellow")
                    if score.skip_reason:
                        reason = clean_text(score.skip_reason)
                        section.append(
                            f"  {clip_text(reason, desc_width)}", style="dim"
                        )
            elif self.state.score_processing:
                section.append("processing batch...", style="cyan italic")
                section.append("\n    ", style="dim")  # Empty second line
            elif should_show_idle(self.state.score_processing, self.state.score_queue):
                section.append("idle", style="dim italic")
                section.append("\n    ", style="dim")  # Empty second line
            else:
                # Queue has items but nothing displayed yet
                section.append("...", style="dim italic")
                section.append("\n    ", style="dim")
            section.append("\n")

        # ENRICH section - always 2 lines for consistent height (skip in scan_only mode)
        if not self.state.scan_only:
            enrich = self.state.current_enrich
            queue_empty = self.state.enrich_queue.is_empty()
            section.append("  ENRICH", style="bold magenta")
            # Show current item if queue still has items OR we're still processing
            # Once queue is drained and worker is idle, show "idle" instead of stale item
            if enrich and (not queue_empty or self.state.enrich_processing):
                section.append(clip_path(enrich.path, self.width - 11), style="white")
                section.append("\n")
                # Stats indented below
                section.append("    ", style="dim")
                # Format bytes nicely
                if enrich.total_bytes >= 1_000_000:
                    size_str = f"{enrich.total_bytes / 1_000_000:.1f}MB"
                elif enrich.total_bytes >= 1_000:
                    size_str = f"{enrich.total_bytes / 1_000:.1f}KB"
                else:
                    size_str = f"{enrich.total_bytes}B"
                section.append(size_str, style="cyan")
                if enrich.total_lines > 0:
                    section.append(f"  {enrich.total_lines:,} LOC", style="cyan")
                # Show top languages
                if enrich.languages:
                    langs = ", ".join(enrich.languages[:3])
                    section.append(f"  [{langs}]", style="green dim")
                # Show multiformat indicator
                if enrich.is_multiformat:
                    section.append("  multiformat", style="yellow")
                elif enrich.read_matches > 0 or enrich.write_matches > 0:
                    section.append(
                        f"  r:{enrich.read_matches} w:{enrich.write_matches}",
                        style="dim",
                    )
            elif self.state.enrich_processing:
                section.append(" processing batch...", style="cyan italic")
                section.append("\n    ", style="dim")  # Empty second line
            elif queue_empty:
                # Queue drained and not processing - truly idle
                section.append(" idle", style="dim italic")
                section.append("\n    ", style="dim")  # Empty second line
            else:
                # Queue has items but nothing displayed yet (waiting for rate limit)
                section.append(" ...", style="dim italic")
                section.append("\n    ", style="dim")

        return section

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges.

        Order: TIME, COST, TOTAL (as requested for visual flow).
        """
        section = Text()

        # TIME row first - elapsed with ETA
        section.append("  TIME    ", style="bold cyan")

        # Estimate total time if we have an ETA
        eta = None if self.state.scan_only else self.state.eta_seconds
        if eta is not None and eta > 0:
            total_est = self.state.elapsed + eta
            section.append_text(
                make_resource_gauge(self.state.elapsed, total_est, self.bar_width)
            )
        else:
            # Unknown total - show elapsed only with full bar (complete or unknown)
            section.append("━" * self.bar_width, style="cyan")

        section.append(f"  {format_time(self.state.elapsed)}", style="bold")

        if eta is not None:
            if eta <= 0:
                # Show which limit was reached - budget, path, or complete
                limit_reason = self.state.limit_reason
                if limit_reason == "cost":
                    section.append("  cost limit reached", style="yellow dim")
                elif limit_reason == "path":
                    section.append("  path limit reached", style="yellow dim")
                else:
                    section.append("  complete", style="green dim")
            else:
                section.append(f"  ETA {format_time(eta)}", style="dim")
        section.append("\n")

        # COST row - this run's cost against budget (hidden in scan_only mode)
        if not self.state.scan_only:
            section.append("  COST    ", style="bold yellow")
            # Cost bar uses cost_limit as 100% - no estimates
            section.append_text(
                make_resource_gauge(
                    self.state.run_cost, self.state.cost_limit, self.bar_width
                )
            )
            section.append(f"  ${self.state.run_cost:.2f}", style="bold")
            section.append(f" / ${self.state.cost_limit:.2f}", style="dim")
            section.append("\n")

            # TOTAL row - progress toward estimated total cost (ETC)
            # Always show if we have any cost data (accumulated or current run)
            total_facility_cost = self.state.accumulated_cost + self.state.run_cost
            if total_facility_cost > 0 or self.state.pending_score > 0:
                section.append("  TOTAL   ", style="bold white")
                # Dynamic ETC based on cost per path and remaining work
                # Include both score and rescore pending work
                cpp = self.state.cost_per_path
                etc = total_facility_cost  # Estimated Total Cost
                if cpp and cpp > 0:
                    pending_score_paths = (
                        self.state.pending_scan + self.state.pending_score
                    )
                    etc += pending_score_paths * cpp
                # Add rescore cost estimate
                if self.state.run_rescored > 0 and self.state._run_rescore_cost > 0:
                    cost_per_rescore = (
                        self.state._run_rescore_cost / self.state.run_rescored
                    )
                    etc += self.state.pending_rescore * cost_per_rescore

                # Progress bar shows current cost toward ETC
                if etc > 0:
                    section.append_text(
                        make_resource_gauge(total_facility_cost, etc, self.bar_width)
                    )
                else:
                    section.append("━" * self.bar_width, style="white")

                section.append(f"  ${total_facility_cost:.2f}", style="bold")
                # Show ETC (dynamic estimate)
                if etc > total_facility_cost:
                    section.append(f"  ETC ${etc:.2f}", style="dim")
                section.append("\n")

        # STATS row - terminal state counts from graph (not session-based)
        # Shows actual graph state, not just this run's processed counts
        section.append("  STATS   ", style="bold magenta")
        section.append(f"depth={self.state.max_depth}", style="cyan")

        # Terminal states (paths that have reached end of pipeline)
        section.append(f"  scored={self.state.scored}", style="green")
        section.append(f"  skipped={self.state.skipped}", style="yellow")

        # Pending work by worker type - show what each worker is waiting for
        # Format: scan:5 expand:12 enrich:3 rescore:0
        pending_parts = []
        if self.state.pending_scan > 0:
            pending_parts.append(f"scan:{self.state.pending_scan}")
        if self.state.pending_expand > 0:
            pending_parts.append(f"expand:{self.state.pending_expand}")
        if self.state.pending_enrich > 0:
            pending_parts.append(f"enrich:{self.state.pending_enrich}")
        if self.state.pending_rescore > 0:
            pending_parts.append(f"rescore:{self.state.pending_rescore}")

        if pending_parts:
            section.append(f"  pending=[{' '.join(pending_parts)}]", style="cyan dim")

        # Excluded last
        section.append(f"  excluded={self.state.excluded}", style="dim")

        return section

    def _build_display(self) -> Panel:
        """Build the complete display."""
        sections = [
            self._build_header(),
            Text("─" * (self.width - 4), style="dim"),
            self._build_progress_section(),
            Text("─" * (self.width - 4), style="dim"),
            self._build_activity_section(),
            Text("─" * (self.width - 4), style="dim"),
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
            width=self.width,
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

        # Track processing state for display
        # Don't clear current_scan when idle - let queue drain naturally
        # via tick(). Only update the processing flag.
        if message == "idle":
            self.state.scan_processing = False
            self._refresh()
            return
        elif "scanning" in message.lower():
            # About to run SSH scan - mark as processing
            self.state.scan_processing = True
        else:
            # Got results back ("scanned N paths")
            self.state.scan_processing = False

        # Queue scan results for streaming (tick() handles rate-limited popping)
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
            # Use actual worker rate, capped for readability
            # Max 2.0/s = each item visible for at least 0.5s
            max_display_rate = 2.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 1.0
            self.state.scan_queue.add(items, display_rate)

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

        # Track processing state for display
        # Don't clear current_score when waiting - let queue drain naturally
        # via tick(). Only update the processing flag.
        if "waiting" in message.lower():
            self.state.score_processing = False
            self._refresh()
            return
        elif "scoring" in message.lower():
            # About to call LLM - mark as processing
            self.state.score_processing = True
        else:
            # Got results back
            self.state.score_processing = False

        # Queue score results for streaming (tick() handles rate-limited popping)
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
                        description=r.get("description", ""),
                        skipped=bool(r.get("skip_reason")),
                        skip_reason=r.get("skip_reason", ""),
                        should_expand=r.get("should_expand", True),
                        terminal_reason=r.get("terminal_reason", ""),
                    )
                )
            # Use actual worker rate, capped for readability
            # Max 2.0/s = each item visible for at least 0.5s
            max_display_rate = 2.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 1.0
            self.state.score_queue.add(items, display_rate)

        self._refresh()

    def update_expand(
        self,
        message: str,
        stats: WorkerStats,
        paths: list[str] | None = None,
        scan_results: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update expand worker state.

        Expand results are added to the scan queue since they are scan operations
        on high-value directories.
        """
        self.state.run_expanded = stats.processed
        self.state.expand_rate = stats.rate

        # Queue expand results to scan stream (they are scans of valuable paths)
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

        self._refresh()

    def update_enrich(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update enrich worker state with enrichment results."""
        self.state.run_enriched = stats.processed
        self.state.enrich_rate = stats.rate

        # Track processing state for display
        # Don't clear current_enrich when waiting - let queue drain naturally
        # via tick(). Only update the processing flag.
        if "waiting" in message.lower():
            self.state.enrich_processing = False
            self._refresh()
            return
        elif "enriching" in message.lower():
            # About to run SSH enrichment - mark as processing
            self.state.enrich_processing = True
        else:
            # Got results back
            self.state.enrich_processing = False

        # Queue enrich results for streaming display
        if results:
            items = []
            for r in results:
                # Parse language breakdown if present
                lang_breakdown = r.get("language_breakdown", {})
                if isinstance(lang_breakdown, str):
                    import json

                    try:
                        lang_breakdown = json.loads(lang_breakdown)
                    except json.JSONDecodeError:
                        lang_breakdown = {}

                # Get top languages by line count
                top_langs = sorted(
                    lang_breakdown.items(), key=lambda x: x[1], reverse=True
                )[:3]
                languages = [lang for lang, _ in top_langs]

                read_matches = r.get("read_matches", 0) or 0
                write_matches = r.get("write_matches", 0) or 0
                is_multi = r.get("is_multiformat", False) or (
                    read_matches > 0 and write_matches > 0
                )

                total_bytes = r.get("total_bytes", 0) or 0
                total_lines = r.get("total_lines", 0) or 0

                # Extract pattern categories (mdsplus, hdf5, imas, etc.)
                pattern_cats = r.get("pattern_categories", {}) or {}
                if isinstance(pattern_cats, str):
                    import json

                    try:
                        pattern_cats = json.loads(pattern_cats)
                    except json.JSONDecodeError:
                        pattern_cats = {}

                # Update aggregate statistics
                self.state.total_bytes_enriched += total_bytes
                self.state.total_lines_enriched += total_lines
                self.state.total_read_matches += read_matches
                self.state.total_write_matches += write_matches
                if is_multi:
                    self.state.multiformat_count += 1

                # Aggregate pattern categories
                for cat, count in pattern_cats.items():
                    self.state.pattern_category_totals[cat] = (
                        self.state.pattern_category_totals.get(cat, 0) + count
                    )

                items.append(
                    EnrichItem(
                        path=r.get("path", ""),
                        total_bytes=total_bytes,
                        total_lines=total_lines,
                        read_matches=read_matches,
                        write_matches=write_matches,
                        pattern_categories=pattern_cats,
                        languages=languages,
                        is_multiformat=is_multi,
                        error=r.get("error"),
                    )
                )
            # Use actual worker rate, capped for readability
            # Max 2.0/s = each item visible for at least 0.5s
            max_display_rate = 2.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 1.0
            self.state.enrich_queue.add(items, display_rate)

        self._refresh()

    def update_rescore(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update rescore worker state.

        Rescore results are added to the score queue since they update scores.
        """
        self.state.run_rescored = stats.processed
        self.state.rescore_rate = stats.rate
        # Track rescore cost separately (cumulative from rescore worker)
        self.state._run_rescore_cost = stats.cost

        # Queue rescore results to score stream
        if results:
            items = []
            for r in results:
                path = r.get("path", "")
                # Use adjustment_reason for display, falling back to "rescored"
                reason = r.get("adjustment_reason", "rescored")
                items.append(
                    ScoreItem(
                        path=path,
                        score=r.get("score"),
                        purpose="rescored",
                        description=reason if reason != "rescored" else "",
                        skipped=False,
                        skip_reason="",
                        should_expand=r.get("should_expand", True),
                    )
                )
            self.state.score_queue.add(items, stats.rate if stats.rate else 1.0)

        self._refresh()

    def refresh_from_graph(self, facility: str) -> None:
        """Refresh totals from graph database."""
        from imas_codex.discovery.paths.frontier import (
            get_accumulated_cost,
            get_discovery_stats,
        )

        stats = get_discovery_stats(facility)
        self.state.total = stats["total"]
        self.state.discovered = stats["discovered"]
        self.state.scanned = stats["scanned"]
        self.state.scored = stats["scored"]
        self.state.skipped = stats["skipped"]
        self.state.excluded = stats["excluded"]
        self.state.max_depth = stats["max_depth"]
        # Calculate pending work counts including expansion_ready
        self._calculate_pending_from_stats(stats)

        # Get accumulated facility cost from graph
        cost_data = get_accumulated_cost(facility)
        self.state.accumulated_cost = cost_data["total_cost"]

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

        next_enrich = self.state.enrich_queue.pop()
        if next_enrich:
            self.state.current_enrich = next_enrich
            updated = True

        if updated:
            self._refresh()

    def get_paths_scored_this_run(self) -> set[str]:
        """Get paths scored during this run."""
        return self.state.scored_paths

    def get_enrichment_aggregates(self) -> dict:
        """Get aggregated enrichment statistics.

        Returns dict with:
            total_bytes: Total bytes across enriched paths
            total_lines: Total lines of code across enriched paths
            total_read_matches: Total format read pattern matches
            total_write_matches: Total format write pattern matches
            multiformat_count: Count of paths with both read+write patterns
            pattern_categories: Dict of pattern category -> match count
        """
        return {
            "total_bytes": self.state.total_bytes_enriched,
            "total_lines": self.state.total_lines_enriched,
            "total_read_matches": self.state.total_read_matches,
            "total_write_matches": self.state.total_write_matches,
            "multiformat_count": self.state.multiformat_count,
            "pattern_categories": dict(self.state.pattern_category_totals),
        }

    def _calculate_pending_from_stats(self, stats: dict) -> None:
        """Calculate pending work counts from graph stats."""
        scanning = stats.get("scanning", 0)
        scoring = stats.get("scoring", 0)
        self.state.pending_scan = stats.get("discovered", 0) + scanning
        self.state.pending_score = stats.get("scanned", 0) + scoring
        self.state.pending_expand = stats.get("expansion_ready", 0)
        self.state.pending_enrich = stats.get("enrichment_ready", 0)

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_display())


def print_discovery_status(
    facility: str,
    console: Console | None = None,
    use_rich: bool = True,
    domain: str | None = None,
) -> None:
    """Print a formatted discovery status report.

    Args:
        facility: Facility ID
        console: Optional Rich console
        use_rich: Whether to use Rich formatting (False for LLM tools)
        domain: Optional domain filter ("paths", "wiki", "signals", or None for all)
    """
    import re

    from imas_codex.discovery.paths.frontier import (
        get_discovery_stats,
        get_high_value_paths,
        get_purpose_distribution,
        get_top_paths_by_purpose,
    )

    def strip_rich_markup(text: str) -> str:
        """Remove Rich markup tags like [bold], [cyan], etc."""
        return re.sub(r"\[/?[a-z_]+\]", "", text)

    def output(text: str) -> None:
        """Print with or without Rich markup."""
        if use_rich:
            console.print(text)
        else:
            print(strip_rich_markup(text))

    console = console or Console()

    # Header
    if domain:
        output(f"\n[bold]Facility: {facility} – {domain.title()} Discovery[/bold]")
    else:
        output(f"\n[bold]Facility: {facility}[/bold]")

    # --------------------------------------------------------------------------
    # Paths domain
    # --------------------------------------------------------------------------
    if domain is None or domain == "paths":
        stats = get_discovery_stats(facility)
        total = stats.get("total", 0)
        if total > 0:
            if domain is None:
                output("\n[bold]Paths Discovery:[/bold]")
            output(f"Total paths: {total:,}")

            discovered = stats.get("discovered", 0)
            scanned = stats.get("scanned", 0)
            scored = stats.get("scored", 0)
            skipped = stats.get("skipped", 0)
            excluded = stats.get("excluded", 0)
            max_depth = stats.get("max_depth", 0)

            output(f"├─ Discovered: {discovered:,} ({discovered / total * 100:.1f}%)")
            output(f"├─ Scanned:    {scanned:,} ({scanned / total * 100:.1f}%)")
            output(f"├─ Scored:     {scored:,} ({scored / total * 100:.1f}%)")
            output(f"├─ Skipped:    {skipped:,} ({skipped / total * 100:.1f}%)")
            output(f"└─ Excluded:   {excluded:,} ({excluded / total * 100:.1f}%)")

            # Purpose distribution with top paths per category
            purpose_dist = get_purpose_distribution(facility)
            if purpose_dist:
                output("\n[bold]By Purpose (top 3 per category):[/bold]")

                categories = [
                    ("Modeling Code", "cyan", ["modeling_code"]),
                    ("Analysis Code", "green", ["analysis_code", "operations_code"]),
                    ("Data", "yellow", ["modeling_data", "experimental_data"]),
                    (
                        "Infrastructure",
                        "blue",
                        ["data_access", "workflow", "visualization"],
                    ),
                    ("Documentation", "magenta", ["documentation"]),
                ]

                for cat_name, color, purposes in categories:
                    purpose_count = sum(purpose_dist.get(p, 0) for p in purposes)
                    if purpose_count == 0:
                        continue

                    output(f"\n[{color}]{cat_name}[/{color}] ({purpose_count:,} paths)")

                    for purpose in purposes:
                        if purpose_dist.get(purpose, 0) == 0:
                            continue
                        top_paths = get_top_paths_by_purpose(facility, purpose, limit=3)
                        if top_paths:
                            for p in top_paths:
                                output(f"  [{p['score']:.2f}] [dim]{p['path']}[/dim]")

                structural_purposes = [
                    "container",
                    "archive",
                    "build_artifact",
                    "system",
                ]
                structural = sum(purpose_dist.get(p, 0) for p in structural_purposes)
                if structural > 0:
                    output(f"\n[dim]Structural[/dim] ({structural:,} paths)")

            frontier = discovered + scanned
            output(f"\nFrontier: {frontier} paths awaiting work")
            output(f"Max depth: {max_depth}")
            coverage = scored / total * 100 if total > 0 else 0
            output(f"Coverage: {coverage:.1f}% scored")

            high_value = get_high_value_paths(facility, min_score=0.7, limit=10)
            if high_value:
                output(f"High-value paths (score > 0.7): {len(high_value)}")
                for p in high_value[:5]:
                    output(f"  [{p['score']:.2f}] {p['path']}")
                if len(high_value) > 5:
                    output(f"  ... and {len(high_value) - 5} more")
        elif domain == "paths":
            output("No paths discovered")

    # --------------------------------------------------------------------------
    # Wiki domain
    # --------------------------------------------------------------------------
    if domain is None or domain == "wiki":
        try:
            from imas_codex.discovery.wiki import get_wiki_discovery_stats

            wiki_stats = get_wiki_discovery_stats(facility)
            wiki_total = wiki_stats.get("total", 0)
            if wiki_total > 0:
                if domain is None:
                    output("\n[bold]Wiki Discovery:[/bold]")
                wiki_scanned = wiki_stats.get("scanned", 0)
                wiki_scored = wiki_stats.get("scored", 0)
                wiki_ingested = wiki_stats.get("ingested", 0)
                wiki_skipped = wiki_stats.get("skipped", 0)
                wiki_cost = wiki_stats.get("accumulated_cost", 0.0)

                output(f"Total pages: {wiki_total:,}")
                output(f"├─ Scanned:   {wiki_scanned:,}")
                output(f"├─ Scored:    {wiki_scored:,}")
                output(f"├─ Ingested:  {wiki_ingested:,}")
                output(f"└─ Skipped:   {wiki_skipped:,}")

                # Artifact stats
                total_artifacts = wiki_stats.get("total_artifacts", 0)
                if total_artifacts > 0:
                    art_scored = wiki_stats.get("artifacts_scored", 0)
                    art_ingested = wiki_stats.get("artifacts_ingested", 0)
                    art_pending_score = wiki_stats.get("pending_artifact_score", 0)
                    art_pending_ingest = wiki_stats.get("pending_artifact_ingest", 0)
                    output(f"\nArtifacts: {total_artifacts:,}")
                    output(f"├─ Scored:    {art_scored:,}")
                    output(f"├─ Ingested:  {art_ingested:,}")
                    output(
                        f"└─ Pending:   "
                        f"score={art_pending_score:,}, "
                        f"ingest={art_pending_ingest:,}"
                    )

                output(f"Accumulated cost: ${wiki_cost:.2f}")
            elif domain == "wiki":
                output("No wiki pages discovered")
        except Exception:
            if domain == "wiki":
                output("Wiki stats unavailable")

    # --------------------------------------------------------------------------
    # Signals domain
    # --------------------------------------------------------------------------
    if domain is None or domain == "signals":
        try:
            from imas_codex.discovery.data import get_data_discovery_stats

            signal_stats = get_data_discovery_stats(facility)
            signal_total = signal_stats.get("total", 0)
            if signal_total > 0:
                if domain is None:
                    output("\n[bold]Signal Discovery:[/bold]")
                scanned = signal_stats.get("scanned", 0)
                enriched = signal_stats.get("enriched", 0)
                checked = signal_stats.get("checked", 0)
                skipped = signal_stats.get("skipped", 0)
                cost = signal_stats.get("accumulated_cost", 0.0)

                output(f"Total signals: {signal_total:,}")
                output(f"├─ Scanned:   {scanned:,}")
                output(f"├─ Enriched:  {enriched:,}")
                output(f"├─ Checked:   {checked:,}")
                output(f"└─ Skipped:   {skipped:,}")
                if cost > 0:
                    output(f"Accumulated cost: ${cost:.2f}")
            elif domain == "signals":
                output("No signals discovered")
        except Exception:
            if domain == "signals":
                output("Signal stats unavailable")
