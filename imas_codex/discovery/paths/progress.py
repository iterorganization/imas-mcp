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
from rich.text import Text

# Import common utilities
from imas_codex.discovery.base.progress import (
    LABEL_WIDTH,
    BaseProgressDisplay,
    PipelineRowConfig,
    ResourceConfig,
    StreamQueue,
    build_pipeline_section,
    build_resource_section,
    clean_text,
    clip_path,
    clip_text,
)

if TYPE_CHECKING:
    from imas_codex.discovery.base.progress import WorkerStats


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
    warnings: list[str] = field(default_factory=list)  # e.g., ["tokei_timeout"]


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
    pending_refine: int = 0  # enriched + not refined

    # Graph-persistent totals (from get_discovery_stats)
    enriched: int = 0  # Total enriched paths in graph
    refined: int = 0  # Total refined paths in graph

    # This run stats
    run_scanned: int = 0
    run_scored: int = 0
    run_expanded: int = 0
    run_enriched: int = 0
    run_refined: int = 0
    # Track score and refine costs separately to avoid double-counting
    _run_score_cost: float = 0.0
    _run_refine_cost: float = 0.0
    scan_rate: float | None = None
    score_rate: float | None = None
    expand_rate: float | None = None
    enrich_rate: float | None = None
    refine_rate: float | None = None

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
        """Average LLM cost per initially scored path (excludes refine cost)."""
        if self.run_scored > 0:
            return self._run_score_cost / self.run_scored
        return None

    @property
    def estimated_total_cost(self) -> float | None:
        """Estimated total cost based on current rate.

        Predicts cost for all remaining LLM work:
        - pending_score paths at cost_per_path rate (initial scoring)
        - pending_refine paths at refine cost rate
        """
        cpp = self.cost_per_path
        if cpp is None:
            return None

        remaining_score_cost = (self.pending_scan + self.pending_score) * cpp

        # Refine cost rate (separate from initial scoring)
        remaining_refine_cost = 0.0
        if self.run_refined > 0 and self._run_refine_cost > 0:
            cost_per_refine = self._run_refine_cost / self.run_refined
            remaining_refine_cost = self.pending_refine * cost_per_refine

        return self.run_cost + remaining_score_cost + remaining_refine_cost

    @property
    def run_cost(self) -> float:
        """Total cost for this run (score + refine)."""
        return self._run_score_cost + self._run_refine_cost

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

        # Score pipeline: pending_score paths at combined score+refine rate
        combined_score_rate = sum(r for r in [self.score_rate, self.refine_rate] if r)
        if self.pending_score > 0 and combined_score_rate > 0:
            worker_etas.append(self.pending_score / combined_score_rate)

        # Expand pipeline: pending_expand at expand rate
        if self.pending_expand > 0 and self.expand_rate and self.expand_rate > 0:
            worker_etas.append(self.pending_expand / self.expand_rate)

        # Enrich pipeline: pending_enrich at enrich rate
        if self.pending_enrich > 0 and self.enrich_rate and self.enrich_rate > 0:
            worker_etas.append(self.pending_enrich / self.enrich_rate)

        # Refine pipeline: pending_refine at refine rate
        if self.pending_refine > 0 and self.refine_rate and self.refine_rate > 0:
            worker_etas.append(self.pending_refine / self.refine_rate)

        if worker_etas:
            return max(worker_etas)

        # No rate data yet - can't estimate
        return None


# ============================================================================
# Main Display Class
# ============================================================================


class ParallelProgressDisplay(BaseProgressDisplay):
    """Clean progress display for parallel discovery.

    Extends ``BaseProgressDisplay`` for the paths discovery pipeline
    (SCAN → SCORE → ENRICH).  Inherits header, servers, worker tracking,
    and live-display lifecycle from the base class.
    """

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
        super().__init__(
            facility=facility,
            cost_limit=cost_limit,
            console=console,
            focus=focus,
            title_suffix="Paths Discovery",
        )
        self.state = ProgressState(
            facility=facility,
            cost_limit=cost_limit,
            path_limit=path_limit,
            model=model,
            focus=focus,
            scan_only=scan_only,
            score_only=score_only,
        )

    def _header_mode_label(self) -> str | None:
        """Show SCAN ONLY / SCORE ONLY mode in header."""
        if self.state.scan_only:
            return "SCAN ONLY"
        if self.state.score_only:
            return "SCORE ONLY"
        return None

    def _build_pipeline_section(self) -> Text:
        """Build the unified pipeline section (progress + activity merged).

        Each pipeline stage gets a 3-line block:
          Line 1: SCAN  ━━━━━━━━━━━━━━━━━━    1,234  42%  77.1/s
          Line 2:       /gss/work/imas/codes/chease/src
          Line 3:       45 files, 3 dirs  code project

        Stages: SCAN → SCORE → ENRICH
        """
        content_width = self.width - 6

        # --- Compute progress data ---

        # SCAN: paths that finished scan / total needing scan
        scanned_paths = (
            self.state.pending_score
            + self.state.scored
            + self.state.skipped
            - self.state.excluded
        )
        scan_total = max(self.state.total - self.state.excluded, 1)

        # SCORE: paths through LLM / total needing scoring
        scored_paths = self.state.scored
        score_total = max(self.state.pending_score + self.state.scored, 1)

        # ENRICH: enriched / total needing enrichment (graph-persistent)
        enrich_total = max(self.state.pending_enrich + self.state.enriched, 1)

        # Combined rates
        scan_rate = (
            sum(r for r in [self.state.scan_rate, self.state.expand_rate] if r) or None
        )
        score_rate = (
            sum(r for r in [self.state.score_rate, self.state.refine_rate] if r) or None
        )

        # Score cost for display
        score_cost = (
            self.state._run_score_cost + self.state._run_refine_cost
            if self.state._run_score_cost > 0 or self.state._run_refine_cost > 0
            else None
        )

        # Worker counts per group
        scan_count, scan_ann = self._count_group_workers("scan")
        score_count, score_ann = self._count_group_workers("score")
        enrich_count, enrich_ann = self._count_group_workers("enrich")

        # --- Build activity data ---

        scan = self.state.current_scan
        score = self.state.current_score
        enrich = self.state.current_enrich

        # Worker completion detection
        scan_complete = self._worker_complete("scan") and not scan
        score_complete = self._worker_complete("score") and not score
        enrich_complete = self._worker_complete("enrich") and not enrich

        # SCAN activity
        scan_text = ""
        scan_detail: list[tuple[str, str]] | None = None
        if scan:
            scan_text = clip_path(scan.path, content_width - LABEL_WIDTH)
            parts: list[tuple[str, str]] = [
                (f"{scan.files} files", "cyan"),
                (", ", "dim"),
                (f"{scan.dirs} dirs", "cyan"),
            ]
            if scan.has_code:
                parts.append(("  ", "dim"))
                parts.append(("code project", "green dim"))
            scan_detail = parts

        # SCORE activity
        score_text = ""
        score_detail: list[tuple[str, str]] | None = None
        if score:
            path_display = clip_path(score.path, content_width - LABEL_WIDTH - 14)
            if not score.should_expand:
                path_display += " terminal"
            score_text = path_display

            parts = []
            if score.score is not None:
                from imas_codex.settings import get_discovery_threshold

                _threshold = get_discovery_threshold()
                style = (
                    "bold green"
                    if score.score >= _threshold
                    else "yellow"
                    if score.score >= 0.4
                    else "red"
                )
                parts.append((f"{score.score:.2f}", style))

                desc_width = content_width - 12
                if score.terminal_reason:
                    desc = score.terminal_reason.replace("_", " ")
                elif score.description:
                    desc = clean_text(score.description)
                elif score.purpose:
                    desc = clean_text(score.purpose)
                else:
                    desc = ""
                if desc:
                    parts.append(
                        (
                            f"  {clip_text(desc, desc_width)}",
                            "italic dim",
                        )
                    )
            elif score.skipped:
                parts.append(("skipped", "yellow"))
                if score.skip_reason:
                    reason = clean_text(score.skip_reason)
                    parts.append((f"  {clip_text(reason, content_width - 16)}", "dim"))
            score_detail = parts or None

        # ENRICH activity
        enrich_text = ""
        enrich_detail: list[tuple[str, str]] | None = None
        queue_empty = self.state.enrich_queue.is_empty()
        if enrich and (not queue_empty or self.state.enrich_processing):
            enrich_text = clip_path(enrich.path, content_width - LABEL_WIDTH)
            parts = []
            if enrich.error:
                parts.append((enrich.error, "red"))
            elif enrich.warnings:
                if enrich.total_bytes >= 1_000_000:
                    size_str = f"{enrich.total_bytes / 1_000_000:.1f}MB"
                elif enrich.total_bytes >= 1_000:
                    size_str = f"{enrich.total_bytes / 1_000:.1f}KB"
                else:
                    size_str = f"{enrich.total_bytes}B"
                parts.append((size_str, "cyan"))
                warn_str = ", ".join(enrich.warnings)
                parts.append((f"  [{warn_str}]", "yellow"))
            else:
                if enrich.total_bytes >= 1_000_000:
                    size_str = f"{enrich.total_bytes / 1_000_000:.1f}MB"
                elif enrich.total_bytes >= 1_000:
                    size_str = f"{enrich.total_bytes / 1_000:.1f}KB"
                else:
                    size_str = f"{enrich.total_bytes}B"
                parts.append((size_str, "cyan"))
                if enrich.total_lines > 0:
                    parts.append((f"  {enrich.total_lines:,} LOC", "cyan"))
                if enrich.languages:
                    langs = ", ".join(enrich.languages[:3])
                    parts.append((f"  [{langs}]", "green dim"))
                if enrich.is_multiformat:
                    parts.append(("  multiformat", "yellow"))
                elif enrich.read_matches > 0 or enrich.write_matches > 0:
                    parts.append(
                        (
                            f"  r:{enrich.read_matches} w:{enrich.write_matches}",
                            "dim",
                        )
                    )
            enrich_detail = parts

        # --- Build pipeline rows ---

        rows = [
            PipelineRowConfig(
                name="SCAN",
                style="bold blue",
                completed=scanned_paths,
                total=scan_total,
                rate=scan_rate,
                disabled=self.state.score_only,
                primary_text=scan_text,
                detail_parts=scan_detail,
                is_processing=self.state.scan_processing,
                is_complete=scan_complete,
                worker_count=scan_count,
                worker_annotation=scan_ann,
                queue_size=(
                    len(self.state.scan_queue)
                    if not self.state.scan_queue.is_empty()
                    and not scan
                    and not self.state.scan_processing
                    else 0
                ),
            ),
            PipelineRowConfig(
                name="SCORE",
                style="bold green",
                completed=scored_paths,
                total=score_total,
                rate=score_rate,
                cost=score_cost,
                disabled=self.state.scan_only,
                primary_text=score_text,
                detail_parts=score_detail,
                is_processing=self.state.score_processing,
                is_complete=score_complete,
                worker_count=score_count,
                worker_annotation=score_ann,
                queue_size=(
                    len(self.state.score_queue)
                    if not self.state.score_queue.is_empty()
                    and not score
                    and not self.state.score_processing
                    else 0
                ),
            ),
            PipelineRowConfig(
                name="ENRICH",
                style="bold magenta",
                completed=self.state.enriched,
                total=enrich_total,
                rate=self.state.enrich_rate,
                disabled=self.state.scan_only,
                primary_text=enrich_text,
                detail_parts=enrich_detail,
                is_processing=self.state.enrich_processing,
                is_complete=enrich_complete,
                worker_count=enrich_count,
                worker_annotation=enrich_ann,
            ),
        ]
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges using unified builder."""
        # Compute ETC — accumulated_cost from graph is source of truth
        # (already includes current-session costs after graph refresh)
        total_facility_cost = self.state.accumulated_cost
        etc = total_facility_cost
        cpp = self.state.cost_per_path
        if cpp and cpp > 0:
            pending = self.state.pending_scan + self.state.pending_score
            etc += pending * cpp
        if self.state.run_refined > 0 and self.state._run_refine_cost > 0:
            cost_per_refine = self.state._run_refine_cost / self.state.run_refined
            etc += self.state.pending_refine * cost_per_refine

        # Build stats
        stats: list[tuple[str, str, str]] = [
            ("depth", str(self.state.max_depth), "cyan"),
            ("scored", str(self.state.scored), "green"),
            ("skipped", str(self.state.skipped), "yellow"),
        ]

        # Extra stats appended at end
        extra_stats: list[tuple[str, str, str]] = [
            ("excluded", str(self.state.excluded), "dim"),
        ]

        config = ResourceConfig(
            elapsed=self.state.elapsed,
            eta=None if self.state.scan_only else self.state.eta_seconds,
            run_cost=self.state.run_cost,
            cost_limit=self.state.cost_limit,
            accumulated_cost=self.state.accumulated_cost,
            etc=etc if etc > total_facility_cost else None,
            scan_only=self.state.scan_only,
            limit_reason=self.state.limit_reason,
            stats=stats + extra_stats,
            pending=[
                ("scan", self.state.pending_scan),
                ("expand", self.state.pending_expand),
                ("enrich", self.state.pending_enrich),
                ("refine", self.state.pending_refine),
            ],
        )
        return build_resource_section(config, self.gauge_width)

    # ========================================================================
    # Public API
    # ========================================================================

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
                        warnings=r.get("warnings", []),
                    )
                )
            # Use actual worker rate, capped for readability
            # Max 2.0/s = each item visible for at least 0.5s
            max_display_rate = 2.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 1.0
            self.state.enrich_queue.add(items, display_rate)

        self._refresh()

    def update_refine(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update refine worker state.

        Refine results are added to the score queue since they update scores.
        """
        self.state.run_refined = stats.processed
        self.state.refine_rate = stats.rate
        # Track refine cost separately (cumulative from refine worker)
        self.state._run_refine_cost = stats.cost

        # Queue refine results to score stream
        if results:
            items = []
            for r in results:
                path = r.get("path", "")
                # Use adjustment_reason for display, falling back to "refined"
                reason = r.get("adjustment_reason", "refined")
                items.append(
                    ScoreItem(
                        path=path,
                        score=r.get("score"),
                        purpose="refined",
                        description=reason if reason != "refined" else "",
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
        """Drain streaming queues for smooth display.

        Clears stale current items when queues have drained and no new
        items have been added for the stale timeout.
        """
        updated = False

        next_scan = self.state.scan_queue.pop()
        if next_scan:
            self.state.current_scan = next_scan
            updated = True
        elif self.state.scan_queue.is_stale() and self.state.current_scan is not None:
            self.state.current_scan = None
            updated = True

        next_score = self.state.score_queue.pop()
        if next_score:
            self.state.current_score = next_score
            updated = True
        elif self.state.score_queue.is_stale() and self.state.current_score is not None:
            self.state.current_score = None
            updated = True

        next_enrich = self.state.enrich_queue.pop()
        if next_enrich:
            self.state.current_enrich = next_enrich
            updated = True
        elif (
            self.state.enrich_queue.is_stale() and self.state.current_enrich is not None
        ):
            self.state.current_enrich = None
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
        self.state.enriched = stats.get("enriched", 0)
        self.state.refined = stats.get("refined", 0)


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

            from imas_codex.settings import get_discovery_threshold

            _threshold = get_discovery_threshold()
            high_value = get_high_value_paths(facility, min_score=_threshold, limit=10)
            if high_value:
                output(f"High-value paths (score > {_threshold}): {len(high_value)}")
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
            from imas_codex.discovery.signals import get_data_discovery_stats

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
