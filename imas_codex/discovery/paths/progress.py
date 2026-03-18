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
    BaseProgressDisplay,
    PipelineRowConfig,
    ResourceConfig,
    StreamQueue,
    build_pipeline_section,
    build_resource_section,
    clean_text,
    compute_projected_etc,
    format_count,
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
class TriageItem:
    """Current triage activity."""

    path: str
    score_composite: float | None = None
    purpose: str = ""  # Category: experimental_data, modeling_code, etc.
    description: str = ""  # LLM reasoning about why this path is valuable
    physics_domain: str = ""  # Primary physics domain (equilibrium, transport, etc.)
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


@dataclass
class ScoreItem:
    """Current score activity (2nd pass with enrichment evidence)."""

    path: str
    score_composite: float | None = None
    previous_score: float | None = None
    purpose: str = ""
    description: str = ""
    physics_domain: str = ""  # Primary physics domain
    adjustment_reason: str = ""
    should_expand: bool = True  # False = terminal (won't explore children)
    terminal_reason: str = ""  # TerminalReason enum value when terminal
    dimension_scores: dict[str, float] = field(
        default_factory=dict
    )  # Per-dimension scores for display


@dataclass
class DedupItem:
    """Current dedup activity — one batch of clone marking."""

    repo_name: str  # SoftwareRepo name (e.g. "MEQ")
    canonical_path: str  # Path retained as canonical
    clones_marked: int  # How many clones were marked terminal this batch
    total_clones: int  # Total clones including canonical


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
    triage_only: bool = False

    # Graph totals (aligned with new state machine)
    total: int = 0
    discovered: int = 0  # Awaiting scan
    scanned: int = 0  # Awaiting triage
    triaged: int = 0  # Triaged (1st pass)
    scored: int = 0  # Scored complete (2nd pass)
    skipped: int = 0  # Low value or dead-end
    explored: int = 0  # Root/navigation paths (scanned but not triaged)
    excluded: int = 0  # Matched exclusion pattern
    max_depth: int = 0  # Maximum tree depth

    # Pending work counts (for progress bars)
    pending_scan: int = 0  # discovered + scanning
    pending_triage: int = 0  # scanned + triaging
    pending_expand: int = 0  # triaged + should_expand + not expanded
    pending_enrich: int = 0  # triaged + should_enrich + not enriched
    pending_score: int = 0  # enriched + not scored

    # Graph-persistent totals (from get_discovery_stats)
    enriched: int = 0  # Total enriched paths in graph

    # This run stats
    run_scanned: int = 0
    run_triaged: int = 0
    run_expanded: int = 0
    run_enriched: int = 0
    run_scored: int = 0
    # Track triage and score costs separately to avoid double-counting
    _run_triage_cost: float = 0.0
    _run_score_cost: float = 0.0
    scan_rate: float | None = None
    triage_rate: float | None = None
    expand_rate: float | None = None
    enrich_rate: float | None = None
    score_rate: float | None = None

    # Accumulated facility cost and time (from graph)
    accumulated_cost: float = 0.0
    accumulated_time: float = 0.0

    # Provider budget exhaustion (API key credit limit hit)
    provider_budget_exhausted: bool = False

    # Current items (and their processing state)
    current_scan: ScanItem | None = None
    current_triage: TriageItem | None = None
    current_enrich: EnrichItem | None = None
    # Processing counters — incremented when a worker starts a batch,
    # decremented when results arrive. Using counters (not bools) because
    # multiple workers of the same type run in parallel, and a single bool
    # would toggle incorrectly when one finishes while another is still active.
    scan_processing: int = 0
    triage_processing: int = 0
    enrich_processing: int = 0

    # Streaming queues - enrich adapts rate based on batch size to fill inter-batch gaps
    scan_queue: StreamQueue = field(default_factory=StreamQueue)
    triage_queue: StreamQueue = field(default_factory=StreamQueue)
    enrich_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.5, min_display_time=0.4, stale_timeout=15.0
        )
    )
    score_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.0, min_display_time=0.5, stale_timeout=15.0
        )
    )

    # Current score item (for streaming display)
    current_score: ScoreItem | None = None
    score_processing: int = 0

    # Dedup tracking
    run_deduped: int = 0  # Clone paths marked terminal this session
    dedup_rate: float | None = None
    current_dedup: DedupItem | None = None
    dedup_processing: int = 0

    # Tracking
    triaged_paths: set[str] = field(default_factory=set)
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
        """Total pending work: scan + triage + expand + enrich queues."""
        return (
            self.pending_scan
            + self.pending_triage
            + self.pending_expand
            + self.pending_enrich
        )

    @property
    def cost_per_path(self) -> float | None:
        """Average LLM cost per triaged path (excludes score cost)."""
        if self.run_triaged > 0:
            return self._run_triage_cost / self.run_triaged
        return None

    @property
    def estimated_total_cost(self) -> float | None:
        """Estimated total cost based on current rate.

        Predicts cost for all remaining LLM work:
        - pending_triage paths at cost_per_path rate (initial triage)
        - pending_score paths at score cost rate
        """
        cpp = self.cost_per_path
        if cpp is None:
            return None

        remaining_triage_cost = (self.pending_scan + self.pending_triage) * cpp

        # Score cost rate (separate from triage)
        remaining_score_cost = 0.0
        if self.run_scored > 0 and self._run_score_cost > 0:
            cost_per_score = self._run_score_cost / self.run_scored
            remaining_score_cost = self.pending_score * cost_per_score

        return self.run_cost + remaining_triage_cost + remaining_score_cost

    @property
    def run_cost(self) -> float:
        """Total cost for this run (triage + score)."""
        return self._run_triage_cost + self._run_score_cost

    @property
    def coverage(self) -> float:
        """Percentage of total paths scored."""
        return (self.scored / self.total * 100) if self.total > 0 else 0

    @property
    def frontier_size(self) -> int:
        """Total paths awaiting work (scan or triage)."""
        return self.pending_scan + self.pending_triage + self.pending_expand

    @property
    def cost_limit_reached(self) -> bool:
        """Check if cost limit has been reached."""
        return self.run_cost >= self.cost_limit if self.cost_limit > 0 else False

    @property
    def path_limit_reached(self) -> bool:
        """Check if path limit has been reached.

        Uses triaged + scored count (terminal states) since both
        represent paths that have completed their pipeline stage.
        """
        if self.path_limit is None or self.path_limit <= 0:
            return False
        return (self.run_triaged + self.run_scored) >= self.path_limit

    @property
    def limit_reason(self) -> str | None:
        """Return which limit was reached, or None if no limit reached."""
        if self.provider_budget_exhausted:
            return "provider budget"
        if self.cost_limit_reached:
            return "cost"
        if self.path_limit_reached:
            return "path"
        return None

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time to complete all pending work.

        Pure work-based ETA: max across parallel worker groups
        (bounded by the slowest pipeline).  Cost and path limits
        are stop conditions, not ETA inputs.

        Uses session-average rates (aggregate across concurrent workers)
        to avoid per-worker EMA undercount.
        """
        from imas_codex.discovery.base.progress import compute_parallel_eta

        def _agg(count: int, fallback: float | None) -> float | None:
            if count > 0 and self.elapsed > 5:
                return count / self.elapsed
            return fallback

        return compute_parallel_eta(
            [
                (self.pending_scan, _agg(self.run_scanned, self.scan_rate)),
                (self.pending_triage, _agg(self.run_triaged, self.triage_rate)),
                (self.pending_expand, _agg(self.run_expanded, self.expand_rate)),
                (self.pending_enrich, _agg(self.run_enriched, self.enrich_rate)),
                (self.pending_score, _agg(self.run_scored, self.score_rate)),
            ]
        )


# ============================================================================
# Main Display Class
# ============================================================================


class ParallelProgressDisplay(BaseProgressDisplay):
    """Clean progress display for parallel discovery.

    Extends ``BaseProgressDisplay`` for the paths discovery pipeline
    (SCAN → TRIAGE → ENRICH → SCORE).  Inherits header, servers, worker tracking,
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
        triage_only: bool = False,
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
            triage_only=triage_only,
        )

    def _header_mode_label(self) -> str | None:
        """Show SCAN ONLY / TRIAGE ONLY mode in header."""
        if self.state.scan_only:
            return "SCAN ONLY"
        if self.state.triage_only:
            return "TRIAGE ONLY"
        return None

    def _build_pipeline_section(self) -> Text:
        """Build the unified pipeline section (progress + activity merged).

        Each pipeline stage gets a 3-line block:
          Line 1: SCAN  ━━━━━━━━━━━━━━━━━━    1,234  42%  77.1/s
          Line 2:       /gss/work/imas/codes/chease/src
          Line 3:       45 files, 3 dirs  code project

        Stages: SCAN → TRIAGE → ENRICH → SCORE
        """

        # --- Compute progress data ---

        # SCAN: paths that finished scan / total needing scan
        scanned_paths = (
            self.state.pending_triage
            + self.state.triaged
            + self.state.scored
            + self.state.skipped
            + self.state.explored
            - self.state.excluded
        )
        scan_total = max(self.state.total - self.state.excluded, 1)

        # TRIAGE: paths through 1st LLM pass / total needing triage
        triaged_paths = self.state.triaged + self.state.scored
        triage_total = max(self.state.pending_triage + triaged_paths, 1)

        # ENRICH: enriched / total needing enrichment (graph-persistent)
        enrich_total = max(self.state.pending_enrich + self.state.enriched, 1)

        # Combined rates
        scan_rate = (
            sum(r for r in [self.state.scan_rate, self.state.expand_rate] if r) or None
        )
        triage_rate = self.state.triage_rate

        # Triage cost for display
        triage_cost = (
            self.state._run_triage_cost if self.state._run_triage_cost > 0 else None
        )

        # Worker counts per group
        scan_count, scan_ann = self._count_group_workers("scan")
        expand_count, _ = self._count_group_workers("expand")
        triage_count, triage_ann = self._count_group_workers("triage")
        enrich_count, enrich_ann = self._count_group_workers("enrich")

        # --- Build activity data ---

        scan = self.state.current_scan
        triage = self.state.current_triage
        enrich = self.state.current_enrich

        # Worker completion detection — scan row is complete when both
        # scan and expand workers are done.
        scan_complete = (
            self._worker_complete("scan")
            and self._worker_complete("expand")
            and not scan
        )
        triage_complete = self._worker_complete("triage") and not triage
        enrich_complete = self._worker_complete("enrich") and not enrich

        # SCAN activity
        scan_text = ""
        scan_desc = ""
        # When all known paths are scanned (100%) but workers are still
        # alive waiting for expand dirs, let the queue drain naturally
        # before switching to "waiting" status.  Only suppress content
        # once the queue is empty so the last batch fully unwinds.
        scan_at_capacity = scanned_paths >= scan_total and not scan_complete
        scan_queue_drained = self.state.scan_queue.is_empty()
        if scan and not (scan_at_capacity and scan_queue_drained):
            scan_text = scan.path
            scan_parts = [f"{scan.files} files, {scan.dirs} dirs"]
            if scan.has_code:
                scan_parts.append("code project")
            scan_desc = "  ".join(scan_parts)

        # TRIAGE activity — uses structured fields for 3-line layout:
        #   Line 2: score  physics_domain  /path/clipped...        rate
        #   Line 3: LLM description spanning full width...          $cost
        triage_path = ""
        triage_value: float | None = None
        triage_domain = ""
        triage_desc = ""
        triage_desc_fallback = ""
        triage_terminal = ""
        if triage:
            triage_path = triage.path

            if triage.score_composite is not None:
                triage_value = triage.score_composite

                # Physics domain (shown on line 2)
                if triage.physics_domain and triage.physics_domain != "general":
                    triage_domain = triage.physics_domain.replace("_", " ")

                # Terminal flag (just "terminal" in muted red)
                if not triage.should_expand:
                    triage_terminal = "terminal"

                # Description for line 3
                if triage.description:
                    triage_desc = clean_text(triage.description)
                elif triage.purpose:
                    triage_desc_fallback = clean_text(triage.purpose)
            elif triage.skipped:
                triage_terminal = "skipped"
                if triage.skip_reason:
                    triage_desc = clean_text(triage.skip_reason)

        # ENRICH activity
        enrich_text = ""
        enrich_desc = ""
        queue_empty = self.state.enrich_queue.is_empty()
        if enrich and (not queue_empty or self.state.enrich_processing > 0):
            enrich_text = enrich.path
            if enrich.error:
                enrich_desc = f"error: {enrich.error}"
            elif enrich.warnings:
                if enrich.total_bytes >= 1_000_000:
                    size_str = f"{enrich.total_bytes / 1_000_000:.1f}MB"
                elif enrich.total_bytes >= 1_000:
                    size_str = f"{enrich.total_bytes / 1_000:.1f}KB"
                else:
                    size_str = f"{enrich.total_bytes}B"
                enrich_desc = f"{size_str}  [{', '.join(enrich.warnings)}]"
            else:
                if enrich.total_bytes >= 1_000_000:
                    size_str = f"{enrich.total_bytes / 1_000_000:.1f}MB"
                elif enrich.total_bytes >= 1_000:
                    size_str = f"{enrich.total_bytes / 1_000:.1f}KB"
                else:
                    size_str = f"{enrich.total_bytes}B"
                desc_parts = [size_str]
                if enrich.total_lines > 0:
                    desc_parts.append(f"{format_count(enrich.total_lines)} LOC")
                if enrich.languages:
                    desc_parts.append(f"[{', '.join(enrich.languages[:3])}]")
                if enrich.pattern_categories:
                    cat_strs = [
                        f"{cat}:{count}"
                        for cat, count in sorted(
                            enrich.pattern_categories.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:4]
                    ]
                    desc_parts.append(" ".join(cat_strs))
                elif enrich.is_multiformat:
                    desc_parts.append("multiformat")
                elif enrich.read_matches > 0 or enrich.write_matches > 0:
                    desc_parts.append(
                        f"r:{enrich.read_matches} w:{enrich.write_matches}"
                    )
                enrich_desc = "  ".join(desc_parts)

        # SCORE activity (2nd pass with enrichment evidence)
        score_count, score_ann = self._count_group_workers("score")
        dedup_count, _ = self._count_group_workers("dedup")
        embed_count, _ = self._count_group_workers("embed")
        score = self.state.current_score
        score_complete = self._worker_complete("score") and not score

        score_text = ""
        score_score_parts: list[tuple[str, str]] | None = None
        score_domain = ""
        score_desc = ""
        score_terminal = ""
        score_queue_empty = self.state.score_queue.is_empty()
        if score and (not score_queue_empty or self.state.score_processing > 0):
            score_text = score.path
            if score.score_composite is not None:
                # Show top dimension score + label instead of combined score
                parts: list[tuple[str, str]] = []

                # Find top dimension for display
                top_dim = ""
                top_val = 0.0
                if score.dimension_scores:
                    for dim, val in score.dimension_scores.items():
                        if val > top_val:
                            top_val = val
                            top_dim = dim

                display_score = top_val if top_dim else score.score_composite
                dim_label = (
                    top_dim.replace("score_", "").replace("_", " ") if top_dim else ""
                )

                from imas_codex.settings import get_discovery_threshold

                _threshold = get_discovery_threshold()
                style = (
                    "bold green"
                    if display_score >= _threshold
                    else "yellow"
                    if display_score >= 0.4
                    else "red"
                )
                parts.append((f"{display_score:.2f}", style))
                if dim_label:
                    parts.append((" ", "dim"))
                    parts.append((dim_label, "cyan italic"))
                score_score_parts = parts or None

                # Physics domain
                if score.physics_domain and score.physics_domain != "general":
                    score_domain = score.physics_domain.replace("_", " ")

                # Terminal flag (just "terminal" in muted red)
                if not score.should_expand and score.terminal_reason:
                    score_terminal = "terminal"

                # Description for line 3 — prefer actual description over scoring reason
                if score.description:
                    score_desc = clean_text(score.description)
                elif score.adjustment_reason:
                    score_desc = clean_text(score.adjustment_reason)

        # SCORE totals
        score_total = max(self.state.pending_score + self.state.scored, 1)

        # Score cost for display
        score_cost = (
            self.state._run_score_cost if self.state._run_score_cost > 0 else None
        )

        # DEDUP runs in background — not shown in pipeline display

        # --- Build pipeline rows ---

        rows = [
            PipelineRowConfig(
                name="SCAN",
                style="bold blue",
                completed=scanned_paths,
                total=scan_total,
                rate=scan_rate,
                disabled=self.state.triage_only,
                primary_text=scan_text,
                description=scan_desc,
                is_processing=self.state.scan_processing > 0
                or (scan_at_capacity and scan_queue_drained),
                processing_label=(
                    "waiting for expand..."
                    if scan_at_capacity
                    and scan_queue_drained
                    and self.state.scan_processing == 0
                    else "processing..."
                ),
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
                name="TRIAGE",
                style="bold green",
                completed=triaged_paths,
                total=triage_total,
                rate=triage_rate,
                cost=triage_cost,
                disabled=self.state.scan_only,
                primary_text=triage_path,
                score_value=triage_value,
                physics_domain=triage_domain,
                terminal_label=triage_terminal,
                description=triage_desc,
                description_fallback=triage_desc_fallback,
                is_processing=self.state.triage_processing > 0,
                is_complete=triage_complete,
                worker_count=triage_count,
                worker_annotation=triage_ann,
                queue_size=(
                    len(self.state.triage_queue)
                    if not self.state.triage_queue.is_empty()
                    and not triage
                    and self.state.triage_processing == 0
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
                description=enrich_desc,
                is_processing=self.state.enrich_processing > 0,
                is_complete=enrich_complete,
                worker_count=enrich_count,
                worker_annotation=enrich_ann,
            ),
            PipelineRowConfig(
                name="SCORE",
                style="bold cyan",
                completed=self.state.scored,
                total=score_total,
                rate=self.state.score_rate,
                cost=score_cost,
                disabled=self.state.scan_only,
                primary_text=score_text,
                score_parts=score_score_parts,
                physics_domain=score_domain,
                terminal_label=score_terminal,
                description=score_desc,
                is_processing=self.state.score_processing > 0,
                is_complete=score_complete,
                worker_count=score_count,
                worker_annotation=score_ann,
            ),
        ]
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges using unified builder."""
        # Compute ETC — sum of per-worker cost projections
        total_facility_cost = self.state.accumulated_cost
        cost_per_score = (
            self.state._run_score_cost / self.state.run_scored
            if self.state.run_scored > 0 and self.state._run_score_cost > 0
            else None
        )
        etc = compute_projected_etc(
            total_facility_cost,
            [
                (
                    self.state.pending_scan + self.state.pending_triage,
                    self.state.cost_per_path,
                ),
                (self.state.pending_score, cost_per_score),
            ],
        )

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
            accumulated_time=self.state.accumulated_time,
            run_cost=self.state.run_cost,
            cost_limit=self.state.cost_limit,
            accumulated_cost=self.state.accumulated_cost,
            etc=etc,
            scan_only=self.state.scan_only,
            limit_reason=self.state.limit_reason,
            stats=stats + extra_stats,
            pending=[
                ("scan", self.state.pending_scan),
                ("expand", self.state.pending_expand),
                ("enrich", self.state.pending_enrich),
                ("score", self.state.pending_score),
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
        self.state.scan_rate = stats.ema_rate or stats.active_rate

        # Track processing state for display
        # Don't clear current_scan when idle - let queue drain naturally
        # via tick(). Only update the processing counter.
        if message == "idle":
            self.state.scan_processing = max(0, self.state.scan_processing - 1)
            self._refresh()
            return
        elif "scanning" in message.lower():
            # About to run SSH scan - mark as processing
            self.state.scan_processing += 1
        else:
            # Got results back ("scanned N paths")
            self.state.scan_processing = max(0, self.state.scan_processing - 1)

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
            # Adaptive rate: use last_batch_time to drain queue
            # just before next batch arrives. EMA as fallback.
            ema = stats.ema_rate or stats.active_rate
            self.state.scan_queue.add(
                items,
                ema,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    def update_triage(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update triage worker state."""
        self.state.run_triaged = stats.processed
        self.state.triage_rate = stats.ema_rate or stats.active_rate
        self.state._run_triage_cost = stats.cost

        if "provider budget" in message.lower():
            self.state.provider_budget_exhausted = True
            self._refresh()
            return

        # Track processing state for display
        # Don't clear current_triage when waiting - let queue drain naturally
        # via tick(). Only update the processing counter.
        if "waiting" in message.lower():
            self.state.triage_processing = max(0, self.state.triage_processing - 1)
            self._refresh()
            return
        elif "triaging" in message.lower():
            # About to call LLM - mark as processing
            self.state.triage_processing += 1
        else:
            # Got results back
            self.state.triage_processing = max(0, self.state.triage_processing - 1)

        # Queue triage results for streaming (tick() handles rate-limited popping)
        if results:
            items = []
            for r in results:
                path = r.get("path", "")
                self.state.triaged_paths.add(path)
                items.append(
                    TriageItem(
                        path=path,
                        score_composite=r.get("score"),
                        purpose=r.get("label", "") or r.get("path_purpose", ""),
                        description=r.get("description", ""),
                        physics_domain=r.get("physics_domain", ""),
                        skipped=bool(r.get("skip_reason")),
                        skip_reason=r.get("skip_reason", ""),
                        should_expand=r.get("should_expand", True),
                        terminal_reason=r.get("terminal_reason", ""),
                    )
                )
            # Adaptive rate: use last_batch_time to drain queue
            # just before next batch arrives. EMA as fallback.
            ema = stats.ema_rate or stats.active_rate
            self.state.triage_queue.add(
                items,
                ema,
                last_batch_time=stats.last_batch_time,
            )

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
        self.state.expand_rate = stats.ema_rate or stats.active_rate

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
            ema = stats.ema_rate or stats.active_rate
            self.state.scan_queue.add(
                items,
                ema,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    def update_enrich(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update enrich worker state with enrichment results."""
        self.state.run_enriched = stats.processed
        self.state.enrich_rate = stats.ema_rate or stats.active_rate

        # Track processing state for display
        # Don't clear current_enrich when waiting - let queue drain naturally
        # via tick(). Only update the processing counter.
        if "waiting" in message.lower():
            self.state.enrich_processing = max(0, self.state.enrich_processing - 1)
            self._refresh()
            return
        elif "enriching" in message.lower():
            # About to run SSH enrichment - mark as processing
            self.state.enrich_processing += 1
        else:
            # Got results back
            self.state.enrich_processing = max(0, self.state.enrich_processing - 1)

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
            # Adaptive rate: use last_batch_time to drain queue
            # just before next batch arrives. EMA as fallback.
            ema = stats.ema_rate or stats.active_rate
            self.state.enrich_queue.add(
                items,
                ema,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    def update_score(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update score worker state with own display row."""
        self.state.run_scored = stats.processed
        self.state.score_rate = stats.ema_rate or stats.active_rate
        # Track score cost separately (cumulative from score worker)
        self.state._run_score_cost = stats.cost

        if "provider budget" in message.lower():
            self.state.provider_budget_exhausted = True
            self._refresh()
            return

        # Track processing state for display
        if "waiting" in message.lower():
            self.state.score_processing = max(0, self.state.score_processing - 1)
            self._refresh()
            return
        elif "scoring" in message.lower():
            self.state.score_processing += 1
        else:
            self.state.score_processing = max(0, self.state.score_processing - 1)

        # Queue score results to score stream (own display row)
        if results:
            items = []
            for r in results:
                path = r.get("path", "")
                reason = r.get("adjustment_reason", "")
                # Collect per-dimension scores for display
                dim_scores = {
                    k: v
                    for k, v in r.items()
                    if k.startswith("score_") and isinstance(v, int | float)
                }
                items.append(
                    ScoreItem(
                        path=path,
                        score_composite=r.get("score"),
                        previous_score=r.get("previous_score"),
                        purpose=r.get("path_purpose", ""),
                        description=r.get("description", ""),
                        physics_domain=r.get("physics_domain", "") or "",
                        adjustment_reason=reason,
                        should_expand=r.get("should_expand", True),
                        terminal_reason=(
                            f"data:{r.get('path_purpose', '')}"
                            if not r.get("should_expand", True)
                            else ""
                        ),
                        dimension_scores=dim_scores,
                    )
                )
            # Adaptive rate: use last_batch_time to drain queue
            # just before next batch arrives. EMA as fallback.
            ema = stats.ema_rate or stats.active_rate
            self.state.score_queue.add(
                items,
                ema,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    def update_dedup(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update dedup worker state."""
        self.state.run_deduped = stats.processed
        self.state.dedup_rate = stats.ema_rate or stats.active_rate

        if "waiting" in message.lower():
            self.state.dedup_processing = max(0, self.state.dedup_processing - 1)
            self.state.current_dedup = None
            self._refresh()
            return
        elif "deduped" in message.lower():
            self.state.dedup_processing = max(0, self.state.dedup_processing - 1)
        else:
            self.state.dedup_processing += 1

        if results:
            # Show the most recent dedup event
            last = results[-1]
            self.state.current_dedup = DedupItem(
                repo_name=last.get("repo_name", "unknown"),
                canonical_path=last.get("canonical_path", ""),
                clones_marked=last.get("clones_marked", 0),
                total_clones=last.get("total_clones", 0),
            )

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
        self.state.triaged = stats["triaged"]
        self.state.scored = stats["scored"]
        self.state.skipped = stats["skipped"]
        self.state.explored = stats.get("explored", 0)
        self.state.excluded = stats["excluded"]
        self.state.max_depth = stats["max_depth"]
        # Calculate pending work counts including expansion_ready
        self._calculate_pending_from_stats(stats)

        # Get accumulated facility cost from graph
        cost_data = get_accumulated_cost(facility)
        self.state.accumulated_cost = cost_data["total_cost"]

        # Accumulated wall-clock time from prior sessions
        from imas_codex.discovery.base.progress import get_accumulated_time

        self.state.accumulated_time = get_accumulated_time(facility, "paths")

        self._refresh()

    def tick(self) -> None:
        """Drain streaming queues for smooth display.

        Only clears a stale current item when the worker is also idle
        (not processing a batch).  This prevents flicker between content
        and "idle" when the queue empties between batches while the
        worker is still actively claiming and processing work.
        """
        updated = False

        next_scan = self.state.scan_queue.pop()
        if next_scan:
            self.state.current_scan = next_scan
            updated = True
        elif (
            self.state.scan_queue.is_stale()
            and self.state.current_scan is not None
            and self.state.scan_processing == 0
        ):
            self.state.current_scan = None
            updated = True

        next_triage = self.state.triage_queue.pop()
        if next_triage:
            self.state.current_triage = next_triage
            updated = True
        elif (
            self.state.triage_queue.is_stale()
            and self.state.current_triage is not None
            and self.state.triage_processing == 0
        ):
            self.state.current_triage = None
            updated = True

        next_enrich = self.state.enrich_queue.pop()
        if next_enrich:
            self.state.current_enrich = next_enrich
            updated = True
        elif (
            self.state.enrich_queue.is_stale()
            and self.state.current_enrich is not None
            and self.state.enrich_processing == 0
        ):
            self.state.current_enrich = None
            updated = True

        next_score = self.state.score_queue.pop()
        if next_score:
            self.state.current_score = next_score
            updated = True
        elif (
            self.state.score_queue.is_stale()
            and self.state.current_score is not None
            and self.state.score_processing == 0
        ):
            self.state.current_score = None
            updated = True

        if updated:
            self._refresh()

    def get_paths_triaged_this_run(self) -> set[str]:
        """Get paths triaged during this run."""
        return self.state.triaged_paths

    def get_paths_scored_this_run(self) -> set[str]:
        """Get paths scored (triaged + scored) during this run."""
        return self.state.triaged_paths

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
        triaging = stats.get("triaging", 0)
        self.state.pending_scan = stats.get("discovered", 0) + scanning
        self.state.pending_triage = stats.get("scanned", 0) + triaging
        self.state.pending_expand = stats.get("expansion_ready", 0)
        self.state.pending_enrich = stats.get("enrichment_ready", 0)
        self.state.pending_score = stats.get("score_ready", 0)
        self.state.enriched = stats.get("enriched", 0)


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
            triaged = stats.get("triaged", 0)
            scored = stats.get("scored", 0)
            skipped = stats.get("skipped", 0)
            excluded = stats.get("excluded", 0)
            max_depth = stats.get("max_depth", 0)
            enriched = stats.get("enriched", 0)

            # Cumulative throughput: each stage includes all downstream
            cum_scanned = scanned + triaged + scored + skipped
            cum_triaged = triaged + scored + skipped

            output(f"├─ Scanned:    {cum_scanned:,} ({cum_scanned / total * 100:.1f}%)")
            output(f"│  ├─ Triaged: {cum_triaged:,} ({cum_triaged / total * 100:.1f}%)")
            if enriched > 0:
                output(f"│  │  ├─ Enriched: {enriched:,}")
            if scored > 0:
                output(f"│  │  └─ Scored:   {scored:,}")
            output(f"│  └─ Skipped: {skipped:,} ({skipped / total * 100:.1f}%)")
            if discovered > 0:
                output(
                    f"├─ Pending:    {discovered:,} ({discovered / total * 100:.1f}%)"
                )
            output(f"└─ Excluded:   {excluded:,} ({excluded / total * 100:.1f}%)")

            # Purpose distribution with top paths per category
            purpose_dist = get_purpose_distribution(facility)
            if purpose_dist:
                output("\n[bold]By Purpose (top 3 per category):[/bold]")

                categories = [
                    ("Modeling Code", "cyan", ["modeling_code"]),
                    ("Analysis Code", "green", ["analysis_code"]),
                    ("Operations Code", "green", ["operations_code"]),
                    (
                        "Data",
                        "yellow",
                        ["modeling_data", "experimental_data"],
                    ),
                    ("Data Access", "blue", ["data_access"]),
                    ("Workflow", "blue", ["workflow"]),
                    ("Visualization", "blue", ["visualization"]),
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
            coverage = (triaged + scored) / total * 100 if total > 0 else 0
            output(f"Coverage: {coverage:.1f}% triaged")

            from imas_codex.settings import get_discovery_threshold

            _threshold = get_discovery_threshold()
            high_value = get_high_value_paths(facility, min_score=_threshold, limit=10)
            if high_value:
                output(f"High-value paths (score > {_threshold}): {len(high_value)}")
                for p in high_value[:5]:
                    # Find top dimension for display
                    top_dim = ""
                    top_val = 0.0
                    for dim_key in [
                        "score_modeling_code",
                        "score_analysis_code",
                        "score_operations_code",
                        "score_data_access",
                        "score_workflow",
                        "score_visualization",
                        "score_documentation",
                        "score_imas",
                    ]:
                        val = p.get(dim_key, 0) or 0
                        if val > top_val:
                            top_val = val
                            top_dim = dim_key
                    dim_label = (
                        top_dim.replace("score_", "").replace("_", " ")
                        if top_dim
                        else ""
                    )
                    score_str = f"{top_val:.2f}" if top_dim else f"{p['score']:.2f}"
                    dim_suffix = f" [dim]{dim_label}[/dim]" if dim_label else ""
                    output(f"  [{score_str}]{dim_suffix} {p['path']}")
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
                # Cumulative throughput: each stage includes all downstream
                cum_scanned = wiki_scanned + wiki_scored + wiki_ingested + wiki_skipped
                cum_scored = wiki_scored + wiki_ingested + wiki_skipped
                output(f"├─ Scanned:   {cum_scanned:,}")
                output(f"│  ├─ Scored:  {cum_scored:,}")
                output(f"│  │  ├─ Ingested: {wiki_ingested:,}")
                output(f"│  │  └─ Skipped:  {wiki_skipped:,}")
                if wiki_scored > 0:
                    output(f"│  └─ Awaiting: {wiki_scored:,}")

                # Document stats
                total_documents = wiki_stats.get("total_documents", 0)
                if total_documents > 0:
                    doc_scored = wiki_stats.get("documents_scored", 0)
                    doc_ingested = wiki_stats.get("docs_ingested", 0)
                    doc_pending_score = wiki_stats.get("pending_document_score", 0)
                    doc_pending_ingest = wiki_stats.get("pending_document_ingest", 0)
                    # Cumulative: scored includes downstream ingested
                    cum_doc_scored = doc_scored + doc_ingested
                    output(f"\nDocuments: {total_documents:,}")
                    output(f"├─ Scored:    {cum_doc_scored:,}")
                    output(f"│  └─ Ingested: {doc_ingested:,}")
                    if doc_pending_score > 0 or doc_pending_ingest > 0:
                        output(
                            f"└─ Pending:   "
                            f"score={doc_pending_score:,}, "
                            f"ingest={doc_pending_ingest:,}"
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
                discovered = signal_stats.get("discovered", 0)
                enriched = signal_stats.get("enriched", 0)
                checked = signal_stats.get("checked", 0)
                skipped = signal_stats.get("skipped", 0)
                cost = signal_stats.get("accumulated_cost", 0.0)

                # Cumulative throughput: each stage includes downstream
                cum_enriched = enriched + checked + skipped

                output(f"Total signals: {signal_total:,}")
                output(f"├─ Enriched:  {cum_enriched:,}")
                output(f"│  ├─ Checked: {checked:,}")
                output(f"│  └─ Skipped: {skipped:,}")
                if discovered > 0:
                    output(f"└─ Pending:   {discovered:,}")
                if cost > 0:
                    output(f"Accumulated cost: ${cost:.2f}")
            elif domain == "signals":
                output("No signals discovered")
        except Exception:
            if domain == "signals":
                output("Signal stats unavailable")

    # --------------------------------------------------------------------------
    # Static domain
    # --------------------------------------------------------------------------
    if domain is None or domain == "static":
        try:
            from imas_codex.discovery.mdsplus.graph_ops import (
                get_static_summary_stats,
            )

            static_stats = get_static_summary_stats(facility)
            versions_total = static_stats.get("versions_total", 0)
            if versions_total > 0:
                if domain is None:
                    output("\n[bold]Static Tree Discovery:[/bold]")
                versions_discovered = static_stats.get("versions_discovered", 0)
                versions_ingested = static_stats.get("versions_ingested", 0)
                nodes_graph = static_stats.get("nodes_graph", 0)
                nodes_enriched = static_stats.get("nodes_enriched", 0)

                # Cumulative: discovered includes downstream ingested
                cum_discovered = versions_discovered + versions_ingested
                output(f"Versions: {versions_total:,}")
                output(
                    f"├─ Discovered: {cum_discovered:,}"
                    f" ({cum_discovered / versions_total * 100:.0f}%)"
                )
                output(
                    f"└─ Ingested:   {versions_ingested:,}"
                    f" ({versions_ingested / versions_total * 100:.0f}%)"
                )
                if nodes_graph > 0:
                    output(f"Nodes: {nodes_graph:,}")
                    enriched_pct = (
                        nodes_enriched / nodes_graph * 100 if nodes_graph else 0
                    )
                    output(f"└─ Enriched:   {nodes_enriched:,} ({enriched_pct:.0f}%)")
            elif domain == "static":
                output("No static trees discovered")
        except Exception:
            if domain == "static":
                output("Static stats unavailable")

    # --------------------------------------------------------------------------
    # Code domain
    # --------------------------------------------------------------------------
    if domain is None or domain == "code":
        try:
            from imas_codex.discovery.code.parallel import get_code_discovery_stats

            code_stats = get_code_discovery_stats(facility)
            code_total = code_stats.get("total", 0)
            if code_total > 0:
                if domain is None:
                    output("\n[bold]Code Discovery:[/bold]")
                discovered = int(code_stats.get("discovered", 0))
                triaged = int(code_stats.get("triaged", 0))
                scored = int(code_stats.get("scored", 0))
                ingested = int(code_stats.get("ingested", 0))
                failed = int(code_stats.get("failed", 0))
                skipped = int(code_stats.get("skipped", 0))
                enriched = int(code_stats.get("enriched_count", 0))

                output(f"Total code files: {code_total:,}")

                # Cumulative throughput
                cum_triaged = triaged + scored + ingested + skipped
                cum_scored = scored + ingested + skipped

                if cum_triaged > 0:
                    output(
                        f"├─ Triaged:   {cum_triaged:,}"
                        f" ({cum_triaged / code_total * 100:.1f}%)"
                    )
                    if enriched > 0:
                        output(f"│  ├─ Enriched: {enriched:,}")
                    if cum_scored > 0:
                        output(f"│  ├─ Scored:   {cum_scored:,}")
                        if ingested > 0:
                            output(f"│  │  └─ Ingested: {ingested:,}")
                    if skipped > 0:
                        output(f"│  └─ Skipped:  {skipped:,}")
                if discovered > 0:
                    output(
                        f"├─ Pending:   {discovered:,}"
                        f" ({discovered / code_total * 100:.1f}%)"
                    )
                if failed > 0:
                    output(f"└─ Failed:    {failed:,}")

                # Language breakdown
                lang_keys = sorted(
                    k for k in code_stats if k.endswith("_files") and code_stats[k] > 0
                )
                if lang_keys:
                    output("\nBy language:")
                    for lk in lang_keys:
                        lang = lk.removesuffix("_files")
                        output(f"  {lang}: {int(code_stats[lk]):,}")

                # Pending work
                pending_triage = int(code_stats.get("pending_triage", 0))
                pending_enrich = int(code_stats.get("pending_enrich", 0))
                pending_score = int(code_stats.get("pending_score", 0))
                pending_ingest = int(code_stats.get("pending_ingest", 0))
                if any([pending_triage, pending_enrich, pending_score, pending_ingest]):
                    output("\nPending work:")
                    if pending_triage:
                        output(f"  Triage:  {pending_triage:,}")
                    if pending_enrich:
                        output(f"  Enrich:  {pending_enrich:,}")
                    if pending_score:
                        output(f"  Score:   {pending_score:,}")
                    if pending_ingest:
                        output(f"  Ingest:  {pending_ingest:,}")
            elif domain == "code":
                output("No code files discovered")
        except Exception:
            if domain == "code":
                output("Code stats unavailable")
