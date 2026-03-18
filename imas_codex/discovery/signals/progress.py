"""
Progress display for parallel data signal discovery.

Design principles (matching wiki and paths progress displays):
- Clean hierarchy: Target → Pipeline → Resources
- Unified per-stage blocks: progress bar + current activity + detail
- Per-stage cost and worker count annotations
- Gradient progress bars with percentage
- Resource gauges for time and cost budgets
- Minimal visual clutter (no emojis, thin progress bars)

Display layout: PIPELINE → RESOURCES
- SCAN: Seed, extract, and promote workers (tree → signal pipeline)
- ENRICH: LLM classification of physics domain, description
- CHECK: Test data access, verify units/sign

Uses common pipeline infrastructure from base.progress module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from imas_codex.discovery.base.progress import (
    BaseProgressDisplay,
    PipelineRowConfig,
    ResourceConfig,
    StreamQueue,
    WorkerStats,
    build_pipeline_section,
    build_resource_section,
    clean_text,
    compute_projected_etc,
    format_count,
    format_rate,
    format_time,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Display Items
# =============================================================================


@dataclass
class ScanItem:
    """Current scan activity."""

    signal_id: str
    data_source_name: str | None = None
    data_source_path: str | None = None
    signals_in_source: int = 0  # Count of signals discovered in current source
    # Epoch detection progress
    epoch_phase: str | None = None  # "coarse", "refine", "build"
    epoch_current_shot: int | None = None
    epoch_shots_scanned: int = 0
    epoch_total_shots: int = 0
    epoch_boundaries_found: int = 0
    epoch_boundaries_refined: int = 0


@dataclass
class EnrichItem:
    """Current enrich activity."""

    signal_id: str
    physics_domain: str | None = None
    description: str = ""


@dataclass
class ExtractItem:
    """Current extract activity."""

    version_id: str
    data_source_name: str | None = None
    version: int | None = None
    node_count: int = 0


@dataclass
class PromoteItem:
    """Current promote/units activity."""

    data_source_name: str
    signals_promoted: int = 0
    units_count: int = 0


@dataclass
class CheckItem:
    """Current check activity."""

    signal_id: str
    shot: int | None = None
    success: bool | None = None
    error: str | None = None
    physics_domain: str | None = None  # For display on second line
    scanner_type: str | None = None  # Scanner name for display prefix


# =============================================================================
# Progress State
# =============================================================================


@dataclass
class DataProgressState:
    """All state for the data progress display."""

    facility: str
    cost_limit: float
    signal_limit: int | None = None
    focus: str = ""

    # Mode flags
    discover_only: bool = False
    enrich_only: bool = False

    # Counts from graph
    total_signals: int = 0  # All FacilitySignal nodes for this facility
    signals_discovered: int = 0
    signals_enriched: int = 0
    signals_checked: int = 0
    signals_skipped: int = 0
    signals_failed: int = 0

    # Pending work counts
    pending_enrich: int = 0
    pending_check: int = 0

    # Signal source tracking
    signal_sources: int = 0
    grouped_signals: int = 0

    # This run stats
    run_discovered: int = 0
    run_enriched: int = 0
    run_checked: int = 0
    _run_enrich_cost: float = 0.0
    discover_rate: float | None = None
    enrich_rate: float | None = None
    check_rate: float | None = None

    # Accumulated facility cost and time (from graph)
    accumulated_cost: float = 0.0
    accumulated_time: float = 0.0

    # Extract/promote stats (tracked internally, displayed as SCAN)
    run_extracted: int = 0
    run_promoted: int = 0
    extract_rate: float | None = None
    promote_rate: float | None = None

    # Current items
    current_scan: ScanItem | None = None
    current_extract: ExtractItem | None = None
    current_promote: PromoteItem | None = None
    current_enrich: EnrichItem | None = None
    current_check: CheckItem | None = None
    scan_processing: bool = False
    extract_processing: bool = False
    promote_processing: bool = False
    current_tree: str | None = None  # Currently scanning tree
    enrich_processing: bool = False
    check_processing: bool = False

    # Worker stats references (for scanner status, error rates, connection timing)
    discover_stats: WorkerStats = field(default_factory=WorkerStats)
    extract_stats: WorkerStats = field(default_factory=WorkerStats)
    check_stats: WorkerStats = field(default_factory=WorkerStats)

    # Streaming queues (extract/promote share scan_queue display slot)
    scan_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=2.0, max_rate=5.0, min_display_time=0.3
        )
    )
    extract_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=1.0, max_rate=3.0, min_display_time=0.4
        )
    )
    promote_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=1.0, max_rate=3.0, min_display_time=0.4
        )
    )
    enrich_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.0, min_display_time=0.4
        )
    )
    check_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.0, min_display_time=0.4
        )
    )

    # Tracking
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def run_cost(self) -> float:
        return self._run_enrich_cost

    @property
    def cost_fraction(self) -> float:
        if self.cost_limit <= 0:
            return 0.0
        return min(1.0, self.run_cost / self.cost_limit)

    @property
    def cost_limit_reached(self) -> bool:
        return self.run_cost >= self.cost_limit if self.cost_limit > 0 else False

    @property
    def signal_limit_reached(self) -> bool:
        if self.signal_limit is None or self.signal_limit <= 0:
            return False
        return self.run_enriched >= self.signal_limit

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time to complete all pending work.

        Pure work-based ETA: max across parallel worker groups
        (bounded by the slowest pipeline).  Cost and signal limits
        are stop conditions, not ETA inputs.

        Uses stage-aware worker rates so ETA matches the rates shown
        in the pipeline rows.
        """
        from imas_codex.discovery.base.progress import compute_parallel_eta

        return compute_parallel_eta([
            (self.pending_enrich, self.enrich_rate),
            (self.pending_check, self.check_rate),
        ])


# =============================================================================
# Main Display Class
# =============================================================================


class DataProgressDisplay(BaseProgressDisplay):
    """Clean progress display for parallel signal discovery.

    Extends ``BaseProgressDisplay`` for the signal discovery pipeline
    (SEED → EXTRACT → PROMOTE → ENRICH → CHECK).  Inherits header, servers, worker tracking,
    and live-display lifecycle from the base class.
    """

    def __init__(
        self,
        facility: str,
        cost_limit: float,
        signal_limit: int | None = None,
        focus: str = "",
        console: Console | None = None,
        discover_only: bool = False,
        enrich_only: bool = False,
    ) -> None:
        super().__init__(
            facility=facility,
            cost_limit=cost_limit,
            console=console,
            focus=focus,
            title_suffix="Signal Discovery",
        )
        self.state = DataProgressState(
            facility=facility,
            cost_limit=cost_limit,
            signal_limit=signal_limit,
            focus=focus,
            discover_only=discover_only,
            enrich_only=enrich_only,
        )

    def _header_mode_label(self) -> str | None:
        """Show SCAN ONLY / ENRICH ONLY mode in header."""
        if self.state.discover_only:
            return "SCAN ONLY"
        if self.state.enrich_only:
            return "ENRICH ONLY"
        return None

    def _build_pipeline_section(self) -> Text:
        """Build the unified pipeline section (progress + activity merged).

        Each pipeline stage gets a 3-line block:
          Line 1: SCAN   ━━━━━━━━━━━━━━━━━━   12,944  87%    84K/s
          Line 2:        extracting v3 results  1,204 nodes
          Line 3:        seeded 4 trees  promoted 12,944 signals

        Stages: SCAN → ENRICH → CHECK
        """

        # --- Compute progress data ---

        max(self.state.total_signals, 1)
        enriched = self.state.signals_enriched + self.state.signals_checked
        checked = self.state.signals_checked
        # Enrich denominator excludes skipped signals (they never need enrichment)
        enrich_denom = max(
            self.state.total_signals - self.state.signals_skipped, 1
        )
        check_denom = max(enriched, 1)

        # Worker counts — SCAN combines seed + extract + promote groups
        seed_count, seed_ann = self._count_group_workers("seed")
        extract_count, extract_ann = self._count_group_workers("extract")
        promote_count, promote_ann = self._count_group_workers("promote")
        scan_count = seed_count + extract_count + promote_count
        # Collect annotations from sub-groups
        scan_ann_parts = [a for a in (seed_ann, extract_ann, promote_ann) if a]
        scan_ann = " ".join(scan_ann_parts)

        enrich_count, enrich_ann = self._count_group_workers("enrich")
        check_count, check_ann = self._count_group_workers("check")

        # Enrich cost
        enrich_cost = (
            self.state._run_enrich_cost if self.state._run_enrich_cost > 0 else None
        )

        # --- Build activity data ---

        # Worker completion detection
        seed_complete = self._worker_complete("seed") and not self.state.current_scan
        extract_complete = (
            self._worker_complete("extract") and not self.state.current_extract
        )
        promote_complete = (
            self._worker_complete("promote") and not self.state.current_promote
        )
        scan_complete = seed_complete and extract_complete and promote_complete
        scan_processing = (
            self.state.scan_processing
            or self.state.extract_processing
            or self.state.promote_processing
        )

        enrich_complete = (
            self._worker_complete("enrich") and not self.state.current_enrich
        )
        check_complete = self._worker_complete("check") and not self.state.current_check

        # SCAN activity — show the most active sub-phase
        # Priority: extract (SSH, slow) > promote (graph) > seed (fast)
        scan_text = ""
        scan_desc = ""

        ext = self.state.current_extract
        prom = self.state.current_promote
        scan = self.state.current_scan

        if ext:
            # Extract is the interesting work — show it prominently
            scan_text = (
                f"extracting v{ext.version} {ext.data_source_name}"
                if ext.version
                else f"extracting {ext.version_id}"
            )
            if ext.node_count > 0:
                scan_desc = f"{format_count(ext.node_count)} nodes"
            else:
                # Use connection timing from extract_stats if available
                conn_status = self.state.extract_stats.format_connection_status()
                scan_desc = conn_status or "connecting..."
        elif prom:
            scan_text = f"promoting {prom.data_source_name}"
            parts = []
            if prom.signals_promoted > 0:
                parts.append(f"{format_count(prom.signals_promoted)} signals")
            if prom.units_count > 0:
                parts.append(f"{format_count(prom.units_count)} units")
            scan_desc = "  ".join(parts)
        elif scan:
            if scan.epoch_phase:
                if scan.epoch_phase == "coarse":
                    pct = (
                        int(100 * scan.epoch_shots_scanned / scan.epoch_total_shots)
                        if scan.epoch_total_shots > 0
                        else 0
                    )
                    scan_text = f"epoch scan {pct}% ({scan.epoch_shots_scanned}/{scan.epoch_total_shots} shots)"
                    if scan.epoch_current_shot:
                        scan_text += f" at shot {scan.epoch_current_shot}"
                elif scan.epoch_phase == "refine":
                    scan_text = (
                        f"refining boundary {scan.epoch_boundaries_refined + 1}/"
                        f"{scan.epoch_boundaries_found}"
                    )
                else:
                    scan_text = f"building {scan.epoch_boundaries_found} epochs"
                if scan.data_source_name:
                    scan_desc = f"tree={scan.data_source_name}"
                if scan.epoch_boundaries_found > 0:
                    scan_desc += f"  {scan.epoch_boundaries_found} epochs detected"
            else:
                path = scan.data_source_path or scan.signal_id
                scan_text = f"seeding {path}"
                if scan.data_source_name:
                    scan_desc = f"tree={scan.data_source_name}"
                if scan.signals_in_source > 0:
                    scan_desc += f"  {format_count(scan.signals_in_source)} versions"
        elif scan_processing and self.state.current_tree:
            scan_text = f"tree={self.state.current_tree}"

        # SCAN progress: use graph state total_signals as ground truth
        # (matches how paths CLI derives scan progress from graph refresh).
        # run_promoted only tracks this-session promotes, which is 0 when
        # running --enrich-only or when scan workers have already finished.
        # Note: run_discovered can double-count (existing + new) since the
        # seed worker starts from graph state then adds per-scanner counts.
        # Use total_signals from graph refresh as the authoritative total.
        scan_completed = (
            self.state.run_promoted
            if self.state.run_promoted > 0
            else self.state.total_signals
        )
        scan_total = max(
            self.state.total_signals,
            self.state.run_promoted,
            1,
        )
        # Use promote rate as the representative rate (output rate)
        # Fall back to extract rate, then discover rate
        scan_rate = (
            self.state.promote_rate
            or self.state.extract_rate
            or self.state.discover_rate
        )

        # ENRICH activity
        enrich = self.state.current_enrich
        enrich_text = ""
        enrich_domain = ""
        enrich_desc = ""
        if enrich:
            # Strip facility prefix for facility-scoped CLI display
            enrich_text = enrich.signal_id
            prefix = f"{self.state.facility}:"
            if enrich_text.startswith(prefix):
                enrich_text = enrich_text[len(prefix):]
            if enrich.physics_domain:
                enrich_domain = enrich.physics_domain
            if enrich.description:
                enrich_desc = clean_text(enrich.description)

        # CHECK activity
        validate = self.state.current_check
        check_text = ""
        check_domain = ""
        check_desc = ""
        if validate:
            # Strip facility prefix for facility-scoped CLI display
            sig_display = validate.signal_id
            prefix = f"{self.state.facility}:"
            if sig_display.startswith(prefix):
                sig_display = sig_display[len(prefix):]
            # Display as "<scanner> <signal_name>" when scanner_type is known
            if validate.scanner_type:
                check_text = f"{validate.scanner_type} {sig_display}"
            else:
                check_text = sig_display
            if validate.physics_domain:
                check_domain = validate.physics_domain
            shot_str = f"shot={validate.shot}" if validate.shot else ""
            if validate.success is True:
                status = "success"
            elif validate.success is False:
                status = validate.error[:40] if validate.error else "failed"
            else:
                status = "testing..."
            check_desc = f"{shot_str}  {status}".strip() if shot_str else status

        # --- Build pipeline rows ---

        # Scanner status line as description fallback for SCAN row
        scanner_status = self.state.discover_stats.format_scanner_status()
        scan_desc_fallback = scanner_status or ""

        rows = [
            PipelineRowConfig(
                name="SCAN",
                style="bold blue",
                completed=scan_completed,
                total=scan_total,
                rate=scan_rate,
                show_pct=True,
                worker_count=scan_count,
                worker_annotation=scan_ann,
                primary_text=scan_text,
                description=scan_desc,
                description_fallback=scan_desc_fallback,
                is_processing=scan_processing,
                is_complete=scan_complete,
                processing_label="scanning...",
            ),
            PipelineRowConfig(
                name="ENRICH",
                style="bold green",
                completed=enriched,
                total=enrich_denom,
                rate=self.state.enrich_rate,
                cost=enrich_cost,
                disabled=self.state.discover_only,
                worker_count=enrich_count,
                worker_annotation=enrich_ann,
                primary_text=enrich_text,
                physics_domain=enrich_domain,
                description=enrich_desc,
                is_processing=self.state.enrich_processing,
                is_complete=enrich_complete,
                processing_label="classifying...",
            ),
            PipelineRowConfig(
                name="CHECK",
                style="bold magenta",
                completed=checked,
                total=check_denom,
                rate=self.state.check_rate,
                disabled=self.state.discover_only or self.state.enrich_only,
                worker_count=check_count,
                worker_annotation=check_ann,
                primary_text=check_text,
                physics_domain=check_domain,
                description=check_desc,
                is_processing=self.state.check_processing,
                is_complete=check_complete,
                processing_label="testing...",
            ),
        ]
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges using unified builder."""
        total = self.state.total_signals
        enriched = self.state.signals_enriched + self.state.signals_checked
        checked = self.state.signals_checked

        # Compute ETC — sum of per-worker cost projections
        total_cost = self.state.accumulated_cost
        cost_per_signal = (
            self.state._run_enrich_cost / self.state.run_enriched
            if self.state.run_enriched > 0
            else None
        )
        etc = compute_projected_etc(total_cost, [
            (self.state.pending_enrich, cost_per_signal),
            (self.state.pending_check, cost_per_signal),
        ])

        # Build stats
        stats: list[tuple[str, str, str]] = [
            ("discovered", str(total), "blue"),
            ("enriched", str(enriched), "green"),
            ("checked", str(checked), "magenta"),
        ]
        if self.state.signal_sources > 0:
            stats.append(("groups", str(self.state.signal_sources), "cyan"))
        if self.state.signals_failed > 0:
            stats.append(("failed", str(self.state.signals_failed), "red"))
        if self.state.signals_skipped > 0:
            stats.append(("skipped", str(self.state.signals_skipped), "dim"))

        # Scanner timing (Phase 1.3)
        scanner_timing = self.state.discover_stats.format_scanner_timing()

        # Error rate annotation (Phase 2.1) — show if check errors are notable
        check_err_pct = self.state.check_stats.error_rate_pct
        if check_err_pct >= 5:
            style = self.state.check_stats.error_health_style
            stats.append((f"err {check_err_pct:.0f}%", "", style))

        config = ResourceConfig(
            elapsed=self.state.elapsed,
            eta=None if self.state.discover_only else self.state.eta_seconds,
            accumulated_time=self.state.accumulated_time,
            run_cost=self.state.run_cost,
            cost_limit=self.state.cost_limit,
            accumulated_cost=self.state.accumulated_cost,
            etc=etc,
            scan_only=self.state.discover_only,
            stats=stats,
            pending=[
                ("enrich", self.state.pending_enrich),
                ("check", self.state.pending_check),
            ],
            scanner_timing=scanner_timing,
        )
        return build_resource_section(config, self.gauge_width)

    # ========================================================================
    # Public API
    # ========================================================================

    def tick(self) -> None:
        """Drain streaming queues for smooth display.

        Clears stale current items when queues have drained and no new
        items have been added for the stale timeout.
        """
        if item := self.state.scan_queue.pop():
            # Extract epoch progress if present
            epoch_progress = item.get("epoch_progress", {})
            self.state.current_scan = ScanItem(
                signal_id=item.get("signal_id", ""),
                data_source_name=item.get("data_source_name"),
                data_source_path=item.get("data_source_path"),
                signals_in_source=item.get("signals_in_source", 0),
                epoch_phase=epoch_progress.get("phase"),
                epoch_current_shot=epoch_progress.get("current_shot"),
                epoch_shots_scanned=epoch_progress.get("shots_scanned", 0),
                epoch_total_shots=epoch_progress.get("total_shots", 0),
                epoch_boundaries_found=epoch_progress.get("boundaries_found", 0),
                epoch_boundaries_refined=epoch_progress.get("boundaries_refined", 0),
            )
            # Track current tree for idle display
            if item.get("data_source_name"):
                self.state.current_tree = item.get("data_source_name")
        elif self.state.scan_queue.is_stale():
            self.state.current_scan = None

        if item := self.state.extract_queue.pop():
            self.state.current_extract = ExtractItem(
                version_id=item.get("id", ""),
                data_source_name=item.get("data_source_name"),
                version=item.get("version"),
                node_count=item.get("node_count", 0),
            )
        elif self.state.extract_queue.is_stale():
            self.state.current_extract = None

        if item := self.state.promote_queue.pop():
            self.state.current_promote = PromoteItem(
                data_source_name=item.get("data_source_name", ""),
                signals_promoted=item.get("signals_promoted", 0),
                units_count=item.get("units_count", 0),
            )
        elif self.state.promote_queue.is_stale():
            self.state.current_promote = None

        if item := self.state.enrich_queue.pop():
            self.state.current_enrich = EnrichItem(
                signal_id=item.get("signal_id", ""),
                physics_domain=item.get("physics_domain"),
                description=item.get("description", ""),
            )
        elif self.state.enrich_queue.is_stale():
            self.state.current_enrich = None

        if item := self.state.check_queue.pop():
            self.state.current_check = CheckItem(
                signal_id=item.get("signal_id", ""),
                shot=item.get("shot"),
                success=item.get("success"),
                error=item.get("error"),
                physics_domain=item.get("physics_domain"),
                scanner_type=item.get("scanner_type"),
            )
        elif self.state.check_queue.is_stale():
            self.state.current_check = None

        self._refresh()

    def update_scan(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
        current_tree: str | None = None,
    ) -> None:
        """Update seed/epoch worker state."""
        self.state.discover_stats = stats
        self.state.run_discovered = stats.processed

        if "idle" in message.lower():
            self.state.scan_processing = False
            stats.mark_idle()
            self.state.discover_rate = stats.active_rate or stats.rate
            self._refresh()
            return

        stats.mark_active()
        self.state.discover_rate = stats.active_rate or stats.rate

        if current_tree:
            self.state.current_tree = current_tree

        if "scanning" in message.lower() or "epoch" in message.lower():
            self.state.scan_processing = True
        else:
            self.state.scan_processing = False

        if results:
            # Get signal count for this tree from results
            tree_counts: dict[str, int] = {}
            for r in results:
                tree = r.get("data_source_name")
                if tree:
                    tree_counts[tree] = tree_counts.get(tree, 0) + 1

            items = [
                {
                    "signal_id": r.get("id", ""),
                    "data_source_name": r.get("data_source_name"),
                    "data_source_path": r.get("data_source_path"),
                    "signals_in_source": tree_counts.get(
                        r.get("data_source_name", ""), 0
                    ),
                    "epoch_progress": r.get("epoch_progress"),  # Include epoch progress
                }
                for r in results
            ]
            max_rate = 5.0
            effective = stats.active_rate or stats.rate
            display_rate = min(effective, max_rate) if effective else 2.0
            self.state.scan_queue.add(
                items,
                display_rate,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    def update_extract(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update extract worker state."""
        self.state.extract_stats = stats
        self.state.run_extracted = stats.processed

        if "idle" in message.lower():
            self.state.extract_processing = False
            stats.mark_idle()
            self.state.extract_rate = stats.active_rate or stats.rate
            self._refresh()
            return

        stats.mark_active()
        self.state.extract_rate = stats.active_rate or stats.rate

        if "extracting" in message.lower() or "v" in message.lower():
            self.state.extract_processing = True
        else:
            self.state.extract_processing = False

        if results:
            items = [
                {
                    "id": r.get("id", ""),
                    "data_source_name": r.get("data_source_name"),
                    "version": r.get("version"),
                    "node_count": r.get("signals_in_source", 0),
                }
                for r in results
            ]
            max_rate = 3.0
            effective = stats.active_rate or stats.rate
            display_rate = min(effective, max_rate) if effective else 1.0
            self.state.extract_queue.add(
                items,
                display_rate,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    def update_promote(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update units/promote worker state."""
        self.state.run_promoted = stats.processed

        if "idle" in message.lower():
            self.state.promote_processing = False
            stats.mark_idle()
            self.state.promote_rate = stats.active_rate or stats.rate
            self._refresh()
            return

        stats.mark_active()
        self.state.promote_rate = stats.active_rate or stats.rate

        if "promoting" in message.lower() or "units" in message.lower():
            self.state.promote_processing = True
        else:
            self.state.promote_processing = False

        if results:
            items = [
                {
                    "data_source_name": r.get("data_source_name", ""),
                    "signals_promoted": r.get("signals_in_source", 0),
                    "units_count": r.get("units_count", 0),
                }
                for r in results
            ]
            max_rate = 3.0
            effective = stats.active_rate or stats.rate
            display_rate = min(effective, max_rate) if effective else 1.0
            self.state.promote_queue.add(
                items,
                display_rate,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    def update_enrich(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update enrich worker state."""
        self.state.run_enriched = stats.processed
        self.state._run_enrich_cost = stats.cost

        msg_lower = message.lower()
        if "idle" in msg_lower:
            self.state.enrich_processing = False
            stats.mark_idle()
            self.state.enrich_rate = stats.active_rate or stats.rate
            self._refresh()
            return
        if "classifying" in msg_lower or "enriching" in msg_lower:
            self.state.enrich_processing = True
            stats.mark_active()
        elif "detected" in msg_lower and "patterns" in msg_lower:
            # Pattern detection — not LLM processing but keep processing flag
            stats.mark_active()
        elif "propagated" in msg_lower:
            # Propagation — processing is active
            self.state.enrich_processing = True
            stats.mark_active()
        else:
            self.state.enrich_processing = False

        self.state.enrich_rate = stats.active_rate or stats.rate

        if results:
            items = [
                {
                    "signal_id": r.get("id", ""),
                    "physics_domain": r.get("physics_domain"),
                    "description": r.get("description", ""),
                }
                for r in results
            ]
            max_rate = 2.0
            effective = stats.active_rate or stats.rate
            display_rate = min(effective, max_rate) if effective else 0.5
            self.state.enrich_queue.add(
                items,
                display_rate,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    def update_check(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update validate worker state."""
        self.state.check_stats = stats
        self.state.run_checked = stats.processed

        msg_lower = message.lower()
        if "idle" in msg_lower:
            self.state.check_processing = False
            stats.mark_idle()
            self.state.check_rate = stats.active_rate or stats.rate
            self._refresh()
            return

        stats.mark_active()
        self.state.check_rate = stats.active_rate or stats.rate

        if "testing" in msg_lower or "validating" in msg_lower:
            self.state.check_processing = True
        else:
            self.state.check_processing = False

        if results:
            items = [
                {
                    "signal_id": r.get("id", ""),
                    "shot": r.get("shot"),
                    "success": r.get("success"),
                    "error": r.get("error"),
                    "physics_domain": r.get("physics_domain"),
                    "scanner_type": r.get("scanner_type"),
                }
                for r in results
            ]
            max_rate = 2.0
            effective = stats.active_rate or stats.rate
            display_rate = min(effective, max_rate) if effective else 0.5
            self.state.check_queue.add(
                items,
                display_rate,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    def update_from_graph(
        self,
        total_signals: int = 0,
        signals_discovered: int = 0,
        signals_enriched: int = 0,
        signals_checked: int = 0,
        signals_skipped: int = 0,
        signals_failed: int = 0,
        pending_enrich: int = 0,
        pending_check: int = 0,
        accumulated_cost: float = 0.0,
        **kwargs,
    ) -> None:
        """Update state from graph statistics."""
        self.state.total_signals = total_signals
        self.state.signals_discovered = signals_discovered
        self.state.signals_enriched = signals_enriched
        self.state.signals_checked = signals_checked
        self.state.signals_skipped = signals_skipped
        self.state.signals_failed = signals_failed
        self.state.pending_enrich = pending_enrich
        self.state.pending_check = pending_check
        self.state.accumulated_cost = accumulated_cost
        if "signal_sources" in kwargs:
            self.state.signal_sources = kwargs["signal_sources"]
        if "grouped_signals" in kwargs:
            self.state.grouped_signals = kwargs["grouped_signals"]

        # Accumulated wall-clock time from prior sessions
        from imas_codex.discovery.base.progress import get_accumulated_time
        self.state.accumulated_time = get_accumulated_time(
            self.state.facility, "signals"
        )

        self._refresh()

    def print_summary(self) -> None:
        """Print final discovery summary."""
        self.console.print()
        self.console.print(
            Panel(
                self._build_summary(),
                title=f"{self.state.facility.upper()} Signal Discovery Complete",
                border_style="green",
                width=self.width,
            )
        )

    def _build_summary(self) -> Text:
        """Build final summary text."""
        summary = Text()

        total = self.state.total_signals
        enriched = self.state.signals_enriched + self.state.signals_checked
        checked = self.state.signals_checked

        # SEED stats
        summary.append("  SEED    ", style="bold blue")
        summary.append(f"seeded={format_count(self.state.run_discovered)}", style="blue")
        if self.state.discover_rate:
            summary.append(f"  {format_rate(self.state.discover_rate)}", style="dim")
        summary.append("\n")

        # EXTRACT stats
        summary.append("  EXTRACT ", style="bold cyan")
        summary.append(f"extracted={format_count(self.state.run_extracted)}", style="cyan")
        if self.state.extract_rate:
            summary.append(f"  {format_rate(self.state.extract_rate)}", style="dim")
        summary.append("\n")

        # PROMOTE stats
        summary.append("  PROMOTE ", style="bold yellow")
        summary.append(f"promoted={format_count(self.state.run_promoted)}", style="yellow")
        summary.append(f"  total={format_count(total)}", style="dim")
        if self.state.promote_rate:
            summary.append(f"  {format_rate(self.state.promote_rate)}", style="dim")
        summary.append("\n")

        # ENRICH stats
        summary.append("  ENRICH  ", style="bold green")
        summary.append(f"enriched={format_count(enriched)}", style="green")
        if self.state.signal_sources > 0:
            summary.append(
                f"  groups={self.state.signal_sources} ({format_count(self.state.grouped_signals)} members)",
                style="cyan",
            )
        summary.append(f"  skipped={format_count(self.state.signals_skipped)}", style="yellow")
        summary.append(f"  cost=${self.state.run_cost:.3f}", style="yellow")
        if self.state.enrich_rate:
            summary.append(f"  {format_rate(self.state.enrich_rate)}", style="dim")
        summary.append("\n")

        # CHECK stats
        summary.append("  CHECK  ", style="bold magenta")
        summary.append(f"checked={format_count(checked)}", style="magenta")
        summary.append(f"  failed={format_count(self.state.signals_failed)}", style="red")
        if self.state.check_rate:
            summary.append(f"  {format_rate(self.state.check_rate)}", style="dim")
        summary.append("\n")

        # USAGE stats
        summary.append("  USAGE ", style="bold cyan")
        summary.append(f"time={format_time(self.state.elapsed)}", style="white")
        summary.append(f"  cost=${self.state.accumulated_cost:.2f}", style="yellow")
        if self.state.run_cost > 0:
            summary.append(f"  session=${self.state.run_cost:.2f}", style="dim")

        return summary
