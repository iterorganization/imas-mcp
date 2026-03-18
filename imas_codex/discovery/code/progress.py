"""
Progress display for parallel file discovery.

Design principles (matching paths, signals, wiki progress displays):
- Clean hierarchy: Target → Pipeline → Resources
- Unified per-stage blocks: progress bar + current activity + detail
- Per-stage cost and worker count annotations
- Gradient progress bars with percentage
- Resource gauges for time and cost budgets
- Minimal visual clutter (no emojis, thin progress bars)

Display layout: PIPELINE → RESOURCES
- SCAN: SSH file enumeration from scored FacilityPaths
- TRIAGE: Per-dimension LLM scoring of discovered CodeFiles
- ENRICH: rg pattern matching + preview extraction on triaged files
- SCORE: Full LLM scoring of enriched CodeFiles
- INGEST: Fetch, tree-sitter chunk, embed scored files

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
    build_pipeline_section,
    build_resource_section,
    clean_text,
    compute_projected_etc,
    format_count,
    format_time,
)

if TYPE_CHECKING:
    from imas_codex.discovery.base.progress import WorkerStats


# =============================================================================
# Display Items
# =============================================================================


@dataclass
class ScanItem:
    """Current scan activity."""

    path: str
    files_found: int = 0
    score_composite: float | None = None  # FacilityPath.score_composite


@dataclass
class TriageItem:
    """Current triage activity."""

    path: str
    triage_composite: float | None = None  # CodeFile.triage_composite
    category: str = ""  # top scoring dimension (modeling, analysis, imas, etc.)
    description: str = ""  # LLM description of what the file contains
    skipped: bool = False


@dataclass
class ScoreItem:
    """Current score activity."""

    path: str
    score_composite: float | None = None
    category: str = ""  # code, document, notebook, config
    description: str = ""  # LLM reasoning about what the file contains
    skipped: bool = False  # LLM says skip this file


@dataclass
class EnrichItem:
    """Current enrich activity."""

    path: str
    triage_composite: float | None = None  # CodeFile.triage_composite
    patterns: int = 0  # total pattern matches
    line_count: int = 0  # lines of code
    pattern_categories: dict[str, int] = field(
        default_factory=dict
    )  # per-category match counts
    preview_snippet: str = ""  # first meaningful line of file content


@dataclass
class IngestItem:
    """Current ingestion activity (code or docs)."""

    path: str
    language: str = ""
    file_type: str = ""  # code, document, notebook, config
    score_composite: float | None = None  # CodeFile.score_composite
    chunks: int = 0  # average chunks per file in batch


@dataclass
class EmbedItem:
    """Current embedding activity for chunk nodes."""

    chunk_id: str  # CodeChunk node ID (e.g. "jet:path/to/file.py:chunk_0")
    label: str = "CodeChunk"  # node label being embedded
    score_composite: float | None = (
        None  # parent CodeFile.score_composite (graph lookup)
    )


# =============================================================================
# Progress State
# =============================================================================


@dataclass
class FileProgressState:
    """All state for the file progress display."""

    facility: str
    cost_limit: float
    focus: str = ""

    # Mode flags
    scan_only: bool = False
    score_only: bool = False

    # Counts from graph
    total: int = 0
    discovered: int = 0
    triaged_count: int = 0
    ingested: int = 0
    failed: int = 0
    skipped: int = 0
    pending_triage: int = 0
    pending_enrich: int = 0
    pending_score: int = 0
    pending_ingest: int = 0
    scored_count: int = 0
    enriched_count: int = 0
    code_files: int = 0

    # This run stats
    run_scanned: int = 0
    run_triaged: int = 0
    run_scored: int = 0
    run_skipped: int = 0
    run_enriched: int = 0
    run_code_ingested: int = 0
    run_embedded: int = 0
    _run_triage_cost: float = 0.0
    _run_score_cost: float = 0.0
    scan_rate: float | None = None
    triage_rate: float | None = None
    score_rate: float | None = None
    enrich_rate: float | None = None
    code_rate: float | None = None
    embed_rate: float | None = None

    # Accumulated facility cost and time (from graph)
    accumulated_cost: float = 0.0
    accumulated_time: float = 0.0

    # Embed stats (from graph)
    total_chunks: int = 0
    embedded_chunks: int = 0
    pending_embed: int = 0

    # Current items
    current_scan: ScanItem | None = None
    current_triage: TriageItem | None = None
    current_score: ScoreItem | None = None
    current_enrich: EnrichItem | None = None
    current_ingest: IngestItem | None = None
    current_embed: EmbedItem | None = None
    scan_processing: bool = False
    triage_processing: bool = False
    score_processing: bool = False
    enrich_processing: bool = False
    ingest_processing: bool = False
    embed_processing: bool = False

    # Streaming queues
    scan_queue: StreamQueue = field(default_factory=StreamQueue)
    triage_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.0, min_display_time=0.4
        )
    )
    score_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.0, min_display_time=0.4
        )
    )
    enrich_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.5, min_display_time=0.4
        )
    )
    ingest_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.0, min_display_time=0.4
        )
    )
    embed_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5,
            max_rate=3.0,
            min_display_time=0.3,
            max_display_time=3.0,
        )
    )

    # Tracking
    start_time: float = field(default_factory=time.time)

    @property
    def run_ingest_total(self) -> int:
        """Code files ingested this run."""
        return self.run_code_ingested

    @property
    def ingest_rate(self) -> float | None:
        """Code ingestion rate."""
        return self.code_rate

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def run_cost(self) -> float:
        return self._run_triage_cost + self._run_score_cost

    @property
    def cost_per_triage(self) -> float | None:
        """Average LLM cost per triaged file."""
        if self.run_triaged > 0:
            return self._run_triage_cost / self.run_triaged
        return None

    @property
    def cost_per_file(self) -> float | None:
        """Average LLM cost per scored file."""
        if self.run_scored > 0:
            return self._run_score_cost / self.run_scored
        return None

    @property
    def cost_limit_reached(self) -> bool:
        return self.run_cost >= self.cost_limit if self.cost_limit > 0 else False

    @property
    def limit_reason(self) -> str | None:
        if self.cost_limit_reached:
            return "cost"
        return None

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time to complete all pending work.

        Pure work-based ETA: max across parallel worker groups
        (bounded by the slowest pipeline).  Cost limits are stop
        conditions, not ETA inputs.

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
                (self.pending_triage, _agg(self.run_triaged, self.triage_rate)),
                (self.pending_enrich, _agg(self.run_enriched, self.enrich_rate)),
                (self.pending_score, _agg(self.run_scored, self.score_rate)),
                (self.pending_ingest, _agg(self.run_code_ingested, self.ingest_rate)),
                (self.pending_embed, _agg(self.run_embedded, self.embed_rate)),
            ]
        )


# =============================================================================
# Main Display Class
# =============================================================================


class FileProgressDisplay(BaseProgressDisplay):
    """Clean progress display for parallel file discovery.

    Extends ``BaseProgressDisplay`` for the file discovery pipeline
    (SCAN -> SCORE -> ENRICH -> INGEST -> IMAGE).  Inherits header, servers,
    worker tracking, and live-display lifecycle from the base class.
    """

    def __init__(
        self,
        facility: str,
        cost_limit: float,
        focus: str = "",
        console: Console | None = None,
        scan_only: bool = False,
        score_only: bool = False,
        min_score: float | None = None,
    ) -> None:
        super().__init__(
            facility=facility,
            cost_limit=cost_limit,
            console=console,
            focus=focus,
            title_suffix="Code Discovery",
        )
        self.min_score = min_score
        self.state = FileProgressState(
            facility=facility,
            cost_limit=cost_limit,
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
        """Build the unified pipeline section.

        Each pipeline stage gets a 3-line block:
          Line 1: SCAN   ━━━━━━━━━━━━━━━━━━    1,234  42%  77.1/s
          Line 2:        /home/codes/liuqe/src
          Line 3:        45 files found

        Stages: SCAN → TRIAGE → ENRICH → SCORE → INGEST
        """

        # --- Compute progress data ---

        # SCAN: files discovered / total from paths
        scan_total = max(self.state.total, 1)

        # TRIAGE: triaged / total needing triage
        triage_completed = self.state.triaged_count + self.state.skipped
        triage_total = max(self.state.pending_triage + triage_completed, 1)

        # ENRICH: enriched / total needing enrichment (triaged files)
        enrich_completed = self.state.enriched_count
        enrich_total = max(self.state.pending_enrich + enrich_completed, 1)

        # SCORE: scored / total needing scoring
        score_completed = self.state.scored_count
        score_total = max(self.state.pending_score + score_completed, 1)

        # INGEST: ingested / total needing ingestion
        ingest_completed = self.state.ingested
        ingest_total = max(self.state.ingested + self.state.pending_ingest, 1)

        # Cost for display
        triage_cost = (
            self.state._run_triage_cost if self.state._run_triage_cost > 0 else None
        )
        score_cost = (
            self.state._run_score_cost if self.state._run_score_cost > 0 else None
        )

        # Worker counts per group
        scan_count, scan_ann = self._count_group_workers("scan")
        triage_count, triage_ann = self._count_group_workers("triage")
        enrich_count, enrich_ann = self._count_group_workers("enrich")
        score_count, score_ann = self._count_group_workers("score")
        code_count, code_ann = self._count_group_workers("code")
        ingest_count = code_count
        ingest_ann = code_ann

        # --- Build activity data ---

        scan = self.state.current_scan
        triage = self.state.current_triage
        enrich = self.state.current_enrich
        score = self.state.current_score
        ingest = self.state.current_ingest

        # Worker completion detection
        scan_complete = self._worker_complete("scan") and not scan
        triage_complete = self._worker_complete("triage") and not triage
        enrich_complete = self._worker_complete("enrich") and not enrich
        score_complete = self._worker_complete("score") and not score
        code_complete = self._worker_complete("code")
        ingest_complete = code_complete and not ingest

        # SCAN activity
        scan_text = ""
        scan_desc = ""
        scan_score_parts: list[tuple[str, str]] | None = None
        if scan:
            scan_text = scan.path
            if scan.score_composite is not None:
                scan_score_parts = [(f"{scan.score_composite:.2f}", "bold blue")]
            if scan.files_found > 0:
                scan_desc = f"{scan.files_found} files found"

        # TRIAGE activity — score + category + description
        triage_text = ""
        triage_score_parts: list[tuple[str, str]] | None = None
        triage_category = ""
        triage_desc = ""
        triage_terminal = ""
        if triage:
            triage_text = triage.path
            if triage.skipped:
                triage_terminal = "skip"
                if triage.description:
                    triage_desc = clean_text(triage.description)
            elif triage.triage_composite is not None:
                triage_score_parts = [(f"{triage.triage_composite:.2f}", "bold green")]
                if triage.category:
                    triage_category = triage.category.replace("_", " ")
                if triage.description:
                    triage_desc = clean_text(triage.description)

        # ENRICH activity — triage score + line count + pattern categories + preview
        enrich_text = ""
        enrich_desc = ""
        enrich_score_parts: list[tuple[str, str]] | None = None
        if enrich:
            enrich_text = enrich.path
            if enrich.triage_composite is not None:
                enrich_score_parts = [(f"{enrich.triage_composite:.2f}", "bold green")]
            desc_parts = []
            if enrich.line_count > 0:
                desc_parts.append(f"{format_count(enrich.line_count)} LOC")
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
            elif enrich.patterns > 0:
                desc_parts.append(f"{enrich.patterns} patterns")
            if enrich.preview_snippet:
                desc_parts.append(enrich.preview_snippet)
            enrich_desc = "  ".join(desc_parts)

        # SCORE activity — score + category + description
        score_text = ""
        score_score_parts: list[tuple[str, str]] | None = None
        score_category = ""
        score_desc = ""
        score_terminal = ""
        if score:
            score_text = score.path
            if score.skipped:
                score_terminal = "skip"
                if score.category:
                    score_category = score.category
                if score.description:
                    score_desc = clean_text(score.description)
            elif score.score_composite is not None:
                score_score_parts = [(f"{score.score_composite:.2f}", "bold cyan")]
                if score.category:
                    score_category = score.category
                if score.description:
                    score_desc = clean_text(score.description)

        # INGEST activity — language + score + chunks
        ingest_text = ""
        ingest_desc = ""
        ingest_score_parts: list[tuple[str, str]] | None = None
        if ingest:
            ingest_text = ingest.path
            if ingest.score_composite is not None:
                ingest_score_parts = [(f"{ingest.score_composite:.2f}", "bold cyan")]
            desc_parts = []
            if ingest.language:
                desc_parts.append(f"[{ingest.language}]")
            elif ingest.file_type:
                desc_parts.append(f"[{ingest.file_type}]")
            if ingest.chunks > 0:
                desc_parts.append(f"~{ingest.chunks} chunks")
            ingest_desc = "  ".join(desc_parts)

        # EMBED activity — chunk ID
        embed = self.state.current_embed
        embed_count, embed_ann = self._count_group_workers("embed")
        embed_complete = self._worker_complete("embed") and not embed

        embed_total = max(self.state.total_chunks, self.state.embedded_chunks + 1)
        embed_text = ""
        embed_desc = ""
        embed_score_parts: list[tuple[str, str]] | None = None
        if embed:
            embed_text = embed.chunk_id
            embed_desc = embed.label
            if embed.score_composite is not None:
                embed_score_parts = [(f"{embed.score_composite:.2f}", "bold cyan")]

        # --- Build pipeline rows ---

        rows = [
            PipelineRowConfig(
                name="SCAN",
                style="bold blue",
                completed=self.state.total,
                total=scan_total,
                rate=self.state.scan_rate,
                disabled=self.state.score_only,
                primary_text=scan_text,
                score_parts=scan_score_parts,
                description=scan_desc,
                is_processing=self.state.scan_processing,
                is_complete=scan_complete,
                worker_count=scan_count,
                worker_annotation=scan_ann,
            ),
            PipelineRowConfig(
                name="TRIAGE",
                style="bold green",
                completed=triage_completed,
                total=triage_total,
                rate=self.state.triage_rate,
                cost=triage_cost,
                disabled=self.state.scan_only,
                primary_text=triage_text,
                score_parts=triage_score_parts,
                physics_domain=triage_category,
                terminal_label=triage_terminal,
                description=triage_desc,
                is_processing=self.state.triage_processing,
                is_complete=triage_complete,
                worker_count=triage_count,
                worker_annotation=triage_ann,
            ),
            PipelineRowConfig(
                name="ENRICH",
                style="bold magenta",
                completed=enrich_completed,
                total=enrich_total,
                rate=self.state.enrich_rate,
                disabled=self.state.scan_only,
                primary_text=enrich_text,
                score_parts=enrich_score_parts,
                description=enrich_desc,
                is_processing=self.state.enrich_processing,
                is_complete=enrich_complete,
                worker_count=enrich_count,
                worker_annotation=enrich_ann,
            ),
            PipelineRowConfig(
                name="SCORE",
                style="bold cyan",
                completed=score_completed,
                total=score_total,
                rate=self.state.score_rate,
                cost=score_cost,
                disabled=self.state.scan_only,
                primary_text=score_text,
                score_parts=score_score_parts,
                physics_domain=score_category,
                terminal_label=score_terminal,
                description=score_desc,
                is_processing=self.state.score_processing,
                is_complete=score_complete,
                worker_count=score_count,
                worker_annotation=score_ann,
            ),
            PipelineRowConfig(
                name="INGEST",
                style="bold yellow",
                completed=ingest_completed,
                total=ingest_total,
                rate=self.state.ingest_rate,
                disabled=self.state.scan_only or self.state.score_only,
                primary_text=ingest_text,
                score_parts=ingest_score_parts,
                description=ingest_desc,
                is_processing=self.state.ingest_processing,
                is_complete=ingest_complete,
                worker_count=ingest_count,
                worker_annotation=ingest_ann,
            ),
            PipelineRowConfig(
                name="EMBED",
                style="bold white",
                completed=self.state.embedded_chunks,
                total=embed_total,
                rate=self.state.embed_rate,
                disabled=self.state.scan_only or self.state.score_only,
                primary_text=embed_text,
                score_parts=embed_score_parts,
                description=embed_desc,
                is_processing=self.state.embed_processing,
                is_complete=embed_complete,
                worker_count=embed_count,
                worker_annotation=embed_ann,
            ),
        ]
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges using unified builder."""
        # Compute ETC — sum of per-worker cost projections
        total_cost = self.state.accumulated_cost
        etc = compute_projected_etc(
            total_cost,
            [
                (self.state.pending_triage, self.state.cost_per_triage),
                (self.state.pending_score, self.state.cost_per_file),
            ],
        )

        # Build stats — completed counts across pipeline
        stats: list[tuple[str, str, str]] = [
            ("total", str(self.state.total), "blue"),
            ("triaged", str(self.state.triaged_count), "green"),
            ("scored", str(self.state.scored_count), "green"),
            ("ingested", str(self.state.ingested), "green"),
        ]
        if self.state.embedded_chunks > 0 or self.state.pending_embed > 0:
            stats.append(("embedded", str(self.state.embedded_chunks), "white"))
        if self.state.skipped > 0:
            stats.append(("skipped", str(self.state.skipped), "yellow"))
        if self.state.failed > 0:
            stats.append(("failed", str(self.state.failed), "red"))

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
            stats=stats,
            pending=[
                ("triage", self.state.pending_triage),
                ("enrich", self.state.pending_enrich),
                ("score", self.state.pending_score),
                ("ingest", self.state.pending_ingest),
                ("embed", self.state.pending_embed),
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
        results: list[dict] | None = None,
    ) -> None:
        """Update scan worker state."""
        self.state.run_scanned = stats.processed
        ema = stats.ema_rate or stats.active_rate
        if ema and ema > 0:
            self.state.scan_rate = ema

        if message == "idle":
            self.state.scan_processing = False
            self._refresh()
            return
        elif "scanning" in message.lower():
            self.state.scan_processing = True
        else:
            self.state.scan_processing = False

        if results:
            items = [
                ScanItem(
                    path=r.get("path", ""),
                    files_found=r.get("files_found", 0),
                    score_composite=r.get("score_composite"),
                )
                for r in results
            ]
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
        """Update triage worker state with own display row."""
        self.state.run_triaged = stats.processed
        self.state._run_triage_cost = stats.cost

        if "idle" in message.lower():
            self.state.triage_processing = False
            stats.mark_idle()
            self._refresh()
            return
        elif "triaging" in message.lower():
            self.state.triage_processing = True
            stats.mark_active()
        elif results:
            self.state.triage_processing = not self.state.triage_queue.is_empty()
            stats.mark_active()

        ema = stats.ema_rate
        if ema and ema > 0:
            self.state.triage_rate = ema

        if results:
            items = [
                TriageItem(
                    path=r.get("path", ""),
                    triage_composite=r.get("triage_composite"),
                    category=r.get("category", ""),
                    description=r.get("description", ""),
                    skipped=r.get("skipped", False),
                )
                for r in results
            ]
            self.state.run_skipped += sum(1 for r in results if r.get("skipped"))
            self.state.triage_queue.add(
                items,
                stats.ema_rate,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    def update_score(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update score worker state."""
        self.state.run_scored = stats.processed
        self.state._run_score_cost = stats.cost

        if "waiting" in message.lower():
            self.state.score_processing = False
            stats.mark_idle()
            self._refresh()
            return
        elif "scoring" in message.lower():
            self.state.score_processing = True
            stats.mark_active()
        elif results:
            self.state.score_processing = not self.state.score_queue.is_empty()
            stats.mark_active()

        ema = stats.ema_rate
        if ema and ema > 0:
            self.state.score_rate = ema

        if results:
            items = [
                ScoreItem(
                    path=r.get("path", ""),
                    score_composite=r.get("score_composite"),
                    category=r.get("category", ""),
                    description=r.get("description", ""),
                    skipped=r.get("skipped", False),
                )
                for r in results
            ]
            self.state.score_queue.add(
                items,
                stats.ema_rate,
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

        if "waiting" in message.lower() or "idle" in message.lower():
            self.state.enrich_processing = False
            stats.mark_idle()
        elif "enriching" in message.lower() or "enriched" in message.lower():
            self.state.enrich_processing = True
            stats.mark_active()
        elif results:
            self.state.enrich_processing = not self.state.enrich_queue.is_empty()
            stats.mark_active()

        ema = stats.ema_rate
        if ema and ema > 0:
            self.state.enrich_rate = ema

        if results:
            items = [
                EnrichItem(
                    path=r.get("path", ""),
                    triage_composite=r.get("triage_composite"),
                    patterns=r.get("patterns", 0),
                    line_count=r.get("line_count", 0),
                    pattern_categories=r.get("pattern_categories", {}),
                    preview_snippet=r.get("preview_snippet", ""),
                )
                for r in results
            ]
            self.state.enrich_queue.add(
                items,
                stats.ema_rate,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    def update_code(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update code ingestion worker state (feeds into INGEST row)."""
        self.state.run_code_ingested = stats.processed

        if "waiting" in message.lower() or "idle" in message.lower():
            self.state.ingest_processing = False
            stats.mark_idle()
            self._refresh()
            return
        elif "ingesting" in message.lower() or "fetching" in message.lower():
            self.state.ingest_processing = True
            stats.mark_active()
        elif results:
            self.state.ingest_processing = not self.state.ingest_queue.is_empty()
            stats.mark_active()

        # Use EMA rate for live display (falls back to active_rate).
        # Only update when we have a valid rate — avoids overwriting the
        # previous rate with None during the first batch after a restart.
        ema = stats.ema_rate
        if ema and ema > 0:
            self.state.code_rate = ema

        if results:
            items = [
                IngestItem(
                    path=r.get("path", ""),
                    language=r.get("language", ""),
                    score_composite=r.get("score_composite"),
                    chunks=r.get("chunks", 0),
                    file_type="code",
                )
                for r in results
            ]
            self.state.ingest_queue.add(
                items,
                stats.ema_rate,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    def update_embed(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update chunk embedding worker state (feeds into EMBED row)."""
        self.state.run_embedded = stats.processed

        if "idle" in message.lower() or "waiting" in message.lower():
            self.state.embed_processing = False
            stats.mark_idle()
            self._refresh()
            return
        elif "embedded" in message.lower():
            self.state.embed_processing = True
            stats.mark_active()
        elif "failed" in message.lower() or "backing off" in message.lower():
            self.state.embed_processing = False
            stats.mark_idle()
        elif results:
            self.state.embed_processing = not self.state.embed_queue.is_empty()
            stats.mark_active()

        ema = stats.ema_rate
        if ema and ema > 0:
            self.state.embed_rate = ema

        if results:
            # Lookup parent CodeFile score_composite via graph traversal
            score_map = self._lookup_chunk_parent_scores(
                [r.get("id", "") for r in results if r.get("label") == "CodeChunk"]
            )
            items = [
                EmbedItem(
                    chunk_id=r.get("id", ""),
                    label=r.get("label", "CodeChunk"),
                    score_composite=score_map.get(r.get("id", "")),
                )
                for r in results
            ]
            self.state.embed_queue.add(
                items,
                stats.ema_rate,
                last_batch_time=stats.last_batch_time,
            )

        self._refresh()

    @staticmethod
    def _lookup_chunk_parent_scores(chunk_ids: list[str]) -> dict[str, float]:
        """Query parent CodeFile score_composite for CodeChunk nodes."""
        if not chunk_ids:
            return {}
        try:
            from imas_codex.graph import GraphClient

            with GraphClient() as gc:
                result = gc.query(
                    """
                    UNWIND $ids AS cid
                    MATCH (cc:CodeChunk {id: cid})
                    MATCH (cf:CodeFile)-[:HAS_EXAMPLE]->(:CodeExample)
                          -[:HAS_CHUNK]->(cc)
                    RETURN cid AS id, cf.score_composite AS score
                    """,
                    ids=chunk_ids,
                )
                return {
                    r["id"]: r["score"] for r in result if r.get("score") is not None
                }
        except Exception:
            return {}

    def refresh_from_graph(self, facility: str) -> None:
        """Refresh totals from graph database."""
        from imas_codex.discovery.code.parallel import get_code_discovery_stats

        stats = get_code_discovery_stats(facility, min_score=self.min_score)
        self.state.total = stats["total"]
        self.state.discovered = stats["discovered"]
        self.state.triaged_count = stats["triaged"]
        self.state.ingested = stats["ingested"]
        self.state.failed = stats["failed"]
        self.state.skipped = stats["skipped"]
        self.state.pending_triage = stats["pending_triage"]
        self.state.pending_enrich = stats["pending_enrich"]
        self.state.pending_score = stats["pending_score"]
        self.state.pending_ingest = stats["pending_ingest"]
        self.state.scored_count = stats["scored"]
        self.state.enriched_count = stats["enriched_count"]
        self.state.code_files = stats["total"]

        # Embed stats
        self.state.total_chunks = stats.get("total_chunks", 0)
        self.state.embedded_chunks = stats.get("embedded_chunks", 0)
        self.state.pending_embed = stats.get("pending_embed", 0)

        # Accumulated LLM cost from graph (source of truth across runs)
        self.state.accumulated_cost = stats.get("accumulated_cost", 0.0)

        # Accumulated wall-clock time from prior sessions
        from imas_codex.discovery.base.progress import get_accumulated_time

        self.state.accumulated_time = get_accumulated_time(self.state.facility, "code")

        self._refresh()

    def tick(self) -> None:
        """Drain streaming queues for smooth display."""
        updated = False

        next_scan = self.state.scan_queue.pop()
        if next_scan:
            self.state.current_scan = next_scan
            updated = True
        elif self.state.scan_queue.is_stale() and self.state.current_scan is not None:
            self.state.current_scan = None
            updated = True

        next_triage = self.state.triage_queue.pop()
        if next_triage:
            self.state.current_triage = next_triage
            updated = True
        elif (
            self.state.triage_queue.is_stale() and self.state.current_triage is not None
        ):
            self.state.current_triage = None
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

        next_score = self.state.score_queue.pop()
        if next_score:
            self.state.current_score = next_score
            updated = True
        elif self.state.score_queue.is_stale() and self.state.current_score is not None:
            self.state.current_score = None
            updated = True

        next_ingest = self.state.ingest_queue.pop()
        if next_ingest:
            self.state.current_ingest = next_ingest
            updated = True
        elif (
            self.state.ingest_queue.is_stale() and self.state.current_ingest is not None
        ):
            self.state.current_ingest = None
            updated = True

        next_embed = self.state.embed_queue.pop()
        if next_embed:
            self.state.current_embed = next_embed
            updated = True
        elif self.state.embed_queue.is_stale() and self.state.current_embed is not None:
            self.state.current_embed = None
            updated = True

        if updated:
            self._refresh()

    def print_summary(self) -> None:
        """Print final discovery summary."""
        self.console.print()
        self.console.print(
            Panel(
                self._build_summary(),
                title=f"{self.state.facility.upper()} Code Discovery Complete",
                border_style="green",
                width=self.width,
            )
        )

    def _build_summary(self) -> Text:
        """Build final summary text."""
        summary = Text()

        # SCAN stats
        summary.append("  SCAN     ", style="bold blue")
        summary.append(f"scanned={format_count(self.state.run_scanned)}", style="blue")
        if self.state.scan_rate:
            summary.append(f"  {self.state.scan_rate:.1f}/s", style="dim")
        summary.append("\n")

        # TRIAGE stats
        summary.append("  TRIAGE   ", style="bold green")
        summary.append(f"triaged={format_count(self.state.run_triaged)}", style="green")
        if self.state._run_triage_cost > 0:
            summary.append(f"  cost=${self.state._run_triage_cost:.3f}", style="yellow")
        if self.state.triage_rate:
            summary.append(f"  {self.state.triage_rate:.1f}/s", style="dim")
        summary.append("\n")

        # ENRICH stats
        summary.append("  ENRICH   ", style="bold magenta")
        summary.append(
            f"enriched={format_count(self.state.run_enriched)}", style="magenta"
        )
        if self.state.enrich_rate:
            summary.append(f"  {self.state.enrich_rate:.1f}/s", style="dim")
        summary.append("\n")

        # SCORE stats
        summary.append("  SCORE    ", style="bold cyan")
        summary.append(f"scored={format_count(self.state.run_scored)}", style="cyan")
        if self.state.run_skipped > 0:
            summary.append(
                f"  skipped={format_count(self.state.run_skipped)}", style="yellow"
            )
        summary.append(f"  cost=${self.state._run_score_cost:.3f}", style="yellow")
        if self.state.score_rate:
            summary.append(f"  {self.state.score_rate:.1f}/s", style="dim")
        summary.append("\n")

        # INGEST stats
        summary.append("  INGEST   ", style="bold yellow")
        summary.append(
            f"code={format_count(self.state.run_code_ingested)}", style="yellow"
        )
        ingest_rate = self.state.ingest_rate
        if ingest_rate:
            summary.append(f"  {ingest_rate:.1f}/s", style="dim")
        summary.append("\n")

        # EMBED stats
        summary.append("  EMBED    ", style="bold white")
        summary.append(
            f"embedded={format_count(self.state.run_embedded)}", style="white"
        )
        if self.state.pending_embed > 0:
            summary.append(
                f"  pending={format_count(self.state.pending_embed)}", style="yellow"
            )
        if self.state.embed_rate:
            summary.append(f"  {self.state.embed_rate:.1f}/s", style="dim")
        summary.append("\n")

        # USAGE stats
        summary.append("  USAGE    ", style="bold white")
        summary.append(f"time={format_time(self.state.elapsed)}", style="white")
        summary.append(f"  cost=${self.state.accumulated_cost:.2f}", style="yellow")
        if self.state.run_cost > 0:
            summary.append(f"  session=${self.state.run_cost:.2f}", style="dim")

        return summary
