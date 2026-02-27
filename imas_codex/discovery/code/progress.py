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
- SCORE: LLM batch scoring of discovered SourceFiles
- ENRICH: rg pattern matching on scored files
- INGEST: Fetch, chunk, embed (code + docs combined)

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
    clip_path,
    clip_text,
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


@dataclass
class ScoreItem:
    """Current score activity."""

    path: str
    score: float | None = None
    category: str = ""  # code, document, notebook, config
    description: str = ""  # LLM reasoning about what the file contains
    skipped: bool = False  # LLM says skip this file


@dataclass
class EnrichItem:
    """Current enrich activity."""

    path: str
    patterns: int = 0  # total pattern matches


@dataclass
class IngestItem:
    """Current ingestion activity (code or docs)."""

    path: str
    language: str = ""
    file_type: str = ""  # code, document, notebook, config


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
    ingested: int = 0
    failed: int = 0
    skipped: int = 0
    pending_score: int = 0
    pending_ingest: int = 0
    scored_count: int = 0
    enriched_count: int = 0
    code_files: int = 0
    document_files: int = 0
    notebook_files: int = 0
    config_files: int = 0

    # This run stats
    run_scanned: int = 0
    run_scored: int = 0
    run_skipped: int = 0
    run_enriched: int = 0
    run_code_ingested: int = 0
    run_docs_ingested: int = 0
    _run_score_cost: float = 0.0
    scan_rate: float | None = None
    score_rate: float | None = None
    enrich_rate: float | None = None
    code_rate: float | None = None
    docs_rate: float | None = None

    # Accumulated facility cost (from graph)
    accumulated_cost: float = 0.0

    # Current items
    current_scan: ScanItem | None = None
    current_score: ScoreItem | None = None
    current_enrich: EnrichItem | None = None
    current_ingest: IngestItem | None = None
    scan_processing: bool = False
    score_processing: bool = False
    enrich_processing: bool = False
    ingest_processing: bool = False

    # Streaming queues
    scan_queue: StreamQueue = field(default_factory=StreamQueue)
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

    # Tracking
    start_time: float = field(default_factory=time.time)

    @property
    def run_ingest_total(self) -> int:
        """Combined code + docs ingested this run."""
        return self.run_code_ingested + self.run_docs_ingested

    @property
    def ingest_rate(self) -> float | None:
        """Combined code + docs rate."""
        rates = [r for r in [self.code_rate, self.docs_rate] if r]
        return sum(rates) if rates else None

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def run_cost(self) -> float:
        return self._run_score_cost

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
        """Estimated time to termination.

        Returns the maximum ETA across all active worker pipelines.
        """
        etas: list[float] = []

        # Cost-based ETA
        if self.run_cost > 0 and self.cost_limit > 0 and self.elapsed > 0:
            cost_rate = self.run_cost / self.elapsed
            if cost_rate > 0:
                remaining_budget = self.cost_limit - self.run_cost
                if remaining_budget > 0:
                    etas.append(remaining_budget / cost_rate)

        # Score pipeline ETA
        if self.score_rate and self.score_rate > 0 and self.pending_score > 0:
            etas.append(self.pending_score / self.score_rate)

        # Ingest pipeline ETA
        ingest_rate = self.ingest_rate
        if ingest_rate and ingest_rate > 0 and self.pending_ingest > 0:
            etas.append(self.pending_ingest / ingest_rate)

        return max(etas) if etas else None


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
    ) -> None:
        super().__init__(
            facility=facility,
            cost_limit=cost_limit,
            console=console,
            focus=focus,
            title_suffix="File Discovery",
        )
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

        Stages: SCAN -> SCORE -> ENRICH -> INGEST
        """
        content_width = self.width - 6

        # --- Compute progress data ---

        # SCAN: files discovered / total from paths
        scan_total = max(self.state.total, 1)

        # SCORE: files scored (graph) / total needing scoring
        score_completed = self.state.scored_count
        score_total = max(self.state.total, 1)

        # ENRICH: enriched (graph) / scored (all scored files need enriching)
        enrich_completed = self.state.enriched_count
        enrich_total = max(self.state.scored_count, 1)

        # INGEST: ingested (graph) / total needing ingestion
        ingest_completed = self.state.ingested
        ingest_total = max(self.state.ingested + self.state.pending_ingest, 1)

        # Score cost for display
        score_cost = (
            self.state._run_score_cost if self.state._run_score_cost > 0 else None
        )

        # Worker counts per group
        scan_count, scan_ann = self._count_group_workers("scan")
        score_count, score_ann = self._count_group_workers("triage")
        enrich_count, enrich_ann = self._count_group_workers("enrich")
        # Combine code + docs workers for ingest display
        code_count, code_ann = self._count_group_workers("code")
        docs_count, docs_ann = self._count_group_workers("docs")
        ingest_count = code_count + docs_count
        ingest_ann = code_ann or docs_ann

        # --- Build activity data ---

        scan = self.state.current_scan
        score = self.state.current_score
        enrich = self.state.current_enrich
        ingest = self.state.current_ingest

        # Worker completion detection
        scan_complete = self._worker_complete("scan") and not scan
        score_complete = self._worker_complete("triage") and not score
        enrich_complete = self._worker_complete("enrich") and not enrich
        code_complete = self._worker_complete("code")
        docs_complete = self._worker_complete("docs")
        ingest_complete = code_complete and docs_complete and not ingest

        # SCAN activity
        scan_text = ""
        scan_detail: list[tuple[str, str]] | None = None
        if scan:
            scan_text = clip_path(scan.path, content_width - 10)
            if scan.files_found > 0:
                scan_detail = [(f"{scan.files_found} files found", "cyan")]

        # SCORE activity (with [category] + description like paths shows purpose)
        score_text = ""
        score_detail: list[tuple[str, str]] | None = None
        if score:
            score_text = clip_path(score.path, content_width - 10)
            parts: list[tuple[str, str]] = []
            if score.skipped:
                parts.append(("skip", "yellow"))
                if score.category:
                    parts.append((f"  [{score.category}]", "dim"))
                if score.description:
                    desc = clean_text(score.description)
                    parts.append(
                        (
                            f"  {clip_text(desc, min(content_width - 20, 55))}",
                            "italic dim",
                        )
                    )
            elif score.score is not None:
                style = (
                    "bold green"
                    if score.score >= 0.7
                    else "yellow"
                    if score.score >= 0.4
                    else "red"
                )
                parts.append((f"{score.score:.2f}", style))
                if score.category:
                    parts.append((f"  [{score.category}]", "cyan dim"))

                desc_width = content_width - 20
                if score.description:
                    desc = clean_text(score.description)
                    parts.append(
                        (
                            f"  {clip_text(desc, min(desc_width, 55))}",
                            "italic dim",
                        )
                    )
            elif score.category:
                parts.append((f"[{score.category}]", "dim"))
            score_detail = parts or None

        # ENRICH activity
        enrich_text = ""
        enrich_detail: list[tuple[str, str]] | None = None
        if enrich:
            enrich_text = clip_path(enrich.path, content_width - 10)
            if enrich.patterns > 0:
                enrich_detail = [(f"{enrich.patterns} patterns", "cyan")]

        # INGEST activity (combined code + docs)
        ingest_text = ""
        ingest_detail: list[tuple[str, str]] | None = None
        if ingest:
            ingest_text = clip_path(ingest.path, content_width - 10)
            parts = []
            if ingest.language:
                parts.append((f"[{ingest.language}]", "green dim"))
            elif ingest.file_type:
                parts.append((f"[{ingest.file_type}]", "dim"))
            ingest_detail = parts or None

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
                detail_parts=scan_detail,
                is_processing=self.state.scan_processing,
                is_complete=scan_complete,
                worker_count=scan_count,
                worker_annotation=scan_ann,
            ),
            PipelineRowConfig(
                name="SCORE",
                style="bold green",
                completed=score_completed,
                total=score_total,
                rate=self.state.score_rate,
                cost=score_cost,
                disabled=self.state.scan_only,
                primary_text=score_text,
                detail_parts=score_detail,
                is_processing=self.state.score_processing,
                is_complete=score_complete,
                worker_count=score_count,
                worker_annotation=score_ann,
            ),
            PipelineRowConfig(
                name="ENRICH",
                style="bold white",
                completed=enrich_completed,
                total=enrich_total,
                rate=self.state.enrich_rate,
                disabled=self.state.scan_only or self.state.score_only,
                primary_text=enrich_text,
                detail_parts=enrich_detail,
                is_processing=self.state.enrich_processing,
                is_complete=enrich_complete,
                worker_count=enrich_count,
                worker_annotation=enrich_ann,
            ),
            PipelineRowConfig(
                name="INGEST",
                style="bold magenta",
                completed=ingest_completed,
                total=ingest_total,
                rate=self.state.ingest_rate,
                disabled=self.state.scan_only or self.state.score_only,
                primary_text=ingest_text,
                detail_parts=ingest_detail,
                is_processing=self.state.ingest_processing,
                is_complete=ingest_complete,
                worker_count=ingest_count,
                worker_annotation=ingest_ann,
            ),
        ]
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges using unified builder."""
        # Compute ETC
        total_cost = self.state.accumulated_cost
        etc = total_cost
        cpf = self.state.cost_per_file
        if cpf and cpf > 0 and self.state.pending_score > 0:
            etc = total_cost + (cpf * self.state.pending_score)

        # Build stats
        stats: list[tuple[str, str, str]] = [
            ("total", str(self.state.total), "blue"),
            ("ingested", str(self.state.ingested), "green"),
        ]
        if self.state.failed > 0:
            stats.append(("failed", str(self.state.failed), "red"))
        if self.state.skipped > 0:
            stats.append(("skipped", str(self.state.skipped), "dim"))

        # File type breakdown
        type_parts: list[tuple[str, str, str]] = []
        if self.state.code_files > 0:
            type_parts.append(("code", str(self.state.code_files), "cyan"))
        if self.state.document_files > 0:
            type_parts.append(("docs", str(self.state.document_files), "magenta"))
        if self.state.notebook_files > 0:
            type_parts.append(("notebooks", str(self.state.notebook_files), "yellow"))

        config = ResourceConfig(
            elapsed=self.state.elapsed,
            eta=None if self.state.scan_only else self.state.eta_seconds,
            run_cost=self.state.run_cost,
            cost_limit=self.state.cost_limit,
            accumulated_cost=self.state.accumulated_cost,
            etc=etc if etc > total_cost else None,
            scan_only=self.state.scan_only,
            limit_reason=self.state.limit_reason,
            stats=stats + type_parts,
            pending=[
                ("score", self.state.pending_score),
                ("ingest", self.state.pending_ingest),
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
        self.state.scan_rate = stats.rate

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
                )
                for r in results
            ]
            max_rate = 2.0
            display_rate = min(stats.rate, max_rate) if stats.rate else 1.0
            self.state.scan_queue.add(items, display_rate)

        self._refresh()

    def update_score(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update score worker state."""
        self.state.run_scored = stats.processed
        self.state.score_rate = stats.rate
        self.state._run_score_cost = stats.cost

        if "waiting" in message.lower():
            self.state.score_processing = False
            self._refresh()
            return
        elif "scoring" in message.lower():
            self.state.score_processing = True
        else:
            self.state.score_processing = False

        if results:
            items = [
                ScoreItem(
                    path=r.get("path", ""),
                    score=r.get("score"),
                    category=r.get("category", r.get("file_category", "")),
                    description=r.get("description", ""),
                    skipped=r.get("skipped", False),
                )
                for r in results
            ]
            # Track skipped files
            self.state.run_skipped += sum(1 for r in results if r.get("skipped"))
            max_rate = 2.0
            display_rate = min(stats.rate, max_rate) if stats.rate else 0.5
            self.state.score_queue.add(items, display_rate)

        self._refresh()

    def update_enrich(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update enrich worker state."""
        self.state.run_enriched = stats.processed
        self.state.enrich_rate = stats.rate

        if "waiting" in message.lower() or "idle" in message.lower():
            self.state.enrich_processing = False
        elif "enriching" in message.lower() or "enriched" in message.lower():
            self.state.enrich_processing = True
        else:
            self.state.enrich_processing = False

        if results:
            items = [
                EnrichItem(
                    path=r.get("path", ""),
                    patterns=r.get("patterns", 0),
                )
                for r in results
            ]
            max_rate = 2.5
            display_rate = min(stats.rate, max_rate) if stats.rate else 0.5
            self.state.enrich_queue.add(items, display_rate)

        self._refresh()

    def update_code(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update code ingestion worker state (feeds into INGEST row)."""
        self.state.run_code_ingested = stats.processed
        self.state.code_rate = stats.rate

        if "waiting" in message.lower():
            self._refresh()
            return
        elif "ingesting" in message.lower() or "fetching" in message.lower():
            self.state.ingest_processing = True

        if results:
            items = [
                IngestItem(
                    path=r.get("path", ""),
                    language=r.get("language", ""),
                    file_type="code",
                )
                for r in results
            ]
            max_rate = 2.0
            display_rate = min(stats.rate, max_rate) if stats.rate else 0.5
            self.state.ingest_queue.add(items, display_rate)

        self._refresh()

    def update_docs(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update docs ingestion worker state (feeds into INGEST row)."""
        self.state.run_docs_ingested = stats.processed
        self.state.docs_rate = stats.rate

        if "waiting" in message.lower():
            self._refresh()
            return
        elif "ingesting" in message.lower():
            self.state.ingest_processing = True

        if results:
            items = [
                IngestItem(
                    path=r.get("path", ""),
                    file_type=r.get("file_type", "document"),
                )
                for r in results
            ]
            max_rate = 2.0
            display_rate = min(stats.rate, max_rate) if stats.rate else 0.5
            self.state.ingest_queue.add(items, display_rate)

        self._refresh()

    def refresh_from_graph(self, facility: str) -> None:
        """Refresh totals from graph database."""
        from imas_codex.discovery.files.parallel import get_file_discovery_stats

        stats = get_file_discovery_stats(facility)
        self.state.total = stats["total"]
        self.state.discovered = stats["discovered"]
        self.state.ingested = stats["ingested"]
        self.state.failed = stats["failed"]
        self.state.skipped = stats["skipped"]
        self.state.pending_score = stats["pending_score"]
        self.state.pending_ingest = stats["pending_ingest"]
        self.state.scored_count = stats["scored_count"]
        self.state.enriched_count = stats["enriched_count"]
        self.state.code_files = stats["code_files"]
        self.state.document_files = stats["document_files"]
        self.state.notebook_files = stats["notebook_files"]
        self.state.config_files = stats["config_files"]
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

        next_ingest = self.state.ingest_queue.pop()
        if next_ingest:
            self.state.current_ingest = next_ingest
            updated = True
        elif (
            self.state.ingest_queue.is_stale() and self.state.current_ingest is not None
        ):
            self.state.current_ingest = None
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
        summary.append(f"scanned={self.state.run_scanned:,}", style="blue")
        if self.state.scan_rate:
            summary.append(f"  {self.state.scan_rate:.1f}/s", style="dim")
        summary.append("\n")

        # SCORE stats
        summary.append("  SCORE    ", style="bold green")
        summary.append(f"scored={self.state.run_scored:,}", style="green")
        if self.state.run_skipped > 0:
            summary.append(f"  skipped={self.state.run_skipped:,}", style="yellow")
        summary.append(f"  cost=${self.state._run_score_cost:.3f}", style="yellow")
        if self.state.score_rate:
            summary.append(f"  {self.state.score_rate:.1f}/s", style="dim")
        summary.append("\n")

        # ENRICH stats
        summary.append("  ENRICH   ", style="bold white")
        summary.append(f"enriched={self.state.run_enriched:,}", style="white")
        if self.state.enrich_rate:
            summary.append(f"  {self.state.enrich_rate:.1f}/s", style="dim")
        summary.append("\n")

        # INGEST stats (combined code + docs)
        summary.append("  INGEST   ", style="bold magenta")
        summary.append(f"code={self.state.run_code_ingested:,}", style="magenta")
        summary.append(f"  docs={self.state.run_docs_ingested:,}", style="yellow")
        ingest_rate = self.state.ingest_rate
        if ingest_rate:
            summary.append(f"  {ingest_rate:.1f}/s", style="dim")
        summary.append("\n")

        # USAGE stats
        summary.append("  USAGE    ", style="bold cyan")
        summary.append(f"time={format_time(self.state.elapsed)}", style="white")
        summary.append(f"  cost=${self.state.accumulated_cost:.2f}", style="yellow")
        if self.state.run_cost > 0:
            summary.append(f"  session=${self.state.run_cost:.2f}", style="dim")

        return summary
