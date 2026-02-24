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
- CODE: Fetch, chunk, embed high-scoring code files
- DOCS: Ingest documents, notebooks, configs

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
    clip_path,
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


@dataclass
class CodeItem:
    """Current code ingestion activity."""

    path: str
    chunks: int = 0
    language: str = ""


@dataclass
class DocsItem:
    """Current docs ingestion activity."""

    path: str
    file_type: str = ""  # document, notebook, config


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
    code_files: int = 0
    document_files: int = 0
    notebook_files: int = 0
    config_files: int = 0

    # This run stats
    run_scanned: int = 0
    run_scored: int = 0
    run_code_ingested: int = 0
    run_docs_ingested: int = 0
    _run_score_cost: float = 0.0
    scan_rate: float | None = None
    score_rate: float | None = None
    code_rate: float | None = None
    docs_rate: float | None = None

    # Accumulated facility cost (from graph)
    accumulated_cost: float = 0.0

    # Current items
    current_scan: ScanItem | None = None
    current_score: ScoreItem | None = None
    current_code: CodeItem | None = None
    current_docs: DocsItem | None = None
    scan_processing: bool = False
    score_processing: bool = False
    code_processing: bool = False
    docs_processing: bool = False

    # Streaming queues
    scan_queue: StreamQueue = field(default_factory=StreamQueue)
    score_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.0, min_display_time=0.4
        )
    )
    code_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.0, min_display_time=0.4
        )
    )
    docs_queue: StreamQueue = field(
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

        # Code pipeline ETA
        if self.code_rate and self.code_rate > 0 and self.pending_ingest > 0:
            etas.append(self.pending_ingest / self.code_rate)

        return max(etas) if etas else None


# =============================================================================
# Main Display Class
# =============================================================================


class FileProgressDisplay(BaseProgressDisplay):
    """Clean progress display for parallel file discovery.

    Extends ``BaseProgressDisplay`` for the file discovery pipeline
    (SCAN → SCORE → CODE → DOCS).  Inherits header, servers, worker
    tracking, and live-display lifecycle from the base class.
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

        Stages: SCAN → SCORE → CODE → DOCS
        """
        content_width = self.width - 6

        # --- Compute progress data ---

        # SCAN: files discovered / total from paths
        scan_total = max(self.state.total, 1)

        # SCORE: files scored / total needing scoring
        score_total = max(self.state.pending_score + self.state.run_scored, 1)

        # CODE: code files ingested / total code files
        code_total = max(self.state.code_files, 1)

        # DOCS: non-code files ingested
        docs_total = max(
            self.state.document_files
            + self.state.notebook_files
            + self.state.config_files,
            1,
        )

        # Score cost for display
        score_cost = (
            self.state._run_score_cost if self.state._run_score_cost > 0 else None
        )

        # Worker counts per group
        scan_count, scan_ann = self._count_group_workers("scan")
        score_count, score_ann = self._count_group_workers("triage")
        code_count, code_ann = self._count_group_workers("code")
        docs_count, docs_ann = self._count_group_workers("docs")

        # --- Build activity data ---

        scan = self.state.current_scan
        score = self.state.current_score
        code = self.state.current_code
        docs = self.state.current_docs

        # Worker completion detection
        scan_complete = self._worker_complete("scan") and not scan
        score_complete = self._worker_complete("triage") and not score
        code_complete = self._worker_complete("code") and not code
        docs_complete = self._worker_complete("docs") and not docs

        # SCAN activity
        scan_text = ""
        scan_detail: list[tuple[str, str]] | None = None
        if scan:
            scan_text = clip_path(scan.path, content_width - 10)
            if scan.files_found > 0:
                scan_detail = [(f"{scan.files_found} files found", "cyan")]

        # SCORE activity
        score_text = ""
        score_detail: list[tuple[str, str]] | None = None
        if score:
            score_text = clip_path(score.path, content_width - 10)
            parts: list[tuple[str, str]] = []
            if score.score is not None:
                style = (
                    "bold green"
                    if score.score >= 0.7
                    else "yellow"
                    if score.score >= 0.4
                    else "red"
                )
                parts.append((f"{score.score:.2f}", style))
            if score.category:
                parts.append((f"  {score.category}", "dim"))
            score_detail = parts or None

        # CODE activity
        code_text = ""
        code_detail: list[tuple[str, str]] | None = None
        if code:
            code_text = clip_path(code.path, content_width - 10)
            parts = []
            if code.chunks > 0:
                parts.append((f"{code.chunks} chunks", "cyan"))
            if code.language:
                parts.append((f"  [{code.language}]", "green dim"))
            code_detail = parts or None

        # DOCS activity
        docs_text = ""
        docs_detail: list[tuple[str, str]] | None = None
        if docs:
            docs_text = clip_path(docs.path, content_width - 10)
            if docs.file_type:
                docs_detail = [(docs.file_type, "dim")]

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
                completed=self.state.run_scored,
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
                name="CODE",
                style="bold magenta",
                completed=self.state.run_code_ingested,
                total=code_total,
                rate=self.state.code_rate,
                disabled=self.state.scan_only or self.state.score_only,
                primary_text=code_text,
                detail_parts=code_detail,
                is_processing=self.state.code_processing,
                is_complete=code_complete,
                worker_count=code_count,
                worker_annotation=code_ann,
            ),
            PipelineRowConfig(
                name="DOCS",
                style="bold yellow",
                completed=self.state.run_docs_ingested,
                total=docs_total,
                rate=self.state.docs_rate,
                disabled=self.state.scan_only or self.state.score_only,
                primary_text=docs_text,
                detail_parts=docs_detail,
                is_processing=self.state.docs_processing,
                is_complete=docs_complete,
                worker_count=docs_count,
                worker_annotation=docs_ann,
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
                    category=r.get("file_category", ""),
                )
                for r in results
            ]
            max_rate = 2.0
            display_rate = min(stats.rate, max_rate) if stats.rate else 0.5
            self.state.score_queue.add(items, display_rate)

        self._refresh()

    def update_code(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update code ingestion worker state."""
        self.state.run_code_ingested = stats.processed
        self.state.code_rate = stats.rate

        if "waiting" in message.lower():
            self.state.code_processing = False
            self._refresh()
            return
        elif "ingesting" in message.lower() or "fetching" in message.lower():
            self.state.code_processing = True
        else:
            self.state.code_processing = False

        if results:
            items = [
                CodeItem(
                    path=r.get("path", ""),
                    chunks=r.get("chunks", 0),
                    language=r.get("language", ""),
                )
                for r in results
            ]
            max_rate = 2.0
            display_rate = min(stats.rate, max_rate) if stats.rate else 0.5
            self.state.code_queue.add(items, display_rate)

        self._refresh()

    def update_docs(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update docs ingestion worker state."""
        self.state.run_docs_ingested = stats.processed
        self.state.docs_rate = stats.rate

        if "waiting" in message.lower():
            self.state.docs_processing = False
            self._refresh()
            return
        elif "ingesting" in message.lower():
            self.state.docs_processing = True
        else:
            self.state.docs_processing = False

        if results:
            items = [
                DocsItem(
                    path=r.get("path", ""),
                    file_type=r.get("file_type", ""),
                )
                for r in results
            ]
            max_rate = 2.0
            display_rate = min(stats.rate, max_rate) if stats.rate else 0.5
            self.state.docs_queue.add(items, display_rate)

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

        next_code = self.state.code_queue.pop()
        if next_code:
            self.state.current_code = next_code
            updated = True
        elif self.state.code_queue.is_stale() and self.state.current_code is not None:
            self.state.current_code = None
            updated = True

        next_docs = self.state.docs_queue.pop()
        if next_docs:
            self.state.current_docs = next_docs
            updated = True
        elif self.state.docs_queue.is_stale() and self.state.current_docs is not None:
            self.state.current_docs = None
            updated = True

        if updated:
            self._refresh()

    def print_summary(self) -> None:
        """Print final discovery summary."""
        self.console.print()
        self.console.print(
            Panel(
                self._build_summary(),
                title=f"{self.state.facility.upper()} File Discovery Complete",
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
        summary.append(f"  cost=${self.state.run_cost:.3f}", style="yellow")
        if self.state.score_rate:
            summary.append(f"  {self.state.score_rate:.1f}/s", style="dim")
        summary.append("\n")

        # CODE stats
        summary.append("  CODE     ", style="bold magenta")
        summary.append(f"ingested={self.state.run_code_ingested:,}", style="magenta")
        if self.state.code_rate:
            summary.append(f"  {self.state.code_rate:.1f}/s", style="dim")
        summary.append("\n")

        # DOCS stats
        summary.append("  DOCS     ", style="bold yellow")
        summary.append(f"ingested={self.state.run_docs_ingested:,}", style="yellow")
        if self.state.docs_rate:
            summary.append(f"  {self.state.docs_rate:.1f}/s", style="dim")
        summary.append("\n")

        # USAGE stats
        summary.append("  USAGE    ", style="bold cyan")
        summary.append(f"time={format_time(self.state.elapsed)}", style="white")
        summary.append(f"  cost=${self.state.accumulated_cost:.2f}", style="yellow")
        if self.state.run_cost > 0:
            summary.append(f"  session=${self.state.run_cost:.2f}", style="dim")

        return summary
