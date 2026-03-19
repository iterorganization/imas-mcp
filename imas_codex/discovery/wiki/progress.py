"""
Progress display for parallel wiki discovery.

Design principles (matching paths parallel_progress.py):
- Minimal visual clutter (no emojis, no stopwatch icons)
- Clear hierarchy: Target → Pipeline → Resources
- Unified per-stage blocks: progress bar + current activity + detail
- Per-stage cost and worker count annotations
- Gradient progress bars with percentage
- Resource gauges for time and cost budgets
- Live embedding source indicator (shows remote/openrouter/local)

Display layout: SERVERS → PIPELINE → RESOURCES
  SCORE:  Content-aware LLM scoring (fetches content, scores with LLM)
  INGEST: Chunk and embed high-value pages (score >= 0.5)
  FILE:   Score and embed wiki file attachments (PDFs, CSVs, etc.)
  IMAGE:  VLM captioning and scoring of wiki images

Progress is tracked against total pages in graph, not just this session.
ETA/ETC metrics calculated like paths discovery.

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
    LABEL_WIDTH,
    BaseProgressDisplay,
    PipelineRowConfig,
    ResourceConfig,
    StreamQueue,
    build_pipeline_section,
    build_resource_section,
    clean_text,
    clip_text,
    compute_projected_etc,
    format_count,
    format_time,
)
from imas_codex.discovery.base.supervision import WorkerState
from imas_codex.embeddings import get_embedding_source
from imas_codex.embeddings.resilience import get_embed_status

if TYPE_CHECKING:
    from imas_codex.discovery.base.progress import WorkerStats


# =============================================================================
# Display Items
# =============================================================================


@dataclass
class ScoreItem:
    """Current score activity."""

    title: str
    score_composite: float | None = None
    physics_domain: str | None = None  # Physics domain if detected
    description: str = ""  # LLM description of page value
    is_physics: bool = False
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class IngestItem:
    """Current ingest activity."""

    title: str
    score_composite: float | None = None  # Redisplay score on ingest
    description: str = ""  # Redisplay LLM description
    physics_domain: str | None = None  # Physics domain if detected
    chunk_count: int = 0


@dataclass
class DocsItem:
    """Current document activity (score or ingest)."""

    filename: str
    document_type: str
    score_composite: float | None = None
    physics_domain: str | None = None
    description: str = ""
    chunk_count: int = 0
    is_score: bool = False  # True if scoring, False if ingesting


@dataclass
class ImageItem:
    """Current image VLM activity."""

    image_id: str
    score_composite: float | None = None
    physics_domain: str | None = None
    description: str = ""
    purpose: str = ""
    source_url: str = ""
    page_title: str = ""
    page_image_count: int = 0  # Total images on this page
    page_image_index: int = 0  # Display index (Nth image shown from this page)

    @property
    def display_name(self) -> str:
        """Human-readable name: page_title:image N/M or fallback."""
        if self.page_title:
            label = self.page_title
            if self.page_image_count > 0 and self.page_image_index > 0:
                label += f":image {self.page_image_index}/{self.page_image_count}"
            else:
                label += ":image"
            return label
        # No page title — extract filename from URL
        if self.source_url:
            from urllib.parse import unquote, urlparse

            path = urlparse(self.source_url).path
            filename = unquote(path.rsplit("/", 1)[-1]) if path else ""
            if filename:
                return filename
        # Strip facility prefix from graph key (e.g., "jet:54dbfb4a" → "image 54dbfb4a")
        if ":" in self.image_id:
            short_hash = self.image_id.split(":", 1)[1][:12]
            return f"image {short_hash}"
        return self.image_id


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
    provider_budget_exhausted: bool = False  # API key credit limit hit (402)

    # Counts from graph (total pages for progress denominator)
    total_pages: int = 0  # All wiki pages in graph for this facility
    pages_scanned: int = 0  # Status = scanned (awaiting score)
    pages_scored: int = 0  # Status = scored (awaiting ingest or skipped)
    pages_ingested: int = 0  # Status = ingested (final state)
    pages_skipped: int = 0  # Skipped (low score or skip_reason)
    pages_failed: int = 0  # Status = failed

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

    # Document stats
    total_documents: int = 0  # All wiki documents in graph (DOC denominator)
    docs_ingested: int = 0
    docs_scored: int = 0
    documents_failed: int = 0
    documents_deferred: int = 0
    documents_skipped: int = 0
    run_docs: int = 0
    run_docs_scored: int = 0
    docs_rate: float | None = None
    document_score_rate: float | None = None
    _run_docs_score_cost: float = 0.0

    # Image stats
    images_scored: int = 0  # From graph
    run_images_scored: int = 0
    image_score_rate: float | None = None
    _run_image_score_cost: float = 0.0

    # Accumulated facility cost and time (from graph)
    accumulated_cost: float = 0.0
    accumulated_time: float = 0.0

    # Per-group accumulated costs (from graph, across all sessions)
    accumulated_page_cost: float = 0.0
    accumulated_document_cost: float = 0.0
    accumulated_image_cost: float = 0.0

    # Final rate snapshots — captured when workers reach done state
    # so the done row still shows the average active rate achieved
    _final_score_rate: float | None = None
    _final_ingest_rate: float | None = None
    _final_docs_rate: float | None = None
    _final_image_rate: float | None = None

    # Historic rates from graph timestamps (fallback when no live rate)
    _historic_score_rate: float | None = None
    _historic_ingest_rate: float | None = None

    # Multi-site tracking (for facilities with multiple wiki instances)
    current_site_name: str = ""
    current_site_index: int = 0
    total_sites: int = 1

    # Accumulated stats from previous sites in multi-site run.
    # These offsets are added to current-site counters for grand totals.
    _offset_scored: int = 0
    _offset_ingested: int = 0
    _offset_score_cost: float = 0.0
    _offset_ingest_cost: float = 0.0
    _offset_docs_score_cost: float = 0.0
    _offset_docs: int = 0
    _offset_docs_scored: int = 0
    _offset_images_scored: int = 0
    _offset_image_score_cost: float = 0.0

    # Current items (and their processing state)
    current_score: ScoreItem | None = None
    current_ingest: IngestItem | None = None
    current_docs: DocsItem | None = None
    current_image: ImageItem | None = None
    score_processing: bool = False
    ingest_processing: bool = False
    docs_processing: bool = False
    image_processing: bool = False

    # Streaming queues - adaptive rate based on worker speed
    score_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.5, min_display_time=0.4, stale_timeout=15.0
        )
    )
    ingest_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5, max_rate=2.5, min_display_time=0.4, stale_timeout=15.0
        )
    )
    document_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.3, max_rate=1.0, min_display_time=0.5, stale_timeout=15.0
        )
    )
    document_score_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.3, max_rate=1.0, min_display_time=0.5, stale_timeout=15.0
        )
    )
    image_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.3, max_rate=1.0, min_display_time=0.5, stale_timeout=15.0
        )
    )

    # Per-page image display counters (tracks Nth image shown per page)
    _page_image_seen: dict[str, int] = field(default_factory=dict)

    # Tracking
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    # Document pending counts (from graph refresh)
    pending_document_score: int = 0  # discovered documents awaiting scoring
    pending_document_ingest: int = 0  # scored documents awaiting ingestion

    # Image pending counts (from graph refresh)
    pending_image_score: int = 0  # ingested images awaiting VLM scoring

    @property
    def run_cost(self) -> float:
        """Total cost for entire run including previous sites."""
        return (
            self._offset_score_cost
            + self._run_score_cost
            + self._offset_ingest_cost
            + self._run_ingest_cost
            + self._offset_docs_score_cost
            + self._run_docs_score_cost
            + self._offset_image_score_cost
            + self._run_image_score_cost
        )

    @property
    def total_run_scored(self) -> int:
        """Total scored across all sites in this run."""
        return self._offset_scored + self.run_scored

    @property
    def total_run_ingested(self) -> int:
        """Total ingested across all sites in this run."""
        return self._offset_ingested + self.run_ingested

    @property
    def total_run_docs(self) -> int:
        """Total documents ingested across all sites."""
        return self._offset_docs + self.run_docs

    @property
    def total_run_docs_scored(self) -> int:
        """Total documents scored across all sites."""
        return self._offset_docs_scored + self.run_docs_scored

    @property
    def total_run_images_scored(self) -> int:
        """Total images scored across all sites."""
        return self._offset_images_scored + self.run_images_scored

    @property
    def total_score_cost(self) -> float:
        """Total page+document+image scoring cost across all sites."""
        return (
            self._offset_score_cost
            + self._run_score_cost
            + self._offset_docs_score_cost
            + self._run_docs_score_cost
            + self._offset_image_score_cost
            + self._run_image_score_cost
        )

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
        return self.total_run_scored >= self.page_limit

    @property
    def limit_reason(self) -> str | None:
        """Return which limit was reached, or None if no limit reached."""
        if self.provider_budget_exhausted:
            return "api budget"
        if self.cost_limit_reached:
            return "cost"
        if self.page_limit_reached:
            return "page"
        return None

    @property
    def cost_per_page(self) -> float | None:
        """Average LLM cost per scored page (page scoring only)."""
        total = self.total_run_scored
        cost = self._offset_score_cost + self._run_score_cost
        if total > 0:
            return cost / total
        return None

    @property
    def cost_per_document_score(self) -> float | None:
        """Average LLM cost per scored document."""
        total = self.total_run_docs_scored
        cost = self._offset_docs_score_cost + self._run_docs_score_cost
        if total > 0:
            return cost / total
        return None

    @property
    def cost_per_image_score(self) -> float | None:
        """Average VLM cost per scored image."""
        total = self.total_run_images_scored
        cost = self._offset_image_score_cost + self._run_image_score_cost
        if total > 0:
            return cost / total
        return None

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time to complete all pending work.

        Pure work-based ETA: max across parallel worker groups
        (bounded by the slowest pipeline).  Cost and page limits
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
                (self.pending_score, _agg(self.run_scored, self.score_rate)),
                (self.pending_ingest, _agg(self.run_ingested, self.ingest_rate)),
                (
                    self.pending_document_score,
                    _agg(self.run_docs_scored, self.document_score_rate),
                ),
                (self.pending_document_ingest, _agg(self.run_docs, self.docs_rate)),
                (
                    self.pending_image_score,
                    _agg(self.run_images_scored, self.image_score_rate),
                ),
            ]
        )


# =============================================================================
# Main Display Class
# =============================================================================


class WikiProgressDisplay(BaseProgressDisplay):
    """Clean progress display for parallel wiki discovery.

    Layout: HEADER → SERVERS → PIPELINE → RESOURCES

    Pipeline stages (unified progress + activity per stage):
    - SCORE: Content-aware LLM scoring (scanned → scored)
    - INGEST: Chunk and embed high-value pages (scored → ingested)
    - FILE: Score and embed wiki file attachments (PDFs, CSVs, etc.)
    - IMAGE: VLM captioning + scoring (ingested → captioned)

    Each stage shows a 3-line block:
      Line 1: progress bar + count + pct
      Line 2: score + domain + name … rate (right-aligned)
      Line 3: description … cost (right-aligned below rate)

    Progress is tracked against total pages in graph, not just this session.
    Uses common pipeline infrastructure from base.progress.
    """

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
        super().__init__(
            facility=facility,
            cost_limit=cost_limit,
            console=console,
            focus=focus,
            title_suffix="Wiki Discovery",
        )
        self.state = ProgressState(
            facility=facility,
            cost_limit=cost_limit,
            page_limit=page_limit,
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

    def _build_header(self) -> Text:
        """Build centered header with facility, focus, and multi-site indicator."""
        header = Text()

        title = f"{self.facility.upper()} {self._title_suffix}"
        mode = self._header_mode_label()
        if mode:
            title += f" ({mode})"

        # Multi-site indicator: show current site URL
        if self.state.total_sites > 1 and self.state.current_site_name:
            title += f"  {self.state.current_site_name}"

        header.append(title.center(self.width - 4), style="bold cyan")

        if self.focus and not self.state.scan_only:
            header.append("\n")
            focus_line = f"Focus: {self.focus}"
            header.append(focus_line.center(self.width - 4), style="italic dim")

        return header

    def _count_group_workers(self, group: str) -> tuple[int, str]:
        """Count workers in a group and build annotation string.

        Extends base version with budget exhaustion annotations for
        triage/docs/images groups.
        """
        wg = self.worker_group
        if not wg:
            return 0, ""

        count = 0
        backoff = 0
        failed = 0
        for _name, status in wg.workers.items():
            if status.group == group:
                count += 1
                if status.state == WorkerState.backoff:
                    backoff += 1
                elif status.state == WorkerState.crashed:
                    failed += 1

        parts: list[str] = []
        if backoff > 0:
            parts.append(f"{backoff} backoff")
        if failed > 0:
            parts.append(f"{failed} failed")

        # Budget annotation for triage/docs/images groups
        if count > 0 and (
            self.state.cost_limit_reached or self.state.provider_budget_exhausted
        ):
            budget_groups = {"triage", "docs", "images"}
            if group in budget_groups:
                all_stopped = all(
                    s.state == WorkerState.stopped
                    for _n, s in wg.workers.items()
                    if s.group == group
                )
                if all_stopped:
                    parts.append(
                        "api budget"
                        if self.state.provider_budget_exhausted
                        else "budget"
                    )

        return count, f"({', '.join(parts)})" if parts else ""

    def _score_detail(
        self,
        score_val: float | None,
        domain: str | None,
        description: str,
        *,
        is_physics: bool = False,
        content_width: int = 80,
    ) -> list[tuple[str, str]]:
        """Build detail parts for a score + domain + description line."""
        from rich.cells import cell_len

        parts: list[tuple[str, str]] = []
        used = 4  # indent
        if score_val is not None:
            style = (
                "bold green"
                if score_val >= 0.7
                else "yellow"
                if score_val >= 0.4
                else "red"
            )
            s = f"{score_val:.2f}"
            parts.append((f"{s}  ", style))
            used += cell_len(s) + 2
        d = domain or ("physics" if is_physics else "")
        if d:
            parts.append((f"{d}  ", "cyan"))
            used += cell_len(d) + 2
        if description:
            desc = clean_text(description)
            parts.append((clip_text(desc, max(10, content_width - used)), "italic dim"))
        return parts

    def _clip_title(self, title: str, max_len: int = 70) -> str:
        """Clip title to max length, preferring end truncation."""
        from imas_codex.discovery.base.progress import clip_text

        return clip_text(title, max_len)

    def _build_pipeline_section(self) -> Text:
        """Build the unified pipeline section (progress + activity merged).

        Each pipeline stage gets a 3-line block:
          Line 1: SCOREx4 ━━━━━━━━━━━━━━━━━━    2,238  29%
          Line 2:         0.00  general  Mailinglists            0.23/s
          Line 3:         Information regarding SPC...           $8.30

        Stages: SCORE → INGEST → FILE → IMAGE
        """
        monitor = self.service_monitor
        is_paused = monitor is not None and monitor.paused

        # --- Compute progress data ---

        # TRIAGE: LLM page scoring (all pages that have been triaged)
        score_total = self.state.total_pages or 1
        scored_pages = (
            self.state.pages_scored
            + self.state.pages_ingested
            + self.state.pages_skipped
            + self.state.pages_failed
        )
        triage_count, triage_ann = self._count_group_workers("triage")

        # PAGES: chunk + embed high-value pages
        ingest_total = max(self.state.pages_scored + self.state.pages_ingested, 1)
        pages_count, pages_ann = self._count_group_workers("pages")

        # DOCS: document scoring + ingestion
        art_scored = self.state.docs_scored
        art_ingested = self.state.docs_ingested
        art_terminal = (
            self.state.documents_failed
            + self.state.documents_deferred
            + self.state.documents_skipped
        )
        art_completed = art_scored + art_ingested + art_terminal
        art_total = self.state.total_documents or art_completed
        art_score_rate = self.state.document_score_rate
        art_ingest_rate = self.state.docs_rate
        art_rate = (
            sum(r for r in [art_score_rate, art_ingest_rate] if r and r > 0) or None
        )
        docs_cost = (
            self.state._run_docs_score_cost
            if self.state._run_docs_score_cost > 0
            else None
        )
        docs_count, docs_ann = self._count_group_workers("docs")

        # IMAGES: VLM captioning + scoring
        img_scored = self.state.images_scored
        img_pending = self.state.pending_image_score
        img_total = img_scored + img_pending
        images_cost = (
            self.state._run_image_score_cost
            if self.state._run_image_score_cost > 0
            else None
        )
        images_count, images_ann = self._count_group_workers("images")

        # --- Build activity data ---

        # TRIAGE activity
        score = self.state.current_score
        triage_text = ""
        triage_score_parts: list[tuple[str, str]] | None = None
        triage_domain = ""
        triage_desc = ""
        triage_desc_fallback = ""
        triage_complete = False
        triage_complete_label = "done"
        triage_at_100 = scored_pages >= score_total > 1
        # Snapshot rate when workers stop (regardless of completion)
        if (self._worker_complete("triage") or triage_at_100) and not score:
            if self.state._final_score_rate is None and self.state.score_rate:
                self.state._final_score_rate = self.state.score_rate
        # Only mark complete when progress is actually at 100%
        if triage_at_100 and not score:
            triage_complete = True
            if self.state.provider_budget_exhausted:
                triage_complete_label = "api budget"
            elif self.state.cost_limit_reached:
                triage_complete_label = "cost limit"
        if score:
            triage_text = score.title
            if score.skipped:
                reason = score.skip_reason[:40] if score.skip_reason else ""
                triage_desc_fallback = f"skipped: {reason}"
            else:
                if score.score_composite is not None:
                    triage_score_parts = [(f"{score.score_composite:.2f}", "bold blue")]
                triage_domain = score.physics_domain or (
                    "physics" if score.is_physics else ""
                )
                triage_desc = score.description

        # PAGES activity
        ingest = self.state.current_ingest
        pages_text = ""
        pages_score_parts: list[tuple[str, str]] | None = None
        pages_domain = ""
        pages_desc = ""
        pages_complete = False
        pages_at_100 = self.state.pages_ingested >= ingest_total > 1
        # Snapshot rate when workers stop (regardless of completion)
        if (self._worker_complete("pages") or pages_at_100) and not ingest:
            if self.state._final_ingest_rate is None and self.state.ingest_rate:
                self.state._final_ingest_rate = self.state.ingest_rate
        # Only mark complete when progress is actually at 100%
        if pages_at_100 and not ingest:
            pages_complete = True
        if ingest:
            pages_text = ingest.title
            if ingest.score_composite is not None:
                pages_score_parts = [(f"{ingest.score_composite:.2f}", "bold blue")]
            pages_domain = ingest.physics_domain or ""
            pages_desc = ingest.description

        # DOCS activity
        document = self.state.current_docs
        docs_text = ""
        docs_score_parts: list[tuple[str, str]] | None = None
        docs_domain = ""
        docs_desc = ""
        docs_desc_fallback = ""
        docs_complete = False
        docs_complete_label = "done"
        docs_at_100 = art_completed >= art_total > 1
        # Snapshot rate when workers stop (regardless of completion)
        if (self._worker_complete("docs") or docs_at_100) and not document:
            if self.state._final_docs_rate is None and art_rate:
                self.state._final_docs_rate = art_rate
        # Only mark complete when progress is actually at 100%
        if docs_at_100 and not document:
            docs_complete = True
            if self.state.provider_budget_exhausted:
                docs_complete_label = "api budget"
            elif self.state.cost_limit_reached:
                docs_complete_label = "cost limit"
        if document:
            display_name = document.filename
            if document.chunk_count > 0:
                display_name += f" ({document.chunk_count} chunks)"
            elif (
                document.score_composite is not None and document.score_composite < 0.5
            ):
                display_name += " (skipped)"
            docs_text = display_name
            if document.score_composite is not None:
                docs_score_parts = [(f"{document.score_composite:.2f}", "bold yellow")]
            docs_domain = document.physics_domain or ""
            docs_desc = document.description
            # Fallback for image-type documents with no description
            if not docs_desc and document.document_type:
                atype = document.document_type.lower()
                if atype in ("png", "jpg", "jpeg", "gif", "svg", "bmp", "tiff"):
                    docs_desc_fallback = f"describing {atype.upper()} image with VLM"
                elif atype in ("pdf",):
                    docs_desc_fallback = "extracting text from PDF"
                else:
                    docs_desc_fallback = f"processing {atype} document"

        # IMAGES activity
        image = self.state.current_image
        images_text = ""
        images_score_parts: list[tuple[str, str]] | None = None
        images_domain = ""
        images_desc = ""
        images_desc_fallback = ""
        images_complete = False
        images_complete_label = "done"
        images_at_100 = img_scored >= img_total > 1
        # Snapshot rate when workers stop (regardless of completion)
        if (self._worker_complete("images") or images_at_100) and not image:
            if self.state._final_image_rate is None and self.state.image_score_rate:
                self.state._final_image_rate = self.state.image_score_rate
        # Only mark complete when progress is actually at 100%
        if images_at_100 and not image:
            images_complete = True
            if self.state.provider_budget_exhausted:
                images_complete_label = "api budget"
            elif self.state.cost_limit_reached:
                images_complete_label = "cost limit"
        if image:
            images_text = image.display_name
            if image.score_composite is not None:
                images_score_parts = [(f"{image.score_composite:.2f}", "bold green")]
            images_domain = image.physics_domain or ""
            images_desc = image.description
            if not images_desc:
                images_desc_fallback = "describing image with VLM"

        # --- Build pipeline rows ---
        #
        # When a worker group is complete, show the accumulated cost from
        # the graph (all sessions) and the final average active rate rather
        # than the live EMA (which decays to None after workers stop).

        scan_only = self.state.scan_only

        # Rate priority: live EMA → final session snapshot → graph historic
        scan_rate = (
            self.state.score_rate
            or self.state._final_score_rate
            or self.state._historic_score_rate
        )
        scan_cost: float | None = (
            self.state.accumulated_page_cost
            if self.state.accumulated_page_cost > 0
            else None
        )

        page_rate = (
            self.state.ingest_rate
            or self.state._final_ingest_rate
            or self.state._historic_ingest_rate
        )

        # DOC rate/cost: prefer live rate, use accumulated graph cost
        docs_display_rate = art_rate or self.state._final_docs_rate
        docs_cost = (
            self.state.accumulated_document_cost
            if self.state.accumulated_document_cost > 0
            else None
        )

        # IMAGE rate/cost: prefer live rate, use accumulated graph cost
        images_display_rate = (
            self.state.image_score_rate or self.state._final_image_rate
        )
        images_cost = (
            self.state.accumulated_image_cost
            if self.state.accumulated_image_cost > 0
            else None
        )

        rows = [
            PipelineRowConfig(
                name="SCORE",
                style="bold blue",
                completed=scored_pages,
                total=score_total,
                rate=scan_rate,
                cost=scan_cost,
                disabled=scan_only,
                worker_count=triage_count,
                worker_annotation=triage_ann,
                primary_text=triage_text,
                score_parts=triage_score_parts,
                physics_domain=triage_domain,
                description=triage_desc,
                description_fallback=triage_desc_fallback,
                is_processing=self.state.score_processing,
                is_complete=triage_complete,
                complete_label=triage_complete_label,
                is_paused=is_paused,
                queue_size=(
                    len(self.state.score_queue)
                    if not self.state.score_queue.is_empty()
                    and not score
                    and not self.state.score_processing
                    else 0
                ),
            ),
            PipelineRowConfig(
                name="INGEST",
                style="bold magenta",
                completed=self.state.pages_ingested,
                total=ingest_total,
                rate=page_rate,
                disabled=scan_only,
                worker_count=pages_count,
                worker_annotation=pages_ann,
                primary_text=pages_text,
                score_parts=pages_score_parts,
                physics_domain=pages_domain,
                description=pages_desc,
                is_processing=self.state.ingest_processing,
                is_complete=pages_complete,
                is_paused=is_paused,
                queue_size=(
                    len(self.state.ingest_queue)
                    if not self.state.ingest_queue.is_empty()
                    and not ingest
                    and not self.state.ingest_processing
                    else 0
                ),
            ),
            PipelineRowConfig(
                name="FILE",
                style="bold yellow",
                completed=art_completed,
                total=max(art_total, 1),
                rate=docs_display_rate,
                cost=docs_cost,
                disabled=scan_only,
                worker_count=docs_count,
                worker_annotation=docs_ann,
                primary_text=docs_text,
                score_parts=docs_score_parts,
                physics_domain=docs_domain,
                description=docs_desc,
                description_fallback=docs_desc_fallback,
                is_processing=self.state.docs_processing,
                is_complete=docs_complete,
                complete_label=docs_complete_label,
                is_paused=is_paused,
                queue_size=(
                    (
                        len(self.state.document_queue)
                        + len(self.state.document_score_queue)
                    )
                    if (
                        not self.state.document_queue.is_empty()
                        or not self.state.document_score_queue.is_empty()
                    )
                    and not document
                    and not self.state.docs_processing
                    else 0
                ),
            ),
            PipelineRowConfig(
                name="IMAGE",
                style="bold green",
                completed=img_scored,
                total=max(img_total, 1),
                rate=images_display_rate,
                cost=images_cost,
                disabled=scan_only,
                worker_count=images_count,
                worker_annotation=images_ann,
                primary_text=images_text,
                score_parts=images_score_parts,
                physics_domain=images_domain,
                description=images_desc,
                description_fallback=images_desc_fallback,
                is_processing=self.state.image_processing,
                is_complete=images_complete,
                complete_label=images_complete_label,
                is_paused=is_paused,
                queue_size=(
                    len(self.state.image_queue)
                    if not self.state.image_queue.is_empty()
                    and not image
                    and not self.state.image_processing
                    else 0
                ),
            ),
        ]

        # Embedding source indicator appended to INGEST row
        embed_indicator = self._get_embed_indicator()
        if embed_indicator and pages_count > 0 and pages_ann:
            pages_ann_with_embed = (
                pages_ann[:-1] + f", {embed_indicator[0]})"
                if pages_ann
                else f"({embed_indicator[0]})"
            )
            rows[1].worker_annotation = pages_ann_with_embed
        elif embed_indicator and not pages_ann:
            rows[1].worker_annotation = f"({embed_indicator[0]})"

        return build_pipeline_section(rows, self.bar_width)

    def _get_embed_indicator(self) -> tuple[str, str] | None:
        """Get embedding source indicator label and style."""
        if self.service_monitor is not None:
            return None  # Service monitor handles embed display
        embed_health = get_embed_status()
        if embed_health != "ready":
            return (f"embed:{embed_health}", "red")
        embed_source = get_embedding_source()
        if embed_source.startswith("iter-"):
            return (f"embed:{embed_source}", "green")
        if embed_source == "remote":
            return ("embed:remote", "green")
        if embed_source == "openrouter":
            return ("embed:openrouter", "yellow")
        if embed_source == "local":
            return ("embed:local", "cyan")
        return (f"embed:{embed_source}", "dim")

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges using unified builder."""
        # Compute ETC — sum of per-worker cost projections
        total_facility_cost = self.state.accumulated_cost
        etc = compute_projected_etc(
            total_facility_cost,
            [
                (self.state.pending_score, self.state.cost_per_page),
                (self.state.pending_document_score, self.state.cost_per_document_score),
                (self.state.pending_image_score, self.state.cost_per_image_score),
            ],
        )

        # Build stats — show pipeline outcome counts.
        # The SCORE progress bar already shows total scored, so STATS
        # focuses on what happened after scoring.
        scored_total = (
            self.state.pages_scored
            + self.state.pages_ingested
            + self.state.pages_skipped
            + self.state.pages_failed
        )
        stats: list[tuple[str, str, str]] = [
            ("scored", format_count(scored_total), "blue"),
            ("ingested", format_count(self.state.pages_ingested), "magenta"),
            ("skipped", format_count(self.state.pages_skipped), "yellow"),
        ]

        # Pending work — only show categories with active workers
        pending_parts: list[tuple[str, int]] = []
        if self.state.pending_score > 0 or self.state.pending_document_score > 0:
            pending_parts.append(
                (
                    "score",
                    self.state.pending_score + self.state.pending_document_score,
                )
            )
        if self.state.pending_ingest > 0 or self.state.pending_document_ingest > 0:
            pending_parts.append(
                (
                    "ingest",
                    self.state.pending_ingest + self.state.pending_document_ingest,
                )
            )
        if self.state.pending_image_score > 0:
            pending_parts.append(("image", self.state.pending_image_score))

        # Determine limit reason
        limit_reason = self.state.limit_reason

        config = ResourceConfig(
            elapsed=self.state.elapsed,
            eta=None if self.state.scan_only else self.state.eta_seconds,
            accumulated_time=self.state.accumulated_time,
            run_cost=self.state.run_cost,
            cost_limit=self.state.cost_limit,
            accumulated_cost=self.state.accumulated_cost,
            etc=etc,
            scan_only=self.state.scan_only,
            limit_reason=limit_reason,
            stats=stats,
            pending=pending_parts,
        )
        return build_resource_section(config, self.gauge_width)

    # ========================================================================
    # Public API
    # ========================================================================

    def tick(self) -> None:
        """Drain streaming queues for smooth display.

        Pops items from each queue and updates the corresponding current
        display item. When a queue has drained and no new items have been
        added for the stale timeout, clears the current item so the
        pipeline row can show its completion/idle state.
        """
        # Pop from score queue
        if item := self.state.score_queue.pop():
            self.state.current_score = ScoreItem(
                title=item.get("title", ""),
                score_composite=item.get("score_composite"),
                physics_domain=item.get("physics_domain"),
                description=item.get("description", ""),
                is_physics=item.get("is_physics", False),
                skipped=item.get("skipped", False),
                skip_reason=item.get("skip_reason", ""),
            )
        elif self.state.score_queue.is_stale():
            self.state.current_score = None

        # Pop from ingest queue
        if item := self.state.ingest_queue.pop():
            self.state.current_ingest = IngestItem(
                title=item.get("title", ""),
                score_composite=item.get("score_composite"),
                description=item.get("description", ""),
                physics_domain=item.get("physics_domain"),
                chunk_count=item.get("chunk_count", 0),
            )
        elif self.state.ingest_queue.is_stale():
            self.state.current_ingest = None

        # Pop from document score queue (priority over ingest queue)
        if item := self.state.document_score_queue.pop():
            self.state.current_docs = DocsItem(
                filename=item.get("filename", ""),
                document_type=item.get("document_type", ""),
                score_composite=item.get("score_composite"),
                physics_domain=item.get("physics_domain"),
                description=item.get("description", ""),
                is_score=True,
            )
        # Pop from document ingest queue
        elif item := self.state.document_queue.pop():
            self.state.current_docs = DocsItem(
                filename=item.get("filename", ""),
                document_type=item.get("document_type", ""),
                score_composite=item.get("score_composite"),
                physics_domain=item.get("physics_domain"),
                description=item.get("description", ""),
                chunk_count=item.get("chunk_count", 0),
                is_score=False,
            )
        elif (
            self.state.document_score_queue.is_stale()
            and self.state.document_queue.is_stale()
        ):
            self.state.current_docs = None

        # Pop from image queue
        if item := self.state.image_queue.pop():
            page_title = item.get("page_title", "")
            page_image_count = item.get("page_image_count", 0)
            # Track display index: Nth image shown from this page
            page_image_index = 0
            if page_title:
                self.state._page_image_seen[page_title] = (
                    self.state._page_image_seen.get(page_title, 0) + 1
                )
                page_image_index = self.state._page_image_seen[page_title]
            self.state.current_image = ImageItem(
                image_id=item.get("id", ""),
                score_composite=item.get("score_composite"),
                physics_domain=item.get("physics_domain"),
                description=item.get("description", ""),
                purpose=item.get("purpose", ""),
                source_url=item.get("source_url", ""),
                page_title=page_title,
                page_image_count=page_image_count,
                page_image_index=page_image_index,
            )
        elif self.state.image_queue.is_stale():
            self.state.current_image = None

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
        self.state._run_score_cost = stats.cost

        # Track processing state and idle/active transitions
        if "waiting" in message.lower() or message == "idle":
            self.state.score_processing = False
            stats.mark_idle()
        elif "scoring" in message.lower() or "fetching" in message.lower():
            self.state.score_processing = True
            stats.mark_active()
        elif "scored" in message.lower() and results:
            self.state.score_processing = not self.state.score_queue.is_empty()
            stats.mark_active()
            stats.record_batch(len(results))
        else:
            self.state.score_processing = False

        # Use EMA rate for live display (falls back to active_rate)
        self.state.score_rate = stats.ema_rate

        # Queue results for streaming with adaptive rate
        if results:
            items = []
            for r in results:
                title = r.get("title") or r.get("id", "?").split(":")[-1]
                items.append(
                    {
                        "title": title,
                        "score_composite": r.get("score_composite"),
                        "physics_domain": r.get("physics_domain"),
                        "description": r.get("description", ""),
                        "is_physics": r.get("is_physics", False),
                        "skipped": r.get("skipped", False),
                        "skip_reason": r.get("skip_reason", ""),
                    }
                )
            max_display_rate = 2.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 1.0
            self.state.score_queue.add(
                items, display_rate, last_batch_time=stats.last_batch_time
            )

        self._refresh()

    def update_ingest(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update ingester state."""
        self.state.run_ingested = stats.processed
        self.state._run_ingest_cost = stats.cost

        # Track processing state and idle/active transitions
        if "waiting" in message.lower() or message == "idle":
            self.state.ingest_processing = False
            stats.mark_idle()
        elif "ingesting" in message.lower():
            self.state.ingest_processing = True
            stats.mark_active()
        elif "ingested" in message.lower() and results:
            self.state.ingest_processing = not self.state.ingest_queue.is_empty()
            stats.mark_active()
            stats.record_batch(len(results))
        else:
            self.state.ingest_processing = False

        self.state.ingest_rate = stats.ema_rate

        # Queue results for streaming with adaptive rate
        if results:
            items = []
            for r in results:
                # Use title if already extracted, otherwise extract from id
                title = r.get("title") or r.get("id", "?").split(":")[-1]
                items.append(
                    {
                        "title": title,
                        "score_composite": r.get("score_composite"),
                        "description": r.get("description", ""),
                        "physics_domain": r.get("physics_domain"),
                        "chunk_count": r.get("chunk_count", 0),
                    }
                )
            max_display_rate = 2.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 1.0
            self.state.ingest_queue.add(
                items, display_rate, last_batch_time=stats.last_batch_time
            )

        self._refresh()

    def update_docs(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update document worker state."""
        self.state.run_docs = stats.processed

        # Track processing state and idle/active transitions
        if "waiting" in message.lower() or message == "idle":
            self.state.docs_processing = False
            stats.mark_idle()
        elif "ingesting" in message.lower():
            self.state.docs_processing = True
            stats.mark_active()
        elif "ingested" in message.lower() and results:
            self.state.docs_processing = not self.state.document_queue.is_empty()
            stats.mark_active()
            stats.record_batch(len(results))
        else:
            self.state.docs_processing = False

        self.state.docs_rate = stats.ema_rate

        # Queue results for streaming
        if results:
            items = []
            for r in results:
                items.append(
                    {
                        "filename": r.get("filename", "unknown"),
                        "document_type": r.get("document_type", ""),
                        "score_composite": r.get("score_composite"),
                        "physics_domain": r.get("physics_domain"),
                        "description": r.get("description", ""),
                        "chunk_count": r.get("chunk_count", 0),
                    }
                )
            max_display_rate = 1.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 0.5
            self.state.document_queue.add(
                items, display_rate, last_batch_time=stats.last_batch_time
            )

        self._refresh()

    def update_docs_score(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update document score worker state."""
        self.state.run_docs_scored = stats.processed
        self.state._run_docs_score_cost = stats.cost

        # Track processing state and idle/active transitions
        if "waiting" in message.lower() or message == "idle":
            self.state.docs_processing = False
            stats.mark_idle()
        elif "scoring" in message.lower() or "extracting" in message.lower():
            self.state.docs_processing = True
            stats.mark_active()
        elif "scored" in message.lower() and results:
            self.state.docs_processing = not self.state.document_score_queue.is_empty()
            stats.mark_active()
            stats.record_batch(len(results))
        else:
            self.state.docs_processing = False

        self.state.document_score_rate = stats.ema_rate

        # Queue results for streaming
        if results:
            items = [
                {
                    "filename": r.get("filename", "unknown"),
                    "document_type": r.get("document_type", ""),
                    "score_composite": r.get("score_composite"),
                    "physics_domain": r.get("physics_domain"),
                    "description": r.get("description", ""),
                }
                for r in results
            ]
            max_display_rate = 1.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 0.5
            self.state.document_score_queue.add(
                items, display_rate, last_batch_time=stats.last_batch_time
            )

        self._refresh()

    def update_image(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update image VLM worker state."""
        self.state.run_images_scored = stats.processed
        self.state._run_image_score_cost = stats.cost

        # Track processing state and idle/active transitions
        if "waiting" in message.lower() or message == "idle":
            self.state.image_processing = False
            stats.mark_idle()
        elif "scoring" in message.lower():
            self.state.image_processing = True
            stats.mark_active()
        elif "scored" in message.lower() and results:
            self.state.image_processing = not self.state.image_queue.is_empty()
            stats.mark_active()
            stats.record_batch(len(results))
        else:
            self.state.image_processing = False

        self.state.image_score_rate = stats.ema_rate

        # Queue results for streaming
        if results:
            items = [
                {
                    "id": r.get("id", "unknown"),
                    "score_composite": r.get("score_composite"),
                    "physics_domain": r.get("physics_domain"),
                    "description": r.get("description", ""),
                    "purpose": r.get("purpose", ""),
                    "source_url": r.get("source_url", ""),
                    "page_title": r.get("page_title", ""),
                    "page_image_count": r.get("page_image_count", 0),
                }
                for r in results
            ]
            max_display_rate = 1.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 0.5
            self.state.image_queue.add(
                items, display_rate, last_batch_time=stats.last_batch_time
            )

        self._refresh()

    def update_from_graph(
        self,
        total_pages: int = 0,
        pages_scanned: int = 0,
        pages_scored: int = 0,
        pages_ingested: int = 0,
        pages_skipped: int = 0,
        pages_failed: int = 0,
        pending_score: int = 0,
        pending_ingest: int = 0,
        pending_document_score: int = 0,
        pending_document_ingest: int = 0,
        accumulated_cost: float = 0.0,
        **kwargs,  # Ignore extra args for compatibility
    ) -> None:
        """Update state from graph statistics."""
        self.state.total_pages = total_pages
        self.state.pages_scanned = pages_scanned
        self.state.pages_scored = pages_scored
        self.state.pages_ingested = pages_ingested
        self.state.pages_skipped = pages_skipped
        self.state.pages_failed = pages_failed
        self.state.pending_score = pending_score
        self.state.pending_ingest = pending_ingest
        self.state.pending_document_score = pending_document_score
        self.state.pending_document_ingest = pending_document_ingest
        self.state.accumulated_cost = accumulated_cost
        # Per-group accumulated costs
        if "accumulated_page_cost" in kwargs:
            self.state.accumulated_page_cost = kwargs["accumulated_page_cost"]
        if "accumulated_document_cost" in kwargs:
            self.state.accumulated_document_cost = kwargs["accumulated_document_cost"]
        if "accumulated_image_cost" in kwargs:
            self.state.accumulated_image_cost = kwargs["accumulated_image_cost"]
        # Update graph-based document counts if provided
        if "total_documents" in kwargs:
            self.state.total_documents = kwargs["total_documents"]
        if "docs_ingested" in kwargs:
            self.state.docs_ingested = kwargs["docs_ingested"]
        if "docs_scored" in kwargs:
            self.state.docs_scored = kwargs["docs_scored"]
        if "documents_failed" in kwargs:
            self.state.documents_failed = kwargs["documents_failed"]
        if "documents_deferred" in kwargs:
            self.state.documents_deferred = kwargs["documents_deferred"]
        if "documents_skipped" in kwargs:
            self.state.documents_skipped = kwargs["documents_skipped"]
        # Update graph-based image counts if provided
        if "images_scored" in kwargs:
            self.state.images_scored = kwargs["images_scored"]
        if "pending_image_score" in kwargs:
            self.state.pending_image_score = kwargs["pending_image_score"]
        # Historic rates from graph timestamps (fallback for done workers)
        if "historic_score_rate" in kwargs:
            self.state._historic_score_rate = kwargs["historic_score_rate"]
        if "historic_ingest_rate" in kwargs:
            self.state._historic_ingest_rate = kwargs["historic_ingest_rate"]

        # Accumulated wall-clock time from prior sessions
        from imas_codex.discovery.base.progress import get_accumulated_time

        self.state.accumulated_time = get_accumulated_time(self.state.facility, "wiki")

        self._refresh()

    def set_site_info(self, site_name: str, site_index: int, total_sites: int) -> None:
        """Set multi-site info for initial display."""
        self.state.current_site_name = site_name
        self.state.current_site_index = site_index
        self.state.total_sites = total_sites
        self._refresh()

    def advance_site(self, site_name: str, site_index: int) -> None:
        """Advance to next wiki site, accumulating stats from current site.

        Saves current-site run counters into offsets so that the display
        shows grand totals across all sites, then resets per-site state
        for the new site's workers.
        """
        # Accumulate current-site stats into offsets
        self.state._offset_scored += self.state.run_scored
        self.state._offset_ingested += self.state.run_ingested
        self.state._offset_score_cost += self.state._run_score_cost
        self.state._offset_ingest_cost += self.state._run_ingest_cost
        self.state._offset_docs_score_cost += self.state._run_docs_score_cost
        self.state._offset_docs += self.state.run_docs
        self.state._offset_docs_scored += self.state.run_docs_scored
        self.state._offset_images_scored += self.state.run_images_scored
        self.state._offset_image_score_cost += self.state._run_image_score_cost

        # Reset per-site counters (new workers will fill these)
        self.state.run_scored = 0
        self.state.run_ingested = 0
        self.state._run_score_cost = 0.0
        self.state._run_ingest_cost = 0.0
        self.state._run_docs_score_cost = 0.0
        self.state.run_docs = 0
        self.state.run_docs_scored = 0
        self.state.run_images_scored = 0
        self.state._run_image_score_cost = 0.0
        self.state.score_rate = None
        self.state.ingest_rate = None
        self.state.docs_rate = None
        self.state.document_score_rate = None
        self.state.image_score_rate = None

        # Reset activity displays
        self.state.current_score = None
        self.state.current_ingest = None
        self.state.current_docs = None
        self.state.current_image = None
        self.state.score_processing = False
        self.state.ingest_processing = False
        self.state.docs_processing = False
        self.state.image_processing = False

        # Reset worker group (set by new run_parallel_wiki_discovery)
        self.worker_group = None

        # Reset streaming queues
        self.state.score_queue = StreamQueue(
            rate=0.5, max_rate=2.5, min_display_time=0.4
        )
        self.state.ingest_queue = StreamQueue(
            rate=0.5, max_rate=2.5, min_display_time=0.4
        )
        self.state.document_queue = StreamQueue(
            rate=0.3, max_rate=1.0, min_display_time=0.5
        )
        self.state.document_score_queue = StreamQueue(
            rate=0.3, max_rate=1.0, min_display_time=0.5
        )
        self.state.image_queue = StreamQueue(
            rate=0.3, max_rate=1.0, min_display_time=0.5
        )
        self.state._page_image_seen = {}

        # Update site info
        self.state.current_site_name = site_name
        self.state.current_site_index = site_index

        self._refresh()

    def print_summary(self) -> None:
        """Print final discovery summary."""
        self.console.print()
        title = f"{self.state.facility.upper()} Wiki Discovery Complete"
        if self.state.total_sites > 1:
            title += f" ({self.state.total_sites} sites)"
        self.console.print(
            Panel(
                self._build_summary(),
                title=title,
                border_style="green",
                width=self.width,
            )
        )

    def _build_summary(self) -> Text:
        """Build final summary text."""
        summary = Text()

        # SCORE stats (pages)
        total_scored = self.state.pages_scored + self.state.pages_ingested
        summary.append(f"{'  SCORE':<{LABEL_WIDTH}}", style="bold blue")
        summary.append(f"scored={format_count(total_scored)}", style="blue")
        summary.append(
            f"  skipped={format_count(self.state.pages_skipped)}", style="yellow"
        )
        if self.state.pages_failed > 0:
            summary.append(
                f"  failed={format_count(self.state.pages_failed)}", style="red"
            )
        page_cost = self.state.accumulated_page_cost
        if page_cost > 0:
            summary.append(f"  cost=${page_cost:.2f}", style="yellow")
        if self.state.score_rate or self.state._final_score_rate:
            rate = self.state.score_rate or self.state._final_score_rate
            summary.append(f"  {rate:.1f}/s", style="dim")
        summary.append("\n")

        # INGEST stats
        total_ingested = self.state.pages_ingested
        summary.append(f"{'  INGEST':<{LABEL_WIDTH}}", style="bold magenta")
        summary.append(f"ingested={format_count(total_ingested)}", style="magenta")
        if self.state.ingest_rate or self.state._final_ingest_rate:
            rate = self.state.ingest_rate or self.state._final_ingest_rate
            summary.append(f"  {rate:.1f}/s", style="dim")
        summary.append("\n")

        # FILE stats (documents)
        art_total = self.state.total_documents
        if art_total > 0:
            art_ingested = self.state.docs_ingested
            art_scored = self.state.docs_scored
            art_failed = self.state.documents_failed
            art_deferred = self.state.documents_deferred
            summary.append(f"{'  FILE':<{LABEL_WIDTH}}", style="bold yellow")
            summary.append(f"scored={format_count(art_scored)}", style="yellow")
            summary.append(f"  ingested={format_count(art_ingested)}", style="yellow")
            if art_failed > 0:
                summary.append(f"  failed={format_count(art_failed)}", style="red")
            if art_deferred > 0:
                summary.append(f"  deferred={format_count(art_deferred)}", style="dim")
            art_cost = self.state.accumulated_document_cost
            if art_cost > 0:
                summary.append(f"  cost=${art_cost:.2f}", style="yellow")
            if self.state.docs_rate or self.state._final_docs_rate:
                rate = self.state.docs_rate or self.state._final_docs_rate
                summary.append(f"  {rate:.1f}/s", style="dim")
            summary.append("\n")

        # IMAGE stats — VLM describes images, producing metadata
        images_described = self.state.images_scored
        if images_described > 0:
            summary.append(f"{'  IMAGE':<{LABEL_WIDTH}}", style="bold green")
            summary.append(f"described={format_count(images_described)}", style="green")
            pending_img = self.state.pending_image_score
            if pending_img > 0:
                summary.append(f"  pending={format_count(pending_img)}", style="yellow")
            img_cost = self.state.accumulated_image_cost
            if img_cost > 0:
                summary.append(f"  cost=${img_cost:.2f}", style="yellow")
            if self.state.image_score_rate or self.state._final_image_rate:
                rate = self.state.image_score_rate or self.state._final_image_rate
                summary.append(f"  {rate:.1f}/s", style="dim")
            summary.append("\n")

        # TOTAL stats — accumulated time + cost from graph (all sessions)
        summary.append(f"{'  TOTAL':<{LABEL_WIDTH}}", style="bold white")
        summary.append(f"time={format_time(self.state.elapsed)}", style="white")
        accumulated = self.state.accumulated_cost
        if accumulated > 0:
            summary.append(f"  cost=${accumulated:.2f}", style="yellow")
        else:
            summary.append(f"  cost=${self.state.run_cost:.2f}", style="yellow")
        if self.state.total_sites > 1:
            summary.append(
                f"  sites={self.state.current_site_index + 1}/{self.state.total_sites}",
                style="cyan",
            )

        # Show coverage percentage (include all terminal states)
        if self.state.total_pages > 0:
            processed = (
                total_scored + self.state.pages_skipped + self.state.pages_failed
            )
            coverage = processed / self.state.total_pages * 100
            summary.append(f"  coverage={coverage:.1f}%", style="cyan")

        # Show limit reason if applicable
        if self.state.provider_budget_exhausted:
            summary.append("\n")
            summary.append(
                "  API key budget exhausted (HTTP 402) — "
                "LLM workers stopped, I/O workers drained queues",
                style="bold yellow",
            )

        return summary
