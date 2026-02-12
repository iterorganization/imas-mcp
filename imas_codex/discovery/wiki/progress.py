"""
Progress display for parallel wiki discovery.

Design principles (matching paths parallel_progress.py):
- Minimal visual clutter (no emojis, no stopwatch icons)
- Clear hierarchy: Target → Progress → Activity → Resources
- Gradient progress bars with percentage
- Resource gauges for time and cost budgets
- Compact current activity with relevant details only
- Live embedding source indicator (shows remote/openrouter/local)

Display layout: WORKERS → PROGRESS → ACTIVITY → RESOURCES
- WORKERS: Live worker status showing counts by task and state
- SCORE: Content-aware LLM scoring (fetches content, scores with LLM)
- PAGE: Chunk and embed high-value pages (score >= 0.5)
- ARTFCT: Artifact scoring and ingestion pipeline
- IMAGE: VLM captioning and scoring of wiki images

Progress is tracked against total pages in graph, not just this session.
ETA/ETC metrics calculated like paths discovery.

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

from imas_codex.discovery.base.progress import (
    GAUGE_METRICS_WIDTH,
    LABEL_WIDTH,
    METRICS_WIDTH,
    MIN_WIDTH,
    StreamQueue,
    build_servers_section,
    build_worker_status_section,
    clean_text,
    clip_text,
    compute_bar_width,
    compute_gauge_width,
    format_time,
    make_bar,
    make_resource_gauge,
)
from imas_codex.discovery.base.supervision import (
    SupervisedWorkerGroup,
    WorkerState,
)
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


@dataclass
class ArtifactItem:
    """Current artifact activity (score or ingest)."""

    filename: str
    artifact_type: str
    score: float | None = None
    physics_domain: str | None = None
    description: str = ""
    chunk_count: int = 0
    is_score: bool = False  # True if scoring, False if ingesting


@dataclass
class ImageItem:
    """Current image VLM activity."""

    image_id: str
    caption: str = ""
    score: float | None = None
    physics_domain: str | None = None
    description: str = ""
    purpose: str = ""


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

    # Worker group for status tracking
    worker_group: SupervisedWorkerGroup | None = None

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

    # Artifact stats
    artifacts_ingested: int = 0
    artifacts_scored: int = 0
    run_artifacts: int = 0
    run_artifacts_scored: int = 0
    artifact_rate: float | None = None
    artifact_score_rate: float | None = None
    _run_artifact_score_cost: float = 0.0

    # Image stats
    images_scored: int = 0  # From graph
    run_images_scored: int = 0
    image_score_rate: float | None = None
    _run_image_score_cost: float = 0.0

    # Accumulated facility cost (from graph)
    accumulated_cost: float = 0.0

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
    _offset_artifact_score_cost: float = 0.0
    _offset_artifacts: int = 0
    _offset_artifacts_scored: int = 0
    _offset_images_scored: int = 0
    _offset_image_score_cost: float = 0.0

    # Current items (and their processing state)
    current_score: ScoreItem | None = None
    current_ingest: IngestItem | None = None
    current_artifact: ArtifactItem | None = None
    current_image: ImageItem | None = None
    score_processing: bool = False
    ingest_processing: bool = False
    artifact_processing: bool = False
    image_processing: bool = False

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
    artifact_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.3, max_rate=1.0, min_display_time=0.5
        )
    )
    artifact_score_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.3, max_rate=1.0, min_display_time=0.5
        )
    )
    image_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.3, max_rate=1.0, min_display_time=0.5
        )
    )

    # Service monitor reference (for SERVERS display row)
    service_monitor: Any = None

    # Tracking
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    # Artifact pending counts (from graph refresh)
    pending_artifact_score: int = 0  # discovered artifacts awaiting scoring
    pending_artifact_ingest: int = 0  # scored artifacts awaiting ingestion

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
            + self._offset_artifact_score_cost
            + self._run_artifact_score_cost
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
    def total_run_artifacts(self) -> int:
        """Total artifacts ingested across all sites."""
        return self._offset_artifacts + self.run_artifacts

    @property
    def total_run_artifacts_scored(self) -> int:
        """Total artifacts scored across all sites."""
        return self._offset_artifacts_scored + self.run_artifacts_scored

    @property
    def total_run_images_scored(self) -> int:
        """Total images scored across all sites."""
        return self._offset_images_scored + self.run_images_scored

    @property
    def total_score_cost(self) -> float:
        """Total page+artifact+image scoring cost across all sites."""
        return (
            self._offset_score_cost
            + self._run_score_cost
            + self._offset_artifact_score_cost
            + self._run_artifact_score_cost
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
    def cost_per_artifact_score(self) -> float | None:
        """Average LLM cost per scored artifact."""
        total = self.total_run_artifacts_scored
        cost = self._offset_artifact_score_cost + self._run_artifact_score_cost
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
        """Estimated time to termination based on limits.

        For limit-based runs (cost or page limit), estimate time to hit limit.
        For unconstrained runs, estimate from the slowest worker group:
        the terminal time is max(score_eta, ingest_eta, artifact_score_eta,
        artifact_ingest_eta) since workers run in parallel.
        """
        # Priority 1: Cost limit - time to exhaust budget
        if self.run_cost > 0 and self.cost_limit > 0:
            cost_rate = self.run_cost / self.elapsed if self.elapsed > 0 else 0
            if cost_rate > 0:
                remaining_budget = self.cost_limit - self.run_cost
                return max(0, remaining_budget / cost_rate)

        # Priority 2: Page limit
        if self.page_limit is not None and self.page_limit > 0:
            total_scored = self.total_run_scored
            if total_scored > 0 and self.elapsed > 0:
                rate = total_scored / self.elapsed
                remaining = self.page_limit - total_scored
                return max(0, remaining / rate) if rate > 0 else None

        # Priority 3: Work-based ETA from slowest worker group
        # Each worker group has its own remaining work and processing rate.
        # Terminal time = max of all worker ETAs (parallel pipeline).
        worker_etas: list[float] = []

        # Page scoring ETA
        if self.pending_score > 0 and self.score_rate and self.score_rate > 0:
            worker_etas.append(self.pending_score / self.score_rate)

        # Page ingestion ETA (scored pages above threshold)
        if self.pending_ingest > 0 and self.ingest_rate and self.ingest_rate > 0:
            worker_etas.append(self.pending_ingest / self.ingest_rate)

        # Artifact scoring ETA
        if (
            self.pending_artifact_score > 0
            and self.artifact_score_rate
            and self.artifact_score_rate > 0
        ):
            worker_etas.append(self.pending_artifact_score / self.artifact_score_rate)

        # Artifact ingestion ETA
        if (
            self.pending_artifact_ingest > 0
            and self.artifact_rate
            and self.artifact_rate > 0
        ):
            worker_etas.append(self.pending_artifact_ingest / self.artifact_rate)

        # Image scoring ETA
        if (
            self.pending_image_score > 0
            and self.image_score_rate
            and self.image_score_rate > 0
        ):
            worker_etas.append(self.pending_image_score / self.image_score_rate)

        if worker_etas:
            return max(worker_etas)

        # No rate data yet - can't estimate
        return None


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
    │  PAGE   ━━━━━━━━━━━━━━━━━━━━━━━━──────────────────────────────     136  35%  0.3/s              │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  SCORE 0.70 Microwave_lab/software                                                               │
    │    [physics domain] Power supply calibration for gyrotron system                                 │
    │  PAGE  Service_Mécanique/Information_TCV/Connexion_dans_l                                        │
    │    0.65 [equilibrium] Vacuum vessel port documentation with coordinates                          │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  TIME    ━━━━━━━━━━━━━━━━━━━━━━  7m     ETA 2h 15m                                              │
    │  COST    ━━━━━━━━━━━━━━━━━━━━━━  $0.04 / $0.01  cost limit reached                              │
    │  TOTAL   ━━━━━━━━━━━━━━━━━━━━━━  $0.04  ETC $1.50                                               │
    │  STATS  scored=388  ingested=136  skipped=0  pending=[score:7386 ingest:252]                     │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘

    Workflow:
    - SCORE: Content-aware LLM scoring (scanned → scored)
    - PAGE: Chunk and embed high-value pages (scored → ingested)
    - ARTFCT: Score and ingest wiki artifacts (PDFs, CSVs, etc.)
    - IMAGE: VLM captioning + scoring (ingested → captioned)
    """

    # Layout constants imported from base.progress
    LABEL_WIDTH = LABEL_WIDTH
    MIN_WIDTH = MIN_WIDTH
    METRICS_WIDTH = METRICS_WIDTH
    GAUGE_METRICS_WIDTH = GAUGE_METRICS_WIDTH

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

    @property
    def width(self) -> int:
        """Get display width based on terminal size (fills terminal)."""
        term_width = self.console.width or 100
        return max(self.MIN_WIDTH, term_width)

    @property
    def bar_width(self) -> int:
        """Calculate progress bar width to fill available space."""
        return compute_bar_width(self.width)

    @property
    def gauge_width(self) -> int:
        """Calculate resource gauge width (shorter than bar to fit metrics)."""
        return compute_gauge_width(self.width)

    def _build_header(self) -> Text:
        """Build centered header with facility and focus."""
        header = Text()

        # Facility name with mode indicator
        title = f"{self.state.facility.upper()} Wiki Discovery"
        if self.state.scan_only:
            title += " (SCAN ONLY)"
        elif self.state.score_only:
            title += " (SCORE ONLY)"

        # Multi-site indicator: show current site URL
        if self.state.total_sites > 1 and self.state.current_site_name:
            title += f"  {self.state.current_site_name}"

        header.append(title.center(self.width - 4), style="bold cyan")

        # Focus (if set and not scan_only)
        if self.state.focus and not self.state.scan_only:
            header.append("\n")
            focus_line = f"Focus: {self.state.focus}"
            header.append(focus_line.center(self.width - 4), style="italic dim")

        return header

    def _build_worker_section(self) -> Text:
        """Build worker status section grouped by functional role.

        Uses the unified builder from base.progress. Workers are grouped
        by their ``group`` field (score vs ingest) set during creation
        in parallel.py.

        Appends embedding source indicator when no service monitor is
        providing that information.
        """
        monitor = self.state.service_monitor
        is_paused = monitor is not None and monitor.paused

        # Build extra indicators (embedding source)
        extra: list[tuple[str, str]] = []
        if self.state.service_monitor is None:
            embed_health = get_embed_status()
            if embed_health != "ready":
                extra.append((f"embed:{embed_health}", "red"))
            else:
                embed_source = get_embedding_source()
                if embed_source.startswith("iter-"):
                    extra.append((f"embed:{embed_source}", "green"))
                elif embed_source == "remote":
                    extra.append(("embed:remote", "green"))
                elif embed_source == "openrouter":
                    extra.append(("embed:openrouter", "yellow"))
                elif embed_source == "local":
                    extra.append(("embed:local", "cyan"))
                else:
                    extra.append((f"embed:{embed_source}", "dim"))

        return build_worker_status_section(
            self.state.worker_group,
            budget_exhausted=self.state.cost_limit_reached,
            is_paused=is_paused,
            budget_sensitive_groups={"score"},
            extra_indicators=extra or None,
        )

    def _build_progress_section(self) -> Text:
        """Build the main progress bars for SCORE and INGEST.

        Progress is measured against total pages in graph, not just this session.
        This shows true coverage of the wiki.
        """
        section = Text()
        bar_width = self.bar_width

        # SCORE row - scoring progress against total pages
        # Total to score = total_pages (all wiki pages for this facility)
        # Completed = pages_scored + pages_ingested (scored includes ingest-ready)
        score_total = self.state.total_pages or 1
        scored_pages = self.state.pages_scored + self.state.pages_ingested
        score_pct = scored_pages / score_total * 100 if score_total > 0 else 0

        if self.state.scan_only:
            section.append("  SCORE   ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  SCORE   ", style="bold blue")
            score_ratio = min(scored_pages / score_total, 1.0) if score_total > 0 else 0
            section.append(make_bar(score_ratio, bar_width), style="blue")
            section.append(f" {scored_pages:>6,}", style="bold")
            section.append(f" {score_pct:>3.0f}%", style="cyan")
            if self.state.score_rate and self.state.score_rate > 0:
                section.append(f" {self.state.score_rate:>5.1f}/s", style="dim")
        section.append("\n")

        # PAGE row - ingestion progress against scored pages
        # Total to ingest = pages that qualify for ingestion (score >= 0.5)
        # For simplicity, use pages_scored + pages_ingested as denominator
        ingest_total = self.state.pages_scored + self.state.pages_ingested
        if ingest_total <= 0:
            ingest_total = 1
        ingest_pct = self.state.pages_ingested / ingest_total * 100

        if self.state.scan_only:
            section.append("  PAGE    ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  PAGE    ", style="bold magenta")
            ingest_ratio = min(self.state.pages_ingested / ingest_total, 1.0)
            section.append(make_bar(ingest_ratio, bar_width), style="magenta")
            section.append(f" {self.state.pages_ingested:>6,}", style="bold")
            section.append(f" {ingest_pct:>3.0f}%", style="cyan")
            if self.state.ingest_rate and self.state.ingest_rate > 0:
                section.append(f" {self.state.ingest_rate:>5.1f}/s", style="dim")

        # ARTIFACTS row - pipeline progress: scored+ingested / total processable
        # Total processable = pending_score + scored + ingested (all supported types)
        # Completed = scored + ingested (artifacts that have been through scoring)
        # This ensures progress shows while scoring (even when most are skipped)
        section.append("\n")
        if self.state.scan_only:
            section.append("  ARTFCT  ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  ARTFCT  ", style="bold yellow")
            scored = self.state.artifacts_scored
            ingested = self.state.artifacts_ingested
            pending_score = self.state.pending_artifact_score
            completed = scored + ingested
            art_total = pending_score + completed
            if art_total > 0:
                art_ratio = min(completed / art_total, 1.0)
                section.append(make_bar(art_ratio, bar_width), style="yellow")
                section.append(f" {completed:>6,}", style="bold")
                art_pct = completed / art_total * 100
                section.append(f" {art_pct:>3.0f}%", style="cyan")
            else:
                section.append("─" * bar_width, style="dim")
                section.append(f" {completed:>6,}", style="bold")
            # Show combined score+ingest rate
            score_rate = self.state.artifact_score_rate
            ingest_rate = self.state.artifact_rate
            combined_rate = sum(r for r in [score_rate, ingest_rate] if r and r > 0)
            if combined_rate > 0:
                section.append(f" {combined_rate:>5.1f}/s", style="dim")

        # IMAGE row - VLM captioning + scoring progress
        section.append("\n")
        if self.state.scan_only:
            section.append("  IMAGE   ", style="dim")
            section.append("─" * bar_width, style="dim")
            section.append("    disabled", style="dim italic")
        else:
            section.append("  IMAGE   ", style="bold green")
            img_scored = self.state.images_scored
            img_pending = self.state.pending_image_score
            img_total = img_scored + img_pending
            if img_total > 0:
                img_ratio = min(img_scored / img_total, 1.0)
                section.append(make_bar(img_ratio, bar_width), style="green")
                section.append(f" {img_scored:>6,}", style="bold")
                img_pct = img_scored / img_total * 100
                section.append(f" {img_pct:>3.0f}%", style="cyan")
            else:
                section.append("─" * bar_width, style="dim")
                section.append(f" {img_scored:>6,}", style="bold")
            if self.state.image_score_rate and self.state.image_score_rate > 0:
                section.append(f" {self.state.image_score_rate:>5.1f}/s", style="dim")

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
        content_width = self.width - 6  # Panel padding + border

        # Check if workers are paused (for showing "paused" instead of "idle")
        monitor = self.state.service_monitor
        is_paused = monitor is not None and monitor.paused

        # Helper to determine if worker should show "idle" or "paused"
        def should_show_idle(processing: bool, queue: StreamQueue) -> bool:
            return not processing and queue.is_empty()

        def is_worker_complete(task_type: str) -> bool:
            """Check if all workers of a given task type have stopped."""
            wg = self.state.worker_group
            if not wg:
                return False
            workers = [
                (name, status.state)
                for name, status in wg.workers.items()
                if task_type in name
            ]
            return len(workers) > 0 and all(
                s == WorkerState.stopped for _, s in workers
            )

        # SCORE section - always 2 lines for consistent height
        if not self.state.scan_only:
            section.append("  SCORE   ", style="bold blue")
            if score:
                # Line 1: Page title (clipped to fit remaining space)
                title_width = content_width - self.LABEL_WIDTH
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
                if is_paused:
                    section.append("paused", style="dim italic")
                else:
                    section.append("processing...", style="cyan italic")
                section.append("\n    ", style="dim")
            elif not self.state.score_queue.is_empty():
                # Items in queue but not yet popped - show waiting state
                queued = len(self.state.score_queue)
                section.append(f"streaming {queued} items...", style="cyan italic")
                section.append("\n    ", style="dim")
            elif is_worker_complete("score"):
                if self.state.cost_limit_reached:
                    section.append("cost limit", style="yellow")
                else:
                    section.append("complete", style="green")
                section.append("\n    ", style="dim")
            elif should_show_idle(self.state.score_processing, self.state.score_queue):
                if is_paused:
                    section.append("paused", style="dim italic")
                else:
                    section.append("idle", style="dim italic")
                section.append("\n    ", style="dim")
            else:
                section.append("...", style="dim italic")
                section.append("\n    ", style="dim")
            section.append("\n")

        # PAGE section - 2 lines: title, then score + domain + description
        if not self.state.scan_only:
            section.append("  PAGE    ", style="bold magenta")
            if ingest:
                # Line 1: Page title (clipped to fit remaining space)
                title_width = content_width - self.LABEL_WIDTH
                section.append(
                    self._clip_title(ingest.title, title_width), style="white"
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
                if is_paused:
                    section.append("paused", style="dim italic")
                else:
                    section.append("processing...", style="cyan italic")
                section.append("\n    ", style="dim")
            elif not self.state.ingest_queue.is_empty():
                # Items in queue but not yet popped - show waiting state
                queued = len(self.state.ingest_queue)
                section.append(f"streaming {queued} items...", style="cyan italic")
                section.append("\n    ", style="dim")
            elif is_worker_complete("ingest"):
                section.append("complete", style="green")
                section.append("\n    ", style="dim")
            elif should_show_idle(
                self.state.ingest_processing, self.state.ingest_queue
            ):
                if is_paused:
                    section.append("paused", style="dim italic")
                else:
                    section.append("idle", style="dim italic")
                section.append("\n    ", style="dim")
            else:
                section.append("...", style="dim italic")
                section.append("\n    ", style="dim")

        # ARTIFACT section - 2 lines: filename, then score + domain + description
        artifact = self.state.current_artifact
        if not self.state.scan_only:
            section.append("\n")
            section.append("  ARTFCT  ", style="bold yellow")
            if artifact:
                # Line 1: Artifact filename
                title_width = content_width - self.LABEL_WIDTH
                display_name = artifact.filename
                if artifact.chunk_count > 0:
                    suffix = f" ({artifact.chunk_count} chunks)"
                    if len(display_name) + len(suffix) > title_width:
                        display_name = (
                            display_name[: title_width - len(suffix) - 3] + "..."
                        )
                    display_name += suffix
                elif artifact.score is not None and artifact.score < 0.5:
                    suffix = " (skipped)"
                    if len(display_name) + len(suffix) > title_width:
                        display_name = (
                            display_name[: title_width - len(suffix) - 3] + "..."
                        )
                    display_name += suffix
                section.append(
                    self._clip_title(display_name, title_width), style="white"
                )
                section.append("\n")

                # Line 2: Score, physics domain, description
                section.append("    ", style="dim")
                score_str = ""
                if artifact.score is not None:
                    if artifact.score >= 0.7:
                        style = "bold green"
                    elif artifact.score >= 0.4:
                        style = "yellow"
                    else:
                        style = "red"
                    score_str = f"{artifact.score:.2f}"
                    section.append(f"{score_str}  ", style=style)

                domain_str = ""
                if artifact.physics_domain:
                    domain_str = artifact.physics_domain
                    section.append(f"{domain_str}  ", style="cyan")

                if artifact.description:
                    desc = clean_text(artifact.description)
                    used = (
                        4
                        + (len(score_str) + 2 if score_str else 0)
                        + (len(domain_str) + 2 if domain_str else 0)
                    )
                    desc_width = content_width - used
                    section.append(
                        clip_text(desc, max(10, desc_width)), style="italic dim"
                    )
            elif self.state.artifact_processing:
                if is_paused:
                    section.append("paused", style="dim italic")
                else:
                    section.append("processing...", style="cyan italic")
                section.append("\n    ", style="dim")
            elif (
                not self.state.artifact_queue.is_empty()
                or not self.state.artifact_score_queue.is_empty()
            ):
                queued = len(self.state.artifact_queue) + len(
                    self.state.artifact_score_queue
                )
                section.append(f"streaming {queued} items...", style="cyan italic")
                section.append("\n    ", style="dim")
            elif is_worker_complete("artifact"):
                if self.state.cost_limit_reached:
                    section.append("cost limit", style="yellow")
                else:
                    section.append("complete", style="green")
                section.append("\n    ", style="dim")
            elif should_show_idle(
                self.state.artifact_processing,
                self.state.artifact_queue,
            ):
                if is_paused:
                    section.append("paused", style="dim italic")
                else:
                    section.append("idle", style="dim italic")
                section.append("\n    ", style="dim")
            else:
                section.append("...", style="dim italic")
                section.append("\n    ", style="dim")

        # IMAGE section - 2 lines: image id, then score + domain + caption
        image = self.state.current_image
        if not self.state.scan_only:
            section.append("\n")
            section.append("  IMAGE   ", style="bold green")
            if image:
                # Line 1: Image ID (clipped)
                title_width = content_width - self.LABEL_WIDTH
                section.append(
                    self._clip_title(image.image_id, title_width), style="white"
                )
                section.append("\n")

                # Line 2: Score, physics domain, caption/description
                section.append("    ", style="dim")
                score_str = ""
                if image.score is not None:
                    if image.score >= 0.7:
                        style = "bold green"
                    elif image.score >= 0.4:
                        style = "yellow"
                    else:
                        style = "red"
                    score_str = f"{image.score:.2f}"
                    section.append(f"{score_str}  ", style=style)

                domain_str = ""
                if image.physics_domain:
                    domain_str = image.physics_domain
                    section.append(f"{domain_str}  ", style="cyan")

                desc = image.caption or image.description
                if desc:
                    desc = clean_text(desc)
                    used = (
                        4
                        + (len(score_str) + 2 if score_str else 0)
                        + (len(domain_str) + 2 if domain_str else 0)
                    )
                    desc_width = content_width - used
                    section.append(
                        clip_text(desc, max(10, desc_width)), style="italic dim"
                    )
            elif self.state.image_processing:
                if is_paused:
                    section.append("paused", style="dim italic")
                else:
                    section.append("processing...", style="cyan italic")
                section.append("\n    ", style="dim")
            elif not self.state.image_queue.is_empty():
                queued = len(self.state.image_queue)
                section.append(f"streaming {queued} items...", style="cyan italic")
                section.append("\n    ", style="dim")
            elif is_worker_complete("image"):
                if self.state.cost_limit_reached:
                    section.append("cost limit", style="yellow")
                else:
                    section.append("complete", style="green")
                section.append("\n    ", style="dim")
            elif should_show_idle(
                self.state.image_processing,
                self.state.image_queue,
            ):
                if is_paused:
                    section.append("paused", style="dim italic")
                else:
                    section.append("idle", style="dim italic")
                section.append("\n    ", style="dim")
            else:
                section.append("...", style="dim italic")
                section.append("\n    ", style="dim")

        return section

    def _build_resources_section(self) -> Text:
        """Build the resource consumption gauges with ETA/ETC like paths CLI."""
        section = Text()
        gw = self.gauge_width

        # TIME row with ETA
        section.append("  TIME    ", style="bold cyan")

        eta = None if self.state.scan_only else self.state.eta_seconds
        if eta is not None and eta > 0:
            total_est = self.state.elapsed + eta
            section.append_text(make_resource_gauge(self.state.elapsed, total_est, gw))
        else:
            section.append("━" * gw, style="cyan")

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
            section.append("  COST    ", style="bold yellow")
            section.append_text(
                make_resource_gauge(self.state.run_cost, self.state.cost_limit, gw)
            )
            section.append(f"  ${self.state.run_cost:.2f}", style="bold")
            section.append(f" / ${self.state.cost_limit:.2f}", style="dim")
            section.append("\n")

            # TOTAL row - cumulative cost with ETC (Estimated Total Cost)
            # ETC predicts cost to complete all pending score work:
            #   remaining page score cost + remaining artifact score cost
            total_facility_cost = self.state.accumulated_cost + self.state.run_cost
            has_pending = (
                self.state.pending_score > 0 or self.state.pending_artifact_score > 0
            )
            if total_facility_cost > 0 or has_pending:
                section.append("  TOTAL   ", style="bold white")

                # Predict remaining cost from per-item rates
                etc = total_facility_cost
                cpp = self.state.cost_per_page
                if cpp and cpp > 0 and self.state.pending_score > 0:
                    etc += self.state.pending_score * cpp
                cpa = self.state.cost_per_artifact_score
                if cpa and cpa > 0 and self.state.pending_artifact_score > 0:
                    etc += self.state.pending_artifact_score * cpa
                cpi = self.state.cost_per_image_score
                if cpi and cpi > 0 and self.state.pending_image_score > 0:
                    etc += self.state.pending_image_score * cpi

                if etc > 0:
                    section.append_text(
                        make_resource_gauge(total_facility_cost, etc, gw)
                    )
                else:
                    section.append("━" * gw, style="white")

                section.append(f"  ${total_facility_cost:.2f}", style="bold")
                if etc > total_facility_cost:
                    section.append(f"  ETC ${etc:.2f}", style="dim")
                section.append("\n")

        # STATS row - graph state with pending work
        section.append("  STATS   ", style="bold magenta")
        section.append(
            f"scored={self.state.pages_scored + self.state.pages_ingested}",
            style="blue",
        )
        section.append(f"  ingested={self.state.pages_ingested}", style="magenta")
        section.append(f"  skipped={self.state.pages_skipped}", style="yellow")

        # Pending work by worker type (combined page + artifact + image)
        pending_score = self.state.pending_score + self.state.pending_artifact_score
        pending_ingest = self.state.pending_ingest + self.state.pending_artifact_ingest
        pending_parts = []
        if pending_score > 0:
            pending_parts.append(f"score:{pending_score}")
        if pending_ingest > 0:
            pending_parts.append(f"ingest:{pending_ingest}")
        if self.state.pending_image_score > 0:
            pending_parts.append(f"image:{self.state.pending_image_score}")
        if pending_parts:
            section.append(f"  pending=[{' '.join(pending_parts)}]", style="cyan dim")

        return section

    def _build_servers_section(self) -> Text | None:
        """Build SERVERS status row from service monitor.

        Always returns a section (even before checks complete) so the
        SERVERS row is visible from the first render.
        """
        monitor = self.state.service_monitor
        if monitor is None:
            return None
        statuses = monitor.get_status()
        if not statuses:
            # Monitor registered but no checks configured yet — show pending
            section = Text()
            section.append("  SERVERS", style="bold white")
            section.append("  checking...", style="dim italic")
            return section
        return build_servers_section(statuses)

    def _build_display(self) -> Panel:
        """Build the complete display."""
        sections = [
            self._build_header(),
        ]

        # SERVERS and WORKERS are always grouped together (no separator between)
        sections.append(Text("─" * (self.width - 4), style="dim"))
        servers = self._build_servers_section()
        if servers is not None:
            sections.append(servers)
        sections.append(self._build_worker_section())

        sections.extend(
            [
                Text("─" * (self.width - 4), style="dim"),
                self._build_progress_section(),
                Text("─" * (self.width - 4), style="dim"),
                self._build_activity_section(),
                Text("─" * (self.width - 4), style="dim"),
                self._build_resources_section(),
            ]
        )

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

    def __enter__(self) -> WikiProgressDisplay:
        """Start live display."""
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
            transient=True,  # Remove live display on exit; summary replaces it
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

        # Pop from artifact score queue (priority over ingest queue)
        if item := self.state.artifact_score_queue.pop():
            self.state.current_artifact = ArtifactItem(
                filename=item.get("filename", ""),
                artifact_type=item.get("artifact_type", ""),
                score=item.get("score"),
                physics_domain=item.get("physics_domain"),
                description=item.get("description", ""),
                is_score=True,
            )
        # Pop from artifact ingest queue
        elif item := self.state.artifact_queue.pop():
            self.state.current_artifact = ArtifactItem(
                filename=item.get("filename", ""),
                artifact_type=item.get("artifact_type", ""),
                score=item.get("score"),
                physics_domain=item.get("physics_domain"),
                description=item.get("description", ""),
                chunk_count=item.get("chunk_count", 0),
                is_score=False,
            )

        # Pop from image queue
        if item := self.state.image_queue.pop():
            self.state.current_image = ImageItem(
                image_id=item.get("id", ""),
                caption=item.get("caption", ""),
                score=item.get("score"),
                physics_domain=item.get("physics_domain"),
                description=item.get("description", ""),
                purpose=item.get("purpose", ""),
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

        # Track processing state - keep processing=True if queue has items pending
        if "waiting" in message.lower() or message == "idle":
            self.state.score_processing = False
        elif "scoring" in message.lower() or "fetching" in message.lower():
            self.state.score_processing = True
        elif "scored" in message.lower() and results:
            # Results arriving - keep processing True until queue drains
            # This prevents the "..." flicker between batch completion and display
            self.state.score_processing = not self.state.score_queue.is_empty()
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
                        "title": title,
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

        # Track processing state - keep processing=True if queue has items pending
        if "waiting" in message.lower() or message == "idle":
            self.state.ingest_processing = False
        elif "ingesting" in message.lower():
            self.state.ingest_processing = True
        elif "ingested" in message.lower() and results:
            # Results arriving - keep processing True until queue drains
            self.state.ingest_processing = not self.state.ingest_queue.is_empty()
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
                        "title": title,
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

    def update_artifact(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update artifact worker state."""
        self.state.run_artifacts = stats.processed
        self.state.artifact_rate = stats.rate

        # Track processing state - keep processing=True if queue has items pending
        if "waiting" in message.lower() or message == "idle":
            self.state.artifact_processing = False
        elif "ingesting" in message.lower():
            self.state.artifact_processing = True
        elif "ingested" in message.lower() and results:
            # Results arriving - keep processing True until queue drains
            self.state.artifact_processing = not self.state.artifact_queue.is_empty()
        else:
            self.state.artifact_processing = False

        # Queue results for streaming
        if results:
            items = []
            for r in results:
                items.append(
                    {
                        "filename": r.get("filename", "unknown"),
                        "artifact_type": r.get("artifact_type", ""),
                        "score": r.get("score"),
                        "physics_domain": r.get("physics_domain"),
                        "description": r.get("description", ""),
                        "chunk_count": r.get("chunk_count", 0),
                    }
                )
            max_display_rate = 1.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 0.5
            self.state.artifact_queue.add(items, display_rate)

        self._refresh()

    def update_artifact_score(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update artifact score worker state."""
        self.state.run_artifacts_scored = stats.processed
        self.state.artifact_score_rate = stats.rate
        self.state._run_artifact_score_cost = stats.cost

        # Track processing state
        if "waiting" in message.lower() or message == "idle":
            self.state.artifact_processing = False
        elif "scoring" in message.lower() or "extracting" in message.lower():
            self.state.artifact_processing = True
        elif "scored" in message.lower() and results:
            self.state.artifact_processing = (
                not self.state.artifact_score_queue.is_empty()
            )
        else:
            self.state.artifact_processing = False

        # Queue results for streaming
        if results:
            items = [
                {
                    "filename": r.get("filename", "unknown"),
                    "artifact_type": r.get("artifact_type", ""),
                    "score": r.get("score"),
                    "physics_domain": r.get("physics_domain"),
                    "description": r.get("description", ""),
                }
                for r in results
            ]
            max_display_rate = 1.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 0.5
            self.state.artifact_score_queue.add(items, display_rate)

        self._refresh()

    def update_image(
        self,
        message: str,
        stats: WorkerStats,
        results: list[dict] | None = None,
    ) -> None:
        """Update image VLM worker state."""
        self.state.run_images_scored = stats.processed
        self.state.image_score_rate = stats.rate
        self.state._run_image_score_cost = stats.cost

        # Track processing state
        if "waiting" in message.lower() or message == "idle":
            self.state.image_processing = False
        elif "scoring" in message.lower():
            self.state.image_processing = True
        elif "scored" in message.lower() and results:
            self.state.image_processing = not self.state.image_queue.is_empty()
        else:
            self.state.image_processing = False

        # Queue results for streaming
        if results:
            items = [
                {
                    "id": r.get("id", "unknown"),
                    "caption": r.get("caption", ""),
                    "score": r.get("score"),
                    "physics_domain": r.get("physics_domain"),
                    "description": r.get("description", ""),
                    "purpose": r.get("purpose", ""),
                }
                for r in results
            ]
            max_display_rate = 1.0
            display_rate = min(stats.rate, max_display_rate) if stats.rate else 0.5
            self.state.image_queue.add(items, display_rate)

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
        pending_artifact_score: int = 0,
        pending_artifact_ingest: int = 0,
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
        self.state.pending_artifact_score = pending_artifact_score
        self.state.pending_artifact_ingest = pending_artifact_ingest
        self.state.accumulated_cost = accumulated_cost
        # Update graph-based artifact counts if provided
        if "artifacts_ingested" in kwargs:
            self.state.artifacts_ingested = kwargs["artifacts_ingested"]
        if "artifacts_scored" in kwargs:
            self.state.artifacts_scored = kwargs["artifacts_scored"]
        # Update graph-based image counts if provided
        if "images_scored" in kwargs:
            self.state.images_scored = kwargs["images_scored"]
        if "pending_image_score" in kwargs:
            self.state.pending_image_score = kwargs["pending_image_score"]
        self._refresh()

    def update_worker_status(self, worker_group: SupervisedWorkerGroup) -> None:
        """Update worker status from supervised worker group."""
        self.state.worker_group = worker_group
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
        self.state._offset_artifact_score_cost += self.state._run_artifact_score_cost
        self.state._offset_artifacts += self.state.run_artifacts
        self.state._offset_artifacts_scored += self.state.run_artifacts_scored
        self.state._offset_images_scored += self.state.run_images_scored
        self.state._offset_image_score_cost += self.state._run_image_score_cost

        # Reset per-site counters (new workers will fill these)
        self.state.run_scored = 0
        self.state.run_ingested = 0
        self.state._run_score_cost = 0.0
        self.state._run_ingest_cost = 0.0
        self.state._run_artifact_score_cost = 0.0
        self.state.run_artifacts = 0
        self.state.run_artifacts_scored = 0
        self.state.run_images_scored = 0
        self.state._run_image_score_cost = 0.0
        self.state.score_rate = None
        self.state.ingest_rate = None
        self.state.artifact_rate = None
        self.state.artifact_score_rate = None
        self.state.image_score_rate = None

        # Reset activity displays
        self.state.current_score = None
        self.state.current_ingest = None
        self.state.current_artifact = None
        self.state.current_image = None
        self.state.score_processing = False
        self.state.ingest_processing = False
        self.state.artifact_processing = False
        self.state.image_processing = False

        # Reset worker group (set by new run_parallel_wiki_discovery)
        self.state.worker_group = None

        # Reset streaming queues
        self.state.score_queue = StreamQueue(
            rate=0.5, max_rate=2.5, min_display_time=0.4
        )
        self.state.ingest_queue = StreamQueue(
            rate=0.5, max_rate=2.5, min_display_time=0.4
        )
        self.state.artifact_queue = StreamQueue(
            rate=0.3, max_rate=1.0, min_display_time=0.5
        )
        self.state.artifact_score_queue = StreamQueue(
            rate=0.3, max_rate=1.0, min_display_time=0.5
        )
        self.state.image_queue = StreamQueue(
            rate=0.3, max_rate=1.0, min_display_time=0.5
        )

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

        # SCORE stats (pages + artifacts combined)
        total_scored = self.state.pages_scored + self.state.pages_ingested
        total_score_cost = self.state.total_score_cost
        summary.append("  SCORE ", style="bold blue")
        summary.append(f"scored={total_scored:,}", style="blue")
        artifacts_scored = self.state.total_run_artifacts_scored
        if artifacts_scored > 0:
            summary.append(f"+{artifacts_scored:,}art", style="blue dim")
        summary.append(f"  skipped={self.state.pages_skipped:,}", style="yellow")
        summary.append(f"  cost=${total_score_cost:.3f}", style="yellow")
        if self.state.score_rate:
            summary.append(f"  {self.state.score_rate:.1f}/s", style="dim")
        summary.append("\n")

        # PAGE stats (pages + artifacts combined)
        total_ingested = self.state.pages_ingested
        summary.append("  PAGE  ", style="bold magenta")
        summary.append(f"  ingested={total_ingested:,}", style="magenta")
        artifacts_ingested = self.state.total_run_artifacts
        if artifacts_ingested > 0:
            summary.append(f"+{artifacts_ingested:,}art", style="magenta dim")
        if self.state.ingest_rate:
            summary.append(f"  {self.state.ingest_rate:.1f}/s", style="dim")
        summary.append("\n")

        # IMAGE stats
        images_scored = self.state.total_run_images_scored
        if images_scored > 0:
            summary.append("  IMAGE ", style="bold green")
            summary.append(f"  scored={images_scored:,}", style="green")
            if self.state.image_score_rate:
                summary.append(f"  {self.state.image_score_rate:.1f}/s", style="dim")
            summary.append("\n")

        # USAGE stats
        summary.append("  USAGE ", style="bold white")
        summary.append(f"time={format_time(self.state.elapsed)}", style="white")
        summary.append(f"  total_cost=${self.state.run_cost:.2f}", style="yellow")
        if self.state.total_sites > 1:
            summary.append(
                f"  sites={self.state.current_site_index + 1}/{self.state.total_sites}",
                style="cyan",
            )

        # Show coverage percentage
        if self.state.total_pages > 0:
            coverage = total_scored / self.state.total_pages * 100
            summary.append(f"  coverage={coverage:.1f}%", style="cyan")

        return summary
