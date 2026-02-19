"""
Common progress display infrastructure for discovery operations.

Provides reusable components for building Rich-based progress displays
across different discovery domains (paths, wiki, code, data).

Design Principles:
- Composable: Build domain-specific displays from common components
- Consistent: Same visual language across all discovery commands
- Clean: Minimal visual clutter (no emojis, thin progress bars)
- Streaming: Rate-limited queues for smooth updates

Usage:
    from imas_codex.discovery.base.progress import (
        ProgressConfig,
        StreamQueue,
        format_time,
        clip_path,
        make_bar,
        make_resource_gauge,
        build_header,
        build_resource_section,
    )

    class MyProgressDisplay(BaseProgressDisplay):
        def __init__(self, ...):
            self.config = ProgressConfig(facility="tcv", cost_limit=10.0)
            ...
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    pass

# Strip ANSI escape codes from text (in case LLM output contains them)
_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def clean_text(text: str) -> str:
    """Remove ANSI codes and escape Rich markup."""
    return escape(_ANSI_PATTERN.sub("", text))


# =============================================================================
# Formatting Utilities
# =============================================================================


def clip_path(path: str, max_len: int = 63) -> str:
    """Clip middle of path with ellipsis: /home/user/.../deep/dir

    Center-clips the path, keeping the start (context) and end (specificity).
    The /.../ format is distinctive and indicates path truncation.
    """
    if len(path) <= max_len:
        return path
    # Keep more of the end (specific part) than the start
    keep_start = max_len // 3
    keep_end = max_len - keep_start - 5  # 5 for "/.../"
    if keep_end < 10:
        # Very short max_len - just clip end
        return path[: max_len - 3] + "..."
    return f"{path[:keep_start]}/.../{path[-keep_end:]}"


def clip_text(text: str, max_len: int = 60) -> str:
    """Clip end of text with ellipsis: Long description text...

    End-clips for descriptions/reasons where the start is most important.
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_time(seconds: float) -> str:
    """Format duration: 1h 23m, 5m 30s, 45s"""
    if seconds < 0:
        return "--"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs:02d}s" if secs else f"{mins}m"
    hours, rem = divmod(int(seconds), 3600)
    mins = rem // 60
    return f"{hours}h {mins:02d}m" if mins else f"{hours}h"


def format_bytes(num_bytes: int) -> str:
    """Format bytes: 1.5MB, 256KB, 1.2GB"""
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.1f}GB"
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.1f}MB"
    if num_bytes >= 1_000:
        return f"{num_bytes / 1_000:.1f}KB"
    return f"{num_bytes}B"


def format_count(count: int) -> str:
    """Format large counts: 1.5M, 256K, 1234"""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


# =============================================================================
# Progress Bar Utilities
# =============================================================================


def make_bar(
    ratio: float, width: int, filled_char: str = "━", empty_char: str = "─"
) -> str:
    """Create a simple thin progress bar string."""
    ratio = max(0.0, min(1.0, ratio))
    filled = int(width * ratio)
    return filled_char * filled + empty_char * (width - filled)


def make_gradient_bar(ratio: float, width: int) -> Text:
    """Create a gradient progress bar (green → yellow → red as it fills)."""
    ratio = max(0.0, min(1.0, ratio))
    filled = int(width * ratio)

    bar = Text()
    for i in range(width):
        if i < filled:
            # Gradient based on position
            pos_ratio = i / width
            if pos_ratio < 0.5:
                bar.append("━", style="green")
            elif pos_ratio < 0.75:
                bar.append("━", style="yellow")
            else:
                bar.append("━", style="red")
        else:
            bar.append("─", style="dim")
    return bar


def make_resource_gauge(
    used: float,
    limit: float,
    width: int = 20,
    unit: str = "",
    borders: bool = False,
) -> Text:
    """Create a resource consumption gauge with color coding.

    Args:
        used: Current value consumed
        limit: Maximum value (100% of gauge)
        width: Width of the gauge bar in characters
        unit: Unused, kept for compatibility
        borders: If True, add │ borders around gauge (legacy)
    """
    ratio = used / limit if limit > 0 else 0
    ratio = max(0.0, min(1.0, ratio))

    # Color based on consumption
    if ratio < 0.5:
        color = "green"
    elif ratio < 0.8:
        color = "yellow"
    else:
        color = "red"

    filled = int(width * ratio)

    gauge = Text()
    if borders:
        gauge.append("│", style="dim")
    gauge.append("━" * filled, style=color)
    gauge.append("─" * (width - filled), style="dim")
    if borders:
        gauge.append("│", style="dim")

    return gauge


# =============================================================================
# Streaming Queue
# =============================================================================


@dataclass
class StreamQueue:
    """Rate-limited queue for smooth display updates.

    Prevents rapid flickering by:
    1. Rate-limiting pops to a maximum rate (default 2.0/s)
    2. Enforcing minimum display time per item (default 0.4s)
    3. Larger queue buffer to absorb batch processing bursts

    The min_display_time ensures items stay visible long enough to read,
    even when processing is fast. This prevents the jarring path/idle flicker.

    Stale detection: tracks when items were last added. When the queue is
    empty and no items have been added for ``stale_timeout`` seconds, the
    queue is considered stale — meaning the displayed item should be cleared.
    """

    items: deque = field(default_factory=deque)
    last_pop: float = field(default_factory=time.time)
    last_add: float = 0.0  # timestamp of last add() call
    rate: float = 2.0  # items per second (capped max rate)
    max_rate: float = 2.5  # never exceed this rate even if worker is faster
    min_display_time: float = 0.4  # minimum seconds each item stays visible
    max_size: int = 500  # larger buffer to absorb batch bursts
    stale_timeout: float = 3.0  # seconds without adds before queue is stale

    def add(self, items: list, rate: float | None = None) -> None:
        """Add items to queue.

        The rate is capped at max_rate to ensure smooth display.
        """
        self.items.extend(items)
        self.last_add = time.time()
        if rate and rate > 0:
            # Cap at max_rate to prevent too-fast streaming
            self.rate = min(rate, self.max_rate)
        # Drop oldest items if queue exceeds max size
        while len(self.items) > self.max_size:
            self.items.popleft()

    def pop(self) -> Any | None:
        """Pop next item if rate limit allows.

        Enforces both rate limiting and minimum display time.
        """
        if not self.items:
            return None
        # Use the slower of: rate interval or min_display_time
        rate_interval = 1.0 / self.rate if self.rate > 0 else 0.5
        interval = max(rate_interval, self.min_display_time)
        now = time.time()
        if now - self.last_pop >= interval:
            self.last_pop = now
            return self.items.popleft()
        return None

    def is_empty(self) -> bool:
        """Check if queue has no pending items."""
        return len(self.items) == 0

    def is_stale(self) -> bool:
        """Check if queue is empty and no items added recently.

        Returns True when the queue has drained AND no new items have
        been added for ``stale_timeout`` seconds, indicating the worker
        has stopped producing and the current displayed item is outdated.
        """
        if not self.is_empty():
            return False
        if self.last_add == 0.0:
            return False  # never received items
        return (time.time() - self.last_add) >= self.stale_timeout

    def clear(self) -> None:
        """Clear the queue. Use on termination to prevent hanging."""
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)


# =============================================================================
# Worker Statistics
# =============================================================================


@dataclass
class WorkerStats:
    """Statistics for a single async worker.

    Reusable across discovery domains (paths, wiki, etc.).

    Tracks both a lifetime average rate and an exponential moving average
    (EMA) over recent batches for smoother, more responsive rate display.
    """

    processed: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    last_batch_time: float = 0.0
    cost: float = 0.0  # LLM cost (for scorer workers)

    # EMA rate tracking
    _ema_rate: float = 0.0  # Exponential moving average rate (items/s)
    _ema_alpha: float = 0.3  # Smoothing factor (0.3 = responsive to recent)
    _prev_processed: int = 0  # Items at last batch
    _prev_batch_time: float = 0.0  # Time at last batch

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def rate(self) -> float | None:
        """Overall average rate (items/second) since start."""
        if self.processed == 0 or self.elapsed <= 0:
            return None
        return self.processed / self.elapsed

    @property
    def ema_rate(self) -> float | None:
        """Exponential moving average rate over recent batches.

        Falls back to overall rate if no batch data available.
        """
        if self._ema_rate > 0:
            return self._ema_rate
        return self.rate

    def record_batch(self, batch_size: int | None = None) -> None:
        """Record a batch completion for EMA rate calculation.

        Call after each batch of items is processed. Uses delta between
        calls to compute instantaneous rate, then smooths with EMA.

        Args:
            batch_size: Items in this batch. If None, inferred from
                ``processed - _prev_processed``.
        """
        now = time.time()
        if batch_size is None:
            batch_size = self.processed - self._prev_processed
        if batch_size <= 0:
            return

        dt = now - self._prev_batch_time if self._prev_batch_time > 0 else 0
        if dt > 0:
            instant_rate = batch_size / dt
            if self._ema_rate > 0:
                self._ema_rate = (
                    self._ema_alpha * instant_rate
                    + (1 - self._ema_alpha) * self._ema_rate
                )
            else:
                self._ema_rate = instant_rate

        self._prev_processed = self.processed
        self._prev_batch_time = now


# =============================================================================
# Progress Configuration
# =============================================================================


# =============================================================================
# Standard Layout Constants
# =============================================================================
#
# All discovery progress displays share the same layout grid to ensure
# consistent visual alignment.  Domain-specific displays import these
# constants rather than redefining them.
#
#   LABEL_WIDTH       – left label column ("  SCAN    ", "  COST    ")
#   METRICS_WIDTH     – right stat column for progress bars (" {n:>6,} {%} {r/s}")
#   GAUGE_METRICS_WIDTH – right stat column for resource gauges (wider: time/cost text)
#   MIN_WIDTH         – minimum panel width
#
# Bar and gauge widths are computed from terminal width:
#   bar_width   = term_width - 4 - LABEL_WIDTH - METRICS_WIDTH
#   gauge_width = term_width - 4 - LABEL_WIDTH - GAUGE_METRICS_WIDTH
#
# The difference prevents resource-gauge rows (TIME, COST, TOTAL) from
# wrapping by reserving extra space for their longer trailing text.

LABEL_WIDTH = 10
METRICS_WIDTH = 22
GAUGE_METRICS_WIDTH = 32
MIN_WIDTH = 80


def compute_bar_width(term_width: int) -> int:
    """Progress-bar width from terminal width."""
    return max(term_width, MIN_WIDTH) - 4 - LABEL_WIDTH - METRICS_WIDTH


def compute_gauge_width(term_width: int) -> int:
    """Resource-gauge width from terminal width."""
    return max(term_width, MIN_WIDTH) - 4 - LABEL_WIDTH - GAUGE_METRICS_WIDTH


@dataclass
class ProgressConfig:
    """Common configuration for progress displays."""

    facility: str
    cost_limit: float = 10.0
    item_limit: int | None = None  # --limit flag (paths, pages, etc.)
    model: str = ""
    focus: str = ""

    # Display settings
    width: int = 88
    bar_width: int = 40
    gauge_width: int = 20

    # Mode flags (domain-specific)
    scan_only: bool = False
    score_only: bool = False


# =============================================================================
# Base Progress State
# =============================================================================


@dataclass
class BaseProgressState:
    """Base class for domain-specific progress states.

    Provides common timing functionality. Extend this for paths, wiki, etc.
    """

    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        """Seconds since start."""
        return time.time() - self.start_time


# =============================================================================
# Unified Pipeline Section (progress + activity merged)
# =============================================================================


@dataclass
class PipelineRowConfig:
    """Configuration for a unified pipeline row (3 lines).

    Unified pipeline row combining progress bar and current activity
    into a single block.  Each pipeline stage renders as:

        TRIAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    2,238  29%  0.2/s  $8.30
               Mailinglists                                            ×4
               0.00  general  Information regarding SPC mailing lists...

    This is the standard layout for discovery CLI tools that want
    integrated per-stage progress and activity display.
    """

    # Identity
    name: str  # e.g., "TRIAGE", "PAGES", "DOCS", "IMAGES"
    style: str  # e.g., "bold blue"

    # Progress bar data
    completed: int = 0
    total: int = 1
    rate: float | None = None  # items/second (EMA preferred)
    cost: float | None = None  # LLM/VLM cost for this stage
    disabled: bool = False
    disabled_msg: str = "disabled"
    show_pct: bool = True

    # Worker annotation
    worker_count: int = 0  # Number of workers in this group
    worker_annotation: str = ""  # e.g., "(1 backoff)" or "(budget)"

    # Activity (current item)
    primary_text: str = ""  # Line 2: current item title/path
    detail_parts: list[tuple[str, str]] | None = None  # Line 3: score+domain+desc

    # Activity state (used when no primary_text)
    is_processing: bool = False
    processing_label: str = "processing..."
    is_complete: bool = False
    complete_label: str = "complete"
    is_paused: bool = False
    queue_size: int = 0

    @property
    def has_content(self) -> bool:
        """True when an item is available to display."""
        return bool(self.primary_text)


def build_pipeline_row(config: PipelineRowConfig, bar_width: int = 40) -> Text:
    """Build a unified pipeline row (progress bar + activity).

    Renders 3 lines:
      Line 1: label + progress bar + count + pct + rate + cost
      Line 2: current item title + worker count annotation
      Line 3: detail (score, domain, description)

    Args:
        config: Pipeline row configuration.
        bar_width: Width of the progress bar.
    """
    row = Text()

    if config.disabled:
        row.append(f"  {config.name:<6} ", style="dim")
        row.append("─" * bar_width, style="dim")
        row.append(f"    {config.disabled_msg}", style="dim italic")
        return row

    # Line 1: Label + progress bar + metrics
    row.append(f"  {config.name:<6} ", style=config.style)

    total = max(config.total, 1)
    ratio = min(config.completed / total, 1.0)
    pct = ratio * 100
    row.append(make_bar(ratio, bar_width), style=config.style.split()[-1])

    row.append(f" {config.completed:>6,}", style="bold")
    if config.show_pct:
        row.append(f" {pct:>3.0f}%", style="cyan")
    else:
        row.append("     ", style="dim")

    if config.rate and config.rate > 0:
        row.append(f" {config.rate:>5.1f}/s", style="dim")

    if config.cost is not None and config.cost > 0:
        row.append(f"  ${config.cost:.2f}", style="yellow dim")

    # Line 2: Current item + worker count
    row.append("\n")
    row.append("         ", style="dim")  # indent to align with bar

    if config.has_content:
        row.append(config.primary_text, style="white")
    elif config.is_processing:
        label = "paused" if config.is_paused else config.processing_label
        style = "dim italic" if config.is_paused else "cyan italic"
        row.append(label, style=style)
    elif config.queue_size > 0:
        row.append(f"streaming {config.queue_size} items...", style="cyan italic")
    elif config.is_complete:
        row.append(config.complete_label, style="green")
    elif config.is_paused:
        row.append("paused", style="dim italic")
    else:
        row.append("idle", style="dim italic")

    # Worker count annotation (right-aligned conceptually, appended after text)
    if config.worker_count > 0:
        annotation = f"  ×{config.worker_count}"
        if config.worker_annotation:
            annotation += f" {config.worker_annotation}"
        row.append(annotation, style="dim")

    # Line 3: Detail parts (score + domain + description)
    row.append("\n")
    row.append("    ", style="dim")  # indent
    if config.detail_parts:
        for text, style in config.detail_parts:
            row.append(text, style=style)

    return row


def build_pipeline_section(
    rows: list[PipelineRowConfig],
    bar_width: int = 40,
) -> Text:
    """Build a unified pipeline section with merged progress + activity.

    Each stage gets a 3-line block: progress bar, current item, detail.
    Stages are separated by a blank line for readability.

    This is the standard layout for discovery CLIs that want
    per-stage visibility.

    Args:
        rows: Pipeline row configurations (one per stage).
        bar_width: Width of progress bars.
    """
    section = Text()
    for i, config in enumerate(rows):
        if config.disabled:
            continue
        if i > 0:
            section.append("\n")
        section.append_text(build_pipeline_row(config, bar_width))
    return section


# =============================================================================
# Common Display Sections
# =============================================================================


def build_worker_status_section(
    worker_group: Any | None,
    *,
    budget_exhausted: bool = False,
    is_paused: bool = False,
    budget_sensitive_groups: set[str] | None = None,
    extra_indicators: list[tuple[str, str]] | None = None,
) -> Text:
    """Build WORKERS status section grouped by functional role.

    Groups workers by their ``group`` field (set during ``create_status()``).
    Shows count, state, and budget/failure annotations per group.

    Workers without a group are grouped by name heuristic (legacy fallback).

    Args:
        worker_group: ``SupervisedWorkerGroup`` with status tracking.
        budget_exhausted: True when cost limit reached.
        is_paused: True when services are degraded (all dimmed).
        budget_sensitive_groups: Group names that stop on budget
            exhaustion.  Default: ``{"score"}``.
        extra_indicators: Additional indicators to append after
            worker groups, e.g. ``[("embed:remote", "green")]``.
    """
    from imas_codex.discovery.base.supervision import WorkerState

    if budget_sensitive_groups is None:
        budget_sensitive_groups = {"score"}

    section = Text()
    section.append("  WORKERS", style="dim" if is_paused else "bold green")

    if not worker_group:
        section.append("  starting...", style="dim italic")
        return section

    # Collect workers into groups
    groups: dict[str, list[tuple[str, WorkerState]]] = {}
    for name, status in worker_group.workers.items():
        grp = status.group or name.split("_worker")[0]
        groups.setdefault(grp, []).append((name, status.state))

    # Render each group
    for grp, workers in groups.items():
        count = len(workers)
        state_counts: dict[WorkerState, int] = {}
        for _, st in workers:
            state_counts[st] = state_counts.get(st, 0) + 1

        all_stopped = state_counts.get(WorkerState.stopped, 0) == count
        is_budget_group = grp in budget_sensitive_groups
        budget_stopped = all_stopped and is_budget_group and budget_exhausted

        # Determine display style
        if is_paused:
            style = "dim"
        elif state_counts.get(WorkerState.crashed, 0) > 0:
            style = "red"
        elif state_counts.get(WorkerState.backoff, 0) > 0:
            style = "yellow"
        elif budget_stopped:
            style = "yellow"
        elif state_counts.get(WorkerState.running, 0) > 0:
            style = "green"
        elif all_stopped:
            style = "green"
        else:
            style = "dim"

        section.append(f"  {grp}:{count}", style=style)

        # Annotations
        running = state_counts.get(WorkerState.running, 0)
        backing_off = state_counts.get(WorkerState.backoff, 0)
        failed = state_counts.get(WorkerState.crashed, 0)

        if budget_stopped:
            section.append(" (budget)", style="yellow dim")
        elif backing_off > 0 or failed > 0:
            parts: list[str] = []
            if running > 0:
                parts.append(f"{running} active")
            if backing_off > 0:
                parts.append(f"{backing_off} backoff")
            if failed > 0:
                parts.append(f"{failed} failed")
            section.append(f" ({', '.join(parts)})", style="dim")

    # Extra indicators (e.g. embed source)
    if extra_indicators:
        for label, style in extra_indicators:
            section.append(f"  {label}", style=style)

    return section


def build_servers_section(
    statuses: list | None = None,
) -> Text | None:
    """Build the SERVERS status row for progress displays.

    Shows live status of external service dependencies (graph, embed, SSH/VPN).
    Returns None if no services are monitored.

    Args:
        statuses: List of ServiceStatus objects from ServiceMonitor.get_status()

    Format:
      SERVERS  graph:iter-login  embed:iter-login  ssh:jt-60sa  auth:vpn
    """
    if not statuses:
        return None

    from imas_codex.discovery.base.services import ServiceState

    section = Text()
    section.append("  SERVERS", style="bold white")

    for s in statuses:
        if s.state == ServiceState.healthy:
            style = "green"
            label = s.detail or "ok"
        elif s.state == ServiceState.unknown:
            # Pending initial check — show grey "pending" instead of "unknown"
            style = "dim"
            label = "pending"
        elif s.state == ServiceState.recovering:
            style = "yellow"
            label = f"recovering ({int(s.downtime_seconds)}s)"
        else:
            # Unhealthy: show grayed label with concise reason
            style = "dim"
            if s.healthy_detail:
                # Was healthy before — show last-known good state grayed out
                label = s.healthy_detail
            elif s.auth_label:
                label = s.auth_label
            else:
                # Derive concise label from error detail
                detail = (s.detail or "").lower()
                if "timeout" in detail or "timed out" in detail:
                    label = "down"
                elif "connection refused" in detail:
                    label = "down"
                elif "no route" in detail:
                    label = "down"
                else:
                    label = "down"
            if s.downtime_seconds > 0:
                label += f" ({int(s.downtime_seconds)}s)"

        section.append(f"  {s.name}:", style="dim")
        section.append(label, style=style)

    return section


def build_header(
    config: ProgressConfig,
    title_suffix: str = "Discovery",
    mode_label: str | None = None,
) -> Text:
    """Build centered header with facility and focus.

    Args:
        config: Progress configuration
        title_suffix: Suffix for title (e.g., "Discovery", "Scan")
        mode_label: Optional mode indicator (e.g., "SCAN ONLY")
    """
    header = Text()

    # Facility name with mode indicator
    title = f"{config.facility.upper()} {title_suffix}"
    if mode_label:
        title += f" ({mode_label})"
    header.append(title.center(config.width - 4), style="bold cyan")

    # Focus (if set and not scan_only)
    if config.focus and not config.scan_only:
        header.append("\n")
        focus_line = f"Focus: {config.focus}"
        header.append(focus_line.center(config.width - 4), style="italic dim")

    return header


# =============================================================================
# Base Progress Display
# =============================================================================


class BaseProgressDisplay(ABC):
    """Abstract base class for discovery progress displays.

    Provides common infrastructure for Live display management.
    Subclasses implement domain-specific rendering logic.
    """

    def __init__(
        self,
        config: ProgressConfig,
        console: Console | None = None,
    ) -> None:
        self.config = config
        self.console = console or Console()
        self._live: Live | None = None
        self.start_time = time.time()

    @property
    def elapsed(self) -> float:
        """Elapsed time since display started."""
        return time.time() - self.start_time

    def __enter__(self) -> BaseProgressDisplay:
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

    @abstractmethod
    def _build_display(self) -> Panel:
        """Build the complete display panel.

        Subclasses must implement this to render their specific content.
        """
        ...

    def tick(self) -> None:  # noqa: B027
        """Drain streaming queues for smooth display.

        Override in subclasses to pop from domain-specific queues.
        Default implementation does nothing.
        """


# =============================================================================
# Unified Resource Section Builder
# =============================================================================


@dataclass
class ResourceConfig:
    """Configuration for the resource consumption section.

    Covers TIME, COST, TOTAL, and STATS rows.
    """

    elapsed: float
    eta: float | None = None

    # Cost tracking
    run_cost: float | None = None
    cost_limit: float | None = None
    accumulated_cost: float = 0.0
    etc: float | None = None  # Estimated Total Cost (pre-computed by caller)

    # Customisation
    scan_only: bool = False
    limit_reason: str | None = None  # "cost", "path", "page", etc.

    # STATS row: list of (label, value, style) tuples
    stats: list[tuple[str, str, str]] | None = None

    # STATS row: pending counts as (label, count) tuples
    pending: list[tuple[str, int]] | None = None


def build_resource_section(
    config: ResourceConfig,
    gauge_width: int = 20,
) -> Text:
    """Build the RESOURCES section (TIME, COST, TOTAL, STATS).

    TIME and COST rows show informational metrics without limit-based gauges.
    TIME shows elapsed + ETA. COST shows spend + ETC projection.

    Args:
        config: Resource configuration.
        gauge_width: Width of resource gauge bars (use ``compute_gauge_width()``).
    """
    section = Text()

    # TIME row — elapsed ticker, no limit gauge
    section.append("  TIME    ", style="bold cyan")

    eta = None if config.scan_only else config.eta
    if eta is not None and eta > 0:
        # Show work-based progress (elapsed / estimated-total)
        total_est = config.elapsed + eta
        section.append_text(make_resource_gauge(config.elapsed, total_est, gauge_width))
    else:
        section.append("━" * gauge_width, style="cyan")

    section.append(f"  {format_time(config.elapsed)}", style="bold")

    if eta is not None:
        if eta <= 0:
            if config.limit_reason:
                section.append(
                    f"  {config.limit_reason} limit reached", style="yellow dim"
                )
            else:
                section.append("  complete", style="green dim")
        else:
            section.append(f"  ETA {format_time(eta)}", style="dim")
    section.append("\n")

    # COST row — informational spend display, no limit gauge
    if not config.scan_only and config.run_cost is not None:
        section.append("  COST    ", style="bold yellow")

        # Show spend-to-projection gauge when ETC is available
        total_cost = config.accumulated_cost + config.run_cost
        etc = config.etc
        if etc is not None and etc > total_cost:
            section.append_text(make_resource_gauge(total_cost, etc, gauge_width))
        else:
            section.append("━" * gauge_width, style="yellow")

        section.append(f"  ${config.run_cost:.2f}", style="bold")
        if etc is not None and etc > total_cost:
            section.append(f"  ETC ${etc:.2f}", style="dim")
        elif config.accumulated_cost > 0:
            section.append(f"  total ${total_cost:.2f}", style="dim")
        section.append("\n")

    # STATS row
    if config.stats:
        section.append("  STATS   ", style="bold magenta")
        for i, (label, value, style) in enumerate(config.stats):
            if i > 0:
                section.append("  ", style="dim")
            section.append(f"{label}={value}", style=style)

        # Pending work
        if config.pending:
            active = [(label, count) for label, count in config.pending if count > 0]
            if active:
                parts = [f"{label}:{count}" for label, count in active]
                section.append(f"  pending=[{' '.join(parts)}]", style="cyan dim")

    return section
