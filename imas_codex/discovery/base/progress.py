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

from rich.cells import cell_len
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from imas_codex.discovery.base.supervision import (
    SupervisedWorkerGroup,
    WorkerState,
)

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
    Uses cell_len for accurate terminal width.
    """
    if cell_len(path) <= max_len:
        return path
    # Keep more of the end (specific part) than the start
    keep_start = max_len // 3
    keep_end = max_len - keep_start - 5  # 5 for "/.../"
    if keep_end < 10:
        # Very short max_len - just clip end
        return clip_text(path, max_len)
    return f"{path[:keep_start]}/.../{path[-keep_end:]}"


def clip_text(text: str, max_len: int = 60) -> str:
    """Clip end of text with ellipsis: Long description text...

    End-clips for descriptions/reasons where the start is most important.
    Uses cell_len for accurate terminal width (handles wide Unicode chars).
    """
    if cell_len(text) <= max_len:
        return text
    # Trim character by character until display width fits
    target = max_len - 3  # room for "..."
    result = []
    width = 0
    for ch in text:
        ch_width = cell_len(ch)
        if width + ch_width > target:
            break
        result.append(ch)
        width += ch_width
    return "".join(result) + "..."


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

    # Idle time tracking (excluded from active_rate)
    _idle_total: float = 0.0  # Cumulative idle seconds
    _idle_start: float | None = None  # When current idle period began

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
    def active_rate(self) -> float | None:
        """Average rate excluding idle time (items/second).

        If no idle time was tracked, falls back to overall rate.
        """
        if self.processed == 0:
            return None
        idle = self._idle_total
        if self._idle_start is not None:
            idle += time.time() - self._idle_start
        active = self.elapsed - idle
        if active <= 0:
            return self.rate
        return self.processed / active

    @property
    def ema_rate(self) -> float | None:
        """Exponential moving average rate over recent batches.

        Falls back to active_rate if no batch data available.
        """
        if self._ema_rate > 0:
            return self._ema_rate
        return self.active_rate

    def mark_idle(self) -> None:
        """Mark the start of an idle period."""
        if self._idle_start is None:
            self._idle_start = time.time()

    def mark_active(self) -> None:
        """Mark the end of an idle period, accumulating idle time."""
        if self._idle_start is not None:
            self._idle_total += time.time() - self._idle_start
            self._idle_start = None

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
#   LABEL_WIDTH       – left label column ("  ARTIFACTx2", "  TIME")
#   METRICS_WIDTH     – right stat column for progress bars (" {n:>6,} {%}")
#   GAUGE_METRICS_WIDTH – right stat column for resource gauges (wider: time/cost text)
#   MIN_WIDTH         – minimum panel width
#
# Bar and gauge widths are computed from terminal width:
#   bar_width   = term_width - 4 - LABEL_WIDTH - METRICS_WIDTH
#   gauge_width = term_width - 4 - LABEL_WIDTH - GAUGE_METRICS_WIDTH
#
# Pipeline line 1 has just count+pct on the right (METRICS_WIDTH).
# Rate and cost are right-aligned on line 2 to the same edge.
# Resource gauges (TIME, COST) keep GAUGE_METRICS_WIDTH for trailing text.

LABEL_WIDTH = 10
METRICS_WIDTH = 12
GAUGE_METRICS_WIDTH = 22
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

        TRIAGEx4  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  7,782 100%
        0.85  electromagnetic  Thomson Scattering            0.23/s
        General documentation for Thomson Scattering        $12.54

    Line 1: NAMExN + bar + count + pct
    Line 2: score + domain + name … rate (right-aligned)
    Line 3: description … cost (right-aligned below rate)

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

    # Activity (current item) — structured fields (preferred)
    primary_text: str = ""  # Resource name (shown on line 2)
    score_value: float | None = None  # Score (shown on line 2, left-aligned with name)
    physics_domain: str = ""  # Physics domain (shown on line 2, at bar start)
    description: str = ""  # LLM/VLM description (shown on line 3, at bar start)
    description_fallback: str = ""  # Shown on line 3 when description is empty

    # Legacy: detail_parts as (text, style) tuples — used when structured fields are not set
    detail_parts: list[tuple[str, str]] | None = None

    # Activity state (used when no primary_text)
    is_processing: bool = False
    processing_label: str = "processing..."
    is_complete: bool = False
    complete_label: str = "done"
    is_paused: bool = False
    queue_size: int = 0

    @property
    def has_content(self) -> bool:
        """True when an item is available to display."""
        return bool(self.primary_text)

    @property
    def has_structured_detail(self) -> bool:
        """True when structured fields are populated (preferred over detail_parts)."""
        return bool(
            self.score_value is not None
            or self.physics_domain
            or self.description
            or self.description_fallback
        )


def build_pipeline_row(config: PipelineRowConfig, bar_width: int = 40) -> Text:
    """Build a unified pipeline row (progress bar + activity).

    Renders 3 lines:
      Line 1: NAMExN + bar + count + pct
      Line 2: score + domain + name … rate (right-aligned)
      Line 3: description … cost (right-aligned below rate)

    Args:
        config: Pipeline row configuration.
        bar_width: Width of the progress bar.
    """
    row = Text()
    # Total content width for lines 2/3 right-alignment
    row_width = LABEL_WIDTH + bar_width + METRICS_WIDTH

    if config.disabled:
        row.append(f"  {config.name}".ljust(LABEL_WIDTH), style="dim")
        row.append("─" * bar_width, style="dim")
        row.append(f"    {config.disabled_msg}", style="dim italic")
        return row

    # ── Line 1: label + progress bar + count + pct ──
    label = Text()
    label.append(f"  {config.name}", style=config.style)
    if config.worker_count > 0:
        label.append(f"x{config.worker_count}", style="dim")
    if config.worker_annotation:
        label.append(f" {config.worker_annotation}", style="dim")
    # Pad to LABEL_WIDTH for aligned bar start
    label_len = len(label.plain)
    label_pad = LABEL_WIDTH - label_len
    if label_pad > 0:
        label.append(" " * label_pad)
    row.append_text(label)

    # Shorten bar when label overflows LABEL_WIDTH to keep line 1 on one line
    effective_bar = bar_width - max(0, label_len - LABEL_WIDTH)

    total = max(config.total, 1)
    ratio = min(config.completed / total, 1.0)
    pct = ratio * 100
    # When displayed percentage rounds to 100%, fill the bar completely
    # to avoid a visual gap between "100%" text and a 99.x% bar
    bar_ratio = 1.0 if round(pct) >= 100 else ratio
    row.append(make_bar(bar_ratio, effective_bar), style=config.style.split()[-1])

    # Right: count + pct only
    count_s = f" {config.completed:>6,}"
    pct_s = f" {pct:>3.0f}%" if config.show_pct else "     "
    pad = max(0, METRICS_WIDTH - len(count_s) - len(pct_s))
    if pad > 0:
        row.append(" " * pad)
    row.append(count_s, style="bold")
    row.append(pct_s, style="cyan" if config.show_pct else "dim")

    # ── Line 2: activity info (left) + rate (right-aligned) ──
    row.append("\n")
    line2 = Text()

    # Pre-compute rate text so we can clip left content to fit
    rate_s = ""
    if config.rate and config.rate > 0:
        rate_s = f"{config.rate:.2f}/s"
    rate_reserve = (len(rate_s) + 4) if rate_s else 0  # 4-char gap

    if config.has_content and config.has_structured_detail:
        line2.append("  ", style="dim")
        # Order: score → domain → name
        if config.score_value is not None:
            score_style = (
                "bold green"
                if config.score_value >= 0.7
                else "yellow"
                if config.score_value >= 0.4
                else "red"
            )
            line2.append(f"{config.score_value:.2f}", style=score_style)
            line2.append("  ", style="dim")
        if config.physics_domain:
            line2.append(config.physics_domain, style="cyan")
            line2.append("  ", style="dim")
        max_name = max(10, row_width - cell_len(line2.plain) - rate_reserve)
        line2.append(clip_text(config.primary_text, max_name), style="white")
    elif config.has_content:
        line2.append("  ", style="dim")
        max_name = max(10, row_width - cell_len(line2.plain) - rate_reserve)
        line2.append(clip_text(config.primary_text, max_name), style="white")
    else:
        # No content: show status text
        line2.append("  ", style="dim")
        if config.is_processing:
            lbl = "paused" if config.is_paused else config.processing_label
            sty = "dim italic" if config.is_paused else "cyan italic"
            line2.append(lbl, style=sty)
        elif config.queue_size > 0:
            line2.append(f"streaming {config.queue_size} items...", style="cyan italic")
        elif config.is_complete:
            line2.append(config.complete_label, style="dim italic")
        elif config.is_paused:
            line2.append("paused", style="dim italic")
        else:
            line2.append("idle", style="dim italic")

    # Right-align rate on line 2
    if rate_s:
        gap = max(1, row_width - cell_len(line2.plain) - len(rate_s))
        line2.append(" " * gap)
        line2.append(rate_s, style="dim")
    row.append_text(line2)

    # ── Line 3: description (left) + cost (right-aligned with rate) ──
    row.append("\n")
    line3 = Text()

    # Pre-compute cost text so we can clip left content to fit
    cost_s = ""
    if config.cost is not None and config.cost > 0:
        cost_s = f"${config.cost:.2f}"
    cost_reserve = (len(cost_s) + 4) if cost_s else 0  # 4-char gap

    if config.has_structured_detail:
        line3.append("  ", style="dim")
        if config.description:
            desc = clean_text(config.description)
            max_desc = max(10, row_width - 4 - cost_reserve)
            line3.append(clip_text(desc, max_desc), style="italic dim")
        elif config.description_fallback:
            max_fb = max(10, row_width - 4 - cost_reserve)
            line3.append(
                clip_text(config.description_fallback, max_fb),
                style="cyan dim italic",
            )
    elif config.detail_parts:
        line3.append("  ", style="dim")
        max_detail = max(10, row_width - 4 - cost_reserve)
        detail_len = 0
        for text, style in config.detail_parts:
            remaining = max_detail - detail_len
            if remaining <= 0:
                break
            if cell_len(text) > remaining:
                text = clip_text(text, remaining)
            line3.append(text, style=style)
            detail_len += cell_len(text)
    # Right-align cost on line 3 (below rate on line 2)
    if cost_s:
        gap = max(1, row_width - cell_len(line3.plain) - len(cost_s))
        line3.append(" " * gap)
        line3.append(cost_s, style="dim")
    row.append_text(line3)

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
            # Unhealthy: show concise error reason from health check
            style = "dim"
            detail = (s.detail or "").lower()
            if s.healthy_detail:
                # Was healthy before — show last-known good state grayed out
                label = s.healthy_detail
            elif s.auth_label:
                label = s.auth_label
            elif "402" in detail or "budget" in detail or "insufficient" in detail:
                label = "no credit"
                style = "yellow"
            elif "401" in detail or "auth" in detail or "api_key" in detail:
                label = "auth error"
            elif "429" in detail or "rate" in detail:
                label = "rate limited"
                style = "yellow"
            elif "timeout" in detail or "timed out" in detail:
                label = "timeout"
            elif "connection refused" in detail:
                label = "refused"
            elif "no route" in detail or "unreachable" in detail:
                label = "unreachable"
            elif "proxy" in detail or "502" in detail or "503" in detail:
                label = "proxy down"
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

    Provides common infrastructure for Live display management,
    worker status tracking, headers, server sections, and the
    standard HEADER → SERVERS → PIPELINE → RESOURCES layout.

    Subclasses must implement:
    - ``_build_pipeline_section()`` — domain-specific pipeline rows
    - ``_build_resources_section()`` — domain-specific resource gauges

    Subclasses may override:
    - ``_header_title_suffix()`` — e.g. "Wiki Discovery", "Signal Discovery"
    - ``_header_mode_label()`` — e.g. "SCAN ONLY"
    """

    def __init__(
        self,
        facility: str,
        cost_limit: float,
        console: Console | None = None,
        focus: str = "",
        title_suffix: str = "Discovery",
    ) -> None:
        self.facility = facility
        self.cost_limit = cost_limit
        self.console = console or Console()
        self.focus = focus
        self._title_suffix = title_suffix
        self._live: Live | None = None
        self.start_time = time.time()
        # Service monitor (set by CLI after construction)
        self.service_monitor: Any = None
        # Worker group (set by on_worker_status callback)
        self.worker_group: SupervisedWorkerGroup | None = None

    # ── Layout properties ──

    @property
    def width(self) -> int:
        """Display width based on terminal size (fills terminal)."""
        term_width = self.console.width or 100
        return max(MIN_WIDTH, term_width)

    @property
    def bar_width(self) -> int:
        """Progress bar width to fill available space."""
        return compute_bar_width(self.width)

    @property
    def gauge_width(self) -> int:
        """Resource gauge width (shorter than bar to fit metrics)."""
        return compute_gauge_width(self.width)

    @property
    def elapsed(self) -> float:
        """Elapsed time since display started."""
        return time.time() - self.start_time

    # ── Worker helpers ──

    def _count_group_workers(self, group: str) -> tuple[int, str]:
        """Count workers in a group and build annotation string.

        Returns:
            (count, annotation) where annotation describes backoff/failed state.
        """
        wg = self.worker_group
        if not wg:
            return 0, ""

        count = 0
        backoff = 0
        failed = 0
        for _name, status in wg.workers.items():
            grp = status.group or _name.split("_worker")[0]
            if grp == group:
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
        return count, f"({', '.join(parts)})" if parts else ""

    def _worker_complete(self, group: str) -> bool:
        """Check if all workers in a group have stopped."""
        wg = self.worker_group
        if not wg:
            return False
        workers = [
            s
            for _n, s in wg.workers.items()
            if (s.group or _n.split("_worker")[0]) == group
        ]
        return len(workers) > 0 and all(s.state == WorkerState.stopped for s in workers)

    def update_worker_status(self, worker_group: SupervisedWorkerGroup) -> None:
        """Update worker status from supervised worker group."""
        self.worker_group = worker_group
        self._refresh()

    # ── Header ──

    def _header_mode_label(self) -> str | None:
        """Override to provide mode label (e.g. 'SCAN ONLY')."""
        return None

    def _build_header(self) -> Text:
        """Build centered header with facility and focus."""
        header = Text()

        title = f"{self.facility.upper()} {self._title_suffix}"
        mode = self._header_mode_label()
        if mode:
            title += f" ({mode})"
        header.append(title.center(self.width - 4), style="bold cyan")

        if self.focus:
            header.append("\n")
            focus_line = f"Focus: {self.focus}"
            header.append(focus_line.center(self.width - 4), style="italic dim")

        return header

    # ── Servers ──

    def _build_servers_section(self) -> Text | None:
        """Build SERVERS status row from service monitor.

        Always returns a section (even before checks complete) so the
        SERVERS row is visible from the first render.
        """
        monitor = self.service_monitor
        if monitor is None:
            return None
        statuses = monitor.get_status()
        if not statuses:
            section = Text()
            section.append("  SERVERS", style="bold white")
            section.append("  checking...", style="dim italic")
            return section
        return build_servers_section(statuses)

    # ── Abstract pipeline/resources ──

    @abstractmethod
    def _build_pipeline_section(self) -> Text:
        """Build domain-specific pipeline section."""
        ...

    @abstractmethod
    def _build_resources_section(self) -> Text:
        """Build domain-specific resource gauges."""
        ...

    # ── Display assembly ──

    def _build_display(self) -> Panel:
        """Build the complete display.

        Standard layout: HEADER → SERVERS → PIPELINE → RESOURCES
        """
        sections: list[Text] = [self._build_header()]

        # SERVERS section (optional)
        servers = self._build_servers_section()
        if servers is not None:
            sections.append(Text("─" * (self.width - 4), style="dim"))
            sections.append(servers)

        # Pipeline section
        sections.append(Text("─" * (self.width - 4), style="dim"))
        sections.append(self._build_pipeline_section())

        # Resource gauges
        sections.append(Text("─" * (self.width - 4), style="dim"))
        sections.append(self._build_resources_section())

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

    # ── Lifecycle ──

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
    ETA/ETC markers are left-aligned at a fixed position after the gauge
    so they don't shift when values change width.

    Args:
        config: Resource configuration.
        gauge_width: Width of resource gauge bars (use ``compute_gauge_width()``).
    """
    section = Text()

    # Fixed column for metrics after gauge: "  $12.34  ETA 5m 30s"
    # Value column starts right after gauge, ETA/ETC at fixed offset
    value_col_width = 10  # "  $12.34" or "  1h 23m"

    # TIME row — elapsed ticker, no limit gauge
    section.append(f"{'  TIME':<{LABEL_WIDTH}}", style="bold cyan")

    eta = None if config.scan_only else config.eta
    if eta is not None and eta > 0:
        # Show work-based progress (elapsed / estimated-total)
        total_est = config.elapsed + eta
        section.append_text(make_resource_gauge(config.elapsed, total_est, gauge_width))
    else:
        section.append("━" * gauge_width, style="cyan")

    elapsed_s = f"  {format_time(config.elapsed)}"
    section.append(elapsed_s, style="bold")
    # Left-align ETA at fixed position
    pad = max(1, value_col_width - len(elapsed_s))
    section.append(" " * pad)

    if eta is not None:
        if eta <= 0:
            if config.limit_reason:
                section.append(
                    f"{config.limit_reason} limit reached", style="yellow dim"
                )
            else:
                section.append("complete", style="green dim")
        else:
            section.append(f"ETA {format_time(eta)}", style="dim")
    section.append("\n")

    # COST row — accumulated cost from graph (source of truth)
    if not config.scan_only and config.run_cost is not None:
        section.append(f"{'  COST':<{LABEL_WIDTH}}", style="bold yellow")

        # accumulated_cost from graph already includes current-session costs
        total_cost = config.accumulated_cost
        etc = config.etc
        if etc is not None and etc > total_cost:
            section.append_text(make_resource_gauge(total_cost, etc, gauge_width))
        else:
            section.append("━" * gauge_width, style="yellow")

        cost_s = f"  ${total_cost:.2f}"
        section.append(cost_s, style="bold")
        # Left-align ETC at fixed position
        pad = max(1, value_col_width - len(cost_s))
        section.append(" " * pad)

        if etc is not None and etc > total_cost:
            section.append(f"ETC ${etc:.2f}", style="dim")
        elif config.run_cost > 0:
            section.append(f"session ${config.run_cost:.2f}", style="dim")
        section.append("\n")

    # STATS row
    if config.stats:
        section.append(f"{'  STATS':<{LABEL_WIDTH}}", style="bold magenta")
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
