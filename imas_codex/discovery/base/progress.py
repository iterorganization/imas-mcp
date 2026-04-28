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
from collections.abc import Callable
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
    """Format large counts compactly: 1.1M, 61.5K, 403.

    Drops trailing ``.0`` for cleaner display (``10K`` not ``10.0K``).
    """
    if count >= 1_000_000:
        val = count / 1_000_000
        return f"{val:.0f}M" if val == int(val) else f"{val:.1f}M"
    if count >= 10_000:
        val = count / 1_000
        return f"{val:.0f}K" if val == int(val) else f"{val:.1f}K"
    if count >= 1_000:
        val = count / 1_000
        return f"{val:.1f}K"
    return str(count)


def format_rate(rate: float, unit: str = "s") -> str:
    """Format items/second with adaptive SI prefix: 1.2Ms/s, 84Ks/s, 3.5s/s

    Args:
        rate: Items per second.
        unit: Base unit label (default ``"s"`` for signals). The output
              reads as ``<value><prefix><unit>/s``, e.g. ``84Ks/s``.
    """
    if rate >= 1_000_000:
        return f"{rate / 1_000_000:.0f}M{unit}/s"
    if rate >= 10_000:
        return f"{rate / 1_000:.0f}K{unit}/s"
    if rate >= 1_000:
        return f"{rate / 1_000:.1f}K{unit}/s"
    if rate >= 10:
        return f"{rate:.0f}{unit}/s"
    if rate >= 1:
        return f"{rate:.1f}{unit}/s"
    return f"{rate:.2f}{unit}/s"


# =============================================================================
# ETA / ETC Calculation Utilities
# =============================================================================


def compute_parallel_eta(
    work_items: list[tuple[int, float | None]],
) -> float | None:
    """Compute ETA for parallel workers: max of (pending / rate).

    Since workers run concurrently, the overall completion time is
    bounded by the slowest worker group.

    Args:
        work_items: List of (pending_count, rate_per_second) tuples.
            Each tuple represents one worker group.  The rate should
            be the **aggregate** throughput for that group (i.e. the
            combined rate of all workers in the group, not a single
            worker's rate).  Use ``WorkerStats.active_rate`` or
            session-average ``run_count / elapsed`` to get aggregate
            rates when multiple workers share a ``WorkerStats``.

    Returns:
        Maximum ETA across all workers, or None if no rate data.
    """
    etas: list[float] = []
    for pending, rate in work_items:
        if pending > 0 and rate and rate > 0:
            etas.append(pending / rate)
    return max(etas) if etas else None


def compute_projected_etc(
    accumulated_cost: float,
    cost_items: list[tuple[int, float | None]],
) -> float | None:
    """Compute projected total cost: accumulated + sum of per-worker projections.

    Cost is additive across workers — each worker that incurs cost
    contributes independently to the total projected spend.

    Args:
        accumulated_cost: Current accumulated cost (source of truth from graph).
        cost_items: List of (pending_count, cost_per_item) tuples.
            Each tuple represents one cost-incurring worker group.

    Returns:
        Projected total cost, or None if no projected costs remain.
    """
    projected = 0.0
    for pending, cost_per_item in cost_items:
        if pending > 0 and cost_per_item and cost_per_item > 0:
            projected += pending * cost_per_item
    if projected > 0:
        return accumulated_cost + projected
    return None


# =============================================================================
# Accumulated Time Tracking
# =============================================================================

# Map discovery domain names to Facility node property names
_DOMAIN_TIME_PROPERTY: dict[str, str] = {
    "paths": "paths_elapsed_seconds",
    "code": "code_elapsed_seconds",
    "signals": "signals_elapsed_seconds",
    "wiki": "wiki_elapsed_seconds",
}


def get_accumulated_time(facility: str, domain: str) -> float:
    """Query accumulated wall-clock discovery time from the graph.

    Args:
        facility: Facility ID (e.g. "tcv", "jet").
        domain: Discovery domain ("paths", "code", "signals", "wiki").

    Returns:
        Accumulated seconds from prior sessions, or 0.0 if not tracked.
    """
    prop = _DOMAIN_TIME_PROPERTY.get(domain)
    if not prop:
        return 0.0
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            result = gc.query(
                f"MATCH (f:Facility {{id: $facility}}) "  # noqa: S608
                f"RETURN f.{prop} AS accumulated_time",
                facility=facility,
            )
        return float(result[0]["accumulated_time"] or 0.0) if result else 0.0
    except Exception:
        return 0.0


def record_session_time(facility: str, domain: str, elapsed: float) -> None:
    """Atomically increment accumulated discovery time on the Facility node.

    Called at session end. Safe for concurrent sessions — each adds
    its own wall-clock time independently.

    Args:
        facility: Facility ID.
        domain: Discovery domain ("paths", "code", "signals", "wiki").
        elapsed: Wall-clock seconds for this session.
    """
    prop = _DOMAIN_TIME_PROPERTY.get(domain)
    if not prop or elapsed <= 0:
        return
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            gc.query(
                f"MATCH (f:Facility {{id: $facility}}) "  # noqa: S608
                f"SET f.{prop} = coalesce(f.{prop}, 0) + $elapsed",
                facility=facility,
                elapsed=elapsed,
            )
    except Exception:
        pass  # Best effort — don't fail the session for a time recording issue


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
    1. Rate-limiting pops to a maximum rate (default 5.0/s)
    2. Enforcing minimum display time per item (default 0.2s)
    3. Adaptive rate calculation: when ``last_batch_time`` is passed to
       ``add()``, the pop rate is automatically tuned so the queue
       drains just before the next batch arrives.
    4. Larger queue buffer to absorb batch processing bursts
    5. Maximum display time per item (default: unlimited) — caps how
       long any single item stays visible.  Useful for slow workers
       (e.g. embed) where the adaptive rate would otherwise hold each
       item for 10+ seconds.

    Stale detection: tracks when items were last added.  Uses adaptive
    timeout based on the last batch processing time — items are cleared
    after 1.2× the batch time, ensuring the display transitions to
    "done" promptly when the worker finishes its last batch.  Falls
    back to ``stale_timeout`` when no batch time is available.
    """

    items: deque = field(default_factory=deque)
    last_pop: float = field(default_factory=time.time)
    last_add: float = 0.0  # timestamp of last add() call
    rate: float = 2.0  # items per second (current pop rate)
    max_rate: float = 5.0  # never exceed this rate even if worker is faster
    min_display_time: float = 0.2  # minimum seconds each item stays visible
    max_display_time: float = 0.0  # max seconds per item (0 = unlimited)
    max_size: int = 500  # larger buffer to absorb batch bursts
    stale_timeout: float = 8.0  # seconds without adds before queue is stale
    _last_batch_time: float = 0.0  # batch processing time for adaptive stale

    def add(
        self,
        items: list,
        rate: float | None = None,
        *,
        last_batch_time: float = 0.0,
    ) -> None:
        """Add items to queue with adaptive rate calculation.

        Clears any previously queued items before adding the new batch.
        This prevents accumulation of stale items from prior batches
        that could keep the display in "streaming" mode long after
        the worker has moved on.

        When ``last_batch_time`` is provided, the pop rate is computed so
        that items drain *slower* than the worker produces them. This
        ensures items remain in the queue when the next batch arrives,
        bridging inter-batch gaps and preventing "processing..." flicker.

        When a batch is larger than the queue can drain at the fastest
        stream rate before the next batch arrives, items are strided
        with ``[::n]`` to ensure users see a representative sample
        across the full batch rather than only the first entries.

        Args:
            items: Items to enqueue.
            rate: Fallback pop rate (items/s). Used when adaptive
                calculation is not possible.
            last_batch_time: Elapsed time of the batch that produced
                these items. Used to estimate when the next batch will
                arrive, so the queue drains smoothly.  Also used for
                adaptive stale detection (1.2× batch time).
        """
        # Stride items if the batch is too large to display at max_rate
        # before the next batch arrives.  This ensures users see items
        # from across the batch, not just the first few.
        if last_batch_time > 0 and len(items) > 0:
            expected_gap = last_batch_time + 1.0
            max_displayable = int(self.max_rate * expected_gap)
            if max_displayable > 0 and len(items) > max_displayable:
                stride = max(1, len(items) // max_displayable)
                items = items[::stride]

        # Reset queue on each new batch — discard un-displayed items
        # from the previous batch to avoid accumulation.
        self.items.clear()
        self.items.extend(items)
        self.last_add = time.time()
        if last_batch_time > 0:
            self._last_batch_time = last_batch_time

        queue_depth = len(self.items)
        new_rate: float | None = None

        if last_batch_time > 0 and queue_depth > 0:
            # Estimate gap until next batch: batch processing time + overhead
            expected_gap = last_batch_time + 1.0
            # Drain over 120% of the expected gap so items remain in the
            # queue when the next batch arrives, bridging the inter-batch
            # gap and preventing "processing..." flicker.
            drain_target = expected_gap * 1.2
            if drain_target > 0:
                new_rate = queue_depth / drain_target

        if new_rate is None and rate and rate > 0:
            new_rate = rate

        if new_rate is not None and new_rate > 0:
            self.rate = min(new_rate, self.max_rate)

        # Drop oldest items if queue exceeds max size
        while len(self.items) > self.max_size:
            self.items.popleft()

    def pop(self) -> Any | None:
        """Pop next item if rate limit allows.

        Enforces rate limiting, minimum display time, and optional
        maximum display time.  When ``max_display_time > 0`` the pop
        interval is clamped so no single item stays visible longer
        than the configured ceiling.
        """
        if not self.items:
            return None
        # Use the slower of: rate interval or min_display_time
        rate_interval = 1.0 / self.rate if self.rate > 0 else 0.5
        interval = max(rate_interval, self.min_display_time)
        # Cap at max_display_time when configured
        if self.max_display_time > 0:
            interval = min(interval, self.max_display_time)
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

        Uses adaptive timeout: 1.2× the last batch processing time
        ensures the display transitions to "done" promptly once the
        worker finishes its last batch.  Falls back to ``stale_timeout``
        when no batch time is tracked.
        """
        if not self.is_empty():
            return False
        if self.last_add == 0.0:
            return False  # never received items
        # Adaptive: use 1.2× batch time when available, otherwise fixed timeout
        timeout = self.stale_timeout
        if self._last_batch_time > 0:
            timeout = min(self._last_batch_time * 1.2, self.stale_timeout)
        return (time.time() - self.last_add) >= timeout

    def clear(self) -> None:
        """Clear the queue. Use on termination to prevent hanging."""
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)


# =============================================================================
# Scanner / Sub-task Progress
# =============================================================================


@dataclass
class ScannerProgress:
    """Progress tracking for a single scanner / sub-task within a worker.

    Used by seed workers to track per-scanner status (e.g. wiki, ppf,
    mdsplus) and by any worker that runs multiple sequential sub-tasks.
    Reusable across all discovery domains.
    """

    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    items_discovered: int = 0
    status: str = "pending"  # pending, running, done, failed
    detail: str = ""  # e.g. "subsystem DA 3/26"
    error: str | None = None

    @property
    def elapsed(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    def mark_running(self, detail: str = "") -> None:
        self.status = "running"
        self.start_time = time.time()
        if detail:
            self.detail = detail

    def mark_done(self, items: int = 0) -> None:
        self.status = "done"
        self.end_time = time.time()
        self.items_discovered += items

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.end_time = time.time()
        self.error = error

    def format_status(self) -> str:
        """Format as compact status string for display.

        Examples: ``wiki✓``, ``ppf✓ 5.2K``, ``jpf: subsystem DA 3/26``
        """
        if self.status == "done":
            count = (
                f" {format_count(self.items_discovered)}"
                if self.items_discovered
                else ""
            )
            return f"{self.name}✓{count}"
        if self.status == "failed":
            return f"{self.name}✗"
        if self.status == "running":
            if self.detail:
                return f"{self.name}: {self.detail}"
            return f"{self.name}…"
        return self.name


# =============================================================================
# Worker Statistics
# =============================================================================


@dataclass
class WorkerStats:
    """Statistics for a single async worker.

    Reusable across discovery domains (paths, wiki, etc.).

    Tracks both a lifetime average rate and an exponential moving average
    (EMA) over recent batches for smoother, more responsive rate display.
    Also tracks rolling error rates and per-scanner progress for
    diagnostic visibility.
    """

    processed: int = 0
    total: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    last_batch_time: float = 0.0
    cost: float = 0.0  # LLM cost (for scorer workers)

    # EMA rate tracking
    _ema_rate: float = 0.0  # Exponential moving average rate (items/s)
    _ema_alpha: float = 0.3  # Smoothing factor (0.3 = responsive to recent)
    _prev_processed: int = 0  # Items at last batch
    _prev_batch_time: float = 0.0  # Time at last batch

    # Frozen rate — captured when phase completes to prevent decay during idle
    _frozen_rate: float | None = None

    # Baseline: items already processed before this session started.
    # Set via ``set_baseline()`` when graph refresh first loads historical
    # counts.  ``session_processed`` = ``processed - _baseline_processed``
    # ensures rate calculations reflect only current-session throughput.
    _baseline_processed: int = 0

    # Idle time tracking (excluded from active_rate)
    _idle_total: float = 0.0  # Cumulative idle seconds
    _idle_start: float | None = None  # When current idle period began

    # Rolling error rate tracking (Phase 2)
    _error_timestamps: list[float] = field(default_factory=list)
    _consecutive_errors: int = 0
    _error_window: float = 60.0  # Rolling window in seconds

    # Per-scanner / sub-task progress (Phase 1)
    scanner_progress: dict[str, ScannerProgress] = field(default_factory=dict)

    # Metadata for structured logging
    metadata: dict[str, Any] = field(default_factory=dict)

    # Current activity description (for display)
    status_text: str = ""

    # Streaming display queue for per-item display in pipeline rows.
    # Items should be dicts with keys matching PipelineRowConfig fields
    # (e.g. ``primary_text``, ``description``, ``physics_domain``).
    stream_queue: StreamQueue = field(
        default_factory=lambda: StreamQueue(
            rate=0.5,
            max_rate=3.0,
            min_display_time=0.3,
            max_display_time=2.0,
        )
    )
    # Currently displayed stream item (popped from stream_queue)
    _current_stream_item: dict[str, Any] | None = None

    # SSH/connection timing
    _connection_start: float | None = None  # When current connection attempt began
    _connection_host: str | None = None  # Host being connected to

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def session_processed(self) -> int:
        """Items processed in the current session only.

        Subtracts ``_baseline_processed`` (items from prior runs) so that
        rate calculations reflect only current-session throughput.
        """
        return max(0, self.processed - self._baseline_processed)

    def set_baseline(self, count: int) -> None:
        """Record the number of already-processed items at session start.

        Call once when graph refresh first loads historical counts, so
        subsequent rate calculations use only current-session items.
        Has no effect if already set (baseline is captured only once).
        """
        if self._baseline_processed == 0 and count > 0:
            self._baseline_processed = count

    @property
    def rate(self) -> float | None:
        """Average rate (items/second) for the current session only.

        Uses ``session_processed`` (items done this session) divided by
        elapsed time, so restarting a process does not inflate the rate
        with items from prior runs.
        """
        if self.session_processed == 0 or self.elapsed <= 0:
            return None
        return self.session_processed / self.elapsed

    @property
    def active_rate(self) -> float | None:
        """Average rate excluding idle time (items/second).

        Uses ``session_processed`` for numerator. If no idle time was
        tracked, falls back to overall session rate.
        """
        if self.session_processed == 0:
            return None
        idle = self._idle_total
        if self._idle_start is not None:
            idle += time.time() - self._idle_start
        active = self.elapsed - idle
        if active <= 0:
            return self.rate
        return self.session_processed / active

    @property
    def ema_rate(self) -> float | None:
        """Exponential moving average rate over recent batches.

        Returns frozen rate if set (phase completed), then EMA,
        then falls back to active_rate.
        """
        if self._frozen_rate is not None:
            return self._frozen_rate
        if self._ema_rate > 0:
            return self._ema_rate
        return self.active_rate

    def freeze_rate(self) -> None:
        """Freeze the current rate so it stays constant after phase completion.

        Captures the best available rate (EMA > active > overall) at the
        moment the phase finishes.  Subsequent calls to ``ema_rate`` return
        this frozen value instead of recomputing from elapsed time.
        """
        self._frozen_rate = self._ema_rate or self.active_rate or self.rate

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

        When ``last_batch_time`` is set (by the worker after each batch),
        uses ``batch_size / last_batch_time`` as instantaneous rate.
        This correctly reflects actual processing throughput even when
        multiple workers share the same stats object and call
        ``record_batch`` nearly simultaneously.

        Falls back to inter-call timing when ``last_batch_time`` is not set.

        Args:
            batch_size: Items in this batch. If None, inferred from
                ``processed - _prev_processed``.
        """
        now = time.time()
        if batch_size is None:
            batch_size = self.processed - self._prev_processed
        if batch_size <= 0:
            return

        # Prefer last_batch_time (actual processing duration) over
        # inter-call timing which is distorted by concurrent workers
        if self.last_batch_time > 0:
            instant_rate = batch_size / self.last_batch_time
        else:
            dt = now - self._prev_batch_time if self._prev_batch_time > 0 else 0
            if dt > 0:
                instant_rate = batch_size / dt
            else:
                instant_rate = 0.0

        if instant_rate > 0:
            if self._ema_rate > 0:
                self._ema_rate = (
                    self._ema_alpha * instant_rate
                    + (1 - self._ema_alpha) * self._ema_rate
                )
            else:
                self._ema_rate = instant_rate

        self._prev_processed = self.processed
        self._prev_batch_time = now

    # ── Error rate tracking ──

    def record_error(self) -> None:
        """Record an error occurrence for rolling rate calculation."""
        now = time.time()
        self.errors += 1
        self._error_timestamps.append(now)
        self._consecutive_errors += 1
        # Prune old timestamps outside the window
        cutoff = now - self._error_window
        self._error_timestamps = [t for t in self._error_timestamps if t >= cutoff]

    def record_success(self) -> None:
        """Record a successful operation, resetting consecutive error count."""
        self._consecutive_errors = 0

    @property
    def error_rate_1m(self) -> float:
        """Errors per second over the last 60 seconds."""
        now = time.time()
        cutoff = now - self._error_window
        recent = [t for t in self._error_timestamps if t >= cutoff]
        if not recent:
            return 0.0
        return len(recent) / self._error_window

    @property
    def consecutive_errors(self) -> int:
        """Number of consecutive errors without a success."""
        return self._consecutive_errors

    @property
    def error_rate_pct(self) -> float:
        """Error rate as a percentage of total processed items.

        Returns 0.0 if no items processed. Used for color coding:
        green (<5%), yellow (5-25%), red (>25%).
        """
        total = self.processed + self.errors
        if total == 0:
            return 0.0
        return (self.errors / total) * 100

    @property
    def error_health_style(self) -> str:
        """Rich style string based on error rate: green/yellow/red."""
        pct = self.error_rate_pct
        if pct < 5:
            return "green"
        if pct < 25:
            return "yellow"
        return "red"

    # ── Scanner progress ──

    def start_scanner(self, name: str, detail: str = "") -> ScannerProgress:
        """Start tracking a scanner / sub-task.

        Args:
            name: Scanner type name (e.g. "wiki", "ppf", "mdsplus").
            detail: Optional detail string for display.

        Returns:
            The ScannerProgress tracker for further updates.
        """
        sp = ScannerProgress(name=name)
        sp.mark_running(detail)
        self.scanner_progress[name] = sp
        return sp

    def finish_scanner(self, name: str, items: int = 0) -> None:
        """Mark a scanner / sub-task as complete."""
        if name in self.scanner_progress:
            self.scanner_progress[name].mark_done(items)

    def fail_scanner(self, name: str, error: str) -> None:
        """Mark a scanner / sub-task as failed."""
        if name in self.scanner_progress:
            self.scanner_progress[name].mark_failed(error)

    def format_scanner_status(self) -> str:
        """Format all scanner statuses as a compact one-line summary.

        Example: ``wiki✓  ppf✓ 5,204  mdsplus✓  jpf: subsystem DA 3/26  device_xml✓``
        """
        if not self.scanner_progress:
            return ""
        return "  ".join(sp.format_status() for sp in self.scanner_progress.values())

    def format_scanner_timing(self) -> str:
        """Format scanner elapsed times for stats display.

        Example: ``wiki:2.1s  ppf:8.4s  mdsplus:0.3s  jpf:45.2s``
        """
        if not self.scanner_progress:
            return ""
        parts = []
        for sp in self.scanner_progress.values():
            elapsed = sp.elapsed
            if elapsed < 60:
                parts.append(f"{sp.name}:{elapsed:.1f}s")
            else:
                parts.append(f"{sp.name}:{format_time(elapsed)}")
        return "  ".join(parts)

    # ── Connection timing ──

    def mark_connecting(self, host: str | None = None) -> None:
        """Mark the start of a connection attempt (SSH, MDSplus, etc.)."""
        self._connection_start = time.time()
        self._connection_host = host

    def mark_connected(self) -> None:
        """Mark a connection attempt as complete."""
        self._connection_start = None
        self._connection_host = None

    @property
    def connection_elapsed(self) -> float | None:
        """Seconds since connection attempt started, or None if not connecting."""
        if self._connection_start is None:
            return None
        return time.time() - self._connection_start

    def format_connection_status(self) -> str:
        """Format connection status with timing annotation.

        Returns empty string if not connecting.
        Examples: ``connecting (3.2s)``, ``connecting (45s ⚡)``
        """
        elapsed = self.connection_elapsed
        if elapsed is None:
            return ""
        host_part = f" to {self._connection_host}" if self._connection_host else ""
        if elapsed > 30:
            return f"connecting{host_part} ({elapsed:.0f}s ⚡)"
        return f"connecting{host_part} ({elapsed:.1f}s)"


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
#   LABEL_WIDTH       – left label column ("  EXTRACTx2", "  TIME")
#   METRICS_WIDTH     – right stat column for progress bars (" {count} {%}")
#   GAUGE_METRICS_WIDTH – right stat column for resource gauges (wider: time/cost text)
#   MIN_WIDTH         – minimum panel width
#
# Bar and gauge widths are computed from terminal width:
#   bar_width   = term_width - 4 - LABEL_WIDTH - METRICS_WIDTH
#   gauge_width = term_width - 4 - LABEL_WIDTH - GAUGE_METRICS_WIDTH
#
# Pipeline line 1 has just count+pct on the right (METRICS_WIDTH).
# Rate and cost are right-aligned on line 2/3 to the same edge.
# Text content on lines 2/3 clips at the bar end (LABEL_WIDTH + bar_width)
# so "..." aligns with the progress bar's right edge.
# Resource gauges (TIME, COST) keep GAUGE_METRICS_WIDTH for trailing text.

LABEL_WIDTH = 12
METRICS_WIDTH = 14
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
    item_limit: int | None = None  # --path-limit flag (paths, pages, etc.)
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
        /home/user/codes/chease  electromagnetic  terminal    0.23/s
        0.85  Thomson Scattering analysis code               $12.54

    Line 1: NAMExN + bar + count + pct
    Line 2: name + domain + terminal … rate (right-aligned in metrics zone)
    Line 3: score + description … cost (right-aligned in metrics zone)

    Text on lines 2-3 clips at the progress bar's right edge so "..."
    aligns with the bar end.  Rate and cost right-align at ``row_width``
    (past the bar, in the metrics column).

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
    primary_text: str = ""  # Resource name (shown on line 2, left-aligned)
    primary_text_style: str = "white"  # Rich style for primary_text
    score_value: float | None = None  # Score (shown on line 3, before description)
    score_parts: list[tuple[str, str]] | None = (
        None  # Custom score rendering (e.g. "0.65"+"+0.20")
    )
    physics_domain: str = ""  # Physics domain (shown on line 2, after name)
    terminal_label: str = (
        ""  # Terminal flag (shown on line 2 as "terminal" in muted red)
    )
    description: str = ""  # LLM/VLM description (shown on line 3, after score)
    description_fallback: str = ""  # Shown on line 3 when description is empty

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


def build_pipeline_row(config: PipelineRowConfig, bar_width: int = 40) -> Text:
    """Build a unified pipeline row (progress bar + activity).

    Renders 3 lines:
      Line 1: NAMExN + bar + count + pct
      Line 2: name + domain + terminal…  (clips at bar end)  rate
      Line 3: score + description…        (clips at bar end)  $cost

    Args:
        config: Pipeline row configuration.
        bar_width: Width of the progress bar.
    """
    row = Text()
    # Total content width for lines 2/3 right-alignment of rate/cost
    row_width = LABEL_WIDTH + bar_width + METRICS_WIDTH

    if config.disabled:
        row.append(f"  {config.name}".ljust(LABEL_WIDTH), style="dim")
        row.append("─" * bar_width, style="dim")
        row.append(f"    {config.disabled_msg}", style="dim italic")
        return row

    # ── Line 1: label + progress bar + count + pct ──
    label = Text()
    label.append(f"  {config.name}", style=config.style)
    if config.worker_count > 1:
        label.append(f"x{config.worker_count}", style="dim")
    if config.worker_annotation:
        label.append(f" {config.worker_annotation}", style="dim")
    # Pad to LABEL_WIDTH for aligned bar start (ensure at least 1 space)
    label_len = len(label.plain)
    label_pad = max(1, LABEL_WIDTH - label_len)
    label.append(" " * label_pad)
    row.append_text(label)

    # Shorten bar when label overflows LABEL_WIDTH to keep line 1 on one line
    effective_bar = bar_width - max(0, label_len + 1 - LABEL_WIDTH)

    total = max(config.total, 1)
    ratio = min(config.completed / total, 1.0)
    pct = ratio * 100
    # When displayed percentage rounds to 100%, fill the bar completely
    # to avoid a visual gap between "100%" text and a 99.x% bar
    bar_ratio = 1.0 if round(pct) >= 100 else ratio
    row.append(make_bar(bar_ratio, effective_bar), style=config.style.split()[-1])

    # Right: count + pct only (format_count for compact display)
    count_s = f" {format_count(config.completed):>8}"
    pct_s = f" {pct:>3.0f}%" if config.show_pct else "     "
    pad = max(0, METRICS_WIDTH - len(count_s) - len(pct_s))
    if pad > 0:
        row.append(" " * pad)
    row.append(count_s, style="bold")
    row.append(pct_s, style="cyan" if config.show_pct else "dim")

    # ── Line 2: name + domain + terminal (left) + rate (right-aligned) ──
    row.append("\n")
    line2 = Text()

    # Pre-compute rate text for right-alignment at row_width
    rate_s = ""
    if config.rate and config.rate > 0:
        rate_s = format_rate(config.rate)

    if config.has_content:
        line2.append("  ", style="dim")
        # Order: name → domain → terminal
        # Pre-compute suffix widths to clip name appropriately
        # Reserve space for rate text + gap so the line never overflows
        suffix_width = 0
        if config.physics_domain:
            suffix_width += cell_len(config.physics_domain) + 2
        if config.terminal_label:
            suffix_width += cell_len(config.terminal_label) + 2
        if rate_s:
            suffix_width += len(rate_s) + 2
        max_name = max(10, row_width - 2 - suffix_width)
        line2.append(
            clip_text(config.primary_text, max_name),
            style=config.primary_text_style,
        )
        if config.physics_domain:
            line2.append("  ", style="dim")
            line2.append(config.physics_domain, style="green")
        if config.terminal_label:
            line2.append("  ", style="dim")
            line2.append(config.terminal_label, style="red dim")
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

    # ── Line 3: score + description (left) + cost (right-aligned) ──
    row.append("\n")
    line3 = Text()

    # Pre-compute cost text for right-alignment
    cost_s = ""
    if config.cost is not None and config.cost > 0:
        cost_s = f"${config.cost:.2f}"

    if config.has_content:
        # Score at start of line 3
        if config.score_parts:
            line3.append("  ", style="dim")
            for text, style in config.score_parts:
                line3.append(text, style=style)
            line3.append("  ", style="dim")
        elif config.score_value is not None:
            line3.append("  ", style="dim")
            score_style = (
                "bold green"
                if config.score_value >= 0.7
                else "yellow"
                if config.score_value >= 0.4
                else "red"
            )
            line3.append(f"{config.score_value:.2f}", style=score_style)
            line3.append("  ", style="dim")

        # Description clipped to remaining space, reserving room for cost
        _desc = config.description or config.description_fallback
        if _desc:
            if not line3.plain.strip():
                line3.append("  ", style="dim")
            _style = "italic dim" if config.description else "cyan dim italic"
            cost_reserve = (len(cost_s) + 2) if cost_s else 0
            max_desc = max(10, row_width - cell_len(line3.plain) - cost_reserve)
            line3.append(clip_text(clean_text(_desc), max_desc), style=_style)

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
                # Show max backoff remaining time across workers in this group
                max_remaining = 0.0
                for name, _st in workers:
                    ws = worker_group.workers.get(name)
                    if ws and ws.backoff_remaining > max_remaining:
                        max_remaining = ws.backoff_remaining
                if max_remaining > 0:
                    parts.append(f"{backing_off} backoff ({int(max_remaining)}s)")
                else:
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
            label = s.detail or "ok"
            # Flag credit/budget issues even when some models are healthy
            if "no credit" in label:
                style = "yellow"
            elif "↑" in label:
                # Load indicator present — show load portion in yellow
                style = "green"
            else:
                style = "green"
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
        # Split load indicator (↑) into separate yellow segment
        if style == "green" and "↑" in label:
            base, load_part = label.split("↑", 1)
            section.append(base, style="green")
            section.append(f"↑{load_part}", style="yellow")
        else:
            section.append(label, style=style)

        # SSH health summary (Phase 2.3) — show avg latency and failure ratio
        # for services with enough check history
        health_summary = (
            s.format_health_summary() if hasattr(s, "format_health_summary") else ""
        )
        if health_summary and s.total_checks >= 3:
            section.append(health_summary, style="dim")

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
        # Shutdown state
        self._shutting_down = False
        self._shutdown_start: float | None = None

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

    def _worker_running(self, group: str) -> bool:
        """Check if any worker in a group is running or idle (waiting)."""
        wg = self.worker_group
        if not wg:
            return False
        return any(
            s.state in (WorkerState.running, WorkerState.idle)
            for _n, s in wg.workers.items()
            if (s.group or _n.split("_worker")[0]) == group
        )

    def _worker_waiting(self, group: str) -> bool:
        """Check if all workers in a group are idle (waiting on dependencies).

        Returns True when workers exist and ALL are in idle state — meaning
        they haven't started processing yet (blocked on ``depends_on``).
        Returns False if any worker is running, stopped, or crashed.
        """
        wg = self.worker_group
        if not wg:
            return False
        workers = [
            s
            for _n, s in wg.workers.items()
            if (s.group or _n.split("_worker")[0]) == group
        ]
        return len(workers) > 0 and all(s.state == WorkerState.idle for s in workers)

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

        # Skip facility prefix for non-facility pseudo-domains (sn, dd) where
        # the facility token is a placeholder rather than a real installation.
        if self.facility.lower() in ("sn", "dd", ""):
            title = self._title_suffix
        else:
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

    # ── Shutdown ──

    def begin_shutdown(self) -> None:
        """Switch display to shutdown mode.

        Called by the signal handler on first Ctrl+C.  Changes the
        panel border to yellow and appends a shutdown progress section
        that tracks worker drain status in real time.  The next
        ``_refresh()`` / ``tick()`` will pick up the new state.
        """
        self._shutting_down = True
        self._shutdown_start = time.time()
        self._refresh()

    def _build_shutdown_section(self) -> Text:
        """Build shutdown progress section showing per-group drain status.

        Each worker group gets a status indicator:
          SHUTDOWN  scan done  score draining (2)  enrich done  3.2s
        After 15s, a hint is shown to press Ctrl+C again for forced exit.
        """
        section = Text()
        section.append("  SHUTDOWN", style="bold yellow")

        wg = self.worker_group
        if wg is None:
            section.append("  stopping workers...", style="yellow")
            return section

        # Collect unique groups preserving discovery order
        seen: dict[str, None] = {}
        for _name, status in wg.workers.items():
            grp = status.group or _name.split("_worker")[0]
            seen.setdefault(grp, None)
        groups = list(seen)

        any_draining = False
        for grp in groups:
            workers = [
                s
                for _n, s in wg.workers.items()
                if (s.group or _n.split("_worker")[0]) == grp
            ]
            active = sum(1 for w in workers if w.is_active)
            total = len(workers)

            if active == 0:
                # All workers in this group have stopped
                section.append(f"  {grp} ", style="dim")
                section.append("done", style="green")
            else:
                any_draining = True
                section.append(f"  {grp} ", style="white")
                section.append(f"draining ({active}/{total})", style="yellow")

        # Elapsed shutdown time
        if self._shutdown_start is not None:
            shutdown_elapsed = time.time() - self._shutdown_start
            section.append(f"  {shutdown_elapsed:.1f}s", style="dim")

            # After 15s of draining, show hint about forced shutdown
            if any_draining and shutdown_elapsed > 15:
                section.append(
                    "\n  Press Ctrl+C again to force exit", style="dim italic"
                )

        return section

    # ── Display assembly ──

    def _build_display(self) -> Panel:
        """Build the complete display.

        Standard layout: HEADER → SERVERS → PIPELINE → RESOURCES
        During shutdown: same layout + SHUTDOWN section appended,
        yellow border to indicate graceful drain in progress.
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

        # Shutdown drain status (appended below when shutting down)
        if self._shutting_down:
            sections.append(Text("─" * (self.width - 4), style="dim"))
            sections.append(self._build_shutdown_section())

        content = Text()
        for i, section in enumerate(sections):
            if i > 0:
                content.append("\n")
            content.append_text(section)

        return Panel(
            content,
            border_style="yellow" if self._shutting_down else "cyan",
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
    accumulated_time: float = 0.0  # Historic time from prior sessions

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

    # SCANNERS row: formatted scanner timing string
    scanner_timing: str = ""


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

    # TIME row — shows accumulated time (historic + session elapsed)
    section.append(f"{'  TIME':<{LABEL_WIDTH}}", style="bold cyan")

    # Total time = prior sessions + current session
    total_time = config.accumulated_time + config.elapsed

    eta = None if config.scan_only else config.eta
    if eta is not None and eta > 0:
        # Show work-based progress (total_time / estimated-total)
        total_est = total_time + eta
        section.append_text(make_resource_gauge(total_time, total_est, gauge_width))
    else:
        section.append("━" * gauge_width, style="cyan")

    elapsed_s = f"  {format_time(total_time)}"
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
            section.append(f"${config.run_cost:.2f}", style="dim")
        section.append("\n")

    # STATS row
    if config.stats:
        # Content width = gauge_width + GAUGE_METRICS_WIDTH (to match gauge rows above)
        content_width = LABEL_WIDTH + gauge_width + GAUGE_METRICS_WIDTH
        section.append(f"{'  STATS':<{LABEL_WIDTH}}", style="bold magenta")
        stats_len = LABEL_WIDTH
        for i, (label, value, style) in enumerate(config.stats):
            entry = f"{label}={value}"
            sep = "  " if i > 0 else ""
            if stats_len + len(sep) + len(entry) > content_width:
                break
            if sep:
                section.append(sep, style="dim")
            section.append(entry, style=style)
            stats_len += len(sep) + len(entry)

        # Pending work — on a new line, clipped to content width
        if config.pending:
            active = [(label, count) for label, count in config.pending if count > 0]
            if active:
                parts = [f"{label}:{format_count(count)}" for label, count in active]
                pending_text = f"pending=[{' '.join(parts)}]"
                max_pending = content_width - LABEL_WIDTH
                if len(pending_text) > max_pending:
                    pending_text = pending_text[: max_pending - 1] + "]"
                section.append("\n")
                section.append(" " * LABEL_WIDTH)
                section.append(pending_text, style="cyan dim")

    # SCANNERS row — per-scanner timing (Phase 1.3)
    if config.scanner_timing:
        section.append("\n")
        section.append(f"{'  SCANNERS':<{LABEL_WIDTH}}", style="bold blue")
        section.append(config.scanner_timing, style="dim")

    return section


# =============================================================================
# Data-Driven Progress Display
# =============================================================================


@dataclass
class StageDisplaySpec:
    """Declarative pipeline stage for :class:`DataDrivenProgressDisplay`.

    Each spec maps a discovery engine worker group to a pipeline row in the
    progress display.  The display observes the engine state directly via
    ``stats_attr`` and ``phase_attr`` field names — no manual callbacks or
    separate ProgressState required.
    """

    name: str
    """Display label (e.g. ``"SCAN"``, ``"VLM"``)."""

    style: str
    """Rich style for the label (e.g. ``"bold blue"``)."""

    group: str
    """Worker group name — used for worker count and completion detection."""

    stats_attr: str
    """Name of the :class:`WorkerStats` field on the engine state."""

    phase_attr: str = ""
    """Name of the :class:`PipelinePhase` field on the engine state (optional)."""

    disabled: bool = False
    """Whether this stage is disabled (shown as ``"disabled"``)."""

    disabled_msg: str = "disabled"
    """Message shown when the stage is disabled."""


class DataDrivenProgressDisplay(BaseProgressDisplay):
    """Progress display driven by :class:`StageDisplaySpec` declarations.

    Observes the engine state directly for :class:`WorkerStats` and
    :class:`PipelinePhase`, eliminating the need for a separate
    ``ProgressState`` dataclass and manual callback wiring.

    Usage::

        display = DataDrivenProgressDisplay(
            facility="tcv",
            cost_limit=2.0,
            stages=[
                StageDisplaySpec("FETCH", "bold blue", "image", "image_stats", "image_phase"),
                StageDisplaySpec("VLM", "bold magenta", "vlm", "image_score_stats",
                                 disabled=scan_only),
            ],
            title_suffix="Document Discovery",
        )
        display.set_engine_state(state)
    """

    def __init__(
        self,
        facility: str,
        cost_limit: float,
        stages: list[StageDisplaySpec],
        *,
        console: Any | None = None,
        focus: str = "",
        title_suffix: str = "Discovery",
        mode_label: str | None = None,
        graph_refresh_fn: Callable[[str], None] | None = None,
        stats_fn: Callable[[], list[tuple[str, str, str]]] | None = None,
        pending_fn: Callable[[], list[tuple[str, int]]] | None = None,
        accumulated_cost_fn: Callable[[], float] | None = None,
        accumulated_time_fn: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(
            facility=facility,
            cost_limit=cost_limit,
            console=console,
            focus=focus,
            title_suffix=title_suffix,
        )
        self.stages = stages
        self._mode_label = mode_label
        self._engine_state: Any | None = None
        self._graph_refresh_fn = graph_refresh_fn
        self._stats_fn = stats_fn
        self._pending_fn = pending_fn
        self._accumulated_cost_fn = accumulated_cost_fn
        self._accumulated_time_fn = accumulated_time_fn

    def set_engine_state(self, state: Any) -> None:
        """Connect display to the live engine state."""
        self._engine_state = state

    def _header_mode_label(self) -> str | None:
        return self._mode_label

    def tick(self) -> None:
        """Drain streaming queues from worker stats."""
        if not self._engine_state:
            return
        for stage in self.stages:
            stats: WorkerStats | None = getattr(
                self._engine_state, stage.stats_attr, None
            )
            if stats is None:
                continue
            item = stats.stream_queue.pop()
            if item is not None:
                stats._current_stream_item = item
            elif stats.stream_queue.is_stale():
                stats._current_stream_item = None

    def _build_pipeline_section(self) -> Text:
        rows: list[PipelineRowConfig] = []
        for stage in self.stages:
            stats: WorkerStats | None = None
            if self._engine_state and stage.stats_attr:
                stats = getattr(self._engine_state, stage.stats_attr, None)

            count, ann = self._count_group_workers(stage.group)
            completed = stats.processed if stats else 0
            total = stats.total if stats and stats.total > 0 else max(completed, 1)
            complete = self._worker_complete(stage.group)
            running = self._worker_running(stage.group)
            waiting = self._worker_waiting(stage.group)

            # Stream item display — use current popped item from queue
            primary_text = stats.status_text if stats else ""
            primary_text_style = "white"
            description = ""
            physics_domain = ""
            score_value: float | None = None
            if stats and stats._current_stream_item:
                si = stats._current_stream_item
                primary_text = si.get("primary_text", primary_text)
                primary_text_style = si.get("primary_text_style", "white")
                description = si.get("description", "")
                physics_domain = si.get("physics_domain", "")
                _sv = si.get("score_value")
                if isinstance(_sv, int | float):
                    score_value = float(_sv)

            rows.append(
                PipelineRowConfig(
                    name=stage.name,
                    style=stage.style,
                    completed=completed,
                    total=total,
                    rate=stats.ema_rate if stats else None,
                    cost=stats.cost if stats and stats.cost > 0 else None,
                    disabled=stage.disabled,
                    disabled_msg=stage.disabled_msg,
                    worker_count=count,
                    worker_annotation=ann,
                    primary_text=primary_text,
                    primary_text_style=primary_text_style,
                    description=description,
                    physics_domain=physics_domain,
                    score_value=score_value,
                    is_complete=complete,
                    is_processing=running and not complete,
                    processing_label="waiting..." if waiting else "processing...",
                )
            )
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        total_cost = 0.0
        cost_items: list[tuple[int, float | None]] = []
        work_items: list[tuple[int, float | None]] = []

        if self._engine_state:
            for stage in self.stages:
                stats = getattr(self._engine_state, stage.stats_attr, None)
                if stats is None:
                    continue

                # Collect per-stage cost projections (ETC = sum)
                if stats.cost > 0:
                    total_cost += stats.cost
                    if stats.processed > 0 and stats.total > stats.processed:
                        cost_per_item = stats.cost / stats.processed
                        remaining = stats.total - stats.processed
                        cost_items.append((remaining, cost_per_item))

                # Collect per-stage time estimates (ETA = max)
                if (
                    stats.total > 0
                    and stats.processed > 0
                    and stats.processed < stats.total
                ):
                    # Use active_rate (aggregate across concurrent workers)
                    # for accurate multi-worker ETA
                    rate = stats.active_rate or stats.ema_rate or stats.rate
                    remaining = stats.total - stats.processed
                    work_items.append((remaining, rate))

        # ETA: max of remaining time across active stages
        eta = compute_parallel_eta(work_items)

        # Accumulated cost: graph-sourced (cross-run) + current session
        accumulated = total_cost
        if self._accumulated_cost_fn:
            graph_cost = self._accumulated_cost_fn()
            accumulated = graph_cost + total_cost

        # ETC: accumulated + sum of per-stage remaining cost projections
        projected_cost = compute_projected_etc(accumulated, cost_items)

        config = ResourceConfig(
            elapsed=self.elapsed,
            eta=eta,
            accumulated_time=(
                self._accumulated_time_fn() if self._accumulated_time_fn else 0.0
            ),
            run_cost=total_cost if total_cost > 0 else None,
            cost_limit=self.cost_limit if self.cost_limit > 0 else None,
            accumulated_cost=accumulated,
            etc=projected_cost,
            stats=self._stats_fn() if self._stats_fn else None,
            pending=self._pending_fn() if self._pending_fn else None,
        )
        return build_resource_section(config, self.gauge_width)

    def refresh_from_graph(self, facility: str) -> None:
        """Refresh display data from graph.

        Delegates to the ``graph_refresh_fn`` provided at construction, if any.
        """
        if self._graph_refresh_fn:
            self._graph_refresh_fn(facility)
        self._refresh()

    def print_summary(self) -> None:
        """Print a brief summary after discovery completes."""
        if self._engine_state is None:
            return
        lines: list[str] = []
        for stage in self.stages:
            stats = getattr(self._engine_state, stage.stats_attr, None)
            if stats and stats.processed > 0:
                parts = [f"{stage.name}: {stats.processed:,}"]
                if stats.cost > 0:
                    parts.append(f"${stats.cost:.2f}")
                lines.append("  ".join(parts))
        if lines:
            self.console.print()
            for line in lines:
                self.console.print(f"  {line}")
        stats = self._stats_fn() if self._stats_fn else None
        if stats:
            summary = "  ".join(f"{label}={value}" for label, value, _style in stats)
            if summary:
                self.console.print(f"  {summary}")
