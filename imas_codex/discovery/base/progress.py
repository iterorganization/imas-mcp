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
    used: float, limit: float, width: int = 20, unit: str = ""
) -> Text:
    """Create a resource consumption gauge with color coding."""
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
    gauge.append("│", style="dim")
    gauge.append("━" * filled, style=color)
    gauge.append("─" * (width - filled), style="dim")
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
    """

    items: deque = field(default_factory=deque)
    last_pop: float = field(default_factory=time.time)
    rate: float = 2.0  # items per second (capped max rate)
    max_rate: float = 2.5  # never exceed this rate even if worker is faster
    min_display_time: float = 0.4  # minimum seconds each item stays visible
    max_size: int = 500  # larger buffer to absorb batch bursts

    def add(self, items: list, rate: float | None = None) -> None:
        """Add items to queue.

        The rate is capped at max_rate to ensure smooth display.
        """
        self.items.extend(items)
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
    """

    processed: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    last_batch_time: float = 0.0
    cost: float = 0.0  # LLM cost (for scorer workers)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def rate(self) -> float | None:
        if self.processed == 0 or self.elapsed <= 0:
            return None
        return self.processed / self.elapsed


# =============================================================================
# Progress Configuration
# =============================================================================


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
# Worker Row Configuration
# =============================================================================


@dataclass
class WorkerRowConfig:
    """Configuration for a single worker progress row."""

    name: str  # e.g., "SCAN", "SCORE"
    style: str  # e.g., "bold blue", "bold green"
    completed: int = 0
    total: int = 1
    rate: float | None = None
    disabled: bool = False
    disabled_msg: str = "disabled"


def build_worker_row(config: WorkerRowConfig, bar_width: int = 40) -> Text:
    """Build a single worker progress row.

    Format: "  SCAN   ━━━━━━━━━━────────────────  1,234  42%  77.1/s"
    """
    row = Text()

    if config.disabled:
        row.append(f"  {config.name:<6} ", style="dim")
        row.append("─" * bar_width, style="dim")
        row.append(f"    {config.disabled_msg}", style="dim italic")
        return row

    # Worker name
    row.append(f"  {config.name:<6} ", style=config.style)

    # Progress bar
    total = max(config.total, 1)
    ratio = min(config.completed / total, 1.0)
    pct = ratio * 100
    row.append(make_bar(ratio, bar_width), style=config.style.split()[-1])

    # Stats
    row.append(f" {config.completed:>6,}", style="bold")
    row.append(f" {pct:>3.0f}%", style="cyan")

    # Rate (if available)
    if config.rate and config.rate > 0:
        row.append(f" {config.rate:>5.1f}/s", style="dim")

    return row


# =============================================================================
# Common Display Sections
# =============================================================================


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


def build_resource_section(
    elapsed: float,
    eta: float | None,
    run_cost: float | None,
    cost_limit: float | None,
    total_cost: float | None = None,
    extra_stats: list[tuple[str, str, str]] | None = None,
    gauge_width: int = 20,
    scan_only: bool = False,
) -> Text:
    """Build resource consumption section (TIME, COST, STATS).

    Args:
        elapsed: Elapsed time in seconds
        eta: Estimated time remaining in seconds (or None)
        run_cost: Cost for this run (or None)
        cost_limit: Cost limit (or None)
        total_cost: Total accumulated cost (or None)
        extra_stats: List of (label, value, style) tuples for STATS row
        gauge_width: Width of resource gauges
        scan_only: If True, hide cost gauges
    """
    section = Text()

    # TIME row
    section.append("  TIME  ", style="bold cyan")

    if eta is not None and eta > 0:
        total_est = elapsed + eta
        section.append_text(make_resource_gauge(elapsed, total_est, gauge_width))
    else:
        section.append("│", style="dim")
        section.append("━" * gauge_width, style="cyan")
        section.append("│", style="dim")

    section.append(f"  {format_time(elapsed)}", style="bold")

    if eta is not None:
        if eta <= 0:
            section.append("  complete", style="green dim")
        else:
            section.append(f"  ETA {format_time(eta)}", style="dim")
    section.append("\n")

    # COST row (hidden in scan_only mode)
    if not scan_only and run_cost is not None and cost_limit is not None:
        section.append("  COST  ", style="bold yellow")
        section.append_text(make_resource_gauge(run_cost, cost_limit, gauge_width))
        section.append(f"  ${run_cost:.2f}", style="bold")
        section.append(f" / ${cost_limit:.2f}", style="dim")

        # Show total accumulated cost if different from run_cost
        if total_cost is not None and total_cost > run_cost:
            section.append(f"  (total: ${total_cost:.2f})", style="dim")
        section.append("\n")

    # STATS row (if extra_stats provided)
    if extra_stats:
        section.append("  STATS ", style="bold magenta")
        for i, (label, value, style) in enumerate(extra_stats):
            if i > 0:
                section.append("  ", style="dim")
            section.append(f"{label}={value}", style=style)

    return section


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
# Common Activity Item Base
# =============================================================================


@dataclass
class ActivityItem:
    """Base class for current activity display items."""

    path: str = ""
    is_processing: bool = False  # True when awaiting batch result
    error: str | None = None
