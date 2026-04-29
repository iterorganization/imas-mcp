"""Per-pool streaming display for the 6-pool SN pipeline.

Six pools rendered individually:

    GENERATE_NAME → REVIEW_NAME → REFINE_NAME
    GENERATE_DOCS → REVIEW_DOCS → REFINE_DOCS

Each pool gets its own progress bar with per-item streaming lines,
accumulated cost, and throughput.  The footer shows overall TIME/ETA,
COST/ETC/CAP, and SERVERS latency.

All rendering functions are **pure** — they accept state dicts and
return :class:`rich.text.Text` renderables.  This makes them unit-
testable without a ``Live`` context.

Display layout (~30 lines):

    GENERATE_NAME  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━────────  720/2000  36%  1.5/s  $1.20
      electron_temperature_in_core_plasma  →  e_temp_core
    REVIEW_NAME    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━────────  640/2000  32%  2.1/s  $0.85
      e_temp_core  0.83  "Good grammar; documentation could be…"
    ...
    ─────────────────────────────────────────────────────────────
    TIME  ━━━━━━━━━━━━────────────────  18m 32s  ETA 2h 14m
    COST  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  $4.55  ETC $9.20  CAP $46.00
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rich.text import Text

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

#: Pool names in display order.
POOL_ORDER: tuple[str, ...] = (
    "generate_name",
    "review_name",
    "refine_name",
    "generate_docs",
    "review_docs",
    "refine_docs",
)

#: Display labels (upper-case, underscore-separated).
POOL_LABELS: dict[str, str] = {
    "generate_name": "GENERATE_NAME",
    "review_name": "REVIEW_NAME",
    "refine_name": "REFINE_NAME",
    "generate_docs": "GENERATE_DOCS",
    "review_docs": "REVIEW_DOCS",
    "refine_docs": "REFINE_DOCS",
}

#: Rich styles per pool.
POOL_STYLES: dict[str, str] = {
    "generate_name": "bold magenta",
    "review_name": "bold yellow",
    "refine_name": "magenta",
    "generate_docs": "bold cyan",
    "review_docs": "bold yellow",
    "refine_docs": "cyan",
}

#: Label column width (right-padded to align bars).
LABEL_COL = 16

#: Progress bar width.
BAR_WIDTH = 36

#: Maximum streamed items kept per pool.
STREAM_MAXLEN = 3

#: Review-score color thresholds.
SCORE_GREEN_THRESHOLD = 0.85
SCORE_YELLOW_THRESHOLD = 0.65


# ═══════════════════════════════════════════════════════════════════════
# Per-item line renderers (pure functions)
# ═══════════════════════════════════════════════════════════════════════


def score_color(score: float) -> str:
    """Return Rich style name for a reviewer score.

    ≥ 0.85 → green, 0.65–0.85 → yellow, < 0.65 → red.
    """
    if score >= SCORE_GREEN_THRESHOLD:
        return "green"
    if score >= SCORE_YELLOW_THRESHOLD:
        return "yellow"
    return "red"


def _clip(text: str, maxlen: int) -> str:
    """Clip text with ellipsis if exceeding *maxlen*."""
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 1] + "…"


def format_item_generate_name(item: dict[str, Any]) -> Text:
    """Render a GENERATE_NAME per-item line.

    Format: ``<source>  →  <name>``
    """
    source = item.get("source", item.get("dd_path", ""))
    name = item.get("name", "")
    line = Text("    ")
    line.append(_clip(str(source), 40), style="dim")
    line.append("  →  ", style="white")
    line.append(str(name), style="bold white")
    return line


def format_item_review_name(item: dict[str, Any]) -> Text:
    """Render a REVIEW_NAME per-item line.

    Format: ``<name>  <score>  "<comment clipped to 80 chars>"``
    """
    name = str(item.get("name", ""))
    raw_score = item.get("score", 0.0)
    sc = float(raw_score) if raw_score is not None else 0.0
    comment = str(item.get("comment", ""))

    line = Text("    ")
    line.append(_clip(name, 30), style="white")
    line.append("  ")
    line.append(f"{sc:.2f}", style=score_color(sc))
    if comment:
        line.append(f'  "{_clip(comment, 80)}"', style="dim")
    return line


def format_item_refine_name(item: dict[str, Any]) -> Text:
    """Render a REFINE_NAME per-item line.

    Format: ``<old_name> (chain=<n>) → <new_name>``
    or on escalation: ``<old_name> (chain=<n>) → escalating to <model>``
    """
    old = str(item.get("old_name", ""))
    chain = int(item.get("chain_length", 0))
    escalated = bool(item.get("escalated", False))
    new = str(item.get("new_name", ""))
    model = str(item.get("model", ""))

    line = Text("    ")
    line.append(_clip(old, 30), style="white")
    line.append(f" (chain={chain})", style="dim")
    if escalated:
        line.append(f" → escalating to {model}", style="bold red")
    else:
        line.append(" → ", style="white")
        line.append(new, style="bold white")
    return line


def format_item_generate_docs(item: dict[str, Any]) -> Text:
    """Render a GENERATE_DOCS per-item line.

    Format: ``<name>  "<description first 100 chars>"``
    """
    name = str(item.get("name", ""))
    desc = str(item.get("description", ""))

    line = Text("    ")
    line.append(_clip(name, 30), style="white")
    if desc:
        line.append(f'  "{_clip(desc, 100)}"', style="dim")
    return line


def format_item_review_docs(item: dict[str, Any]) -> Text:
    """Render a REVIEW_DOCS per-item line.

    Same format as REVIEW_NAME but for docs context.
    """
    return format_item_review_name(item)


def format_item_refine_docs(item: dict[str, Any]) -> Text:
    """Render a REFINE_DOCS per-item line.

    Format: ``<name> (rev=<n>) "<description first 100 chars>"``
    """
    name = str(item.get("name", ""))
    rev = int(item.get("revision", 0))
    desc = str(item.get("description", ""))

    line = Text("    ")
    line.append(_clip(name, 30), style="white")
    line.append(f" (rev={rev})", style="dim")
    if desc:
        line.append(f'  "{_clip(desc, 100)}"', style="dim")
    return line


#: Registry of per-item formatters by pool name.
ITEM_FORMATTERS: dict[str, Any] = {
    "generate_name": format_item_generate_name,
    "review_name": format_item_review_name,
    "refine_name": format_item_refine_name,
    "generate_docs": format_item_generate_docs,
    "review_docs": format_item_review_docs,
    "refine_docs": format_item_refine_docs,
}


# ═══════════════════════════════════════════════════════════════════════
# Pool state dataclass
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PoolDisplayState:
    """Observable state for a single pool's display panel.

    All fields are updated by the pool loop or watchers; the display
    reads them on each tick.
    """

    name: str
    completed: int = 0
    total: int = 0
    cost: float = 0.0
    start_time: float = field(default_factory=time.time)

    #: Latest streamed items (newest at right / bottom).
    items: deque = field(default_factory=lambda: deque(maxlen=STREAM_MAXLEN))

    #: Throttle state — set when backlog exceeds cap.
    throttled: bool = False
    throttle_reason: str = ""  # e.g. "review_name backlog 207>200"

    def add_item(self, item: dict[str, Any]) -> None:
        """Push a streamed item into the display deque."""
        self.items.append(item)

    @property
    def rate(self) -> float | None:
        """Items per second (session average)."""
        elapsed = time.time() - self.start_time
        if elapsed <= 0 or self.completed <= 0:
            return None
        return self.completed / elapsed

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)

    @property
    def ratio(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(self.completed / self.total, 1.0)

    @property
    def pct(self) -> float:
        return self.ratio * 100.0


# ═══════════════════════════════════════════════════════════════════════
# Progress bar builder (pure)
# ═══════════════════════════════════════════════════════════════════════


def make_bar(ratio: float, width: int = BAR_WIDTH) -> str:
    """Create a thin Unicode progress bar.

    ``━`` for filled, ``─`` for empty.
    """
    ratio = max(0.0, min(1.0, ratio))
    filled = int(width * ratio)
    return "━" * filled + "─" * (width - filled)


# ═══════════════════════════════════════════════════════════════════════
# Pool panel renderer (pure)
# ═══════════════════════════════════════════════════════════════════════


def render_pool_panel(state: PoolDisplayState) -> Text:
    """Render one pool's block: header line + up to STREAM_MAXLEN item lines.

    Returns a :class:`rich.text.Text` that can be joined with others
    to compose the full display.
    """
    pool_name = state.name
    label = POOL_LABELS.get(pool_name, pool_name.upper())
    style = POOL_STYLES.get(pool_name, "white")

    # ── Header line ───────────────────────────────────────────────
    header = Text()

    # Pool label (with optional throttle suffix)
    if state.throttled:
        header.append(f"  {label}", style=style)
        header.append(f" [paused: {state.throttle_reason}]", style="bold red")
    else:
        header.append(f"  {label}", style=style)

    # Pad to align bar
    label_len = len(header.plain)
    pad = max(1, LABEL_COL - label_len)
    header.append(" " * pad)

    # Progress bar
    filled_count = int(BAR_WIDTH * state.ratio)
    header.append("━" * filled_count, style="green")
    header.append("─" * (BAR_WIDTH - filled_count), style="dim")
    header.append("  ")

    # Counts
    header.append(f"{state.completed}/{state.total}", style="white")
    header.append(f"  {state.pct:.0f}%", style="dim")

    # Rate
    r = state.rate
    if r is not None:
        header.append(f"  {r:.1f}/s", style="dim")

    # Cost
    if state.cost > 0:
        header.append(f"  ${state.cost:.2f}", style="green")

    # ── Streamed items ────────────────────────────────────────────
    result = Text()
    result.append_text(header)

    formatter = ITEM_FORMATTERS.get(pool_name)
    if formatter and state.items:
        for item in state.items:
            result.append("\n")
            result.append_text(formatter(item))

    return result


# ═══════════════════════════════════════════════════════════════════════
# Footer renderers (pure)
# ═══════════════════════════════════════════════════════════════════════


def format_time_value(seconds: float) -> str:
    """Format a duration as ``Xh Ym``, ``Xm Ys``, or ``Xs``."""
    if seconds < 0:
        return "--"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s" if s else f"{m}m"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m:02d}m" if m else f"{h}h"


def compute_eta(
    pools: list[PoolDisplayState],
) -> float | None:
    """ETA in seconds based on aggregate throughput.

    ``remaining_total / current_throughput``.
    """
    total_remaining = sum(p.remaining for p in pools)
    if total_remaining <= 0:
        return 0.0

    total_completed = sum(p.completed for p in pools)
    # Compute overall elapsed from earliest start
    if not pools:
        return None
    earliest = min(p.start_time for p in pools)
    elapsed = time.time() - earliest
    if elapsed <= 0 or total_completed <= 0:
        return None
    throughput = total_completed / elapsed
    return total_remaining / throughput


def compute_etc(
    pools: list[PoolDisplayState],
) -> float | None:
    """Estimated Total Cost: current_cost + cost_per_item × remaining.

    ``cost_per_item`` is the rolling average across all pools.
    """
    total_cost = sum(p.cost for p in pools)
    total_completed = sum(p.completed for p in pools)
    total_remaining = sum(p.remaining for p in pools)

    if total_completed <= 0 or total_remaining <= 0:
        return None

    cost_per_item = total_cost / total_completed
    return total_cost + cost_per_item * total_remaining


def render_footer(
    pools: list[PoolDisplayState],
    *,
    cost_limit: float = 0.0,
    graph_latency_ms: float | None = None,
    llm_latency_s: float | None = None,
    graph_host: str = "graph",
    llm_host: str = "llm",
) -> Text:
    """Render the footer block: TIME + COST + SERVERS.

    All fields are optional; omitted when data is unavailable.
    """
    footer = Text()

    # ── Separator ─────────────────────────────────────────────────
    sep_width = LABEL_COL + BAR_WIDTH + 30
    footer.append("─" * sep_width, style="dim")

    # ── TIME row ──────────────────────────────────────────────────
    earliest = min(p.start_time for p in pools) if pools else time.time()
    elapsed = time.time() - earliest
    eta = compute_eta(pools)

    footer.append("\n  TIME", style="bold white")
    pad = max(1, LABEL_COL - 6)
    footer.append(" " * pad)

    # Time bar: elapsed / (elapsed + eta)
    total_expected = elapsed + (eta if eta and eta > 0 else 0)
    time_ratio = elapsed / total_expected if total_expected > 0 else 1.0
    bar_filled = int(BAR_WIDTH * min(time_ratio, 1.0))
    footer.append("━" * bar_filled, style="blue")
    footer.append("─" * (BAR_WIDTH - bar_filled), style="dim")
    footer.append(f"  {format_time_value(elapsed)}", style="white")
    if eta is not None and eta > 0:
        footer.append(f"  ETA {format_time_value(eta)}", style="dim")

    # ── COST row ──────────────────────────────────────────────────
    total_cost = sum(p.cost for p in pools)
    if total_cost > 0 or cost_limit > 0:
        etc = compute_etc(pools)
        footer.append("\n  COST", style="bold white")
        pad = max(1, LABEL_COL - 6)
        footer.append(" " * pad)

        # Cost bar: cost / limit
        cost_ratio = total_cost / cost_limit if cost_limit > 0 else 0.0
        bar_filled = int(BAR_WIDTH * min(cost_ratio, 1.0))
        cost_color = (
            "green" if cost_ratio < 0.5 else ("yellow" if cost_ratio < 0.8 else "red")
        )
        footer.append("━" * bar_filled, style=cost_color)
        footer.append("─" * (BAR_WIDTH - bar_filled), style="dim")
        footer.append(f"  ${total_cost:.2f}", style="white")
        if etc is not None:
            footer.append(f"  ETC ${etc:.2f}", style="dim")
        if cost_limit > 0:
            footer.append(f"  CAP ${cost_limit:.2f}", style="dim")

    # ── SERVERS row ───────────────────────────────────────────────
    server_parts: list[str] = []
    if graph_latency_ms is not None:
        server_parts.append(f"{graph_host} (avg {graph_latency_ms:.0f}ms)")
    if llm_latency_s is not None:
        server_parts.append(f"{llm_host} (avg {llm_latency_s:.1f}s)")
    if server_parts:
        footer.append("\n  SERVERS", style="bold white")
        pad = max(1, LABEL_COL - 9)
        footer.append(" " * pad)
        footer.append("  ".join(server_parts), style="dim")

    return footer


# ═══════════════════════════════════════════════════════════════════════
# Full display composer
# ═══════════════════════════════════════════════════════════════════════


def render_full_display(
    pools: dict[str, PoolDisplayState],
    *,
    cost_limit: float = 0.0,
    graph_latency_ms: float | None = None,
    llm_latency_s: float | None = None,
    graph_host: str = "graph",
    llm_host: str = "llm",
) -> Text:
    """Compose the full display from 6 pool panels + footer.

    Args:
        pools: Mapping of pool name → :class:`PoolDisplayState`.
        cost_limit: Budget cap in USD.
        graph_latency_ms: Rolling avg graph query latency.
        llm_latency_s: Rolling avg LLM call latency.

    Returns:
        A single :class:`rich.text.Text` suitable for ``Live.update()``.
    """
    display = Text()

    # Title
    display.append("  Standard Name Pipeline\n", style="bold white")

    pool_list = [pools[name] for name in POOL_ORDER if name in pools]

    for i, pstate in enumerate(pool_list):
        if i > 0:
            display.append("\n")
        display.append_text(render_pool_panel(pstate))

    display.append("\n")
    display.append_text(
        render_footer(
            pool_list,
            cost_limit=cost_limit,
            graph_latency_ms=graph_latency_ms,
            llm_latency_s=llm_latency_s,
            graph_host=graph_host,
            llm_host=llm_host,
        )
    )

    return display


# ═══════════════════════════════════════════════════════════════════════
# Live display adapter for run_discovery() harness
# ═══════════════════════════════════════════════════════════════════════


class SN6PoolDisplay:
    """Live display adapter that bridges the 6-pool renderers with the
    ``run_discovery()`` harness.

    Implements the context-manager protocol (``__enter__``/``__exit__``)
    expected by :func:`~imas_codex.cli.discover.common.run_discovery` and
    exposes a ``tick()`` method for periodic display refresh.

    Usage::

        display = SN6PoolDisplay(
            cost_limit=5.0,
            pending_fn=_pending_fn,
            accumulated_cost_fn=_cost_fn,
        )
        display.on_event({"pool": "review_name", "name": "e_temp", "score": 0.85, ...})

    The ``on_event`` method is the callback wired into worker pools.
    Each call pushes an item into the corresponding pool's deque and
    increments its completed counter.
    """

    def __init__(
        self,
        *,
        cost_limit: float = 0.0,
        console: Any | None = None,
        pending_fn: Any | None = None,
        accumulated_cost_fn: Any | None = None,
    ) -> None:
        from rich.console import Console

        self.console = console or Console()
        self.cost_limit = cost_limit
        self._pending_fn = pending_fn
        self._accumulated_cost_fn = accumulated_cost_fn

        # Per-pool display state (6 pools).
        self.pools: dict[str, PoolDisplayState] = {
            name: PoolDisplayState(name=name) for name in POOL_ORDER
        }

        # Lifecycle
        self._live: Any | None = None

        # Service monitor (set by run_discovery after construction).
        self.service_monitor: Any = None

        # Shutdown state (for compatibility with run_discovery harness).
        self._shutting_down = False
        self._shutdown_start: float | None = None

    # ── Event callback (wired into workers) ───────────────────────────

    def on_event(self, ev: dict[str, Any]) -> None:
        """Push a per-item event into the display.

        Called by workers after each successful persist.  Thread-safe
        because deque.append is atomic in CPython.

        Args:
            ev: Event dict with at minimum ``"pool"`` key matching one
                of :data:`POOL_ORDER`.  Additional keys are pool-specific
                and consumed by the per-item formatters.
        """
        pool_name = ev.get("pool", "")
        state = self.pools.get(pool_name)
        if state is None:
            return
        state.add_item(ev)
        state.completed += 1
        cost = ev.get("cost", 0.0)
        if cost:
            state.cost += float(cost)

    # ── Pending count / total refresh ─────────────────────────────────

    def refresh_pending(self) -> None:
        """Refresh pool totals from the pending-count callback.

        Mapping from pending-fn keys to pool names:

        - ``generate_name`` ← ``draft`` (pending) + ``draft_done`` (baseline)
        - ``review_name``   ← ``review_names`` + ``review_names_done``
        - ``refine_name``   ← ``revise``
        - ``generate_docs`` ← ``enrich`` + ``enrich_done``
        - ``review_docs``   ← ``review_docs`` + ``review_docs_done``
        - ``refine_docs``   — no direct graph query; total stays at 0.
        """
        if self._pending_fn is None:
            return
        try:
            counts = self._pending_fn()
        except Exception:
            return

        # counts may be either a dict or a list of (name, count) tuples
        # depending on which _pending_fn variant is wired.
        if isinstance(counts, list):
            counts = dict(counts)

        _MAP: dict[str, tuple[str, str | None]] = {
            "generate_name": ("draft", "draft_done"),
            "review_name": ("review_names", "review_names_done"),
            "refine_name": ("revise", None),
            "generate_docs": ("enrich", "enrich_done"),
            "review_docs": ("review_docs", "review_docs_done"),
        }
        for pool_name, (pending_key, done_key) in _MAP.items():
            state = self.pools.get(pool_name)
            if state is None:
                continue
            pending = int(counts.get(pending_key, 0))
            # Seed baseline from done counts on first refresh
            if done_key is not None:
                done = int(counts.get(done_key, 0))
                if done > state.completed:
                    state.completed = done
            new_total = state.completed + pending
            if new_total > state.total:
                state.total = new_total

    # ── Display rendering ─────────────────────────────────────────────

    def _build_display(self) -> Any:
        """Build the complete display renderable."""
        from rich.panel import Panel

        content = render_full_display(
            self.pools,
            cost_limit=self.cost_limit,
        )

        # Shutdown indicator
        if self._shutting_down:
            content.append("\n")
            content.append("  SHUTTING DOWN", style="bold yellow")
            if self._shutdown_start is not None:
                import time as _time

                elapsed = _time.time() - self._shutdown_start
                content.append(f"  ({elapsed:.1f}s)", style="dim")

        return Panel(
            content,
            border_style="yellow" if self._shutting_down else "cyan",
            padding=(0, 1),
        )

    # ── Tick (called by run_discovery ticker task) ────────────────────

    def tick(self) -> None:
        """Periodic refresh: update pending counts and repaint."""
        self.refresh_pending()
        self._refresh()

    def _refresh(self) -> None:
        """Repaint the Live display."""
        if self._live is not None:
            self._live.update(self._build_display())

    # ── Lifecycle (context manager for run_discovery) ─────────────────

    def __enter__(self) -> SN6PoolDisplay:
        from rich.live import Live

        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
            vertical_overflow="visible",
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._live is not None:
            self._live.__exit__(*args)

    # ── Harness compatibility ─────────────────────────────────────────

    def refresh_from_graph(self, facility: str) -> None:
        """Called by run_discovery graph-refresh task.  Updates pending."""
        self.refresh_pending()
        self._refresh()

    def print_summary(self) -> None:
        """Print a compact summary after the run completes."""
        total_items = sum(p.completed for p in self.pools.values())
        total_cost = sum(p.cost for p in self.pools.values())
        if total_items > 0 or total_cost > 0:
            self.console.print()
            for name in POOL_ORDER:
                p = self.pools[name]
                if p.completed > 0:
                    parts = [
                        f"  {POOL_LABELS[name]}: {p.completed:,}",
                    ]
                    if p.cost > 0:
                        parts.append(f"${p.cost:.2f}")
                    self.console.print("  ".join(parts))
            if total_cost > 0:
                self.console.print(f"  TOTAL COST: ${total_cost:.2f}")

    def signal_shutdown(self) -> None:
        """Signal graceful shutdown (3-press handler)."""
        self._shutting_down = True
        if self._shutdown_start is None:
            import time as _time

            self._shutdown_start = _time.time()
        self._refresh()

    # Alias expected by cli/shutdown.py
    begin_shutdown = signal_shutdown


__all__ = [
    "BAR_WIDTH",
    "ITEM_FORMATTERS",
    "LABEL_COL",
    "POOL_LABELS",
    "POOL_ORDER",
    "POOL_STYLES",
    "PoolDisplayState",
    "SCORE_GREEN_THRESHOLD",
    "SCORE_YELLOW_THRESHOLD",
    "SN6PoolDisplay",
    "STREAM_MAXLEN",
    "compute_eta",
    "compute_etc",
    "format_item_generate_docs",
    "format_item_generate_name",
    "format_item_refine_docs",
    "format_item_refine_name",
    "format_item_review_docs",
    "format_item_review_name",
    "format_time_value",
    "make_bar",
    "render_footer",
    "render_full_display",
    "render_pool_panel",
    "score_color",
]
