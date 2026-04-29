"""Per-pool streaming display for the 6-pool SN pipeline.

Six pools rendered individually using the canonical ``BaseProgressDisplay``
layout (full-width panel, per-worker streaming via ``PipelineRowConfig``):

    DRAFT         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━────────      720  36%
    electron_temperature_in_core_plasma                          1.5/s
    → e_temp_core                                                $1.20
    REVIEW NAME   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━────────      640  32%
    e_temp_core                                                  2.1/s
    0.83  "Good grammar; documentation could be…"                $0.85
    ...
    ─────────────────────────────────────────────────────────────
    TIME  ━━━━━━━━━━━━────────────────  18m 32s  ETA 2h 14m
    COST  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  $4.55  ETC $9.20

Inherits ``BaseProgressDisplay`` for full-width rendering, canonical
HEADER → SERVERS → PIPELINE → RESOURCES layout, and service-monitor
integration.  Per-item streaming uses ``PipelineRowConfig`` (3-line
per pool) with the standard ``build_pipeline_section`` renderer.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rich.text import Text

from imas_codex.discovery.base.progress import (
    BaseProgressDisplay,
    PipelineRowConfig,
    ResourceConfig,
    WorkerStats,
    build_pipeline_section,
    build_resource_section,
    compute_parallel_eta,
)

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

#: Display labels — short labels that fit the canonical LABEL_WIDTH (12).
POOL_LABELS: dict[str, str] = {
    "generate_name": "DRAFT NAME",
    "review_name": "REVIEW NAME",
    "refine_name": "REFINE NAME",
    "generate_docs": "DRAFT DOCS",
    "review_docs": "REVIEW DOCS",
    "refine_docs": "REFINE DOCS",
}

#: Legacy long labels (upper-case, underscore-separated).
#: Used by legacy :func:`render_pool_panel` for backward compat.
_LEGACY_POOL_LABELS: dict[str, str] = {
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

#: Maximum streamed items kept per pool (legacy compat).
STREAM_MAXLEN = 3

#: Review-score color thresholds.
SCORE_GREEN_THRESHOLD = 0.85
SCORE_YELLOW_THRESHOLD = 0.65


# ═══════════════════════════════════════════════════════════════════════
# Event → PipelineRowConfig field mapping (per-pool)
# ═══════════════════════════════════════════════════════════════════════


def _map_generate_name(ev: dict[str, Any]) -> dict[str, Any]:
    """Map a generate_name event to PipelineRowConfig stream fields."""
    source = ev.get("source", ev.get("dd_path", ""))
    name = ev.get("name", "")
    return {
        "primary_text": str(source),
        "primary_text_style": "dim",
        "description": f"→ {name}" if name else "",
    }


def _map_review_name(ev: dict[str, Any]) -> dict[str, Any]:
    """Map a review_name event to PipelineRowConfig stream fields."""
    name = str(ev.get("name", ""))
    raw_score = ev.get("score")
    sc = float(raw_score) if raw_score is not None else None
    comment = str(ev.get("comment", ""))
    return {
        "primary_text": name,
        "primary_text_style": "white",
        "score_value": sc,
        "description": f'"{comment}"' if comment else "",
    }


def _map_refine_name(ev: dict[str, Any]) -> dict[str, Any]:
    """Map a refine_name event to PipelineRowConfig stream fields."""
    old = str(ev.get("old_name", ""))
    chain = int(ev.get("chain_length", 0))
    escalated = bool(ev.get("escalated", False))
    new = str(ev.get("new_name", ""))
    model = str(ev.get("model", ""))
    if escalated:
        desc = f"(chain={chain}) → escalating to {model}"
    else:
        desc = f"(chain={chain}) → {new}"
    return {
        "primary_text": old,
        "primary_text_style": "white",
        "description": desc,
    }


def _map_generate_docs(ev: dict[str, Any]) -> dict[str, Any]:
    """Map a generate_docs event to PipelineRowConfig stream fields."""
    name = str(ev.get("name", ""))
    desc = str(ev.get("description", ""))
    return {
        "primary_text": name,
        "primary_text_style": "white",
        "description": f'"{desc}"' if desc else "",
    }


def _map_review_docs(ev: dict[str, Any]) -> dict[str, Any]:
    """Map a review_docs event to PipelineRowConfig stream fields."""
    return _map_review_name(ev)


def _map_refine_docs(ev: dict[str, Any]) -> dict[str, Any]:
    """Map a refine_docs event to PipelineRowConfig stream fields."""
    name = str(ev.get("name", ""))
    rev = int(ev.get("revision", 0))
    desc = str(ev.get("description", ""))
    return {
        "primary_text": name,
        "primary_text_style": "white",
        "description": f'(rev={rev}) "{desc}"' if desc else f"(rev={rev})",
    }


#: Registry mapping pool name → event-to-stream-fields mapper.
_EVENT_MAPPERS: dict[str, Any] = {
    "generate_name": _map_generate_name,
    "review_name": _map_review_name,
    "refine_name": _map_refine_name,
    "generate_docs": _map_generate_docs,
    "review_docs": _map_review_docs,
    "refine_docs": _map_refine_docs,
}

# ═══════════════════════════════════════════════════════════════════════
# Legacy per-item renderers (kept for backward compat / test imports)
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
    """Render a GENERATE_NAME per-item line."""
    source = item.get("source", item.get("dd_path", ""))
    name = item.get("name", "")
    line = Text("    ")
    line.append(_clip(str(source), 40), style="dim")
    line.append("  →  ", style="white")
    line.append(str(name), style="bold white")
    return line


def format_item_review_name(item: dict[str, Any]) -> Text:
    """Render a REVIEW_NAME per-item line."""
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
    """Render a REFINE_NAME per-item line."""
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
    """Render a GENERATE_DOCS per-item line."""
    name = str(item.get("name", ""))
    desc = str(item.get("description", ""))
    line = Text("    ")
    line.append(_clip(name, 30), style="white")
    if desc:
        line.append(f'  "{_clip(desc, 100)}"', style="dim")
    return line


def format_item_review_docs(item: dict[str, Any]) -> Text:
    """Render a REVIEW_DOCS per-item line."""
    return format_item_review_name(item)


def format_item_refine_docs(item: dict[str, Any]) -> Text:
    """Render a REFINE_DOCS per-item line."""
    name = str(item.get("name", ""))
    rev = int(item.get("revision", 0))
    desc = str(item.get("description", ""))
    line = Text("    ")
    line.append(_clip(name, 30), style="white")
    line.append(f" (rev={rev})", style="dim")
    if desc:
        line.append(f'  "{_clip(desc, 100)}"', style="dim")
    return line


#: Registry of per-item formatters by pool name (legacy).
ITEM_FORMATTERS: dict[str, Any] = {
    "generate_name": format_item_generate_name,
    "review_name": format_item_review_name,
    "refine_name": format_item_refine_name,
    "generate_docs": format_item_generate_docs,
    "review_docs": format_item_review_docs,
    "refine_docs": format_item_refine_docs,
}


# ═══════════════════════════════════════════════════════════════════════
# Pool state dataclass (legacy — kept for backward compat / tests)
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
    throttle_reason: str = ""

    #: Timestamp of the last completed item (for stall detection).
    last_completion_at: float | None = None

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
# Legacy rendering utilities (kept for backward compat / tests)
# ═══════════════════════════════════════════════════════════════════════

#: Legacy label column width (superseded by LABEL_WIDTH from base).
LABEL_COL = 16

#: Legacy progress bar width (superseded by terminal-responsive bar_width).
BAR_WIDTH = 36


def make_bar(ratio: float, width: int = BAR_WIDTH) -> str:
    """Create a thin Unicode progress bar.

    ``━`` for filled, ``─`` for empty.
    """
    ratio = max(0.0, min(1.0, ratio))
    filled = int(width * ratio)
    return "━" * filled + "─" * (width - filled)


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


def compute_eta(pools: list[PoolDisplayState]) -> float | None:
    """ETA in seconds based on aggregate throughput."""
    total_remaining = sum(p.remaining for p in pools)
    if total_remaining <= 0:
        return 0.0
    total_completed = sum(p.completed for p in pools)
    if not pools:
        return None
    earliest = min(p.start_time for p in pools)
    elapsed = time.time() - earliest
    if elapsed <= 0 or total_completed <= 0:
        return None
    throughput = total_completed / elapsed
    return total_remaining / throughput


def compute_etc(pools: list[PoolDisplayState]) -> float | None:
    """Estimated Total Cost: current_cost + cost_per_item × remaining."""
    total_cost = sum(p.cost for p in pools)
    total_completed = sum(p.completed for p in pools)
    total_remaining = sum(p.remaining for p in pools)
    if total_completed <= 0 or total_remaining <= 0:
        return None
    cost_per_item = total_cost / total_completed
    return total_cost + cost_per_item * total_remaining


def render_pool_panel(state: PoolDisplayState) -> Text:
    """Render one pool's block using legacy custom layout.

    .. deprecated:: Use ``SN6PoolDisplay._build_pipeline_section`` instead.
    """
    pool_name = state.name
    label = _LEGACY_POOL_LABELS.get(pool_name, pool_name.upper())
    style = POOL_STYLES.get(pool_name, "white")

    header = Text()
    if state.throttled:
        header.append(f"  {label}", style=style)
        header.append(f" [paused: {state.throttle_reason}]", style="bold red")
    else:
        header.append(f"  {label}", style=style)

    label_len = len(header.plain)
    pad = max(1, LABEL_COL - label_len)
    header.append(" " * pad)
    filled_count = int(BAR_WIDTH * state.ratio)
    header.append("━" * filled_count, style="green")
    header.append("─" * (BAR_WIDTH - filled_count), style="dim")
    header.append("  ")
    header.append(f"{state.completed}/{state.total}", style="white")
    header.append(f"  {state.pct:.0f}%", style="dim")
    r = state.rate
    if r is not None:
        header.append(f"  {r:.1f}/s", style="dim")
    if state.cost > 0:
        header.append(f"  ${state.cost:.2f}", style="green")

    result = Text()
    result.append_text(header)
    formatter = ITEM_FORMATTERS.get(pool_name)
    if formatter and state.items:
        for item in state.items:
            result.append("\n")
            result.append_text(formatter(item))
    return result


def render_footer(
    pools: list[PoolDisplayState],
    *,
    cost_limit: float = 0.0,
    graph_latency_ms: float | None = None,
    llm_latency_s: float | None = None,
    graph_host: str = "graph",
    llm_host: str = "llm",
) -> Text:
    """Render the footer block using legacy custom layout.

    .. deprecated:: Use ``SN6PoolDisplay._build_resources_section`` instead.
    """
    footer = Text()
    sep_width = LABEL_COL + BAR_WIDTH + 30
    footer.append("─" * sep_width, style="dim")
    earliest = min(p.start_time for p in pools) if pools else time.time()
    elapsed = time.time() - earliest
    eta = compute_eta(pools)
    footer.append("\n  TIME", style="bold white")
    pad = max(1, LABEL_COL - 6)
    footer.append(" " * pad)
    total_expected = elapsed + (eta if eta and eta > 0 else 0)
    time_ratio = elapsed / total_expected if total_expected > 0 else 1.0
    bar_filled = int(BAR_WIDTH * min(time_ratio, 1.0))
    footer.append("━" * bar_filled, style="blue")
    footer.append("─" * (BAR_WIDTH - bar_filled), style="dim")
    footer.append(f"  {format_time_value(elapsed)}", style="white")
    if eta is not None and eta > 0:
        footer.append(f"  ETA {format_time_value(eta)}", style="dim")
    total_cost = sum(p.cost for p in pools)
    if total_cost > 0 or cost_limit > 0:
        etc = compute_etc(pools)
        footer.append("\n  COST", style="bold white")
        pad = max(1, LABEL_COL - 6)
        footer.append(" " * pad)
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


def render_full_display(
    pools: dict[str, PoolDisplayState],
    *,
    cost_limit: float = 0.0,
    graph_latency_ms: float | None = None,
    llm_latency_s: float | None = None,
    graph_host: str = "graph",
    llm_host: str = "llm",
) -> Text:
    """Compose legacy display from 6 pool panels + footer.

    .. deprecated:: Use ``SN6PoolDisplay`` (canonical BaseProgressDisplay
        subclass) instead.
    """
    display = Text()
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
# SN6PoolDisplay — canonical BaseProgressDisplay subclass
# ═══════════════════════════════════════════════════════════════════════


class SN6PoolDisplay(BaseProgressDisplay):
    """Full-width 6-pool display using the canonical ``BaseProgressDisplay``.

    Renders 6 pipeline rows (one per pool) with per-worker streaming
    via ``PipelineRowConfig``, plus TIME/COST resource gauges.  Inherits
    full-width panel, service-monitor SERVERS section, and shutdown
    handling from ``BaseProgressDisplay``.

    Usage::

        display = SN6PoolDisplay(
            cost_limit=5.0,
            pending_fn=_pending_fn,
            accumulated_cost_fn=_cost_fn,
        )
        display.on_event({"pool": "review_name", "name": "e_temp", "score": 0.85, ...})

    The ``on_event`` method is the callback wired into worker pools.
    Each call pushes an item into the corresponding pool's
    ``WorkerStats.stream_queue`` and increments its counter.
    """

    def __init__(
        self,
        *,
        cost_limit: float = 0.0,
        console: Any | None = None,
        pending_fn: Any | None = None,
        accumulated_cost_fn: Any | None = None,
    ) -> None:
        super().__init__(
            facility="sn",
            cost_limit=cost_limit,
            console=console,
            title_suffix="Standard Name",
        )
        self._pending_fn = pending_fn
        self._accumulated_cost_fn = accumulated_cost_fn

        # Per-pool observable state (6 pools).
        # PoolDisplayState tracks completed/total/cost; WorkerStats drives
        # the canonical streaming display via stream_queue.
        self.pools: dict[str, PoolDisplayState] = {
            name: PoolDisplayState(name=name) for name in POOL_ORDER
        }
        self._pool_stats: dict[str, WorkerStats] = {
            name: WorkerStats() for name in POOL_ORDER
        }

    # ── Event callback (wired into workers) ───────────────────────────

    def on_event(self, ev: dict[str, Any]) -> None:
        """Push a per-item event into the display.

        Called by workers after each successful persist.  Thread-safe
        because deque.append is atomic in CPython.

        Args:
            ev: Event dict with at minimum ``"pool"`` key matching one
                of :data:`POOL_ORDER`.  Additional keys are pool-specific.
        """
        pool_name = ev.get("pool", "")
        state = self.pools.get(pool_name)
        if state is None:
            return
        state.add_item(ev)
        state.completed += 1
        state.last_completion_at = time.time()
        cost = ev.get("cost", 0.0)
        if cost:
            state.cost += float(cost)

        # Push to canonical stream queue for per-worker display.
        ws = self._pool_stats.get(pool_name)
        if ws is not None:
            mapper = _EVENT_MAPPERS.get(pool_name)
            if mapper:
                stream_item = mapper(ev)
                ws.stream_queue.add([stream_item])
            ws.processed = state.completed
            ws.total = max(state.total, state.completed)
            ws.cost = state.cost

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

        # counts may be either a dict or a list of (name, count) tuples.
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
            if done_key is not None:
                done = int(counts.get(done_key, 0))
                if done > state.completed:
                    state.completed = done
            new_total = state.completed + pending
            if new_total > state.total:
                state.total = new_total

            # Sync WorkerStats for canonical rendering.
            ws = self._pool_stats.get(pool_name)
            if ws is not None:
                ws.processed = state.completed
                ws.total = max(state.total, state.completed)
                ws.cost = state.cost

    # ── Canonical display methods (override BaseProgressDisplay) ──────

    def _build_pipeline_section(self) -> Text:
        """Build pipeline section using canonical PipelineRowConfig."""
        rows: list[PipelineRowConfig] = []
        for pool_name in POOL_ORDER:
            state = self.pools[pool_name]
            ws = self._pool_stats[pool_name]
            label = POOL_LABELS[pool_name]
            style = POOL_STYLES[pool_name]

            completed = state.completed
            total = max(state.total, completed, 1)

            # Stream item from WorkerStats queue.
            primary_text = ""
            primary_text_style = "white"
            description = ""
            score_value: float | None = None
            si = ws._current_stream_item
            if si:
                primary_text = si.get("primary_text", "")
                primary_text_style = si.get("primary_text_style", "white")
                description = si.get("description", "")
                _sv = si.get("score_value")
                if isinstance(_sv, int | float):
                    score_value = float(_sv)

            rows.append(
                PipelineRowConfig(
                    name=label,
                    style=style,
                    completed=completed,
                    total=total,
                    rate=state.rate,
                    cost=state.cost if state.cost > 0 else None,
                    primary_text=primary_text,
                    primary_text_style=primary_text_style,
                    description=description,
                    score_value=score_value,
                    is_processing=completed > 0 and completed < total,
                    is_complete=completed > 0 and completed >= total,
                )
            )
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        """Build TIME + COST resource gauges using pipeline-aware ETC.

        Replaces the old per-pool independent projection with a
        hybrid pipeline-flow model that accounts for upstream work
        flowing through downstream pools.
        """
        from imas_codex.standard_names.cost_model import (
            compute_cycle_estimates,
            compute_pipeline_etc,
            detect_stall,
            resolve_pool_cpi,
        )

        total_cost = sum(p.cost for p in self.pools.values())

        # ETA: parallel ETA across pools with remaining work.
        work_items: list[tuple[int, float | None]] = []
        for pool_name in POOL_ORDER:
            state = self.pools[pool_name]
            remaining = state.remaining
            if remaining > 0 and state.rate is not None and state.rate > 0:
                work_items.append((remaining, state.rate))

        eta = compute_parallel_eta(work_items)

        # Accumulated cost from graph (cross-run).
        accumulated = total_cost
        if self._accumulated_cost_fn:
            try:
                graph_cost = self._accumulated_cost_fn()
                accumulated = graph_cost + total_cost
            except Exception:
                pass

        # --- Pipeline-aware ETC ---
        projected: float | None = None
        stalled = False
        try:
            from imas_codex.standard_names.graph_ops import (
                query_historical_cpi,
                query_pipeline_buckets,
            )

            buckets = query_pipeline_buckets()

            # Build CycleEstimates from this-run pool counters.
            gn = self.pools["generate_name"]
            rn = self.pools["review_name"]
            rfn = self.pools["refine_name"]
            rd = self.pools["review_docs"]
            rfd = self.pools["refine_docs"]

            cycles = compute_cycle_estimates(
                refine_name_done=rfn.completed,
                name_review_first_pass_done=rn.completed,
                refine_docs_done=rfd.completed,
                docs_review_first_pass_done=rd.completed,
                # accepted_count: names reviewed with score >= threshold
                accepted_count=max(rn.completed - rfn.completed, 0),
                total_completed_name_stage=rn.completed,
                sources_attempted=gn.total if gn.total > 0 else 0,
                names_drafted=gn.completed,
            )

            # Resolve CPI per pool.
            historical = query_historical_cpi()

            # Build sibling CPI fallback map.
            # refine_name ≈ 1.0 × review_name; refine_docs ≈ 1.0 × review_docs
            _sibling_map: dict[str, str] = {
                "refine_name": "review_name",
                "refine_docs": "review_docs",
            }

            cpis: dict[str, Any] = {}
            for pool_name in POOL_ORDER:
                state = self.pools[pool_name]
                sibling_pool = _sibling_map.get(pool_name)
                sibling_cpi: float | None = None
                if sibling_pool and sibling_pool in cpis:
                    sibling_cpi = cpis[sibling_pool].value

                cpis[pool_name] = resolve_pool_cpi(
                    pool=pool_name,
                    observed_cost=state.cost,
                    observed_completed=state.completed,
                    historical=historical,
                    sibling_cpi=sibling_cpi,
                )

            projected = compute_pipeline_etc(
                buckets=buckets,
                cycles=cycles,
                cpis=cpis,
                accumulated_cost=accumulated,
            )

            # Stall detection.
            pool_pending = {name: self.pools[name].remaining for name in POOL_ORDER}
            pool_last_at = {
                name: self.pools[name].last_completion_at for name in POOL_ORDER
            }
            stalled = detect_stall(pool_pending, pool_last_at, time.time())
        except Exception:
            # Fallback: no projection if graph queries fail.
            projected = None

        # Format ETC for display.
        # When stalled, suppress ETC (canonical renderer can't render "∞").
        etc_value: float | None = projected
        if stalled:
            etc_value = None

        config = ResourceConfig(
            elapsed=self.elapsed,
            eta=eta,
            run_cost=total_cost if total_cost > 0 else None,
            cost_limit=self.cost_limit if self.cost_limit > 0 else None,
            accumulated_cost=accumulated,
            etc=etc_value,
        )
        return build_resource_section(config, self.gauge_width)

    # ── Tick (called by run_discovery ticker task) ────────────────────

    def tick(self) -> None:
        """Periodic refresh: drain stream queues, update pending, repaint."""
        # Drain stream queues → _current_stream_item for each pool.
        for ws in self._pool_stats.values():
            item = ws.stream_queue.pop()
            if item is not None:
                ws._current_stream_item = item
            elif ws.stream_queue.is_stale():
                ws._current_stream_item = None

        self.refresh_pending()
        self._refresh()

    # ── Harness compatibility ─────────────────────────────────────────

    def refresh_from_graph(self, facility: str) -> None:
        """Called by run_discovery graph-refresh task."""
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

    def on_worker_status(self, group: Any) -> None:
        """Callback for worker status updates (harness compat)."""
        self.update_worker_status(group)


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
