"""Progress display for standard name build pipeline.

Contains:

- :class:`StandardNameProgressDisplay` — 3-stage display for the single-pass
  ``--paths`` pipeline (extract → compose → finalize).
- :class:`SNLoopState` — observable state for loop-mode ``sn run``, consumed
  by :class:`DataDrivenProgressDisplay` via ``StageDisplaySpec`` declarations.
- :func:`build_sn_loop_stages` — stage specs for the 5 loop-mode phases.
- :class:`SNPoolState` — observable state for pool-mode ``sn run`` (Phase 8),
  aggregating 5 pools into 3 display rows with per-subpool health.
- :func:`build_sn_pool_stages` — stage specs for the 3 pool-mode rows.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.progress import (
    DataDrivenProgressDisplay,
    StageDisplaySpec,
    WorkerStats,
)
from imas_codex.discovery.base.supervision import SupervisedWorkerGroup

if TYPE_CHECKING:
    from imas_codex.standard_names.pools import PoolHealth

logger = logging.getLogger(__name__)


def build_sn_stages() -> list[StageDisplaySpec]:
    """Build the stage specs for the SN progress display.

    3 rows: EXTRACT → COMPOSE → FINALIZE
    """
    return [
        StageDisplaySpec(
            name="EXTRACT",
            style="bold blue",
            group="extract",
            stats_attr="extract_stats",
            phase_attr="extract_phase",
        ),
        StageDisplaySpec(
            name="COMPOSE",
            style="bold magenta",
            group="compose",
            stats_attr="compose_stats",
            phase_attr="compose_phase",
        ),
        StageDisplaySpec(
            name="FINALIZE",
            style="bold green",
            group="finalize",
            stats_attr="finalize_stats",
        ),
    ]


class StandardNameProgressDisplay(DataDrivenProgressDisplay):
    """Rich progress display for the SN build pipeline.

    Shows 3 phases: Extract → Compose → Finalize
    where Finalize groups validate + consolidate + persist.
    """

    def __init__(
        self,
        source: str = "dd",
        *,
        console: Any | None = None,
        cost_limit: float = 5.0,
        mode_label: str | None = None,
    ):
        stages = build_sn_stages()
        super().__init__(
            facility="sn",
            cost_limit=cost_limit,
            console=console,
            stages=stages,
            title_suffix="Standard Name Build",
            mode_label=mode_label,
        )
        self.source = source

    def on_worker_status(self, group: SupervisedWorkerGroup) -> None:
        """Callback for worker status updates."""
        self.update_worker_status(group)

    def refresh_from_graph(self, facility: str) -> None:
        """Refresh display data from graph (no-op for SN build)."""
        self._refresh()

    def print_summary(self) -> None:
        """Print a brief summary after build completes."""
        if self._engine_state is None:
            return

        lines: list[str] = []
        for label, attr in [
            ("EXTRACT", "extract_stats"),
            ("COMPOSE", "compose_stats"),
            ("FINALIZE", "finalize_stats"),
        ]:
            stats = getattr(self._engine_state, attr, None)
            if stats and stats.processed > 0:
                parts = [f"{label}: {stats.processed:,}"]
                if stats.cost > 0:
                    parts.append(f"${stats.cost:.2f}")
                lines.append("  ".join(parts))

        if lines:
            self.console.print()
            for line in lines:
                self.console.print(f"  {line}")


# ═══════════════════════════════════════════════════════════════════════
# Loop-mode state + stage specs (sn run without --paths)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SNLoopState:
    """Observable state for the SN loop ``DataDrivenProgressDisplay``.

    Six independent worker stats feed the display (one per phase function):

    - ``extract_stats``       — DD/signal source extraction (graph queries)
    - ``draft_stats``         — initial compose (Source extracted → composed)
    - ``revise_stats``        — regen with reviewer feedback (low-score names)
    - ``describe_stats``      — enrichment (named → enriched)
    - ``review_names_stats``  — name-side review (reviewed_name_at)
    - ``review_docs_stats``   — docs-side review (reviewed_docs_at)

    Each spec gets a unique ``StageDisplaySpec.group`` so the display framework
    tracks running/completion/worker-counts per row independently. Visual
    cohesion comes from adjacent placement and shared color families, not
    shared groups.
    """

    extract_stats: WorkerStats = field(default_factory=WorkerStats)
    draft_stats: WorkerStats = field(default_factory=WorkerStats)
    revise_stats: WorkerStats = field(default_factory=WorkerStats)
    describe_stats: WorkerStats = field(default_factory=WorkerStats)
    review_names_stats: WorkerStats = field(default_factory=WorkerStats)
    review_docs_stats: WorkerStats = field(default_factory=WorkerStats)


def build_sn_loop_stages(
    *,
    skip_generate: bool = False,
    skip_enrich: bool = False,
    skip_review: bool = False,
    min_score: float | None = None,
) -> list[StageDisplaySpec]:
    """Build the stage specs for the SN loop progress display.

    Six rows, one per phase function. Each row has a unique ``group`` so the
    framework tracks per-row state independently:

    - **EXTRACT**       — graph-side extraction (DD/signal source nodes)
    - **DRAFT**         — initial compose
    - **REVISE**        — regen with reviewer feedback (always visible when
      generate is enabled; pending=0 when ``min_score`` is unset)
    - **DESCRIBE**      — enrichment
    - **NAMES**         — name-side review (was REVIEW NAMES)
    - **DOCUMENTATION** — docs-side review (was REVIEW DOCS)

    The ``min_score`` parameter is accepted for backwards compatibility but no
    longer affects whether REVISE is visible — it always shows when the
    generate group is enabled.
    """
    del min_score  # accepted for compatibility; revise visibility no longer gated on it
    revise_disabled = skip_generate
    return [
        StageDisplaySpec(
            name="EXTRACT",
            style="bold blue",
            group="extract",
            stats_attr="extract_stats",
            disabled=skip_generate,
        ),
        StageDisplaySpec(
            name="DRAFT",
            style="bold magenta",
            group="draft",
            stats_attr="draft_stats",
            disabled=skip_generate,
        ),
        StageDisplaySpec(
            name="REVISE",
            style="magenta",
            group="revise",
            stats_attr="revise_stats",
            disabled=revise_disabled,
        ),
        StageDisplaySpec(
            name="DESCRIBE",
            style="bold cyan",
            group="describe",
            stats_attr="describe_stats",
            disabled=skip_enrich,
        ),
        StageDisplaySpec(
            name="NAMES",
            style="bold yellow",
            group="review_names",
            stats_attr="review_names_stats",
            disabled=skip_review,
        ),
        StageDisplaySpec(
            name="DOCUMENTATION",
            style="yellow",
            group="review_docs",
            stats_attr="review_docs_stats",
            disabled=skip_review,
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════
# Pool-mode state + stage specs (Phase 8 concurrent pools)
# ═══════════════════════════════════════════════════════════════════════

# Default wedge threshold: a subpool is wedged when
# ``(now - last_progress_at) > WEDGE_THRESHOLD`` AND pending > 0.
# 2 × backoff cap (3s base) ≈ 6s is a reasonable default.
WEDGE_THRESHOLD: float = 6.0

# Mapping from display row → (subpool label, pool name) pairs.
# Used by :func:`format_pool_health_text` and :class:`SNPoolState`.
_GENERATE_SUBPOOLS: tuple[tuple[str, str], ...] = (
    ("compose", "generate"),
    ("regen", "regen"),
)
_REVIEW_SUBPOOLS: tuple[tuple[str, str], ...] = (
    ("names", "review_names"),
    ("docs", "review_docs"),
)
_ENRICH_SUBPOOLS: tuple[tuple[str, str], ...] = (("pending", "enrich"),)


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string (e.g. '5m ago')."""
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m ago"
    hours = minutes / 60
    return f"{hours:.1f}h ago"


def format_pool_health_text(
    subpools: tuple[tuple[str, str], ...],
    health_map: dict[str, PoolHealth],
    *,
    wedge_threshold: float = WEDGE_THRESHOLD,
    now: float | None = None,
) -> str:
    """Compose per-subpool status text with wedge detection.

    Args:
        subpools: ``(display_label, pool_name)`` pairs for this row.
        health_map: Mapping of pool name → :class:`PoolHealth`.
        wedge_threshold: Seconds since last progress before a subpool
            is considered wedged (provided ``pending_count > 0``).
        now: Current time (``time.time()``); defaults to ``time.time()``.

    Returns:
        Status text string with optional Rich markup.  Examples::

            "names=42 docs=8"
            "compose=15 regen=3"
            "pending=27"
            "names=42 [red]docs[/red]=8 — wedged docs (5m ago)"
    """
    ts = now if now is not None else time.time()

    # Single subpool → simple "pending=N" form.
    if len(subpools) == 1:
        label, pool_name = subpools[0]
        ph = health_map.get(pool_name)
        if ph is None:
            return ""
        pending = ph.pending_count
        if pending <= 0:
            return ""
        wedged = ph.is_wedged(poll_interval=wedge_threshold / 2.0, now=ts)
        if wedged:
            elapsed = _format_elapsed(ts - ph.last_progress_at)
            return f"[red]{label}[/red]={pending} — wedged {label} ({elapsed})"
        return f"{label}={pending}"

    # Multi-subpool → "label1=N label2=M" with per-subpool wedge.
    parts: list[str] = []
    wedged_labels: list[str] = []
    wedged_elapsed: list[str] = []

    for label, pool_name in subpools:
        ph = health_map.get(pool_name)
        if ph is None:
            continue
        pending = ph.pending_count
        if pending <= 0:
            continue
        wedged = ph.is_wedged(poll_interval=wedge_threshold / 2.0, now=ts)
        if wedged:
            parts.append(f"[red]{label}[/red]={pending}")
            wedged_labels.append(label)
            elapsed = _format_elapsed(ts - ph.last_progress_at)
            wedged_elapsed.append(f"{label} ({elapsed})")
        else:
            parts.append(f"{label}={pending}")

    if not parts:
        return ""

    text = " ".join(parts)
    if wedged_labels:
        detail = ", ".join(wedged_elapsed)
        text += f" — wedged {detail}"
    return text


@dataclass
class SNPoolState:
    """Observable state for the Phase 8 concurrent pool display.

    Three display rows, each backed by a :class:`WorkerStats`:

    - **GENERATE** — aggregates ``generate`` (compose) + ``regen`` pools.
    - **ENRICH** — single ``enrich`` pool.
    - **REVIEW** — aggregates ``review_names`` + ``review_docs`` pools.

    :class:`PoolHealth` references are injected after pool construction
    via :meth:`set_pool_health`.  :meth:`refresh_pool_health` reads live
    health data and composes ``status_text`` (with Rich markup for wedge
    indicators) on each ``WorkerStats``.
    """

    generate_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    review_stats: WorkerStats = field(default_factory=WorkerStats)

    # Injected after pool construction — maps pool name → PoolHealth.
    _pool_health: dict[str, PoolHealth] = field(default_factory=dict)

    def set_pool_health(self, pool_name: str, health: PoolHealth) -> None:
        """Register a pool's health reference for display consumption."""
        self._pool_health[pool_name] = health

    def is_wedged(self, pool_name: str) -> bool:
        """Check whether a specific subpool is wedged.

        A pool is wedged when ``(now - last_progress_at) > WEDGE_THRESHOLD``
        AND ``pending_count > 0``.
        """
        ph = self._pool_health.get(pool_name)
        if ph is None:
            return False
        return ph.is_wedged(poll_interval=WEDGE_THRESHOLD / 2.0)

    def refresh_pool_health(self, *, now: float | None = None) -> None:
        """Update ``status_text`` on each WorkerStats from live PoolHealth.

        Called on each display tick (e.g. from the ``pending_fn`` callback).
        Composes per-subpool pending counts and wedge indicators with Rich
        markup (``[red]...[/red]``).
        """
        ts = now if now is not None else time.time()

        # GENERATE row: compose + regen
        gen_text = format_pool_health_text(
            _GENERATE_SUBPOOLS,
            self._pool_health,
            now=ts,
        )
        self.generate_stats.status_text = gen_text
        self.generate_stats.status_markup = "[red]" in gen_text

        # ENRICH row: single pool
        enrich_text = format_pool_health_text(
            _ENRICH_SUBPOOLS,
            self._pool_health,
            now=ts,
        )
        self.enrich_stats.status_text = enrich_text
        self.enrich_stats.status_markup = "[red]" in enrich_text

        # REVIEW row: review_names + review_docs
        review_text = format_pool_health_text(
            _REVIEW_SUBPOOLS,
            self._pool_health,
            now=ts,
        )
        self.review_stats.status_text = review_text
        self.review_stats.status_markup = "[red]" in review_text


def build_sn_pool_stages(
    *,
    skip_generate: bool = False,
    skip_enrich: bool = False,
    skip_review: bool = False,
) -> list[StageDisplaySpec]:
    """Build the 3 stage specs for the Phase 8 pool display.

    Three rows mapping to the 5 concurrent pools:

    - **GENERATE** — compose + regen pools.
    - **ENRICH** — enrich pool.
    - **REVIEW** — review_names + review_docs pools.
    """
    return [
        StageDisplaySpec(
            name="GENERATE",
            style="bold magenta",
            group="generate",
            stats_attr="generate_stats",
            disabled=skip_generate,
        ),
        StageDisplaySpec(
            name="ENRICH",
            style="bold cyan",
            group="enrich",
            stats_attr="enrich_stats",
            disabled=skip_enrich,
        ),
        StageDisplaySpec(
            name="REVIEW",
            style="bold yellow",
            group="review",
            stats_attr="review_stats",
            disabled=skip_review,
        ),
    ]
