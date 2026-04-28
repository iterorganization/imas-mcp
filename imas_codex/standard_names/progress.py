"""Progress display for standard name build pipeline.

Contains:

- :class:`StandardNameProgressDisplay` — 3-stage display for the single-pass
  ``--paths`` pipeline (extract → compose → finalize).
- :class:`SNLoopState` — observable state for loop-mode ``sn run``, consumed
  by :class:`DataDrivenProgressDisplay` via ``StageDisplaySpec`` declarations.
- :func:`build_sn_loop_stages` — stage specs for the 5 loop-mode phases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import (
    DataDrivenProgressDisplay,
    StageDisplaySpec,
    WorkerStats,
)
from imas_codex.discovery.base.supervision import SupervisedWorkerGroup

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

    Five independent worker stats feed the display (one per phase function):

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

    Five rows, one per phase function. Each row has a unique ``group`` so the
    framework tracks per-row state independently:

    - **DRAFT**         — initial compose
    - **REVISE**        — regen with reviewer feedback (disabled when
      ``min_score is None`` since the loop forces ``skip_regen``)
    - **DESCRIBE**      — enrichment
    - **REVIEW NAMES**  — name-side review
    - **REVIEW DOCS**   — docs-side review
    """
    revise_disabled = skip_generate or min_score is None
    return [
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
            name="REVIEW NAMES",
            style="bold yellow",
            group="review_names",
            stats_attr="review_names_stats",
            disabled=skip_review,
        ),
        StageDisplaySpec(
            name="REVIEW DOCS",
            style="yellow",
            group="review_docs",
            stats_attr="review_docs_stats",
            disabled=skip_review,
        ),
    ]
