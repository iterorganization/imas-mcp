"""Progress display for standard name build pipeline."""

from __future__ import annotations

import logging
from typing import Any

from imas_codex.discovery.base.progress import (
    DataDrivenProgressDisplay,
    ResourceConfig,
    StageDisplaySpec,
    WorkerStats,
    build_resource_section,
)
from imas_codex.discovery.base.supervision import SupervisedWorkerGroup

logger = logging.getLogger(__name__)


def build_sn_stages() -> list[StageDisplaySpec]:
    """Build the stage specs for the SN progress display.

    4 rows: EXTRACT → COMPOSE → REVIEW → FINALIZE
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
            name="REVIEW",
            style="bold yellow",
            group="review",
            stats_attr="review_stats",
            phase_attr="review_phase",
        ),
        StageDisplaySpec(
            name="FINALIZE",
            style="bold green",
            group="finalize",
            stats_attr="finalize_stats",
        ),
    ]


class SNProgressDisplay(DataDrivenProgressDisplay):
    """Rich progress display for the SN build pipeline.

    Shows 3–4 phases: Extract → Compose → [Review] → Finalize
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

    def _build_resources_section(self):
        """Build resource gauges: elapsed, cost."""
        total_cost = 0.0
        if self._engine_state:
            for attr in ("compose_stats", "review_stats"):
                stats: WorkerStats | None = getattr(self._engine_state, attr, None)
                if stats and stats.cost > 0:
                    total_cost += stats.cost

        config = ResourceConfig(
            elapsed=self.elapsed,
            run_cost=total_cost if total_cost > 0 else None,
            cost_limit=self.cost_limit if self.cost_limit > 0 else None,
        )
        return build_resource_section(config, self.gauge_width)

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
            ("REVIEW", "review_stats"),
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
