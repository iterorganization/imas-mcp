"""Progress display for standard name build pipeline."""

from __future__ import annotations

import logging
from typing import Any

from rich.text import Text

from imas_codex.discovery.base.progress import (
    BaseProgressDisplay,
    PipelineRowConfig,
    ResourceConfig,
    WorkerStats,
    build_pipeline_section,
    build_resource_section,
)
from imas_codex.discovery.base.supervision import SupervisedWorkerGroup

logger = logging.getLogger(__name__)


class SNProgressDisplay(BaseProgressDisplay):
    """Rich progress display for the SN build pipeline.

    Shows six phases: Extract → Compose → Review → Validate → Consolidate → Persist
    with per-phase progress bars, rates, and cost tracking.
    """

    def __init__(
        self,
        source: str = "dd",
        *,
        console: Any | None = None,
        cost_limit: float = 5.0,
        mode_label: str | None = None,
    ):
        super().__init__(
            facility="sn",
            cost_limit=cost_limit,
            console=console,
            title_suffix="Standard Name Build",
        )
        self.source = source
        self._mode_label = mode_label
        self._engine_state: Any | None = None

    def set_engine_state(self, state: Any) -> None:
        """Connect display to the live engine state."""
        self._engine_state = state

    def on_worker_status(self, group: SupervisedWorkerGroup) -> None:
        """Callback for worker status updates."""
        self.update_worker_status(group)

    def _header_mode_label(self) -> str | None:
        return self._mode_label

    def _get_stage_stats(self, attr: str) -> WorkerStats | None:
        """Safely get stats from engine state."""
        if self._engine_state is None:
            return None
        return getattr(self._engine_state, attr, None)

    def _build_pipeline_section(self) -> Text:
        """Build pipeline section showing Extract → Compose → Review → Validate → Consolidate → Persist."""
        stages = [
            ("EXTRACT", "bold blue", "extract", "extract_stats"),
            ("COMPOSE", "bold magenta", "compose", "compose_stats"),
            ("REVIEW", "bold yellow", "review", "review_stats"),
            ("VALIDATE", "bold green", "validate", "validate_stats"),
            ("CONSOLIDATE", "bold white", "consolidate", "consolidate_stats"),
            ("PERSIST", "bold cyan", "persist", "persist_stats"),
        ]

        rows: list[PipelineRowConfig] = []
        for name, style, group, stats_attr in stages:
            stats = self._get_stage_stats(stats_attr)
            count, ann = self._count_group_workers(group)
            completed = stats.processed if stats else 0
            total = stats.total if stats and stats.total > 0 else max(completed, 1)
            complete = self._worker_complete(group)
            running = self._worker_running(group)
            waiting = self._worker_waiting(group)

            primary_text = stats.status_text if stats else ""
            if stats and stats._current_stream_item:
                si = stats._current_stream_item
                primary_text = si.get("primary_text", primary_text)

            rows.append(
                PipelineRowConfig(
                    name=name,
                    style=style,
                    completed=completed,
                    total=total,
                    rate=stats.ema_rate if stats else None,
                    cost=stats.cost if stats and stats.cost > 0 else None,
                    worker_count=count,
                    worker_annotation=ann,
                    primary_text=primary_text,
                    is_complete=complete,
                    is_processing=running and not complete,
                    processing_label="waiting..." if waiting else "processing...",
                )
            )
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        """Build resource gauges: elapsed, cost, ETA."""
        total_cost = 0.0
        stats_attrs = [
            "extract_stats",
            "compose_stats",
            "review_stats",
            "validate_stats",
            "consolidate_stats",
            "persist_stats",
        ]

        for attr in stats_attrs:
            stats = self._get_stage_stats(attr)
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
            ("VALIDATE", "validate_stats"),
            ("CONSOLIDATE", "consolidate_stats"),
            ("PERSIST", "persist_stats"),
        ]:
            stats = self._get_stage_stats(attr)
            if stats and stats.processed > 0:
                parts = [f"{label}: {stats.processed:,}"]
                if stats.cost > 0:
                    parts.append(f"${stats.cost:.2f}")
                lines.append("  ".join(parts))

        if lines:
            self.console.print()
            for line in lines:
                self.console.print(f"  {line}")
