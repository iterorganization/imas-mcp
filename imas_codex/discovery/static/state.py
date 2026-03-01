"""Shared state for parallel static tree discovery."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.supervision import PipelinePhase


@dataclass
class StaticDiscoveryState:
    """Shared state for parallel static tree discovery workers."""

    facility: str
    ssh_host: str
    tree_name: str
    tree_config: dict[str, Any]

    # Limits
    cost_limit: float = 2.0
    timeout: int = 600
    batch_size: int = 40
    enrich: bool = True
    deadline: float | None = None
    force: bool = False

    # Service monitor for health checks
    service_monitor: Any = None

    # Worker stats
    extract_stats: WorkerStats = field(default_factory=WorkerStats)
    units_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    ingest_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False

    # Pipeline phases
    extract_phase: PipelinePhase = field(
        default_factory=lambda: PipelinePhase("extract")
    )
    units_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("units"))
    enrich_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("enrich"))
    ingest_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("ingest"))

    def should_stop(self) -> bool:
        """Check if discovery should stop."""
        if self.stop_requested:
            return True
        if self.deadline is not None and time.time() >= self.deadline:
            return True
        return False

    @property
    def budget_exhausted(self) -> bool:
        """Check if cost limit reached."""
        return self.enrich_stats.cost >= self.cost_limit

    @property
    def total_cost(self) -> float:
        """Total LLM cost across all workers."""
        return self.enrich_stats.cost
