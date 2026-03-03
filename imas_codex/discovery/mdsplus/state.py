"""Shared state for parallel tree discovery."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.supervision import PipelinePhase


@dataclass
class TreeDiscoveryState:
    """Shared state for parallel tree discovery workers.

    Supports versioned (machine-description), shot-scoped (dynamic),
    and epoched MDSplus trees.
    """

    facility: str
    ssh_host: str
    tree_name: str
    tree_config: dict[str, Any]

    # Limits
    cost_limit: float = 2.0
    timeout: int = 600
    batch_size: int = 40
    deadline: float | None = None
    force: bool = False

    # Service monitor for health checks
    service_monitor: Any = None

    # Worker stats
    extract_stats: WorkerStats = field(default_factory=WorkerStats)
    units_stats: WorkerStats = field(default_factory=WorkerStats)
    promote_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False

    # Pipeline phases
    extract_phase: PipelinePhase = field(
        default_factory=lambda: PipelinePhase("extract")
    )
    units_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("units"))
    promote_phase: PipelinePhase = field(
        default_factory=lambda: PipelinePhase("promote")
    )

    def should_stop(self) -> bool:
        """Check if discovery should stop."""
        if self.stop_requested:
            return True
        if self.deadline is not None and time.time() >= self.deadline:
            return True
        return False


# Backward-compatible alias
StaticDiscoveryState = TreeDiscoveryState
