"""Shared state for parallel tree discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import PipelinePhase


@dataclass
class TreeDiscoveryState(DiscoveryStateBase):
    """Shared state for parallel tree discovery workers.

    Supports versioned (machine-description), shot-scoped (dynamic),
    and epoched MDSplus trees.
    """

    ssh_host: str = ""
    data_source_name: str = ""
    tree_config: dict[str, Any] = field(default_factory=dict)

    # Limits
    cost_limit: float = 2.0
    timeout: int = 600
    batch_size: int = 40
    force: bool = False

    # Worker stats
    extract_stats: WorkerStats = field(default_factory=WorkerStats)
    units_stats: WorkerStats = field(default_factory=WorkerStats)
    promote_stats: WorkerStats = field(default_factory=WorkerStats)

    # Pipeline phases
    extract_phase: PipelinePhase = field(
        default_factory=lambda: PipelinePhase("extract")
    )
    units_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("units"))
    promote_phase: PipelinePhase = field(
        default_factory=lambda: PipelinePhase("promote")
    )


# Backward-compatible alias
StaticDiscoveryState = TreeDiscoveryState
