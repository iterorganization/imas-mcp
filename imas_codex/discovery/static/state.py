"""Shared state for parallel static tree discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import PipelinePhase


@dataclass
class StaticDiscoveryState(DiscoveryStateBase):
    """Shared state for parallel static tree discovery workers."""

    ssh_host: str = ""
    data_source_name: str = ""
    tree_config: dict[str, Any] = field(default_factory=dict)

    # Limits (cost_limit, deadline inherited from base)
    cost_limit: float = 2.0
    timeout: int = 600
    batch_size: int = 40
    enrich: bool = True
    force: bool = False

    # Worker stats
    extract_stats: WorkerStats = field(default_factory=WorkerStats)
    units_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    ingest_stats: WorkerStats = field(default_factory=WorkerStats)

    # Pipeline phases
    extract_phase: PipelinePhase = field(
        default_factory=lambda: PipelinePhase("extract")
    )
    units_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("units"))
    enrich_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("enrich"))
    ingest_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("ingest"))

    @property
    def total_cost(self) -> float:
        return self.enrich_stats.cost
