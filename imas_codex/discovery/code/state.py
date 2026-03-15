"""File discovery state management.

Shared state dataclass for parallel file discovery workers.
Tracks worker statistics, controls stopping conditions, and manages
the SSH host connection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import PipelinePhase

logger = logging.getLogger(__name__)


@dataclass
class FileDiscoveryState(DiscoveryStateBase):
    """Shared state for parallel file discovery."""

    ssh_host: str = ""

    # Limits
    cost_limit: float = 5.0
    min_score: float = 0.9
    max_paths: int = 100
    focus: str | None = None

    # Worker stats
    scan_stats: WorkerStats = field(default_factory=WorkerStats)
    triage_stats: WorkerStats = field(default_factory=WorkerStats)
    score_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    code_stats: WorkerStats = field(default_factory=WorkerStats)
    link_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    scan_only: bool = False
    score_only: bool = False

    # Pipeline phases
    scan_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("scan"))
    triage_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("triage"))
    score_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("score"))
    enrich_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("enrich"))
    code_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("code"))
    link_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("link"))

    @property
    def total_cost(self) -> float:
        """Total LLM cost across all workers."""
        return self.triage_stats.cost + self.score_stats.cost

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time across all workers."""
        return max(
            self.scan_stats.elapsed,
            self.score_stats.elapsed,
            self.code_stats.elapsed,
        )
