"""File discovery state management.

Shared state dataclass for parallel file discovery workers.
Tracks worker statistics, controls stopping conditions, and manages
the SSH host connection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.supervision import PipelinePhase

logger = logging.getLogger(__name__)


@dataclass
class FileDiscoveryState:
    """Shared state for parallel file discovery."""

    facility: str
    ssh_host: str

    # Service monitor for worker gating (set by parallel.py)
    service_monitor: Any = field(default=None, repr=False)

    # Limits
    cost_limit: float = 5.0
    min_score: float = 0.5
    max_paths: int = 100
    focus: str | None = None
    deadline: float | None = None

    # Worker stats
    scan_stats: WorkerStats = field(default_factory=WorkerStats)
    score_stats: WorkerStats = field(default_factory=WorkerStats)
    code_stats: WorkerStats = field(default_factory=WorkerStats)
    docs_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False
    scan_only: bool = False
    score_only: bool = False

    # Pipeline phases
    scan_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("scan"))
    score_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("score"))
    code_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("code"))
    docs_phase: PipelinePhase = field(default_factory=lambda: PipelinePhase("docs"))

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
        total_cost = self.score_stats.cost
        return total_cost >= self.cost_limit

    @property
    def total_cost(self) -> float:
        """Total LLM cost across all workers."""
        return self.score_stats.cost

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time across all workers."""
        return max(
            self.scan_stats.elapsed,
            self.score_stats.elapsed,
            self.code_stats.elapsed,
            self.docs_stats.elapsed,
        )
