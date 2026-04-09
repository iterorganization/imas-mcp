"""SN build pipeline state.

Extends :class:`DiscoveryStateBase` with per-phase stats and pipeline
phases for the EXTRACT → COMPOSE → VALIDATE → PERSIST standard-name
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import PipelinePhase


@dataclass
class SNBuildState(DiscoveryStateBase):
    """Shared state for the standard-name build pipeline.

    Each phase has its own ``WorkerStats`` and ``PipelinePhase``.
    Workers update stats and shared data as they progress.

    State fields are all past tense, named after the phase that writes them:
    ``extracted``, ``composed``, ``reviewed``, ``validated``.

    The ``facility`` field is inherited from ``DiscoveryStateBase``.
    For DD source pipelines use ``facility="dd"``; for signal source
    pipelines pass the real facility identifier.
    """

    # Build configuration
    source: str = "dd"  # "dd" or "signals"
    ids_filter: str | None = None  # For DD source: restrict to a single IDS
    domain_filter: str | None = None  # Physics domain filter
    facility_filter: str | None = None  # For signals source: facility to query
    dry_run: bool = False
    force: bool = False  # Bypass source-level skip
    limit: int | None = None  # Cap on paths to process

    # Review configuration
    skip_review: bool = False
    review_model: str | None = None

    # In-memory pipeline data (extract → compose → review → validate)
    extracted: list[Any] = field(default_factory=list)  # ExtractionBatch objects
    composed: list[dict[str, Any]] = field(default_factory=list)
    reviewed: list[dict[str, Any]] = field(default_factory=list)
    validated: list[dict[str, Any]] = field(default_factory=list)

    # Accumulated results
    stats: dict[str, Any] = field(default_factory=dict)

    # Per-phase progress (observed by display)
    extract_stats: WorkerStats = field(default_factory=WorkerStats)
    compose_stats: WorkerStats = field(default_factory=WorkerStats)
    review_stats: WorkerStats = field(default_factory=WorkerStats)
    validate_stats: WorkerStats = field(default_factory=WorkerStats)
    persist_stats: WorkerStats = field(default_factory=WorkerStats)

    # Pipeline phases
    extract_phase: PipelinePhase = field(init=False)
    compose_phase: PipelinePhase = field(init=False)
    review_phase: PipelinePhase = field(init=False)
    validate_phase: PipelinePhase = field(init=False)
    persist_phase: PipelinePhase = field(init=False)

    def __post_init__(self) -> None:
        self.extract_phase = PipelinePhase("extract")
        self.compose_phase = PipelinePhase("compose")
        self.review_phase = PipelinePhase("review")
        self.validate_phase = PipelinePhase("validate")
        self.persist_phase = PipelinePhase("persist")

    # ------------------------------------------------------------------
    # Cost / stopping
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        """Total LLM cost — compose and review phases call the LLM."""
        return self.compose_stats.cost + self.review_stats.cost

    def should_stop(self) -> bool:
        """Stop when base conditions met or budget exhausted."""
        if super().should_stop():
            return True
        if self.budget_exhausted:
            return True
        return False
