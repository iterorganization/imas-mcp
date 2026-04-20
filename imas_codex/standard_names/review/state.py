"""Shared state for the ``sn review`` pipeline.

Extends :class:`DiscoveryStateBase` with per-phase stats and pipeline
phases for the EXTRACT → ENRICH → REVIEW → PERSIST review pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import PipelinePhase


@dataclass
class StandardNameReviewState(DiscoveryStateBase):
    """Shared state for the sn review pipeline.

    Each phase has its own ``WorkerStats`` and ``PipelinePhase``.
    Workers update stats and shared data as they progress.
    """

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    ids_filter: str | None = None
    domain_filter: str | None = None
    status_filter: str = "drafted"
    unreviewed_only: bool = False
    force_review: bool = False
    skip_audit: bool = False
    name_only: bool = False

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    review_model: str | None = None
    batch_size: int = 15
    neighborhood_k: int = 10
    concurrency: int = 2
    dry_run: bool = False

    # Cross-family reviewer diversity
    secondary_models: list[str] = field(default_factory=list)
    disagreement_threshold: float = 0.2

    # ------------------------------------------------------------------
    # Pipeline data
    # ------------------------------------------------------------------
    all_names: list[dict] = field(default_factory=list)  # Full catalog for audits
    target_names: list[dict] = field(default_factory=list)  # Scoped targets for review
    audit_report: Any = None  # AuditReport from Layer 1
    review_batches: list[dict] = field(default_factory=list)
    review_results: list[dict] = field(default_factory=list)

    # Budget manager (set by CLI)
    budget_manager: Any = None  # ReviewBudgetManager

    # ------------------------------------------------------------------
    # Per-phase stats
    # ------------------------------------------------------------------
    extract_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    review_stats: WorkerStats = field(default_factory=WorkerStats)
    persist_stats: WorkerStats = field(default_factory=WorkerStats)

    # Pipeline phases (init=False → created in __post_init__)
    extract_phase: PipelinePhase = field(init=False)
    enrich_phase: PipelinePhase = field(init=False)
    review_phase: PipelinePhase = field(init=False)
    persist_phase: PipelinePhase = field(init=False)

    # Accumulated results
    stats: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.extract_phase = PipelinePhase("extract")
        self.enrich_phase = PipelinePhase("enrich")
        self.review_phase = PipelinePhase("review")
        self.persist_phase = PipelinePhase("persist")

    # ------------------------------------------------------------------
    # Cost / stopping
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        """Total LLM cost — review is the only LLM phase."""
        return self.review_stats.cost

    def should_stop(self) -> bool:
        """Stop when base conditions met, budget exhausted, or manager drained."""
        if super().should_stop():
            return True
        if self.budget_exhausted:
            return True
        if self.budget_manager and self.budget_manager.exhausted():
            return True
        return False
