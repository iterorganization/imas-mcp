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
    # Review rubric target. One of: "names" (4-dim name rubric, /80),
    # "docs" (4-dim docs rubric, /80).
    # ``name_only`` is kept as a back-compat alias and is kept in sync
    # with ``target == "names"``.
    target: str = "names"

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    review_model: str | None = None
    batch_size: int = 15
    neighborhood_k: int = 10
    concurrency: int = 8
    dry_run: bool = False

    # Reviewer list (N >= 1). review_models[0] is canonical.
    review_models: list[str] = field(default_factory=list)
    disagreement_threshold: float = 0.2
    # Deprecated alias — kept only to avoid breaking callers passing
    # ``secondary_models=[]`` as kwarg. Not used by the pipeline anymore.
    secondary_models: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Pipeline data
    # ------------------------------------------------------------------
    all_names: list[dict] = field(default_factory=list)  # Full catalog for audits
    target_names: list[dict] = field(default_factory=list)  # Scoped targets for review
    audit_report: Any = None  # AuditReport from Layer 1
    review_batches: list[dict] = field(default_factory=list)
    review_results: list[dict] = field(default_factory=list)
    # Per-model review records built during REVIEW phase, persisted in PERSIST
    review_records: list[dict] = field(default_factory=list)
    canonical_review_model: str | None = None

    # Budget manager (set by CLI)
    budget_manager: Any = None  # ReviewBudgetManager
    # Optional phase tag for budget attribution (e.g. "review_names", "review_docs").
    budget_phase_tag: str = ""
    # Optional loop-level WorkerStats for live display updates.
    loop_stats: WorkerStats | None = None

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
