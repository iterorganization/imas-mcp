"""Enrich pipeline state.

Extends :class:`DiscoveryStateBase` with per-phase stats and pipeline
phases for the EXTRACT → CONTEXTUALISE → DOCUMENT → VALIDATE → PERSIST
standard-name enrichment pipeline.

The enrich pipeline runs *after* the generate pipeline completes.
It takes ``review_status='named'`` StandardName nodes and enriches
them with documentation, description, and cross-reference links,
transitioning them through ``enriched`` → ``reviewable``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import PipelinePhase


@dataclass
class StandardNameEnrichState(DiscoveryStateBase):
    """Shared state for the standard-name enrich pipeline.

    Each phase has its own ``WorkerStats`` and ``PipelinePhase``.
    Workers update stats and shared data as they progress.

    State fields mirror the generate pipeline's ``StandardNameBuildState``
    shape, with phases renamed for the enrich DAG.
    """

    # --- Filters / configuration ---
    domain: list[str] | None = None
    ids: str | None = None
    limit: int | None = None
    force: bool = False
    dry_run: bool = False
    batch_size: int = 8
    model: str | None = None
    status_filter: list[str] | None = None

    # --- Batch / accumulator data ---
    batches: list[Any] = field(default_factory=list)
    enriched: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)

    # --- Accumulated cost / token counters ---
    cost: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    # --- Per-stage counters ---
    stats: dict[str, Any] = field(default_factory=dict)

    # --- Per-phase progress (observed by display) ---
    extract_stats: WorkerStats = field(default_factory=WorkerStats)
    contextualise_stats: WorkerStats = field(default_factory=WorkerStats)
    document_stats: WorkerStats = field(default_factory=WorkerStats)
    validate_stats: WorkerStats = field(default_factory=WorkerStats)
    persist_stats: WorkerStats = field(default_factory=WorkerStats)

    # --- Pipeline phases ---
    extract_phase: PipelinePhase = field(init=False)
    contextualise_phase: PipelinePhase = field(init=False)
    document_phase: PipelinePhase = field(init=False)
    validate_phase: PipelinePhase = field(init=False)
    persist_phase: PipelinePhase = field(init=False)

    def __post_init__(self) -> None:
        self.extract_phase = PipelinePhase("extract")
        self.contextualise_phase = PipelinePhase("contextualise")
        self.document_phase = PipelinePhase("document")
        self.validate_phase = PipelinePhase("validate")
        self.persist_phase = PipelinePhase("persist")

    # ------------------------------------------------------------------
    # Cost / stopping
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        """Total LLM cost — document is the primary LLM phase."""
        return self.cost

    def should_stop(self) -> bool:
        """Stop when base conditions met or budget exhausted."""
        if super().should_stop():
            return True
        if self.budget_exhausted:
            return True
        return False
