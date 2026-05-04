"""SN build pipeline state.

Extends :class:`DiscoveryStateBase` with per-phase stats and pipeline
phases for the EXTRACT → COMPOSE → VALIDATE → CONSOLIDATE → PERSIST
standard-name pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import PipelinePhase


@dataclass
class StandardNameBuildState(DiscoveryStateBase):
    """Shared state for the standard-name build pipeline.

    Each phase has its own ``WorkerStats`` and ``PipelinePhase``.
    Workers update stats and shared data as they progress.

    State fields are all past tense, named after the phase that writes them:
    ``extracted``, ``composed``, ``validated``, ``consolidated``.

    The ``facility`` field is inherited from ``DiscoveryStateBase``.
    For DD source pipelines use ``facility="dd"``; for signal source
    pipelines pass the real facility identifier.
    """

    # Build configuration
    source: str = "dd"  # "dd" or "signals"
    ids_filter: str | None = None  # For DD source: restrict to a single IDS
    domain_filter: str | None = None  # Physics domain filter
    facility_filter: str | None = None  # For signals source: facility to query
    paths_list: list[str] | None = None  # Explicit DD paths (bypass query+classifier)
    dry_run: bool = False
    force: bool = False  # Bypass source-level skip
    # Regeneration mode. When True, extraction targets existing reviewed
    # StandardNames whose ``reviewer_score`` is below ``min_score`` and
    # re-composes them with their prior reviewer critique injected. When
    # False, the extract worker runs the broad DD path / signals source.
    regen: bool = False
    # Reviewer-score threshold for regen-mode selection. Names with
    # ``reviewer_score < min_score`` are eligible. None in regen mode is a
    # no-op (nothing selected); paired with the CLI ``--min-score`` flag.
    min_score: float | None = None
    # Run provenance — stamped on every StandardName touched by this run
    # via write_run_provenance. Set by the CLI / loop; default placeholders
    # allow standalone pipeline invocations to still produce a node.
    run_id: str | None = None
    limit: int | None = None  # Cap on paths to process
    from_model: str | None = None  # Regenerate names produced by this model (substring)

    # Name-only batching mode (Workstream 2a): group by (physics_domain × unit)
    # in larger bins and skip L2 exemplar / L4 theme / IDS-prefetch enrichment.
    # Produces named candidates faster during bootstrap; downstream review
    # / enrichment passes restore the deeper context.
    name_only: bool = False
    name_only_batch_size: int = 50

    # Budget manager (lease-style)
    budget_manager: Any = None  # BudgetManager from .budget
    # Optional phase tag for budget attribution (e.g. "generate", "regen").
    budget_phase_tag: str = ""
    # Optional loop-level WorkerStats for live display updates.
    # Workers increment ``loop_stats.processed`` and push to
    # ``loop_stats.stream_queue`` after each LLM charge.
    loop_stats: WorkerStats | None = None
    # Optional loop-level WorkerStats specifically for the EXTRACT phase so
    # the loop display can show extraction progress in real time. The
    # extract_worker mirrors progress updates here when set.
    loop_extract_stats: WorkerStats | None = None

    # Model overrides (None = use defaults from pyproject.toml)
    compose_model: str | None = None  # Override for compose step (default: reasoning)

    # COCOS provenance (resolved once at extract start)
    dd_version: str | None = None
    cocos_version: int | None = None
    cocos_params: dict | None = None

    # In-memory pipeline data (extract → compose → validate → consolidate)
    extracted: list[Any] = field(default_factory=list)  # ExtractionBatch objects
    composed: list[dict[str, Any]] = field(default_factory=list)
    validated: list[dict[str, Any]] = field(default_factory=list)
    consolidated: list[dict[str, Any]] = field(default_factory=list)

    # Accumulated results
    stats: dict[str, Any] = field(default_factory=dict)

    # Quality lever tracking (L3, L6, L7)
    grammar_retries: int = 0
    grammar_retries_succeeded: int = 0
    opus_revisions_attempted: int = 0
    opus_revisions_accepted: int = 0
    audits_run: int = 0
    audits_failed: int = 0

    # Per-phase progress (observed by display)
    extract_stats: WorkerStats = field(default_factory=WorkerStats)
    compose_stats: WorkerStats = field(default_factory=WorkerStats)
    validate_stats: WorkerStats = field(default_factory=WorkerStats)
    consolidate_stats: WorkerStats = field(default_factory=WorkerStats)
    persist_stats: WorkerStats = field(default_factory=WorkerStats)
    finalize_stats: WorkerStats = field(default_factory=WorkerStats)

    # Pipeline phases
    extract_phase: PipelinePhase = field(init=False)
    compose_phase: PipelinePhase = field(init=False)
    validate_phase: PipelinePhase = field(init=False)
    consolidate_phase: PipelinePhase = field(init=False)
    persist_phase: PipelinePhase = field(init=False)

    def __post_init__(self) -> None:
        self.extract_phase = PipelinePhase("extract")
        self.compose_phase = PipelinePhase("compose")
        self.validate_phase = PipelinePhase("validate")
        self.consolidate_phase = PipelinePhase("consolidate")
        self.persist_phase = PipelinePhase("persist")

    # ------------------------------------------------------------------
    # Cost / stopping
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        """Total LLM cost — compose is the only LLM phase."""
        return self.compose_stats.cost

    def should_stop(self) -> bool:
        """Stop when base conditions met or budget exhausted."""
        if super().should_stop():
            return True
        if self.budget_exhausted:
            return True
        if self.budget_manager and self.budget_manager.exhausted():
            return True
        return False

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def is_regen_mode(self) -> bool:
        """Return True when extraction should target reviewed names below ``min_score``.

        Triggered by ``--min-score F`` (or by the loop's regen phase). An
        explicit ``paths_list`` or ``from_model`` short-circuits to those
        narrower sources instead. ``--domain`` / ``--limit`` act as
        narrowing filters within regen mode, not overrides.
        """
        if not self.regen:
            return False
        if self.paths_list:
            return False
        if self.from_model:
            return False
        return self.min_score is not None
