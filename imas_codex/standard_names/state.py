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
    # Scope extraction to existing SNs in validation_status='needs_revision'
    # (set by --regen-only or by the rotator's regen phase). When False, the
    # extract worker runs the broad DD path / signals source.
    regen_only: bool = False
    # Inject prior reviewer critique (tier, score, comments) into the compose
    # prompt so the LLM can directly address it on regeneration. Always-on by
    # default — the CLI exposes ``--no-review-feedback`` to disable.
    inject_review_feedback: bool = True
    limit: int | None = None  # Cap on paths to process
    from_model: str | None = None  # Regenerate names produced by this model (substring)

    # Name-only batching mode (Workstream 2a): group by (physics_domain × unit)
    # in larger bins and skip L2 exemplar / L4 theme / IDS-prefetch enrichment.
    # Produces named candidates faster during bootstrap; downstream review
    # / enrichment passes restore the deeper context.
    name_only: bool = False
    name_only_batch_size: int = 50

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
        return False

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def is_regen_only_mode(self) -> bool:
        """Return True when extraction should be scoped to needs_revision SNs only.

        Triggered by ``--regen-only`` (or by the rotator's regen phase) when no
        narrower source-selection flag overrides the default extraction scope.
        ``--paths`` (``paths_list``) and ``--from-model`` both narrow to an
        explicit source set, so they short-circuit regen-only mode.
        ``--domain`` / ``--ids`` / ``--limit`` are *narrowing* filters
        within regen-only mode, not overrides.
        """
        if not self.regen_only:
            return False
        if self.paths_list:
            return False
        if self.from_model:
            return False
        return True
