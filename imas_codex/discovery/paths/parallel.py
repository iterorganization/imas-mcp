"""
Parallel discovery engine with true concurrent scan and score workers.

Architecture:
- Two independent async workers: Scanner and Scorer
- Graph is the coordination mechanism (no locks needed)
- Atomic status transitions prevent race conditions:
  - discovered → listing → listed (Scanner worker)
  - listed → scoring → scored (Scorer worker)
- Workers continuously poll graph for work
- Cost-based termination for Scorer
- Orphan recovery: paths stuck in transient states >10 min are reset

Key insight: The graph acts as a thread-safe work queue. Each worker
claims work by atomically updating status, processes it, then marks complete.
No two workers can claim the same path.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.supervision import (
    OrphanRecoverySpec,
    PipelinePhase,
    SupervisedWorkerGroup,
    make_orphan_recovery_tick,
    run_supervised_loop,
    supervised_worker,
)
from imas_codex.graph.models import PathStatus, TerminalReason

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryState:
    """Shared state for parallel discovery."""

    facility: str
    cost_limit: float
    path_limit: int | None = None
    focus: str | None = None
    threshold: float = 0.7
    root_filter: list[str] | None = None  # Restrict work to these roots
    auto_enrich_threshold: float | None = None  # Also enrich paths scoring >= this
    deadline: float | None = None  # Unix timestamp when discovery should stop

    # Worker stats
    scan_stats: WorkerStats = field(default_factory=WorkerStats)
    expand_stats: WorkerStats = field(default_factory=WorkerStats)
    score_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    refine_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    stop_requested: bool = False

    # Pipeline phases (initialized in __post_init__)
    scan_phase: PipelinePhase = field(init=False)
    expand_phase: PipelinePhase = field(init=False)
    score_phase: PipelinePhase = field(init=False)
    enrich_phase: PipelinePhase = field(init=False)
    refine_phase: PipelinePhase = field(init=False)

    def __post_init__(self) -> None:
        self.scan_phase = PipelinePhase("scan")
        self.expand_phase = PipelinePhase("expand")
        self.score_phase = PipelinePhase("score")
        self.enrich_phase = PipelinePhase("enrich")
        self.refine_phase = PipelinePhase("refine")

    # Backwards-compat idle count properties for progress display
    @property
    def scan_idle_count(self) -> int:
        return self.scan_phase.idle_count

    @scan_idle_count.setter
    def scan_idle_count(self, value: int) -> None:
        if value >= 3:
            self.scan_phase._idle_count = value
        else:
            self.scan_phase._idle_count = value

    @property
    def expand_idle_count(self) -> int:
        return self.expand_phase.idle_count

    @expand_idle_count.setter
    def expand_idle_count(self, value: int) -> None:
        self.expand_phase._idle_count = value

    @property
    def score_idle_count(self) -> int:
        return self.score_phase.idle_count

    @score_idle_count.setter
    def score_idle_count(self, value: int) -> None:
        self.score_phase._idle_count = value

    @property
    def enrich_idle_count(self) -> int:
        return self.enrich_phase.idle_count

    @enrich_idle_count.setter
    def enrich_idle_count(self, value: int) -> None:
        self.enrich_phase._idle_count = value

    @property
    def refine_idle_count(self) -> int:
        return self.refine_phase.idle_count

    @refine_idle_count.setter
    def refine_idle_count(self, value: int) -> None:
        self.refine_phase._idle_count = value

    # SSH retry tracking for exponential backoff
    ssh_retry_count: int = 0
    max_ssh_retries: int = 5
    ssh_error_message: str | None = None

    # Session tracking for --limit
    initial_terminal_count: int | None = None

    @property
    def total_cost(self) -> float:
        return self.score_stats.cost + self.refine_stats.cost

    @property
    def total_processed(self) -> int:
        return self.scan_stats.processed + self.score_stats.processed

    @property
    def terminal_count(self) -> int:
        """Count of paths in terminal states (scored, not pending expand/enrich).

        For --limit purposes, we only count paths that have completed their
        pipeline: scored and not awaiting expansion or enrichment.
        """
        from imas_codex.graph import GraphClient

        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
                WHERE p.status = $scored
                  AND (p.should_expand = false OR p.expanded_at IS NOT NULL)
                  AND (p.should_enrich = false OR p.is_enriched = true)
                RETURN count(p) AS terminal_count
                """,
                facility=self.facility,
                scored=PathStatus.scored.value,
            )
            return result[0]["terminal_count"] if result else 0

    @property
    def session_terminal_count(self) -> int:
        """Count of terminal paths created in this session."""
        if self.initial_terminal_count is None:
            return 0
        return max(0, self.terminal_count - self.initial_terminal_count)

    @property
    def budget_exhausted(self) -> bool:
        return self.total_cost >= self.cost_limit

    @property
    def deadline_expired(self) -> bool:
        """Check if the deadline has been reached."""
        if self.deadline is None:
            return False
        import time

        return time.time() >= self.deadline

    @property
    def path_limit_reached(self) -> bool:
        """Check if path limit reached using session terminal count.

        Uses paths completed in THIS SESSION, not cumulative graph total.
        E.g., with 28 existing paths and --limit 30, we process 30 more.
        """
        if self.path_limit is None:
            return False
        return self.session_terminal_count >= self.path_limit

    def should_stop(self) -> bool:
        """Check if discovery should terminate.

        Uses PipelinePhase.done for each phase, which combines idle
        detection with graph-level pending work checks via
        has_pending_work().  This replaces the old idle-counter
        approach that was prone to race conditions.
        """
        if self.stop_requested:
            return True
        if self.budget_exhausted:
            return True
        if self.path_limit_reached:
            return True
        if self.deadline_expired:
            return True
        # All phases must be done (idle + no graph work)
        all_done = (
            self.scan_phase.done
            and self.expand_phase.done
            and self.score_phase.done
            and self.enrich_phase.done
            and self.refine_phase.done
        )
        if all_done:
            # Final confirmation: no pending work at all
            if has_pending_work(self.facility):
                # Graph has work — reset all phases so workers re-poll
                self.scan_phase.reset()
                self.expand_phase.reset()
                self.score_phase.reset()
                self.enrich_phase.reset()
                self.refine_phase.reset()
                return False
            logger.debug(
                f"Terminating: all phases done, no pending work "
                f"(scan={self.scan_stats.processed}, score={self.score_stats.processed})"
            )
            return True
        return False


# Claim timeout for orphan recovery (same as wiki/signals)
CLAIM_TIMEOUT_SECONDS = 600  # 10 minutes


def has_pending_work(facility: str) -> bool:
    """Check if there's pending or in-progress work in the graph.

    Returns True if any of:
    - Discovered paths awaiting first scan (including actively claimed)
    - Scanned paths awaiting scoring (including actively claimed)
    - Scored paths with should_expand=true that haven't been expanded yet
    - Scored paths with should_enrich=true that haven't been enriched yet
    - Enriched paths that haven't been refined yet

    Note: Unlike claim functions, this does NOT filter out claimed paths.
    Claimed paths represent active work in progress and must count as
    pending to prevent premature termination while workers are mid-task.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WITH p,
                 CASE WHEN p.status = $discovered AND p.score IS NULL
                      THEN 'discovered' ELSE null END AS disc,
                 CASE WHEN p.status = $scanned AND p.score IS NULL
                      THEN 'scanned' ELSE null END AS scn,
                 CASE WHEN p.status = $scored AND p.should_expand = true
                      AND p.expanded_at IS NULL
                      THEN 'expand' ELSE null END AS exp,
                 CASE WHEN p.status = $scored AND p.should_enrich = true
                      AND (p.is_enriched IS NULL OR p.is_enriched = false)
                      THEN 'enrich' ELSE null END AS enr,
                 CASE WHEN p.is_enriched = true AND p.refined_at IS NULL
                      THEN 'refine' ELSE null END AS rsc
            WHERE disc IS NOT NULL OR scn IS NOT NULL OR exp IS NOT NULL
                  OR enr IS NOT NULL OR rsc IS NOT NULL
            RETURN count(p) AS pending,
                   count(disc) AS pending_discovered,
                   count(scn) AS pending_scanned,
                   count(exp) AS pending_expand,
                   count(enr) AS pending_enrich,
                   count(rsc) AS pending_refine
            """,
            facility=facility,
            discovered=PathStatus.discovered.value,
            scanned=PathStatus.scanned.value,
            scored=PathStatus.scored.value,
        )
        if result:
            pending = result[0]["pending"]
            if pending > 0:
                logger.debug(
                    f"Pending work: discovered={result[0]['pending_discovered']}, "
                    f"scanned={result[0]['pending_scanned']}, "
                    f"expand={result[0]['pending_expand']}, "
                    f"enrich={result[0]['pending_enrich']}, "
                    f"refine={result[0]['pending_refine']}"
                )
            return pending > 0
        return False


# ============================================================================
# Orphan Recovery (timeout-based, safe for parallel instances)
# ============================================================================


def reset_orphaned_claims(
    facility: str, *, silent: bool = False, force: bool = False
) -> int:
    """Release stale claims older than CLAIM_TIMEOUT_SECONDS.

    Uses timeout-based recovery so multiple CLI instances can run
    concurrently on the same facility without wiping each other's claims.
    Only claims older than the timeout are considered orphaned.

    Args:
        facility: Facility identifier
        silent: If True, suppress logging (caller will log)
        force: Clear ALL claims regardless of age (use at startup)

    Returns:
        Number of paths with claims cleared
    """
    from imas_codex.discovery.base.claims import reset_stale_claims

    return reset_stale_claims(
        "FacilityPath",
        facility,
        timeout_seconds=0 if force else CLAIM_TIMEOUT_SECONDS,
        silent=silent,
    )


# ============================================================================
# Graph-based work claiming (atomic status transitions)
# ============================================================================


def claim_paths_for_scanning(
    facility: str, limit: int = 50, root_filter: list[str] | None = None
) -> list[dict[str, Any]]:
    """Atomically claim discovered paths for initial scanning.

    Claims only unscored discovered paths (first scan, enumerate only).
    Expansion is handled by separate expand_worker.

    Uses claimed_at timestamp for coordination (no status change during claim).

    Args:
        facility: Facility ID
        limit: Maximum paths to claim
        root_filter: If set, only claim paths under these roots
    """
    from imas_codex.graph import GraphClient

    # Build root filter clause
    root_clause = ""
    if root_filter:
        root_clause = "AND any(root IN $root_filter WHERE p.path STARTS WITH root)"

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        # Claim unscored discovered paths (breadth-first by depth)
        result = gc.query(
            f"""
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.status = $discovered AND p.score IS NULL
              AND (p.claimed_at IS NULL OR p.claimed_at < datetime() - duration($cutoff))
            {root_clause}
            WITH p ORDER BY p.depth ASC, p.path ASC LIMIT $limit
            SET p.claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.depth AS depth, false AS is_expanding
            """,
            facility=facility,
            limit=limit,
            discovered=PathStatus.discovered.value,
            cutoff=cutoff,
            root_filter=root_filter or [],
        )
        return list(result)


def claim_paths_for_expanding(
    facility: str, limit: int = 50, root_filter: list[str] | None = None
) -> list[dict[str, Any]]:
    """Atomically claim scored paths for expansion scanning.

    Claims paths with should_expand=true that haven't been expanded yet.
    These are scored high-value directories that need child enumeration.

    Uses claimed_at timestamp for coordination (no status change during claim).

    Args:
        facility: Facility ID
        limit: Maximum paths to claim
        root_filter: If set, only claim paths under these roots
    """
    from imas_codex.graph import GraphClient

    # Build root filter clause
    root_clause = ""
    if root_filter:
        root_clause = "AND any(root IN $root_filter WHERE p.path STARTS WITH root)"

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        # Claim expansion paths (score-descending for valuable first)
        result = gc.query(
            f"""
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.status = $scored
              AND p.should_expand = true
              AND p.expanded_at IS NULL
              AND (p.claimed_at IS NULL OR p.claimed_at < datetime() - duration($cutoff))
            {root_clause}
            WITH p ORDER BY p.score DESC, p.depth ASC LIMIT $limit
            SET p.claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.depth AS depth, true AS is_expanding
            """,
            facility=facility,
            limit=limit,
            scored=PathStatus.scored.value,
            cutoff=cutoff,
            root_filter=root_filter or [],
        )
        return list(result)


def claim_paths_for_scoring(
    facility: str, limit: int = 25, root_filter: list[str] | None = None
) -> list[dict[str, Any]]:
    """Atomically claim scanned paths for scoring.

    Uses claimed_at timestamp for coordination (no status change during claim).
    Returns paths that this worker now owns.

    Args:
        facility: Facility ID
        limit: Maximum paths to claim
        root_filter: If set, only claim paths under these roots
    """
    from imas_codex.graph import GraphClient

    # Build root filter clause
    root_clause = ""
    if root_filter:
        root_clause = "AND any(root IN $root_filter WHERE p.path STARTS WITH root)"

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.status = $scanned AND p.score IS NULL
              AND (p.claimed_at IS NULL OR p.claimed_at < datetime() - duration($cutoff))
            {root_clause}
            WITH p ORDER BY p.depth ASC, p.path ASC LIMIT $limit
            SET p.claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.depth AS depth,
                   p.total_files AS total_files, p.total_dirs AS total_dirs,
                   p.file_type_counts AS file_type_counts,
                   p.has_readme AS has_readme, p.has_makefile AS has_makefile,
                   p.has_git AS has_git, p.git_remote_url AS git_remote_url,
                   p.vcs_type AS vcs_type, p.vcs_remote_url AS vcs_remote_url,
                   p.vcs_remote_accessible AS vcs_remote_accessible,
                   p.patterns_detected AS patterns_detected,
                   p.child_names AS child_names,
                   p.tree_context AS tree_context,
                   p.numeric_dir_ratio AS numeric_dir_ratio
            """,
            facility=facility,
            limit=limit,
            scanned=PathStatus.scanned.value,
            cutoff=cutoff,
            root_filter=root_filter or [],
        )
        return list(result)


async def mark_scan_complete(
    facility: str,
    scan_results: list[tuple[str, dict, list[dict], str | None, bool]],
    excluded: list[tuple[str, str, str]] | None = None,
) -> dict[str, int]:
    """Mark scanned paths complete and conditionally create children.

    Transition: listing → listed (first scan) or scored (expansion scan)

    Args:
        facility: Facility ID
        scan_results: List of (path, stats_dict, child_dirs, error, is_expanding) tuples.
                      child_dirs is list of {path, is_symlink, realpath, device_inode} dicts.
        excluded: Optional list of (path, parent_path, reason) for excluded dirs
    """
    from imas_codex.discovery.paths.frontier import persist_scan_results

    return await persist_scan_results(facility, scan_results, excluded=excluded)


def mark_score_complete(
    facility: str,
    score_data: list[dict[str, Any]],
) -> int:
    """Apply structural expansion overrides and persist scored paths.

    Transition: scoring → scored
    """
    from imas_codex.discovery.paths.frontier import (
        apply_expansion_overrides,
        mark_paths_scored,
    )

    apply_expansion_overrides(facility, score_data)
    return mark_paths_scored(facility, score_data)


def claim_paths_for_enriching(
    facility: str,
    limit: int = 25,
    root_filter: list[str] | None = None,
    auto_enrich_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Atomically claim scored paths for enrichment.

    Claims paths where:
    - status = 'scored'
    - should_enrich = true OR score >= auto_enrich_threshold
    - is_enriched is null or false

    Uses claimed_at timestamp for orphan recovery.
    Returns paths that this worker now owns.

    Args:
        facility: Facility ID
        limit: Maximum paths to claim
        root_filter: If set, only claim paths under these roots
        auto_enrich_threshold: If set, also claim paths with score >= threshold
    """
    from imas_codex.graph import GraphClient

    # Build root filter clause
    root_clause = ""
    if root_filter:
        root_clause = "AND any(root IN $root_filter WHERE p.path STARTS WITH root)"

    # Build enrich condition: should_enrich=true OR (threshold and score >= threshold)
    if auto_enrich_threshold is not None:
        enrich_clause = "(p.should_enrich = true OR p.score >= $auto_enrich_threshold)"
    else:
        enrich_clause = "p.should_enrich = true"

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.status = $scored
              AND {enrich_clause}
              AND (p.is_enriched IS NULL OR p.is_enriched = false)
              AND (p.claimed_at IS NULL OR p.claimed_at < datetime() - duration($cutoff))
            {root_clause}
            WITH p ORDER BY p.score DESC, p.depth ASC LIMIT $limit
            SET p.claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.depth AS depth, p.score AS score,
                   p.path_purpose AS path_purpose
            """,
            facility=facility,
            limit=limit,
            scored=PathStatus.scored.value,
            cutoff=cutoff,
            root_filter=root_filter or [],
            auto_enrich_threshold=auto_enrich_threshold,
        )
        return list(result)


def claim_paths_for_refining(
    facility: str, limit: int = 10, root_filter: list[str] | None = None
) -> list[dict[str, Any]]:
    """Atomically claim enriched paths for rescoring.

    Claims paths where:
    - is_enriched = true
    - score >= 0.7 (aligned with downstream code discovery threshold)
    - refined_at is null

    Rescoring uses enrichment data (total_bytes, total_lines, language_breakdown)
    to refine the score.

    Returns paths that this worker now owns.

    Args:
        facility: Facility ID
        limit: Maximum paths to claim
        root_filter: If set, only claim paths under these roots
    """
    from imas_codex.graph import GraphClient

    # Build root filter clause
    root_clause = ""
    if root_filter:
        root_clause = "AND any(root IN $root_filter WHERE p.path STARTS WITH root)"

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.is_enriched = true
              AND p.score >= 0.7
              AND p.refined_at IS NULL
              AND (p.claimed_at IS NULL OR p.claimed_at < datetime() - duration($cutoff))
            {root_clause}
            WITH p ORDER BY p.score DESC LIMIT $limit
            SET p.claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.depth AS depth,
                   p.score AS score,
                   p.score_modeling_code AS score_modeling_code,
                   p.score_analysis_code AS score_analysis_code,
                   p.score_operations_code AS score_operations_code,
                   p.score_modeling_data AS score_modeling_data,
                   p.score_experimental_data AS score_experimental_data,
                   p.score_data_access AS score_data_access,
                   p.score_workflow AS score_workflow,
                   p.score_visualization AS score_visualization,
                   p.score_documentation AS score_documentation,
                   p.score_imas AS score_imas,
                   p.score_convention AS score_convention,
                   p.total_bytes AS total_bytes, p.total_lines AS total_lines,
                   p.language_breakdown AS language_breakdown,
                   p.is_multiformat AS is_multiformat,
                   p.pattern_categories AS pattern_categories,
                   p.read_matches AS read_matches,
                   p.write_matches AS write_matches,
                   p.path_purpose AS path_purpose,
                   p.description AS description,
                   p.keywords AS keywords,
                   p.physics_domain AS physics_domain,
                   p.child_names AS child_names,
                   p.total_files AS total_files,
                   p.total_dirs AS total_dirs,
                   p.file_type_counts AS file_type_counts,
                   p.has_readme AS has_readme,
                   p.has_makefile AS has_makefile,
                   p.vcs_type AS vcs_type,
                   p.expansion_reason AS expansion_reason,
                   p.enrich_warnings AS enrich_warnings
            """,
            facility=facility,
            limit=limit,
            cutoff=cutoff,
            root_filter=root_filter or [],
        )
        return list(result)


def mark_enrichment_complete(
    facility: str,
    enrichment_results: list[dict[str, Any]],
) -> int:
    """Mark paths as enriched with deep scan data.

    Args:
        facility: Facility ID
        enrichment_results: List of dicts with path and enrichment data

    Returns:
        Number of paths updated
    """
    from datetime import UTC, datetime

    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    updated = 0

    with GraphClient() as gc:
        for result in enrichment_results:
            path_id = f"{facility}:{result['path']}"

            if result.get("error"):
                # Clear claimed_at so path can be retried, and store error
                gc.query(
                    """
                    MATCH (p:FacilityPath {id: $id})
                    SET p.claimed_at = null,
                        p.enrich_error = $error
                    """,
                    id=path_id,
                    error=result["error"][:200],
                )
                continue

            # Serialize language_breakdown dict to JSON (Neo4j rejects dicts)
            lang_breakdown = result.get("language_breakdown")
            if isinstance(lang_breakdown, dict):
                import json

                lang_breakdown = json.dumps(lang_breakdown) if lang_breakdown else None

            # Serialize pattern_categories dict to JSON
            pattern_cats = result.get("pattern_categories")
            if isinstance(pattern_cats, dict):
                import json

                pattern_cats = json.dumps(pattern_cats) if pattern_cats else None

            # Serialize warnings list to comma-separated string for Neo4j
            warnings = result.get("warnings", [])
            warn_str = ", ".join(warnings) if warnings else None

            gc.query(
                """
                MATCH (p:FacilityPath {id: $id})
                SET p.is_enriched = true,
                    p.enriched_at = $now,
                    p.total_bytes = $total_bytes,
                    p.total_lines = $total_lines,
                    p.language_breakdown = $language_breakdown,
                    p.is_multiformat = $is_multiformat,
                    p.pattern_categories = $pattern_categories,
                    p.read_matches = $read_matches,
                    p.write_matches = $write_matches,
                    p.enrich_warnings = $enrich_warnings,
                    p.claimed_at = null
                """,
                id=path_id,
                now=now,
                total_bytes=result.get("total_bytes"),
                total_lines=result.get("total_lines"),
                language_breakdown=lang_breakdown,
                is_multiformat=result.get("is_multiformat", False),
                pattern_categories=pattern_cats,
                read_matches=result.get("read_matches", 0),
                write_matches=result.get("write_matches", 0),
                enrich_warnings=warn_str,
            )
            updated += 1

    return updated


def mark_refine_complete(
    facility: str,
    refine_results: list[dict[str, Any]],
) -> int:
    """Mark paths with refined scores after rescoring.

    Persists adjusted scores and evidence from pattern matching.

    Args:
        facility: Facility ID
        refine_results: List of dicts with path, new scores, and evidence

    Returns:
        Number of paths updated
    """
    import json
    from datetime import UTC, datetime

    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    updated = 0

    with GraphClient() as gc:
        for result in refine_results:
            path_id = f"{facility}:{result['path']}"

            # Serialize evidence lists to JSON
            primary_evidence = result.get("primary_evidence", [])
            if isinstance(primary_evidence, list):
                primary_evidence = json.dumps(primary_evidence)

            # Build per-dimension SET clauses for scores that were adjusted
            dim_params: dict[str, float] = {}
            dim_set_clauses: list[str] = []
            from imas_codex.discovery.paths.frontier import SCORE_DIMENSIONS

            for dim in SCORE_DIMENSIONS:
                if dim in result:
                    dim_params[dim] = result[dim]
                    dim_set_clauses.append(f"p.{dim} = ${dim}")

            extra_set = ""
            if dim_set_clauses:
                extra_set = ",\n                    " + ",\n                    ".join(
                    dim_set_clauses
                )

            # Serialize keywords to JSON for graph storage
            keywords = result.get("keywords", [])
            if isinstance(keywords, list):
                keywords = json.dumps(keywords)

            gc.query(
                f"""
                MATCH (p:FacilityPath {{id: $id}})
                SET p.refined_at = $now,
                    p.score = $score,
                    p.score_cost = coalesce(p.score_cost, 0) + $score_cost,
                    p.refine_reason = $adjustment_reason,
                    p.primary_evidence = $primary_evidence,
                    p.evidence_summary = $evidence_summary,
                    p.description = $description,
                    p.keywords = $keywords,
                    p.path_purpose = $path_purpose,
                    p.physics_domain = $physics_domain,
                    p.claimed_at = null{extra_set}
                """,
                id=path_id,
                now=now,
                score=result.get("score"),
                score_cost=result.get("score_cost", 0.0),
                adjustment_reason=result.get("adjustment_reason", ""),
                primary_evidence=primary_evidence,
                evidence_summary=result.get("evidence_summary", ""),
                description=result.get("description", ""),
                keywords=keywords,
                path_purpose=result.get("path_purpose"),
                physics_domain=result.get("physics_domain"),
                **dim_params,
            )
            updated += 1

    return updated


# ============================================================================
# Async Workers
# ============================================================================


async def scan_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[str] | None, list[dict] | None], None]
    | None = None,
    batch_size: int = 50,
) -> None:
    """Async scanner worker.

    Continuously claims pending paths, scans via SSH, marks complete.
    Runs until stop_requested or no more pending paths.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per SSH call (default 50)
    """
    from imas_codex.discovery.paths.scanner import async_scan_paths

    while not state.should_stop():
        # Claim work from graph
        paths = claim_paths_for_scanning(
            state.facility, limit=batch_size, root_filter=state.root_filter
        )

        if not paths:
            state.scan_phase.record_idle()
            if on_progress:
                on_progress("idle", state.scan_stats, None, None)
            # Wait before polling again
            await asyncio.sleep(1.0)
            continue

        state.scan_phase.record_activity(len(paths))
        path_strs = [p["path"] for p in paths]

        if on_progress:
            on_progress(f"scanning {len(paths)} paths", state.scan_stats, None, None)

        # Async SSH scan — fully cancellable, no thread executor
        start = time.time()
        try:
            results = await async_scan_paths(
                state.facility, path_strs, enable_rg=False, enable_size=False
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            # Transient failure (timeout or SSH connection error)
            # Revert paths to 'discovered' for retry
            state.ssh_retry_count += 1
            state.ssh_error_message = str(e)[:100]

            # Exponential backoff: 2s, 4s, 8s, 16s, 32s
            backoff_seconds = min(2**state.ssh_retry_count, 32)

            logger.warning(
                f"SSH failure {state.ssh_retry_count}/{state.max_ssh_retries}, "
                f"retry in {backoff_seconds}s: {e}"
            )
            _revert_listing_claim(state.facility, path_strs)
            state.scan_stats.errors += len(paths)

            if state.ssh_retry_count >= state.max_ssh_retries:
                logger.error(
                    f"SSH connection to {state.facility} failed after "
                    f"{state.max_ssh_retries} attempts. "
                    f"Scan worker stopping (other workers continue)."
                )
                # Only stop this scan worker — don't kill all workers.
                # Enrich, refine, and embed workers may still have work.
                state.scan_phase.mark_done()  # Mark as done so should_stop can evaluate
                if on_progress:
                    on_progress(
                        f"SSH failed: {state.ssh_error_message}",
                        state.scan_stats,
                        None,
                        None,
                    )
                break

            if on_progress:
                on_progress(
                    f"SSH retry {state.ssh_retry_count} in {backoff_seconds}s",
                    state.scan_stats,
                    None,
                    None,
                )
            await asyncio.sleep(backoff_seconds)
            continue

        # SSH succeeded - reset retry counter
        state.ssh_retry_count = 0
        state.ssh_error_message = None
        state.scan_stats.last_batch_time = time.time() - start

        # Persist results (marks scanning → scanned)
        # Run in executor to avoid blocking event loop
        # is_expanding=False since scan_worker only handles initial scans
        # Convert ChildDirInfo objects to dicts for serialization
        batch_data = [
            (
                r.path,
                r.stats.to_dict(),
                [
                    {
                        "path": c.path,
                        "is_symlink": c.is_symlink,
                        "realpath": c.realpath,
                        "device_inode": c.device_inode,
                    }
                    for c in r.child_dirs
                ],
                r.error,
                False,  # Not expanding - that's handled by expand_worker
            )
            for r in results
        ]

        # Collect excluded directories with parent paths and reasons
        excluded_data = []
        for r in results:
            for excluded_path, reason in r.excluded_dirs:
                excluded_data.append((excluded_path, r.path, reason))

        stats = await mark_scan_complete(
            state.facility,
            batch_data,
            excluded=excluded_data if excluded_data else None,
        )

        state.scan_stats.processed += stats["scanned"]
        state.scan_stats.errors += stats["errors"]

        # Build detailed scan results for progress display
        scan_results = [
            {
                "path": r.path,
                "total_files": r.stats.total_files,
                "total_dirs": r.stats.total_dirs,
                "has_readme": r.stats.has_readme,
                "has_makefile": r.stats.has_makefile,
                "has_git": r.stats.has_git,
                "file_types": {},  # Could be extracted if available
                "error": r.error,
            }
            for r in results
            if not r.error
        ]

        # Pass detailed scan results to progress callback
        scanned_paths = [r.path for r in results if not r.error]
        if on_progress:
            on_progress(
                f"scanned {stats['scanned']}",
                state.scan_stats,
                scanned_paths,
                scan_results,
            )

        # Brief yield to allow score worker to run
        await asyncio.sleep(0.1)


async def expand_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[str] | None, list[dict] | None], None]
    | None = None,
    batch_size: int = 50,
) -> None:
    """Async expansion worker.

    Expands scored high-value paths by enumerating their children.
    Runs independently of scan_worker, claiming paths with should_expand=true.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per SSH call (default 50)
    """
    from imas_codex.discovery.paths.scanner import async_scan_paths

    while not state.should_stop():
        # Claim expansion work from graph - paths with should_expand=true
        paths = claim_paths_for_expanding(
            state.facility, limit=batch_size, root_filter=state.root_filter
        )

        if not paths:
            state.expand_phase.record_idle()
            if on_progress:
                on_progress("idle", state.expand_stats, None, None)
            # Wait before polling again
            await asyncio.sleep(1.0)
            continue

        state.expand_phase.record_activity(len(paths))
        path_strs = [p["path"] for p in paths]

        if on_progress:
            on_progress(f"expanding {len(paths)} paths", state.expand_stats, None, None)

        # Async SSH scan — fully cancellable, no thread executor
        start = time.time()
        try:
            results = await async_scan_paths(
                state.facility, path_strs, enable_rg=False, enable_size=False
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            # Transient failure - revert paths for retry
            logger.warning(f"Expand SSH failure: {e}")
            _revert_listing_claim(state.facility, path_strs)
            state.expand_stats.errors += len(paths)
            await asyncio.sleep(5.0)
            continue

        state.expand_stats.last_batch_time = time.time() - start

        # Persist results with is_expanding=True to create child paths
        # Convert ChildDirInfo objects to dicts for serialization
        batch_data = [
            (
                r.path,
                r.stats.to_dict(),
                [
                    {
                        "path": c.path,
                        "is_symlink": c.is_symlink,
                        "realpath": c.realpath,
                        "device_inode": c.device_inode,
                    }
                    for c in r.child_dirs
                ],
                r.error,
                True,  # is_expanding - creates child paths
            )
            for r in results
        ]

        # Collect excluded directories
        excluded_data = []
        for r in results:
            for excluded_path, reason in r.excluded_dirs:
                excluded_data.append((excluded_path, r.path, reason))

        stats = await mark_scan_complete(
            state.facility,
            batch_data,
            excluded=excluded_data if excluded_data else None,
        )

        state.expand_stats.processed += stats["scanned"]
        state.expand_stats.errors += stats["errors"]

        # Build detailed scan results for progress display
        scan_results = [
            {
                "path": r.path,
                "total_files": r.stats.total_files,
                "total_dirs": r.stats.total_dirs,
                "has_readme": r.stats.has_readme,
                "has_makefile": r.stats.has_makefile,
                "has_git": r.stats.has_git,
                "error": r.error,
            }
            for r in results
            if not r.error
        ]

        expanded_paths = [r.path for r in results if not r.error]
        if on_progress:
            on_progress(
                f"expanded {stats['scanned']}",
                state.expand_stats,
                expanded_paths,
                scan_results,
            )

        await asyncio.sleep(0.1)


async def score_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[dict] | None], None] | None = None,
    batch_size: int = 25,
) -> None:
    """Async scorer worker.

    Continuously claims scanned paths, scores via LLM, marks complete.
    Runs until stop_requested, budget exhausted, or no more scanned paths.

    Optimization: Empty directories (total_files=0 AND total_dirs=0) are
    auto-skipped without LLM call since they have no content to evaluate.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per LLM call (default 25)
    """
    from imas_codex.discovery.paths.scorer import DirectoryScorer

    scorer = DirectoryScorer(facility=state.facility)
    loop = asyncio.get_running_loop()

    while not state.should_stop():
        # Check budget before claiming work
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.score_stats, None)
            break

        # Claim work from graph
        paths = claim_paths_for_scoring(
            state.facility, limit=batch_size, root_filter=state.root_filter
        )

        if not paths:
            state.score_phase.record_idle()
            if state.score_phase.idle_count <= 3:
                logger.debug(
                    f"Score worker idle ({state.score_phase.idle_count}), "
                    f"scan_processed={state.scan_stats.processed}"
                )
            if on_progress:
                on_progress("waiting for scanned paths", state.score_stats, None)
            # Wait before polling again
            await asyncio.sleep(2.0)
            continue

        state.score_phase.record_activity(len(paths))
        logger.debug(f"Score worker claimed {len(paths)} paths")

        # Split paths into empty (auto-skip) and non-empty (need LLM)
        empty_paths = []
        paths_to_score = []
        for p in paths:
            total_files = p.get("total_files", 0) or 0
            total_dirs = p.get("total_dirs", 0) or 0
            if total_files == 0 and total_dirs == 0:
                empty_paths.append(p)
            else:
                paths_to_score.append(p)

        # Auto-skip empty directories with 0.0 score
        skipped_results = []
        if empty_paths:
            skip_data = [
                {
                    "path": p["path"],
                    "score": 0.0,
                    "path_purpose": "empty_directory",
                    "description": "Empty directory - no files or subdirectories",
                    "score_modeling_code": 0.0,
                    "score_analysis_code": 0.0,
                    "score_operations_code": 0.0,
                    "score_modeling_data": 0.0,
                    "score_experimental_data": 0.0,
                    "score_data_access": 0.0,
                    "score_workflow": 0.0,
                    "score_visualization": 0.0,
                    "score_documentation": 0.0,
                    "score_imas": 0.0,
                    "score_convention": 0.0,
                    "should_expand": False,
                    "skip_reason": "empty",
                    "terminal_reason": TerminalReason.empty.value,
                }
                for p in empty_paths
            ]
            await loop.run_in_executor(
                None, lambda sd=skip_data: mark_score_complete(state.facility, sd)
            )
            state.score_stats.processed += len(empty_paths)

            skipped_results = [
                {
                    "path": p["path"],
                    "score": 0.0,
                    "label": "empty_directory",
                    "path_purpose": "empty_directory",
                    "description": "Empty directory - no files or subdirectories",
                    "score_imas": 0.0,
                    "score_convention": 0.0,
                    "skip_reason": "empty",
                    "should_expand": False,
                    "total_files": 0,
                }
                for p in empty_paths
            ]

            if on_progress and not paths_to_score:
                # Only skipped paths, show progress
                on_progress(
                    f"skipped {len(empty_paths)} empty",
                    state.score_stats,
                    skipped_results,
                )
                continue

        if not paths_to_score:
            continue

        if on_progress:
            on_progress(f"scoring {len(paths_to_score)} paths", state.score_stats, None)

        # Async LLM scoring — fully cancellable
        start = time.time()
        try:
            result = await scorer.async_score_batch(
                directories=paths_to_score,
                focus=state.focus,
                threshold=state.threshold,
            )
            state.score_stats.last_batch_time = time.time() - start
            state.score_stats.cost += result.total_cost

            # Persist results (marks scoring → scored)
            # Run in executor to avoid blocking event loop
            score_data = [d.to_graph_dict() for d in result.scored_dirs]
            await loop.run_in_executor(
                None, lambda sd=score_data: mark_score_complete(state.facility, sd)
            )

            state.score_stats.processed += len(result.scored_dirs)

            # Build detailed score results for progress callback
            detailed_results = [
                {
                    "path": d.path,
                    "score": d.score,
                    "label": d.path_purpose.value if d.path_purpose else "",
                    "path_purpose": d.path_purpose.value if d.path_purpose else "",
                    "description": d.description,
                    "score_imas": d.score_imas,
                    "skip_reason": d.skip_reason or "",
                    "should_expand": d.should_expand,
                    "total_files": 0,  # Not available at score time
                }
                for d in result.scored_dirs
            ]
            # Combine with any skipped results
            all_results = skipped_results + detailed_results
            if on_progress:
                skipped_msg = (
                    f" (+{len(skipped_results)} skipped)" if skipped_results else ""
                )
                on_progress(
                    f"scored {len(result.scored_dirs)} (${result.total_cost:.3f}){skipped_msg}",
                    state.score_stats,
                    all_results,
                )

        except ValueError:
            # LLM validation error - revert paths to 'scanned' status for retry
            # DO NOT increment error count - this will be retried automatically
            logger.warning(
                f"LLM validation error for batch of {len(paths_to_score)} paths. "
                "Reverting to scanned status for retry."
            )
            _revert_scoring_claim(state.facility, [p["path"] for p in paths_to_score])
            # Don't show validation errors in progress display
        except Exception as e:
            # Other errors - increment error count and revert
            logger.exception(f"Score error: {e}")
            state.score_stats.errors += len(paths_to_score)
            _revert_scoring_claim(state.facility, [p["path"] for p in paths_to_score])

        # Brief yield
        await asyncio.sleep(0.1)


def _revert_path_claims(facility: str, paths: list[str]) -> None:
    """Release claimed paths on error by clearing claimed_at.

    Unified handler for all path claim reverts (scanning, listing,
    scoring, enrichment).  Uses facility-scoped query to avoid
    accidentally releasing paths from another facility.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE p.path IN $paths
            SET p.claimed_at = null
            """,
            facility=facility,
            paths=paths,
        )


# Backwards-compatible aliases
_revert_scoring_claim = _revert_path_claims
_revert_listing_claim = _revert_path_claims
_revert_enrich_claim = _revert_path_claims


async def enrich_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[dict] | None], None] | None = None,
    batch_size: int = 25,
) -> None:
    """Async enrichment worker.

    Continuously claims scored paths with should_enrich=true, runs deep
    analysis (du, tokei, patterns) via SSH, marks complete.
    Runs in PARALLEL with scan/score/expand workers.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per SSH call (default 25)
    """
    # Use local claim_paths_for_enriching and mark_enrichment_complete
    # (defined above in this module with root_filter support)
    from imas_codex.discovery.paths.enrichment import async_enrich_paths

    loop = asyncio.get_running_loop()

    while not state.should_stop():
        # Claim work from graph
        paths = claim_paths_for_enriching(
            state.facility,
            limit=batch_size,
            root_filter=state.root_filter,
            auto_enrich_threshold=state.auto_enrich_threshold,
        )

        if not paths:
            state.enrich_phase.record_idle()
            if on_progress:
                on_progress("waiting for enrichable paths", state.enrich_stats, None)
            # Wait before polling again
            await asyncio.sleep(2.0)
            continue

        state.enrich_phase.record_activity(len(paths))
        path_strs = [p["path"] for p in paths]
        # Build path -> purpose mapping for targeted pattern selection
        path_purposes = {p["path"]: p.get("path_purpose") for p in paths}

        if on_progress:
            on_progress(f"enriching {len(paths)} paths", state.enrich_stats, None)

        # Async SSH enrichment — fully cancellable
        start = time.time()
        try:
            results = await async_enrich_paths(
                state.facility, path_strs, path_purposes=path_purposes
            )
            state.enrich_stats.last_batch_time = time.time() - start

            # Convert EnrichmentResult to dict for persistence and display
            result_dicts = [
                {
                    "path": r.path,
                    "total_bytes": r.total_bytes,
                    "total_lines": r.total_lines,
                    "language_breakdown": r.language_breakdown,
                    "pattern_categories": r.pattern_categories,
                    "is_multiformat": r.is_multiformat,
                    "read_matches": r.read_matches,
                    "write_matches": r.write_matches,
                    "error": r.error,
                    "warnings": r.warnings if hasattr(r, "warnings") else [],
                }
                for r in results
            ]

            # Persist results
            enriched = await loop.run_in_executor(
                None,
                lambda rd=result_dicts: mark_enrichment_complete(state.facility, rd),
            )

            state.enrich_stats.processed += enriched
            state.enrich_stats.errors += len([r for r in results if r.error])

            if on_progress:
                on_progress(f"enriched {enriched}", state.enrich_stats, result_dicts)

        except Exception as e:
            logger.exception(f"Enrich error: {e}")
            state.enrich_stats.errors += len(paths)
            _revert_enrich_claim(state.facility, path_strs)

        # Brief yield
        await asyncio.sleep(0.1)


async def refine_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[dict] | None], None] | None = None,
    batch_size: int = 10,
) -> None:
    """Async refine worker.

    Continuously claims enriched paths, refines using enrichment data
    (total_bytes, total_lines, language_breakdown) for refined scoring.
    Cheaper LLM call since we use a simpler prompt with concrete metrics.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per refine operation (default 10)
    """
    # Use local claim_paths_for_refining and mark_refine_complete
    # (defined above in this module with root_filter support)

    loop = asyncio.get_running_loop()

    while not state.should_stop():
        # Check budget before claiming work
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.refine_stats, None)
            break

        # Claim work from graph
        paths = claim_paths_for_refining(
            state.facility, limit=batch_size, root_filter=state.root_filter
        )

        if not paths:
            state.refine_phase.record_idle()
            if on_progress:
                on_progress("waiting for enriched paths", state.refine_stats, None)
            # Wait before polling again
            await asyncio.sleep(2.0)
            continue

        state.refine_phase.record_activity(len(paths))

        if on_progress:
            on_progress(f"rescoring {len(paths)} paths", state.refine_stats, None)

        # Run LLM-based rescoring in thread pool
        start = time.time()
        try:
            # LLM rescoring uses enrichment data for intelligent score refinement
            # Pass facility for cross-facility example injection
            llm_results, cost = await _async_refine_with_llm(
                paths, state.facility, focus=state.focus
            )

            # Add should_expand from original data and cost tracking
            refine_results = []
            failed_count = 0
            cost_per_path = cost / len(paths) if paths else 0.0
            for llm_r in llm_results:
                # Skip failed results — don't corrupt graph with stale scores
                if llm_r.get("_failed"):
                    failed_count += 1
                    continue
                # Find original path data for should_expand
                orig = next((p for p in paths if p["path"] == llm_r["path"]), {})
                result: dict = {
                    "path": llm_r["path"],
                    "score": llm_r["score"],
                    "score_cost": cost_per_path,
                    "should_expand": orig.get("should_expand", True),
                    "adjustment_reason": llm_r.get("adjustment_reason", ""),
                    "primary_evidence": llm_r.get("primary_evidence", []),
                    "evidence_summary": llm_r.get("evidence_summary", ""),
                    "description": llm_r.get("description", ""),
                    "keywords": llm_r.get("keywords", []),
                    "path_purpose": llm_r.get("path_purpose"),
                    "physics_domain": llm_r.get("physics_domain"),
                }
                # Copy per-dimension scores from LLM results
                for dim in [
                    "score_modeling_code",
                    "score_analysis_code",
                    "score_operations_code",
                    "score_modeling_data",
                    "score_experimental_data",
                    "score_data_access",
                    "score_workflow",
                    "score_visualization",
                    "score_documentation",
                    "score_imas",
                    "score_convention",
                ]:
                    if dim in llm_r:
                        result[dim] = llm_r[dim]
                refine_results.append(result)

            if failed_count:
                logger.warning(
                    "Skipped %d/%d failed refinements (not written to graph)",
                    failed_count,
                    len(llm_results),
                )
                state.refine_stats.errors += failed_count

            state.refine_stats.last_batch_time = time.time() - start
            state.refine_stats.cost += cost

            # Persist only successful results
            refined = 0
            if refine_results:
                refined = await loop.run_in_executor(
                    None,
                    lambda rr=refine_results: mark_refine_complete(state.facility, rr),
                )

            state.refine_stats.processed += refined

            if on_progress:
                on_progress(f"refined {refined}", state.refine_stats, refine_results)

        except Exception as e:
            logger.exception(f"Refine error: {e}")
            state.refine_stats.errors += len(paths)

        # Brief yield
        await asyncio.sleep(0.1)


def _refine_with_llm(
    paths: list[dict],
    facility: str | None = None,
    focus: str | None = None,
) -> tuple[list[dict], float]:
    """Refine paths using LLM with enrichment data.

    Injects cross-facility enriched examples into the prompt for
    consistent scoring calibration across facilities.

    Args:
        paths: List of path dicts with enrichment data
        facility: Current facility for preferring same-facility examples
        focus: Optional focus string for scoring

    Returns:
        Tuple of (refine_results, cost)
    """
    import json

    from imas_codex.agentic.prompt_loader import render_prompt
    from imas_codex.discovery.paths.frontier import sample_enriched_paths
    from imas_codex.discovery.paths.models import RefineBatch
    from imas_codex.settings import get_model

    # Build prompt context with enriched examples
    context: dict = {}

    # Sample enriched paths for calibration (cross-facility)
    enriched_examples = sample_enriched_paths(
        facility=facility,
        per_category=2,
        cross_facility=True,
    )
    has_examples = any(enriched_examples.get(cat) for cat in enriched_examples)
    if has_examples:
        context["enriched_examples"] = enriched_examples
    if focus:
        context["focus"] = focus

    # Render prompt with examples
    system_prompt = render_prompt("discovery/refiner", context)

    # Build user prompt with FULL context: initial scoring + enrichment data
    lines = ["Refine these directories using their enrichment metrics:\n"]
    for p in paths:
        # Parse language breakdown if it's a JSON string
        lang = p.get("language_breakdown")
        if isinstance(lang, str):
            try:
                lang = json.loads(lang)
            except json.JSONDecodeError:
                lang = {}

        lines.append(f"\n## Path: {p['path']}")
        lines.append(f"Depth: {p.get('depth', 0)}")

        # Initial scoring context (what the scorer saw and decided)
        lines.append(f"Purpose: {p.get('path_purpose', 'unknown')}")
        if p.get("physics_domain"):
            lines.append(f"Physics domain: {p['physics_domain']}")
        if p.get("description"):
            lines.append(f"Description: {p['description']}")
        if p.get("keywords"):
            keywords = p["keywords"]
            if isinstance(keywords, list):
                keywords = ", ".join(keywords)
            lines.append(f"Keywords: {keywords}")
        if p.get("expansion_reason"):
            lines.append(f"Expansion reason: {p['expansion_reason']}")

        # Filesystem structure
        total_files = p.get("total_files")
        total_dirs = p.get("total_dirs")
        if total_files is not None or total_dirs is not None:
            lines.append(f"Files: {total_files or 0}, Dirs: {total_dirs or 0}")
        file_types = p.get("file_type_counts")
        if file_types:
            if isinstance(file_types, str):
                lines.append(f"File types: {file_types}")
            else:
                lines.append(f"File types: {file_types}")
        for indicator in ["has_readme", "has_makefile", "vcs_type"]:
            val = p.get(indicator)
            if val:
                lines.append(f"  {indicator}: {val}")

        # Child contents (truncated) - helps understand what's in the directory
        child_names = p.get("child_names")
        if child_names:
            if isinstance(child_names, str):
                # Truncate long child lists
                if len(child_names) > 200:
                    child_names = child_names[:200] + "..."
            lines.append(f"Contents: {child_names}")

        # Enrichment metrics (new data from deep analysis)
        lines.append("\nEnrichment data:")
        lines.append(f"  Total lines: {p.get('total_lines') or 0}")
        lines.append(f"  Total bytes: {p.get('total_bytes') or 0}")
        lines.append(f"  Language breakdown: {lang or {}}")
        lines.append(f"  Is multiformat: {p.get('is_multiformat', False)}")

        # Pattern match evidence (key data for rescoring)
        pattern_cats = p.get("pattern_categories")
        if isinstance(pattern_cats, str):
            try:
                pattern_cats = json.loads(pattern_cats)
            except json.JSONDecodeError:
                pattern_cats = {}
        if pattern_cats:
            lines.append(f"  Pattern categories: {pattern_cats}")
        lines.append(f"  Read matches: {p.get('read_matches') or 0}")
        lines.append(f"  Write matches: {p.get('write_matches') or 0}")

        # Enrichment warnings (tokei timeout, du timeout, etc.)
        enrich_warnings = p.get("enrich_warnings")
        if enrich_warnings:
            if isinstance(enrich_warnings, str):
                lines.append(f"  Enrichment warnings: {enrich_warnings}")
                lines.append(
                    "  Note: Some metrics may be incomplete due to timeouts on large directories."
                )

        # Initial per-dimension scores (what we're potentially adjusting)
        # Use `or 0.0` since .get() returns None if key exists with None value
        lines.append("\nInitial scores:")
        lines.append(
            f"  score_modeling_code: {p.get('score_modeling_code') or 0.0:.2f}"
        )
        lines.append(
            f"  score_analysis_code: {p.get('score_analysis_code') or 0.0:.2f}"
        )
        lines.append(
            f"  score_operations_code: {p.get('score_operations_code') or 0.0:.2f}"
        )
        lines.append(
            f"  score_modeling_data: {p.get('score_modeling_data') or 0.0:.2f}"
        )
        lines.append(
            f"  score_experimental_data: {p.get('score_experimental_data') or 0.0:.2f}"
        )
        lines.append(f"  score_data_access: {p.get('score_data_access') or 0.0:.2f}")
        lines.append(f"  score_workflow: {p.get('score_workflow') or 0.0:.2f}")
        lines.append(
            f"  score_visualization: {p.get('score_visualization') or 0.0:.2f}"
        )
        lines.append(
            f"  score_documentation: {p.get('score_documentation') or 0.0:.2f}"
        )
        lines.append(f"  score_imas: {p.get('score_imas') or 0.0:.2f}")
        lines.append(f"  score_convention: {p.get('score_convention') or 0.0:.2f}")
        lines.append(f"  combined_score: {p.get('score') or 0.0:.2f}")

    user_prompt = "\n".join(lines)

    # Get model
    model = get_model("language")  # Use same model as scorer

    # Call LLM with shared retry+parse loop — retries on both API errors
    # and JSON/validation errors (same resilience as wiki pipeline).
    from imas_codex.discovery.base.llm import call_llm_structured

    try:
        batch, cost, _tokens = call_llm_structured(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=RefineBatch,
        )
    except ValueError:
        # All retries exhausted — mark all as failed so they're skipped
        return [
            {
                "path": p["path"],
                "score": p.get("score", 0.5),
                "adjustment_reason": "parse error",
                "_failed": True,
            }
            for p in paths
        ], 0.0

    # Build results dict with per-dimension updates
    path_to_result = {r.path: r for r in batch.results}
    results = []

    for p in paths:
        if p["path"] in path_to_result:
            r = path_to_result[p["path"]]
            result: dict = {
                "path": p["path"],
                "score": max(0.0, min(1.0, r.new_score)),
                "adjustment_reason": (r.adjustment_reason[:80] or ""),
                "primary_evidence": r.primary_evidence or [],
                "evidence_summary": (r.evidence_summary or "")[:200],
                "description": r.description or p.get("description", ""),
                "keywords": r.keywords or p.get("keywords", []),
                "path_purpose": (r.path_purpose.value if r.path_purpose else None)
                or p.get("path_purpose"),
                "physics_domain": (r.physics_domain.value if r.physics_domain else None)
                or p.get("physics_domain"),
            }

            # Add per-dimension scores (use original if LLM returned None)
            for dim in [
                "score_modeling_code",
                "score_analysis_code",
                "score_operations_code",
                "score_modeling_data",
                "score_experimental_data",
                "score_data_access",
                "score_workflow",
                "score_visualization",
                "score_documentation",
                "score_imas",
                "score_convention",
            ]:
                llm_value = getattr(r, dim, None)
                if llm_value is not None:
                    result[dim] = max(0.0, min(1.0, llm_value))
                else:
                    # Keep original value
                    result[dim] = p.get(dim, 0.0)

            results.append(result)
        else:
            # Path not in response — mark as failed so it's skipped
            results.append(
                {
                    "path": p["path"],
                    "score": p.get("score", 0.5),
                    "adjustment_reason": "not in response",
                    "_failed": True,
                }
            )

    return results, cost


async def _async_refine_with_llm(
    paths: list[dict],
    facility: str | None = None,
    focus: str | None = None,
) -> tuple[list[dict], float]:
    """Refine paths using LLM with enrichment data (async/cancellable).

    Async version of _refine_with_llm using acall_llm_structured for
    native async LLM calls that respond to asyncio.cancel().

    Injects cross-facility enriched examples into the prompt for
    consistent scoring calibration across facilities.

    Args:
        paths: List of path dicts with enrichment data
        facility: Current facility for preferring same-facility examples
        focus: Optional focus string for scoring

    Returns:
        Tuple of (refine_results, cost)
    """
    import json

    from imas_codex.agentic.prompt_loader import render_prompt
    from imas_codex.discovery.paths.frontier import sample_enriched_paths
    from imas_codex.discovery.paths.models import RefineBatch
    from imas_codex.settings import get_model

    # Build prompt context with enriched examples
    context: dict = {}

    # Sample enriched paths for calibration (cross-facility)
    enriched_examples = sample_enriched_paths(
        facility=facility,
        per_category=2,
        cross_facility=True,
    )
    has_examples = any(enriched_examples.get(cat) for cat in enriched_examples)
    if has_examples:
        context["enriched_examples"] = enriched_examples
    if focus:
        context["focus"] = focus

    # Render prompt with examples
    system_prompt = render_prompt("discovery/refiner", context)

    # Build user prompt with FULL context: initial scoring + enrichment data
    lines_prompt = ["Refine these directories using their enrichment metrics:\n"]
    for p in paths:
        # Parse language breakdown if it's a JSON string
        lang = p.get("language_breakdown")
        if isinstance(lang, str):
            try:
                lang = json.loads(lang)
            except json.JSONDecodeError:
                lang = {}

        lines_prompt.append(f"\n## Path: {p['path']}")
        lines_prompt.append(f"Depth: {p.get('depth', 0)}")

        # Initial scoring context (what the scorer saw and decided)
        lines_prompt.append(f"Purpose: {p.get('path_purpose', 'unknown')}")
        if p.get("physics_domain"):
            lines_prompt.append(f"Physics domain: {p['physics_domain']}")
        if p.get("description"):
            lines_prompt.append(f"Description: {p['description']}")
        if p.get("keywords"):
            keywords = p["keywords"]
            if isinstance(keywords, list):
                keywords = ", ".join(keywords)
            lines_prompt.append(f"Keywords: {keywords}")
        if p.get("expansion_reason"):
            lines_prompt.append(f"Expansion reason: {p['expansion_reason']}")

        # Filesystem structure
        total_files = p.get("total_files")
        total_dirs = p.get("total_dirs")
        if total_files is not None or total_dirs is not None:
            lines_prompt.append(f"Files: {total_files or 0}, Dirs: {total_dirs or 0}")
        file_types = p.get("file_type_counts")
        if file_types:
            if isinstance(file_types, str):
                lines_prompt.append(f"File types: {file_types}")
            else:
                lines_prompt.append(f"File types: {file_types}")
        for indicator in ["has_readme", "has_makefile", "vcs_type"]:
            val = p.get(indicator)
            if val:
                lines_prompt.append(f"  {indicator}: {val}")

        # Child contents (truncated) - helps understand what's in the directory
        child_names = p.get("child_names")
        if child_names:
            if isinstance(child_names, str):
                # Truncate long child lists
                if len(child_names) > 200:
                    child_names = child_names[:200] + "..."
            lines_prompt.append(f"Contents: {child_names}")

        # Enrichment metrics (new data from deep analysis)
        lines_prompt.append("\nEnrichment data:")
        lines_prompt.append(f"  Total lines: {p.get('total_lines') or 0}")
        lines_prompt.append(f"  Total bytes: {p.get('total_bytes') or 0}")
        lines_prompt.append(f"  Language breakdown: {lang or {}}")
        lines_prompt.append(f"  Is multiformat: {p.get('is_multiformat', False)}")

        # Pattern match evidence (key data for rescoring)
        pattern_cats = p.get("pattern_categories")
        if isinstance(pattern_cats, str):
            try:
                pattern_cats = json.loads(pattern_cats)
            except json.JSONDecodeError:
                pattern_cats = {}
        if pattern_cats:
            lines_prompt.append(f"  Pattern categories: {pattern_cats}")
        lines_prompt.append(f"  Read matches: {p.get('read_matches') or 0}")
        lines_prompt.append(f"  Write matches: {p.get('write_matches') or 0}")

        # Enrichment warnings (tokei timeout, etc.)
        enrich_warnings = p.get("enrich_warnings")
        if enrich_warnings:
            if isinstance(enrich_warnings, str):
                lines_prompt.append(f"  Enrichment warnings: {enrich_warnings}")
                lines_prompt.append(
                    "  Note: Some metrics may be incomplete due to timeouts on large directories."
                )

        # Initial per-dimension scores (what we're potentially adjusting)
        # Use `or 0.0` since .get() returns None if key exists with None value
        lines_prompt.append("\nInitial scores:")
        lines_prompt.append(
            f"  score_modeling_code: {p.get('score_modeling_code') or 0.0:.2f}"
        )
        lines_prompt.append(
            f"  score_analysis_code: {p.get('score_analysis_code') or 0.0:.2f}"
        )
        lines_prompt.append(
            f"  score_operations_code: {p.get('score_operations_code') or 0.0:.2f}"
        )
        lines_prompt.append(
            f"  score_modeling_data: {p.get('score_modeling_data') or 0.0:.2f}"
        )
        lines_prompt.append(
            f"  score_experimental_data: {p.get('score_experimental_data') or 0.0:.2f}"
        )
        lines_prompt.append(
            f"  score_data_access: {p.get('score_data_access') or 0.0:.2f}"
        )
        lines_prompt.append(f"  score_workflow: {p.get('score_workflow') or 0.0:.2f}")
        lines_prompt.append(
            f"  score_visualization: {p.get('score_visualization') or 0.0:.2f}"
        )
        lines_prompt.append(
            f"  score_documentation: {p.get('score_documentation') or 0.0:.2f}"
        )
        lines_prompt.append(f"  score_imas: {p.get('score_imas') or 0.0:.2f}")
        lines_prompt.append(
            f"  score_convention: {p.get('score_convention') or 0.0:.2f}"
        )
        lines_prompt.append(f"  combined_score: {p.get('score') or 0.0:.2f}")

    user_prompt = "\n".join(lines_prompt)

    # Get model
    model = get_model("language")  # Use same model as scorer

    # Call LLM with shared retry+parse loop — retries on both API errors
    # and JSON/validation errors (same resilience as wiki pipeline).
    from imas_codex.discovery.base.llm import acall_llm_structured

    try:
        batch, cost, _tokens = await acall_llm_structured(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=RefineBatch,
        )
    except ValueError:
        # All retries exhausted — mark all as failed so they're skipped
        return [
            {
                "path": p["path"],
                "score": p.get("score", 0.5),
                "adjustment_reason": "parse error",
                "_failed": True,
            }
            for p in paths
        ], 0.0

    # Build results dict with per-dimension updates
    path_to_result = {r.path: r for r in batch.results}
    results = []

    for p in paths:
        if p["path"] in path_to_result:
            r = path_to_result[p["path"]]
            result: dict = {
                "path": p["path"],
                "score": max(0.0, min(1.0, r.new_score)),
                "adjustment_reason": (r.adjustment_reason[:80] or ""),
                "primary_evidence": r.primary_evidence or [],
                "evidence_summary": (r.evidence_summary or "")[:200],
                "description": r.description or p.get("description", ""),
                "keywords": r.keywords or p.get("keywords", []),
                "path_purpose": (r.path_purpose.value if r.path_purpose else None)
                or p.get("path_purpose"),
                "physics_domain": (r.physics_domain.value if r.physics_domain else None)
                or p.get("physics_domain"),
            }

            # Add per-dimension scores (use original if LLM returned None)
            for dim in [
                "score_modeling_code",
                "score_analysis_code",
                "score_operations_code",
                "score_modeling_data",
                "score_experimental_data",
                "score_data_access",
                "score_workflow",
                "score_visualization",
                "score_documentation",
                "score_imas",
                "score_convention",
            ]:
                llm_value = getattr(r, dim, None)
                if llm_value is not None:
                    result[dim] = max(0.0, min(1.0, llm_value))
                else:
                    # Keep original value
                    result[dim] = p.get(dim, 0.0)

            results.append(result)
        else:
            # Path not in response — mark as failed so it's skipped
            results.append(
                {
                    "path": p["path"],
                    "score": p.get("score", 0.5),
                    "adjustment_reason": "not in response",
                    "_failed": True,
                }
            )

    return results, cost


# ============================================================================
# SSH Preflight Check
# ============================================================================


def check_ssh_connectivity(facility: str, timeout: int = 10) -> tuple[bool, str]:
    """Check if SSH connection to facility is working.

    Args:
        facility: Facility ID
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, message)
    """
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.remote.tools import run

    try:
        config = get_facility(facility)
        ssh_host = config.get("ssh_host", facility)
    except ValueError:
        ssh_host = facility

    try:
        result = run("echo ok", facility=ssh_host, timeout=timeout)
        if "ok" in result:
            return True, f"SSH to {ssh_host} working"
        return False, f"SSH to {ssh_host} returned unexpected output"
    except subprocess.TimeoutExpired:
        return False, f"SSH to {ssh_host} timed out after {timeout}s"
    except subprocess.CalledProcessError as e:
        return False, f"SSH to {ssh_host} failed: {e}"
    except Exception as e:
        return False, f"SSH check failed: {e}"


# ============================================================================
# Main Discovery Loop
# ============================================================================


async def run_parallel_discovery(
    facility: str,
    cost_limit: float = 10.0,
    path_limit: int | None = None,
    focus: str | None = None,
    threshold: float = 0.7,
    root_filter: list[str] | None = None,
    auto_enrich_threshold: float | None = None,
    num_scan_workers: int = 1,
    num_expand_workers: int = 1,
    num_score_workers: int = 1,  # Single scorer: refine also uses LLM (total=2)
    num_enrich_workers: int = 2,  # Two enrichment workers for parallel SSH
    num_refine_workers: int = 1,  # Enabled by default for score refinement
    scan_batch_size: int = 50,
    expand_batch_size: int = 50,
    score_batch_size: int = 50,  # Increased: more work per API call
    enrich_batch_size: int = 10,  # Smaller: heavy SSH operations (du/tokei)
    refine_batch_size: int = 10,  # Smaller batches: 20+ fields per path in structured output
    on_scan_progress: Callable[
        [str, WorkerStats, list[str] | None, list[dict] | None], None
    ]
    | None = None,
    on_expand_progress: Callable[
        [str, WorkerStats, list[str] | None, list[dict] | None], None
    ]
    | None = None,
    on_score_progress: Callable[[str, WorkerStats, list[dict] | None], None]
    | None = None,
    on_enrich_progress: Callable[[str, WorkerStats, list[dict] | None], None]
    | None = None,
    on_refine_progress: Callable[[str, WorkerStats, list[dict] | None], None]
    | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
    graceful_shutdown_timeout: float = 5.0,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Run parallel discovery with all worker types.

    Workers:
    - scan: Lists directories, discovers child paths (SSH-bound)
    - expand: Expands scored high-value paths (should_expand=true)
    - score: LLM-based scoring with should_expand/should_enrich decisions
    - enrich: Deep analysis (du/tokei/patterns) for should_enrich=true paths
    - refine: Refines scores using enrichment data (optional)

    All workers run in PARALLEL - expand/enrich do not wait for each other.

    Args:
        facility: Facility ID to discover
        cost_limit: Maximum LLM cost in dollars
        path_limit: Maximum paths to process (optional)
        focus: Focus string for scoring (optional)
        threshold: Score threshold for expansion
        root_filter: Restrict work to paths under these roots (optional)
        num_scan_workers: Number of concurrent scan workers (default: 1)
        num_expand_workers: Number of concurrent expand workers (default: 1)
        num_score_workers: Number of concurrent score workers (default: 2)
        num_enrich_workers: Number of concurrent enrich workers (default: 1)
        num_refine_workers: Number of concurrent refine workers (default: 1)
        scan_batch_size: Paths per SSH call (default: 50)
        expand_batch_size: Paths per expand SSH call (default: 50)
        score_batch_size: Paths per LLM call (default: 50)
        enrich_batch_size: Paths per SSH call (default: 10)
        refine_batch_size: Paths per refine (default: 50)
        on_scan_progress: Callback for scan progress
        on_expand_progress: Callback for expand progress
        on_score_progress: Callback for score progress
        on_enrich_progress: Callback for enrich progress
        on_refine_progress: Callback for refine progress
        on_worker_status: Callback for worker status changes. Called with
            SupervisedWorkerGroup for display integration.
        graceful_shutdown_timeout: Seconds to wait for workers to finish after
            limit reached before cancelling (default: 5.0)
        deadline: Unix timestamp when discovery should stop (optional)

    Terminates when:
    - Cost limit reached
    - Path limit reached (if set)
    - Deadline expired (if set)
    - All workers idle (no more work)

    Returns:
        Summary dict with scanned, expanded, scored, enriched, refined, cost, elapsed, rates
    """
    from imas_codex.discovery import get_discovery_stats, seed_facility_roots

    # SSH preflight check - fail fast if facility is unreachable
    ssh_ok, ssh_message = check_ssh_connectivity(facility, timeout=15)
    if not ssh_ok:
        logger.error(f"SSH preflight failed: {ssh_message}")
        raise ConnectionError(f"Cannot connect to facility {facility}: {ssh_message}")

    # Release ALL claims from previous runs.  At startup, this process is the
    # only one that should own claims for this facility, so force-clearing is
    # safe and prevents stale claims from a recently-crashed run blocking work.
    reset_orphaned_claims(facility, force=True)

    # Ensure we have paths to discover
    stats = get_discovery_stats(facility)
    if stats["total"] == 0:
        seed_facility_roots(facility)

    # Create shared state
    state = DiscoveryState(
        facility=facility,
        cost_limit=cost_limit,
        path_limit=path_limit,
        focus=focus,
        threshold=threshold,
        root_filter=root_filter,
        auto_enrich_threshold=auto_enrich_threshold,
        deadline=deadline,
    )

    # Mark phases as done for disabled workers so should_stop() works correctly
    # When num_*_workers=0, those workers never run and never produce/consume work
    if num_scan_workers == 0:
        state.scan_phase.mark_done()
    if num_expand_workers == 0:
        state.expand_phase.mark_done()
    if num_score_workers == 0:
        state.score_phase.mark_done()
    if num_enrich_workers == 0:
        state.enrich_phase.mark_done()
    if num_refine_workers == 0:
        state.refine_phase.mark_done()

    # Capture initial terminal count for session-based --limit tracking
    state.initial_terminal_count = state.terminal_count

    # Create supervised worker group
    worker_group = SupervisedWorkerGroup()

    # Scan workers (group="scan")
    for i in range(num_scan_workers):
        worker_name = f"scan_worker_{i}"
        status = worker_group.create_status(worker_name, group="scan")
        worker_group.add_task(
            asyncio.create_task(
                supervised_worker(
                    scan_worker,
                    worker_name,
                    state,
                    state.should_stop,
                    on_progress=on_scan_progress,
                    batch_size=scan_batch_size,
                    status_tracker=status,
                )
            )
        )

    # Expand workers (group="scan" — same pipeline stage)
    for i in range(num_expand_workers):
        worker_name = f"expand_worker_{i}"
        status = worker_group.create_status(worker_name, group="scan")
        worker_group.add_task(
            asyncio.create_task(
                supervised_worker(
                    expand_worker,
                    worker_name,
                    state,
                    state.should_stop,
                    on_progress=on_expand_progress,
                    batch_size=expand_batch_size,
                    status_tracker=status,
                )
            )
        )

    # Score workers (group="score")
    for i in range(num_score_workers):
        worker_name = f"score_worker_{i}"
        status = worker_group.create_status(worker_name, group="score")
        worker_group.add_task(
            asyncio.create_task(
                supervised_worker(
                    score_worker,
                    worker_name,
                    state,
                    state.should_stop,
                    on_progress=on_score_progress,
                    batch_size=score_batch_size,
                    status_tracker=status,
                )
            )
        )

    # Enrich workers (group="enrich")
    for i in range(num_enrich_workers):
        worker_name = f"enrich_worker_{i}"
        status = worker_group.create_status(worker_name, group="enrich")
        worker_group.add_task(
            asyncio.create_task(
                supervised_worker(
                    enrich_worker,
                    worker_name,
                    state,
                    state.should_stop,
                    on_progress=on_enrich_progress,
                    batch_size=enrich_batch_size,
                    status_tracker=status,
                )
            )
        )

    # Refine workers (group="score" — same pipeline stage)
    for i in range(num_refine_workers):
        worker_name = f"refine_worker_{i}"
        status = worker_group.create_status(worker_name, group="score")
        worker_group.add_task(
            asyncio.create_task(
                supervised_worker(
                    refine_worker,
                    worker_name,
                    state,
                    state.should_stop,
                    on_progress=on_refine_progress,
                    batch_size=refine_batch_size,
                    status_tracker=status,
                )
            )
        )

    # Embed description worker (group="scan" — lightweight I/O)
    from imas_codex.discovery.base.embed_worker import embed_description_worker

    embed_status = worker_group.create_status("embed_worker", group="scan")
    worker_group.add_task(
        asyncio.create_task(
            supervised_worker(
                embed_description_worker,
                "embed_worker",
                state,
                state.should_stop,
                labels=["FacilityPath"],
                status_tracker=embed_status,
            )
        )
    )

    logger.info(
        f"Started {worker_group.get_active_count()} supervised workers: "
        f"scan={num_scan_workers}, expand={num_expand_workers}, "
        f"score={num_score_workers}, enrich={num_enrich_workers}, "
        f"refine={num_refine_workers}, embed=1"
    )

    # Periodic orphan recovery during discovery (every 60s)
    orphan_tick = make_orphan_recovery_tick(
        facility,
        [OrphanRecoverySpec("FacilityPath", timeout_seconds=CLAIM_TIMEOUT_SECONDS)],
    )

    # Run supervision loop — handles status updates and clean shutdown
    await run_supervised_loop(
        worker_group,
        state.should_stop,
        on_worker_status=on_worker_status,
        on_tick=orphan_tick,
        shutdown_timeout=graceful_shutdown_timeout,
    )

    # Determine why we stopped for logging
    if state.budget_exhausted:
        logger.info("Budget limit reached")
    elif state.path_limit_reached:
        logger.info("Path limit reached")
    elif state.stop_requested:
        logger.info("Stop requested")
    else:
        logger.info("Discovery complete (no pending work)")

    state.stop_requested = True

    # Graceful shutdown: reset any in-progress claims for next run
    reset_count = reset_orphaned_claims(facility, silent=True)
    if reset_count:
        logger.info(f"Shutdown cleanup: {reset_count} claimed paths reset")

    # Auto-normalize scores after scoring completes
    if state.score_stats.processed > 0:
        from imas_codex.discovery.paths.frontier import normalize_scores

        normalize_scores(facility)

    # Force garbage collection while the event loop is still running so that
    # orphaned asyncio subprocess transports are cleaned up before the loop
    # closes.  Without this, BaseSubprocessTransport.__del__ fires after the
    # loop is closed and prints harmless but noisy RuntimeError tracebacks.
    import gc

    gc.collect()

    elapsed = max(
        state.scan_stats.elapsed,
        state.expand_stats.elapsed,
        state.score_stats.elapsed,
        state.enrich_stats.elapsed,
        state.refine_stats.elapsed,
    )

    return {
        "scanned": state.scan_stats.processed,
        "expanded": state.expand_stats.processed,
        "scored": state.score_stats.processed,
        "enriched": state.enrich_stats.processed,
        "refined": state.refine_stats.processed,
        "cost": state.total_cost,
        "elapsed_seconds": elapsed,
        "scan_rate": state.scan_stats.rate,
        "expand_rate": state.expand_stats.rate,
        "score_rate": state.score_stats.rate,
        "enrich_rate": state.enrich_stats.rate,
        "refine_rate": state.refine_stats.rate,
        "scan_errors": state.scan_stats.errors,
        "expand_errors": state.expand_stats.errors,
        "score_errors": state.score_stats.errors,
        "enrich_errors": state.enrich_stats.errors,
        "refine_errors": state.refine_stats.errors,
    }
