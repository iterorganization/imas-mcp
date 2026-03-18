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

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.discovery.base.llm import ProviderBudgetExhausted
from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import (
    OrphanRecoverySpec,
    PipelinePhase,
    is_infrastructure_error,
)
from imas_codex.discovery.paths.models import ScoreBatch
from imas_codex.graph.models import PathStatus, TerminalReason

if TYPE_CHECKING:
    from collections.abc import Callable

    from imas_codex.discovery.base.supervision import SupervisedWorkerGroup
    from imas_codex.remote.ssh_worker import SSHWorkerPool

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryState(DiscoveryStateBase):
    """Shared state for parallel discovery."""

    path_limit: int | None = None
    focus: str | None = None
    threshold: float | None = None  # Uses get_discovery_threshold() when None
    root_filter: list[str] | None = None  # Restrict work to these roots
    auto_enrich_threshold: float | None = None  # Also enrich paths scoring >= this

    # Worker stats
    scan_stats: WorkerStats = field(default_factory=WorkerStats)
    expand_stats: WorkerStats = field(default_factory=WorkerStats)
    triage_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    score_stats: WorkerStats = field(default_factory=WorkerStats)
    dedup_stats: WorkerStats = field(default_factory=WorkerStats)
    user_stats: WorkerStats = field(default_factory=WorkerStats)

    # Control
    provider_budget_exhausted: bool = False  # API key credit limit hit (402)

    # Pipeline phases (initialized in __post_init__)
    scan_phase: PipelinePhase = field(init=False)
    expand_phase: PipelinePhase = field(init=False)
    triage_phase: PipelinePhase = field(init=False)
    enrich_phase: PipelinePhase = field(init=False)
    score_phase: PipelinePhase = field(init=False)
    user_phase: PipelinePhase = field(init=False)

    def __post_init__(self) -> None:
        self.scan_phase = PipelinePhase("scan")
        self.expand_phase = PipelinePhase("expand")
        self.triage_phase = PipelinePhase("triage")
        self.enrich_phase = PipelinePhase("enrich")
        self.score_phase = PipelinePhase("score")
        self.user_phase = PipelinePhase("user")

    # SSH retry tracking for exponential backoff
    ssh_retry_count: int = 0
    max_ssh_retries: int = 5
    ssh_error_message: str | None = None

    # Session tracking for --path-limit
    initial_terminal_count: int | None = None

    @property
    def total_cost(self) -> float:
        return self.triage_stats.cost + self.score_stats.cost

    @property
    def total_processed(self) -> int:
        return self.scan_stats.processed + self.score_stats.processed

    @property
    def terminal_count(self) -> int:
        """Count of paths in terminal states (triaged or scored, not pending work).

        For --path-limit purposes, count paths that have completed their pipeline:
        - triaged: below threshold, terminal after 1st pass
        - scored: completed 2nd pass scoring with enrichment evidence
        Both must not be awaiting expansion or enrichment.
        """
        from imas_codex.graph import GraphClient

        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
                WHERE p.status IN [$triaged, $scored]
                  AND (p.should_expand = false OR p.expanded_at IS NOT NULL)
                  AND (p.should_enrich = false OR p.is_enriched = true)
                RETURN count(p) AS terminal_count
                """,
                facility=self.facility,
                triaged=PathStatus.triaged.value,
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
        return super().budget_exhausted or self.provider_budget_exhausted

    @property
    def path_limit_reached(self) -> bool:
        """Check if path limit reached using session terminal count.

        Uses paths completed in THIS SESSION, not cumulative graph total.
        E.g., with 28 existing paths and --path-limit 30, we process 30 more.
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
        if super().should_stop():
            return True
        if self.budget_exhausted:
            return True
        if self.path_limit_reached:
            return True
        # All phases must be done (idle + no graph work)
        all_done = (
            self.scan_phase.done
            and self.expand_phase.done
            and self.triage_phase.done
            and self.enrich_phase.done
            and self.score_phase.done
        )
        if all_done:
            # Final confirmation: no pending work at all
            if has_pending_work(self.facility):
                # Graph has work — reset all phases so workers re-poll
                self.scan_phase.reset()
                self.expand_phase.reset()
                self.triage_phase.reset()
                self.enrich_phase.reset()
                self.score_phase.reset()
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
    - Triaged paths with should_expand=true that haven't been expanded yet
    - Triaged paths with should_enrich=true that haven't been enriched yet
    - Enriched paths that haven't been scored yet

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
                 CASE WHEN p.status = $discovered AND p.triage_composite IS NULL
                      THEN 'discovered' ELSE null END AS disc,
                 CASE WHEN p.status = $scanned AND p.triage_composite IS NULL
                      THEN 'scanned' ELSE null END AS scn,
                 CASE WHEN p.status = $triaged AND p.should_expand = true
                      AND p.expanded_at IS NULL
                      THEN 'expand' ELSE null END AS exp,
                 CASE WHEN p.status = $triaged AND p.should_enrich = true
                      AND p.triage_composite >= 0.15
                      AND (p.is_enriched IS NULL OR p.is_enriched = false)
                      THEN 'enrich' ELSE null END AS enr,
                 CASE WHEN p.is_enriched = true
                      AND p.scored_at IS NULL
                      THEN 'score' ELSE null END AS rsc
            WHERE disc IS NOT NULL OR scn IS NOT NULL OR exp IS NOT NULL
                  OR enr IS NOT NULL OR rsc IS NOT NULL
            RETURN count(p) AS pending,
                   count(disc) AS pending_discovered,
                   count(scn) AS pending_scanned,
                   count(exp) AS pending_expand,
                   count(enr) AS pending_enrich,
                   count(rsc) AS pending_score
            """,
            facility=facility,
            discovered=PathStatus.discovered.value,
            scanned=PathStatus.scanned.value,
            triaged=PathStatus.triaged.value,
        )
        if result:
            pending = result[0]["pending"]
            if pending > 0:
                logger.debug(
                    f"Pending work: discovered={result[0]['pending_discovered']}, "
                    f"scanned={result[0]['pending_scanned']}, "
                    f"expand={result[0]['pending_expand']}, "
                    f"enrich={result[0]['pending_enrich']}, "
                    f"score={result[0]['pending_score']}"
                )
            return pending > 0
        return False


# ============================================================================
# Per-phase pending work checks (lightweight, targeted graph queries)
# ============================================================================


def _has_pending_scan_work(facility: str) -> bool:
    """Check if there are discovered paths awaiting scanning."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE p.status = $discovered AND p.triage_composite IS NULL
            RETURN count(p) > 0 AS has_work
            """,
            facility=facility,
            discovered=PathStatus.discovered.value,
        )
        return bool(result and result[0]["has_work"])


def _has_pending_expand_work(facility: str) -> bool:
    """Check if there are triaged paths awaiting expansion."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE p.status = $triaged
              AND p.should_expand = true
              AND p.expanded_at IS NULL
            RETURN count(p) > 0 AS has_work
            """,
            facility=facility,
            triaged=PathStatus.triaged.value,
        )
        return bool(result and result[0]["has_work"])


def _has_pending_triage_work(facility: str) -> bool:
    """Check if there are scanned paths awaiting triage."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE p.status = $scanned AND p.triage_composite IS NULL
            RETURN count(p) > 0 AS has_work
            """,
            facility=facility,
            scanned=PathStatus.scanned.value,
        )
        return bool(result and result[0]["has_work"])


def _has_pending_enrich_work(facility: str) -> bool:
    """Check if there are triaged paths awaiting enrichment."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE p.status = $triaged
              AND p.should_enrich = true
              AND p.triage_composite >= 0.15
              AND (p.is_enriched IS NULL OR p.is_enriched = false)
            RETURN count(p) > 0 AS has_work
            """,
            facility=facility,
            triaged=PathStatus.triaged.value,
        )
        return bool(result and result[0]["has_work"])


def _has_pending_score_work(facility: str) -> bool:
    """Check if there are enriched paths awaiting scoring (2nd pass).

    Matches the same criteria as claim_paths_for_scoring:
    all enriched paths that haven't been scored yet.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE p.is_enriched = true
              AND p.scored_at IS NULL
            RETURN count(p) > 0 AS has_work
            """,
            facility=facility,
        )
        return bool(result and result[0]["has_work"])


def _has_pending_user_work(facility: str) -> bool:
    """Check if there are FacilityUser nodes without Person links.

    Users are created by persist_scan_results() during scanning.  Person
    linking (ORCID lookup) is handled asynchronously by user_worker.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (u:FacilityUser)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE u.person_id IS NULL
              AND (u.claimed_at IS NULL
                   OR u.claimed_at < datetime() - duration($cutoff))
            RETURN count(u) > 0 AS has_work
            """,
            facility=facility,
            cutoff=f"PT{CLAIM_TIMEOUT_SECONDS}S",
        )
        return bool(result and result[0]["has_work"])


@retry_on_deadlock()
def claim_users_for_linking(facility: str, limit: int = 10) -> list[dict[str, Any]]:
    """Atomically claim FacilityUser nodes for Person linking (ORCID).

    Claims users where person_id IS NULL (no Person link yet).
    Uses claimed_at with ORDER BY rand() for orphan recovery.

    Args:
        facility: Facility ID
        limit: Maximum users to claim per batch
    """
    import uuid as _uuid

    from imas_codex.graph import GraphClient

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(_uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (u:FacilityUser)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE u.person_id IS NULL
              AND (u.claimed_at IS NULL
                   OR u.claimed_at < datetime() - duration($cutoff))
            WITH u ORDER BY rand() LIMIT $limit
            SET u.claimed_at = datetime(), u.claim_token = $token
            """,
            facility=facility,
            limit=limit,
            cutoff=cutoff,
            token=claim_token,
        )
        result = gc.query(
            """
            MATCH (u:FacilityUser {claim_token: $token})-[:AT_FACILITY]->(f:Facility {id: $facility})
            RETURN u.id AS id, u.username AS username,
                   u.name AS name, u.given_name AS given_name,
                   u.family_name AS family_name, u.email AS email
            """,
            facility=facility,
            token=claim_token,
        )
        return list(result)


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


@retry_on_deadlock()
def claim_paths_for_scanning(
    facility: str, limit: int = 50, root_filter: list[str] | None = None
) -> list[dict[str, Any]]:
    """Atomically claim discovered paths for initial scanning.

    Claims only unscored discovered paths (first scan, enumerate only).
    Uses claim_token pattern with ORDER BY rand() to prevent deadlocks.

    Args:
        facility: Facility ID
        limit: Maximum paths to claim
        root_filter: If set, only claim paths under these roots
    """
    import uuid as _uuid

    from imas_codex.graph import GraphClient

    root_clause = ""
    if root_filter:
        root_clause = "AND any(root IN $root_filter WHERE p.path STARTS WITH root)"

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(_uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            f"""
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.status = $discovered AND p.triage_composite IS NULL
              AND (p.claimed_at IS NULL OR p.claimed_at < datetime() - duration($cutoff))
            {root_clause}
            WITH p ORDER BY rand() LIMIT $limit
            SET p.claimed_at = datetime(), p.claim_token = $token
            """,
            facility=facility,
            limit=limit,
            discovered=PathStatus.discovered.value,
            cutoff=cutoff,
            root_filter=root_filter or [],
            token=claim_token,
        )
        result = gc.query(
            f"""
            MATCH (p:FacilityPath {{claim_token: $token}})-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.status = $discovered
            {root_clause}
            RETURN p.id AS id, p.path AS path, p.depth AS depth, false AS is_expanding
            """,
            facility=facility,
            discovered=PathStatus.discovered.value,
            root_filter=root_filter or [],
            token=claim_token,
        )
        return list(result)


@retry_on_deadlock()
def claim_paths_for_expanding(
    facility: str, limit: int = 50, root_filter: list[str] | None = None
) -> list[dict[str, Any]]:
    """Atomically claim triaged paths for expansion scanning.

    Claims paths with should_expand=true that haven't been expanded yet.
    Uses claim_token pattern with ORDER BY rand() to prevent deadlocks.

    Args:
        facility: Facility ID
        limit: Maximum paths to claim
        root_filter: If set, only claim paths under these roots
    """
    import uuid as _uuid

    from imas_codex.graph import GraphClient

    root_clause = ""
    if root_filter:
        root_clause = "AND any(root IN $root_filter WHERE p.path STARTS WITH root)"

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(_uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            f"""
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.status = $triaged
              AND p.should_expand = true
              AND p.expanded_at IS NULL
              AND (p.claimed_at IS NULL OR p.claimed_at < datetime() - duration($cutoff))
            {root_clause}
            WITH p ORDER BY rand() LIMIT $limit
            SET p.claimed_at = datetime(), p.claim_token = $token
            """,
            facility=facility,
            limit=limit,
            triaged=PathStatus.triaged.value,
            cutoff=cutoff,
            root_filter=root_filter or [],
            token=claim_token,
        )
        result = gc.query(
            f"""
            MATCH (p:FacilityPath {{claim_token: $token}})-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.status = $triaged AND p.should_expand = true
            {root_clause}
            RETURN p.id AS id, p.path AS path, p.depth AS depth, true AS is_expanding
            """,
            facility=facility,
            triaged=PathStatus.triaged.value,
            root_filter=root_filter or [],
            token=claim_token,
        )
        return list(result)


@retry_on_deadlock()
def claim_paths_for_triaging(
    facility: str, limit: int = 25, root_filter: list[str] | None = None
) -> list[dict[str, Any]]:
    """Atomically claim scanned paths for triage.

    Uses claim_token pattern with ORDER BY rand() to prevent deadlocks.

    Args:
        facility: Facility ID
        limit: Maximum paths to claim
        root_filter: If set, only claim paths under these roots
    """
    import uuid as _uuid

    from imas_codex.graph import GraphClient

    root_clause = ""
    if root_filter:
        root_clause = "AND any(root IN $root_filter WHERE p.path STARTS WITH root)"

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(_uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            f"""
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.status = $scanned AND p.triage_composite IS NULL
              AND (p.claimed_at IS NULL OR p.claimed_at < datetime() - duration($cutoff))
            {root_clause}
            WITH p ORDER BY rand() LIMIT $limit
            SET p.claimed_at = datetime(), p.claim_token = $token
            """,
            facility=facility,
            limit=limit,
            scanned=PathStatus.scanned.value,
            cutoff=cutoff,
            root_filter=root_filter or [],
            token=claim_token,
        )
        result = gc.query(
            f"""
            MATCH (p:FacilityPath {{claim_token: $token}})-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.status = $scanned
            {root_clause}
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
            scanned=PathStatus.scanned.value,
            root_filter=root_filter or [],
            token=claim_token,
        )
        return list(result)


async def mark_scan_complete(
    facility: str,
    scan_results: list[tuple[str, dict, list[dict], str | None, bool]],
    excluded: list[tuple[str, str, str]] | None = None,
) -> dict[str, int]:
    """Mark scanned paths complete and conditionally create children.

    Runs persist_scan_results in a thread executor to avoid blocking the
    event loop — it performs synchronous SSH, HTTP, and Neo4j operations.

    Transition: listing → listed (first scan) or scored (expansion scan)

    Args:
        facility: Facility ID
        scan_results: List of (path, stats_dict, child_dirs, error, is_expanding) tuples.
                      child_dirs is list of {path, is_symlink, realpath, device_inode} dicts.
        excluded: Optional list of (path, parent_path, reason) for excluded dirs
    """
    from imas_codex.discovery.paths.frontier import persist_scan_results

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: persist_scan_results(facility, scan_results, excluded=excluded),
    )


def mark_triage_complete(
    facility: str,
    score_data: list[dict[str, Any]],
) -> int:
    """Apply structural expansion overrides and persist triaged paths.

    Transition: scanned → triaged
    """
    from imas_codex.discovery.paths.frontier import (
        apply_expansion_overrides,
        mark_paths_triaged,
    )

    apply_expansion_overrides(facility, score_data)
    return mark_paths_triaged(facility, score_data)


@retry_on_deadlock()
def claim_paths_for_enriching(
    facility: str,
    limit: int = 25,
    root_filter: list[str] | None = None,
    auto_enrich_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Atomically claim triaged paths for enrichment.

    Claims paths where:
    - status = 'triaged'
    - should_enrich = true OR triage_composite >= auto_enrich_threshold
    - is_enriched is null or false

    Uses claim_token pattern with ORDER BY rand() to prevent deadlocks.

    Args:
        facility: Facility ID
        limit: Maximum paths to claim
        root_filter: If set, only claim paths under these roots
        auto_enrich_threshold: If set, also claim paths with score >= threshold
    """
    import uuid as _uuid

    from imas_codex.graph import GraphClient

    root_clause = ""
    if root_filter:
        root_clause = "AND any(root IN $root_filter WHERE p.path STARTS WITH root)"

    min_enrich_score = 0.15
    if auto_enrich_threshold is not None:
        enrich_clause = (
            "(p.should_enrich = true OR p.triage_composite >= $auto_enrich_threshold)"
        )
    else:
        enrich_clause = "p.should_enrich = true"

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(_uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            f"""
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.status = $triaged
              AND {enrich_clause}
              AND p.triage_composite >= $min_enrich_score
              AND (p.is_enriched IS NULL OR p.is_enriched = false)
              AND (p.claimed_at IS NULL OR p.claimed_at < datetime() - duration($cutoff))
              AND (p.total_dirs IS NULL OR p.total_dirs <= 500)
            {root_clause}
            WITH p ORDER BY rand() LIMIT $limit
            SET p.claimed_at = datetime(), p.claim_token = $token
            """,
            facility=facility,
            limit=limit,
            triaged=PathStatus.triaged.value,
            cutoff=cutoff,
            root_filter=root_filter or [],
            auto_enrich_threshold=auto_enrich_threshold,
            min_enrich_score=min_enrich_score,
            token=claim_token,
        )
        result = gc.query(
            f"""
            MATCH (p:FacilityPath {{claim_token: $token}})-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.status = $triaged
            {root_clause}
            RETURN p.id AS id, p.path AS path, p.depth AS depth, p.triage_composite AS triage_composite,
                   p.path_purpose AS path_purpose
            """,
            facility=facility,
            triaged=PathStatus.triaged.value,
            root_filter=root_filter or [],
            token=claim_token,
        )
        return list(result)


@retry_on_deadlock()
def claim_paths_for_scoring(
    facility: str,
    limit: int = 10,
    root_filter: list[str] | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """Atomically claim enriched paths for 2nd-pass scoring.

    Claims paths where:
    - is_enriched = true
    - scored_at is null

    Uses claim_token pattern with ORDER BY rand() to prevent deadlocks.

    Args:
        facility: Facility ID
        limit: Maximum paths to claim
        root_filter: If set, only claim paths under these roots
        min_score: Minimum triage_composite for scoring (default: 0.0 = score all)
    """
    import uuid as _uuid

    from imas_codex.graph import GraphClient

    threshold = min_score if min_score is not None else 0.0

    root_clause = ""
    if root_filter:
        root_clause = "AND any(root IN $root_filter WHERE p.path STARTS WITH root)"

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(_uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            f"""
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.is_enriched = true
              AND p.triage_composite >= $min_score
              AND p.scored_at IS NULL
              AND (p.claimed_at IS NULL OR p.claimed_at < datetime() - duration($cutoff))
            {root_clause}
            WITH p ORDER BY rand() LIMIT $limit
            SET p.claimed_at = datetime(), p.claim_token = $token
            """,
            facility=facility,
            limit=limit,
            cutoff=cutoff,
            root_filter=root_filter or [],
            min_score=threshold,
            token=claim_token,
        )
        result = gc.query(
            f"""
            MATCH (p:FacilityPath {{claim_token: $token}})-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE p.is_enriched = true
            {root_clause}
            RETURN p.id AS id, p.path AS path, p.depth AS depth,
                   p.triage_composite AS triage_composite,
                   p.triage_modeling_code AS triage_modeling_code,
                   p.triage_analysis_code AS triage_analysis_code,
                   p.triage_operations_code AS triage_operations_code,
                   p.triage_modeling_data AS triage_modeling_data,
                   p.triage_experimental_data AS triage_experimental_data,
                   p.triage_data_access AS triage_data_access,
                   p.triage_workflow AS triage_workflow,
                   p.triage_visualization AS triage_visualization,
                   p.triage_documentation AS triage_documentation,
                   p.triage_imas AS triage_imas,
                   p.triage_convention AS triage_convention,
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
            root_filter=root_filter or [],
            token=claim_token,
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
    import json
    from datetime import UTC, datetime

    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()

    # Separate errors from successes for batch processing
    error_items: list[dict[str, Any]] = []
    success_items: list[dict[str, Any]] = []

    for result in enrichment_results:
        path_id = f"{facility}:{result['path']}"

        if result.get("error"):
            error_msg = result["error"][:200]
            permanent = error_msg in ("not a directory", "permission denied")
            error_items.append(
                {
                    "id": path_id,
                    "error": error_msg,
                    "permanent": permanent,
                }
            )
        else:
            lang_breakdown = result.get("language_breakdown")
            if isinstance(lang_breakdown, dict):
                lang_breakdown = json.dumps(lang_breakdown) if lang_breakdown else None

            pattern_cats = result.get("pattern_categories")
            if isinstance(pattern_cats, dict):
                pattern_cats = json.dumps(pattern_cats) if pattern_cats else None

            warnings = result.get("warnings", [])
            warn_str = ", ".join(warnings) if warnings else None

            success_items.append(
                {
                    "id": path_id,
                    "now": now,
                    "total_bytes": result.get("total_bytes"),
                    "total_lines": result.get("total_lines"),
                    "language_breakdown": lang_breakdown,
                    "is_multiformat": result.get("is_multiformat", False),
                    "pattern_categories": pattern_cats,
                    "read_matches": result.get("read_matches", 0),
                    "write_matches": result.get("write_matches", 0),
                    "enrich_warnings": warn_str,
                }
            )

    updated = 0
    with GraphClient() as gc:
        if error_items:
            gc.query(
                """
                UNWIND $items AS item
                MATCH (p:FacilityPath {id: item.id})
                SET p.claimed_at = null,
                    p.enrich_error = item.error,
                    p.should_enrich = CASE WHEN item.permanent
                        THEN false ELSE p.should_enrich END
                """,
                items=error_items,
            )

        if success_items:
            gc.query(
                """
                UNWIND $items AS item
                MATCH (p:FacilityPath {id: item.id})
                SET p.is_enriched = true,
                    p.enriched_at = item.now,
                    p.total_bytes = item.total_bytes,
                    p.total_lines = item.total_lines,
                    p.language_breakdown = item.language_breakdown,
                    p.is_multiformat = item.is_multiformat,
                    p.pattern_categories = item.pattern_categories,
                    p.read_matches = item.read_matches,
                    p.write_matches = item.write_matches,
                    p.enrich_warnings = item.enrich_warnings,
                    p.claimed_at = null
                """,
                items=success_items,
            )
            updated = len(success_items)

    return updated


def mark_score_complete(
    facility: str,
    score_results: list[dict[str, Any]],
) -> int:
    """Mark paths with scored results after 2nd-pass scoring.

    Persists adjusted scores and evidence from pattern matching.
    Sets status to 'scored'.

    Args:
        facility: Facility ID
        score_results: List of dicts with path, new scores, and evidence

    Returns:
        Number of paths updated
    """
    import json
    from datetime import UTC, datetime

    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()

    from imas_codex.discovery.paths.frontier import SCORE_DIMENSIONS

    # Build batch items with all dimensions included (null if absent)
    items: list[dict[str, Any]] = []
    child_skip_ids: list[dict[str, Any]] = []

    for result in score_results:
        path_id = f"{facility}:{result['path']}"

        primary_evidence = result.get("primary_evidence", [])
        if isinstance(primary_evidence, list):
            primary_evidence = json.dumps(primary_evidence)

        keywords = result.get("keywords", [])
        if isinstance(keywords, list):
            keywords = json.dumps(keywords)

        item: dict[str, Any] = {
            "id": path_id,
            "now": now,
            "score": result.get("score"),
            "score_cost": result.get("score_cost", 0.0),
            "scored": PathStatus.scored.value,
            "adjustment_reason": result.get("adjustment_reason", ""),
            "primary_evidence": primary_evidence,
            "evidence_summary": result.get("evidence_summary", ""),
            "description": result.get("description", ""),
            "keywords": keywords,
            "path_purpose": result.get("path_purpose"),
            "physics_domain": result.get("physics_domain"),
            "should_expand": result.get("should_expand", True),
        }
        for dim in SCORE_DIMENSIONS:
            item[dim] = result.get(dim)
        items.append(item)

        # Collect child-skip candidates
        path_purpose = result.get("path_purpose")
        should_expand = result.get("should_expand", True)
        if path_purpose in {"modeling_data", "experimental_data"} and not should_expand:
            child_skip_ids.append(
                {
                    "id": path_id,
                    "reason": f"parent_{path_purpose}",
                }
            )

    if not items:
        return 0

    updated = len(items)

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $items AS item
            MATCH (p:FacilityPath {id: item.id})
            SET p.status = item.scored,
                p.scored_at = item.now,
                p.score_composite = item.score,
                p.score_cost = coalesce(p.score_cost, 0) + item.score_cost,
                p.score_reason = item.adjustment_reason,
                p.primary_evidence = item.primary_evidence,
                p.evidence_summary = item.evidence_summary,
                p.description = item.description,
                p.keywords = item.keywords,
                p.path_purpose = item.path_purpose,
                p.physics_domain = item.physics_domain,
                p.should_expand = item.should_expand,
                p.claimed_at = null,
                p.score_modeling_code = coalesce(item.score_modeling_code, p.score_modeling_code),
                p.score_analysis_code = coalesce(item.score_analysis_code, p.score_analysis_code),
                p.score_operations_code = coalesce(item.score_operations_code, p.score_operations_code),
                p.score_modeling_data = coalesce(item.score_modeling_data, p.score_modeling_data),
                p.score_experimental_data = coalesce(item.score_experimental_data, p.score_experimental_data),
                p.score_data_access = coalesce(item.score_data_access, p.score_data_access),
                p.score_workflow = coalesce(item.score_workflow, p.score_workflow),
                p.score_visualization = coalesce(item.score_visualization, p.score_visualization),
                p.score_documentation = coalesce(item.score_documentation, p.score_documentation),
                p.score_imas = coalesce(item.score_imas, p.score_imas),
                p.score_convention = coalesce(item.score_convention, p.score_convention)
            """,
            items=items,
        )

        # Batch child-skip for data containers
        if child_skip_ids:
            gc.query(
                """
                UNWIND $parents AS parent
                MATCH (child:FacilityPath)-[:IN_DIRECTORY]->(p:FacilityPath {id: parent.id})
                WHERE child.status = 'discovered'
                SET child.status = $skipped,
                    child.skipped_at = $now,
                    child.terminal_reason = $terminal_reason,
                    child.skip_reason = parent.reason
                """,
                parents=child_skip_ids,
                now=now,
                terminal_reason=TerminalReason.parent_terminal.value,
                skipped=PathStatus.skipped.value,
            )

    return updated


# ============================================================================
# Deduplication helpers (graph-only, no SSH)
# ============================================================================


def _find_clone_groups(
    facility: str,
    limit: int = 50,
) -> list[tuple[str, str, list[dict]]]:
    """Find SoftwareRepos with multiple non-terminal paths at this facility.

    Returns a list of (repo_id, repo_name, active_paths) tuples where
    active_paths is sorted: canonical first (accessible remote > has remote
    URL > shallowest depth > earliest discovered).

    Args:
        facility: Facility ID
        limit: Maximum number of repos to process per call

    Returns:
        List of (repo_id, repo_name, sorted_paths) for repos with >1 clone.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (r:SoftwareRepo)
            MATCH (p:FacilityPath)-[:INSTANCE_OF]->(r)
            WHERE p.status IN [$scanned, $scored]
              AND p.terminal_reason IS NULL
            WITH r,
                 collect({
                     id: p.id,
                     path: p.path,
                     depth: coalesce(p.depth, 99),
                     accessible: coalesce(p.vcs_remote_accessible, false),
                     has_remote: CASE WHEN p.vcs_remote_url IS NOT NULL THEN 1 ELSE 0 END,
                     discovered_at: coalesce(toString(p.discovered_at), "")
                 }) AS active_paths
            WHERE size(active_paths) > 1
               OR (size(active_paths) = 1
                   AND size([p IN active_paths WHERE p.accessible]) > 0)
            RETURN r.id AS repo_id,
                   coalesce(r.name, split(r.id, "/")[-1]) AS repo_name,
                   active_paths
            ORDER BY size(active_paths) DESC
            LIMIT $limit
            """,
            scanned=PathStatus.scanned.value,
            scored=PathStatus.scored.value,
            limit=limit,
        )

    groups = []
    for row in rows or []:
        # Sort Python-side: canonical = accessible remote > has remote > shallowest > earliest
        paths = sorted(
            row["active_paths"],
            key=lambda p: (
                0 if p["accessible"] else 1,
                1 - p["has_remote"],
                p["depth"],
                p["discovered_at"],
            ),
        )
        groups.append((row["repo_id"], row["repo_name"], paths))
    return groups


def _mark_clones_terminal(
    clone_ids: list[str],
    canonical_path: str,
    repo_id: str,
) -> int:
    """Mark non-canonical clone paths as terminal (clone).

    Only marks paths that are not already terminal, not currently claimed
    by another worker, and are in scanned or scored status.

    Args:
        clone_ids: FacilityPath IDs to mark terminal
        canonical_path: Path of the canonical instance (for skip_reason)
        repo_id: SoftwareRepo ID (for skip_reason)

    Returns:
        Number of paths actually marked.
    """
    from datetime import UTC, datetime

    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    skip_reason = f"Clone of {canonical_path} ({repo_id})"

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $clone_ids AS clone_id
            MATCH (p:FacilityPath {id: clone_id})
            WHERE p.terminal_reason IS NULL
              AND p.status IN [$scanned, $scored]
              AND p.claimed_at IS NULL
            SET p.status = $skipped,
                p.terminal_reason = $reason,
                p.skip_reason = $skip_reason,
                p.skipped_at = $now,
                p.should_expand = false,
                p.should_enrich = false
            RETURN count(p) AS marked
            """,
            clone_ids=clone_ids,
            scanned=PathStatus.scanned.value,
            scored=PathStatus.scored.value,
            skipped=PathStatus.skipped.value,
            reason=TerminalReason.clone.value,
            skip_reason=skip_reason,
            now=now,
        )
    return result[0]["marked"] if result else 0


def _mark_accessible_elsewhere(
    path_id: str,
    repo_id: str,
) -> bool:
    """Mark a path as terminal because its repo is externally accessible.

    Used by dedup_worker when the canonical clone's repo is publicly
    accessible — no need to scan/ingest locally.

    Args:
        path_id: FacilityPath ID to mark terminal
        repo_id: SoftwareRepo ID (for skip_reason)

    Returns:
        True if the path was marked.
    """
    from datetime import UTC, datetime

    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    skip_reason = f"Repo accessible elsewhere ({repo_id})"

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath {id: $path_id})
            WHERE p.terminal_reason IS NULL
              AND p.status IN [$scanned, $scored]
              AND p.claimed_at IS NULL
            SET p.status = $skipped,
                p.terminal_reason = $reason,
                p.skip_reason = $skip_reason,
                p.skipped_at = $now,
                p.should_expand = false,
                p.should_enrich = false
            RETURN count(p) AS marked
            """,
            path_id=path_id,
            scanned=PathStatus.scanned.value,
            scored=PathStatus.scored.value,
            skipped=PathStatus.skipped.value,
            reason=TerminalReason.accessible_elsewhere.value,
            skip_reason=skip_reason,
            now=now,
        )
    return bool(result and result[0]["marked"] > 0)


# ============================================================================
# Async Workers
# ============================================================================


async def scan_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[str] | None, list[dict] | None], None]
    | None = None,
    batch_size: int = 5,
    pool: SSHWorkerPool | None = None,
) -> None:
    """Async scanner worker.

    Continuously claims pending paths, scans via SSH, marks complete.
    Runs until stop_requested or no more pending paths.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per SSH call (default 5)
        pool: Optional persistent SSH worker pool for session reuse
    """
    from imas_codex.discovery.paths.scanner import async_scan_paths

    loop = asyncio.get_running_loop()

    while not state.should_stop():
        # Claim work from graph (run in executor to avoid blocking event loop)
        paths = await loop.run_in_executor(
            None,
            lambda: claim_paths_for_scanning(
                state.facility, limit=batch_size, root_filter=state.root_filter
            ),
        )

        if not paths:
            state.scan_phase.record_idle()
            state.scan_stats.mark_idle()
            if on_progress:
                on_progress("idle", state.scan_stats, None, None)
            # Wait before polling again
            await asyncio.sleep(1.0)
            continue

        state.scan_phase.record_activity(len(paths))
        state.scan_stats.mark_active()
        path_strs = [p["path"] for p in paths]

        if on_progress:
            on_progress(f"scanning {len(paths)} paths", state.scan_stats, None, None)

        # Async SSH scan — fully cancellable, no thread executor
        start = time.time()
        try:
            results = await async_scan_paths(
                state.facility,
                path_strs,
                enable_rg=False,
                enable_size=False,
                enable_git_metadata=True,
                enable_vcs_remote_check=True,
                pool=pool,
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
            _revert_path_claims(state.facility, path_strs)
            state.scan_stats.errors += len(paths)

            if state.ssh_retry_count >= state.max_ssh_retries:
                logger.error(
                    f"SSH connection to {state.facility} failed after "
                    f"{state.max_ssh_retries} attempts. "
                    f"Stopping discovery (rerun to retry)."
                )
                # Stop all workers. Without SSH, discovered paths remain
                # unscanned and has_pending_work() would loop forever
                # resetting phases. The process is reentrant — rerun to
                # continue from where it left off.
                state.stop_requested = True
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
        state.scan_stats.record_batch(stats["scanned"])

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


async def dedup_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[dict] | None], None] | None = None,
    batch_size: int = 50,
) -> None:
    """Async deduplication worker — graph-only, no SSH, no LLM cost.

    Identifies FacilityPaths that are clones of the same upstream repository
    by grouping on shared SoftwareRepo nodes (identity = remote_url or
    git_root_commit).  For each clone group, the canonical instance is
    retained (accessible remote > has remote URL > shallowest depth >
    earliest discovered) and all others are marked terminal with
    terminal_reason=clone, status=skipped, should_expand=false,
    should_enrich=false.

    This blocks ALL downstream work on duplicate paths:
    - Paths scoring (claim_paths_for_scoring requires status=scanned)
    - Path expansion (claim_paths_for_expanding requires status=scored)
    - Path enrichment (claim_paths_for_enriching requires status=scored)
    - Score phase (claim_paths_for_scoring requires is_enriched=true)
    - Code file scanning (claim_paths_for_file_scan requires status=scored)
    - Ingestion pipeline (queue_files requires scored FacilityPath parent)

    The worker runs entirely against the graph — it fires a Cypher query per
    cycle, marks clones terminal, and sleeps briefly.  When no clone groups
    remain it backs off to 5 s polls and eventually idles.

    Args:
        state: Shared discovery state
        on_progress: Progress callback receiving (message, stats, results)
        batch_size: Maximum SoftwareRepo groups to process per cycle
    """
    loop = asyncio.get_running_loop()
    idle_sleep = 5.0  # longer sleep when no work found

    while not state.should_stop():
        groups = await loop.run_in_executor(
            None,
            lambda: _find_clone_groups(state.facility, batch_size),
        )

        if not groups:
            state.dedup_stats.mark_idle()
            if on_progress:
                on_progress("waiting for clone groups", state.dedup_stats, None)
            await asyncio.sleep(idle_sleep)
            continue

        state.dedup_stats.mark_active()
        results = []

        for repo_id, repo_name, active_paths in groups:
            canonical = active_paths[0]
            clones = active_paths[1:]
            clone_ids = [p["id"] for p in clones]

            marked = await loop.run_in_executor(
                None,
                lambda cids=clone_ids, cp=canonical["path"], rid=repo_id: (
                    _mark_clones_terminal(cids, cp, rid)
                ),
            )

            # If the canonical instance's repo is externally accessible,
            # mark it too — no need to scan/ingest locally when the code
            # is available from a public source.
            canonical_also_skipped = False
            if canonical["accessible"]:
                canonical_also_skipped = await loop.run_in_executor(
                    None,
                    lambda pid=canonical["id"], rid=repo_id: _mark_accessible_elsewhere(
                        pid, rid
                    ),
                )
                if canonical_also_skipped:
                    marked += 1

            if marked > 0:
                state.dedup_stats.processed += marked
                state.dedup_stats.record_batch(marked)
                results.append(
                    {
                        "repo_id": repo_id,
                        "repo_name": repo_name,
                        "canonical_path": canonical["path"],
                        "clones_marked": marked,
                        "total_clones": len(active_paths),
                        "canonical_skipped": canonical_also_skipped,
                    }
                )
                logger.info(
                    "Dedup: marked %d paths of %s terminal (canonical: %s%s)",
                    marked,
                    repo_name,
                    canonical["path"],
                    ", also skipped (accessible)" if canonical_also_skipped else "",
                )

        if results:
            if on_progress:
                on_progress(
                    f"deduped {state.dedup_stats.processed}",
                    state.dedup_stats,
                    results,
                )

        # Short yield to not starve other workers
        await asyncio.sleep(0.5)


async def expand_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[str] | None, list[dict] | None], None]
    | None = None,
    batch_size: int = 10,
    pool: SSHWorkerPool | None = None,
) -> None:
    """Async expansion worker.

    Expands scored high-value paths by enumerating their children.
    Runs independently of scan_worker, claiming paths with should_expand=true.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per SSH call (default 10)
        pool: Optional persistent SSH worker pool for session reuse
    """
    from imas_codex.discovery.paths.scanner import async_scan_paths

    loop = asyncio.get_running_loop()

    while not state.should_stop():
        # Claim expansion work from graph (run in executor to avoid blocking event loop)
        paths = await loop.run_in_executor(
            None,
            lambda: claim_paths_for_expanding(
                state.facility, limit=batch_size, root_filter=state.root_filter
            ),
        )

        if not paths:
            state.expand_phase.record_idle()
            state.expand_stats.mark_idle()
            if on_progress:
                on_progress("idle", state.expand_stats, None, None)
            # Wait before polling again
            await asyncio.sleep(1.0)
            continue

        state.expand_phase.record_activity(len(paths))
        state.expand_stats.mark_active()
        path_strs = [p["path"] for p in paths]

        if on_progress:
            on_progress(f"expanding {len(paths)} paths", state.expand_stats, None, None)

        # Async SSH scan — fully cancellable, no thread executor
        start = time.time()
        try:
            results = await async_scan_paths(
                state.facility,
                path_strs,
                enable_rg=False,
                enable_size=False,
                enable_git_metadata=True,
                enable_vcs_remote_check=True,
                pool=pool,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            # Transient failure - revert paths for retry
            logger.warning(f"Expand SSH failure: {e}")
            _revert_path_claims(state.facility, path_strs)
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
        state.expand_stats.record_batch(stats["scanned"])

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


async def triage_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[dict] | None], None] | None = None,
    batch_size: int = 25,
) -> None:
    """Async triage worker.

    Continuously claims scanned paths, triages via LLM, marks complete.
    Runs until stop_requested, budget exhausted, or no more scanned paths.

    Optimization: Empty directories (total_files=0 AND total_dirs=0) are
    auto-skipped without LLM call since they have no content to evaluate.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per LLM call (default 25)
    """
    from imas_codex.discovery.paths.scorer import DirectoryTriager

    scorer = DirectoryTriager(facility=state.facility)
    loop = asyncio.get_running_loop()

    while not state.should_stop():
        # Check budget before claiming work
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.triage_stats, None)
            break

        # Claim work from graph (run in executor to avoid blocking event loop)
        paths = await loop.run_in_executor(
            None,
            lambda: claim_paths_for_triaging(
                state.facility, limit=batch_size, root_filter=state.root_filter
            ),
        )

        if not paths:
            state.triage_phase.record_idle()
            state.triage_stats.mark_idle()
            if state.triage_phase.idle_count <= 3:
                logger.debug(
                    f"Triage worker idle ({state.triage_phase.idle_count}), "
                    f"scan_processed={state.scan_stats.processed}"
                )
            if on_progress:
                on_progress("waiting for scanned paths", state.triage_stats, None)
            # Wait before polling again
            await asyncio.sleep(2.0)
            continue

        state.triage_phase.record_activity(len(paths))
        state.triage_stats.mark_active()
        logger.debug(f"Triage worker claimed {len(paths)} paths")

        # Split paths into auto-skip categories and paths needing LLM
        empty_paths = []
        accessible_vcs_paths = []
        paths_to_score = []

        from imas_codex.config.discovery_config import is_repo_accessible_elsewhere

        for p in paths:
            total_files = p.get("total_files", 0) or 0
            total_dirs = p.get("total_dirs", 0) or 0
            if total_files == 0 and total_dirs == 0:
                empty_paths.append(p)
            elif (
                p.get("has_git") or p.get("vcs_type")
            ) and is_repo_accessible_elsewhere(
                remote_url=p.get("vcs_remote_url") or p.get("git_remote_url"),
                scanner_accessible=p.get("vcs_remote_accessible"),
                facility=state.facility,
            ):
                accessible_vcs_paths.append(p)
            else:
                paths_to_score.append(p)

        # Auto-skip empty directories with 0.0 score
        skipped_results = []
        if empty_paths:
            skip_data = [
                {
                    "path": p["path"],
                    "triage_composite": 0.0,
                    "path_purpose": "empty",
                    "description": "Empty directory - no files or subdirectories",
                    "triage_modeling_code": 0.0,
                    "triage_analysis_code": 0.0,
                    "triage_operations_code": 0.0,
                    "triage_modeling_data": 0.0,
                    "triage_experimental_data": 0.0,
                    "triage_data_access": 0.0,
                    "triage_workflow": 0.0,
                    "triage_visualization": 0.0,
                    "triage_documentation": 0.0,
                    "triage_imas": 0.0,
                    "triage_convention": 0.0,
                    "should_expand": False,
                    "should_enrich": False,
                    "enrich_skip_reason": "empty directory",
                    "skip_reason": "empty",
                    "terminal_reason": TerminalReason.empty.value,
                }
                for p in empty_paths
            ]
            await loop.run_in_executor(
                None, lambda sd=skip_data: mark_triage_complete(state.facility, sd)
            )
            state.triage_stats.processed += len(empty_paths)
            state.triage_stats.record_batch(len(empty_paths))

            skipped_results = [
                {
                    "path": p["path"],
                    "score": 0.0,
                    "label": "empty",
                    "path_purpose": "empty",
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
                    state.triage_stats,
                    skipped_results,
                )
                if not accessible_vcs_paths and not paths_to_score:
                    continue

        # Auto-skip VCS repos with accessible remotes (no LLM needed)
        if accessible_vcs_paths:
            vcs_skip_data = [
                {
                    "path": p["path"],
                    "triage_composite": 0.15,
                    "path_purpose": "software_project",
                    "description": (
                        f"VCS repository accessible elsewhere"
                        f" ({p.get('vcs_remote_url') or p.get('git_remote_url', '')})"
                    ),
                    "triage_modeling_code": 0.1,
                    "triage_analysis_code": 0.1,
                    "triage_operations_code": 0.0,
                    "triage_modeling_data": 0.0,
                    "triage_experimental_data": 0.0,
                    "triage_data_access": 0.1,
                    "triage_workflow": 0.0,
                    "triage_visualization": 0.0,
                    "triage_documentation": 0.0,
                    "triage_imas": 0.0,
                    "triage_convention": 0.0,
                    "should_expand": False,
                    "should_enrich": False,
                    "skip_reason": (
                        f"{p.get('vcs_type') or 'git'} repo accessible elsewhere"
                    ),
                    "enrich_skip_reason": (
                        f"{p.get('vcs_type') or 'git'} repo accessible elsewhere"
                    ),
                    "terminal_reason": TerminalReason.accessible_elsewhere.value,
                }
                for p in accessible_vcs_paths
            ]
            await loop.run_in_executor(
                None,
                lambda sd=vcs_skip_data: mark_triage_complete(state.facility, sd),
            )
            state.triage_stats.processed += len(accessible_vcs_paths)
            state.triage_stats.record_batch(len(accessible_vcs_paths))

            vcs_results = [
                {
                    "path": p["path"],
                    "score": 0.15,
                    "label": "software_project",
                    "path_purpose": "software_project",
                    "description": (
                        f"VCS repository accessible elsewhere"
                        f" ({p.get('vcs_remote_url') or p.get('git_remote_url', '')})"
                    ),
                    "score_imas": 0.0,
                    "score_convention": 0.0,
                    "skip_reason": (
                        f"{p.get('vcs_type') or 'git'} repo accessible elsewhere"
                    ),
                    "should_expand": False,
                    "terminal_reason": TerminalReason.accessible_elsewhere.value,
                    "total_files": p.get("total_files", 0),
                }
                for p in accessible_vcs_paths
            ]
            skipped_results.extend(vcs_results)

            if on_progress and not paths_to_score:
                skipped_msg = (
                    f"skipped {len(empty_paths)} empty, " if empty_paths else "skipped "
                )
                on_progress(
                    f"{skipped_msg}{len(accessible_vcs_paths)} accessible VCS repos",
                    state.triage_stats,
                    skipped_results,
                )
                continue

        if not paths_to_score:
            continue

        if on_progress:
            on_progress(
                f"triaging {len(paths_to_score)} paths", state.triage_stats, None
            )

        # Async LLM triage — fully cancellable
        start = time.time()
        try:
            result = await scorer.async_triage_batch(
                directories=paths_to_score,
                focus=state.focus,
                threshold=state.threshold,
            )
            state.triage_stats.last_batch_time = time.time() - start
            state.triage_stats.cost += result.total_cost

            # Persist results (marks scanned → triaged)
            # Run in executor to avoid blocking event loop
            score_data = [d.to_graph_dict() for d in result.triaged_dirs]
            await loop.run_in_executor(
                None, lambda sd=score_data: mark_triage_complete(state.facility, sd)
            )

            state.triage_stats.processed += len(result.triaged_dirs)
            state.triage_stats.record_batch(len(result.triaged_dirs))

            # Build detailed score results for progress callback
            detailed_results = [
                {
                    "path": d.path,
                    "score": d.score,
                    "label": d.path_purpose.value if d.path_purpose else "",
                    "path_purpose": d.path_purpose.value if d.path_purpose else "",
                    "description": d.description,
                    "physics_domain": (
                        d.physics_domain.value if d.physics_domain else ""
                    ),
                    "score_imas": d.score_imas,
                    "skip_reason": d.skip_reason or "",
                    "should_expand": d.should_expand,
                    "terminal_reason": d.skip_reason or "",
                    "total_files": 0,  # Not available at score time
                }
                for d in result.triaged_dirs
            ]
            # Combine with any skipped results
            all_results = skipped_results + detailed_results
            if on_progress:
                skipped_msg = (
                    f" (+{len(skipped_results)} skipped)" if skipped_results else ""
                )
                on_progress(
                    f"triaged {len(result.triaged_dirs)} (${result.total_cost:.3f}){skipped_msg}",
                    state.triage_stats,
                    all_results,
                )

        except ProviderBudgetExhausted as e:
            logger.error("API key budget exhausted — halting triage workers: %s", e)
            _revert_path_claims(state.facility, [p["path"] for p in paths_to_score])
            state.provider_budget_exhausted = True
            if on_progress:
                on_progress("provider budget exhausted", state.triage_stats, None)
            break
        except ValueError:
            # LLM validation error - revert paths to 'scanned' status for retry
            # DO NOT increment error count - this will be retried automatically
            logger.warning(
                f"LLM validation error for batch of {len(paths_to_score)} paths. "
                "Reverting to scanned status for retry."
            )
            _revert_path_claims(state.facility, [p["path"] for p in paths_to_score])
            # Don't show validation errors in progress display
        except Exception as e:
            # Other errors - increment error count and revert
            logger.exception(f"Triage error: {e}")
            state.triage_stats.errors += len(paths_to_score)
            _revert_path_claims(state.facility, [p["path"] for p in paths_to_score])
            if is_infrastructure_error(e):
                raise

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


async def enrich_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[dict] | None], None] | None = None,
    batch_size: int = 25,
    pool: SSHWorkerPool | None = None,
) -> None:
    """Async enrichment worker.

    Continuously claims scored paths with should_enrich=true, runs deep
    analysis (du, tokei, patterns) via SSH, marks complete.
    Runs in PARALLEL with scan/score/expand workers.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per SSH call (default 25)
        pool: Optional persistent SSH worker pool for session reuse
    """
    # Use local claim_paths_for_enriching and mark_enrichment_complete
    # (defined above in this module with root_filter support)
    from imas_codex.discovery.paths.enrichment import async_enrich_paths

    loop = asyncio.get_running_loop()

    while not state.should_stop():
        # Claim work from graph (run in executor to avoid blocking event loop)
        paths = await loop.run_in_executor(
            None,
            lambda: claim_paths_for_enriching(
                state.facility,
                limit=batch_size,
                root_filter=state.root_filter,
                auto_enrich_threshold=state.auto_enrich_threshold,
            ),
        )

        if not paths:
            state.enrich_phase.record_idle()
            state.enrich_stats.mark_idle()
            if on_progress:
                on_progress("waiting for enrichable paths", state.enrich_stats, None)
            # Wait before polling again
            await asyncio.sleep(2.0)
            continue

        state.enrich_phase.record_activity(len(paths))
        state.enrich_stats.mark_active()
        path_strs = [p["path"] for p in paths]
        # Build path -> purpose mapping for targeted pattern selection
        path_purposes = {p["path"]: p.get("path_purpose") for p in paths}

        if on_progress:
            on_progress(f"enriching {len(paths)} paths", state.enrich_stats, None)

        # Async SSH enrichment — fully cancellable
        start = time.time()
        try:
            results = await async_enrich_paths(
                state.facility,
                path_strs,
                path_purposes=path_purposes,
                pool=pool,
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
            state.enrich_stats.record_batch(enriched)

            if on_progress:
                on_progress(f"enriched {enriched}", state.enrich_stats, result_dicts)

        except Exception as e:
            logger.exception(f"Enrich error: {e}")
            state.enrich_stats.errors += len(paths)
            _revert_path_claims(state.facility, path_strs)
            if is_infrastructure_error(e):
                raise

        # Brief yield
        await asyncio.sleep(0.1)


async def score_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[dict] | None], None] | None = None,
    batch_size: int = 10,
) -> None:
    """Async score worker (2nd pass).

    Continuously claims enriched paths, scores using enrichment data
    (total_bytes, total_lines, language_breakdown) for authoritative scoring.
    Does NOT see numeric triage scores to avoid anchoring bias.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Paths per score operation (default 10)
    """
    # Use local claim_paths_for_scoring and mark_score_complete
    # (defined above in this module with root_filter support)

    loop = asyncio.get_running_loop()

    while not state.should_stop():
        # Check budget before claiming work
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.score_stats, None)
            break

        # Claim work from graph (run in executor to avoid blocking event loop)
        paths = await loop.run_in_executor(
            None,
            lambda: claim_paths_for_scoring(
                state.facility, limit=batch_size, root_filter=state.root_filter
            ),
        )

        if not paths:
            state.score_phase.record_idle()
            state.score_stats.mark_idle()
            if on_progress:
                on_progress("waiting for enriched paths", state.score_stats, None)
            # Wait before polling again
            await asyncio.sleep(2.0)
            continue

        state.score_phase.record_activity(len(paths))
        state.score_stats.mark_active()

        if on_progress:
            on_progress(f"scoring {len(paths)} paths", state.score_stats, None)

        # Run LLM-based scoring in thread pool
        start = time.time()
        try:
            # LLM scoring uses enrichment data for authoritative scoring
            # Pass facility for cross-facility example injection
            llm_results, cost = await _async_score_with_llm(
                paths, state.facility, focus=state.focus
            )

            # Add should_expand from original data and cost tracking
            score_results = []
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
                    "previous_score": orig.get("triage_composite"),
                    "score_cost": cost_per_path,
                    "should_expand": llm_r.get(
                        "should_expand", orig.get("should_expand", True)
                    ),
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
                score_results.append(result)

            if failed_count:
                logger.warning(
                    "Skipped %d/%d failed scores (not written to graph)",
                    failed_count,
                    len(llm_results),
                )
                state.score_stats.errors += failed_count

            state.score_stats.last_batch_time = time.time() - start
            state.score_stats.cost += cost

            # Programmatic data directory detection: override should_expand
            # for paths that structurally look like data containers
            _apply_score_data_dir_overrides(score_results, paths)

            # Persist only successful results
            scored = 0
            if score_results:
                scored = await loop.run_in_executor(
                    None,
                    lambda rr=score_results: mark_score_complete(state.facility, rr),
                )

            state.score_stats.processed += scored
            state.score_stats.record_batch(scored)

            if on_progress:
                on_progress(f"scored {scored}", state.score_stats, score_results)

        except ProviderBudgetExhausted as e:
            logger.error("API key budget exhausted — halting score workers: %s", e)
            state.provider_budget_exhausted = True
            if on_progress:
                on_progress("provider budget exhausted", state.score_stats, None)
            break
        except Exception as e:
            logger.exception(f"Score error: {e}")
            state.score_stats.errors += len(paths)
            if is_infrastructure_error(e):
                raise

        # Brief yield
        await asyncio.sleep(0.1)


def _apply_score_data_dir_overrides(
    score_results: list[dict],
    orig_paths: list[dict],
) -> None:
    """Programmatic data directory detection for score results.

    Overrides should_expand=false for paths that structurally look like
    data containers based on enrichment metadata, even if the LLM didn't
    flag them. Checks:
    1. High data-dimension scores with no code lines
    2. Directories with mostly numeric child names (shot numbers)
    3. path_purpose already set to data types by LLM

    Mutates score_results in place.
    """
    data_purposes = {"modeling_data", "experimental_data"}
    orig_by_path = {p["path"]: p for p in orig_paths}

    for result in score_results:
        if not result.get("should_expand", True):
            continue  # Already marked terminal

        path_purpose = result.get("path_purpose")
        orig = orig_by_path.get(result["path"], {})
        total_lines = orig.get("total_lines") or 0
        total_bytes = orig.get("total_bytes") or 0
        child_names = orig.get("child_names") or ""

        # 1. Data purpose with no code → terminal
        if path_purpose in data_purposes:
            result["should_expand"] = False
            logger.debug(
                "Score data-dir override: %s (purpose=%s)",
                result["path"],
                path_purpose,
            )
            continue

        # 2. Mostly numeric children (shot directories like 1001/, 1002/)
        if child_names:
            import json as _json

            try:
                names = (
                    _json.loads(child_names)
                    if isinstance(child_names, str)
                    else child_names
                )
            except (ValueError, TypeError):
                names = []
            if len(names) >= 5:
                numeric_count = sum(1 for n in names if str(n).strip("/").isdigit())
                if numeric_count / len(names) >= 0.6:
                    result["should_expand"] = False
                    if not path_purpose:
                        result["path_purpose"] = "experimental_data"
                    logger.debug(
                        "Score data-dir override (numeric children): %s "
                        "(%d/%d numeric)",
                        result["path"],
                        numeric_count,
                        len(names),
                    )
                    continue

        # 3. Large byte count with zero code lines → likely raw data
        if total_bytes > 100_000_000 and total_lines == 0:
            result["should_expand"] = False
            if not path_purpose:
                result["path_purpose"] = "experimental_data"
            logger.debug(
                "Score data-dir override (large no-code): %s (%d bytes)",
                result["path"],
                total_bytes,
            )


def _build_score_results(
    batch: ScoreBatch,
    paths: list[dict],
) -> list[dict]:
    """Build results dict from LLM score batch (shared by sync/async)."""
    path_to_result = {r.path: r for r in batch.results}
    results = []

    for p in paths:
        if p["path"] in path_to_result:
            r = path_to_result[p["path"]]
            result: dict = {
                "path": p["path"],
                "score": max(0.0, min(1.0, r.new_score)),
                "adjustment_reason": r.scoring_reason or "",
                "primary_evidence": r.primary_evidence or [],
                "evidence_summary": r.evidence_summary or "",
                "description": r.description or p.get("description", ""),
                "keywords": r.keywords or p.get("keywords", []),
                "path_purpose": (r.path_purpose.value if r.path_purpose else None)
                or p.get("path_purpose"),
                "physics_domain": (r.physics_domain.value if r.physics_domain else None)
                or p.get("physics_domain"),
                "should_expand": r.should_expand,
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
                    # Fall back to triage-phase value
                    triage_dim = dim.replace("score_", "triage_")
                    result[dim] = p.get(triage_dim, 0.0)

            results.append(result)
        else:
            results.append(
                {
                    "path": p["path"],
                    "score": p.get("triage_composite", 0.5),
                    "adjustment_reason": "not in response",
                    "_failed": True,
                }
            )

    return results


def _score_with_llm(
    paths: list[dict],
    facility: str | None = None,
    focus: str | None = None,
) -> tuple[list[dict], float]:
    """Score paths using LLM with enrichment data.

    Injects cross-facility enriched examples into the prompt for
    consistent scoring calibration across facilities.

    CRITICAL: Does NOT show numeric triage scores to the LLM — only
    enrichment evidence. This eliminates anchoring bias.

    Args:
        paths: List of path dicts with enrichment data
        facility: Current facility for preferring same-facility examples
        focus: Optional focus string for scoring

    Returns:
        Tuple of (score_results, cost)
    """
    import json

    from imas_codex.discovery.paths.frontier import (
        sample_dimension_calibration_examples,
        sample_enriched_paths,
    )
    from imas_codex.discovery.paths.models import ScoreBatch
    from imas_codex.llm.prompt_loader import render_prompt
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
        context["score_calibration"] = enriched_examples

    # Add per-dimension calibration examples (cached with 60s TTL)
    # phase='score' draws from scored peers (2nd-pass dimensions).
    dimension_calibration = sample_dimension_calibration_examples(
        facility=facility,
        per_level=5,
        tolerance=0.1,
        phase="score",
    )
    has_dim_calibration = any(
        any(examples for examples in dim_levels.values())
        for dim_levels in dimension_calibration.values()
    )
    if has_dim_calibration:
        context["dimension_calibration"] = dimension_calibration

    if focus:
        context["focus"] = focus

    # Render prompt with examples
    system_prompt = render_prompt("paths/scorer", context)

    # Build user prompt with enrichment data but NO numeric triage scores
    lines = ["Score these directories using their enrichment evidence:\n"]
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

        # Qualitative context from triage (no numeric scores)
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

        # Enrichment metrics (concrete evidence for scoring)
        lines.append("\nEnrichment data:")
        lines.append(f"  Total lines: {p.get('total_lines') or 0}")
        lines.append(f"  Total bytes: {p.get('total_bytes') or 0}")
        lines.append(f"  Language breakdown: {lang or {}}")
        lines.append(f"  Is multiformat: {p.get('is_multiformat', False)}")

        # Pattern match evidence (key data for scoring)
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

    user_prompt = "\n".join(lines)

    # Get model
    model = get_model("language")

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
            response_model=ScoreBatch,
        )
    except ValueError:
        # All retries exhausted — mark all as failed so they're skipped
        return [
            {
                "path": p["path"],
                "score": p.get("triage_composite", 0.5),
                "adjustment_reason": "parse error",
                "_failed": True,
            }
            for p in paths
        ], 0.0

    results = _build_score_results(batch, paths)
    return results, cost


async def _async_score_with_llm(
    paths: list[dict],
    facility: str | None = None,
    focus: str | None = None,
) -> tuple[list[dict], float]:
    """Score paths using LLM with enrichment data (async/cancellable).

    Async version of _score_with_llm using acall_llm_structured for
    native async LLM calls that respond to asyncio.cancel().

    CRITICAL: Does NOT show numeric triage scores to the LLM — only
    enrichment evidence. This eliminates anchoring bias.

    Args:
        paths: List of path dicts with enrichment data
        facility: Current facility for preferring same-facility examples
        focus: Optional focus string for scoring

    Returns:
        Tuple of (score_results, cost)
    """
    import json

    from imas_codex.discovery.paths.frontier import (
        sample_dimension_calibration_examples,
        sample_enriched_paths,
    )
    from imas_codex.discovery.paths.models import ScoreBatch
    from imas_codex.llm.prompt_loader import render_prompt
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
        context["score_calibration"] = enriched_examples

    # Add per-dimension calibration examples (cached with 60s TTL)
    # phase='score' draws from scored peers (2nd-pass dimensions).
    dimension_calibration = sample_dimension_calibration_examples(
        facility=facility,
        per_level=5,
        tolerance=0.1,
        phase="score",
    )
    has_dim_calibration = any(
        any(examples for examples in dim_levels.values())
        for dim_levels in dimension_calibration.values()
    )
    if has_dim_calibration:
        context["dimension_calibration"] = dimension_calibration

    if focus:
        context["focus"] = focus

    # Render prompt with examples
    system_prompt = render_prompt("paths/scorer", context)

    # Build user prompt with enrichment data but NO numeric triage scores
    lines_prompt = ["Score these directories using their enrichment evidence:\n"]
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

        # Qualitative context from triage (no numeric scores)
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

        # Enrichment metrics (concrete evidence for scoring)
        lines_prompt.append("\nEnrichment data:")
        lines_prompt.append(f"  Total lines: {p.get('total_lines') or 0}")
        lines_prompt.append(f"  Total bytes: {p.get('total_bytes') or 0}")
        lines_prompt.append(f"  Language breakdown: {lang or {}}")
        lines_prompt.append(f"  Is multiformat: {p.get('is_multiformat', False)}")

        # Pattern match evidence (key data for scoring)
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

    user_prompt = "\n".join(lines_prompt)

    # Get model
    model = get_model("language")

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
            response_model=ScoreBatch,
        )
    except ValueError:
        # All retries exhausted — mark all as failed so they're skipped
        return [
            {
                "path": p["path"],
                "score": p.get("triage_composite", 0.5),
                "adjustment_reason": "parse error",
                "_failed": True,
            }
            for p in paths
        ], 0.0

    results = _build_score_results(batch, paths)
    return results, cost


# ============================================================================
# SSH Preflight Check
# ============================================================================


def check_ssh_connectivity(facility: str, timeout: int = 10) -> tuple[bool, str]:
    """Check if SSH connection to facility is working.

    Delegates to :func:`~imas_codex.remote.executor.ssh_preflight` — the
    canonical SSH connectivity check that handles socket directory creation,
    stale socket cleanup, and retry logic.

    Args:
        facility: Facility ID
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, message)
    """
    from imas_codex.remote.executor import ssh_preflight

    # Resolve facility → SSH host alias
    ssh_host = facility
    try:
        from imas_codex.discovery.base.facility import get_facility

        config = get_facility(facility)
        ssh_host = config.get("ssh_host", facility)
    except (ValueError, Exception):
        pass

    ok, detail = ssh_preflight(ssh_host, timeout=timeout, retry=True)
    if ok:
        return True, f"SSH to {facility} working"
    return False, f"Cannot connect to {facility}: {detail}"


def _clear_user_claim(user_id: str) -> None:
    """Clear claimed_at on a FacilityUser so it can be retried."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        gc.query(
            "MATCH (u:FacilityUser {id: $id}) SET u.claimed_at = null",
            id=user_id,
        )


# ============================================================================
# User Worker (async Person linking with ORCID)
# ============================================================================


async def user_worker(
    state: DiscoveryState,
    on_progress: Callable[[str, WorkerStats, list[str] | None], None] | None = None,
    batch_size: int = 10,
) -> None:
    """Async user worker — links FacilityUser nodes to Person nodes via ORCID.

    Runs independently of the scan pipeline to avoid blocking path discovery.
    Claims FacilityUser nodes without person_id and performs ORCID lookups
    to create cross-facility Person identities.

    This worker is non-critical: if it falls behind, scan/triage/score
    continue unimpeded. Person links are enrichment, not pipeline-blocking.

    Args:
        state: Shared discovery state
        on_progress: Progress callback
        batch_size: Users per claim batch (default 10)
    """
    from datetime import UTC, datetime

    from imas_codex.discovery.paths.frontier import _create_person_link
    from imas_codex.graph import GraphClient

    loop = asyncio.get_running_loop()

    while not state.should_stop():
        # Claim unclaimed users from graph
        users = await loop.run_in_executor(
            None,
            lambda: claim_users_for_linking(state.facility, limit=batch_size),
        )

        if not users:
            state.user_phase.record_idle()
            state.user_stats.mark_idle()
            if on_progress:
                on_progress("idle", state.user_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.user_phase.record_activity(len(users))
        state.user_stats.mark_active()

        if on_progress:
            on_progress(f"linking {len(users)} users", state.user_stats, None)

        now = datetime.now(UTC).isoformat()
        linked = 0
        errors = 0

        for user in users:
            if state.should_stop():
                break
            try:
                gc = await loop.run_in_executor(None, GraphClient)
                try:
                    await _create_person_link(
                        gc,
                        facility_user_id=user["id"],
                        username=user["username"],
                        name=user.get("name"),
                        given_name=user.get("given_name"),
                        family_name=user.get("family_name"),
                        email=user.get("email"),
                        now=now,
                    )
                    linked += 1
                finally:
                    await loop.run_in_executor(None, gc.close)
            except Exception as e:
                logger.debug(f"Person link failed for {user['id']}: {e}")
                errors += 1
                # Clear claim so user can be retried
                try:
                    await loop.run_in_executor(
                        None,
                        lambda uid=user["id"]: _clear_user_claim(uid),
                    )
                except Exception:
                    pass

        state.user_stats.processed += linked
        state.user_stats.errors += errors
        state.user_stats.record_batch(linked)

        if on_progress:
            on_progress(
                f"linked {linked} users",
                state.user_stats,
                [u["username"] for u in users[:linked]],
            )

        # Yield to other workers
        await asyncio.sleep(0.5)


# ============================================================================
# Main Discovery Loop
# ============================================================================


async def run_parallel_discovery(
    facility: str,
    cost_limit: float = 10.0,
    path_limit: int | None = None,
    focus: str | None = None,
    threshold: float | None = None,
    root_filter: list[str] | None = None,
    auto_enrich_threshold: float | None = None,
    num_scan_workers: int = 1,
    num_expand_workers: int = 1,
    num_triage_workers: int = 1,  # Single triage worker (LLM)
    num_enrich_workers: int = 2,  # Two enrichment workers for parallel SSH
    num_score_workers: int = 1,  # Score worker uses enrichment evidence (LLM)
    scan_batch_size: int = 20,  # 20 paths per SSH call (amortize connection overhead)
    expand_batch_size: int = 25,  # Expansion needs child enumeration
    triage_batch_size: int = 50,  # More work per API call
    enrich_batch_size: int = 25,  # Larger batches: amortize SSH connection cost
    score_batch_size: int = 10,  # Smaller batches: 20+ fields per path in structured output
    dedup_batch_size: int = 50,  # Repos to check per dedup cycle
    on_scan_progress: Callable[
        [str, WorkerStats, list[str] | None, list[dict] | None], None
    ]
    | None = None,
    on_expand_progress: Callable[
        [str, WorkerStats, list[str] | None, list[dict] | None], None
    ]
    | None = None,
    on_triage_progress: Callable[[str, WorkerStats, list[dict] | None], None]
    | None = None,
    on_enrich_progress: Callable[[str, WorkerStats, list[dict] | None], None]
    | None = None,
    on_score_progress: Callable[[str, WorkerStats, list[dict] | None], None]
    | None = None,
    on_dedup_progress: Callable[[str, WorkerStats, list[dict] | None], None]
    | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
    graceful_shutdown_timeout: float = 2.0,
    deadline: float | None = None,
    stop_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    """Run parallel discovery with all worker types.

    Workers:
    - scan: Lists directories, discovers child paths (SSH-bound)
    - expand: Expands triaged high-value paths (should_expand=true)
    - triage: LLM-based triage with should_expand/should_enrich decisions
    - enrich: Deep analysis (du/tokei/patterns) for should_enrich=true paths
    - score: Scores using enrichment data (no numeric triage scores shown to LLM)
    - dedup: Marks clone paths terminal using SoftwareRepo identity (graph-only)

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
        num_triage_workers: Number of concurrent triage workers (default: 1)
        num_enrich_workers: Number of concurrent enrich workers (default: 2)
        num_score_workers: Number of concurrent score workers (default: 1)
        scan_batch_size: Paths per SSH call (default: 20)
        expand_batch_size: Paths per expand SSH call (default: 25)
        triage_batch_size: Paths per triage LLM call (default: 50)
        enrich_batch_size: Paths per SSH call (default: 25)
        score_batch_size: Paths per score LLM call (default: 10)
        dedup_batch_size: SoftwareRepo clone groups per dedup cycle (default: 50)
        on_scan_progress: Callback for scan progress
        on_expand_progress: Callback for expand progress
        on_triage_progress: Callback for triage progress
        on_enrich_progress: Callback for enrich progress
        on_score_progress: Callback for score progress
        on_dedup_progress: Callback for dedup progress
        on_worker_status: Callback for worker status changes. Called with
            SupervisedWorkerGroup for display integration.
        graceful_shutdown_timeout: Seconds to wait for workers to finish after
            limit reached before cancelling (default: 5.0)
        deadline: Unix timestamp when discovery should stop (optional)
        stop_event: External asyncio.Event for graceful shutdown. When set,
            ``state.stop_requested`` is activated so workers finish their
            current batch and exit cleanly.

    Terminates when:
    - Cost limit reached
    - Path limit reached (if set)
    - Deadline expired (if set)
    - All workers idle (no more work)

    Returns:
        Summary dict with scanned, expanded, triaged, enriched, scored, deduped, cost, elapsed, rates
    """
    from imas_codex.discovery import get_discovery_stats, seed_facility_roots

    # SSH preflight check - fail fast if facility is unreachable.
    # Use 30s timeout to accommodate proxy-jump connections (e.g., TCV via lac912)
    # and allow time for stale control socket cleanup.
    ssh_ok, ssh_message = check_ssh_connectivity(facility, timeout=30)
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

    # Resolve thresholds from settings when not explicitly provided
    from imas_codex.settings import get_discovery_threshold

    resolved_threshold = (
        threshold if threshold is not None else get_discovery_threshold()
    )
    resolved_enrich = (
        auto_enrich_threshold
        if auto_enrich_threshold is not None
        else get_discovery_threshold()
    )

    # Create shared state
    state = DiscoveryState(
        facility=facility,
        cost_limit=cost_limit,
        path_limit=path_limit,
        focus=focus,
        threshold=resolved_threshold,
        root_filter=root_filter,
        auto_enrich_threshold=resolved_enrich,
        deadline=deadline,
    )

    # Wire up graph-backed has_work_fn on each phase so PipelinePhase.done
    # verifies pending work in the graph before declaring a phase complete.
    # This prevents workers from going idle while upstream phases still have
    # work to produce.  Each downstream phase also checks if its upstream
    # is done, so it stays alive until the upstream finishes.
    state.scan_phase.set_has_work_fn(lambda: _has_pending_scan_work(facility))
    state.expand_phase.set_has_work_fn(lambda: _has_pending_expand_work(facility))
    state.triage_phase.set_has_work_fn(
        lambda: (
            _has_pending_triage_work(facility)
            or not state.scan_phase.done
            or not state.expand_phase.done
        )
    )
    state.enrich_phase.set_has_work_fn(
        lambda: _has_pending_enrich_work(facility) or not state.triage_phase.done
    )
    state.score_phase.set_has_work_fn(
        lambda: _has_pending_score_work(facility) or not state.enrich_phase.done
    )
    state.user_phase.set_has_work_fn(
        lambda: _has_pending_user_work(facility) or not state.scan_phase.done
    )

    # Capture initial terminal count for session-based --limit tracking
    state.initial_terminal_count = state.terminal_count

    # Watch external stop event (from CLI signal handler)
    # NOTE: run_discovery_engine handles stop_event internally, but we need
    # to track user_interrupted separately for post-engine cleanup decisions.

    # Create persistent SSH worker pool to amortize PAM session overhead.
    # Each SSH session to some facilities costs ~2.6s (pam_systemd.so scope
    # creation). Persistent workers pay this cost once, then execute subsequent
    # commands in ~12ms (network RTT only).
    ssh_pool: SSHWorkerPool | None = None
    num_ssh_workers = num_scan_workers + num_expand_workers + num_enrich_workers
    if num_ssh_workers > 0:
        from imas_codex.discovery.base.facility import get_facility
        from imas_codex.remote.ssh_worker import SSHWorkerPool as _SSHWorkerPool

        try:
            config = get_facility(facility)
            ssh_host = config.get("ssh_host", facility)
        except ValueError:
            ssh_host = facility

        from imas_codex.remote.executor import is_local_host

        if not is_local_host(ssh_host):
            try:
                ssh_pool = _SSHWorkerPool(
                    ssh_host=ssh_host,
                    max_workers=num_ssh_workers,
                )
                await ssh_pool.start()
                logger.info(
                    "Started persistent SSH pool for %s with %d workers",
                    ssh_host,
                    num_ssh_workers,
                )
            except Exception as e:
                logger.warning(
                    "Failed to start SSH worker pool for %s, "
                    "falling back to per-command SSH: %s",
                    ssh_host,
                    e,
                )
                ssh_pool = None

    try:
        # Build worker specs
        workers: list[WorkerSpec] = [
            WorkerSpec(
                "scan",
                "scan_phase",
                scan_worker,
                count=num_scan_workers,
                on_progress=on_scan_progress,
                kwargs={"batch_size": scan_batch_size, "pool": ssh_pool},
            ),
            WorkerSpec(
                "expand",
                "expand_phase",
                expand_worker,
                count=num_expand_workers,
                on_progress=on_expand_progress,
                kwargs={"batch_size": expand_batch_size, "pool": ssh_pool},
            ),
            WorkerSpec(
                "triage",
                "triage_phase",
                triage_worker,
                count=num_triage_workers,
                on_progress=on_triage_progress,
                kwargs={"batch_size": triage_batch_size},
            ),
            WorkerSpec(
                "enrich",
                "enrich_phase",
                enrich_worker,
                count=num_enrich_workers,
                on_progress=on_enrich_progress,
                kwargs={"batch_size": enrich_batch_size, "pool": ssh_pool},
            ),
            WorkerSpec(
                "score",
                "score_phase",
                score_worker,
                count=num_score_workers,
                on_progress=on_score_progress,
                kwargs={"batch_size": score_batch_size},
            ),
        ]

        # Embed description worker
        from imas_codex.discovery.base.embed_worker import embed_description_worker

        workers.append(
            WorkerSpec(
                "embed",
                "score_phase",  # embed lifecycle follows score
                embed_description_worker,
                group="embed",
                kwargs={
                    "labels": ["FacilityPath"],
                    "done_check": lambda: state.score_phase.done,
                },
            )
        )

        # Dedup worker (graph-only, independent)
        workers.append(
            WorkerSpec(
                "dedup",
                "triage_phase",  # dedup lifecycle follows triage
                dedup_worker,
                group="dedup",
                on_progress=on_dedup_progress,
                kwargs={"batch_size": dedup_batch_size},
            )
        )

        # User worker (async Person linking with ORCID)
        workers.append(
            WorkerSpec(
                "user",
                "user_phase",
                user_worker,
                group="user",
            )
        )

        orphan_specs = [
            OrphanRecoverySpec("FacilityPath", timeout_seconds=CLAIM_TIMEOUT_SECONDS),
            OrphanRecoverySpec("FacilityUser", timeout_seconds=CLAIM_TIMEOUT_SECONDS),
        ]

        await run_discovery_engine(
            state,
            workers,
            stop_event=stop_event,
            orphan_specs=orphan_specs,
            on_worker_status=on_worker_status,
            shutdown_timeout=graceful_shutdown_timeout,
        )

        # Determine why we stopped for logging
        if state.budget_exhausted:
            logger.info("Budget limit reached")
        elif state.path_limit_reached:
            logger.info("Path limit reached")
        elif stop_event is not None and stop_event.is_set():
            logger.info("Stop requested")
        else:
            logger.info("Discovery complete (no pending work)")

        user_interrupted = stop_event is not None and stop_event.is_set()

        # Skip heavyweight cleanup when the user pressed Ctrl+C — these are
        # nice-to-have ops that will run at the start of the next session.
        if not user_interrupted:
            # Graceful shutdown: reset any in-progress claims for next run
            reset_count = reset_orphaned_claims(facility, silent=True)
            if reset_count:
                logger.info(f"Shutdown cleanup: {reset_count} claimed paths reset")

            # Auto-normalize scores after scoring completes
            if state.score_stats.processed > 0:
                from imas_codex.discovery.paths.frontier import normalize_scores

                normalize_scores(facility)

                # Force garbage collection while the event loop is still running so
                # that orphaned asyncio subprocess transports are cleaned up before
                # the loop closes.  Without this,
                # BaseSubprocessTransport.__del__ fires after the loop is closed and
                # prints harmless but noisy RuntimeError tracebacks.
                import gc

                gc.collect()
        else:
            logger.info(
                "User interrupted — skipping score normalization and orphan reset"
            )

        elapsed = max(
            state.scan_stats.elapsed,
            state.expand_stats.elapsed,
            state.triage_stats.elapsed,
            state.enrich_stats.elapsed,
            state.score_stats.elapsed,
            state.dedup_stats.elapsed,
        )

        return {
            "scanned": state.scan_stats.processed,
            "expanded": state.expand_stats.processed,
            "triaged": state.triage_stats.processed,
            "enriched": state.enrich_stats.processed,
            "scored": state.score_stats.processed,
            "deduped": state.dedup_stats.processed,
            "users_linked": state.user_stats.processed,
            "cost": state.total_cost,
            "elapsed_seconds": elapsed,
            "scan_rate": state.scan_stats.rate,
            "expand_rate": state.expand_stats.rate,
            "triage_rate": state.triage_stats.rate,
            "enrich_rate": state.enrich_stats.rate,
            "score_rate": state.score_stats.rate,
            "scan_errors": state.scan_stats.errors,
            "expand_errors": state.expand_stats.errors,
            "triage_errors": state.triage_stats.errors,
            "enrich_errors": state.enrich_stats.errors,
            "score_errors": state.score_stats.errors,
            "user_errors": state.user_stats.errors,
        }
    finally:
        # Clean up persistent SSH workers on exit (normal, exception, or Ctrl+C)
        if ssh_pool is not None:
            if stop_event is not None and stop_event.is_set():
                # User interrupted — force-kill immediately instead of
                # waiting 3-5s per worker for graceful close.
                logger.info("Force-killing SSH worker pool (user interrupt)")
                ssh_pool.force_kill_all()
            else:
                logger.info("Shutting down persistent SSH worker pool")
                try:
                    await asyncio.wait_for(ssh_pool.close(), timeout=10.0)
                except (TimeoutError, asyncio.CancelledError):
                    logger.warning("SSH pool close timed out — force-killing workers")
                    ssh_pool.force_kill_all()
