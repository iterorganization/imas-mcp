"""Graph operations for file discovery claim coordination.

Provides claim/release functions for both scan and score phases of
file discovery, enabling safe parallel execution of ``discover code``.

Scan phase: Claims FacilityPath nodes via ``files_claimed_at`` (separate
from the paths module's ``claimed_at``) to prevent duplicate SSH scans.

Score phase: Claims CodeFile nodes via ``claimed_at`` to prevent
duplicate LLM scoring calls.
"""

from __future__ import annotations

import logging
from typing import Any

from imas_codex.discovery.base.claims import (
    DEFAULT_CLAIM_TIMEOUT_SECONDS,
    release_claim,
    release_claims_batch,
    reset_stale_claims,
)
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

CLAIM_TIMEOUT_SECONDS = DEFAULT_CLAIM_TIMEOUT_SECONDS  # 300s (5 minutes)


# ---------------------------------------------------------------------------
# Startup reset
# ---------------------------------------------------------------------------


def reset_orphaned_file_claims(
    facility: str, *, silent: bool = False
) -> dict[str, int]:
    """Release stale claims for file discovery (scan + score phases).

    Recovers from crashed workers by releasing claims older than
    ``CLAIM_TIMEOUT_SECONDS``.  Safe for parallel execution — only
    truly orphaned claims are released.

    Args:
        facility: Facility ID
        silent: Suppress logging

    Returns:
        Dict with ``code_file_reset`` and ``facility_path_reset`` counts
    """
    # CodeFile claims (scoring phase)
    sf_reset = reset_stale_claims(
        "CodeFile",
        facility,
        timeout_seconds=CLAIM_TIMEOUT_SECONDS,
        silent=silent,
    )

    # FacilityPath file-scan claims (uses separate files_claimed_at field)
    fp_reset = reset_stale_claims(
        "FacilityPath",
        facility,
        timeout_seconds=CLAIM_TIMEOUT_SECONDS,
        claimed_field="files_claimed_at",
        silent=silent,
    )

    total = sf_reset + fp_reset
    if total and not silent:
        logger.info(
            "Released %d orphaned file claims (%d CodeFile, %d FacilityPath)",
            total,
            sf_reset,
            fp_reset,
        )

    return {"code_file_reset": sf_reset, "facility_path_reset": fp_reset}


# ---------------------------------------------------------------------------
# Scan-phase claiming (FacilityPath with files_claimed_at)
# ---------------------------------------------------------------------------


def claim_paths_for_file_scan(
    facility: str,
    min_score: float = 0.5,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Atomically claim scored FacilityPaths for file scanning.

    Uses ``files_claimed_at`` (separate from the paths module's ``claimed_at``)
    to avoid conflicts with the paths discovery pipeline.

    Skips paths that have already been scanned (files_scanned > 0), paths
    linked to publicly-accessible software repos (INSTANCE_OF → SoftwareRepo
    with remote_url containing github/gitlab/bitbucket), and paths where the
    VCS remote is known to be accessible externally.

    Prioritizes paths with high scores on mapping-valuable dimensions
    (data_access, imas, convention, analysis, modeling).

    Args:
        facility: Facility ID
        min_score: Minimum path score to include
        limit: Maximum paths to claim

    Returns:
        List of dicts with ``id``, ``path``, ``score``, ``purpose``, ``files_scanned``
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath {facility_id: $facility})
            WHERE p.status IN ['scored', 'explored']
              AND coalesce(p.score_composite, 0) >= $min_score
              AND p.path IS NOT NULL
              AND (p.files_claimed_at IS NULL
                   OR p.files_claimed_at < datetime() - duration($cutoff))
              AND coalesce(p.vcs_remote_accessible, false) = false
              AND NOT EXISTS {
                MATCH (p)-[:INSTANCE_OF]->(r:SoftwareRepo)
                WHERE r.source_type IN ['github', 'gitlab', 'bitbucket']
              }
            OPTIONAL MATCH (f:Facility {id: $facility})
            WITH p, f
            WHERE p.last_file_scan_at IS NULL
               OR (f.files_scan_after IS NOT NULL
                   AND p.last_file_scan_at < f.files_scan_after)
            WITH p
            ORDER BY (
                coalesce(p.score_data_access, 0) * 3
                + coalesce(p.score_imas, 0) * 3
                + coalesce(p.score_convention, 0) * 2
                + coalesce(p.score_analysis_code, 0)
                + coalesce(p.score_modeling_code, 0)
            ) DESC, p.score_composite DESC
            LIMIT $limit
            SET p.files_claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.score_composite AS score,
                   p.purpose AS purpose,
                   coalesce(p.files_scanned, 0) AS files_scanned
            """,
            facility=facility,
            min_score=min_score,
            limit=limit,
            cutoff=cutoff,
        )
        paths = list(result)
        if paths:
            logger.debug(
                "Claimed %d FacilityPaths for file scanning (facility=%s)",
                len(paths),
                facility,
            )
        return paths


def mark_path_file_scanned(path_id: str, file_count: int) -> None:
    """Mark a FacilityPath as scanned with the file count.

    Sets ``files_scanned`` so the path won't be re-claimed for scanning.
    Called for all paths including those with 0 files.
    """
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (p:FacilityPath {id: $id})
                SET p.files_scanned = $count,
                    p.last_file_scan_at = datetime()
                """,
                id=path_id,
                count=file_count,
            )
    except Exception as e:
        logger.warning("Failed to mark path %s as scanned: %s", path_id, e)


def release_path_file_scan_claim(path_id: str) -> None:
    """Release file-scan claim on a FacilityPath.

    Clears ``files_claimed_at`` (not ``claimed_at``, which belongs
    to the paths discovery module).
    """
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (p:FacilityPath {id: $id})
                SET p.files_claimed_at = null
                """,
                id=path_id,
            )
    except Exception as e:
        logger.warning("Failed to release file scan claim for %s: %s", path_id, e)


def release_path_file_scan_claims_batch(path_ids: list[str]) -> int:
    """Release file-scan claims on multiple FacilityPaths."""
    if not path_ids:
        return 0
    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                UNWIND $ids AS pid
                MATCH (p:FacilityPath {id: pid})
                WHERE p.files_claimed_at IS NOT NULL
                SET p.files_claimed_at = null
                RETURN count(p) AS released
                """,
                ids=path_ids,
            )
            return result[0]["released"] if result else 0
    except Exception as e:
        logger.warning("Failed to release file scan claims: %s", e)
        return 0


def set_files_scan_after(facility: str) -> None:
    """Set files_scan_after on the Facility node to trigger rescan.

    All FacilityPaths with last_file_scan_at before this timestamp
    become eligible for re-scanning.
    """
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (f:Facility {id: $facility})
            SET f.files_scan_after = datetime()
            """,
            facility=facility,
        )
    logger.info("Set files_scan_after on %s — paths will be rescanned", facility)


# ---------------------------------------------------------------------------
# Triage-phase claiming (CodeFile: discovered → triaged)
# ---------------------------------------------------------------------------


def claim_files_for_triage(
    facility: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Atomically claim discovered CodeFiles for LLM triage.

    Claims files with ``status='discovered'`` and no ``triage_composite``.
    Returns minimal context: path, language, parent directory path and
    description.  NO numeric scores from the parent — the triage LLM
    should assess from filename and directory context alone.

    Files are ordered by parent path score descending so files from
    the highest-value directories are triaged first.

    Args:
        facility: Facility ID
        limit: Maximum files to claim

    Returns:
        List of dicts with file info + parent path/description
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.triage_composite IS NULL
              AND (sf.claimed_at IS NULL
                   OR sf.claimed_at < datetime() - duration($cutoff))
            OPTIONAL MATCH (sf)-[:IN_DIRECTORY]->(p:FacilityPath)
            WITH sf, p
            ORDER BY coalesce(p.score_composite, 0) DESC, sf.discovered_at ASC
            LIMIT $limit
            SET sf.claimed_at = datetime()
            RETURN sf.id AS id, sf.path AS path,
                   sf.language AS language,
                   p.id AS parent_path_id, p.path AS parent_path,
                   p.description AS parent_description
            """,
            facility=facility,
            limit=limit,
            cutoff=cutoff,
        )
        files = list(result)
        if files:
            logger.debug(
                "Claimed %d CodeFiles for triage (facility=%s)",
                len(files),
                facility,
            )
        return files


def release_file_triage_claim(file_id: str) -> None:
    """Release triage claim on a single CodeFile."""
    release_claim("CodeFile", file_id)


def release_file_triage_claims(file_ids: list[str]) -> None:
    """Release triage claims on multiple CodeFiles."""
    release_claims_batch("CodeFile", file_ids)


# ---------------------------------------------------------------------------
# Enrich-phase claiming (CodeFile: triaged → enriched)
# ---------------------------------------------------------------------------


def claim_files_for_enrichment(
    facility: str,
    limit: int = 200,
    min_triage_composite: float = 0.75,
) -> list[dict[str, Any]]:
    """Atomically claim triaged CodeFiles for rg pattern enrichment.

    Claims files with ``status='triaged'`` and ``triage_composite``
    above the threshold.  Enrichment runs rg pattern matching and
    extracts preview text for the subsequent scoring pass.

    Prioritizes files with high triage composite scores.

    Args:
        facility: Facility ID
        limit: Maximum files to claim
        min_triage_composite: Minimum triage composite to enrich

    Returns:
        List of dicts with id, path, language, triage_composite
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'triaged'
              AND sf.triage_composite >= $min_triage
              AND coalesce(sf.is_enriched, false) = false
              AND (sf.claimed_at IS NULL
                   OR sf.claimed_at < datetime() - duration($cutoff))
            WITH sf ORDER BY sf.triage_composite DESC LIMIT $limit
            SET sf.claimed_at = datetime()
            RETURN sf.id AS id, sf.path AS path,
                   sf.language AS language,
                   sf.triage_composite AS triage_composite
            """,
            facility=facility,
            limit=limit,
            cutoff=cutoff,
            min_triage=min_triage_composite,
        )
        files = list(result)
        if files:
            logger.debug(
                "Claimed %d CodeFiles for enrichment (facility=%s)",
                len(files),
                facility,
            )
        return files


def release_file_enrich_claims(file_ids: list[str]) -> None:
    """Release enrichment claims on multiple CodeFiles."""
    release_claims_batch("CodeFile", file_ids)


# ---------------------------------------------------------------------------
# Score-phase claiming (CodeFile: triaged+enriched → scored)
# ---------------------------------------------------------------------------


def claim_files_for_scoring(
    facility: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Atomically claim enriched CodeFiles for full LLM scoring.

    Claims files with ``status='triaged'`` that have been enriched
    (``is_enriched=true``).  Returns triage description (qualitative
    only, NO triage numeric scores) plus enrichment evidence (pattern
    categories, line count) and parent directory context.

    Args:
        facility: Facility ID
        limit: Maximum files to claim

    Returns:
        List of dicts with file info + triage description + enrichment data
    """
    import json as _json

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'triaged'
              AND sf.is_enriched = true
              AND (sf.claimed_at IS NULL
                   OR sf.claimed_at < datetime() - duration($cutoff))
            OPTIONAL MATCH (sf)-[:IN_DIRECTORY]->(p:FacilityPath)
            WITH sf, p
            ORDER BY sf.triage_composite DESC, sf.triaged_at ASC
            LIMIT $limit
            SET sf.claimed_at = datetime()
            RETURN sf.id AS id, sf.path AS path,
                   sf.language AS language,
                   sf.triage_description AS triage_description,
                   sf.pattern_categories AS pattern_categories_json,
                   sf.total_pattern_matches AS total_pattern_matches,
                   sf.line_count AS line_count,
                   p.id AS parent_path_id, p.path AS parent_path,
                   p.description AS parent_description
            """,
            facility=facility,
            limit=limit,
            cutoff=cutoff,
        )
        files = []
        for row in result:
            f = dict(row)
            # Parse pattern_categories JSON
            pj = f.pop("pattern_categories_json", None)
            if pj:
                try:
                    f["pattern_categories"] = _json.loads(pj)
                except (_json.JSONDecodeError, TypeError):
                    f["pattern_categories"] = {}
            else:
                f["pattern_categories"] = {}
            f.setdefault("total_pattern_matches", 0)
            f.setdefault("line_count", 0)
            files.append(f)
        if files:
            logger.debug(
                "Claimed %d CodeFiles for scoring (facility=%s)",
                len(files),
                facility,
            )
        return files


def release_file_score_claim(file_id: str) -> None:
    """Release scoring claim on a single CodeFile."""
    release_claim("CodeFile", file_id)


def release_file_score_claims(file_ids: list[str]) -> None:
    """Release scoring claims on multiple CodeFiles."""
    release_claims_batch("CodeFile", file_ids)


# ---------------------------------------------------------------------------
# Has-pending-work queries (used by PipelinePhase.has_work_fn)
# ---------------------------------------------------------------------------


def has_pending_scan_work(facility: str, min_score: float = 0.5) -> bool:
    """Check if there are FacilityPaths remaining to scan for files."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath {facility_id: $facility})
            WHERE p.status IN ['scored', 'explored']
              AND coalesce(p.score_composite, 0) >= $min_score
              AND p.path IS NOT NULL
              AND coalesce(p.vcs_remote_accessible, false) = false
              AND NOT EXISTS {
                MATCH (p)-[:INSTANCE_OF]->(r:SoftwareRepo)
                WHERE r.source_type IN ['github', 'gitlab', 'bitbucket']
              }
            OPTIONAL MATCH (f:Facility {id: $facility})
            WITH p, f
            WHERE p.last_file_scan_at IS NULL
               OR (f.files_scan_after IS NOT NULL
                   AND p.last_file_scan_at < f.files_scan_after)
            RETURN count(p) > 0 AS has_work
            """,
            facility=facility,
            min_score=min_score,
        )
        return result[0]["has_work"] if result else False


def has_pending_score_work(facility: str) -> bool:
    """Check if there are triaged+enriched CodeFiles needing scoring."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'triaged'
              AND sf.is_enriched = true
            RETURN count(sf) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


def has_pending_triage_work(facility: str) -> bool:
    """Check if there are discovered CodeFiles needing triage."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.triage_composite IS NULL
            RETURN count(sf) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


def has_pending_enrich_work(facility: str) -> bool:
    """Check if there are triaged CodeFiles needing rg enrichment."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'triaged'
              AND coalesce(sf.is_enriched, false) = false
            RETURN count(sf) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


def has_pending_code_work(
    facility: str, min_score: float = 0.75, max_line_count: int = 10000
) -> bool:
    """Check if there are scored code files needing ingestion."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'scored'
              AND sf.interest_score >= $min_score
              AND coalesce(sf.line_count, 0) <= $max_line_count
            RETURN count(sf) > 0 AS has_work
            """,
            facility=facility,
            min_score=min_score,
            max_line_count=max_line_count,
        )
        return result[0]["has_work"] if result else False


def has_pending_link_work(facility: str) -> bool:
    """Check if there are ingested CodeFiles with unlinked code evidence."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'ingested'
              AND coalesce(sf.evidence_linked, false) = false
            RETURN count(sf) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


def link_code_evidence_to_signals(facility: str) -> dict[str, int]:
    """Link code evidence to FacilitySignals via DataReference → TreeNode → Signal.

    Traverses the chain:
      CodeChunk → CONTAINS_REF → DataReference → RESOLVES_TO_TREE_NODE → TreeNode
      FacilitySignal → SOURCE_NODE → TreeNode

    Sets code_evidence_count and has_code_evidence on matched signals.
    Marks processed CodeFiles as evidence_linked=true.

    Returns:
        Dict with signals_linked, refs_resolved counts
    """
    with GraphClient() as gc:
        # Step 1: Ensure DataReference → TreeNode links exist
        # (may not have been created yet for new ingestions)
        resolve_result = gc.query(
            """
            MATCH (d:DataReference {ref_type: 'mdsplus_path', facility_id: $facility})
            WHERE NOT (d)-[:RESOLVES_TO_TREE_NODE]->()
            MATCH (t:TreeNode {facility_id: $facility})
            WHERE t.path = d.raw_string
               OR toUpper(t.path) = toUpper(d.raw_string)
            MERGE (d)-[:RESOLVES_TO_TREE_NODE]->(t)
            RETURN count(*) AS resolved
            """,
            facility=facility,
        )
        refs_resolved = resolve_result[0]["resolved"] if resolve_result else 0

        # Step 2: Propagate code evidence to FacilitySignals
        # Find signals whose SOURCE_NODE TreeNode has DataReferences from code
        link_result = gc.query(
            """
            MATCH (dr:DataReference {facility_id: $facility})
                  -[:RESOLVES_TO_TREE_NODE]->(tn:TreeNode)
                  <-[:SOURCE_NODE]-(sig:FacilitySignal {facility_id: $facility})
            WITH sig, count(DISTINCT dr) AS ref_count
            SET sig.code_evidence_count = ref_count,
                sig.has_code_evidence = true
            RETURN count(sig) AS signals_linked
            """,
            facility=facility,
        )
        signals_linked = link_result[0]["signals_linked"] if link_result else 0

        # Step 3: Mark processed CodeFiles as evidence_linked
        gc.query(
            """
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'ingested'
              AND coalesce(sf.evidence_linked, false) = false
            SET sf.evidence_linked = true
            """,
            facility=facility,
        )

        return {
            "refs_resolved": refs_resolved,
            "signals_linked": signals_linked,
        }


__all__ = [
    "CLAIM_TIMEOUT_SECONDS",
    "claim_files_for_enrichment",
    "claim_files_for_scoring",
    "claim_files_for_triage",
    "claim_paths_for_file_scan",
    "has_pending_code_work",
    "has_pending_enrich_work",
    "has_pending_link_work",
    "has_pending_scan_work",
    "has_pending_score_work",
    "has_pending_triage_work",
    "link_code_evidence_to_signals",
    "release_file_enrich_claims",
    "release_file_score_claim",
    "release_file_score_claims",
    "release_file_triage_claim",
    "release_file_triage_claims",
    "release_path_file_scan_claim",
    "release_path_file_scan_claims_batch",
    "reset_orphaned_file_claims",
]
