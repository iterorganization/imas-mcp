"""Graph operations for file discovery claim coordination.

Provides claim/release functions for both scan and score phases of
file discovery, enabling safe parallel execution of ``discover files``.

Scan phase: Claims FacilityPath nodes via ``files_claimed_at`` (separate
from the paths module's ``claimed_at``) to prevent duplicate SSH scans.

Score phase: Claims SourceFile nodes via ``claimed_at`` to prevent
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
        Dict with ``source_file_reset`` and ``facility_path_reset`` counts
    """
    # SourceFile claims (scoring phase)
    sf_reset = reset_stale_claims(
        "SourceFile",
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
            "Released %d orphaned file claims (%d SourceFile, %d FacilityPath)",
            total,
            sf_reset,
            fp_reset,
        )

    return {"source_file_reset": sf_reset, "facility_path_reset": fp_reset}


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

    Skips paths that have already been scanned (files_scanned > 0) and paths
    linked to publicly-accessible software repos (INSTANCE_OF → SoftwareRepo
    with remote_url containing github/gitlab/bitbucket).

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
              AND coalesce(p.score, 0) >= $min_score
              AND p.path IS NOT NULL
              AND p.last_file_scan_at IS NULL
              AND (p.files_claimed_at IS NULL
                   OR p.files_claimed_at < datetime() - duration($cutoff))
              AND NOT EXISTS {
                MATCH (p)-[:INSTANCE_OF]->(r:SoftwareRepo)
                WHERE r.source_type IN ['github', 'gitlab', 'bitbucket']
              }
            WITH p ORDER BY p.score DESC LIMIT $limit
            SET p.files_claimed_at = datetime()
            RETURN p.id AS id, p.path AS path, p.score AS score,
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


# ---------------------------------------------------------------------------
# Score-phase claiming (SourceFile with claimed_at)
# ---------------------------------------------------------------------------


def claim_files_for_scoring(
    facility: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Atomically claim discovered SourceFiles for LLM scoring.

    Claims files with ``status='discovered'`` and no ``interest_score``.
    Uses ``claimed_at`` timeout for orphan recovery.

    Files are ordered by ``path_heuristic_score`` descending so the
    highest-value files (by path pattern matching) are scored first.

    Args:
        facility: Facility ID
        limit: Maximum files to claim

    Returns:
        List of dicts with ``id``, ``path``, ``language``, ``file_category``
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.interest_score IS NULL
              AND (sf.claimed_at IS NULL
                   OR sf.claimed_at < datetime() - duration($cutoff))
            WITH sf ORDER BY coalesce(sf.path_heuristic_score, 0) DESC,
                             sf.discovered_at ASC
            LIMIT $limit
            SET sf.claimed_at = datetime()
            RETURN sf.id AS id, sf.path AS path, sf.language AS language,
                   sf.file_category AS file_category
            """,
            facility=facility,
            limit=limit,
            cutoff=cutoff,
        )
        files = list(result)
        if files:
            logger.debug(
                "Claimed %d SourceFiles for scoring (facility=%s)",
                len(files),
                facility,
            )
        return files


def release_file_score_claim(file_id: str) -> None:
    """Release scoring claim on a single SourceFile."""
    release_claim("SourceFile", file_id)


def release_file_score_claims(file_ids: list[str]) -> None:
    """Release scoring claims on multiple SourceFiles."""
    release_claims_batch("SourceFile", file_ids)


__all__ = [
    "CLAIM_TIMEOUT_SECONDS",
    "claim_files_for_scoring",
    "claim_paths_for_file_scan",
    "release_file_score_claim",
    "release_file_score_claims",
    "release_path_file_scan_claim",
    "release_path_file_scan_claims_batch",
    "reset_orphaned_file_claims",
]
