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
              AND coalesce(p.score, 0) >= $min_score
              AND p.path IS NOT NULL
              AND (p.files_claimed_at IS NULL
                   OR p.files_claimed_at < datetime() - duration($cutoff))
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
            ) DESC, p.score DESC
            LIMIT $limit
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
# Score-phase claiming (SourceFile with claimed_at)
# ---------------------------------------------------------------------------


def claim_files_for_scoring(
    facility: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Atomically claim discovered SourceFiles for LLM scoring.

    Claims files with ``status='discovered'`` and no ``interest_score``.
    Uses ``claimed_at`` timeout for orphan recovery.

    Joins through CONTAINS → FacilityPath to return parent directory
    enrichment data (pattern matches, scores, description) so the
    scorer can group files by parent and include enrichment context.

    Files are ordered by parent path score descending so files from
    the highest-value directories are scored first.

    Args:
        facility: Facility ID
        limit: Maximum files to claim

    Returns:
        List of dicts with file info + parent enrichment fields
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
            OPTIONAL MATCH (p:FacilityPath)-[:CONTAINS]->(sf)
            WITH sf, p
            ORDER BY coalesce(p.score, 0) DESC, sf.discovered_at ASC
            LIMIT $limit
            SET sf.claimed_at = datetime()
            RETURN sf.id AS id, sf.path AS path,
                   sf.language AS language, sf.file_category AS file_category,
                   sf.pattern_categories AS patterns_json,
                   sf.total_pattern_matches AS total_matches,
                   sf.line_count AS line_count,
                   sf.is_enriched AS is_enriched,
                   p.id AS parent_path_id, p.path AS parent_path,
                   p.score AS parent_score, p.purpose AS parent_purpose,
                   p.description AS parent_description,
                   p.pattern_categories AS parent_patterns,
                   p.read_matches AS parent_read_matches,
                   p.write_matches AS parent_write_matches,
                   p.is_multiformat AS parent_multiformat,
                   p.score_data_access AS parent_score_data_access,
                   p.score_imas AS parent_score_imas,
                   p.score_convention AS parent_score_convention,
                   p.score_analysis_code AS parent_score_analysis_code,
                   p.score_modeling_code AS parent_score_modeling_code,
                   p.score_operations_code AS parent_score_operations_code,
                   p.score_workflow AS parent_score_workflow,
                   p.score_visualization AS parent_score_visualization,
                   p.score_documentation AS parent_score_documentation
            """,
            facility=facility,
            limit=limit,
            cutoff=cutoff,
        )
        files = []
        for row in result:
            f = dict(row)
            # Parse per-file pattern_categories JSON into a dict
            patterns_json = f.pop("patterns_json", None)
            if patterns_json:
                try:
                    import json

                    f["patterns"] = json.loads(patterns_json)
                except (json.JSONDecodeError, TypeError):
                    f["patterns"] = {}
            else:
                f["patterns"] = {}
            f.setdefault("total_matches", 0)
            f.setdefault("line_count", 0)
            files.append(f)
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


# ---------------------------------------------------------------------------
# Enrich-phase claiming (SourceFile with claimed_at, scored + not enriched)
# ---------------------------------------------------------------------------


def claim_files_for_enrichment(
    facility: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Atomically claim scored SourceFiles for rg pattern enrichment.

    Claims files that have been scored (interest_score IS NOT NULL) but
    not yet enriched (is_enriched IS NULL or false). Prioritizes files
    with high composite scores.

    Args:
        facility: Facility ID
        limit: Maximum files to claim

    Returns:
        List of dicts with id, path, language, interest_score
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.interest_score IS NOT NULL
              AND coalesce(sf.is_enriched, false) = false
              AND (sf.claimed_at IS NULL
                   OR sf.claimed_at < datetime() - duration($cutoff))
            WITH sf ORDER BY sf.interest_score DESC LIMIT $limit
            SET sf.claimed_at = datetime()
            RETURN sf.id AS id, sf.path AS path,
                   sf.language AS language,
                   sf.interest_score AS interest_score
            """,
            facility=facility,
            limit=limit,
            cutoff=cutoff,
        )
        files = list(result)
        if files:
            logger.debug(
                "Claimed %d SourceFiles for enrichment (facility=%s)",
                len(files),
                facility,
            )
        return files


def release_file_enrich_claims(file_ids: list[str]) -> None:
    """Release enrichment claims on multiple SourceFiles."""
    release_claims_batch("SourceFile", file_ids)


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
              AND coalesce(p.score, 0) >= $min_score
              AND p.path IS NOT NULL
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
    """Check if there are SourceFiles needing LLM scoring."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.interest_score IS NULL
            RETURN count(sf) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


def has_pending_enrich_work(facility: str) -> bool:
    """Check if there are scored SourceFiles needing rg enrichment."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.interest_score IS NOT NULL
              AND coalesce(sf.is_enriched, false) = false
            RETURN count(sf) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


def has_pending_code_work(facility: str, min_score: float = 0.0) -> bool:
    """Check if there are scored code files needing ingestion."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.interest_score IS NOT NULL
              AND sf.interest_score >= $min_score
              AND sf.file_category = 'code'
            RETURN count(sf) > 0 AS has_work
            """,
            facility=facility,
            min_score=min_score,
        )
        return result[0]["has_work"] if result else False


def has_pending_docs_work(facility: str, min_score: float = 0.0) -> bool:
    """Check if there are scored document/notebook/config files needing ingestion."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.interest_score IS NOT NULL
              AND sf.interest_score >= $min_score
              AND sf.file_category IN ['document', 'notebook', 'config']
            RETURN count(sf) > 0 AS has_work
            """,
            facility=facility,
            min_score=min_score,
        )
        return result[0]["has_work"] if result else False


def has_pending_image_work(facility: str) -> bool:
    """Check if there are image files needing ingestion."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.interest_score IS NOT NULL
              AND sf.file_category = 'image'
            RETURN count(sf) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


def has_pending_image_score_work(facility: str) -> bool:
    """Check if there are ingested images needing VLM scoring."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (img:Image {facility_id: $facility})
            WHERE img.status = 'ingested'
              AND img.description IS NULL
            RETURN count(img) > 0 AS has_work
            """,
            facility=facility,
        )
        return result[0]["has_work"] if result else False


__all__ = [
    "CLAIM_TIMEOUT_SECONDS",
    "claim_files_for_enrichment",
    "claim_files_for_scoring",
    "claim_paths_for_file_scan",
    "has_pending_code_work",
    "has_pending_docs_work",
    "has_pending_enrich_work",
    "has_pending_image_score_work",
    "has_pending_image_work",
    "has_pending_scan_work",
    "has_pending_score_work",
    "release_file_enrich_claims",
    "release_file_score_claim",
    "release_file_score_claims",
    "release_path_file_scan_claim",
    "release_path_file_scan_claims_batch",
    "reset_orphaned_file_claims",
]
