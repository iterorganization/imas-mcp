"""File scanner for discovering source files in scored FacilityPaths.

Enumerates files at remote facilities using SSH, filtering by supported
extensions. Uses depth=1 since the paths pipeline has already walked
subdirectories — each FacilityPath is processed at its own level only.

Combines file enumeration with rg pattern matching in a single SSH call
via the discover_files.py remote script, providing per-file enrichment
evidence that feeds into dual-pass LLM scoring.

Creates SourceFile nodes with status='discovered' and per-file enrichment
data (pattern matches, line counts).
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable

from imas_codex.graph import GraphClient
from imas_codex.ingestion.readers.remote import (
    ALL_SUPPORTED_EXTENSIONS,
    detect_file_category,
    detect_language,
)

logger = logging.getLogger(__name__)

# Progress callback: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]

# Default file size limit: 1 MB — typical user analysis code is small.
# Machine data files, large caches, binaries are excluded.
DEFAULT_MAX_FILE_SIZE = 1 * 1024 * 1024

# Max files per path — much lower now that we scan at depth=1
DEFAULT_MAX_FILES_PER_PATH = 500


def _get_extensions_list() -> list[str]:
    """Get sorted list of supported file extensions (without dots)."""
    return sorted({e.lstrip(".").lower() for e in ALL_SUPPORTED_EXTENSIONS})


def _get_pattern_categories() -> dict[str, str]:
    """Get pattern categories from the paths enrichment PATTERN_REGISTRY.

    Flattens into category → regex for rg execution on remote host.
    """
    from imas_codex.discovery.paths.enrichment import PATTERN_REGISTRY

    flat: dict[str, str] = {}
    for _category, (patterns_dict, _score_dim) in PATTERN_REGISTRY.items():
        for name, regex in patterns_dict.items():
            flat[name] = regex
    return flat


def _scan_remote_paths_batch(
    facility: str,
    remote_paths: list[str],
    ssh_host: str | None = None,
    max_depth: int = 1,
    timeout: int = 300,
    max_files_per_path: int = DEFAULT_MAX_FILES_PER_PATH,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> dict[str, list[dict]]:
    """Scan multiple remote paths for files with enrichment in a single SSH call.

    Uses the discover_files.py remote script that combines file enumeration
    (fd/find at depth=1) with rg pattern matching. Each file gets per-file
    enrichment data (pattern matches, line counts).

    Args:
        facility: Facility ID
        remote_paths: Remote directory paths to scan
        ssh_host: SSH host alias (defaults to facility)
        max_depth: Maximum directory depth (default 1 = files at this level only)
        timeout: SSH command timeout in seconds
        max_files_per_path: Maximum files per path
        max_file_size: Maximum file size in bytes (default 1 MB)

    Returns:
        Dict mapping path -> list of file info dicts with enrichment data
    """
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.remote.executor import run_python_script

    if not remote_paths:
        return {}

    host = ssh_host
    if not host:
        try:
            config = get_facility(facility)
            host = config.get("ssh_host", facility)
        except ValueError:
            host = facility

    input_data = {
        "paths": remote_paths,
        "extensions": _get_extensions_list(),
        "max_depth": max_depth,
        "max_files_per_path": max_files_per_path,
        "max_file_size": max_file_size,
        "pattern_categories": _get_pattern_categories(),
    }

    try:
        output = run_python_script(
            "discover_files.py",
            input_data=input_data,
            ssh_host=host,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "Batch scan timed out after %ds for %s (%d paths)",
            timeout,
            facility,
            len(remote_paths),
        )
        return {p: [] for p in remote_paths}
    except subprocess.CalledProcessError as e:
        logger.warning("Batch scan failed for %s: %s", facility, e)
        return {p: [] for p in remote_paths}

    # Parse JSON output
    try:
        if "[stderr]:" in output:
            output = output.split("[stderr]:")[0].strip()
        results_data = json.loads(output)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse scan output for %s: %s", facility, e)
        return {p: [] for p in remote_paths}

    # Convert to file info dicts with enrichment data
    result_map: dict[str, list[dict]] = {}
    for entry in results_data:
        path = entry.get("path", "")
        error = entry.get("error")
        if error:
            logger.warning("Scan error for %s:%s: %s", facility, path, error)
            result_map[path] = []
            continue

        files = []
        for file_info in entry.get("files", []):
            # file_info is now a dict with path, patterns, total_matches, line_count
            if isinstance(file_info, str):
                # Backwards compatibility with old scan_files.py format
                file_path = file_info
                patterns = {}
                total_matches = 0
                line_count = 0
            else:
                file_path = file_info.get("path", "")
                patterns = file_info.get("patterns", {})
                total_matches = file_info.get("total_matches", 0)
                line_count = file_info.get("line_count", 0)

            files.append(
                {
                    "path": file_path,
                    "language": detect_language(file_path),
                    "file_category": detect_file_category(file_path),
                    "facility_id": facility,
                    "patterns": patterns,
                    "total_matches": total_matches,
                    "line_count": line_count,
                }
            )

        if entry.get("truncated"):
            logger.warning(
                "Path %s:%s truncated at %d files",
                facility,
                path,
                max_files_per_path,
            )

        if files:
            logger.info("Found %d files in %s:%s", len(files), facility, path)

        result_map[path] = files

    return result_map


async def async_scan_remote_paths_batch(
    facility: str,
    remote_paths: list[str],
    ssh_host: str | None = None,
    max_depth: int = 1,
    timeout: int = 300,
    max_files_per_path: int = DEFAULT_MAX_FILES_PER_PATH,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> dict[str, list[dict]]:
    """Async version of _scan_remote_paths_batch."""
    import asyncio

    return await asyncio.to_thread(
        _scan_remote_paths_batch,
        facility,
        remote_paths,
        ssh_host=ssh_host,
        max_depth=max_depth,
        timeout=timeout,
        max_files_per_path=max_files_per_path,
        max_file_size=max_file_size,
    )


def _scan_remote_path(
    facility: str,
    remote_path: str,
    ssh_host: str | None = None,
    max_depth: int = 1,
    timeout: int = 60,
    max_files_per_path: int = DEFAULT_MAX_FILES_PER_PATH,
) -> list[dict]:
    """Scan a single remote path for supported files via SSH.

    Wraps _scan_remote_paths_batch for single-path convenience.
    """
    result_map = _scan_remote_paths_batch(
        facility,
        [remote_path],
        ssh_host=ssh_host,
        max_depth=max_depth,
        timeout=timeout,
        max_files_per_path=max_files_per_path,
    )
    return result_map.get(remote_path, [])


def _get_scannable_paths(
    facility: str,
    min_score: float = 0.5,
    limit: int = 100,
) -> list[dict]:
    """Get scored FacilityPaths that are ready for file scanning.

    .. deprecated::
        Use :func:`~imas_codex.discovery.files.graph_ops.claim_paths_for_file_scan`
        instead for parallel-safe claiming.

    Returns paths with status 'scored' and score >= min_score,
    ordered by score descending.
    """
    with GraphClient() as client:
        result = client.query(
            """
            MATCH (p:FacilityPath {facility_id: $facility})
            WHERE p.status IN ['scored', 'explored']
              AND coalesce(p.score, 0) >= $min_score
              AND p.path IS NOT NULL
            RETURN p.id AS id, p.path AS path, p.score AS score,
                   p.purpose AS purpose, coalesce(p.files_scanned, 0) AS files_scanned
            ORDER BY p.score DESC
            LIMIT $limit
            """,
            facility=facility,
            min_score=min_score,
            limit=limit,
        )
        return list(result)


def _persist_discovered_files(
    facility: str,
    files: list[dict],
    source_path_id: str | None = None,
) -> dict[str, int]:
    """Create SourceFile nodes from scanned files with enrichment data.

    Stores per-file enrichment data (pattern matches, line counts) from
    the discover_files.py remote script so it's available for LLM scoring.

    Args:
        facility: Facility ID
        files: List of file info dicts (with patterns, total_matches, line_count)
        source_path_id: FacilityPath ID that contained these files

    Returns:
        Dict with discovered, skipped counts
    """
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    items = []
    for f in files:
        file_id = f"{facility}:{f['path']}"
        item = {
            "id": file_id,
            "facility_id": facility,
            "path": f["path"],
            "language": f.get("language", "python"),
            "file_category": f.get("file_category", "code"),
            "status": "discovered",
            "discovered_at": now,
            "in_directory": source_path_id,
        }
        # Store per-file enrichment from discover_files.py
        patterns = f.get("patterns")
        if patterns:
            item["pattern_categories"] = json.dumps(patterns)
            item["total_pattern_matches"] = f.get("total_matches", 0)
            item["is_enriched"] = True
        line_count = f.get("line_count", 0)
        if line_count:
            item["line_count"] = line_count
        items.append(item)

    if not items:
        return {"discovered": 0, "skipped": 0}

    # Batch large writes to avoid overwhelming Neo4j.
    # Keep batches small (100) to reduce lock contention with concurrent
    # score workers that claim SourceFile nodes.
    BATCH_SIZE = 100
    total_discovered = 0
    total_skipped = 0

    with GraphClient() as client:
        for batch_start in range(0, len(items), BATCH_SIZE):
            batch = items[batch_start : batch_start + BATCH_SIZE]

            # MERGE to skip already-discovered files.
            # Catch constraint violations from concurrent scan workers
            # racing on the same file id.
            try:
                result = client.query(
                    """
                    UNWIND $items AS item
                    MERGE (sf:SourceFile {id: item.id})
                    ON CREATE SET sf += item
                    WITH sf, item
                    WHERE sf.status = 'discovered' AND sf.discovered_at = item.discovered_at
                    MATCH (f:Facility {id: item.facility_id})
                    MERGE (sf)-[:AT_FACILITY]->(f)
                    RETURN count(CASE WHEN sf.discovered_at = item.discovered_at THEN 1 END) AS discovered,
                           count(CASE WHEN sf.discovered_at <> item.discovered_at THEN 1 END) AS skipped
                    """,
                    items=batch,
                )
            except Exception as e:
                if "ConstraintValidation" in str(
                    type(e).__name__
                ) or "ConstraintValidation" in str(e):
                    logger.debug(
                        "Constraint violation (concurrent scan), retrying individually: %s",
                        e,
                    )
                    result = [{"discovered": 0, "skipped": len(batch)}]
                else:
                    raise

            counts = result[0] if result else {"discovered": 0, "skipped": 0}
            total_discovered += counts.get("discovered", len(batch))
            total_skipped += counts.get("skipped", 0)

            # Link to parent FacilityPath
            if source_path_id:
                client.query(
                    """
                    UNWIND $items AS item
                    MATCH (sf:SourceFile {id: item.id})
                    MATCH (p:FacilityPath {id: $parent_id})
                    MERGE (p)-[:CONTAINS]->(sf)
                    """,
                    items=batch,
                    parent_id=source_path_id,
                )

        # Update FacilityPath scan count (once, after all batches)
        if source_path_id:
            client.query(
                """
                MATCH (p:FacilityPath {id: $parent_id})
                SET p.files_scanned = $count,
                    p.last_file_scan_at = datetime()
                """,
                parent_id=source_path_id,
                count=len(files),
            )

        return {
            "discovered": total_discovered,
            "skipped": total_skipped,
        }


def scan_facility_files(
    facility: str,
    min_score: float = 0.5,
    max_paths: int = 100,
    max_depth: int = 5,
    ssh_host: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Scan scored FacilityPaths for source files.

    Enumerates files in high-scoring directories, creates SourceFile nodes.
    Uses claim coordination via ``files_claimed_at`` on FacilityPath to
    enable safe parallel execution.

    Args:
        facility: Facility ID
        min_score: Minimum FacilityPath score to scan
        max_paths: Maximum paths to scan
        max_depth: Maximum directory depth per path
        ssh_host: SSH host override
        progress_callback: Optional progress callback

    Returns:
        Dict with total_files, total_paths, new_files, skipped_files
    """
    from imas_codex.discovery.files.graph_ops import (
        claim_paths_for_file_scan,
        release_path_file_scan_claim,
    )

    stats = {
        "total_files": 0,
        "total_paths": 0,
        "new_files": 0,
        "skipped_files": 0,
    }

    def report(current: int, total: int, msg: str) -> None:
        if progress_callback:
            progress_callback(current, total, msg)
        logger.info("[%d/%d] %s", current, total, msg)

    # Claim paths atomically (parallel-safe)
    paths = claim_paths_for_file_scan(facility, min_score=min_score, limit=max_paths)
    if not paths:
        report(0, 0, "No scored paths to scan (or all claimed by another worker)")
        return stats

    # Ensure Facility node exists so AT_FACILITY relationships don't fail
    with GraphClient() as gc:
        gc.ensure_facility(facility)

    total = len(paths)
    report(0, total, f"Scanning {total} paths for source files")

    for idx, path_info in enumerate(paths):
        path = path_info["path"]
        path_id = path_info["id"]
        score = path_info.get("score", 0)

        report(idx, total, f"Scanning {path} (score={score:.2f})")

        try:
            files = _scan_remote_path(
                facility, path, ssh_host=ssh_host, max_depth=max_depth
            )

            if files:
                result = _persist_discovered_files(
                    facility, files, source_path_id=path_id
                )
                stats["new_files"] += result["discovered"]
                stats["skipped_files"] += result["skipped"]
                stats["total_files"] += len(files)

            stats["total_paths"] += 1

        except Exception as e:
            logger.warning("Error scanning %s: %s", path, e)
            # Release claim on error so another worker can retry
            release_path_file_scan_claim(path_id)
            continue

        # Release claim after successful scan
        release_path_file_scan_claim(path_id)

    report(
        total,
        total,
        f"Scan complete: {stats['new_files']} new files in {stats['total_paths']} paths",
    )
    return stats


def get_file_discovery_stats(facility: str) -> dict[str, int]:
    """Get file discovery statistics for a facility.

    Returns:
        Dict with counts by status, total, plus path scan stats
    """
    with GraphClient() as client:
        result = client.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            RETURN sf.status AS status, count(*) AS count
            """,
            facility=facility,
        )
        stats = {row["status"]: row["count"] for row in result}
        stats["total"] = sum(stats.values())

        # Count by category
        cat_result = client.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            RETURN sf.file_category AS category, count(*) AS count
            """,
            facility=facility,
        )
        for row in cat_result:
            if row["category"]:
                stats[f"cat_{row['category']}"] = row["count"]

        return stats


def clear_facility_files(facility: str, batch_size: int = 1000) -> dict[str, int]:
    """Clear all SourceFile nodes and dependent nodes for a facility.

    Cascades in referential-integrity order:
    1. DataReference nodes linked to CodeChunks from facility SourceFiles
    2. CodeChunk nodes linked to CodeExamples from facility SourceFiles
    3. CodeExample nodes linked to facility SourceFiles
    4. SourceFile nodes by facility_id
    5. Orphaned Image nodes (no remaining HAS_IMAGE from other domains)

    Args:
        facility: Facility ID
        batch_size: Nodes to delete per batch (default 1000)

    Returns:
        Dict with counts of deleted nodes
    """
    results = {
        "source_files_deleted": 0,
        "code_chunks_deleted": 0,
    }

    with GraphClient() as client:
        # Delete CodeChunks (and their DataReferences cascade via DETACH)
        # linked through CodeExample -> SourceFile chain
        while True:
            result = client.query(
                """
                MATCH (sf:SourceFile {facility_id: $facility})
                      <-[:FROM_FILE]-(ce)
                      -[:HAS_CHUNK]->(cc:CodeChunk)
                WITH cc LIMIT $batch_size
                DETACH DELETE cc
                RETURN count(cc) AS deleted
                """,
                facility=facility,
                batch_size=batch_size,
            )
            deleted = result[0]["deleted"] if result else 0
            results["code_chunks_deleted"] += deleted
            if deleted < batch_size:
                break

        # Delete CodeExample nodes linked to facility SourceFiles
        while True:
            result = client.query(
                """
                MATCH (sf:SourceFile {facility_id: $facility})
                      <-[:FROM_FILE]-(ce)
                WITH ce LIMIT $batch_size
                DETACH DELETE ce
                RETURN count(ce) AS deleted
                """,
                facility=facility,
                batch_size=batch_size,
            )
            deleted = result[0]["deleted"] if result else 0
            if deleted < batch_size:
                break

        # Delete SourceFile nodes in batches
        while True:
            result = client.query(
                """
                MATCH (sf:SourceFile {facility_id: $facility})
                WITH sf LIMIT $batch_size
                DETACH DELETE sf
                RETURN count(sf) AS deleted
                """,
                facility=facility,
                batch_size=batch_size,
            )
            deleted = result[0]["deleted"] if result else 0
            results["source_files_deleted"] += deleted
            if deleted < batch_size:
                break

        # Delete orphaned Image nodes (no remaining HAS_IMAGE from any domain)
        images_deleted = 0
        while True:
            result = client.query(
                """
                MATCH (img:Image {facility_id: $facility})
                WHERE NOT (img)<-[:HAS_IMAGE]-()
                WITH img LIMIT $batch_size
                DETACH DELETE img
                RETURN count(img) AS deleted
                """,
                facility=facility,
                batch_size=batch_size,
            )
            deleted = result[0]["deleted"] if result else 0
            images_deleted += deleted
            if deleted < batch_size:
                break
        if images_deleted > 0:
            results["images_deleted"] = images_deleted

        logger.info(
            "Deleted %d SourceFile + %d CodeChunk nodes for %s",
            results["source_files_deleted"],
            results["code_chunks_deleted"],
            facility,
        )

    return results


__all__ = [
    "clear_facility_files",
    "get_file_discovery_stats",
    "scan_facility_files",
]
