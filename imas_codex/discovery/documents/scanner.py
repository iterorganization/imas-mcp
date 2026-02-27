"""Document scanner: enumerate document + image files from scored FacilityPaths.

Creates Document nodes for non-code files (PDF, markdown, notebooks, images, etc.).
Reuses the SSH scanning infrastructure from the code scanner with different
file extension sets.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from imas_codex.graph import GraphClient
from imas_codex.ingestion.readers.remote import (
    DOCUMENT_EXTENSIONS,
    IMAGE_EXTENSIONS,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

# Document + image extensions combined
_DOCUMENT_AND_IMAGE_EXTENSIONS: set[str] = set(DOCUMENT_EXTENSIONS) | IMAGE_EXTENSIONS

# Extension → DocumentType mapping
_EXT_TO_DOCUMENT_TYPE: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "spreadsheet",  # Word docs → closer to spreadsheet than presentation
    ".pptx": "presentation",
    ".xlsx": "spreadsheet",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".rst": "text",
    ".txt": "text",
    ".ipynb": "notebook",
    # Image extensions
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".svg": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
    ".webp": "image",
}


def detect_document_type(path: str) -> str:
    """Detect document type from file extension.

    Returns DocumentType enum value.
    """
    from pathlib import PurePosixPath

    ext = PurePosixPath(path).suffix.lower()
    return _EXT_TO_DOCUMENT_TYPE.get(ext, "other")


def _get_document_extensions_list() -> list[str]:
    """Get sorted list of document + image file extensions (without dots)."""
    return sorted({e.lstrip(".").lower() for e in _DOCUMENT_AND_IMAGE_EXTENSIONS})


def _persist_document_files(
    facility: str,
    files: list[dict],
    source_path_id: str | None = None,
) -> dict[str, int]:
    """Create Document nodes from scanned document/image files.

    Args:
        facility: Facility ID
        files: List of file info dicts with path, document_type
        source_path_id: FacilityPath ID that contained these files

    Returns:
        Dict with discovered, skipped counts
    """
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    items = []
    for f in files:
        file_id = f"{facility}:{f['path']}"
        doc_type = f.get("document_type", detect_document_type(f["path"]))
        item = {
            "id": file_id,
            "facility_id": facility,
            "path": f["path"],
            "document_type": doc_type,
            "status": "discovered",
            "discovered_at": now,
            "in_directory": source_path_id,
        }
        items.append(item)

    if not items:
        return {"discovered": 0, "skipped": 0}

    BATCH_SIZE = 100
    total_discovered = 0
    total_skipped = 0

    with GraphClient() as client:
        for batch_start in range(0, len(items), BATCH_SIZE):
            batch = items[batch_start : batch_start + BATCH_SIZE]

            try:
                result = client.query(
                    """
                    UNWIND $items AS item
                    MERGE (d:Document {id: item.id})
                    ON CREATE SET d += item
                    WITH d, item
                    WHERE d.status = 'discovered' AND d.discovered_at = item.discovered_at
                    MATCH (f:Facility {id: item.facility_id})
                    MERGE (d)-[:AT_FACILITY]->(f)
                    RETURN count(CASE WHEN d.discovered_at = item.discovered_at THEN 1 END) AS discovered,
                           count(CASE WHEN d.discovered_at <> item.discovered_at THEN 1 END) AS skipped
                    """,
                    items=batch,
                )
            except Exception as e:
                if "ConstraintValidation" in str(
                    type(e).__name__
                ) or "ConstraintValidation" in str(e):
                    logger.debug("Constraint violation (concurrent scan): %s", e)
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
                    MATCH (d:Document {id: item.id})
                    MATCH (p:FacilityPath {id: $parent_id})
                    MERGE (p)-[:CONTAINS]->(d)
                    """,
                    items=batch,
                    parent_id=source_path_id,
                )

    return {"discovered": total_discovered, "skipped": total_skipped}


def scan_facility_documents(
    facility: str,
    min_score: float = 0.5,
    max_paths: int = 100,
    ssh_host: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Scan scored FacilityPaths for document and image files.

    Enumerates non-code files in high-scoring directories, creates Document nodes.

    Args:
        facility: Facility ID
        min_score: Minimum FacilityPath score to scan
        max_paths: Maximum paths to scan
        ssh_host: SSH host override
        progress_callback: Optional progress callback

    Returns:
        Dict with total_files, total_paths, new_files, skipped_files
    """
    from imas_codex.discovery.code.graph_ops import (
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

    paths = claim_paths_for_file_scan(facility, min_score=min_score, limit=max_paths)
    if not paths:
        report(0, 0, "No scored paths to scan")
        return stats

    with GraphClient() as gc:
        gc.ensure_facility(facility)

    total = len(paths)
    report(0, total, f"Scanning {total} paths for documents")

    for idx, path_info in enumerate(paths):
        path = path_info["path"]
        path_id = path_info["id"]

        report(idx, total, f"Scanning {path}")

        try:
            files = _scan_remote_path_for_documents(
                facility,
                path,
                ssh_host=ssh_host,
            )

            if files:
                result = _persist_document_files(
                    facility, files, source_path_id=path_id
                )
                stats["new_files"] += result["discovered"]
                stats["skipped_files"] += result["skipped"]
                stats["total_files"] += len(files)

            stats["total_paths"] += 1

        except Exception as e:
            logger.warning("Error scanning %s: %s", path, e)
            release_path_file_scan_claim(path_id)
            continue

        release_path_file_scan_claim(path_id)

    report(
        total,
        total,
        f"Scan complete: {stats['new_files']} new documents in {stats['total_paths']} paths",
    )
    return stats


def _scan_remote_path_for_documents(
    facility: str,
    remote_path: str,
    ssh_host: str | None = None,
    max_depth: int = 1,
    timeout: int = 60,
) -> list[dict]:
    """Scan a remote path for document + image files via SSH fd."""
    import subprocess

    from imas_codex.discovery.base.facility import get_facility

    if ssh_host is None:
        config = get_facility(facility)
        ssh_host = config.get("ssh_host", facility)

    extensions = _get_document_extensions_list()
    ext_args = " ".join(f"-e {ext}" for ext in extensions)

    cmd = f"fd --type f --max-depth {max_depth} {ext_args} '{remote_path}' 2>/dev/null"

    try:
        result = subprocess.run(
            ["ssh", ssh_host, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Document scan timed out for %s:%s", facility, remote_path)
        return []

    if not result.stdout.strip():
        return []

    files = []
    for line in result.stdout.strip().splitlines():
        file_path = line.strip()
        if file_path:
            files.append(
                {
                    "path": file_path,
                    "document_type": detect_document_type(file_path),
                    "facility_id": facility,
                }
            )

    return files


def get_document_discovery_stats(facility: str) -> dict[str, int]:
    """Get document discovery statistics for a facility."""
    with GraphClient() as client:
        result = client.query(
            """
            MATCH (d:Document)-[:AT_FACILITY]->(f:Facility {id: $facility})
            RETURN d.status AS status, count(*) AS count
            """,
            facility=facility,
        )
        stats: dict[str, int] = {row["status"]: row["count"] for row in result}
        stats["total"] = sum(stats.values())

        type_result = client.query(
            """
            MATCH (d:Document)-[:AT_FACILITY]->(f:Facility {id: $facility})
            RETURN d.document_type AS doc_type, count(*) AS count
            """,
            facility=facility,
        )
        for row in type_result:
            if row["doc_type"]:
                stats[f"type_{row['doc_type']}"] = row["count"]

        # Image stats
        image_result = client.query(
            """
            MATCH (img:Image {facility_id: $facility})
            RETURN count(img) AS total_images,
                   count(CASE WHEN img.description IS NOT NULL THEN 1 END) AS scored_images
            """,
            facility=facility,
        )
        if image_result:
            stats["total_images"] = image_result[0]["total_images"]
            stats["scored_images"] = image_result[0]["scored_images"]

        return stats


def clear_facility_documents(facility: str, batch_size: int = 1000) -> dict[str, int]:
    """Clear all Document nodes and orphaned Images for a facility.

    Args:
        facility: Facility ID
        batch_size: Nodes per batch

    Returns:
        Dict with counts of deleted nodes
    """
    with GraphClient() as client:
        doc_result = client.query(
            """
            MATCH (d:Document {facility_id: $facility})
            WITH d LIMIT $batch
            DETACH DELETE d
            RETURN count(*) AS deleted
            """,
            facility=facility,
            batch=batch_size,
        )
        docs_deleted = doc_result[0]["deleted"] if doc_result else 0

        # Delete remaining in batches
        while docs_deleted > 0:
            more = client.query(
                """
                MATCH (d:Document {facility_id: $facility})
                WITH d LIMIT $batch
                DETACH DELETE d
                RETURN count(*) AS deleted
                """,
                facility=facility,
                batch=batch_size,
            )
            batch_deleted = more[0]["deleted"] if more else 0
            if batch_deleted == 0:
                break
            docs_deleted += batch_deleted

        # Clean orphaned filesystem Images (not linked to any Document or WikiPage)
        orphan_result = client.query(
            """
            MATCH (img:Image {facility_id: $facility, source_type: 'filesystem'})
            WHERE NOT (img)<-[:HAS_IMAGE]-()
            WITH img LIMIT $batch
            DETACH DELETE img
            RETURN count(*) AS deleted
            """,
            facility=facility,
            batch=batch_size,
        )
        images_deleted = orphan_result[0]["deleted"] if orphan_result else 0

        return {
            "documents_deleted": docs_deleted,
            "images_deleted": images_deleted,
        }
