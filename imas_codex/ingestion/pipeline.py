"""Unified ingestion pipeline for all content types.

Graph-driven ingestion via CodeFile queue:
- Automatic deduplication (skips already-ingested files)
- Per-file atomic commits (interrupt-safe)
- Auto-updates FacilityPath status to 'explored'
- Links extracted MDSplus paths to SignalNode entities
- Routes files to appropriate splitters based on language/type

Replaces code_examples.pipeline.
"""

import asyncio
import hashlib
import logging
import re
import time as _time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_codex.graph import GraphClient

if TYPE_CHECKING:
    from imas_codex.embeddings.encoder import Encoder

from .chunkers import chunk_code, chunk_text
from .extractors.ids import extract_ids_references
from .extractors.mdsplus import extract_mdsplus_paths
from .graph import (
    link_chunks_to_data_nodes,
    link_chunks_to_imas_paths,
    link_examples_to_facility,
)
from .queue import get_pending_files, update_source_file_status
from .readers.remote import TEXT_SPLITTER_LANGUAGES, fetch_remote_files

logger = logging.getLogger(__name__)

# Progress callback type: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]

# Target chunk size in characters for code chunking.
# Matches embed worker's TARGET_EMBED_TEXT_CHARS to avoid segment splitting
# at embed time.  4000 chars ≈ 1K tokens for Qwen3-Embedding.
DEFAULT_CHUNK_MAX_CHARS = 4000


def _split_and_extract(
    content: str,
    language: str,
    metadata: dict[str, Any],
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    chunk_lines: int = 40,
    chunk_lines_overlap: int = 10,
    use_text_splitter: bool = False,
) -> list[dict[str, Any]]:
    """Split content into chunks and extract metadata.

    Replaces LlamaIndex IngestionPipeline: splits text using tree-sitter
    or sliding window, then runs IDS and MDSplus extraction.

    Args:
        content: Source code text
        language: Programming language
        metadata: Base metadata to attach to each chunk
        max_chars: Maximum characters per chunk
        chunk_lines: Target lines per chunk
        chunk_lines_overlap: Overlap lines between chunks
        use_text_splitter: Force text-based splitting

    Returns:
        List of chunk dicts with text, metadata, and extracted references
    """
    if use_text_splitter or language in TEXT_SPLITTER_LANGUAGES:
        chunks = chunk_text(
            content,
            chunk_size=max_chars,
            chunk_overlap=chunk_lines_overlap * 60,
        )
    else:
        chunks = chunk_code(
            content,
            language=language,
            max_chars=max_chars,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
        )

    result: list[dict[str, Any]] = []
    for chunk in chunks:
        # Extract IDS references
        ids_refs = extract_ids_references(chunk.text)
        related_ids = sorted(ids_refs) if ids_refs else []

        # Extract MDSplus paths
        mdsplus_refs = extract_mdsplus_paths(chunk.text)
        mdsplus_paths = [r.path for r in mdsplus_refs]

        chunk_dict: dict[str, Any] = {
            "text": chunk.text,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            **metadata,
        }
        if related_ids:
            chunk_dict["related_ids"] = related_ids
            chunk_dict["related_ids_count"] = len(related_ids)
        if mdsplus_paths:
            chunk_dict["mdsplus_paths"] = mdsplus_paths
            chunk_dict["mdsplus_ref_count"] = len(mdsplus_paths)

        result.append(chunk_dict)

    return result


def _generate_example_id(facility: str, remote_path: str) -> str:
    """Generate unique ID for a code example."""
    content = f"{facility}:{remote_path}"
    hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{facility}:{Path(remote_path).stem}:{hash_suffix}"


def _extract_author(path: str) -> str | None:
    """Extract username from path like /home/username/..."""
    match = re.match(r"/home/(\w+)/", path)
    return match.group(1) if match else None


def _check_already_ingested(
    graph_client: GraphClient,
    facility: str,
    remote_paths: list[str],
) -> tuple[list[str], list[str]]:
    """Check which files are already ingested."""
    result = graph_client.query(
        """
        MATCH (e:CodeExample)
        WHERE e.facility_id = $facility AND e.source_file IN $paths
        RETURN e.source_file AS path
        """,
        facility=facility,
        paths=remote_paths,
    )

    already_ingested = {r["path"] for r in result}
    to_ingest = [p for p in remote_paths if p not in already_ingested]

    return to_ingest, list(already_ingested)


async def ingest_files(
    facility: str,
    remote_paths: list[str] | None = None,
    description: str | None = None,
    progress_callback: ProgressCallback | None = None,
    force: bool = False,
    limit: int | None = None,
    encoder: "Encoder | None" = None,
) -> dict[str, int]:
    """Ingest files from a remote facility.

    Can be called in two modes:
    1. **Path list mode**: Provide remote_paths explicitly
    2. **Graph-driven mode**: Omit remote_paths to process queued CodeFile nodes

    Embedding is deferred: CodeChunk nodes are written with
    ``embedding = null``.  The ``embed_text_worker`` populates
    embeddings asynchronously on the GPU.

    Features:
    - Deduplication: Skips files that are already ingested (unless force=True)
    - Interrupt-safe: Each file is committed atomically
    - Auto status update: CodeFile nodes are marked 'ingested'
    - MDSplus linking: Extracted paths are linked to SignalNode entities

    Args:
        facility: Facility SSH host alias (e.g., "tcv")
        remote_paths: List of remote file paths to ingest (if None, uses graph queue)
        description: Optional description for all files
        progress_callback: Optional callback for progress reporting
        force: If True, re-ingest files even if already present
        limit: Maximum files to process from graph queue
        encoder: Unused (kept for backward compatibility).  Embedding
            is now handled by the ``embed_text_worker``.

    Returns:
        Dict with counts: files, chunks, ids_found, mdsplus_paths, skipped, data_nodes_linked
    """
    stats = {
        "files": 0,
        "chunks": 0,
        "ids_found": 0,
        "mdsplus_paths": 0,
        "skipped": 0,
        "data_nodes_linked": 0,
    }

    def report(current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(current, total, message)
        logger.info("[%d/%d] %s", current, total, message)

    # Determine source of files
    source_file_ids: dict[str, str] = {}

    if remote_paths is None:
        query_limit = limit if limit is not None else 10000
        pending = get_pending_files(facility, limit=query_limit)
        if not pending:
            report(0, 0, "No pending files in queue")
            return stats

        remote_paths = [p["path"] for p in pending]
        source_file_ids = {p["path"]: p["id"] for p in pending}
        report(0, len(pending), f"Processing {len(pending)} queued files")

    total_files = len(remote_paths)
    report(0, total_files, f"Starting ingestion of {total_files} files")

    # Ensure Facility node exists so AT_FACILITY relationships don't fail
    with GraphClient() as gc:
        gc.ensure_facility(facility)

        # Ingestion gating: verify this graph allows the target facility
        try:
            from imas_codex.graph.meta import gate_ingestion

            gate_ingestion(gc, facility)
        except ValueError as e:
            logger.error("Ingestion gated: %s", e)
            report(0, 0, f"Ingestion blocked: {e}")
            return stats

    # Deduplication check
    paths_to_ingest = remote_paths
    if not force:
        with GraphClient() as check_client:
            paths_to_ingest, already_ingested = _check_already_ingested(
                check_client, facility, remote_paths
            )
            stats["skipped"] = len(already_ingested)
            if already_ingested:
                report(
                    0,
                    total_files,
                    f"Skipping {len(already_ingested)} already-ingested files",
                )

    if not paths_to_ingest:
        report(total_files, total_files, "All files already ingested")
        return stats

    # Collect file content grouped by language
    files_by_language: dict[str, list[dict[str, Any]]] = {}
    file_metadata: dict[str, dict[str, Any]] = {}

    # Fetch files off the event loop thread (SSH blocks synchronously)
    fetched_files = await asyncio.to_thread(
        lambda: list(fetch_remote_files(facility, paths_to_ingest))
    )

    for idx, (remote_path, content, language) in enumerate(fetched_files):
        filename = Path(remote_path).name
        report(idx, len(paths_to_ingest), f"Fetched {filename} ({language})")

        example_id = _generate_example_id(facility, remote_path)
        author = _extract_author(remote_path)

        file_metadata[example_id] = {
            "facility_id": facility,
            "source_file": remote_path,
            "language": language,
            "title": filename,
            "description": description or f"Code example from {remote_path}",
            "author": author,
            "ingested_at": datetime.now(UTC).isoformat(),
            "_source_file_id": source_file_ids.get(remote_path),
        }

        file_info = {
            "content": content,
            "remote_path": remote_path,
            "example_id": example_id,
        }

        if language not in files_by_language:
            files_by_language[language] = []
        files_by_language[language].append(file_info)

    if not files_by_language:
        report(total_files, total_files, "No files to process")
        return stats

    # Flatten all files — process together regardless of language.
    # Previous code grouped by language and ran separate chunk→embed→write
    # cycles per group. With 10 files across 5 languages, that meant 5
    # separate embedding calls and ~100 individual graph queries.
    all_files: list[dict[str, Any]] = []
    for language, file_list in files_by_language.items():
        for file_info in file_list:
            file_info["language"] = language
            all_files.append(file_info)

    total_to_process = len(all_files)
    stats["files"] = 0

    BATCH_SIZE = 20

    processed_files = 0
    for batch_start in range(0, total_to_process, BATCH_SIZE):
        batch_files = all_files[batch_start : batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, total_to_process)

        report(
            processed_files,
            total_to_process,
            f"Processing files {batch_start + 1}-{batch_end}/{total_to_process}",
        )

        # Split and extract for each file (language-aware per file)
        all_chunks: list[dict[str, Any]] = []
        chunk_example_ids: list[str] = []
        t_chunk_start = _time.monotonic()

        for file_info in batch_files:
            example_id = file_info["example_id"]
            language = file_info["language"]
            chunk_metadata = {
                "source_file": file_info["remote_path"],
                "facility_id": facility,
                "language": language,
                "code_example_id": example_id,
            }

            try:
                chunks = await asyncio.to_thread(
                    _split_and_extract,
                    file_info["content"],
                    language,
                    chunk_metadata,
                )
            except Exception:
                logger.warning(
                    "Failed to parse %s with tree-sitter, trying text splitter",
                    file_info["remote_path"],
                )
                try:
                    chunks = await asyncio.to_thread(
                        _split_and_extract,
                        file_info["content"],
                        language,
                        chunk_metadata,
                        use_text_splitter=True,
                    )
                except Exception as e2:
                    logger.error(
                        "Failed to process %s: %s", file_info["remote_path"], e2
                    )
                    meta = file_metadata.get(example_id, {})
                    sf_id = meta.get("_source_file_id")
                    if sf_id:
                        update_source_file_status(sf_id, "failed", error=str(e2))
                    continue

            # Generate chunk IDs
            for i, chunk in enumerate(chunks):
                content_hash = hashlib.md5(chunk["text"].encode()).hexdigest()[:8]
                chunk["id"] = f"{example_id}:chunk_{i}:{content_hash}"

            all_chunks.extend(chunks)
            chunk_example_ids.append(example_id)

        if not all_chunks:
            for file_info in batch_files:
                meta = file_metadata.get(file_info["example_id"], {})
                sf_id = meta.get("_source_file_id")
                if sf_id:
                    update_source_file_status(sf_id, "failed", error="No chunks")
            continue

        t_chunk_elapsed = _time.monotonic() - t_chunk_start

        # Deferred embedding: write chunks to graph WITHOUT embeddings.
        # The embed_text_worker picks up CodeChunk nodes where
        # embedding IS NULL and embeds them asynchronously on the GPU.
        # This decouples ingestion throughput from embedding latency.

        # Count stats
        batch_ids_found = 0
        batch_mdsplus_paths = 0
        for chunk in all_chunks:
            batch_ids_found += len(chunk.get("related_ids", []))
            batch_mdsplus_paths += len(chunk.get("mdsplus_paths", []))

        stats["chunks"] += len(all_chunks)
        stats["ids_found"] += batch_ids_found
        stats["mdsplus_paths"] += batch_mdsplus_paths

        # Batched graph writes — one session for all files in the batch.
        # Previously did N × create_nodes("CodeExample") + N × individual queries;
        # now batches into single UNWIND calls (~6 queries vs ~5N).
        t_graph_start = _time.monotonic()
        step_times: dict[str, float] = {}
        with GraphClient() as graph_client:
            # Step 1: Batch create CodeExample nodes (auto-creates AT_FACILITY)
            all_example_props: list[dict[str, Any]] = []
            source_file_map: dict[str, str] = {}
            for file_info in batch_files:
                example_id = file_info["example_id"]
                if example_id not in chunk_example_ids:
                    continue
                meta = file_metadata.get(example_id)
                if not meta:
                    continue

                source_file_id = meta.pop("_source_file_id", None)
                if source_file_id:
                    source_file_map[example_id] = source_file_id

                from_file_id = f"{facility}:{meta['source_file']}"
                all_example_props.append(
                    {
                        "id": example_id,
                        **meta,
                        "from_file": from_file_id,
                    }
                )

            t_s = _time.monotonic()
            if all_example_props:
                graph_client.create_nodes("CodeExample", all_example_props)
            step_times["create_examples"] = _time.monotonic() - t_s

            # Step 2: Batch FacilityPath status update + HAS_EXAMPLE
            t_s = _time.monotonic()
            if all_example_props:
                graph_client.query(
                    """
                    UNWIND $items AS item
                    MATCH (p:FacilityPath {facility_id: $facility})
                    WHERE item.source_file STARTS WITH p.path
                    MATCH (e:CodeExample {id: item.example_id})
                    SET p.status = 'explored',
                        p.last_ingested_at = datetime(),
                        p.files_ingested = coalesce(p.files_ingested, 0) + 1
                    MERGE (p)-[:HAS_EXAMPLE]->(e)
                    """,
                    facility=facility,
                    items=[
                        {"source_file": ep["source_file"], "example_id": ep["id"]}
                        for ep in all_example_props
                    ],
                )
            step_times["facility_path"] = _time.monotonic() - t_s

            # Step 3: Batch CodeFile → HAS_EXAMPLE
            t_s = _time.monotonic()
            if all_example_props:
                graph_client.query(
                    """
                    UNWIND $pairs AS pair
                    MATCH (cf:CodeFile {id: pair.cf_id})
                    MATCH (ce:CodeExample {id: pair.ce_id})
                    MERGE (cf)-[:HAS_EXAMPLE]->(ce)
                    """,
                    pairs=[
                        {
                            "cf_id": f"{facility}:{ep['source_file']}",
                            "ce_id": ep["id"],
                        }
                        for ep in all_example_props
                    ],
                )
            step_times["codefile_has_example"] = _time.monotonic() - t_s

            # Step 4: Batch CodeFile status update
            # (replaces per-file update_source_file_status which opened N connections)
            t_s = _time.monotonic()
            now = datetime.now(UTC).isoformat()
            if source_file_map:
                graph_client.query(
                    """
                    UNWIND $items AS item
                    MATCH (sf:CodeFile {id: item.sf_id})
                    SET sf.status = 'ingested',
                        sf.completed_at = $now,
                        sf.code_example_id = item.ce_id,
                        sf.error = null
                    """,
                    items=[
                        {"sf_id": sf_id, "ce_id": ex_id}
                        for ex_id, sf_id in source_file_map.items()
                    ],
                    now=now,
                )
            step_times["codefile_status"] = _time.monotonic() - t_s

            # Step 5: Create CodeChunk nodes (relationships handled below)
            # Disable auto-relationship creation — the pipeline already
            # creates HAS_CHUNK (step 6), CONTAINS_REF (step 7), and
            # AT_FACILITY in the final linking step.  Skipping the 3-5
            # separate relationship queries per batch of 50 saves 30-70s.
            t_s = _time.monotonic()
            graph_client.create_nodes(
                "CodeChunk", all_chunks, create_relationships=False
            )
            step_times["create_chunks"] = _time.monotonic() - t_s

            # Step 6: Create HAS_CHUNK + AT_FACILITY for CodeChunks
            # Combined into one query to avoid extra round-trip (previously
            # AT_FACILITY was auto-created by create_nodes, now handled here).
            t_s = _time.monotonic()
            graph_client.query(
                """
                MATCH (c:CodeChunk)
                WHERE c.code_example_id IN $example_ids
                MATCH (e:CodeExample {id: c.code_example_id})
                MERGE (e)-[:HAS_CHUNK]->(c)
                WITH c
                WHERE c.facility_id IS NOT NULL
                MATCH (f:Facility {id: c.facility_id})
                MERGE (c)-[:AT_FACILITY]->(f)
                """,
                example_ids=chunk_example_ids,
            )
            step_times["has_chunk"] = _time.monotonic() - t_s

            # Step 7: Link MDSplus refs to data nodes (batched, not per-example)
            t_s = _time.monotonic()
            if batch_mdsplus_paths > 0:
                linked = link_chunks_to_data_nodes(
                    graph_client, example_ids=chunk_example_ids
                )
                stats["data_nodes_linked"] += linked
            step_times["link_data_nodes"] = _time.monotonic() - t_s

        t_graph_elapsed = _time.monotonic() - t_graph_start

        step_detail = " ".join(
            f"{k}={v:.1f}s" for k, v in step_times.items() if v >= 0.1
        )
        logger.info(
            "Batch %d-%d timing: chunk=%.1fs graph=%.1fs (%d chunks) [%s]",
            batch_start + 1,
            batch_end,
            t_chunk_elapsed,
            t_graph_elapsed,
            len(all_chunks),
            step_detail,
        )

        processed_files += len(batch_files)
        stats["files"] = processed_files

    # Final relationship linking — safety net for any relationships not
    # created in the per-batch step above (e.g. cross-batch references).
    all_example_ids = list(file_metadata.keys())
    report(processed_files, total_to_process, "Creating final graph relationships...")

    t_link_start = _time.monotonic()
    with GraphClient() as graph_client:
        link_chunks_to_imas_paths(graph_client, example_ids=all_example_ids)
        if stats["mdsplus_paths"] > 0:
            link_chunks_to_data_nodes(graph_client, example_ids=all_example_ids)
        link_examples_to_facility(graph_client, example_ids=all_example_ids)
    t_link_elapsed = _time.monotonic() - t_link_start

    report(
        total_to_process,
        total_to_process,
        f"Completed: {stats['files']} files, {stats['chunks']} chunks, "
        f"{stats['skipped']} skipped, {stats['data_nodes_linked']} data nodes linked",
    )
    logger.info("Final linking: %.1fs", t_link_elapsed)
    return stats


__all__ = [
    "ProgressCallback",
    "ingest_files",
]
