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
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_codex.graph import GraphClient

if TYPE_CHECKING:
    from imas_codex.embeddings.encoder import Encoder

from .chunkers import chunk_code, chunk_text
from .extractors.ids import extract_ids_references
from .extractors.mdsplus import extract_mdsplus_paths
from .graph import (
    link_chunks_to_data_nodes,
    link_chunks_to_imas_paths,
    link_example_mdsplus_paths,
    link_examples_to_facility,
)
from .queue import get_pending_files, update_source_file_status
from .readers.remote import TEXT_SPLITTER_LANGUAGES, fetch_remote_files

logger = logging.getLogger(__name__)

# Progress callback type: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]

# Maximum number of chunk texts to embed in a single call.
# Prevents CUDA OOM on large batches — each chunk can be up to 10KB.
# With 1024-dim embeddings, 32 chunks ≈ 320KB text → ~1 GiB GPU memory.
MAX_CHUNKS_PER_EMBED = 16


def _embed_chunks_safe(
    encoder: "Encoder",
    chunk_texts: list[str],
    max_per_call: int = MAX_CHUNKS_PER_EMBED,
) -> np.ndarray:
    """Embed chunks in sub-batches with OOM-aware retry.

    Splits large embedding requests into sub-batches to fit in GPU memory.
    On CUDA OOM, halves the sub-batch size and retries. Single-chunk OOMs
    skip only that chunk (zero vector) instead of failing the entire batch.

    Args:
        encoder: Encoder instance (local or remote)
        chunk_texts: List of text strings to embed
        max_per_call: Maximum chunks per embedding call

    Returns:
        Numpy array of embeddings (zero vectors for skipped chunks)
    """
    if not chunk_texts:
        return np.array([])

    all_embeddings: list[np.ndarray] = []
    i = 0
    current_batch = max_per_call
    dim: int | None = None

    while i < len(chunk_texts):
        batch = chunk_texts[i : i + current_batch]
        try:
            emb = encoder.embed_texts(batch)
            all_embeddings.append(emb)
            if dim is None:
                dim = emb.shape[1]
            i += len(batch)
            # Gradually restore batch size after success
            if current_batch < max_per_call:
                current_batch = min(current_batch * 2, max_per_call)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                # Free cached GPU memory before retrying
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                if current_batch > 1:
                    current_batch = max(1, current_batch // 2)
                    logger.warning(
                        "OOM with %d chunks, retrying with %d",
                        len(batch),
                        current_batch,
                    )
                    continue
                else:
                    # Single chunk OOM — skip it with a zero vector and continue
                    logger.warning(
                        "OOM embedding single chunk (%d chars), inserting zero vector",
                        len(batch[0]),
                    )
                    if dim is None:
                        from imas_codex.settings import get_embedding_dimension

                        dim = get_embedding_dimension()
                    all_embeddings.append(np.zeros((1, dim), dtype=np.float32))
                    i += 1
                    current_batch = max_per_call
                    continue
            else:
                raise

    return np.vstack(all_embeddings)


def _split_and_extract(
    content: str,
    language: str,
    metadata: dict[str, Any],
    max_chars: int = 10000,
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


def _update_facility_path_status(
    graph_client: GraphClient,
    facility: str,
    source_file: str,
    example_id: str,
) -> None:
    """Update FacilityPath status to 'explored' and link to CodeExample."""
    graph_client.query(
        """
        MATCH (p:FacilityPath {facility_id: $facility})
        WHERE $source_file STARTS WITH p.path
        MATCH (e:CodeExample {id: $example_id})
        SET p.status = 'explored',
            p.last_ingested_at = datetime(),
            p.files_ingested = coalesce(p.files_ingested, 0) + 1
        MERGE (p)-[:HAS_EXAMPLE]->(e)
        """,
        facility=facility,
        source_file=source_file,
        example_id=example_id,
    )


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
        encoder: Optional shared Encoder instance (avoids loading model per call)

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

    for idx, (remote_path, content, language) in enumerate(
        fetch_remote_files(facility, paths_to_ingest)
    ):
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

    total_to_process = sum(len(files) for files in files_by_language.values())
    stats["files"] = 0

    # Load encoder for embeddings (reuse shared instance if provided)
    if encoder is None:
        from imas_codex.embeddings.encoder import Encoder

        encoder = Encoder()

    BATCH_SIZE = 20

    processed_files = 0
    for language, file_list in files_by_language.items():
        for batch_start in range(0, len(file_list), BATCH_SIZE):
            batch_files = file_list[batch_start : batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, len(file_list))

            report(
                processed_files,
                total_to_process,
                f"Processing {language} files {batch_start + 1}-{batch_end}/{len(file_list)}",
            )

            # Split and extract for each file in batch
            all_chunks: list[dict[str, Any]] = []
            chunk_example_ids: list[str] = []

            for file_info in batch_files:
                example_id = file_info["example_id"]
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

            # Generate embeddings in safe sub-batches (OOM-aware)
            chunk_texts = [c["text"] for c in all_chunks]
            try:
                embeddings = await asyncio.to_thread(
                    _embed_chunks_safe, encoder, chunk_texts
                )
                for i, chunk in enumerate(all_chunks):
                    chunk["embedding"] = embeddings[i].tolist()
            except Exception as e:
                logger.error("Embedding failed for %d chunks: %s", len(chunk_texts), e)
                for file_info in batch_files:
                    meta = file_metadata.get(file_info["example_id"], {})
                    sf_id = meta.get("_source_file_id")
                    if sf_id:
                        update_source_file_status(sf_id, "failed", error=str(e)[:200])
                continue

            # Count stats
            batch_ids_found = 0
            batch_mdsplus_paths = 0
            for chunk in all_chunks:
                batch_ids_found += len(chunk.get("related_ids", []))
                batch_mdsplus_paths += len(chunk.get("mdsplus_paths", []))

            stats["chunks"] += len(all_chunks)
            stats["ids_found"] += batch_ids_found
            stats["mdsplus_paths"] += batch_mdsplus_paths

            # Write to graph using create_nodes() for schema-correct relationships.
            # Order matters: CodeExample must exist before CodeChunk (for CODE_EXAMPLE_ID).
            with GraphClient() as graph_client:
                # Step 1: Create CodeExample nodes first (auto-creates AT_FACILITY)
                for file_info in batch_files:
                    example_id = file_info["example_id"]
                    meta = file_metadata.get(example_id)
                    if not meta:
                        continue

                    source_file_id = meta.pop("_source_file_id", None)

                    # Compute from_file for the FROM_FILE relationship
                    from_file_id = f"{facility}:{meta['source_file']}"
                    example_props = {
                        "id": example_id,
                        **meta,
                        "from_file": from_file_id,
                    }

                    graph_client.create_nodes("CodeExample", [example_props])

                    _update_facility_path_status(
                        graph_client, facility, meta["source_file"], example_id
                    )

                    # Link CodeFile → HAS_EXAMPLE → CodeExample
                    graph_client.query(
                        """
                        MATCH (cf:CodeFile {id: $cf_id})
                        MATCH (ce:CodeExample {id: $ce_id})
                        MERGE (cf)-[:HAS_EXAMPLE]->(ce)
                        """,
                        cf_id=from_file_id,
                        ce_id=example_id,
                    )

                    if source_file_id:
                        update_source_file_status(
                            source_file_id, "ingested", code_example_id=example_id
                        )

                # Step 2: Create CodeChunk nodes (auto-creates CODE_EXAMPLE_ID, AT_FACILITY)
                graph_client.create_nodes("CodeChunk", all_chunks)

                # Step 3: Create HAS_CHUNK (inverse traversal convenience)
                graph_client.query(
                    """
                    MATCH (c:CodeChunk)
                    WHERE c.code_example_id IN $example_ids
                    MATCH (e:CodeExample {id: c.code_example_id})
                    MERGE (e)-[:HAS_CHUNK]->(c)
                    """,
                    example_ids=chunk_example_ids,
                )

                for example_id in chunk_example_ids:
                    linked = link_example_mdsplus_paths(graph_client, example_id)
                    stats["data_nodes_linked"] += linked

            processed_files += len(batch_files)
            stats["files"] = processed_files

    # Final relationship linking — scoped to only the examples we just created.
    # This avoids catastrophic O(all_refs × all_signals) global scans that
    # previously spent 300+ seconds per batch scanning the entire graph.
    all_example_ids = list(file_metadata.keys())
    report(processed_files, total_to_process, "Creating final graph relationships...")

    with GraphClient() as graph_client:
        link_chunks_to_imas_paths(graph_client, example_ids=all_example_ids)
        link_chunks_to_data_nodes(graph_client, example_ids=all_example_ids)
        link_examples_to_facility(graph_client, example_ids=all_example_ids)

    report(
        total_to_process,
        total_to_process,
        f"Completed: {stats['files']} files, {stats['chunks']} chunks, "
        f"{stats['skipped']} skipped, {stats['data_nodes_linked']} data nodes linked",
    )
    return stats


__all__ = [
    "ProgressCallback",
    "ingest_files",
]
