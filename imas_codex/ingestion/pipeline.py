"""Unified ingestion pipeline for all content types.

Graph-driven ingestion via SourceFile queue:
- Automatic deduplication (skips already-ingested files)
- Per-file atomic commits (interrupt-safe)
- Auto-updates FacilityPath status to 'explored'
- Links extracted MDSplus paths to TreeNode entities
- Routes files to appropriate splitters based on language/type

Replaces code_examples.pipeline.
"""

import hashlib
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

from imas_codex.embeddings import get_embed_model
from imas_codex.graph import GraphClient
from imas_codex.settings import (
    get_embedding_dimension,
    get_graph_password,
    get_graph_uri,
    get_graph_username,
)

from .extractors.ids import IDSExtractor
from .extractors.mdsplus import MDSplusExtractor
from .graph import (
    link_chunks_to_imas_paths,
    link_chunks_to_tree_nodes,
    link_example_mdsplus_paths,
    link_examples_to_facility,
)
from .queue import get_pending_files, update_source_file_status
from .readers.remote import TEXT_SPLITTER_LANGUAGES, fetch_remote_files

logger = logging.getLogger(__name__)

# Progress callback type: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]


def create_vector_store(
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
) -> Neo4jVectorStore:
    """Create Neo4jVectorStore for code chunks.

    Connection settings are resolved from centralized settings when not
    explicitly provided (env var → pyproject.toml → default).

    Args:
        neo4j_uri: Neo4j connection URI (default: from settings)
        neo4j_user: Neo4j username (default: from settings)
        neo4j_password: Neo4j password (default: from settings)

    Returns:
        Configured Neo4jVectorStore
    """
    return Neo4jVectorStore(
        username=neo4j_user or get_graph_username(),
        password=neo4j_password or get_graph_password(),
        url=neo4j_uri or get_graph_uri(),
        embedding_dimension=get_embedding_dimension(),
        index_name="code_chunk_embedding",
        node_label="CodeChunk",
        text_node_property="content",
        embedding_node_property="embedding",
    )


def create_pipeline(
    vector_store: Neo4jVectorStore | None = None,
    language: str = "python",
    chunk_lines: int = 40,
    chunk_lines_overlap: int = 10,
    max_chars: int = 10000,
    use_text_splitter: bool = False,
) -> IngestionPipeline:
    """Create LlamaIndex ingestion pipeline.

    Args:
        vector_store: Optional pre-configured Neo4jVectorStore
        language: Programming language for CodeSplitter
        chunk_lines: Target lines per chunk
        chunk_lines_overlap: Overlap between chunks
        max_chars: Maximum characters per chunk
        use_text_splitter: Use text-based splitting instead of tree-sitter

    Returns:
        Configured IngestionPipeline
    """
    vs = vector_store or create_vector_store()

    if use_text_splitter or language in TEXT_SPLITTER_LANGUAGES:
        splitter = SentenceSplitter(
            chunk_size=max_chars,
            chunk_overlap=chunk_lines_overlap * 60,
            separator="\n",
        )
    else:
        splitter = CodeSplitter(
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
        )

    return IngestionPipeline(
        transformations=[
            splitter,
            IDSExtractor(),
            MDSplusExtractor(),
            get_embed_model(),
        ],
        vector_store=vs,
    )


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
        MERGE (p)-[:PRODUCED]->(e)
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
) -> dict[str, int]:
    """Ingest files from a remote facility.

    Can be called in two modes:
    1. **Path list mode**: Provide remote_paths explicitly
    2. **Graph-driven mode**: Omit remote_paths to process queued SourceFile nodes

    Features:
    - Deduplication: Skips files that are already ingested (unless force=True)
    - Interrupt-safe: Each file is committed atomically
    - Auto status update: SourceFile nodes are marked 'ingested'
    - MDSplus linking: Extracted paths are linked to TreeNode entities

    Args:
        facility: Facility SSH host alias (e.g., "tcv")
        remote_paths: List of remote file paths to ingest (if None, uses graph queue)
        description: Optional description for all files
        progress_callback: Optional callback for progress reporting
        force: If True, re-ingest files even if already present
        limit: Maximum files to process from graph queue

    Returns:
        Dict with counts: files, chunks, ids_found, mdsplus_paths, skipped, tree_nodes_linked
    """
    stats = {
        "files": 0,
        "chunks": 0,
        "ids_found": 0,
        "mdsplus_paths": 0,
        "skipped": 0,
        "tree_nodes_linked": 0,
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

    vector_store = create_vector_store()

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

    # Group documents by language
    docs_by_language: dict[str, list[Document]] = {}
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

        doc = Document(
            text=content,
            metadata={
                "source_file": remote_path,
                "facility_id": facility,
                "language": language,
                "code_example_id": example_id,
                "_full_doc_text": content,
            },
        )

        if language not in docs_by_language:
            docs_by_language[language] = []
        docs_by_language[language].append(doc)

    if not docs_by_language:
        report(total_files, total_files, "No files to process")
        return stats

    total_to_process = sum(len(docs) for docs in docs_by_language.values())
    stats["files"] = 0

    BATCH_SIZE = 25

    processed_files = 0
    for language, documents in docs_by_language.items():
        try:
            pipeline = create_pipeline(vector_store=vector_store, language=language)
        except Exception as e:
            logger.error("Failed to create pipeline for %s: %s", language, e)
            continue

        for batch_start in range(0, len(documents), BATCH_SIZE):
            batch_docs = documents[batch_start : batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, len(documents))

            report(
                processed_files,
                total_to_process,
                f"Embedding {language} files {batch_start + 1}-{batch_end}/{len(documents)}",
            )

            try:
                nodes = await pipeline.arun(documents=batch_docs)
            except Exception as e:
                logger.warning(
                    "Failed to parse %s batch with tree-sitter, trying text splitter: %s",
                    language,
                    e,
                )
                try:
                    fallback_pipeline = create_pipeline(
                        vector_store=vector_store,
                        language=language,
                        use_text_splitter=True,
                    )
                    nodes = await fallback_pipeline.arun(documents=batch_docs)
                except Exception as e2:
                    logger.error("Failed to process %s batch: %s", language, e2)
                    for doc in batch_docs:
                        example_id = doc.metadata.get("code_example_id")
                        meta = file_metadata.get(example_id, {})
                        sf_id = meta.get("_source_file_id")
                        if sf_id:
                            update_source_file_status(sf_id, "failed", error=str(e2))
                    continue

            # Count stats
            batch_ids_found = 0
            batch_mdsplus_paths = 0
            for node in nodes:
                related_ids = node.metadata.get("related_ids", [])
                batch_ids_found += len(related_ids)
                mdsplus_paths = node.metadata.get("mdsplus_paths", [])
                batch_mdsplus_paths += len(mdsplus_paths)

            stats["chunks"] += len(nodes)
            stats["ids_found"] += batch_ids_found
            stats["mdsplus_paths"] += batch_mdsplus_paths

            # Commit batch to graph (interrupt-safe)
            with GraphClient() as graph_client:
                for doc in batch_docs:
                    example_id = doc.metadata.get("code_example_id")
                    meta = file_metadata.get(example_id)
                    if not meta:
                        continue

                    source_file_id = meta.pop("_source_file_id", None)

                    graph_client.query(
                        """
                        MERGE (e:CodeExample {id: $id})
                        SET e += $props
                        """,
                        id=example_id,
                        props=meta,
                    )

                    _update_facility_path_status(
                        graph_client, facility, meta["source_file"], example_id
                    )

                    if source_file_id:
                        update_source_file_status(
                            source_file_id, "ingested", code_example_id=example_id
                        )

                batch_example_ids = [
                    doc.metadata.get("code_example_id") for doc in batch_docs
                ]
                graph_client.query(
                    """
                    MATCH (c:CodeChunk)
                    WHERE c.code_example_id IN $example_ids
                    MATCH (e:CodeExample {id: c.code_example_id})
                    MERGE (e)-[:HAS_CHUNK]->(c)
                    """,
                    example_ids=batch_example_ids,
                )

                for example_id in batch_example_ids:
                    if example_id:
                        linked = link_example_mdsplus_paths(graph_client, example_id)
                        stats["tree_nodes_linked"] += linked

            processed_files += len(batch_docs)
            stats["files"] = processed_files

    # Final relationship linking
    report(processed_files, total_to_process, "Creating final graph relationships...")

    with GraphClient() as graph_client:
        link_chunks_to_imas_paths(graph_client)
        link_chunks_to_tree_nodes(graph_client)
        link_examples_to_facility(graph_client)

    report(
        total_to_process,
        total_to_process,
        f"Completed: {stats['files']} files, {stats['chunks']} chunks, "
        f"{stats['skipped']} skipped, {stats['tree_nodes_linked']} tree nodes linked",
    )
    return stats


__all__ = [
    "ProgressCallback",
    "create_pipeline",
    "create_vector_store",
    "ingest_files",
]
