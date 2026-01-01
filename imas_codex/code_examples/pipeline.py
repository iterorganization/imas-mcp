"""LlamaIndex IngestionPipeline for code examples.

Configures the pipeline with CodeSplitter, IDSExtractor, and
Neo4jVectorStore for semantic code search.

Features:
- Graph-driven ingestion via SourceFile queue
- Automatic deduplication (skips already-ingested files)
- Per-file atomic commits (interrupt-safe)
- Auto-updates FacilityPath status to 'ingested'
- Links extracted MDSplus paths to TreeNode entities
"""

import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import CodeSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

from imas_codex.graph import GraphClient
from imas_codex.settings import get_imas_embedding_model

from .facility_reader import fetch_remote_files
from .graph_linker import (
    link_chunks_to_imas_paths,
    link_chunks_to_tree_nodes,
    link_examples_to_facility,
)
from .ids_extractor import IDSExtractor
from .mdsplus_extractor import MDSplusExtractor
from .queue import get_pending_files, update_source_file_status

logger = logging.getLogger(__name__)

# Progress callback type: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]


def get_embed_model() -> HuggingFaceEmbedding:
    """Get the project's standard embedding model."""
    model_name = get_imas_embedding_model()
    return HuggingFaceEmbedding(
        model_name=model_name,
        trust_remote_code=False,
    )


def create_vector_store(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "imas-codex",
) -> Neo4jVectorStore:
    """Create Neo4jVectorStore for code chunks.

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        Configured Neo4jVectorStore
    """
    return Neo4jVectorStore(
        username=neo4j_user,
        password=neo4j_password,
        url=neo4j_uri,
        embedding_dimension=384,  # all-MiniLM-L6-v2
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
    max_chars: int = 3000,
) -> IngestionPipeline:
    """Create LlamaIndex ingestion pipeline for code examples.

    Args:
        vector_store: Optional pre-configured Neo4jVectorStore
        language: Programming language for CodeSplitter (default: python)
        chunk_lines: Target lines per chunk
        chunk_lines_overlap: Overlap between chunks
        max_chars: Maximum characters per chunk

    Returns:
        Configured IngestionPipeline
    """
    vs = vector_store or create_vector_store()

    return IngestionPipeline(
        transformations=[
            CodeSplitter(
                language=language,
                chunk_lines=chunk_lines,
                chunk_lines_overlap=chunk_lines_overlap,
                max_chars=max_chars,
            ),
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
    import re

    match = re.match(r"/home/(\w+)/", path)
    return match.group(1) if match else None


def _check_already_ingested(
    graph_client: GraphClient,
    facility: str,
    remote_paths: list[str],
) -> tuple[list[str], list[str]]:
    """Check which files are already ingested.

    Args:
        graph_client: GraphClient instance
        facility: Facility ID
        remote_paths: List of remote file paths to check

    Returns:
        Tuple of (paths_to_ingest, already_ingested_paths)
    """
    # Query for existing CodeExample nodes with matching source_file
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
    """Update FacilityPath status to 'ingested' and link to CodeExample.

    Finds the FacilityPath that contains this source file and updates it.

    Args:
        graph_client: GraphClient instance
        facility: Facility ID
        source_file: Remote file path that was ingested
        example_id: ID of the created CodeExample
    """
    # Find containing FacilityPath and update status
    graph_client.query(
        """
        MATCH (p:FacilityPath {facility_id: $facility})
        WHERE $source_file STARTS WITH p.path
        MATCH (e:CodeExample {id: $example_id})
        SET p.status = 'ingested',
            p.last_ingested_at = datetime(),
            p.files_ingested = coalesce(p.files_ingested, 0) + 1
        MERGE (p)-[:PRODUCED]->(e)
        """,
        facility=facility,
        source_file=source_file,
        example_id=example_id,
    )


def _link_example_mdsplus_paths(
    graph_client: GraphClient,
    example_id: str,
) -> int:
    """Link a CodeExample to TreeNodes via its chunks' MDSplus paths.

    Creates REFERENCES_NODE relationships from the CodeExample directly
    to TreeNode entities for easier querying.

    Args:
        graph_client: GraphClient instance
        example_id: ID of the CodeExample

    Returns:
        Number of TreeNode links created
    """
    result = graph_client.query(
        """
        MATCH (e:CodeExample {id: $example_id})-[:HAS_CHUNK]->(c:CodeChunk)
        WHERE c.mdsplus_paths IS NOT NULL
        UNWIND c.mdsplus_paths AS mds_path
        MATCH (t:TreeNode)
        WHERE t.path = mds_path
           OR t.path ENDS WITH substring(mds_path, 1)
           OR t.name = split(mds_path, '::')[-1]
        MERGE (e)-[:REFERENCES_NODE]->(t)
        RETURN count(DISTINCT t) AS linked
        """,
        example_id=example_id,
    )
    return result[0]["linked"] if result else 0


async def ingest_code_files(
    facility: str,
    remote_paths: list[str] | None = None,
    description: str | None = None,
    progress_callback: ProgressCallback | None = None,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Ingest code files from a remote facility using LlamaIndex pipeline.

    Can be called in two modes:
    1. **Path list mode**: Provide remote_paths explicitly
    2. **Graph-driven mode**: Omit remote_paths to process queued SourceFile nodes

    Fetches files via SSH, processes with IngestionPipeline, and creates
    graph relationships for IMAS and MDSplus path linking. Files are grouped
    by language and processed with language-specific CodeSplitter.

    Features:
    - **Deduplication**: Skips files that are already ingested (unless force=True)
    - **Interrupt-safe**: Each file is committed atomically
    - **Auto status update**: SourceFile nodes are marked 'ready'
    - **MDSplus linking**: Extracted paths are linked to TreeNode entities

    Args:
        facility: Facility SSH host alias (e.g., "epfl")
        remote_paths: List of remote file paths to ingest (if None, uses graph queue)
        description: Optional description for all files
        progress_callback: Optional callback for progress reporting
        force: If True, re-ingest files even if already present
        limit: Maximum files to process from graph queue (None = all queued files)

    Returns:
        Dict with counts: {
            "files": N,
            "chunks": M,
            "ids_found": K,
            "mdsplus_paths": L,
            "skipped": S,
            "tree_nodes_linked": T
        }
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

    # Determine source of files: explicit paths or graph queue
    source_file_ids: dict[str, str] = {}  # path -> source_file_id for status updates

    if remote_paths is None:
        # Graph-driven mode: get pending files from queue
        query_limit = limit if limit is not None else 10000  # Large number for "all"
        pending = get_pending_files(facility, limit=query_limit)
        if not pending:
            report(0, 0, "No pending files in queue")
            return stats

        remote_paths = [p["path"] for p in pending]
        source_file_ids = {p["path"]: p["id"] for p in pending}
        report(0, len(pending), f"Processing {len(pending)} queued files")

    total_files = len(remote_paths)
    report(0, total_files, f"Starting ingestion of {total_files} files")

    # Create vector store (shared across all languages)
    vector_store = create_vector_store()

    # Check for already-ingested files (deduplication)
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

    # Update SourceFile status to 'fetching'
    for path in paths_to_ingest:
        if path in source_file_ids:
            update_source_file_status(source_file_ids[path], "fetching")

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

        # Store metadata for CodeExample creation
        file_metadata[example_id] = {
            "facility_id": facility,
            "source_file": remote_path,
            "language": language,
            "title": filename,
            "description": description or f"Code example from {remote_path}",
            "author": author,
            "ingested_at": datetime.now(UTC).isoformat(),
            "_source_file_id": source_file_ids.get(remote_path),  # For status update
        }

        # Create Document for pipeline
        doc = Document(
            text=content,
            metadata={
                "source_file": remote_path,
                "facility_id": facility,
                "language": language,
                "code_example_id": example_id,
                "_full_doc_text": content,  # For line number calculation
            },
        )

        # Group by language
        if language not in docs_by_language:
            docs_by_language[language] = []
        docs_by_language[language].append(doc)
        stats["files"] += 1

    if not docs_by_language:
        report(total_files, total_files, "No files to process")
        return stats

    # Update status to 'embedding' for files being processed
    for meta in file_metadata.values():
        if meta.get("_source_file_id"):
            update_source_file_status(meta["_source_file_id"], "embedding")

    # Process each language group with appropriate pipeline
    all_nodes = []
    processed_files = 0
    for language, documents in docs_by_language.items():
        report(
            processed_files,
            stats["files"],
            f"Embedding {len(documents)} {language} files...",
        )
        try:
            pipeline = create_pipeline(vector_store=vector_store, language=language)
            nodes = await pipeline.arun(documents=documents)
            all_nodes.extend(nodes)
            processed_files += len(documents)
        except Exception as e:
            # Fallback: try with Python parser (works for most languages)
            logger.warning(
                "Failed to parse %s with %s parser, trying python: %s",
                language,
                language,
                e,
            )
            try:
                pipeline = create_pipeline(vector_store=vector_store, language="python")
                nodes = await pipeline.arun(documents=documents)
                all_nodes.extend(nodes)
                processed_files += len(documents)
            except Exception as e2:
                logger.error("Failed to process %s files: %s", language, e2)
                # Mark these files as failed
                for doc in documents:
                    sf_id = source_file_ids.get(doc.metadata.get("source_file"))
                    if sf_id:
                        update_source_file_status(sf_id, "failed", error=str(e2))
                continue

    stats["chunks"] = len(all_nodes)

    # Count IDS and MDSplus references
    for node in all_nodes:
        related_ids = node.metadata.get("related_ids", [])
        stats["ids_found"] += len(related_ids)
        mdsplus_paths = node.metadata.get("mdsplus_paths", [])
        stats["mdsplus_paths"] += len(mdsplus_paths)

    # Create CodeExample nodes and relationships
    report(processed_files, stats["files"], "Creating graph relationships...")

    with GraphClient() as graph_client:
        # Create CodeExample nodes and update FacilityPath status
        for example_id, meta in file_metadata.items():
            # Remove internal fields before storing
            source_file_id = meta.pop("_source_file_id", None)

            graph_client.query(
                """
                MERGE (e:CodeExample {id: $id})
                SET e += $props
                """,
                id=example_id,
                props=meta,
            )
            # Update FacilityPath status automatically
            _update_facility_path_status(
                graph_client, facility, meta["source_file"], example_id
            )

            # Update SourceFile status to 'ready'
            if source_file_id:
                update_source_file_status(
                    source_file_id, "ready", code_example_id=example_id
                )

        # Link chunks to examples (based on metadata)
        graph_client.query(
            """
            MATCH (c:CodeChunk)
            WHERE c.code_example_id IS NOT NULL
            MATCH (e:CodeExample {id: c.code_example_id})
            MERGE (e)-[:HAS_CHUNK]->(c)
            """
        )

        # Create RELATED_PATHS, REFERENCES_NODE, and FACILITY_ID relationships
        link_chunks_to_imas_paths(graph_client)
        link_chunks_to_tree_nodes(graph_client)
        link_examples_to_facility(graph_client)

        # Link CodeExamples directly to TreeNodes for easier querying
        for example_id in file_metadata:
            linked = _link_example_mdsplus_paths(graph_client, example_id)
            stats["tree_nodes_linked"] += linked

    report(
        total_files,
        total_files,
        f"Completed: {stats['files']} files, {stats['chunks']} chunks, "
        f"{stats['skipped']} skipped, {stats['tree_nodes_linked']} tree nodes linked",
    )
    return stats


__all__ = [
    "ProgressCallback",
    "create_pipeline",
    "create_vector_store",
    "get_embed_model",
    "ingest_code_files",
]
