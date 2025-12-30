"""LlamaIndex IngestionPipeline for code examples.

Configures the pipeline with CodeSplitter, IDSExtractor, and
Neo4jVectorStore for semantic code search.
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
from .graph_linker import link_chunks_to_imas_paths, link_examples_to_facility
from .ids_extractor import IDSExtractor

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
    chunk_lines: int = 40,
    chunk_lines_overlap: int = 10,
    max_chars: int = 3000,
) -> IngestionPipeline:
    """Create LlamaIndex ingestion pipeline for code examples.

    Args:
        vector_store: Optional pre-configured Neo4jVectorStore
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
                language="python",
                chunk_lines=chunk_lines,
                chunk_lines_overlap=chunk_lines_overlap,
                max_chars=max_chars,
            ),
            IDSExtractor(),
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


async def ingest_code_files(
    facility: str,
    remote_paths: list[str],
    description: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Ingest code files from a remote facility using LlamaIndex pipeline.

    Fetches files via SSH, processes with IngestionPipeline, and creates
    graph relationships for IMAS path linking.

    Args:
        facility: Facility SSH host alias (e.g., "epfl")
        remote_paths: List of remote file paths to ingest
        description: Optional description for all files
        progress_callback: Optional callback for progress reporting

    Returns:
        Dict with counts: {"files": N, "chunks": M, "ids_found": K}
    """
    stats = {"files": 0, "chunks": 0, "ids_found": 0}
    total_files = len(remote_paths)

    def report(current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(current, total, message)
        logger.info("[%d/%d] %s", current, total, message)

    report(0, total_files, f"Starting ingestion of {total_files} files")

    # Create pipeline and graph client
    pipeline = create_pipeline()
    graph_client = GraphClient()

    # Fetch and process files
    documents: list[Document] = []
    file_metadata: dict[str, dict[str, Any]] = {}

    for idx, (remote_path, content, language) in enumerate(
        fetch_remote_files(facility, remote_paths)
    ):
        filename = Path(remote_path).name
        report(idx, total_files, f"Fetched {filename}")

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
        }

        # Create Document for pipeline
        doc = Document(
            text=content,
            metadata={
                "source_file": remote_path,
                "facility_id": facility,
                "language": language,
                "code_example_id": example_id,
            },
        )
        documents.append(doc)
        stats["files"] += 1

    if not documents:
        report(total_files, total_files, "No files to process")
        return stats

    # Run pipeline
    report(stats["files"], total_files, "Running ingestion pipeline")
    nodes = await pipeline.arun(documents=documents)

    stats["chunks"] = len(nodes)

    # Count IDS references
    for node in nodes:
        related_ids = node.metadata.get("related_ids", [])
        stats["ids_found"] += len(related_ids)

    # Create CodeExample nodes and relationships
    report(stats["files"], total_files, "Creating graph relationships")

    with graph_client:
        # Create CodeExample nodes
        for example_id, meta in file_metadata.items():
            graph_client.query(
                """
                MERGE (e:CodeExample {id: $id})
                SET e += $props
                """,
                id=example_id,
                props=meta,
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

        # Create RELATED_PATHS and FACILITY_ID relationships
        link_chunks_to_imas_paths(graph_client)
        link_examples_to_facility(graph_client)

    report(
        total_files,
        total_files,
        f"Completed: {stats['files']} files, {stats['chunks']} chunks",
    )
    return stats


__all__ = [
    "ProgressCallback",
    "create_pipeline",
    "create_vector_store",
    "get_embed_model",
    "ingest_code_files",
]
