"""Code Examples module for ingesting and searching facility code snippets.

This module uses LlamaIndex's IngestionPipeline and Neo4jVectorStore for:
- Fetching code files from remote facilities via SSH/SCP
- Chunking code into searchable segments using tree-sitter CodeSplitter
- Generating embeddings using HuggingFace sentence-transformers
- Extracting IMAS IDS references from code via custom IDSExtractor
- Extracting MDSplus paths from code via custom MDSplusExtractor
- Storing code examples and chunks in Neo4j with vector embeddings
- Semantic search with optional IDS and facility filtering

Graph-driven ingestion workflow:
1. Scouts use queue_source_files() to mark files for ingestion
2. CLI command 'imas-codex ingest <facility>' processes the queue
3. SourceFile nodes track lifecycle: queued -> fetching -> embedding -> ready
"""

from .facility_reader import (
    EXTENSION_TO_LANGUAGE,
    TEXT_SPLITTER_LANGUAGES,
    detect_language,
    fetch_remote_files,
)
from .graph_linker import (
    link_chunks_to_imas_paths,
    link_chunks_to_tree_nodes,
    link_examples_to_facility,
)
from .ids_extractor import IDSExtractor, get_known_ids
from .mdsplus_extractor import (
    MDSplusExtractor,
    MDSplusReference,
    extract_mdsplus_paths,
    normalize_mdsplus_path,
)
from .pipeline import (
    ProgressCallback,
    create_pipeline,
    create_vector_store,
    get_embed_model,
    ingest_code_files,
)
from .queue import (
    QueuedFile,
    get_pending_files,
    get_queue_stats,
    queue_source_files,
    update_source_file_status,
)
from .search import CodeExampleSearch, CodeSearchResult

__all__ = [
    "CodeExampleSearch",
    "CodeSearchResult",
    "EXTENSION_TO_LANGUAGE",
    "IDSExtractor",
    "MDSplusExtractor",
    "MDSplusReference",
    "ProgressCallback",
    "QueuedFile",
    "TEXT_SPLITTER_LANGUAGES",
    "create_pipeline",
    "create_vector_store",
    "detect_language",
    "extract_mdsplus_paths",
    "fetch_remote_files",
    "get_embed_model",
    "get_known_ids",
    "get_pending_files",
    "get_queue_stats",
    "ingest_code_files",
    "link_chunks_to_imas_paths",
    "link_chunks_to_tree_nodes",
    "link_examples_to_facility",
    "normalize_mdsplus_path",
    "queue_source_files",
    "update_source_file_status",
]
