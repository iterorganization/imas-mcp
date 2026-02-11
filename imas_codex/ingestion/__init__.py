"""Unified ingestion module for all content types.

Routes files to appropriate processors based on FileCategory:
- code: tree-sitter chunking (Python, Fortran, MATLAB, etc.)
- document: text extraction + sentence chunking (PDF, DOCX, PPTX, HTML)
- notebook: cell-aware chunking (Jupyter)
- data/config/other: metadata-only ingestion

All content types share:
- Embedding via EncoderEmbedding (local/remote/openrouter)
- Entity extraction (IDS references, MDSplus paths, units, conventions)
- Unified Chunk graph persistence with NEXT_CHUNK linked lists
- VLM captioning for embedded images

Graph-driven workflow:
1. discover files <facility> → scan + score → SourceFile nodes
2. ingest run <facility> → fetch + chunk + embed + link
"""

from .extractors.ids import IDSExtractor, get_known_ids
from .extractors.mdsplus import (
    MDSplusExtractor,
    MDSplusReference,
    extract_mdsplus_paths,
    normalize_mdsplus_path,
)
from .graph import link_chunks_to_imas_paths, link_chunks_to_tree_nodes
from .pipeline import ingest_files
from .queue import (
    get_pending_files,
    get_queue_stats,
    queue_source_files,
    update_source_file_status,
)
from .readers.remote import (
    EXTENSION_TO_LANGUAGE,
    TEXT_SPLITTER_LANGUAGES,
    detect_language,
    fetch_remote_files,
)
from .search import ChunkSearch, ChunkSearchResult

__all__ = [
    "ChunkSearch",
    "ChunkSearchResult",
    "EXTENSION_TO_LANGUAGE",
    "IDSExtractor",
    "MDSplusExtractor",
    "MDSplusReference",
    "TEXT_SPLITTER_LANGUAGES",
    "detect_language",
    "extract_mdsplus_paths",
    "fetch_remote_files",
    "get_known_ids",
    "get_pending_files",
    "get_queue_stats",
    "ingest_files",
    "link_chunks_to_imas_paths",
    "link_chunks_to_tree_nodes",
    "normalize_mdsplus_path",
    "queue_source_files",
    "update_source_file_status",
]
