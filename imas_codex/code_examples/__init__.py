"""Code Examples module for ingesting and searching facility code snippets.

This module uses LlamaIndex's IngestionPipeline and Neo4jVectorStore for:
- Fetching code files from remote facilities via SSH/SCP
- Chunking code into searchable segments using tree-sitter CodeSplitter
- Generating embeddings using HuggingFace sentence-transformers
- Extracting IMAS IDS references from code via custom IDSExtractor
- Storing code examples and chunks in Neo4j with vector embeddings
- Semantic search with optional IDS and facility filtering
"""

from .facility_reader import EXTENSION_TO_LANGUAGE, detect_language, fetch_remote_files
from .graph_linker import link_chunks_to_imas_paths, link_examples_to_facility
from .ids_extractor import IDSExtractor, get_known_ids
from .pipeline import (
    ProgressCallback,
    create_pipeline,
    create_vector_store,
    get_embed_model,
    ingest_code_files,
)
from .search import CodeExampleSearch, CodeSearchResult

__all__ = [
    "CodeExampleSearch",
    "CodeSearchResult",
    "EXTENSION_TO_LANGUAGE",
    "IDSExtractor",
    "ProgressCallback",
    "create_pipeline",
    "create_vector_store",
    "detect_language",
    "fetch_remote_files",
    "get_embed_model",
    "get_known_ids",
    "ingest_code_files",
    "link_chunks_to_imas_paths",
    "link_examples_to_facility",
]
