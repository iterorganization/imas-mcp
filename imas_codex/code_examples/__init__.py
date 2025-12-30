"""Code Examples module for ingesting and searching facility code snippets.

This module provides tools for:
- Fetching code files from remote facilities via SSH/SCP
- Chunking code into searchable segments using LlamaIndex CodeSplitter
- Generating embeddings using HuggingFace sentence-transformers
- Extracting IMAS path references from code
- Storing code examples and chunks in the Neo4j graph
- Queue-based offline processing for async embedding generation
"""

from .ingester import (
    CodeExampleIngester,
    ProgressCallback,
    get_code_splitter,
    get_embed_model,
)
from .processor import AsyncEmbeddingProcessor, run_processor
from .queue import EmbeddingQueue, FileStatus, QueuedFile
from .search import CodeExampleSearch

__all__ = [
    "AsyncEmbeddingProcessor",
    "CodeExampleIngester",
    "CodeExampleSearch",
    "EmbeddingQueue",
    "FileStatus",
    "ProgressCallback",
    "QueuedFile",
    "get_code_splitter",
    "get_embed_model",
    "run_processor",
]
