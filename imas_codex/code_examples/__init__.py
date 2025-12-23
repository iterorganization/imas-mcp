"""Code Examples module for ingesting and searching facility code snippets.

This module provides tools for:
- Fetching code files from remote facilities via SSH/SCP
- Chunking code into searchable segments using LlamaIndex
- Generating embeddings for semantic search
- Extracting IMAS path references from code
- Storing code examples and chunks in the Neo4j graph
"""

from .ingester import CodeExampleIngester, ProgressCallback
from .search import CodeExampleSearch

__all__ = ["CodeExampleIngester", "CodeExampleSearch", "ProgressCallback"]
