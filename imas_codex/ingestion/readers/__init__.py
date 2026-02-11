"""Content readers for ingesting files into the knowledge graph.

Readers handle file I/O for different content types:
- remote: SSH/SCP file fetching from remote facilities
- office: PDF, DOCX, PPTX, XLSX extraction
- html: HTML/TWiki text extraction
- notebook: Jupyter notebook cell extraction
"""

from .remote import (
    EXTENSION_TO_LANGUAGE,
    TEXT_SPLITTER_LANGUAGES,
    detect_language,
    fetch_remote_files,
)

__all__ = [
    "EXTENSION_TO_LANGUAGE",
    "TEXT_SPLITTER_LANGUAGES",
    "detect_language",
    "fetch_remote_files",
]
