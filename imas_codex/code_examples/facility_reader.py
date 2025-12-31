"""Remote facility file fetching via SSH/SCP.

Provides functions for transferring code files from remote facilities
to local staging for ingestion into the knowledge graph.
"""

import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path

from fabric import Connection

logger = logging.getLogger(__name__)

# Extension to language mapping (tree-sitter language names)
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".m": "matlab",
    ".f90": "fortran",
    ".f": "fortran",
    ".for": "fortran",
    ".pro": "python",  # IDL -> fallback to Python-like parsing
    ".jl": "julia",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".fun": "c",  # TDI -> fallback to C-like parsing
    ".FUN": "c",  # TDI (case sensitive filesystems)
}


def detect_language(path: str) -> str:
    """Detect programming language from file extension.

    Args:
        path: File path

    Returns:
        Language identifier for tree-sitter
    """
    ext = Path(path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, "python")


def fetch_remote_files(
    facility: str,
    remote_paths: list[str],
) -> Iterator[tuple[str, str, str]]:
    """Fetch files from remote facility via SCP.

    Uses Fabric for SSH connection with connection reuse.
    Files are downloaded to a temporary directory.

    Args:
        facility: SSH host alias from ~/.ssh/config
        remote_paths: List of remote file paths to fetch

    Yields:
        Tuples of (remote_path, content, language)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with Connection(facility) as conn:
            for remote_path in remote_paths:
                try:
                    local_path = Path(tmpdir) / Path(remote_path).name
                    conn.get(remote_path, str(local_path))

                    content = local_path.read_text(encoding="utf-8", errors="replace")
                    language = detect_language(remote_path)

                    yield remote_path, content, language
                except Exception as e:
                    logger.warning("Failed to fetch %s: %s", remote_path, e)
                    continue


__all__ = ["detect_language", "fetch_remote_files", "EXTENSION_TO_LANGUAGE"]
