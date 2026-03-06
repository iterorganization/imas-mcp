"""Remote facility file fetching via SSH.

Provides functions for transferring files from remote facilities
to local staging for ingestion into the knowledge graph.

Two transfer strategies, both via subprocess SSH:
- Sequential SSH cat for single files (~0.6s fixed SSH setup)
- Batch gzip tar for 2+ files (amortises SSH setup, 16x compression)

Benchmarked on JET (100 Python files, 4 MB total):
  SCP sequential:   ~1 file/s   (1s SSH setup per file)
  SSH cat:           1.6 file/s  (single file, no temp dir)
  tar gzip 5 files:  7.8 files/s
  tar gzip 50 files: 55 files/s
  tar gzip 100 files: 98 files/s
"""

import io
import logging
import shlex
import subprocess
import tarfile
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

# Use gzip tar for 2+ files.  Even at 5 files gzip tar achieves
# 7.8 files/s vs 1.1 files/s for sequential SCP.
TAR_BATCH_THRESHOLD = 2

# Extension to language mapping for code files
# Languages with direct tree-sitter integration: python, matlab, fortran, julia, cpp, c, idl
# Languages needing text splitter: tdi
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".m": "matlab",
    ".f90": "fortran",
    ".f": "fortran",
    ".for": "fortran",
    ".F90": "fortran",
    ".F": "fortran",
    ".pro": "idl",
    ".jl": "julia",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".fun": "tdi",
    ".FUN": "tdi",
}

# Document extensions (not code — use text splitter or dedicated reader)
DOCUMENT_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".ipynb": "notebook",
}

# Image extensions (not text — processed via VLM pipeline)
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}

# Languages that require text-based splitting (no tree-sitter grammar).
# Includes document types that have no tree-sitter parser.
# IDL is parsed by tree-sitter-gdl and NOT in this set.
TEXT_SPLITTER_LANGUAGES = {
    "tdi",
    "text",
    "markdown",
    "rst",
    "html",
    "pdf",
    "docx",
    "pptx",
    "xlsx",
    "notebook",
}

# All supported extensions (code + documents + images)
ALL_SUPPORTED_EXTENSIONS = (
    set(EXTENSION_TO_LANGUAGE) | set(DOCUMENT_EXTENSIONS) | IMAGE_EXTENSIONS
)


def detect_language(path: str) -> str:
    """Detect programming language from file extension.

    Args:
        path: File path

    Returns:
        Language identifier for tree-sitter, or document type
    """
    ext = Path(path).suffix
    # Try case-sensitive first (for .F90 vs .f90)
    if ext in EXTENSION_TO_LANGUAGE:
        return EXTENSION_TO_LANGUAGE[ext]
    # Then case-insensitive for code
    ext_lower = ext.lower()
    if ext_lower in EXTENSION_TO_LANGUAGE:
        return EXTENSION_TO_LANGUAGE[ext_lower]
    # Check document types
    if ext_lower in DOCUMENT_EXTENSIONS:
        return DOCUMENT_EXTENSIONS[ext_lower]
    return "python"  # Default fallback


def detect_file_category(path: str) -> str:
    """Detect file category from extension.

    Maps to FileCategory enum values: code, document, notebook, config, data, other.

    Args:
        path: File path

    Returns:
        Category string matching FileCategory enum
    """
    ext = Path(path).suffix.lower()
    if ext in {k.lower() for k in EXTENSION_TO_LANGUAGE}:
        return "code"
    if ext == ".ipynb":
        return "notebook"
    if ext in {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".html",
        ".htm",
        ".md",
        ".rst",
        ".txt",
    }:
        return "document"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in {".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".json"}:
        return "config"
    if ext in {".csv", ".h5", ".hdf5", ".nc", ".mat"}:
        return "data"
    return "other"


def _fetch_sequential(
    facility: str,
    remote_paths: list[str],
) -> Iterator[tuple[str, str, str]]:
    """Fetch files one at a time via SSH cat.

    Uses a single SSH subprocess per file — faster than SCP because
    it avoids temp file creation and Fabric overhead (~0.6s vs ~1.0s).

    Args:
        facility: SSH host alias
        remote_paths: List of remote file paths

    Yields:
        Tuples of (remote_path, content, language)
    """
    for remote_path in remote_paths:
        try:
            result = subprocess.run(
                ["ssh", facility, "cat", shlex.quote(remote_path)],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(
                    "Failed to fetch %s: exit %d", remote_path, result.returncode
                )
                continue

            content = result.stdout.decode("utf-8", errors="replace")
            language = detect_language(remote_path)

            yield remote_path, content, language
        except subprocess.TimeoutExpired:
            logger.warning("Timed out fetching %s", remote_path)
            continue
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", remote_path, e)
            continue


def _fetch_batch_tar(
    facility: str,
    remote_paths: list[str],
) -> Iterator[tuple[str, str, str]]:
    """Fetch multiple files as a gzip-compressed tar stream.

    Gzip compression reduces payload ~16x for source code, making this
    faster than both plain tar and sequential SCP at all batch sizes ≥2.
    Falls back to sequential if tar fails.

    Args:
        facility: SSH host alias
        remote_paths: List of remote file paths

    Yields:
        Tuples of (remote_path, content, language)
    """
    paths_arg = " ".join(shlex.quote(p) for p in remote_paths)
    cmd = f"tar -czf - {paths_arg} 2>/dev/null"

    logger.info("Batch fetching %d files via tar+gzip", len(remote_paths))

    try:
        result = subprocess.run(
            ["ssh", facility, cmd],
            capture_output=True,
            check=True,
            timeout=300,
        )

        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue

                try:
                    original_path = "/" + member.name.lstrip("/")

                    f = tf.extractfile(member)
                    if f is None:
                        continue

                    content = f.read().decode("utf-8", errors="replace")
                    language = detect_language(original_path)

                    yield original_path, content, language
                except Exception as e:
                    logger.warning("Failed to extract %s: %s", member.name, e)
                    continue

    except subprocess.CalledProcessError as e:
        logger.warning(
            "Tar batch failed (exit %d), falling back to sequential", e.returncode
        )
        yield from _fetch_sequential(facility, remote_paths)
    except subprocess.TimeoutExpired:
        logger.warning("Tar batch timed out, falling back to sequential")
        yield from _fetch_sequential(facility, remote_paths)


def fetch_remote_files(
    facility: str,
    remote_paths: list[str],
    strategy: str = "auto",
) -> Iterator[tuple[str, str, str]]:
    """Fetch files from remote facility via SSH.

    Automatically selects optimal transfer strategy:
    - SSH cat for single files (~0.6s SSH overhead)
    - Batch gzip tar for 2+ files (amortises SSH setup, 16x compression)

    Args:
        facility: SSH host alias from ~/.ssh/config
        remote_paths: List of remote file paths to fetch
        strategy: "auto" (default), "sequential", or "tar"

    Yields:
        Tuples of (remote_path, content, language)
    """
    if not remote_paths:
        return

    n = len(remote_paths)

    if strategy == "auto":
        strategy = "sequential" if n <= TAR_BATCH_THRESHOLD else "tar"

    logger.info("Fetching %d files using %s strategy", n, strategy)

    if strategy == "tar":
        yield from _fetch_batch_tar(facility, remote_paths)
    else:
        yield from _fetch_sequential(facility, remote_paths)


__all__ = [
    "ALL_SUPPORTED_EXTENSIONS",
    "DOCUMENT_EXTENSIONS",
    "EXTENSION_TO_LANGUAGE",
    "TAR_BATCH_THRESHOLD",
    "TEXT_SPLITTER_LANGUAGES",
    "detect_file_category",
    "detect_language",
    "fetch_remote_files",
]
