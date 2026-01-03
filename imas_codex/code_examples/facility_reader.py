"""Remote facility file fetching via SSH/SCP.

Provides functions for transferring code files from remote facilities
to local staging for ingestion into the knowledge graph.
"""

import io
import logging
import shlex
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path

from fabric import Connection

logger = logging.getLogger(__name__)

# Threshold for switching to batch tar transfer
TAR_BATCH_THRESHOLD = 10

# Extension to language mapping
# Languages with tree-sitter support: python, matlab, fortran, julia, cpp, c
# Languages needing text splitter: tdi, idl
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
    ".fun": "tdi",  # TDI (MDSplus Tree Data Interface)
    ".FUN": "tdi",  # TDI (case sensitive filesystems)
}

# Languages that require text-based splitting (no tree-sitter grammar)
TEXT_SPLITTER_LANGUAGES = {"tdi", "idl"}


def detect_language(path: str) -> str:
    """Detect programming language from file extension.

    Args:
        path: File path

    Returns:
        Language identifier for tree-sitter
    """
    ext = Path(path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, "python")


def _fetch_sequential(
    facility: str,
    remote_paths: list[str],
) -> Iterator[tuple[str, str, str]]:
    """Fetch files one at a time via SCP.

    Uses Fabric for SSH connection with connection reuse.

    Args:
        facility: SSH host alias
        remote_paths: List of remote file paths

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


def _fetch_batch_tar(
    facility: str,
    remote_paths: list[str],
) -> Iterator[tuple[str, str, str]]:
    """Fetch multiple files as a single tar stream.

    Significantly faster for 10+ files by eliminating per-file SCP overhead.
    Falls back to sequential fetch if tar fails.

    Args:
        facility: SSH host alias
        remote_paths: List of remote file paths

    Yields:
        Tuples of (remote_path, content, language)
    """
    # Build tar command with quoted paths
    paths_arg = " ".join(shlex.quote(p) for p in remote_paths)
    cmd = f"tar -cf - {paths_arg} 2>/dev/null"

    logger.info("Batch fetching %d files via tar", len(remote_paths))

    try:
        result = subprocess.run(
            ["ssh", facility, cmd],
            capture_output=True,
            check=True,
            timeout=300,  # 5 minute timeout
        )

        # Extract files from tar stream
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue

                try:
                    # Get original path from tar member name
                    # tar stores paths without leading /
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
    - Sequential SCP for small batches (≤10 files)
    - Batch tar for larger batches (>10 files)

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
    "detect_language",
    "fetch_remote_files",
    "EXTENSION_TO_LANGUAGE",
    "TAR_BATCH_THRESHOLD",
]
