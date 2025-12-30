"""Code example ingestion pipeline using LlamaIndex.

Fetches code from remote facilities, chunks it with tree-sitter,
generates embeddings using sentence-transformers, extracts IMAS
references, and stores in Neo4j.
"""

import hashlib
import logging
import re
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fabric import Connection
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from imas_codex.graph import GraphClient
from imas_codex.settings import get_imas_embedding_model

from .queue import EmbeddingQueue, QueuedFile

logger = logging.getLogger(__name__)

# Progress callback type: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]

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
}

# Regex patterns for IMAS IDS detection
IDS_PATTERNS = [
    # Python: ids_factory.new("equilibrium")
    r'\.new\(["\'](\w+)["\']\)',
    # Python: factory.equilibrium()
    r"factory\.(\w+)\(\)",
    # String literals that are IDS names
    r'["\'](\w+)["\']',
]

# Known IMAS IDS names for validation
KNOWN_IDS = {
    "equilibrium",
    "core_profiles",
    "core_sources",
    "core_transport",
    "summary",
    "pf_active",
    "magnetics",
    "thomson_scattering",
    "interferometer",
    "ece",
    "mse",
    "bolometer",
    "soft_x_rays",
    "charge_exchange",
    "nbi",
    "ec_launchers",
    "ic_antennas",
    "wall",
    "controllers",
    "pulse_schedule",
}


@dataclass
class CodeChunkResult:
    """Result of chunking a code file."""

    content: str
    function_name: str | None
    start_line: int
    end_line: int
    metadata: dict[str, Any] = field(default_factory=dict)


def get_embed_model() -> HuggingFaceEmbedding:
    """Get the project's standard embedding model."""
    model_name = get_imas_embedding_model()
    return HuggingFaceEmbedding(
        model_name=model_name,
        trust_remote_code=False,
    )


def get_code_splitter(
    language: str,
    chunk_lines: int = 40,
    chunk_lines_overlap: int = 10,
    max_chars: int = 3000,
) -> CodeSplitter:
    """Get a LlamaIndex CodeSplitter for the given language.

    Args:
        language: Programming language (python, matlab, fortran, etc.)
        chunk_lines: Target number of lines per chunk
        chunk_lines_overlap: Number of overlapping lines between chunks
        max_chars: Maximum characters per chunk

    Returns:
        Configured CodeSplitter instance
    """
    return CodeSplitter(
        language=language,
        chunk_lines=chunk_lines,
        chunk_lines_overlap=chunk_lines_overlap,
        max_chars=max_chars,
    )


@dataclass
class CodeExampleIngester:
    """Ingests code examples from remote facilities into the knowledge graph.

    Uses Fabric for SSH file transfer, LlamaIndex CodeSplitter for
    language-aware chunking, and HuggingFace embeddings for semantic search.
    """

    embed_model: HuggingFaceEmbedding = field(default_factory=get_embed_model)
    graph_client: GraphClient = field(default_factory=GraphClient)
    chunk_lines: int = 40
    chunk_lines_overlap: int = 10
    max_chars: int = 3000
    progress_callback: ProgressCallback | None = None
    queue: EmbeddingQueue | None = None

    def __post_init__(self) -> None:
        """Cache CodeSplitters by language for efficiency."""
        self._splitters: dict[str, CodeSplitter] = {}

    def _get_splitter(self, language: str) -> CodeSplitter:
        """Get or create a CodeSplitter for the given language."""
        if language not in self._splitters:
            self._splitters[language] = get_code_splitter(
                language=language,
                chunk_lines=self.chunk_lines,
                chunk_lines_overlap=self.chunk_lines_overlap,
                max_chars=self.max_chars,
            )
        return self._splitters[language]

    def ingest_files(
        self,
        facility: str,
        remote_paths: list[str],
        description: str | None = None,
    ) -> dict[str, int]:
        """Ingest multiple code files from a remote facility.

        Args:
            facility: Facility SSH host alias (e.g., "epfl")
            remote_paths: List of remote file paths to ingest
            description: Optional description for all files

        Returns:
            Dict with counts: {"files": N, "chunks": M, "ids_found": K}
        """
        stats = {"files": 0, "chunks": 0, "ids_found": 0}
        total_files = len(remote_paths)

        self._report_progress(
            0, total_files, f"Starting ingestion of {total_files} files"
        )

        with self.graph_client:
            for idx, (remote_path, local_path) in enumerate(
                self._fetch_files(facility, remote_paths)
            ):
                filename = Path(remote_path).name
                self._report_progress(idx, total_files, f"Processing {filename}")
                try:
                    result = self._ingest_single_file(
                        facility=facility,
                        remote_path=remote_path,
                        local_path=local_path,
                        description=description,
                    )
                    stats["files"] += 1
                    stats["chunks"] += result["chunks"]
                    stats["ids_found"] += result["ids_found"]
                    self._report_progress(
                        idx + 1,
                        total_files,
                        f"Ingested {filename}: {result['chunks']} chunks",
                    )
                except Exception as e:
                    logger.exception("Failed to ingest %s: %s", remote_path, e)
                    self._report_progress(
                        idx + 1, total_files, f"Failed to ingest {filename}: {e}"
                    )

        self._report_progress(
            total_files,
            total_files,
            f"Completed: {stats['files']} files, {stats['chunks']} chunks",
        )
        return stats

    def _report_progress(self, current: int, total: int, message: str) -> None:
        """Report progress via callback if set."""
        if self.progress_callback:
            self.progress_callback(current, total, message)
        logger.info(f"[{current}/{total}] {message}")

    def queue_files(
        self,
        facility: str,
        remote_paths: list[str],
        description: str | None = None,
    ) -> list[QueuedFile]:
        """Queue files for offline embedding processing.

        Downloads files from remote facility and stages them locally
        for async processing. Does not block on embedding generation.

        Args:
            facility: Facility SSH host alias (e.g., "epfl")
            remote_paths: List of remote file paths to queue
            description: Optional description for all files

        Returns:
            List of QueuedFile objects representing queued items
        """
        if self.queue is None:
            self.queue = EmbeddingQueue()

        queued = []
        total = len(remote_paths)

        self._report_progress(0, total, f"Downloading {total} files for queuing")

        for idx, (remote_path, local_path) in enumerate(
            self._fetch_files(facility, remote_paths)
        ):
            try:
                content = local_path.read_text(encoding="utf-8", errors="replace")
                extension = Path(remote_path).suffix.lower()
                language = EXTENSION_TO_LANGUAGE.get(extension, "python")

                qf = self.queue.add_file(
                    facility_id=facility,
                    remote_path=remote_path,
                    content=content,
                    language=language,
                    description=description,
                )
                queued.append(qf)
                self._report_progress(
                    idx + 1, total, f"Queued: {Path(remote_path).name}"
                )
            except Exception as e:
                logger.warning(f"Failed to queue {remote_path}: {e}")

        self._report_progress(
            total, total, f"Queued {len(queued)} files for processing"
        )
        return queued

    def process_queue(self, max_files: int | None = None) -> dict[str, int]:
        """Process pending files from the embedding queue.

        Args:
            max_files: Maximum number of files to process (None = all)

        Returns:
            Dict with counts: {"processed": N, "failed": M, "chunks": K}
        """
        if self.queue is None:
            self.queue = EmbeddingQueue()

        pending = self.queue.get_pending()
        if max_files:
            pending = pending[:max_files]

        stats = {"processed": 0, "failed": 0, "chunks": 0}
        total = len(pending)

        self._report_progress(0, total, f"Processing {total} queued files")

        with self.graph_client:
            for idx, qf in enumerate(pending):
                self.queue.mark_processing(qf.id)
                filename = Path(qf.remote_path).name

                try:
                    local_path = Path(qf.local_path)
                    if not local_path.exists():
                        raise FileNotFoundError(f"Staged file missing: {qf.local_path}")

                    result = self._ingest_single_file(
                        facility=qf.facility_id,
                        remote_path=qf.remote_path,
                        local_path=local_path,
                        description=qf.description,
                    )

                    self.queue.mark_completed(qf.id)
                    stats["processed"] += 1
                    stats["chunks"] += result["chunks"]
                    self._report_progress(
                        idx + 1,
                        total,
                        f"Processed: {filename} ({result['chunks']} chunks)",
                    )
                except Exception as e:
                    self.queue.mark_failed(qf.id, str(e))
                    stats["failed"] += 1
                    logger.exception(f"Failed to process {filename}: {e}")
                    self._report_progress(idx + 1, total, f"Failed: {filename}")

        self._report_progress(
            total,
            total,
            f"Completed: {stats['processed']} processed, {stats['failed']} failed",
        )
        return stats

    def _ingest_single_file(
        self,
        facility: str,
        remote_path: str,
        local_path: Path,
        description: str | None = None,
    ) -> dict[str, int]:
        """Ingest a single code file using LlamaIndex."""
        content = local_path.read_text(encoding="utf-8", errors="replace")
        filename = Path(remote_path).name
        extension = Path(remote_path).suffix.lower()
        language = EXTENSION_TO_LANGUAGE.get(extension, "python")

        # Generate example ID
        example_id = self._generate_id(facility, remote_path)

        # Extract author from path if possible
        author = self._extract_author(remote_path)

        # Extract IDS references from the full file
        related_ids = self._extract_ids_references(content)

        # Create CodeExample node
        example_props = {
            "id": example_id,
            "facility_id": facility,
            "source_file": remote_path,
            "language": language,
            "title": filename,
            "description": description or f"Code example from {remote_path}",
            "author": author,
            "related_ids": list(related_ids),
            "ingested_at": datetime.now(UTC).isoformat(),
        }

        self.graph_client.create_node(
            "CodeExample", example_id, example_props, id_field="id"
        )

        # Link to facility
        self.graph_client.create_relationship(
            "CodeExample",
            example_id,
            "Facility",
            facility,
            "FACILITY_ID",
            from_id_field="id",
        )

        # Chunk the code using LlamaIndex
        chunks = list(self._chunk_code(content, language))

        # Generate embeddings for all chunks in batch
        chunk_texts = [c.content for c in chunks]
        embeddings = self._batch_embed(chunk_texts) if chunk_texts else []

        # Collect all chunk data for batch insertion
        chunk_props_list: list[dict[str, Any]] = []
        chunk_ids_map: dict[str, set[str]] = {}

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            chunk_id = f"{example_id}:chunk_{i}"

            # Extract IDS references from chunk
            chunk_ids = self._extract_ids_references(chunk.content)
            if chunk_ids:
                chunk_ids_map[chunk_id] = chunk_ids

            chunk_props_list.append(
                {
                    "id": chunk_id,
                    "code_example_id": example_id,
                    "content": chunk.content,
                    "function_name": chunk.function_name,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "embedding": embedding,
                }
            )

        # Batch insert all chunks
        if chunk_props_list:
            self.graph_client.create_nodes(
                "CodeChunk",
                chunk_props_list,
                id_field="id",
                facility_id_field=None,
            )

            # Batch create HAS_CHUNK relationships
            self.graph_client.query(
                """
                UNWIND $chunks AS chunk
                MATCH (c:CodeChunk {id: chunk.id})
                MATCH (e:CodeExample {id: $example_id})
                MERGE (c)-[:HAS_CHUNK]->(e)
                """,
                chunks=[{"id": c["id"]} for c in chunk_props_list],
                example_id=example_id,
            )

            # Batch create IMAS path relationships
            imas_links = [
                {"chunk_id": cid, "ids": ids_name}
                for cid, ids_set in chunk_ids_map.items()
                for ids_name in ids_set
            ]
            if imas_links:
                self.graph_client.query(
                    """
                    UNWIND $links AS link
                    MERGE (i:IMASPath {path: link.ids})
                    ON CREATE SET i.ids = link.ids
                    WITH i, link
                    MATCH (c:CodeChunk {id: link.chunk_id})
                    MERGE (c)-[:RELATED_PATHS]->(i)
                    """,
                    links=imas_links,
                )

        chunk_count = len(chunk_props_list)
        logger.info(
            f"Ingested {filename}: {chunk_count} chunks, {len(related_ids)} IDS refs"
        )

        return {"chunks": chunk_count, "ids_found": len(related_ids)}

    def _fetch_files(
        self, facility: str, remote_paths: list[str]
    ) -> Iterator[tuple[str, Path]]:
        """Fetch files from remote facility via SCP with connection reuse.

        Yields (remote_path, local_path) tuples.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with Connection(facility) as conn:
                for remote_path in remote_paths:
                    local_path = Path(tmpdir) / Path(remote_path).name
                    try:
                        conn.get(remote_path, str(local_path))
                        yield remote_path, local_path
                    except Exception as e:
                        logger.warning("Failed to fetch %s: %s", remote_path, e)

    def _chunk_code(self, content: str, language: str) -> Iterator[CodeChunkResult]:
        """Chunk code using LlamaIndex CodeSplitter.

        Uses tree-sitter for language-aware parsing that respects
        function and class boundaries.
        """
        try:
            splitter = self._get_splitter(language)
            doc = Document(text=content)
            nodes = splitter.get_nodes_from_documents([doc])

            for node in nodes:
                # Extract line information from node metadata
                start_line = node.metadata.get("start_line", 1)
                end_line = node.metadata.get(
                    "end_line", start_line + node.text.count("\n")
                )

                # Try to extract function name from first line
                function_name = self._extract_function_name(node.text, language)

                yield CodeChunkResult(
                    content=node.text,
                    function_name=function_name,
                    start_line=start_line,
                    end_line=end_line,
                    metadata=dict(node.metadata),
                )
        except Exception as e:
            logger.warning(f"CodeSplitter failed for {language}, falling back: {e}")
            yield from self._chunk_by_size(content)

    def _extract_function_name(self, text: str, language: str) -> str | None:
        """Extract function/class name from chunk text."""
        first_line = text.strip().split("\n")[0] if text else ""

        patterns = {
            "python": r"(?:async\s+)?(?:def|class)\s+(\w+)",
            "matlab": r"function\s+(?:\[?\w+\]?\s*=\s*)?(\w+)",
            "fortran": r"(?:subroutine|function)\s+(\w+)",
            "julia": r"function\s+(\w+)",
            "cpp": r"(?:\w+\s+)+(\w+)\s*\(",
            "c": r"(?:\w+\s+)+(\w+)\s*\(",
        }

        pattern = patterns.get(language)
        if pattern:
            match = re.match(pattern, first_line, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _chunk_by_size(self, content: str) -> Iterator[CodeChunkResult]:
        """Fallback size-based chunking."""
        lines = content.split("\n")

        for i in range(0, len(lines), self.chunk_lines - self.chunk_lines_overlap):
            chunk = lines[i : i + self.chunk_lines]
            yield CodeChunkResult(
                content="\n".join(chunk),
                function_name=None,
                start_line=i + 1,
                end_line=min(i + self.chunk_lines, len(lines)),
            )

    def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch."""
        if not texts:
            return []

        embeddings = self.embed_model.get_text_embedding_batch(
            texts, show_progress=False
        )
        return [list(e) for e in embeddings]

    def _extract_ids_references(self, content: str) -> set[str]:
        """Extract IMAS IDS references from code content."""
        found: set[str] = set()

        for pattern in IDS_PATTERNS:
            for match in re.finditer(pattern, content):
                candidate = match.group(1).lower()
                if candidate in KNOWN_IDS:
                    found.add(candidate)

        return found

    def _generate_id(self, facility: str, remote_path: str) -> str:
        """Generate unique ID for a code example."""
        content = f"{facility}:{remote_path}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{facility}:{Path(remote_path).stem}:{hash_suffix}"

    def _extract_author(self, path: str) -> str | None:
        """Extract username from path like /home/username/..."""
        match = re.match(r"/home/(\w+)/", path)
        return match.group(1) if match else None


__all__ = [
    "CodeExampleIngester",
    "ProgressCallback",
    "get_embed_model",
    "get_code_splitter",
]
