"""Code example ingestion pipeline.

Fetches code from remote facilities, chunks it, generates embeddings,
extracts IMAS references, and stores in Neo4j.
"""

import hashlib
import logging
import re
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fabric import Connection

from imas_codex.embeddings.encoder import Encoder
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# Extension to language mapping
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".m": "matlab",
    ".f90": "fortran",
    ".f": "fortran",
    ".for": "fortran",
    ".pro": "idl",
    ".jl": "julia",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
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


@dataclass
class CodeExampleIngester:
    """Ingests code examples from remote facilities into the knowledge graph.

    Uses Fabric for SSH file transfer, LlamaIndex for code chunking,
    and the existing Encoder for embeddings.
    """

    encoder: Encoder = field(default_factory=Encoder)
    graph_client: GraphClient = field(default_factory=GraphClient)
    chunk_size: int = 1000
    chunk_overlap: int = 200

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

        with self.graph_client:
            for remote_path, local_path in self._fetch_files(facility, remote_paths):
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
                except Exception as e:
                    logger.exception("Failed to ingest %s: %s", remote_path, e)

        return stats

    def _ingest_single_file(
        self,
        facility: str,
        remote_path: str,
        local_path: Path,
        description: str | None = None,
    ) -> dict[str, int]:
        """Ingest a single code file."""
        content = local_path.read_text(encoding="utf-8", errors="replace")
        filename = Path(remote_path).name
        extension = Path(remote_path).suffix.lower()
        language = EXTENSION_TO_LANGUAGE.get(extension, "python")

        # Generate example ID
        example_id = self._generate_id(facility, remote_path)

        # Extract author from path if possible (e.g., /home/username/...)
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

        # Chunk the code
        chunks = list(self._chunk_code(content, language))
        chunk_count = 0

        for i, chunk in enumerate(chunks):
            chunk_id = f"{example_id}:chunk_{i}"

            # Generate embedding for chunk
            embedding = self.encoder.embed_texts([chunk.content])[0]

            # Extract IDS references from chunk
            chunk_ids = self._extract_ids_references(chunk.content)

            chunk_props = {
                "id": chunk_id,
                "code_example_id": example_id,
                "content": chunk.content,
                "function_name": chunk.function_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "embedding": embedding.tolist(),
            }

            self.graph_client.create_node(
                "CodeChunk", chunk_id, chunk_props, id_field="id"
            )

            # Create relationship to parent
            self.graph_client.create_relationship(
                "CodeChunk",
                chunk_id,
                "CodeExample",
                example_id,
                "HAS_CHUNK",
                from_id_field="id",
            )

            # Create relationships to IMAS paths for fine-grained linking
            for ids_name in chunk_ids:
                # Create IMASPath node if needed and link
                imas_path = f"{ids_name}"
                self.graph_client.query(
                    """
                    MERGE (i:IMASPath {path: $path})
                    ON CREATE SET i.ids = $ids
                    WITH i
                    MATCH (c:CodeChunk {id: $chunk_id})
                    MERGE (c)-[:RELATED_PATHS]->(i)
                    """,
                    path=imas_path,
                    ids=ids_name,
                    chunk_id=chunk_id,
                )

            chunk_count += 1

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
        """Chunk code into searchable segments.

        Uses function-level chunking for Python, falls back to
        size-based chunking for other languages.
        """
        if language == "python":
            yield from self._chunk_python_functions(content)
        else:
            yield from self._chunk_by_size(content)

    def _chunk_python_functions(self, content: str) -> Iterator[CodeChunkResult]:
        """Extract Python functions/classes as chunks."""
        lines = content.split("\n")
        current_chunk: list[str] = []
        current_name: str | None = None
        current_start: int = 0
        in_definition = False
        base_indent: int = 0

        for i, line in enumerate(lines, 1):
            # Detect function/class definition
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if stripped.startswith(("def ", "class ", "async def ")):
                # Yield previous chunk if any
                if current_chunk and current_name:
                    yield CodeChunkResult(
                        content="\n".join(current_chunk),
                        function_name=current_name,
                        start_line=current_start,
                        end_line=i - 1,
                    )

                # Start new chunk
                match = re.match(r"(?:async\s+)?(?:def|class)\s+(\w+)", stripped)
                current_name = match.group(1) if match else None
                current_chunk = [line]
                current_start = i
                in_definition = True
                base_indent = indent
            elif in_definition:
                # Continue current definition
                if (
                    stripped
                    and indent <= base_indent
                    and not stripped.startswith(("@", "#"))
                ):
                    # End of definition
                    yield CodeChunkResult(
                        content="\n".join(current_chunk),
                        function_name=current_name,
                        start_line=current_start,
                        end_line=i - 1,
                    )
                    current_chunk = [line]
                    current_name = None
                    current_start = i
                    in_definition = False
                else:
                    current_chunk.append(line)
            else:
                # Module-level code
                if not current_chunk:
                    current_start = i
                current_chunk.append(line)

        # Yield final chunk
        if current_chunk:
            yield CodeChunkResult(
                content="\n".join(current_chunk),
                function_name=current_name,
                start_line=current_start,
                end_line=len(lines),
            )

    def _chunk_by_size(self, content: str) -> Iterator[CodeChunkResult]:
        """Fall back to size-based chunking."""
        lines = content.split("\n")
        chunk_lines = self.chunk_size // 50  # Approximate lines per chunk

        for i in range(0, len(lines), chunk_lines - self.chunk_overlap // 50):
            chunk = lines[i : i + chunk_lines]
            yield CodeChunkResult(
                content="\n".join(chunk),
                function_name=None,
                start_line=i + 1,
                end_line=min(i + chunk_lines, len(lines)),
            )

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
