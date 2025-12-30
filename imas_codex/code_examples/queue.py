"""Offline code embedding queue for async processing.

Provides a queue-based workflow where files are staged locally
for embedding, processed in the background, and ingested into
the knowledge graph without blocking agent operations.
"""

import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FileStatus(str, Enum):
    """Status of a file in the embedding queue."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueuedFile(BaseModel):
    """A file queued for embedding processing."""

    id: str
    facility_id: str
    remote_path: str
    local_path: str
    language: str
    status: FileStatus = FileStatus.PENDING
    description: str | None = None
    error: str | None = None
    queued_at: str = ""
    processed_at: str | None = None

    def model_post_init(self, _context: Any) -> None:
        """Set default timestamp if not provided."""
        if not self.queued_at:
            self.queued_at = datetime.now(UTC).isoformat()


@dataclass
class EmbeddingQueue:
    """Manages a staging directory for offline code embedding.

    Files are downloaded to a staging directory and processed
    asynchronously. This allows embedding generation to run
    without blocking LLM agent operations.
    """

    staging_dir: Path = field(
        default_factory=lambda: Path.home()
        / ".cache"
        / "imas-codex"
        / "embedding-queue"
    )
    manifest_file: str = "queue_manifest.json"

    def __post_init__(self) -> None:
        """Ensure staging directory exists."""
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.staging_dir / self.manifest_file

    def _generate_file_id(self, facility_id: str, remote_path: str) -> str:
        """Generate unique ID for a queued file."""
        content = f"{facility_id}:{remote_path}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _load_manifest(self) -> dict[str, QueuedFile]:
        """Load queue manifest from disk."""
        if not self._manifest_path.exists():
            return {}
        try:
            import json

            data = json.loads(self._manifest_path.read_text())
            return {k: QueuedFile.model_validate(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}")
            return {}

    def _save_manifest(self, manifest: dict[str, QueuedFile]) -> None:
        """Save queue manifest to disk."""
        import json

        data = {k: v.model_dump() for k, v in manifest.items()}
        self._manifest_path.write_text(json.dumps(data, indent=2))

    def add_file(
        self,
        facility_id: str,
        remote_path: str,
        content: str,
        language: str,
        description: str | None = None,
    ) -> QueuedFile:
        """Add a file to the embedding queue.

        Args:
            facility_id: Facility identifier (e.g., "epfl")
            remote_path: Original remote file path
            content: File content to embed
            language: Programming language (python, matlab, etc.)
            description: Optional description

        Returns:
            QueuedFile object representing the queued item
        """
        file_id = self._generate_file_id(facility_id, remote_path)
        local_path = self.staging_dir / f"{file_id}_{Path(remote_path).name}"

        # Write content to staging
        local_path.write_text(content, encoding="utf-8")

        queued = QueuedFile(
            id=file_id,
            facility_id=facility_id,
            remote_path=remote_path,
            local_path=str(local_path),
            language=language,
            description=description,
        )

        # Update manifest
        manifest = self._load_manifest()
        manifest[file_id] = queued
        self._save_manifest(manifest)

        logger.info(f"Queued file: {remote_path} -> {local_path}")
        return queued

    def add_files(
        self,
        facility_id: str,
        files: list[tuple[str, str, str]],
        description: str | None = None,
    ) -> list[QueuedFile]:
        """Add multiple files to the queue.

        Args:
            facility_id: Facility identifier
            files: List of (remote_path, content, language) tuples
            description: Optional description for all files

        Returns:
            List of QueuedFile objects
        """
        queued = []
        manifest = self._load_manifest()

        for remote_path, content, language in files:
            file_id = self._generate_file_id(facility_id, remote_path)
            local_path = self.staging_dir / f"{file_id}_{Path(remote_path).name}"
            local_path.write_text(content, encoding="utf-8")

            qf = QueuedFile(
                id=file_id,
                facility_id=facility_id,
                remote_path=remote_path,
                local_path=str(local_path),
                language=language,
                description=description,
            )
            manifest[file_id] = qf
            queued.append(qf)

        self._save_manifest(manifest)
        logger.info(f"Queued {len(queued)} files for facility {facility_id}")
        return queued

    def get_pending(self) -> list[QueuedFile]:
        """Get all files pending processing."""
        manifest = self._load_manifest()
        return [f for f in manifest.values() if f.status == FileStatus.PENDING]

    def get_status(self) -> dict[str, Any]:
        """Get queue status summary."""
        manifest = self._load_manifest()
        by_status: dict[str, int] = {}
        for f in manifest.values():
            by_status[f.status] = by_status.get(f.status, 0) + 1
        return {
            "total": len(manifest),
            "by_status": by_status,
            "staging_dir": str(self.staging_dir),
        }

    def mark_processing(self, file_id: str) -> None:
        """Mark a file as currently being processed."""
        manifest = self._load_manifest()
        if file_id in manifest:
            manifest[file_id].status = FileStatus.PROCESSING
            self._save_manifest(manifest)

    def mark_completed(self, file_id: str) -> None:
        """Mark a file as completed and remove from staging."""
        manifest = self._load_manifest()
        if file_id in manifest:
            qf = manifest[file_id]
            qf.status = FileStatus.COMPLETED
            qf.processed_at = datetime.now(UTC).isoformat()

            # Remove staged file
            local_path = Path(qf.local_path)
            if local_path.exists():
                local_path.unlink()
                logger.debug(f"Removed staged file: {local_path}")

            self._save_manifest(manifest)

    def mark_failed(self, file_id: str, error: str) -> None:
        """Mark a file as failed with error message."""
        manifest = self._load_manifest()
        if file_id in manifest:
            manifest[file_id].status = FileStatus.FAILED
            manifest[file_id].error = error
            manifest[file_id].processed_at = datetime.now(UTC).isoformat()
            self._save_manifest(manifest)

    def clear_completed(self) -> int:
        """Remove completed entries from manifest."""
        manifest = self._load_manifest()
        original_count = len(manifest)
        manifest = {
            k: v for k, v in manifest.items() if v.status != FileStatus.COMPLETED
        }
        self._save_manifest(manifest)
        return original_count - len(manifest)

    def clear_all(self) -> None:
        """Clear all queued files and staging directory."""
        # Remove all staged files
        for f in self.staging_dir.iterdir():
            if f.name != self.manifest_file:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)

        # Clear manifest
        self._save_manifest({})
        logger.info("Cleared embedding queue")


__all__ = ["EmbeddingQueue", "QueuedFile", "FileStatus"]
