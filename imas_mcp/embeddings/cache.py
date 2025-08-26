"""Embedding cache with validation and metadata."""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EmbeddingCache:
    """Cache for document embeddings with metadata."""

    embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    path_ids: list[str] = field(default_factory=list)
    model_name: str = ""
    created_at: float = field(default_factory=time.time)
    document_count: int = 0
    ids_set: set | None = None  # IDS set used for this cache
    source_content_hash: str = ""  # Hash of source data directory content
    source_max_mtime: float = 0.0  # Maximum modification time of source files
    dd_version: str | None = None  # Data Dictionary version

    def is_valid(
        self,
        current_doc_count: int,
        current_model: str,
        current_ids_set: set | None = None,
        source_data_dir: Path | None = None,
    ) -> bool:
        """Check if cache is valid for current state."""
        return self.validate_with_reason(
            current_doc_count, current_model, current_ids_set, source_data_dir
        )[0]

    def validate_with_reason(
        self,
        current_doc_count: int,
        current_model: str,
        current_ids_set: set | None = None,
        source_data_dir: Path | None = None,
    ) -> tuple[bool, str]:
        """Check if cache is valid and return detailed reason if invalid."""
        # Basic validation checks
        if self.document_count != current_doc_count:
            return (
                False,
                f"Document count mismatch: cached={self.document_count}, current={current_doc_count}",
            )

        if self.model_name != current_model:
            return (
                False,
                f"Model name mismatch: cached='{self.model_name}', current='{current_model}'",
            )

        if len(self.embeddings) == 0:
            return False, "Cache has no embeddings data"

        if len(self.path_ids) == 0:
            return False, "Cache has no path IDs"

        if self.ids_set != current_ids_set:
            return (
                False,
                f"IDS set mismatch: cached={self.ids_set}, current={current_ids_set}",
            )

        # Enhanced validation with DD version checking
        if source_data_dir is not None:
            # Check DD version instead of file modification times
            current_dd_version = self._get_current_dd_version(source_data_dir)
            if self.dd_version and current_dd_version:
                if current_dd_version != self.dd_version:
                    return (
                        False,
                        f"Data Dictionary version changed: cached='{self.dd_version}', current='{current_dd_version}'",
                    )
            elif not self.dd_version:
                # Cache without DD version - rebuild required
                return False, "Cache requires rebuild"

            # Check source content hash if available (fallback validation)
            if self.source_content_hash:
                current_hash = self._compute_source_content_hash(source_data_dir)
                if current_hash != self.source_content_hash:
                    return False, "Source content has changed (hash mismatch)"

        return True, "Cache is valid"

    def _has_modified_source_files(self, source_data_dir: Path) -> bool:
        """Check if any source JSON files have been modified after cache creation."""
        return len(self._get_modified_source_files(source_data_dir)) > 0

    def _get_modified_source_files(self, source_data_dir: Path) -> list[str]:
        """Get list of source files that have been modified after cache creation.

        Returns a list of relative file paths for source files whose modification
        time is newer than the cache creation time, indicating they have been
        changed since the cache was built.
        """
        modified_files = []
        try:
            # Check catalog file
            catalog_path = source_data_dir / "ids_catalog.json"
            if catalog_path.exists() and catalog_path.stat().st_mtime > self.created_at:
                modified_files.append("ids_catalog.json")

            # Check detailed files
            detailed_dir = source_data_dir / "detailed"
            if detailed_dir.exists():
                for json_file in detailed_dir.glob("*.json"):
                    if json_file.stat().st_mtime > self.created_at:
                        modified_files.append(f"detailed/{json_file.name}")

        except Exception:
            # If we can't check files, assume they might have been modified
            pass

        return modified_files

    def _compute_source_content_hash(self, source_data_dir: Path) -> str:
        """Compute hash of source data directory content."""
        hash_data = str(source_data_dir.resolve())

        # Include IDS set in hash for proper cache isolation
        if self.ids_set:
            ids_str = "|".join(sorted(self.ids_set))
            hash_data += f"|ids:{ids_str}"

        return hashlib.md5(hash_data.encode()).hexdigest()

    def _get_max_source_mtime(self, source_data_dir: Path) -> float:
        """Get the maximum modification time of all source files."""
        max_mtime = 0.0

        try:
            # Check catalog file
            catalog_path = source_data_dir / "ids_catalog.json"
            if catalog_path.exists():
                max_mtime = max(max_mtime, catalog_path.stat().st_mtime)

            # Check detailed files
            detailed_dir = source_data_dir / "detailed"
            if detailed_dir.exists():
                for json_file in detailed_dir.glob("*.json"):
                    max_mtime = max(max_mtime, json_file.stat().st_mtime)
        except Exception:
            pass

        return max_mtime

    def update_source_metadata(self, source_data_dir: Path) -> None:
        """Update source file metadata for cache validation."""
        self.source_content_hash = self._compute_source_content_hash(source_data_dir)
        self.source_max_mtime = self._get_max_source_mtime(source_data_dir)
        self.dd_version = self._get_current_dd_version(source_data_dir)

    def _get_current_dd_version(self, source_data_dir: Path) -> str | None:
        """Get the current Data Dictionary version from catalog."""
        try:
            catalog_path = source_data_dir / "ids_catalog.json"
            if catalog_path.exists():
                with open(catalog_path, encoding="utf-8") as f:
                    catalog_data = json.load(f)
                return catalog_data.get("metadata", {}).get("version")
        except Exception:
            pass
        return None
