"""
Storage and persistence for physics extraction system.

Handles saving/loading of physics databases, progress tracking,
and conflict management.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import shutil
from contextlib import contextmanager

from .models import (
    PhysicsQuantity,
    PhysicsDatabase,
    ExtractionProgress,
    ConflictResolution,
    ConflictResolutionStrategy,
    ExtractionStatus,
)

logger = logging.getLogger(__name__)


@contextmanager
def file_lock(file_path: Path):
    """
    Simple file locking using lock files.

    Args:
        file_path: Path to file to lock
    """
    lock_file = file_path.with_suffix(file_path.suffix + ".lock")

    try:
        # Simple lock file approach - create lock file
        if lock_file.exists():
            # Check if lock is stale (older than 1 hour)
            try:
                lock_time = datetime.fromtimestamp(lock_file.stat().st_mtime)
                if (datetime.now() - lock_time).total_seconds() > 3600:
                    lock_file.unlink()
                else:
                    raise RuntimeError(f"File is locked: {file_path}")
            except OSError:
                pass  # Lock file might be corrupted, proceed

        # Create lock file
        with open(lock_file, "w") as f:
            f.write(f"Locked at {datetime.now().isoformat()}")

        yield

    finally:
        # Clean up lock file
        if lock_file.exists():
            try:
                lock_file.unlink()
            except Exception:
                pass


class PhysicsStorage:
    """
    Handles storage and retrieval of physics databases.

    Provides atomic saves, backup management, and data integrity.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.database_file = self.storage_dir / "physics_database.json"
        self.backup_dir = self.storage_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

    def save_database(
        self, database: PhysicsDatabase, create_backup: bool = True
    ) -> bool:
        """
        Save physics database with atomic write and optional backup.

        Args:
            database: PhysicsDatabase to save
            create_backup: Whether to create backup before saving

        Returns:
            True if successful, False otherwise
        """
        try:
            with file_lock(self.database_file):
                # Create backup if requested and file exists
                if create_backup and self.database_file.exists():
                    self._create_backup()

                # Atomic write using temporary file
                temp_file = self.database_file.with_suffix(".tmp")

                # Convert to JSON-serializable format
                database_dict = self._database_to_dict(database)

                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(database_dict, f, indent=2, default=str)

                # Atomic move
                temp_file.replace(self.database_file)

                logger.info(
                    f"Saved physics database with {database.total_quantities} quantities"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            return False

    def load_database(self) -> Optional[PhysicsDatabase]:
        """
        Load physics database from storage.

        Returns:
            PhysicsDatabase if successful, None otherwise
        """
        if not self.database_file.exists():
            logger.info("No existing database found, creating new one")
            return PhysicsDatabase(version="1.0.0")

        try:
            with file_lock(self.database_file):
                with open(self.database_file, "r", encoding="utf-8") as f:
                    database_dict = json.load(f)

                database = self._dict_to_database(database_dict)
                logger.info(
                    f"Loaded physics database with {database.total_quantities} quantities"
                )
                return database

        except Exception as e:
            logger.error(f"Failed to load database: {e}")

            # Try to restore from backup
            backup_database = self._restore_from_backup()
            if backup_database:
                logger.info("Restored database from backup")
                return backup_database

            # Create new database if all else fails
            logger.warning("Creating new database due to load failure")
            return PhysicsDatabase(version="1.0.0")

    def _create_backup(self):
        """Create timestamped backup of current database."""
        if not self.database_file.exists():
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"physics_database_{timestamp}.json"

        try:
            shutil.copy2(self.database_file, backup_file)

            # Keep only last 10 backups
            backups = sorted(self.backup_dir.glob("physics_database_*.json"))
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    old_backup.unlink()

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")

    def _restore_from_backup(self) -> Optional[PhysicsDatabase]:
        """Restore database from most recent backup."""
        backups = sorted(self.backup_dir.glob("physics_database_*.json"), reverse=True)

        for backup_file in backups:
            try:
                with open(backup_file, "r", encoding="utf-8") as f:
                    database_dict = json.load(f)

                database = self._dict_to_database(database_dict)
                logger.info(f"Restored from backup: {backup_file.name}")
                return database

            except Exception as e:
                logger.error(f"Failed to restore from {backup_file.name}: {e}")
                continue

        return None

    def _database_to_dict(self, database: PhysicsDatabase) -> Dict[str, Any]:
        """Convert PhysicsDatabase to JSON-serializable dictionary."""
        quantities_dict = {}
        for qid, quantity in database.quantities.items():
            quantities_dict[qid] = {
                "id": quantity.id,
                "name": quantity.name,
                "symbol": quantity.symbol,
                "unit": quantity.unit,
                "dimensions": quantity.dimensions,
                "typical_range": quantity.typical_range,
                "description": quantity.description,
                "physics_context": quantity.physics_context,
                "related_quantities": quantity.related_quantities,
                "imas_paths": quantity.imas_paths,
                "ids_sources": list(quantity.ids_sources),
                "extraction_confidence": quantity.extraction_confidence,
                "ai_source": quantity.ai_source,
                "human_verified": quantity.human_verified,
                "last_updated": quantity.last_updated.isoformat(),
                "version": quantity.version,
                "created_by": quantity.created_by,
            }

        return {
            "version": database.version,
            "created_at": database.created_at.isoformat(),
            "last_updated": database.last_updated.isoformat(),
            "quantities": quantities_dict,
            "quantities_by_name": database.quantities_by_name,
            "quantities_by_ids": database.quantities_by_ids,
            "quantities_by_context": database.quantities_by_context,
            "total_quantities": database.total_quantities,
            "verified_quantities": database.verified_quantities,
            "coverage_by_ids": database.coverage_by_ids,
            "extraction_sessions": database.extraction_sessions,
            "conflicts_resolved": database.conflicts_resolved,
        }

    def _dict_to_database(self, data: Dict[str, Any]) -> PhysicsDatabase:
        """Convert dictionary to PhysicsDatabase."""
        # Reconstruct quantities
        quantities = {}
        for qid, q_data in data.get("quantities", {}).items():
            quantities[qid] = PhysicsQuantity(
                id=q_data["id"],
                name=q_data["name"],
                symbol=q_data.get("symbol"),
                unit=q_data.get("unit"),
                dimensions=q_data.get("dimensions"),
                typical_range=q_data.get("typical_range"),
                description=q_data["description"],
                physics_context=q_data.get("physics_context"),
                related_quantities=q_data.get("related_quantities", []),
                imas_paths=q_data.get("imas_paths", []),
                ids_sources=set(q_data.get("ids_sources", [])),
                extraction_confidence=q_data.get("extraction_confidence", 0.0),
                ai_source=q_data.get("ai_source"),
                human_verified=q_data.get("human_verified", False),
                last_updated=datetime.fromisoformat(q_data["last_updated"]),
                version=q_data.get("version", 1),
                created_by=q_data.get("created_by", "unknown"),
            )

        return PhysicsDatabase(
            version=data.get("version", "1.0.0"),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            quantities=quantities,
            quantities_by_name=data.get("quantities_by_name", {}),
            quantities_by_ids=data.get("quantities_by_ids", {}),
            quantities_by_context=data.get("quantities_by_context", {}),
            total_quantities=data.get("total_quantities", len(quantities)),
            verified_quantities=data.get("verified_quantities", 0),
            coverage_by_ids=data.get("coverage_by_ids", {}),
            extraction_sessions=data.get("extraction_sessions", []),
            conflicts_resolved=data.get("conflicts_resolved", 0),
        )


class ProgressTracker:
    """
    Tracks and persists extraction progress.

    Provides resumable processing and detailed progress reporting.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.progress_file = self.storage_dir / "extraction_progress.json"

    def save_progress(self, progress: ExtractionProgress) -> bool:
        """Save extraction progress."""
        try:
            with file_lock(self.progress_file):
                progress_dict = {
                    "session_id": progress.session_id,
                    "started_at": progress.started_at.isoformat(),
                    "last_updated": progress.last_updated.isoformat(),
                    "total_ids": progress.total_ids,
                    "completed_ids": progress.completed_ids,
                    "failed_ids": progress.failed_ids,
                    "ids_status": {k: v.value for k, v in progress.ids_status.items()},
                    "ids_progress": progress.ids_progress,
                    "total_quantities_found": progress.total_quantities_found,
                    "total_quantities_updated": progress.total_quantities_updated,
                    "total_conflicts": progress.total_conflicts,
                    "total_paths_processed": progress.total_paths_processed,
                    "total_processing_time": progress.total_processing_time,
                    "paths_per_batch": progress.paths_per_batch,
                    "confidence_threshold": progress.confidence_threshold,
                }

                with open(self.progress_file, "w", encoding="utf-8") as f:
                    json.dump(progress_dict, f, indent=2)

                return True

        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
            return False

    def load_progress(self) -> Optional[ExtractionProgress]:
        """Load extraction progress."""
        if not self.progress_file.exists():
            return None

        try:
            with file_lock(self.progress_file):
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Convert status strings back to enums
                ids_status = {
                    k: ExtractionStatus(v)
                    for k, v in data.get("ids_status", {}).items()
                }

                progress = ExtractionProgress(
                    session_id=data["session_id"],
                    started_at=datetime.fromisoformat(data["started_at"]),
                    last_updated=datetime.fromisoformat(data["last_updated"]),
                    total_ids=data.get("total_ids", 0),
                    completed_ids=data.get("completed_ids", 0),
                    failed_ids=data.get("failed_ids", 0),
                    ids_status=ids_status,
                    ids_progress=data.get("ids_progress", {}),
                    total_quantities_found=data.get("total_quantities_found", 0),
                    total_quantities_updated=data.get("total_quantities_updated", 0),
                    total_conflicts=data.get("total_conflicts", 0),
                    total_paths_processed=data.get("total_paths_processed", 0),
                    total_processing_time=data.get("total_processing_time", 0.0),
                    paths_per_batch=data.get("paths_per_batch", 10),
                    confidence_threshold=data.get("confidence_threshold", 0.5),
                )

                return progress

        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return None

    def get_remaining_ids(self, all_ids: List[str]) -> List[str]:
        """Get list of IDS that haven't been completed yet."""
        progress = self.load_progress()
        if not progress:
            return all_ids

        completed_or_failed = set()
        for ids_name, status in progress.ids_status.items():
            if status in [ExtractionStatus.COMPLETED, ExtractionStatus.FAILED]:
                completed_or_failed.add(ids_name)

        return [ids_name for ids_name in all_ids if ids_name not in completed_or_failed]


class ConflictManager:
    """
    Manages conflicts between physics quantities.

    Handles detection, resolution, and tracking of conflicts.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.conflicts_file = self.storage_dir / "conflicts.json"

    def detect_conflicts(
        self,
        existing_quantities: List[PhysicsQuantity],
        new_quantities: List[PhysicsQuantity],
    ) -> List[ConflictResolution]:
        """
        Detect conflicts between existing and new quantities.

        Args:
            existing_quantities: Current quantities in database
            new_quantities: New quantities to check

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Create lookup by name for efficiency
        existing_by_name = {q.name: q for q in existing_quantities}

        for new_quantity in new_quantities:
            if new_quantity.name in existing_by_name:
                existing_quantity = existing_by_name[new_quantity.name]

                # Check for differences
                conflict_fields = self._find_conflicting_fields(
                    existing_quantity, new_quantity
                )

                if conflict_fields:
                    conflict = ConflictResolution(
                        quantity_name=new_quantity.name,
                        existing_quantity=existing_quantity,
                        new_quantity=new_quantity,
                        conflict_fields=conflict_fields,
                        requires_human_review=self._requires_human_review(
                            conflict_fields
                        ),
                    )
                    conflicts.append(conflict)

        return conflicts

    def _find_conflicting_fields(
        self, existing: PhysicsQuantity, new: PhysicsQuantity
    ) -> List[str]:
        """Find fields that differ between quantities."""
        conflicts = []

        # Check important fields for conflicts
        fields_to_check = [
            "description",
            "unit",
            "dimensions",
            "typical_range",
            "physics_context",
            "symbol",
        ]

        for field in fields_to_check:
            existing_val = getattr(existing, field)
            new_val = getattr(new, field)

            # Handle None values
            if existing_val != new_val:
                # Don't conflict if one is None and the other isn't empty
                if existing_val is None and new_val:
                    continue
                if new_val is None and existing_val:
                    continue

                conflicts.append(field)

        # Check lists/sets
        if set(existing.imas_paths) != set(new.imas_paths):
            conflicts.append("imas_paths")

        if existing.ids_sources != new.ids_sources:
            conflicts.append("ids_sources")

        return conflicts

    def _requires_human_review(self, conflict_fields: List[str]) -> bool:
        """Determine if conflicts require human review."""
        critical_fields = {"description", "unit", "dimensions"}
        return bool(set(conflict_fields) & critical_fields)

    def resolve_conflict(
        self, conflict: ConflictResolution, strategy: ConflictResolutionStrategy
    ) -> PhysicsQuantity:
        """
        Resolve a conflict using the specified strategy.

        Args:
            conflict: ConflictResolution to resolve
            strategy: How to resolve the conflict

        Returns:
            Resolved PhysicsQuantity
        """
        resolved_quantity = conflict.resolve(strategy, "system")

        # Save conflict resolution
        self._save_conflict_resolution(conflict)

        return resolved_quantity

    def auto_resolve_conflicts(
        self, conflicts: List[ConflictResolution]
    ) -> Tuple[List[PhysicsQuantity], List[ConflictResolution]]:
        """
        Automatically resolve conflicts that don't require human review.

        Args:
            conflicts: List of conflicts to resolve

        Returns:
            Tuple of (resolved_quantities, unresolved_conflicts)
        """
        resolved_quantities = []
        unresolved_conflicts = []

        for conflict in conflicts:
            if not conflict.requires_human_review:
                try:
                    # Use MERGE strategy for auto-resolution
                    resolved = self.resolve_conflict(
                        conflict, ConflictResolutionStrategy.MERGE
                    )
                    resolved_quantities.append(resolved)
                except Exception as e:
                    logger.error(
                        f"Failed to auto-resolve conflict for {conflict.quantity_name}: {e}"
                    )
                    unresolved_conflicts.append(conflict)
            else:
                unresolved_conflicts.append(conflict)

        return resolved_quantities, unresolved_conflicts

    def _save_conflict_resolution(self, conflict: ConflictResolution):
        """Save conflict resolution for tracking."""
        try:
            # Load existing conflicts
            conflicts_data = []
            if self.conflicts_file.exists():
                with open(self.conflicts_file, "r", encoding="utf-8") as f:
                    conflicts_data = json.load(f)

            # Add new conflict
            conflict_dict = {
                "conflict_id": conflict.conflict_id,
                "quantity_name": conflict.quantity_name,
                "conflict_fields": conflict.conflict_fields,
                "resolution_strategy": conflict.resolution_strategy.value
                if conflict.resolution_strategy
                else None,
                "resolved_at": conflict.resolved_at.isoformat()
                if conflict.resolved_at
                else None,
                "resolved_by": conflict.resolved_by,
                "is_resolved": conflict.is_resolved,
                "requires_human_review": conflict.requires_human_review,
            }

            conflicts_data.append(conflict_dict)

            # Save back to file
            with open(self.conflicts_file, "w", encoding="utf-8") as f:
                json.dump(conflicts_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save conflict resolution: {e}")

    def get_unresolved_conflicts(self) -> List[Dict[str, Any]]:
        """Get list of unresolved conflicts requiring human review."""
        if not self.conflicts_file.exists():
            return []

        try:
            with open(self.conflicts_file, "r", encoding="utf-8") as f:
                conflicts_data = json.load(f)

            return [
                conflict
                for conflict in conflicts_data
                if not conflict.get("is_resolved", False)
                and conflict.get("requires_human_review", False)
            ]

        except Exception as e:
            logger.error(f"Failed to load conflicts: {e}")
            return []
