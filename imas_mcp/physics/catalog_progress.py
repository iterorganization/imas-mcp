"""
Progress tracking using IDS catalog metadata.

Provides detailed progress tracking based on the IMAS IDS catalog,
including per-IDS path counts and physics domain categorization.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from imas_mcp.physics.models import ExtractionStatus

logger = logging.getLogger(__name__)


class CatalogBasedProgressTracker:
    """
    Enhanced progress tracker that uses IDS catalog metadata.

    Provides detailed tracking including:
    - Per-IDS path counts from catalog
    - Physics domain categorization
    - Weighted progress based on actual path counts
    """

    def __init__(self, storage_dir: Path, catalog_file: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.catalog_file = Path(catalog_file)
        self.progress_file = self.storage_dir / "extraction_progress.json"

        # Load catalog metadata
        self._load_catalog()

    def _load_catalog(self):
        """Load IDS catalog metadata."""
        try:
            with open(self.catalog_file, encoding="utf-8") as f:
                catalog_data = json.load(f)

            self.catalog_metadata = catalog_data.get("metadata", {})
            self.ids_catalog = catalog_data.get("ids_catalog", {})

            logger.info(f"Loaded catalog with {len(self.ids_catalog)} IDS definitions")

        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            self.catalog_metadata = {}
            self.ids_catalog = {}

    def get_catalog_stats(self) -> dict[str, Any]:
        """Get overall catalog statistics."""
        return {
            "total_ids": self.catalog_metadata.get("total_ids", 0),
            "total_paths": self.catalog_metadata.get("total_paths", 0),
            "total_leaf_nodes": self.catalog_metadata.get("total_leaf_nodes", 0),
            "catalog_version": self.catalog_metadata.get("version", "unknown"),
            "generation_date": self.catalog_metadata.get("generation_date", "unknown"),
        }

    def get_ids_info(self, ids_name: str) -> dict[str, Any] | None:
        """Get information about a specific IDS from catalog."""
        return self.ids_catalog.get(ids_name)

    def get_ids_path_count(self, ids_name: str) -> int:
        """Get the number of paths for a specific IDS."""
        ids_info = self.get_ids_info(ids_name)
        return ids_info.get("path_count", 0) if ids_info else 0

    def get_available_ids_with_counts(self) -> dict[str, int]:
        """Get all available IDS with their path counts."""
        return {
            ids_name: info.get("path_count", 0)
            for ids_name, info in self.ids_catalog.items()
        }

    def get_physics_domains(self) -> dict[str, list[str]]:
        """Group IDS by physics domain."""
        domains = {}
        for ids_name, info in self.ids_catalog.items():
            domain = info.get("physics_domain", "general")
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(ids_name)
        return domains

    def create_enhanced_progress(
        self,
        session_id: str,
        ids_list: list[str],
        paths_per_ids: int = 10,
        confidence_threshold: float = 0.5,
    ) -> "EnhancedExtractionProgress":
        """
        Create enhanced progress tracker with catalog metadata.

        Args:
            session_id: Session identifier
            ids_list: List of IDS to process
            paths_per_ids: Default paths per IDS (may be overridden by catalog)
            confidence_threshold: AI confidence threshold

        Returns:
            EnhancedExtractionProgress instance
        """
        # Calculate total paths from catalog
        total_catalog_paths = sum(
            self.get_ids_path_count(ids_name) for ids_name in ids_list
        )

        # Calculate actual paths to process
        total_processing_paths = 0
        ids_path_info = {}

        for ids_name in ids_list:
            catalog_paths = self.get_ids_path_count(ids_name)
            processing_paths = (
                min(paths_per_ids, catalog_paths)
                if catalog_paths > 0
                else paths_per_ids
            )

            ids_path_info[ids_name] = {
                "catalog_paths": catalog_paths,
                "processing_paths": processing_paths,
                "physics_domain": self.ids_catalog.get(ids_name, {}).get(
                    "physics_domain", "general"
                ),
                "description": self.ids_catalog.get(ids_name, {}).get(
                    "description", ""
                ),
            }

            total_processing_paths += processing_paths

        return EnhancedExtractionProgress(
            session_id=session_id,
            total_ids=len(ids_list),
            total_catalog_paths=total_catalog_paths,
            total_processing_paths=total_processing_paths,
            ids_path_info=ids_path_info,
            confidence_threshold=confidence_threshold,
        )

    def save_progress(self, progress: "EnhancedExtractionProgress") -> bool:
        """Save enhanced extraction progress."""
        try:
            progress_dict = {
                "session_id": progress.session_id,
                "started_at": progress.started_at.isoformat(),
                "last_updated": progress.last_updated.isoformat(),
                # Basic counts
                "total_ids": progress.total_ids,
                "completed_ids": progress.completed_ids,
                "failed_ids": progress.failed_ids,
                # Enhanced path tracking
                "total_catalog_paths": progress.total_catalog_paths,
                "total_processing_paths": progress.total_processing_paths,
                "processed_paths": progress.processed_paths,
                # Per-IDS information
                "ids_path_info": progress.ids_path_info,
                "ids_status": {k: v.value for k, v in progress.ids_status.items()},
                "ids_progress": progress.ids_progress,
                "ids_processed_paths": progress.ids_processed_paths,
                # Results
                "total_quantities_found": progress.total_quantities_found,
                "total_quantities_updated": progress.total_quantities_updated,
                "total_conflicts": progress.total_conflicts,
                "total_processing_time": progress.total_processing_time,
                # Configuration
                "confidence_threshold": progress.confidence_threshold,
                # Domain statistics
                "domain_stats": progress.domain_stats,
            }

            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_dict, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to save enhanced progress: {e}")
            return False

    def load_progress(self) -> Optional["EnhancedExtractionProgress"]:
        """Load enhanced extraction progress."""
        if not self.progress_file.exists():
            return None

        try:
            with open(self.progress_file, encoding="utf-8") as f:
                data = json.load(f)

            # Convert status strings back to enums
            ids_status = {
                k: ExtractionStatus(v) for k, v in data.get("ids_status", {}).items()
            }

            progress = EnhancedExtractionProgress(
                session_id=data["session_id"],
                started_at=datetime.fromisoformat(data["started_at"]),
                last_updated=datetime.fromisoformat(data["last_updated"]),
                total_ids=data.get("total_ids", 0),
                completed_ids=data.get("completed_ids", 0),
                failed_ids=data.get("failed_ids", 0),
                total_catalog_paths=data.get("total_catalog_paths", 0),
                total_processing_paths=data.get("total_processing_paths", 0),
                processed_paths=data.get("processed_paths", 0),
                ids_path_info=data.get("ids_path_info", {}),
                ids_status=ids_status,
                ids_progress=data.get("ids_progress", {}),
                ids_processed_paths=data.get("ids_processed_paths", {}),
                total_quantities_found=data.get("total_quantities_found", 0),
                total_quantities_updated=data.get("total_quantities_updated", 0),
                total_conflicts=data.get("total_conflicts", 0),
                total_processing_time=data.get("total_processing_time", 0.0),
                confidence_threshold=data.get("confidence_threshold", 0.5),
                domain_stats=data.get("domain_stats", {}),
            )

            return progress

        except Exception as e:
            logger.error(f"Failed to load enhanced progress: {e}")
            return None


class EnhancedExtractionProgress:
    """
    Enhanced extraction progress with catalog-based tracking.

    Provides detailed progress information using IDS catalog metadata.
    """

    def __init__(
        self,
        session_id: str,
        total_ids: int = 0,
        total_catalog_paths: int = 0,
        total_processing_paths: int = 0,
        ids_path_info: dict[str, dict[str, Any]] | None = None,
        confidence_threshold: float = 0.5,
        **kwargs,
    ):
        # Session info
        self.session_id = session_id
        self.started_at = kwargs.get("started_at", datetime.utcnow())
        self.last_updated = kwargs.get("last_updated", datetime.utcnow())

        # Basic tracking
        self.total_ids = total_ids
        self.completed_ids = kwargs.get("completed_ids", 0)
        self.failed_ids = kwargs.get("failed_ids", 0)

        # Enhanced path tracking
        self.total_catalog_paths = total_catalog_paths
        self.total_processing_paths = total_processing_paths
        self.processed_paths = kwargs.get("processed_paths", 0)

        # Per-IDS detailed information
        self.ids_path_info = ids_path_info or {}
        self.ids_status: dict[str, ExtractionStatus] = kwargs.get("ids_status", {})
        self.ids_progress: dict[str, float] = kwargs.get("ids_progress", {})
        self.ids_processed_paths: dict[str, int] = kwargs.get("ids_processed_paths", {})

        # Results tracking
        self.total_quantities_found = kwargs.get("total_quantities_found", 0)
        self.total_quantities_updated = kwargs.get("total_quantities_updated", 0)
        self.total_conflicts = kwargs.get("total_conflicts", 0)
        self.total_processing_time = kwargs.get("total_processing_time", 0.0)

        # Configuration
        self.confidence_threshold = confidence_threshold

        # Domain statistics
        self.domain_stats: dict[str, dict[str, int]] = kwargs.get("domain_stats", {})

    @property
    def completion_percentage(self) -> float:
        """Calculate overall completion percentage based on IDS."""
        if self.total_ids == 0:
            return 0.0
        return (self.completed_ids / self.total_ids) * 100.0

    @property
    def path_completion_percentage(self) -> float:
        """Calculate completion percentage based on actual paths processed."""
        if self.total_processing_paths == 0:
            return 0.0
        return (self.processed_paths / self.total_processing_paths) * 100.0

    @property
    def catalog_coverage_percentage(self) -> float:
        """Calculate what percentage of catalog paths we're processing."""
        if self.total_catalog_paths == 0:
            return 0.0
        return (self.total_processing_paths / self.total_catalog_paths) * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if extraction is complete."""
        return self.completed_ids + self.failed_ids == self.total_ids

    def update_ids_progress(
        self,
        ids_name: str,
        status: ExtractionStatus,
        progress: float = 0.0,
        paths_processed: int = 0,
    ):
        """
        Update progress for a specific IDS.

        Args:
            ids_name: IDS name
            status: Current status
            progress: Progress percentage (0.0-1.0)
            paths_processed: Number of paths actually processed
        """
        self.ids_status[ids_name] = status
        self.ids_progress[ids_name] = progress
        self.ids_processed_paths[ids_name] = paths_processed
        self.last_updated = datetime.utcnow()

        # Update overall counters
        completed = sum(
            1 for s in self.ids_status.values() if s == ExtractionStatus.COMPLETED
        )
        failed = sum(
            1 for s in self.ids_status.values() if s == ExtractionStatus.FAILED
        )

        self.completed_ids = completed
        self.failed_ids = failed

        # Update total processed paths
        self.processed_paths = sum(self.ids_processed_paths.values())

        # Update domain statistics
        self._update_domain_stats()

    def _update_domain_stats(self):
        """Update statistics by physics domain."""
        self.domain_stats = {}

        for ids_name, ids_info in self.ids_path_info.items():
            domain = ids_info.get("physics_domain", "general")

            if domain not in self.domain_stats:
                self.domain_stats[domain] = {
                    "total_ids": 0,
                    "completed_ids": 0,
                    "failed_ids": 0,
                    "total_paths": 0,
                    "processed_paths": 0,
                }

            # Update counts
            self.domain_stats[domain]["total_ids"] += 1
            self.domain_stats[domain]["total_paths"] += ids_info.get(
                "processing_paths", 0
            )
            self.domain_stats[domain]["processed_paths"] += (
                self.ids_processed_paths.get(ids_name, 0)
            )

            # Update status counts
            status = self.ids_status.get(ids_name, ExtractionStatus.PENDING)
            if status == ExtractionStatus.COMPLETED:
                self.domain_stats[domain]["completed_ids"] += 1
            elif status == ExtractionStatus.FAILED:
                self.domain_stats[domain]["failed_ids"] += 1

    def get_detailed_status(self) -> dict[str, Any]:
        """Get comprehensive status information."""
        return {
            "session_info": {
                "session_id": self.session_id,
                "started_at": self.started_at.isoformat(),
                "last_updated": self.last_updated.isoformat(),
                "is_complete": self.is_complete,
            },
            "overall_progress": {
                "completion_percentage": self.completion_percentage,
                "path_completion_percentage": self.path_completion_percentage,
                "catalog_coverage_percentage": self.catalog_coverage_percentage,
                "total_ids": self.total_ids,
                "completed_ids": self.completed_ids,
                "failed_ids": self.failed_ids,
                "pending_ids": self.total_ids - self.completed_ids - self.failed_ids,
            },
            "path_statistics": {
                "total_catalog_paths": self.total_catalog_paths,
                "total_processing_paths": self.total_processing_paths,
                "processed_paths": self.processed_paths,
                "remaining_paths": self.total_processing_paths - self.processed_paths,
            },
            "results_summary": {
                "total_quantities_found": self.total_quantities_found,
                "total_quantities_updated": self.total_quantities_updated,
                "total_conflicts": self.total_conflicts,
                "total_processing_time": self.total_processing_time,
                "avg_processing_time_per_ids": (
                    self.total_processing_time / max(1, self.completed_ids)
                    if self.completed_ids > 0
                    else 0.0
                ),
            },
            "domain_breakdown": self.domain_stats,
            "confidence_threshold": self.confidence_threshold,
        }

    def get_ids_details(self) -> dict[str, Any]:
        """Get detailed per-IDS information."""
        details = {}

        for ids_name, ids_info in self.ids_path_info.items():
            status = self.ids_status.get(ids_name, ExtractionStatus.PENDING)
            progress = self.ids_progress.get(ids_name, 0.0)
            processed_paths = self.ids_processed_paths.get(ids_name, 0)

            details[ids_name] = {
                "description": ids_info.get("description", ""),
                "physics_domain": ids_info.get("physics_domain", "general"),
                "catalog_paths": ids_info.get("catalog_paths", 0),
                "processing_paths": ids_info.get("processing_paths", 0),
                "processed_paths": processed_paths,
                "status": status.value,
                "progress_percentage": progress * 100.0,
                "completion_ratio": f"{processed_paths}/{ids_info.get('processing_paths', 0)}",
            }

        return details
