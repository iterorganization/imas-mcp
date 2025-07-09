"""
Coordination and orchestration for physics extraction system.

Handles high-level coordination of extraction processes, resource management,
and user interaction workflows.
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import json

from .models import (
    PhysicsDatabase,
    ExtractionProgress,
    ExtractionResult,
    ExtractionStatus,
    ConflictResolutionStrategy,
)

from .catalog_progress import CatalogBasedProgressTracker
from .extractors import AIPhysicsExtractor, BatchProcessor, CatalogBatchProcessor
from .storage import PhysicsStorage, ProgressTracker, ConflictManager

logger = logging.getLogger(__name__)


class LockManager:
    """
    Simple lock management for coordinating concurrent access.

    Prevents multiple extraction processes from interfering with each other.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.lock_file = self.storage_dir / "extraction.lock"

    def acquire_lock(self, session_id: str) -> bool:
        """
        Acquire extraction lock.

        Args:
            session_id: ID of the session requesting lock

        Returns:
            True if lock acquired, False otherwise
        """
        if self.lock_file.exists():
            # Check if lock is stale (older than 1 hour)
            try:
                lock_data = json.loads(self.lock_file.read_text())
                lock_time = datetime.fromisoformat(lock_data["timestamp"])

                if (datetime.utcnow() - lock_time).total_seconds() > 3600:
                    logger.warning("Removing stale extraction lock")
                    self.lock_file.unlink()
                else:
                    logger.info(
                        f"Extraction locked by session {lock_data['session_id']}"
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to read lock file: {e}")
                # Remove corrupted lock file
                self.lock_file.unlink()

        # Create new lock
        try:
            lock_data = {
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.lock_file.write_text(json.dumps(lock_data, indent=2))
            logger.info(f"Acquired extraction lock for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False

    def release_lock(self, session_id: str) -> bool:
        """
        Release extraction lock.

        Args:
            session_id: ID of the session releasing lock

        Returns:
            True if lock released, False otherwise
        """
        if not self.lock_file.exists():
            return True

        try:
            lock_data = json.loads(self.lock_file.read_text())
            if lock_data["session_id"] == session_id:
                self.lock_file.unlink()
                logger.info(f"Released extraction lock for session {session_id}")
                return True
            else:
                logger.warning("Cannot release lock owned by different session")
                return False
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
            return False

    def is_locked(self) -> bool:
        """Check if extraction is currently locked."""
        return self.lock_file.exists()


class ExtractionCoordinator:
    """
    Main coordinator for physics extraction operations.

    Provides high-level interface for managing extraction workflows,
    handling conflicts, and tracking progress.
    """

    def __init__(
        self,
        json_data_dir: Path,
        storage_dir: Path,
        catalog_file: Optional[Path] = None,
        ai_model: str = "gpt-4",
        confidence_threshold: float = 0.5,
    ):
        self.json_data_dir = Path(json_data_dir)
        self.storage_dir = Path(storage_dir)

        # Initialize components
        self.extractor = AIPhysicsExtractor(
            ai_model=ai_model, confidence_threshold=confidence_threshold
        )
        self.batch_processor = BatchProcessor(
            json_data_dir=json_data_dir, extractor=self.extractor
        )

        # Storage components
        self.physics_storage = PhysicsStorage(storage_dir)
        self.progress_tracker = ProgressTracker(storage_dir)
        self.conflict_manager = ConflictManager(storage_dir)
        self.lock_manager = LockManager(storage_dir)

        # Initialize catalog components if catalog is available
        if catalog_file and Path(catalog_file).exists():
            self.catalog_file = Path(catalog_file)
            self.catalog_batch_processor = CatalogBatchProcessor(
                json_data_dir=json_data_dir,
                catalog_file=catalog_file,
                extractor=self.extractor,
            )
            self.catalog_progress_tracker = CatalogBasedProgressTracker(
                storage_dir=storage_dir, catalog_file=catalog_file
            )
            logger.info("Extraction coordinator initialized with catalog support")
        else:
            self.catalog_file = None
            self.catalog_batch_processor = None
            self.catalog_progress_tracker = None
            logger.info("Basic extraction coordinator initialized (no catalog)")

        # Load or create database
        loaded_database = self.physics_storage.load_database()
        self.database: PhysicsDatabase = (
            loaded_database
            if loaded_database is not None
            else PhysicsDatabase(version="1.0.0")
        )

    def get_extraction_status(self) -> Dict[str, Any]:
        """
        Get current extraction status and statistics.

        Returns:
            Dictionary with status information
        """
        progress = self.progress_tracker.load_progress()
        available_ids = self.batch_processor.get_available_ids()

        if progress:
            remaining_ids = self.progress_tracker.get_remaining_ids(available_ids)
        else:
            remaining_ids = available_ids

        unresolved_conflicts = self.conflict_manager.get_unresolved_conflicts()

        return {
            "database_stats": self.database.get_stats(),
            "available_ids": len(available_ids),
            "remaining_ids": len(remaining_ids),
            "remaining_ids_list": remaining_ids[:10],  # Show first 10
            "extraction_progress": {
                "completion_percentage": progress.completion_percentage
                if progress
                else 0.0,
                "total_ids": progress.total_ids if progress else 0,
                "completed_ids": progress.completed_ids if progress else 0,
                "failed_ids": progress.failed_ids if progress else 0,
                "is_complete": progress.is_complete if progress else False,
            },
            "conflicts": {
                "total_unresolved": len(unresolved_conflicts),
                "requiring_review": len(
                    [c for c in unresolved_conflicts if c.get("requires_human_review")]
                ),
            },
            "is_locked": self.lock_manager.is_locked(),
            "last_updated": datetime.utcnow().isoformat(),
        }

    def start_extraction(
        self,
        ids_list: Optional[List[str]] = None,
        paths_per_ids: int = 10,
        auto_resolve_conflicts: bool = True,
    ) -> str:
        """
        Start physics extraction process.

        Args:
            ids_list: List of IDS to process (None for all available)
            paths_per_ids: Number of paths to process per IDS
            auto_resolve_conflicts: Whether to auto-resolve simple conflicts

        Returns:
            Session ID for tracking progress
        """
        # Generate session ID
        session_id = f"extraction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Acquire lock
        if not self.lock_manager.acquire_lock(session_id):
            raise RuntimeError("Extraction is already in progress")

        try:
            # Get IDS list
            if ids_list is None:
                available_ids = self.batch_processor.get_available_ids()
                ids_list = self.progress_tracker.get_remaining_ids(available_ids)

            # Initialize progress
            progress = ExtractionProgress(
                session_id=session_id,
                total_ids=len(ids_list),
                paths_per_batch=paths_per_ids,
            )

            # Mark all IDS as pending
            for ids_name in ids_list:
                progress.update_ids_progress(ids_name, ExtractionStatus.PENDING)

            # Save initial progress
            self.progress_tracker.save_progress(progress)

            logger.info(
                f"Started extraction session {session_id} for {len(ids_list)} IDS"
            )
            return session_id

        except Exception as e:
            self.lock_manager.release_lock(session_id)
            raise e

    def process_next_ids(self, session_id: str) -> Optional[ExtractionResult]:
        """
        Process the next IDS in the extraction queue.

        Args:
            session_id: Session ID

        Returns:
            ExtractionResult for the processed IDS, or None if complete
        """
        # Load current progress
        progress = self.progress_tracker.load_progress()
        if not progress or progress.session_id != session_id:
            raise ValueError(f"Invalid session ID: {session_id}")

        # Find next IDS to process
        next_ids = None
        for ids_name, status in progress.ids_status.items():
            if status == ExtractionStatus.PENDING:
                next_ids = ids_name
                break

        if not next_ids:
            logger.info("No more IDS to process")
            return None

        # Update status to in progress
        progress.update_ids_progress(next_ids, ExtractionStatus.IN_PROGRESS)
        self.progress_tracker.save_progress(progress)

        try:
            # Process the IDS
            result = self.batch_processor.process_ids(
                next_ids, progress.paths_per_batch
            )

            # Update progress with results
            if result.status == ExtractionStatus.COMPLETED:
                progress.update_ids_progress(next_ids, ExtractionStatus.COMPLETED, 1.0)
                progress.total_quantities_found += len(result.quantities_found)
                progress.total_paths_processed += len(result.paths_processed)
                progress.total_processing_time += result.processing_time

                # Add quantities to database and handle conflicts
                self._integrate_extraction_results(result, progress)

            else:
                progress.update_ids_progress(next_ids, ExtractionStatus.FAILED, 0.0)

            # Save updated progress
            self.progress_tracker.save_progress(progress)

            logger.info(f"Processed {next_ids}: {result.status.value}")
            return result

        except Exception as e:
            logger.error(f"Failed to process {next_ids}: {e}")
            progress.update_ids_progress(next_ids, ExtractionStatus.FAILED, 0.0)
            self.progress_tracker.save_progress(progress)
            raise e

    def _integrate_extraction_results(
        self, result: ExtractionResult, progress: ExtractionProgress
    ):
        """
        Integrate extraction results into the database.

        Args:
            result: ExtractionResult to integrate
            progress: Current progress tracker
        """
        if not result.quantities_found:
            return

        # Get existing quantities for conflict detection
        existing_quantities = list(self.database.quantities.values())

        # Detect conflicts
        conflicts = self.conflict_manager.detect_conflicts(
            existing_quantities, result.quantities_found
        )

        if conflicts:
            logger.info(f"Detected {len(conflicts)} conflicts for {result.ids_name}")
            progress.total_conflicts += len(conflicts)

            # Auto-resolve simple conflicts
            resolved_quantities, unresolved_conflicts = (
                self.conflict_manager.auto_resolve_conflicts(conflicts)
            )

            # Add resolved quantities
            for quantity in resolved_quantities:
                if quantity.name in self.database.quantities_by_name:
                    # Update existing
                    self.database.update_quantity(quantity)
                    progress.total_quantities_updated += 1
                else:
                    # Add new
                    self.database.add_quantity(quantity)

            logger.info(
                f"Auto-resolved {len(resolved_quantities)} conflicts, {len(unresolved_conflicts)} remain"
            )

        # Add non-conflicting quantities
        non_conflicting = [
            q
            for q in result.quantities_found
            if q.name not in [c.quantity_name for c in conflicts]
        ]

        for quantity in non_conflicting:
            self.database.add_quantity(quantity)

        # Save updated database
        self.physics_storage.save_database(self.database)

    def complete_extraction(self, session_id: str) -> Dict[str, Any]:
        """
        Complete extraction session and release resources.

        Args:
            session_id: Session ID to complete

        Returns:
            Final extraction summary
        """
        progress = self.progress_tracker.load_progress()
        if not progress or progress.session_id != session_id:
            raise ValueError(f"Invalid session ID: {session_id}")

        # Update database extraction history
        self.database.extraction_sessions.append(session_id)
        self.physics_storage.save_database(self.database)

        # Release lock
        self.lock_manager.release_lock(session_id)

        # Generate summary
        summary = {
            "session_id": session_id,
            "completed_at": datetime.utcnow().isoformat(),
            "total_ids_processed": progress.completed_ids + progress.failed_ids,
            "successful_ids": progress.completed_ids,
            "failed_ids": progress.failed_ids,
            "total_quantities_found": progress.total_quantities_found,
            "total_quantities_updated": progress.total_quantities_updated,
            "total_conflicts": progress.total_conflicts,
            "total_paths_processed": progress.total_paths_processed,
            "total_processing_time": progress.total_processing_time,
            "database_stats": self.database.get_stats(),
        }

        logger.info(f"Completed extraction session {session_id}")
        return summary

    def get_conflicts_for_review(self) -> List[Dict[str, Any]]:
        """Get conflicts that require human review."""
        return self.conflict_manager.get_unresolved_conflicts()

    def resolve_conflict_manually(
        self,
        conflict_id: str,
        resolution: ConflictResolutionStrategy,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Manually resolve a specific conflict.

        Args:
            conflict_id: ID of the conflict to resolve
            resolution: How to resolve it
            notes: Optional notes about the resolution

        Returns:
            True if resolved successfully
        """
        # This would need to load the specific conflict and resolve it
        # For now, this is a placeholder for the manual resolution workflow
        logger.info(
            f"Manual conflict resolution requested for {conflict_id}: {resolution.value}"
        )
        return True

    def export_database(self, output_file: Path, format: str = "json") -> bool:
        """
        Export physics database to external format.

        Args:
            output_file: Where to save the export
            format: Export format ("json", "yaml", "csv")

        Returns:
            True if export successful
        """
        try:
            if format.lower() == "json":
                # Export as structured JSON
                export_data = {
                    "metadata": {
                        "version": self.database.version,
                        "exported_at": datetime.utcnow().isoformat(),
                        "total_quantities": self.database.total_quantities,
                        "verified_quantities": self.database.verified_quantities,
                    },
                    "quantities": {},
                }

                for qid, quantity in self.database.quantities.items():
                    export_data["quantities"][qid] = {
                        "name": quantity.name,
                        "symbol": quantity.symbol,
                        "description": quantity.description,
                        "unit": quantity.unit,
                        "dimensions": quantity.dimensions,
                        "typical_range": quantity.typical_range,
                        "physics_context": quantity.physics_context,
                        "imas_paths": quantity.imas_paths,
                        "ids_sources": list(quantity.ids_sources),
                        "extraction_confidence": quantity.extraction_confidence,
                        "human_verified": quantity.human_verified,
                    }

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2)

                logger.info(f"Exported database to {output_file}")
                return True
            else:
                logger.error(f"Unsupported export format: {format}")
                return False

        except Exception as e:
            logger.error(f"Failed to export database: {e}")
            return False

    def get_extraction_status_with_catalog(self) -> Dict[str, Any]:
        """
        Get extraction status with catalog-based information.

        Returns:
            Comprehensive status including catalog metadata
        """
        # Get basic status
        basic_status = self.get_extraction_status()

        if not self.catalog_batch_processor or not self.catalog_progress_tracker:
            return basic_status

        # Add catalog-specific information
        catalog_stats = self.catalog_progress_tracker.get_catalog_stats()
        physics_domains = self.catalog_batch_processor.get_physics_domain_summary()
        ids_with_metadata = self.catalog_batch_processor.get_ids_with_metadata()

        # Load catalog progress if available
        catalog_progress = self.catalog_progress_tracker.load_progress()
        detailed_progress = (
            catalog_progress.get_detailed_status() if catalog_progress else None
        )

        return {
            **basic_status,
            "catalog_info": catalog_stats,
            "physics_domains": physics_domains,
            "available_ids_metadata": {
                "count": len(ids_with_metadata),
                "sample": dict(list(ids_with_metadata.items())[:5]),  # Show first 5
            },
            "catalog_progress": detailed_progress,
        }

    def start_catalog_extraction(
        self,
        ids_list: Optional[List[str]] = None,
        paths_per_ids: int = 10,
        auto_resolve_conflicts: bool = True,
        physics_domain_filter: Optional[str] = None,
    ) -> str:
        """
        Start extraction with catalog-based progress tracking.

        Args:
            ids_list: List of IDS to process (None for all available)
            paths_per_ids: Number of paths to process per IDS
            auto_resolve_conflicts: Whether to auto-resolve simple conflicts
            physics_domain_filter: Optional filter by physics domain

        Returns:
            Session ID for tracking progress
        """
        if not self.catalog_batch_processor or not self.catalog_progress_tracker:
            # Fall back to basic extraction
            return self.start_extraction(
                ids_list, paths_per_ids, auto_resolve_conflicts
            )

        # Generate session ID
        session_id = f"catalog_extraction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Acquire lock
        if not self.lock_manager.acquire_lock(session_id):
            raise RuntimeError("Extraction is already in progress")

        try:
            # Get IDS list with filtering
            if ids_list is None:
                ids_with_metadata = self.catalog_batch_processor.get_ids_with_metadata()

                if physics_domain_filter:
                    ids_list = [
                        ids_name
                        for ids_name, metadata in ids_with_metadata.items()
                        if metadata.get("physics_domain") == physics_domain_filter
                    ]
                else:
                    ids_list = list(ids_with_metadata.keys())

                # Remove already processed IDS
                basic_progress = self.progress_tracker.load_progress()
                if basic_progress:
                    remaining_ids = self.progress_tracker.get_remaining_ids(ids_list)
                    ids_list = remaining_ids

            # Create catalog progress tracker
            catalog_progress = self.catalog_progress_tracker.create_enhanced_progress(
                session_id=session_id,
                ids_list=ids_list,
                paths_per_ids=paths_per_ids,
                confidence_threshold=self.extractor.confidence_threshold,
            )

            # Mark all IDS as pending
            for ids_name in ids_list:
                catalog_progress.update_ids_progress(ids_name, ExtractionStatus.PENDING)

            # Save initial progress
            self.catalog_progress_tracker.save_progress(catalog_progress)

            # Generate processing estimates
            estimates = self.catalog_batch_processor.estimate_processing_time(
                ids_list, paths_per_ids
            )

            logger.info(
                f"Started catalog extraction session {session_id} for {len(ids_list)} IDS. "
                f"Estimated time: {estimates['estimated_time_minutes']:.1f} minutes"
            )

            return session_id

        except Exception as e:
            self.lock_manager.release_lock(session_id)
            raise e

    def process_next_ids_with_catalog(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Process the next IDS with catalog tracking.

        Args:
            session_id: Session ID

        Returns:
            Result information with catalog metadata, or None if complete
        """
        if not self.catalog_batch_processor or not self.catalog_progress_tracker:
            # Fall back to basic processing
            result = self.process_next_ids(session_id)
            return (
                {
                    "ids_name": result.ids_name,
                    "extraction_result": {
                        "status": result.status.value,
                        "quantities_found": len(result.quantities_found),
                        "paths_processed": len(result.paths_processed),
                        "processing_time": result.processing_time,
                        "warnings": result.warnings,
                    },
                }
                if result
                else None
            )

        # Load catalog progress
        catalog_progress = self.catalog_progress_tracker.load_progress()
        if not catalog_progress or catalog_progress.session_id != session_id:
            raise ValueError(f"Invalid session ID: {session_id}")

        # Find next IDS to process
        next_ids = None
        for ids_name, status in catalog_progress.ids_status.items():
            if status == ExtractionStatus.PENDING:
                next_ids = ids_name
                break

        if not next_ids:
            logger.info("No more IDS to process")
            return None

        # Get catalog info for this IDS
        ids_info = catalog_progress.ids_path_info.get(next_ids, {})
        processing_paths = ids_info.get("processing_paths", 10)

        # Update status to in progress
        catalog_progress.update_ids_progress(next_ids, ExtractionStatus.IN_PROGRESS)
        self.catalog_progress_tracker.save_progress(catalog_progress)

        try:
            # Process the IDS with catalog information
            result = self.catalog_batch_processor.process_ids_with_catalog_info(
                next_ids, processing_paths
            )

            # Update catalog progress with results
            if result.status == ExtractionStatus.COMPLETED:
                catalog_progress.update_ids_progress(
                    next_ids,
                    ExtractionStatus.COMPLETED,
                    1.0,
                    len(result.paths_processed),
                )
                catalog_progress.total_quantities_found += len(result.quantities_found)
                catalog_progress.total_processing_time += result.processing_time

                # Integrate results into database
                self._integrate_extraction_results_with_catalog(
                    result, catalog_progress
                )

            else:
                catalog_progress.update_ids_progress(
                    next_ids, ExtractionStatus.FAILED, 0.0
                )

            # Save updated progress
            self.catalog_progress_tracker.save_progress(catalog_progress)

            # Create result with catalog information
            catalog_result = {
                "ids_name": next_ids,
                "extraction_result": {
                    "status": result.status.value,
                    "quantities_found": len(result.quantities_found),
                    "paths_processed": len(result.paths_processed),
                    "processing_time": result.processing_time,
                    "warnings": result.warnings,
                },
                "catalog_info": getattr(result, "catalog_metadata", {}),
                "progress_info": {
                    "overall_completion": catalog_progress.completion_percentage,
                    "path_completion": catalog_progress.path_completion_percentage,
                    "remaining_ids": catalog_progress.total_ids
                    - catalog_progress.completed_ids
                    - catalog_progress.failed_ids,
                },
            }

            logger.info(
                f"Catalog processing completed for {next_ids}: {result.status.value}"
            )
            return catalog_result

        except Exception as e:
            logger.error(f"Catalog processing failed for {next_ids}: {e}")
            catalog_progress.update_ids_progress(next_ids, ExtractionStatus.FAILED, 0.0)
            self.catalog_progress_tracker.save_progress(catalog_progress)
            raise e

    def _integrate_extraction_results_with_catalog(self, result, catalog_progress):
        """
        Integrate extraction results with catalog conflict tracking.

        Args:
            result: ExtractionResult to integrate
            catalog_progress: Current catalog progress tracker
        """
        if not result.quantities_found:
            return

        # Get existing quantities for conflict detection
        existing_quantities = list(self.database.quantities.values())

        # Detect conflicts
        conflicts = self.conflict_manager.detect_conflicts(
            existing_quantities, result.quantities_found
        )

        if conflicts:
            logger.info(f"Detected {len(conflicts)} conflicts for {result.ids_name}")
            catalog_progress.total_conflicts += len(conflicts)

            # Auto-resolve simple conflicts
            resolved_quantities, unresolved_conflicts = (
                self.conflict_manager.auto_resolve_conflicts(conflicts)
            )

            # Add resolved quantities
            for quantity in resolved_quantities:
                if quantity.name in self.database.quantities_by_name:
                    # Update existing
                    self.database.update_quantity(quantity)
                    catalog_progress.total_quantities_updated += 1
                else:
                    # Add new
                    self.database.add_quantity(quantity)

            logger.info(
                f"Auto-resolved {len(resolved_quantities)} conflicts, {len(unresolved_conflicts)} remain"
            )

        # Add non-conflicting quantities
        non_conflicting = [
            q
            for q in result.quantities_found
            if q.name not in [c.quantity_name for c in conflicts]
        ]

        for quantity in non_conflicting:
            self.database.add_quantity(quantity)

        # Save updated database
        self.physics_storage.save_database(self.database)

    def get_domain_progress(self) -> Dict[str, Any]:
        """Get progress breakdown by physics domain."""
        if not self.catalog_progress_tracker:
            return {}

        enhanced_progress = self.catalog_progress_tracker.load_progress()
        if not enhanced_progress:
            return {}

        return enhanced_progress.domain_stats

    def get_ids_details(self) -> Dict[str, Any]:
        """Get detailed per-IDS information."""
        if not self.catalog_progress_tracker:
            return {}

        catalog_progress = self.catalog_progress_tracker.load_progress()
        if not catalog_progress:
            return {}

        return catalog_progress.get_ids_details()

    def export_catalog_report(self, output_file: Path) -> bool:
        """
        Export comprehensive extraction report.

        Args:
            output_file: Where to save the report

        Returns:
            True if successful
        """
        try:
            catalog_progress = None
            catalog_stats = {}

            if self.catalog_progress_tracker:
                catalog_progress = self.catalog_progress_tracker.load_progress()
                catalog_stats = self.catalog_progress_tracker.get_catalog_stats()

            report = {
                "report_metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "report_type": "catalog_extraction_report",
                },
                "catalog_information": catalog_stats,
                "extraction_progress": catalog_progress.get_detailed_status()
                if catalog_progress
                else None,
                "ids_details": self.get_ids_details(),
                "domain_progress": self.get_domain_progress(),
                "database_statistics": self.database.get_stats(),
                "physics_domains_summary": self.catalog_batch_processor.get_physics_domain_summary()
                if self.catalog_batch_processor
                else {},
            }

            import json

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Catalog extraction report exported to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to export catalog report: {e}")
            return False
