import abc
from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import hashlib
import logging
import os
from pathlib import Path
import sys
import time
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
)

from packaging.version import Version
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from imas_mcp.schema_accessor import SchemaAccessor
from imas_mcp.dd_accessor import create_dd_accessor, DataDictionaryAccessor
from imas_mcp.core.progress_monitor import create_progress_monitor

IndexPrefixT = Literal["lexicographic", "semantic"]

# Performance tuning constants
DEFAULT_BATCH_SIZE = (
    1000  # Balanced batch size for good performance with smooth progress
)
PROGRESS_LOG_INTERVAL = 50

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class DataDictionaryIndex(abc.ABC):
    """Abstract base class for IMAS Data Dictionary methods and attributes."""

    dirname: Path = field(
        default_factory=lambda: Path(__file__).parent / "resources" / "index_data"
    )
    ids_set: Optional[Set[str]] = None
    use_rich: Optional[bool] = None  # Auto-detect if None
    _dd_accessor: Optional[DataDictionaryAccessor] = field(default=None, init=False)
    indexname: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Common initialization for all handlers."""
        logger.info(f"Initializing DataDictionaryIndex with ids_set: {self.ids_set}")

        # Ensure the resources directory exists
        self.dirname.mkdir(parents=True, exist_ok=True)

        # Create the DD accessor first
        self._dd_accessor = create_dd_accessor(
            metadata_dir=self.dirname,
            index_name=None,  # We'll set this after we can determine the name
            index_prefix=self.index_prefix,
        )

        # Create JSON data accessor for new JSON-based processing
        self._json_accessor = SchemaAccessor()

        # Now we can get the index name
        self.indexname = self._get_index_name()

        logger.info(
            f"Initialized Data Dictionary index: {self.indexname} in {self.dirname}"
        )

    @property
    @abc.abstractmethod
    def index_prefix(self) -> IndexPrefixT:
        """Return the index name prefix."""
        pass

    @contextmanager
    def _performance_timer(self, operation_name: str):
        """Context manager for timing operations with logging."""
        start_time = time.time()
        logger.info(f"Starting {operation_name}")
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            logger.info(f"Completed {operation_name} in {elapsed:.2f}s")

    @functools.cached_property
    def dd_version(self) -> Version:
        """Return the IMAS DD version."""
        return self.dd_accessor.get_version()

    def _get_index_name(self) -> str:
        """Return the full index name based on prefix, IMAS DD version, and ids_set."""
        # Ensure dd_version is available
        dd_version = self.dd_version.public  # Access dd_version property
        indexname = f"{self.index_prefix}_{dd_version}"
        # Ensure ids_set is treated consistently (e.g., handle None or empty set)
        if self.ids_set is not None and len(self.ids_set) > 0:
            ids_str = ",".join(
                sorted(list(self.ids_set))
            )  # Convert set to sorted list for consistent hash
            hash_suffix = hashlib.md5(ids_str.encode("utf-8")).hexdigest()[
                :8
            ]  # Specify encoding
            return f"{indexname}-{hash_suffix}"
        return indexname

    def _get_ids_set(self) -> Set[str]:
        """Return a set of IDS names to process."""
        if self.ids_set is not None:
            return self.ids_set

        # Get available IDS from JSON data instead of XML
        available_ids = self._json_accessor.get_available_ids()
        all_ids_names = set(available_ids)

        if not all_ids_names:
            logger.warning("No IDS names found in the JSON data.")
        return all_ids_names

    @functools.cached_property
    def ids_names(self) -> List[str]:
        """Return a list of IDS names relevant to this index."""
        return sorted(list(self._get_ids_set()))

    @contextmanager
    def _progress_tracker(self, description: str, total: Optional[int] = None):
        """Context manager for progress tracking with fallback for non-interactive environments."""
        if self._is_interactive_environment():
            # Use rich progress for interactive environments
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=20),
                MofNCompleteColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(description, total=total)
                yield progress, task
        else:
            # Use fallback logging for non-interactive environments (Docker, CI/CD, etc.)
            logger.info(f"Starting: {description} (total: {total or 'unknown'})")
            start_time = (
                time.time()
            )  # Simple progress tracker for non-interactive environments

            class SimpleProgressTracker:
                def __init__(self):
                    self.completed = 0
                    self.last_log_time = start_time
                    self.last_percentage = 0

                def advance(self, task_id=None):
                    """Advance progress counter. Task parameter accepted but ignored for compatibility."""
                    self.completed += 1
                    current_time = time.time()

                    # Log progress every 10% or every 30 seconds for cleaner output
                    if total:
                        percentage = (self.completed / total) * 100
                        time_since_log = current_time - self.last_log_time

                        if (percentage - self.last_percentage >= 10) or (
                            time_since_log >= 30
                        ):
                            elapsed = current_time - start_time
                            remaining = 0
                            if percentage > 0:
                                remaining = (elapsed / percentage) * (100 - percentage)
                            logger.info(
                                f"Progress: {self.completed}/{total} "
                                f"({percentage:.1f}%) - {elapsed:.1f}s elapsed, "
                                f"~{remaining:.1f}s remaining"
                            )
                            self.last_log_time = current_time
                            self.last_percentage = percentage
                    else:
                        # Log every 100 items if total is unknown
                        if self.completed % 100 == 0:
                            elapsed = current_time - start_time
                            logger.info(
                                f"Progress: {self.completed} items - {elapsed:.1f}s elapsed"
                            )

                def update(self, task_id=None, **kwargs):
                    """Update progress description. Parameters accepted but ignored for compatibility."""
                    pass

            simple_progress = SimpleProgressTracker()
            simple_task = object()  # Dummy task object for compatibility

            try:
                yield simple_progress, simple_task
            finally:
                elapsed = time.time() - start_time
                logger.info(
                    f"Completed: {description} - {simple_progress.completed} items "
                    f"processed in {elapsed:.1f}s"
                )

    @functools.cached_property
    def _total_elements(self) -> int:
        """Calculate and cache the total number of elements to process."""
        ids_to_process = self._get_ids_set()
        total_paths = 0

        for ids_name in ids_to_process:
            try:
                ids_data = self._json_accessor.get_ids_detailed_data(ids_name)
                paths = ids_data.get("paths", {})
                total_paths += len(paths)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Could not get data for IDS '{ids_name}': {e}")
                continue

        logger.debug(f"Total elements to process: {total_paths}")
        return total_paths

    def _get_document(self, progress_tracker=None) -> Iterable[Dict[str, Any]]:
        """Get document entries from JSON data instead of XML."""
        logger.debug(
            f"Starting JSON-based extraction for DD version {self.dd_version.public}"
        )

        ids_to_process = self._get_ids_set()
        logger.debug(f"Processing {len(ids_to_process)} IDS: {sorted(ids_to_process)}")

        # Count total elements for progress tracking
        total_elements = self._total_elements

        # Use provided progress tracker or create new one
        if progress_tracker:
            progress, task = progress_tracker
            context_manager = None
        else:
            context_manager = self._progress_tracker(
                "Extracting Data Dictionary documents", total=total_elements
            )
            progress, task = context_manager.__enter__()

        try:
            document_count = 0
            ids_count = len(ids_to_process)
            current_ids_index = 0

            for ids_name in ids_to_process:
                current_ids_index += 1

                # Log per-IDS progress only for non-interactive environments
                if not self._is_interactive_environment():
                    logger.info(
                        f"Processing IDS {current_ids_index}/{ids_count}: {ids_name}"
                    )

                try:
                    # Get detailed data from JSON instead of XML
                    ids_data = self._json_accessor.get_ids_detailed_data(ids_name)
                    paths = ids_data.get("paths", {})

                    # Process all paths in the IDS
                    for path_name, path_data in paths.items():
                        entry = self._build_json_entry(path_data, ids_name)
                        if entry:
                            yield entry
                            document_count += 1
                            if progress and hasattr(progress, "advance") and task:
                                progress.advance(task)  # type: ignore

                        # Update description periodically for better time estimates (Rich only)
                        if document_count % PROGRESS_LOG_INTERVAL == 0:
                            if progress and hasattr(progress, "update") and task:
                                try:
                                    progress.update(  # type: ignore
                                        task,
                                        description=f"Processing {ids_name}",  # type: ignore
                                    )
                                except TypeError:
                                    # Fallback for incompatible progress tracker
                                    pass

                    # Log completion for each IDS only in non-interactive environments
                    if not self._is_interactive_environment():
                        logger.info(
                            f"  Completed {ids_name}: {len(paths)} elements processed"
                        )

                except (FileNotFoundError, ValueError) as e:
                    logger.warning(f"Could not process IDS '{ids_name}': {e}")
                    continue

        finally:
            # Only exit context manager if we created it
            if context_manager:
                context_manager.__exit__(None, None, None)

        logger.debug(
            f"Finished extracting {document_count} document entries from JSON data."
        )

    def _build_json_entry(
        self, path_data: Dict[str, Any], ids_name: str
    ) -> Optional[Dict[str, Any]]:
        """Build a document entry from JSON path data."""
        path = path_data.get("path")
        if not path:
            return None

        # Extract basic fields
        documentation = path_data.get("documentation", "")
        units = path_data.get("units", "none")

        # Process coordinates as text for searching
        coordinates = path_data.get("coordinates", [])
        coordinates_text = " ".join(coordinates) if coordinates else ""

        # Process lifecycle
        lifecycle = path_data.get("lifecycle", "active")

        # Process data type
        data_type = path_data.get("data_type", "")

        # Process physics context
        physics_context = path_data.get("physics_context")
        physics_context_text = ""
        if physics_context:
            domain = physics_context.get("domain", "")
            phenomena = physics_context.get("phenomena", [])
            physics_context_text = f"{domain} {' '.join(phenomena)}"

        # Process related paths as searchable text
        related_paths = path_data.get("related_paths", [])
        related_paths_text = " ".join(related_paths) if related_paths else ""

        # Process usage examples as searchable text
        usage_examples = path_data.get("usage_examples", [])
        usage_examples_text = ""
        if usage_examples:
            example_texts = []
            for example in usage_examples:
                scenario = example.get("scenario", "")
                code = example.get("code", "")
                notes = example.get("notes", "")
                example_texts.append(f"{scenario} {code} {notes}")
            usage_examples_text = " ".join(example_texts)

        # Process validation rules as text
        validation_rules = path_data.get("validation_rules", {})
        validation_text = ""
        if validation_rules:
            parts = []
            if validation_rules.get("units_required"):
                parts.append("units_required")
            if validation_rules.get("min_value"):
                parts.append(f"min_value_{validation_rules['min_value']}")
            if validation_rules.get("max_value"):
                parts.append(f"max_value_{validation_rules['max_value']}")
            validation_text = " ".join(parts)

        # Process relationships as searchable text
        relationships = path_data.get("relationships", {})
        relationships_text = ""
        if relationships:
            all_relations = []
            for rel_type, rel_list in relationships.items():
                if isinstance(rel_list, list):
                    all_relations.extend(rel_list)
            relationships_text = " ".join(all_relations)

        # Additional XML attributes
        introduced_after = path_data.get("introduced_after", "")
        coordinate1 = path_data.get("coordinate1", "")
        coordinate2 = path_data.get("coordinate2", "")
        timebase = path_data.get("timebase", "")
        type_field = path_data.get("type", "")

        return {
            "path": path,
            "documentation": documentation,
            "units": units,
            "ids_name": ids_name,
            # Extended fields from JSON
            "coordinates": coordinates_text,
            "lifecycle": lifecycle,
            "data_type": data_type,
            "physics_context": physics_context_text,
            "related_paths": related_paths_text,
            "usage_examples": usage_examples_text,
            "validation_rules": validation_text,
            "relationships": relationships_text,
            "introduced_after": introduced_after,
            "coordinate1": coordinate1,
            "coordinate2": coordinate2,
            "timebase": timebase,
            "type": type_field,
        }

    def _get_document_batch(
        self, batch_size: int = DEFAULT_BATCH_SIZE, use_rich: bool = True
    ) -> Iterable[List[Dict[str, Any]]]:
        """Get document entries from JSON data in batches."""
        # Get the list of IDS to process for proper progress monitoring
        ids_to_process = list(self._get_ids_set())

        # Pre-calculate total expected batches for smooth progress
        total_elements = self._total_elements
        estimated_batches = max(1, (total_elements + batch_size - 1) // batch_size)

        documents_batch = []
        processed_paths = set()
        total_documents = 0
        batch_count = 0

        # Use create_progress_monitor with estimated batch count for smooth progress
        batch_names = [f"Batch {i + 1}" for i in range(estimated_batches)]
        progress = create_progress_monitor(
            use_rich=use_rich, logger=logger, item_names=batch_names
        )

        try:
            progress.start_processing(batch_names, "Processing document batches")

            current_ids_index = 0

            # Process each IDS individually but track progress by batches
            for ids_name in ids_to_process:
                current_ids_index += 1

                try:
                    # Get detailed data from JSON for this specific IDS
                    ids_data = self._json_accessor.get_ids_detailed_data(ids_name)
                    paths = ids_data.get("paths", {})

                    # Process all paths in this IDS
                    for path_name, path_data in paths.items():
                        entry = self._build_json_entry(path_data, ids_name)
                        if not entry:
                            continue

                        path = entry.get("path")
                        if not path or path in processed_paths:
                            continue

                        if all(
                            key in entry
                            for key in ["path", "documentation", "ids_name"]
                        ):
                            documents_batch.append(entry)
                            processed_paths.add(path)
                            total_documents += 1

                            if len(documents_batch) >= batch_size:
                                batch_count += 1
                                # Update progress description to show current IDS
                                progress.set_current_item(
                                    f"Processing {ids_name} (Batch {batch_count})"
                                )
                                # Advance progress by one batch
                                progress.update_progress(f"Batch {batch_count}")

                                yield list(documents_batch)
                                documents_batch.clear()

                    # No individual IDS progress updates - we track by batches now

                except (FileNotFoundError, ValueError) as e:
                    logger.warning(f"Could not process IDS '{ids_name}': {e}")
                    continue

            # Handle final partial batch
            if documents_batch:
                batch_count += 1
                current_ids_name = ids_to_process[-1] if ids_to_process else "Final"
                progress.set_current_item(
                    f"Processing {current_ids_name} (Final batch)"
                )
                progress.update_progress(f"Batch {batch_count}")
                yield list(documents_batch)

        except Exception as e:
            logger.error(f"Error during document batch generation: {e}")
            raise
        finally:
            progress.finish_processing()

        logger.debug(
            f"Completed document batch generation: {batch_count} batches, {total_documents} total documents"
        )

    @abc.abstractmethod
    def build_index(self) -> None:
        """Builds the index from the Data Dictionary JSON data."""
        raise NotImplementedError

    def _is_interactive_environment(self) -> bool:
        """Check if we're running in an interactive environment where rich progress should be displayed."""
        # Check for non-interactive environments first
        if os.getenv("CI") or os.getenv("GITHUB_ACTIONS") or os.getenv("GITLAB_CI"):
            return False

        # Check if we're in a Docker container
        if os.path.exists("/.dockerenv"):
            return False

        # Check if stdout is a TTY (can display rich progress)
        if not sys.stdout.isatty():
            return False

        # If we have a TTY and we're not in CI/Docker, assume interactive
        return True

    @property
    def dd_accessor(self) -> DataDictionaryAccessor:
        """Return the data dictionary accessor."""
        if self._dd_accessor is None:
            raise RuntimeError("Data dictionary accessor not initialized")
        return self._dd_accessor
