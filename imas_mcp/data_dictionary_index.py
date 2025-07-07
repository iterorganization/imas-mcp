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
import xml.etree.ElementTree as ET

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

from imas_mcp.core.xml_utils import DocumentationBuilder
from imas_mcp.dd_accessor import create_dd_accessor, DataDictionaryAccessor

IndexPrefixT = Literal["lexicographic", "semantic"]

# Performance tuning constants
DEFAULT_BATCH_SIZE = 500
PROGRESS_LOG_INTERVAL = 50

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class DataDictionaryIndex(abc.ABC):
    """Abstract base class for IMAS Data Dictionary methods and attributes."""

    ids_set: Optional[Set[str]] = None  # Set of IDS names to index
    dirname: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
        / "resources"
        / "index_data"
    )
    indexname: Optional[str] = field(default=None)
    _dd_accessor: Optional[DataDictionaryAccessor] = field(
        default=None, init=False, repr=False
    )

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
    def _xml_root(self) -> ET.Element:
        """Cache XML root element for repeated access."""
        root = self._dd_etree.getroot()
        if root is None:
            raise ValueError("Root element not found in XML tree after parsing.")
        return root

    @functools.cached_property
    def _ids_elements(self) -> List[ET.Element]:
        """Cache IDS elements to avoid repeated XPath queries."""
        return self._xml_root.findall(".//IDS[@name]")

    @functools.cached_property
    def dd_version(self) -> Version:
        """Return the IMAS DD version."""
        return self.dd_accessor.get_version()

    @functools.cached_property
    def _dd_etree(self) -> ET.ElementTree:
        """Return the IMAS DD XML element tree."""
        return self.dd_accessor.get_xml_tree()

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
        """Return a set of IDS names to process.
        If self.ids_set is provided, it's used. Otherwise, all IDS names from the DD are used.
        """
        if self.ids_set is not None:
            return self.ids_set

        logger.info(
            "No specific ids_set provided, using all IDS names from Data Dictionary."
        )
        all_ids_names: Set[str] = set()  # Explicit type
        # Ensure dd_etree is accessed correctly
        for elem in self._dd_etree.findall(
            ".//IDS[@name]"
        ):  # all IDS elements with a 'name' attribute
            name = elem.get("name")
            if name:  # Ensure name is not None or empty
                all_ids_names.add(name)
        if not all_ids_names:
            logger.warning("No IDS names found in the Data Dictionary XML.")
        return all_ids_names

    def _build_hierarchical_documentation(
        self, documentation_parts: Dict[str, str]
    ) -> str:
        """Build hierarchical documentation string from path-based documentation parts.

        Delegates to the shared DocumentationBuilder utility for consistent
        hierarchical documentation formatting across all components.

        Args:
            documentation_parts: Dictionary where keys are hierarchical paths
                and values are the documentation strings for each node.

        Returns:
            Formatted markdown string with hierarchical context.
        """
        return DocumentationBuilder.build_hierarchical_documentation(
            documentation_parts
        )

    def _build_element_entry(
        self,
        elem: ET.Element,
        ids_node: ET.Element,
        ids_name: str,
        parent_map: Dict[ET.Element, ET.Element],
    ) -> Optional[Dict[str, Any]]:
        """Build a single element entry efficiently."""
        path_parts = []
        units = elem.get("units", "")

        # Collect documentation using shared utility
        documentation_parts = DocumentationBuilder.collect_documentation_hierarchy(
            elem, ids_node, ids_name, parent_map
        )

        # Walk up tree for path building and units inheritance
        walker = elem
        while walker is not None and walker != ids_node:
            walker_name = walker.get("name")
            if walker_name:
                path_parts.insert(0, walker_name)

            # Handle units inheritance
            parent_walker = parent_map.get(walker)
            if units == "as_parent" and parent_walker is not None:
                parent_units = parent_walker.get("units")
                if parent_units:
                    units = parent_units

            walker = parent_walker

        if not path_parts:
            return None

        full_path = f"{ids_name}/{'/'.join(path_parts)}"
        combined_documentation = self._build_hierarchical_documentation(
            documentation_parts
        )

        return {
            "path": full_path,
            "documentation": combined_documentation or elem.get("documentation", ""),
            "units": units or "none",
            "ids_name": ids_name,
        }

    @functools.cached_property
    def ids_names(self) -> List[str]:
        """Return a list of IDS names relevant to this index.
        Extracts from DD based on current configuration.
        """
        logger.info(
            "Extracting IDS names from DD for current configuration."
        )  # Use _get_ids_set() which respects self.ids_set if provided, or gets all from DD
        ids_set = self._get_ids_set()
        relevant_names = sorted(list(ids_set))  # Sort for consistency

        return relevant_names

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

                    # Log progress every 5% or every 15 seconds for better visibility
                    if total:
                        percentage = (self.completed / total) * 100
                        time_since_log = current_time - self.last_log_time

                        if (percentage - self.last_percentage >= 5) or (
                            time_since_log >= 15
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
                        # Log every 50 items if total is unknown
                        if self.completed % 50 == 0:
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
        """
        Calculate and cache the total number of elements to process.        Returns:
            int: Total count of IDS root elements plus all their named descendants
        """
        ids_to_process = self._get_ids_set()
        root_node = self._xml_root

        # Pre-filter IDS nodes efficiently
        ids_nodes = [
            node
            for node in root_node.findall(".//IDS[@name]")
            if node.get("name") in ids_to_process
        ]

        total_elements = 0
        for ids_node in ids_nodes:
            total_elements += 1  # Count IDS root
            total_elements += len(ids_node.findall(".//*[@name]"))  # Count descendants

        logger.info(f"Total elements to process: {total_elements}")
        return total_elements

    def _get_document(self, progress_tracker=None) -> Iterable[Dict[str, Any]]:
        """
        Get document entries from IMAS Data Dictionary XML.

        Args:
            progress_tracker: Optional tuple of (progress, task) for external tracking        Yields:
            Dict[str, Any]: Document entries
        """
        logger.info(
            f"Starting optimized extraction for DD version {self.dd_version.public}"
        )

        ids_to_process = self._get_ids_set()
        logger.info(f"Processing {len(ids_to_process)} IDS: {sorted(ids_to_process)}")

        root_node = self._xml_root

        # Cache parent map - major performance improvement
        parent_map = {
            c: p for p in root_node.iter() for c in p
        }  # Pre-filter IDS nodes efficiently
        ids_nodes = [
            node
            for node in root_node.findall(".//IDS[@name]")
            if node.get("name") in ids_to_process
        ]

        if not ids_nodes:
            logger.warning(f"No IDS found for ids_set: {ids_to_process}")
            return  # Count total elements for progress tracking
        total_elements = (
            self._total_elements
        )  # Use provided progress tracker or create new one

        if progress_tracker:
            progress, task = progress_tracker
            # Don't update total when using shared progress tracker
            context_manager = None
        else:
            context_manager = self._progress_tracker(
                "Extracting Data Dictionary documents", total=total_elements
            )
            progress, task = context_manager.__enter__()

        try:
            document_count = 0
            ids_count = len(ids_nodes)
            current_ids_index = 0

            for ids_node in ids_nodes:
                ids_name = ids_node.get("name")
                if not ids_name:
                    continue

                current_ids_index += 1

                # Log per-IDS progress for non-interactive environments
                if not self._is_interactive_environment():
                    logger.info(
                        f"Processing IDS {current_ids_index}/{ids_count}: {ids_name}"
                    )  # Yield IDS root entry
                yield {
                    "path": ids_name,
                    "documentation": ids_node.get("documentation", ""),
                    "units": ids_node.get("units", "none"),
                    "ids_name": ids_name,
                }
                document_count += 1
                if progress:
                    progress.advance(task)  # type: ignore[arg-type]                # Count descendants for per-IDS logging
                descendants = ids_node.findall(".//*[@name]")
                if not self._is_interactive_environment() and descendants:
                    logger.info(
                        f"  Processing {len(descendants)} elements in {ids_name}"
                    )

                # Process all named descendants
                for elem in descendants:
                    entry = self._build_element_entry(
                        elem, ids_node, ids_name, parent_map
                    )
                    if entry:
                        yield entry
                        document_count += 1
                        if progress:
                            progress.advance(task)  # type: ignore[arg-type]                        # Update description periodically for better time estimates (Rich only)
                        if document_count % PROGRESS_LOG_INTERVAL == 0:
                            if progress and hasattr(progress, "update"):
                                progress.update(
                                    task, description=f"Processing {ids_name}"
                                )  # type: ignore[arg-type]

                # Log completion for each IDS in non-interactive environments
                if not self._is_interactive_environment():
                    logger.info(
                        f"  Completed {ids_name}: {len(descendants) + 1} elements processed"
                    )

        finally:
            # Only exit context manager if we created it
            if context_manager:
                context_manager.__exit__(None, None, None)

        logger.info(f"Finished extracting {document_count} document entries from DD.")

    def _get_document_batch(
        self, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> Iterable[List[Dict[str, Any]]]:
        """
        Get document entries from Data Dictionary XML in batches.

        Args:
            batch_size: Number of documents per batch

        Yields:
            List[Dict[str, Any]]: Batches of document entries
        """
        logger.info(f"Generating document batches from index: {self.indexname}")

        # Use cached total elements calculation for accurate progress tracking
        total_elements = self._total_elements

        documents_batch = []
        processed_paths = set()
        total_documents = 0
        batch_count = 0

        # Single progress tracker shared with _get_document, with pre-calculated total
        with self._progress_tracker(
            "Processing IDS attributes", total=total_elements
        ) as (
            progress,
            task,
        ):
            try:  # Use shared progress tracker for consistent updates
                for entry_dict in self._get_document(progress_tracker=(progress, task)):
                    path = entry_dict.get("path")
                    if not path or path in processed_paths:
                        continue

                    if all(
                        key in entry_dict
                        for key in ["path", "documentation", "ids_name"]
                    ):
                        documents_batch.append(entry_dict)
                        processed_paths.add(path)
                        total_documents += 1

                        if len(documents_batch) >= batch_size:
                            batch_count += 1
                            yield list(documents_batch)
                            documents_batch.clear()

                if documents_batch:
                    batch_count += 1
                    yield list(documents_batch)

            except Exception as e:
                logger.error(f"Error during document batch generation: {e}")
                raise

        logger.info(
            f"Completed document batch generation: {batch_count} batches, {total_documents} total documents"
        )

    @abc.abstractmethod
    def build_index(self) -> None:
        """Builds the index from the Data Dictionary IDSDef XML file."""
        raise NotImplementedError

    def _is_interactive_environment(self) -> bool:
        """
        Detect if we're running in an interactive environment where rich progress is viewable.

        Returns:
            bool: True if rich progress should be displayed, False for fallback logging
        """
        # Check if we're in a Docker container
        if os.path.exists("/.dockerenv"):
            return False

        # Check if stdout is a TTY (terminal)
        if not sys.stdout.isatty():
            return False

        # Check for CI/CD environment variables
        ci_vars = ["CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "GITLAB_CI"]
        if any(os.getenv(var) for var in ci_vars):
            return False

        # Check if TERM is set to a non-interactive value
        term = os.getenv("TERM", "")
        if term in ["dumb", ""]:
            return False

        return True

    @property
    def dd_accessor(self) -> DataDictionaryAccessor:
        """Return the data dictionary accessor."""
        if self._dd_accessor is None:
            raise RuntimeError("Data dictionary accessor not initialized")
        return self._dd_accessor
