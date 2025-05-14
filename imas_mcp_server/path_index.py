"""
PathIndex class refactored to use Whoosh for text indexing and search
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# Third-party imports


logger = logging.getLogger(__name__)


@dataclass
class PathIndex:
    """Optimized index of IDS paths with Whoosh for text indexing and search"""

    version: str  # IMAS DD version
    index_dir: Path  # Whoosh index directory

    # Internal Whoosh index
    _index: Any = field(default=None, repr=False)  # Whoosh index
    schema: Any = field(default=None, repr=False)  # Whoosh schema
    _writer: Any = field(default=None, repr=False)  # Whoosh writer for batch operations
    _writer_lock: bool = field(
        default=False, repr=False
    )  # Lock to prevent multiple writer instances





    @contextmanager
    def batch_add(self, optimize=False):
        """Context manager for batch addition of paths to improve performance.

        Example usage:
            with path_index.batch_add(optimize=True) as writer:
                for path, doc in paths_and_docs:
                    writer.add_document(path=path, ...)

        Args:
            optimize: If True, optimize the index after committing (slower but more efficient searches)

        Yields:
            Whoosh writer object for the current batch operation
        """
        if self._index is None:
            logger.warning("Whoosh index not initialized. Call _setup_index() first.")
            yield None
            return

        if self.writer is not None:
            # Writer already exists, likely nested context
            logger.warning(
                "Writer already exists. Nested batch operations not supported."
            )
            yield self.writer
            return

        # Create a new writer
        new_writer = self._index.writer(
            procs=4,  # Use multiple processors for better performance
            multisegment=True,  # Better for incremental indexing
            limitmb=256,  # Increased memory limit for better performance
        )
        self.writer = new_writer

        try:
            yield self.writer
        except Exception as e:
            # On exception, cancel the changes
            logger.error(f"Error during batch operation: {e}")
            if self.writer is not None:
                self.writer.cancel()
            raise
        finally:
            # Always clean up
            if self.writer is not None:
                try:
                    self.writer.commit(optimize=optimize)
                except Exception as e:
                    logger.error(f"Error committing changes: {e}")
                    self.writer.cancel()
                    raise
                finally:
                    self.writer = None

    def add_path(self, path: str, documentation: str = "") -> None:
        """Add a path to the index

        Args:
            path: The path to add
            documentation: The documentation for the path
        """
        # Check if Whoosh index is available
        if self._index is None:
            logger.warning("Whoosh index not initialized. Call _setup_index() first.")
            return

        # Extract root IDS
        segments = path.split("/")
        ids = segments[0]
        segments_text = " ".join(segments)

        # If a writer is already active, use it
        if self.writer is not None:
            self.writer.delete_by_term("path", path)
            self.writer.add_document(
                path=path,
                segment=segments_text,
                content=documentation,
                ids=ids,
            )
            # Don't commit - it's being handled by the active batch operation
        else:
            # Create a new writer, add the document, and commit
            with self.batch_add() as writer:
                if writer is not None:
                    writer.delete_by_term("path", path)
                    writer.add_document(
                        path=path,
                        segment=segments_text,
                        content=documentation,
                        ids=ids,
                    )

    def batch_add_paths(
        self, paths_and_docs: Dict[str, str], optimize: bool = False
    ) -> None:
        """Add multiple paths to the index in a single batch operation for better performance.

        Args:
            paths_and_docs: Dictionary with paths as keys and documentation as values
            optimize: Whether to optimize the index after committing
        """
        if not paths_and_docs:
            return

        with self.batch_add(optimize=optimize) as writer:
            if writer is None:
                return

            for path, documentation in paths_and_docs.items():
                # Extract root IDS
                segments = path.split("/")
                ids = segments[0]
                segments_text = " ".join(segments)

                # Add document to the index
                writer.delete_by_term("path", path)
                writer.add_document(
                    path=path,
                    segment=segments_text,
                    content=documentation,
                    ids=ids,
                )



    def search_by_semantic_similarity(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for paths using semantic similarity with the query.

        Note: This is a placeholder for future implementation.
        Currently falls back to keyword search.

        Args:
            query: Natural language query
            limit: Maximum number of results to return

        Returns:
            List of dictionaries with path, relevance score, and documentation
        """
        return self.search_by_keywords(query, limit)

    def count_paths(self) -> int:
        """Return the number of paths in the index."""
        if self._index is None:
            return 0

        with self._index.searcher() as searcher:
            return searcher.doc_count()

    def get_ids_set(self) -> set:
        """Extract the set of IDS names from the index."""
        if self._index is None:
            return set()

        ids_set = set()
        with self._index.searcher() as searcher:
            for doc in searcher.all_stored_fields():
                if "ids" in doc:
                    ids_set.add(doc["ids"])
        return ids_set

    def get_document(self, path: str) -> str:
        """Return the documentation for a path."""
        if self._index is None:
            return ""

        with self._index.searcher() as searcher:
            doc = searcher.document(path=path)
            if doc and "content" in doc:
                return doc["content"]
        return ""
