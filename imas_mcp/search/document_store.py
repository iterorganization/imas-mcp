"""
In-memory document store for IMAS data with SQLite3 full-text search.

This module provides fast access to IMAS JSON documents optimized for LLM tools
and sentence transformer search. Uses in-memory storage with SQLite3 for
complex queries and full-text search.
"""

import json
import logging
import sqlite3
import threading
import importlib.resources as resources
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.unit_loader import (
    load_unit_contexts,
    get_unit_category,
    get_unit_physics_domains,
    load_unit_categories,
    load_physics_domain_hints,
    get_unit_dimensionality,
    get_unit_name,
)

logger = logging.getLogger(__name__)


@dataclass
class Units:
    """Manages unit-related information and context."""

    unit_str: str
    name: str = ""
    context: str = ""
    category: Optional[str] = None
    physics_domains: List[str] = field(default_factory=list)
    dimensionality: str = ""

    @classmethod
    def from_unit_string(
        cls,
        unit_str: str,
        unit_contexts: Dict[str, str],
        unit_categories: Dict[str, List[str]],
        physics_domain_hints: Dict[str, List[str]],
    ) -> "Units":
        """Create Units instance from unit string and loaded contexts."""
        name = get_unit_name(unit_str)
        context = unit_contexts.get(unit_str, "")
        category = get_unit_category(unit_str, unit_categories)
        physics_domains = get_unit_physics_domains(
            unit_str, unit_categories, physics_domain_hints
        )
        dimensionality = get_unit_dimensionality(unit_str)

        return cls(
            unit_str=unit_str,
            name=name,
            context=context,
            category=category,
            physics_domains=physics_domains,
            dimensionality=dimensionality,
        )

    def has_meaningful_units(self) -> bool:
        """Check if this represents meaningful physical units."""
        return bool(self.unit_str and self.unit_str not in ("", "none", "1"))

    def get_embedding_components(self) -> List[str]:
        """Get components for embedding text generation."""
        components = []

        if self.has_meaningful_units():
            # Combine short and long form units
            if self.name:
                components.append(f"Units: {self.unit_str} ({self.name})")
            else:
                components.append(f"Units: {self.unit_str}")

            if self.context:
                components.append(f"Physical quantity: {self.context}")

            if self.category:
                components.append(f"Unit category: {self.category}")

            if self.physics_domains:
                components.append(f"Physics domains: {' '.join(self.physics_domains)}")

            if self.dimensionality:
                components.append(f"Dimensionality: {self.dimensionality}")

        return components


@dataclass(frozen=True)
class DocumentMetadata:
    """Immutable metadata for a document path."""

    path_id: str
    ids_name: str
    path_name: str
    units: str = ""
    data_type: str = ""
    coordinates: tuple = field(default_factory=tuple)
    physics_domain: str = ""
    physics_phenomena: tuple = field(default_factory=tuple)


@dataclass
class Document:
    """A complete document with content and metadata."""

    metadata: DocumentMetadata
    documentation: str = ""
    physics_context: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    units: Optional[Units] = None

    def set_units(
        self,
        unit_contexts: Dict[str, str],
        unit_categories: Dict[str, List[str]],
        physics_domain_hints: Dict[str, List[str]],
    ) -> None:
        """Set units information from loaded contexts."""
        if self.metadata.units:
            self.units = Units.from_unit_string(
                self.metadata.units,
                unit_contexts,
                unit_categories,
                physics_domain_hints,
            )

    @property
    def embedding_text(self) -> str:
        """Generate text optimized for sentence transformer embedding."""
        components = [
            f"IDS: {self.metadata.ids_name}",
            f"Path: {self.metadata.path_name}",
        ]

        # Prioritize primary documentation and units for better semantic distinction
        if self.documentation:
            # Extract the primary description (first sentence before hierarchical context)
            primary_doc = self.documentation.split(".")[0].strip()
            if primary_doc:
                components.append(f"Description: {primary_doc}")

        # Add unit-related components
        if self.units:
            components.extend(self.units.get_embedding_components())

        # Add full documentation after primary description and units
        if self.documentation:
            components.append(f"Documentation: {self.documentation}")

        if self.metadata.physics_domain:
            components.append(f"Physics domain: {self.metadata.physics_domain}")

        if self.metadata.physics_phenomena:
            components.append(
                f"Physics phenomena: {' '.join(self.metadata.physics_phenomena)}"
            )

        if self.metadata.coordinates:
            components.append(f"Coordinates: {' '.join(self.metadata.coordinates)}")

        if self.metadata.data_type:
            components.append(f"Data type: {self.metadata.data_type}")

        return " | ".join(components)


@dataclass
class SearchIndex:
    """In-memory search indices for fast lookups."""

    # Primary indices
    by_path_id: Dict[str, Document] = field(default_factory=dict)
    by_ids_name: Dict[str, List[str]] = field(default_factory=dict)

    # Search indices
    by_physics_domain: Dict[str, Set[str]] = field(default_factory=dict)
    by_units: Dict[str, Set[str]] = field(default_factory=dict)
    by_coordinates: Dict[str, Set[str]] = field(default_factory=dict)

    # Full-text indices
    documentation_words: Dict[str, Set[str]] = field(default_factory=dict)
    path_segments: Dict[str, Set[str]] = field(default_factory=dict)

    # Statistics
    total_documents: int = 0
    total_ids: int = 0

    def add_document(self, document: Document) -> None:
        """Add a document to all relevant indices."""
        path_id = document.metadata.path_id
        ids_name = document.metadata.ids_name

        # Primary indices
        self.by_path_id[path_id] = document

        if ids_name not in self.by_ids_name:
            self.by_ids_name[ids_name] = []
        self.by_ids_name[ids_name].append(path_id)

        # Search indices
        if document.metadata.physics_domain:
            domain = document.metadata.physics_domain
            if domain not in self.by_physics_domain:
                self.by_physics_domain[domain] = set()
            self.by_physics_domain[domain].add(path_id)

        if document.metadata.units and document.metadata.units not in ("", "none", "1"):
            units = document.metadata.units
            if units not in self.by_units:
                self.by_units[units] = set()
            self.by_units[units].add(path_id)

        for coord in document.metadata.coordinates:
            if coord not in self.by_coordinates:
                self.by_coordinates[coord] = set()
            self.by_coordinates[coord].add(path_id)

        # Full-text indices
        if document.documentation:
            words = document.documentation.lower().split()
            for word in words:
                if len(word) > 2:  # Skip very short words
                    if word not in self.documentation_words:
                        self.documentation_words[word] = set()
                    self.documentation_words[word].add(path_id)

        # Path segment index
        path_parts = document.metadata.path_name.lower().split("/")
        for part in path_parts:
            if len(part) > 1:
                if part not in self.path_segments:
                    self.path_segments[part] = set()
                self.path_segments[part].add(path_id)

        self.total_documents += 1


@dataclass
class DocumentStore:
    """
    In-memory document store for IMAS data with intelligent SQLite3 caching.

    Optimized for LLM tools and sentence transformer embedding. Loads all
    JSON data into memory for O(1) access with SQLite3 for complex queries.

    Features intelligent cache management:
    - Only rebuilds SQLite index when data changes or explicitly requested
    - Validates cache using file modification times and metadata
    - Provides cache inspection and management methods
    """

    # Configuration
    auto_load: bool = True  # Whether to automatically load all documents on init

    # Internal state
    _index: SearchIndex = field(default_factory=SearchIndex, init=False)
    _data_dir: Path = field(init=False)
    _sqlite_path: Path = field(init=False)
    _loaded: bool = field(default=False, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _unit_contexts: Dict[str, str] = field(default_factory=dict, init=False)
    _unit_categories: Dict[str, List[str]] = field(default_factory=dict, init=False)
    _physics_domain_hints: Dict[str, List[str]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        """Initialize the document store."""
        # Use importlib.resources to locate JSON data
        self._data_dir = self._get_resources_path()

        # Build cache in resources directory
        self._sqlite_path = self._data_dir / ".imas_fts.db"

        # Load unit contexts from YAML file
        self._unit_contexts = load_unit_contexts()
        self._unit_categories = load_unit_categories()
        self._physics_domain_hints = load_physics_domain_hints()
        logger.info(
            f"Loaded {len(self._unit_contexts)} unit context definitions with categories and domain hints"
        )

        if self.auto_load:
            self.load_all_documents()

    def load_ids_set(self, ids_set: set[str]) -> None:
        """Initialize DocumentStore with specific IDS for faster loading."""
        if not self._loaded:
            # Convert set to list for the load method
            self.load_all_documents(ids_filter=list(ids_set))

    def _get_resources_path(self) -> Path:
        """Get the path to the resources directory using importlib.resources."""
        try:
            # Use the new files() API instead of deprecated path()
            resources_dir = resources.files("imas_mcp.resources") / "json_data"
            # Convert Traversable to Path using str conversion
            return Path(str(resources_dir))
        except (ImportError, FileNotFoundError):
            # Fallback to package relative path
            import imas_mcp

            package_path = Path(imas_mcp.__file__).parent
            return package_path / "resources" / "json_data"

    def is_available(self) -> bool:
        """Check if IMAS data is available."""
        return (self._data_dir / "ids_catalog.json").exists()

    def load_all_documents(
        self, force_rebuild_index: bool = False, ids_filter: Optional[List[str]] = None
    ) -> None:
        """Load JSON documents into memory with indexing.

        Args:
            force_rebuild_index: Force rebuild of SQLite FTS index
            ids_filter: Optional list of IDS names to load (for faster testing)
        """
        with self._lock:
            if self._loaded:
                return

            logger.info("Loading IMAS documents into memory...")

            # Load catalog to get available IDS
            available_ids = self._get_available_ids()

            # Filter IDS if specified
            if ids_filter:
                available_ids = [ids for ids in available_ids if ids in ids_filter]
                logger.info(f"Filtering to {len(available_ids)} IDS: {available_ids}")
                # Force rebuild when using IDS filter since cache won't match
                force_rebuild_index = True

            self._index.total_ids = len(available_ids)

            # Load each IDS detailed file
            for ids_name in available_ids:
                self._load_ids_documents(ids_name)

            # Build or validate SQLite FTS index
            if force_rebuild_index or self._should_rebuild_fts_index():
                self._build_sqlite_fts_index()
            else:
                logger.info("Using existing SQLite FTS5 index")

            self._loaded = True
            logger.info(
                f"Loaded {self._index.total_documents} documents from "
                f"{self._index.total_ids} IDS into memory"
            )

    def _get_available_ids(self) -> List[str]:
        """Get list of available IDS names."""
        catalog_path = self._data_dir / "ids_catalog.json"
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {catalog_path}")

        with open(catalog_path, encoding="utf-8") as f:
            catalog = json.load(f)

        return list(catalog.get("ids_catalog", {}).keys())

    def _load_ids_documents(self, ids_name: str) -> None:
        """Load all documents for a specific IDS."""
        detailed_file = self._data_dir / "detailed" / f"{ids_name}.json"
        if not detailed_file.exists():
            logger.warning(f"Missing detailed file for {ids_name}")
            return

        try:
            with open(detailed_file, encoding="utf-8") as f:
                ids_data = json.load(f)

            paths = ids_data.get("paths", {})
            for path_name, path_data in paths.items():
                document = self._create_document(ids_name, path_name, path_data)
                self._index.add_document(document)

        except Exception as e:
            logger.error(f"Failed to load {ids_name}: {e}")

    def _create_document(
        self, ids_name: str, path_name: str, path_data: Dict[str, Any]
    ) -> Document:
        """Create a Document object from raw path data."""
        # Create unique path ID
        path_id = (
            f"{ids_name}/{path_name}"
            if not path_name.startswith(ids_name)
            else path_name
        )

        # Extract physics context
        physics_context = path_data.get("physics_context", {})
        physics_domain = ""
        physics_phenomena = ()

        if isinstance(physics_context, dict):
            physics_domain = physics_context.get("domain", "")
            phenomena = physics_context.get("phenomena", [])
            if isinstance(phenomena, list):
                physics_phenomena = tuple(phenomena)

        # Extract coordinates
        coordinates = path_data.get("coordinates", [])
        if isinstance(coordinates, list):
            coordinates = tuple(coordinates)
        else:
            coordinates = ()

        # Create metadata
        metadata = DocumentMetadata(
            path_id=path_id,
            ids_name=ids_name,
            path_name=path_name,
            units=path_data.get("units", ""),
            data_type=path_data.get("data_type", ""),
            coordinates=coordinates,
            physics_domain=physics_domain,
            physics_phenomena=physics_phenomena,
        )

        # Create document
        document = Document(
            metadata=metadata,
            documentation=path_data.get("documentation", ""),
            physics_context=physics_context,
            relationships=path_data.get("relationships", {}),
            raw_data=path_data,
        )

        # Set unit contexts for this document
        document.set_units(
            self._unit_contexts, self._unit_categories, self._physics_domain_hints
        )

        return document

    # Fast access methods for LLM tools
    def get_document(self, path_id: str) -> Optional[Document]:
        """Get document by path ID with O(1) lookup."""
        return self._index.by_path_id.get(path_id)

    def get_documents_by_ids(self, ids_name: str) -> List[Document]:
        """Get all documents for an IDS."""
        path_ids = self._index.by_ids_name.get(ids_name, [])
        return [self._index.by_path_id[pid] for pid in path_ids]

    def get_all_documents(self) -> List[Document]:
        """Get all documents for embedding generation."""
        return list(self._index.by_path_id.values())

    def search_by_keywords(
        self, keywords: List[str], max_results: int = 50
    ) -> List[Document]:
        """Fast keyword search using in-memory indices."""
        matching_path_ids = set()

        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Search documentation words
            if keyword_lower in self._index.documentation_words:
                matching_path_ids.update(self._index.documentation_words[keyword_lower])

            # Search path segments
            if keyword_lower in self._index.path_segments:
                matching_path_ids.update(self._index.path_segments[keyword_lower])

            # Search path IDs directly
            for path_id in self._index.by_path_id:
                if keyword_lower in path_id.lower():
                    matching_path_ids.add(path_id)

        # Return documents
        results = [self._index.by_path_id[pid] for pid in matching_path_ids]
        return results[:max_results]

    def search_by_physics_domain(self, domain: str) -> List[Document]:
        """Search by physics domain using index."""
        path_ids = self._index.by_physics_domain.get(domain, set())
        return [self._index.by_path_id[pid] for pid in path_ids]

    def search_by_units(self, units: str) -> List[Document]:
        """Search by units using index."""
        path_ids = self._index.by_units.get(units, set())
        return [self._index.by_path_id[pid] for pid in path_ids]

    def search_by_coordinates(self, coordinate: str) -> List[Document]:
        """Search by coordinate system using index."""
        path_ids = self._index.by_coordinates.get(coordinate, set())
        return [self._index.by_path_id[pid] for pid in path_ids]

    # SQLite3 Full-Text Search Integration
    def _build_sqlite_fts_index(self) -> None:
        """Build SQLite3 FTS5 index for advanced text search with metadata tracking."""
        logger.info("Building SQLite FTS5 index...")

        with sqlite3.connect(str(self._sqlite_path)) as conn:
            # Always ensure clean tables for consistent schema
            conn.execute("DROP TABLE IF EXISTS documents")
            conn.execute("DROP TABLE IF EXISTS index_metadata")

            # Create metadata table (key-value pairs)
            conn.execute("""
                CREATE TABLE index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # Create FTS5 virtual table (content table, not contentless)
            conn.execute("""
                CREATE VIRTUAL TABLE documents USING fts5(
                    path_id UNINDEXED,
                    ids_name,
                    path_name,
                    documentation,
                    physics_domain,
                    units,
                    coordinates,
                    data_type,
                    embedding_text
                )
            """)

            # Insert all documents
            for document in self._index.by_path_id.values():
                coords_str = " ".join(document.metadata.coordinates)

                conn.execute(
                    """
                    INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        document.metadata.path_id,
                        document.metadata.ids_name,
                        document.metadata.path_name,
                        document.documentation,
                        document.metadata.physics_domain,
                        document.metadata.units,
                        coords_str,
                        document.metadata.data_type,
                        document.embedding_text,
                    ),
                )

            # Store metadata for cache validation
            import time

            conn.execute(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
                ("created_at", str(time.time())),
            )
            conn.execute(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
                ("document_count", str(self._index.total_documents)),
            )
            conn.execute(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
                ("ids_count", str(self._index.total_ids)),
            )
            conn.execute(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
                ("data_dir_hash", self._compute_data_dir_hash()),
            )

            conn.commit()

        logger.info("SQLite FTS5 index built successfully")

    @contextmanager
    def _sqlite_connection(self):
        """Context manager for SQLite connections."""
        conn = sqlite3.connect(str(self._sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def search_full_text(
        self, query: str, fields: Optional[List[str]] = None, max_results: int = 50
    ) -> List[Document]:
        """
        Advanced full-text search using SQLite FTS5.

        Args:
            query: FTS5 query string (supports AND, OR, NOT, quotes, etc.)
            fields: Specific fields to search in (default: all)
            max_results: Maximum results to return

        Returns:
            List of matching documents

        Examples:
            search_full_text('plasma temperature')
            search_full_text('physics_domain:transport AND units:eV')
            search_full_text('"electron density" OR "ion density"')
        """
        with self._sqlite_connection() as conn:
            # Build FTS5 query
            if fields:
                # Search specific fields
                field_queries = []
                for field in fields:
                    field_queries.append(f"{field}:{query}")
                fts_query = " OR ".join(field_queries)
            else:
                # Search all fields
                fts_query = query

            try:
                cursor = conn.execute(
                    """
                    SELECT path_id, rank
                    FROM documents 
                    WHERE documents MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """,
                    (fts_query, max_results),
                )

                results = []
                for row in cursor:
                    document = self._index.by_path_id.get(row["path_id"])
                    if document:
                        results.append(document)

                return results

            except sqlite3.OperationalError as e:
                logger.error(f"FTS query failed: {e}")
                return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the document store."""
        cache_info = self.get_cache_info()

        return {
            "total_documents": self._index.total_documents,
            "total_ids": self._index.total_ids,
            "physics_domains": len(self._index.by_physics_domain),
            "unique_units": len(self._index.by_units),
            "coordinate_systems": len(self._index.by_coordinates),
            "documentation_terms": len(self._index.documentation_words),
            "path_segments": len(self._index.path_segments),
            "cache": cache_info,
            "data_directory": str(self._data_dir),
        }

    def get_physics_domains(self) -> List[str]:
        """Get all available physics domains."""
        return list(self._index.by_physics_domain.keys())

    def get_available_units(self) -> List[str]:
        """Get all available units."""
        return list(self._index.by_units.keys())

    def get_coordinate_systems(self) -> List[str]:
        """Get all available coordinate systems."""
        return list(self._index.by_coordinates.keys())

    def get_available_ids(self) -> List[str]:
        """Get list of available IDS names."""
        return list(self._index.by_ids_name.keys())

    def _should_rebuild_fts_index(self) -> bool:
        """Check if FTS index needs rebuilding based on cache validation."""
        if not self._sqlite_path.exists():
            logger.debug("SQLite index does not exist, needs rebuild")
            return True

        try:
            with sqlite3.connect(str(self._sqlite_path)) as conn:
                conn.row_factory = sqlite3.Row  # Enable dict-like access
                # Check if required tables exist
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('documents', 'index_metadata')
                """)
                tables = {row[0] for row in cursor.fetchall()}

                if not {"documents", "index_metadata"}.issubset(tables):
                    logger.debug("Required tables missing, needs rebuild")
                    return True

                # Check index metadata using key-value pairs
                cursor = conn.execute("SELECT key, value FROM index_metadata")
                metadata = {row["key"]: row["value"] for row in cursor}

                if not metadata:
                    logger.debug("No index metadata found, needs rebuild")
                    return True

                stored_doc_count = int(metadata.get("document_count", 0))
                stored_ids_count = int(metadata.get("ids_count", 0))
                stored_data_hash = metadata.get("data_dir_hash", "")
                index_timestamp = float(metadata.get("created_at", 0))

                # Check if document counts match
                if (
                    stored_doc_count != self._index.total_documents
                    or stored_ids_count != self._index.total_ids
                ):
                    logger.debug(
                        f"Document count mismatch: stored={stored_doc_count}, current={self._index.total_documents}"
                    )
                    return True

                # Check if data directory has changed
                current_data_hash = self._compute_data_dir_hash()
                if stored_data_hash != current_data_hash:
                    logger.debug("Data directory changed, needs rebuild")
                    return True

                # Check if any source files are newer than the index
                if self._has_newer_source_files(index_timestamp):
                    logger.debug("Source files newer than index, needs rebuild")
                    return True

                return False

        except sqlite3.Error as e:
            logger.warning(f"Error checking SQLite index: {e}, will rebuild")
            return True

    def _compute_data_dir_hash(self) -> str:
        """Compute hash of data directory path for cache validation."""
        import hashlib

        return hashlib.md5(str(self._data_dir.resolve()).encode()).hexdigest()

    def _has_newer_source_files(self, index_timestamp: float) -> bool:
        """Check if any source JSON files are newer than the index timestamp."""
        # Check catalog file
        catalog_path = self._data_dir / "ids_catalog.json"
        if catalog_path.exists() and catalog_path.stat().st_mtime > index_timestamp:
            return True

        # Check detailed files
        detailed_dir = self._data_dir / "detailed"
        if detailed_dir.exists():
            for json_file in detailed_dir.glob("*.json"):
                if json_file.stat().st_mtime > index_timestamp:
                    return True

        return False

    def clear_cache(self) -> None:
        """Clear the SQLite FTS cache and force rebuild on next access."""
        try:
            # Remove cache file if it exists
            if self._sqlite_path.exists():
                self._sqlite_path.unlink()
                logger.info("SQLite FTS cache cleared")
            else:
                logger.info("No SQLite cache to clear")

        except PermissionError as e:
            # On Windows, file might be locked - try to handle gracefully
            logger.warning(f"Could not remove cache file (file locked): {e}")
            logger.info("Cache will be rebuilt on next access")

    def rebuild_index(self) -> None:
        """Force rebuild of the SQLite FTS index."""
        logger.info("Force rebuilding SQLite FTS index...")

        # Remove the cache file and rebuild
        try:
            if self._sqlite_path.exists():
                self._sqlite_path.unlink()
        except PermissionError:
            logger.warning("Could not remove cache file - will recreate")

        # Force rebuild
        self._build_sqlite_fts_index()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the SQLite cache."""
        if not self._sqlite_path.exists():
            return {
                "cached": False,
                "file_path": str(self._sqlite_path),
                "message": "No cache file exists",
            }

        try:
            with sqlite3.connect(str(self._sqlite_path)) as conn:
                conn.row_factory = sqlite3.Row  # Enable dict-like access
                # Get basic file info
                file_size_bytes = self._sqlite_path.stat().st_size
                file_size_mb = file_size_bytes / (1024 * 1024)

                # Get index metadata using key-value pairs
                cursor = conn.execute("SELECT key, value FROM index_metadata")
                metadata = {row["key"]: row["value"] for row in cursor}

                if not metadata:
                    return {
                        "cached": True,
                        "file_path": str(self._sqlite_path),
                        "file_size_mb": round(file_size_mb, 2),
                        "status": "invalid",
                        "message": "Cache file exists but missing metadata",
                    }

                created_at = float(metadata.get("created_at", 0))
                doc_count = int(metadata.get("document_count", 0))
                ids_count = int(metadata.get("ids_count", 0))
                data_hash = metadata.get("data_dir_hash", "")
                version = metadata.get("version", "unknown")

                # Get document count from FTS table
                cursor = conn.execute("SELECT COUNT(*) FROM documents")
                fts_doc_count = cursor.fetchone()[0]

                # Format timestamp
                import time

                created_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(created_at)
                )

                # Check if cache is still valid
                current_data_hash = self._compute_data_dir_hash()
                is_valid = (
                    data_hash == current_data_hash
                    and doc_count == self._index.total_documents
                    and ids_count == self._index.total_ids
                    and not self._has_newer_source_files(created_at)
                )

                return {
                    "cached": True,
                    "file_path": str(self._sqlite_path),
                    "file_size_mb": round(file_size_mb, 2),
                    "created_at": created_time,
                    "document_count": doc_count,
                    "ids_count": ids_count,
                    "fts_document_count": fts_doc_count,
                    "version": version or "1.0",
                    "data_dir_hash": data_hash,
                    "current_data_hash": current_data_hash,
                    "is_valid": is_valid,
                    "status": "valid" if is_valid else "stale",
                    "message": "Cache is up to date"
                    if is_valid
                    else "Cache needs rebuild",
                }

        except sqlite3.Error as e:
            return {
                "cached": True,
                "file_path": str(self._sqlite_path),
                "status": "error",
                "error": str(e),
                "message": "Error reading cache file",
            }

    def close(self) -> None:
        """Close any open database connections."""
        # Note: We use context managers for all connections, so nothing to close
        logger.debug(
            "DocumentStore close() called - using context managers, no persistent connections"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure connections are closed."""
        self.close()
