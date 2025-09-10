"""
In-memory document store for IMAS data with SQLite3 full-text search.

This module provides fast access to IMAS JSON documents optimized for LLM tools
and sentence transformer search. Uses in-memory storage with SQLite3 for
complex queries and full-text search.
"""

import hashlib
import importlib.resources as resources
import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imas_mcp
from imas_mcp.core.data_model import IdentifierSchema, IdsNode, PhysicsContext
from imas_mcp.core.physics_accessors import UnitAccessor
from imas_mcp.core.unit_loader import get_unit_dimensionality, get_unit_name

logger = logging.getLogger(__name__)


@dataclass
class Units:
    """Manages unit-related information and context."""

    unit_str: str
    name: str = ""
    context: str = ""
    category: str | None = None
    physics_domains: list[str] = field(default_factory=list)
    dimensionality: str = ""

    @classmethod
    def from_unit_string(
        cls,
        unit_str: str,
        unit_accessor: UnitAccessor,
    ) -> "Units":
        """Create Units instance from unit string using UnitAccessor."""
        # Get unit context from the physics accessor
        context = unit_accessor.get_unit_context(unit_str) or ""
        category = unit_accessor.get_category_for_unit(unit_str)
        physics_domains = unit_accessor.get_domains_for_unit(unit_str)

        name = get_unit_name(unit_str)
        dimensionality = get_unit_dimensionality(unit_str)

        return cls(
            unit_str=unit_str,
            name=name,
            context=context,
            category=category,
            physics_domains=[domain.value for domain in physics_domains],
            dimensionality=dimensionality,
        )

    def has_meaningful_units(self) -> bool:
        """Check if this represents meaningful physical units."""
        return bool(self.unit_str and self.unit_str not in ("", "none", "1"))

    def get_embedding_components(self) -> list[str]:
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
    physics_context: dict[str, Any] = field(default_factory=dict)
    relationships: dict[str, list[str]] = field(default_factory=dict)
    raw_data: dict[str, Any] = field(default_factory=dict)
    units: Units | None = None

    def set_units(self, unit_accessor: UnitAccessor) -> None:
        """Set units information using UnitAccessor."""
        if self.metadata.units:
            self.units = Units.from_unit_string(
                self.metadata.units,
                unit_accessor,
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

    def to_datapath(self) -> IdsNode:
        """Convert Document to DataPath for consistent API responses."""
        # Build physics context if available
        physics_context = None
        if self.metadata.physics_domain:
            physics_context = PhysicsContext(
                domain=self.metadata.physics_domain,
                phenomena=list(self.metadata.physics_phenomena)
                if self.metadata.physics_phenomena
                else [],
                typical_values={},
            )

        # Build identifier schema if available
        identifier_schema = None
        if self.raw_data.get("identifier_schema"):
            schema_data = self.raw_data["identifier_schema"]
            if isinstance(schema_data, dict):
                identifier_schema = IdentifierSchema(
                    schema_path=schema_data.get("schema_path", ""),
                    documentation=schema_data.get("documentation"),
                    options=schema_data.get("options", []),
                    metadata=schema_data.get("metadata", {}),
                )

        return IdsNode(
            path=self.metadata.path_name,
            documentation=self.documentation,
            units=self.units.unit_str if self.units else "",
            coordinates=list(self.metadata.coordinates)
            if self.metadata.coordinates
            else [],
            lifecycle=self.raw_data.get(
                "lifecycle", "active"
            ),  # Extract from raw_data with fallback
            data_type=self.metadata.data_type,
            introduced_after_version=self.raw_data.get("introduced_after_version"),
            lifecycle_status=self.raw_data.get("lifecycle_status"),
            lifecycle_version=self.raw_data.get("lifecycle_version"),
            physics_context=physics_context,
            related_paths=self.raw_data.get("related_paths", []),
            usage_examples=self.raw_data.get("usage_examples", []),
            validation_rules=self.raw_data.get("validation_rules"),
            identifier_schema=identifier_schema,
            coordinate1=self.raw_data.get("coordinate1"),
            coordinate2=self.raw_data.get("coordinate2"),
            timebase=self.raw_data.get("timebase"),
            type=self.raw_data.get("type"),
            structure_reference=self.raw_data.get("structure_reference"),
        )


@dataclass
class SearchIndex:
    """In-memory search indices for fast lookups."""

    # Primary indices
    by_path_id: dict[str, Document] = field(default_factory=dict)
    by_ids_name: dict[str, list[str]] = field(default_factory=dict)

    # Search indices
    by_physics_domain: dict[str, set[str]] = field(default_factory=dict)
    by_units: dict[str, set[str]] = field(default_factory=dict)
    by_coordinates: dict[str, set[str]] = field(default_factory=dict)

    # Full-text indices
    documentation_words: dict[str, set[str]] = field(default_factory=dict)
    path_segments: dict[str, set[str]] = field(default_factory=dict)

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
    ids_set: set[str] | None = None  # Specific IDS to load (for testing/performance)

    # Internal state
    _index: SearchIndex = field(default_factory=SearchIndex, init=False)
    _data_dir: Path = field(init=False)
    _sqlite_path: Path = field(init=False)
    _loaded: bool = field(default=False, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _unit_contexts: dict[str, str] = field(default_factory=dict, init=False)
    _unit_categories: dict[str, list[str]] = field(default_factory=dict, init=False)
    _physics_domain_hints: dict[str, list[str]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        """Initialize the document store with on-demand loading."""
        # Use importlib.resources to locate JSON data
        self._data_dir = self._get_resources_path()

        # Build cache in dedicated database directory with consistent naming
        sqlite_dir = self._get_sqlite_dir()
        sqlite_dir.mkdir(parents=True, exist_ok=True)
        self._sqlite_path = sqlite_dir / self._generate_db_filename()

        # Initialize unit accessor for physics context
        self._unit_accessor = UnitAccessor()

        # Get all available unit contexts for logging
        all_unit_contexts = self._unit_accessor.get_all_unit_contexts()
        logger.info(
            f"Loaded {len(all_unit_contexts)} unit context definitions via physics integration"
        )

        # Initialize lazy loading state
        self._loaded_ids: set[str] = set()  # Track which IDS are loaded
        self._available_ids: list[str] | None = None  # Cache of available IDS
        self._identifier_catalog_loaded = False  # Track identifier catalog loading

        # Don't auto-load documents - use on-demand loading per IDS

    def _ensure_ids_loaded(self, ids_names: list[str]) -> None:
        """Ensure specific IDS are loaded on-demand."""
        # Check which IDS need to be loaded
        to_load = [
            ids_name for ids_name in ids_names if ids_name not in self._loaded_ids
        ]

        if not to_load:
            return  # All requested IDS already loaded

        logger.info(f"Loading {len(to_load)} IDS")

        # Load each missing IDS
        for ids_name in to_load:
            self._load_ids_documents(ids_name)
            self._loaded_ids.add(ids_name)

        # Load identifier catalog if not already loaded
        if not self._identifier_catalog_loaded:
            self._load_identifier_catalog_documents()
            self._identifier_catalog_loaded = True

        # Update the loaded flag if we have any documents
        if self._index.total_documents > 0:
            self._loaded = True

        # Always refresh total_ids to reflect currently loaded IDS (even for partial loads)
        # Keys of by_ids_name represent each distinct IDS loaded so far.
        self._index.total_ids = len(self._index.by_ids_name)

    def _ensure_loaded(self) -> None:
        """Ensure all available documents are loaded (fallback for full loading)."""
        if not self._loaded:
            available_ids = self._get_available_ids()  # This now respects ids_set
            if self.ids_set is not None:
                logger.info(
                    f"Loading {len(available_ids)} filtered IDS: {available_ids}"
                )
            else:
                logger.info("Loading all available IDS...")
            self._ensure_ids_loaded(available_ids)

    def _get_resources_path(self) -> Path:
        """Get the path to the resources directory using importlib.resources."""
        try:
            # Use the new files() API instead of deprecated path()
            resources_dir = resources.files("imas_mcp") / "resources" / "schemas"
            # Convert Traversable to Path using str conversion
            return Path(str(resources_dir))
        except (ImportError, FileNotFoundError):
            # Fallback to package relative path
            package_path = Path(imas_mcp.__file__).parent
            return package_path / "resources" / "schemas"

    def _get_sqlite_dir(self) -> Path:
        """Get the database directory for database files."""
        try:
            # Use resources directory for database files
            database_dir = resources.files("imas_mcp") / "resources" / "database"
            return Path(str(database_dir))
        except (ImportError, FileNotFoundError):
            # Fallback to package relative path
            package_path = Path(imas_mcp.__file__).parent
            return package_path / "resources" / "database"

    def _get_dd_version(self) -> str:
        """Extract IMAS data dictionary version from ids_catalog.json metadata.

        Returns:
            Version string if present, else "unknown".
        """
        catalog_path = self._data_dir / "ids_catalog.json"
        if not catalog_path.exists():  # pragma: no cover - defensive
            return "unknown"
        try:
            with open(catalog_path, encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("metadata", {}) if isinstance(data, dict) else {}
            version = meta.get("version")
            return version or "unknown"
        except Exception:  # pragma: no cover - defensive
            return "unknown"

    # Public accessor for data dictionary version
    def get_dd_version(self) -> str:
        """Return the IMAS data dictionary version (fallback to package version).

        This provides an external, stable API rather than relying on the
        private ``_get_dd_version`` helper.
        """
        version = self._get_dd_version()
        if not version or version == "unknown":  # defensive fallback
            try:
                import importlib.metadata

                version = importlib.metadata.version("imas-mcp")
            except Exception:  # pragma: no cover - defensive
                version = "unknown"
        return version

    def _generate_db_filename(self) -> str:
        """Generate consistent database filename based on configuration."""
        # Create hash from ids_set for consistent naming
        if self.ids_set:
            ids_str = "_".join(sorted(self.ids_set))
            ids_hash = hashlib.md5(ids_str.encode()).hexdigest()[:8]
            return f"imas_fts_{ids_hash}.db"
        else:
            return "imas_fts.db"

    def is_available(self) -> bool:
        """Check if IMAS data is available."""
        return (self._data_dir / "ids_catalog.json").exists()

    def load_all_documents(
        self, force_rebuild_index: bool = False, ids_filter: list[str] | None = None
    ) -> None:
        """Load JSON documents into memory with indexing.

        Args:
            force_rebuild_index: Force rebuild of SQLite FTS index
            ids_filter: Optional list of IDS names to load (for faster testing)
                       If not provided, uses self.ids_set if available
        """
        with self._lock:
            if self._loaded:
                return

            logger.info("Loading IMAS documents into memory...")

            # Load catalog to get available IDS
            available_ids = self._get_available_ids()

            # Determine which IDS to load
            target_ids = ids_filter
            if target_ids is None and self.ids_set is not None:
                target_ids = list(self.ids_set)

            # Filter IDS if specified
            if target_ids:
                available_ids = sorted(
                    [ids for ids in available_ids if ids in target_ids]
                )
                logger.info(f"Filtering to {len(available_ids)} IDS: {available_ids}")
                # Force rebuild when using IDS filter since cache won't match
                force_rebuild_index = True

            self._index.total_ids = len(available_ids)

            # Load each IDS detailed file (in sorted order for deterministic results)
            for ids_name in sorted(available_ids):
                self._load_ids_documents(ids_name)

            # Load identifier catalog documents
            self._load_identifier_catalog_documents()

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

    def _get_available_ids(self) -> list[str]:
        """Get list of available IDS names without loading documents, respecting ids_set filter."""
        if self._available_ids is not None:
            # If ids_set is specified, filter the cached available IDS
            if self.ids_set is not None:
                return [ids for ids in self._available_ids if ids in self.ids_set]
            return self._available_ids

        catalog_path = self._data_dir / "ids_catalog.json"
        if not catalog_path.exists():
            logger.warning(f"Catalog not found: {catalog_path}")
            self._available_ids = []
            return self._available_ids

        try:
            with open(catalog_path, encoding="utf-8") as f:
                catalog = json.load(f)
            all_ids = list(catalog.get("ids_catalog", {}).keys())
            self._available_ids = all_ids

            # If ids_set is specified, filter the available IDS
            if self.ids_set is not None:
                return [ids for ids in all_ids if ids in self.ids_set]
            return all_ids
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            self._available_ids = []
            return self._available_ids

    def get_available_ids(self) -> list[str]:
        """Get list of available IDS names (public method)."""
        return self._get_available_ids()

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

    def _load_identifier_catalog_documents(self) -> None:
        """Load identifier catalog as special documents for search and access."""
        identifier_catalog_file = self._data_dir / "identifier_catalog.json"
        if not identifier_catalog_file.exists():
            logger.debug("No identifier catalog found, skipping identifier documents")
            return

        try:
            with open(identifier_catalog_file, encoding="utf-8") as f:
                catalog_data = json.load(f)

            # Prepare to limit schemas to only those referenced by allowed IDS when ids_set is active
            schemas = catalog_data.get("schemas", {})
            paths_by_ids = catalog_data.get("paths_by_ids", {})

            # Determine which IDS are allowed
            allowed_ids = (
                set(self.ids_set)
                if self.ids_set is not None
                else set(paths_by_ids.keys())
            )

            # Build set of referenced schema names from allowed IDS only
            referenced_schema_names: set[str] = set()
            for ids_name in sorted(paths_by_ids.keys()):
                if ids_name not in allowed_ids:
                    continue
                for path_data in paths_by_ids[ids_name]:
                    schema_name = path_data.get("schema_name")
                    if schema_name:
                        referenced_schema_names.add(schema_name)

            # Load schema documents: if filtered, only index referenced schemas to avoid leaking unrelated schemas
            for schema_name in sorted(schemas.keys()):
                if (
                    self.ids_set is not None
                    and schema_name not in referenced_schema_names
                ):
                    continue
                schema_data = schemas[schema_name]
                document = self._create_identifier_schema_document(
                    schema_name, schema_data
                )
                self._index.add_document(document)

            # Load path documents that reference identifier schemas (sorted for deterministic order)
            for ids_name in sorted(paths_by_ids.keys()):
                identifier_paths = paths_by_ids[ids_name]
                # Apply IDS filter for identifier paths
                if self.ids_set is not None and ids_name not in allowed_ids:
                    continue

                # Sort identifier paths for deterministic order
                for path_data in sorted(
                    identifier_paths, key=lambda x: x.get("path", "")
                ):
                    document = self._create_identifier_path_document(
                        ids_name, path_data
                    )
                    self._index.add_document(document)

            logger.info(
                f"Loaded {len(schemas)} identifier schemas and {sum(len(paths) for paths in paths_by_ids.values())} identifier paths"
            )

        except Exception as e:
            logger.error(f"Failed to load identifier catalog: {e}")

    def _create_identifier_schema_document(
        self, schema_name: str, schema_data: dict[str, Any]
    ) -> Document:
        """Create a Document for an identifier schema."""
        path_id = f"identifier_schema/{schema_name.lower().replace(' ', '_')}"

        # Create documentation text with all options
        options_text = []
        for option in schema_data.get("options", []):
            options_text.append(
                f"{option['index']}: {option['name']} - {option['description']}"
            )

        documentation = f"""Identifier Schema: {schema_name}

{schema_data.get("description", "")}

Available Options ({schema_data.get("total_options", 0)} total):
{chr(10).join(options_text)}

Usage: Used in {schema_data.get("usage_count", 0)} paths across IMAS
Branching Complexity: {schema_data.get("branching_complexity", 0):.2f}
Physics Domains: {", ".join(schema_data.get("physics_domains", []))}

Paths using this schema:
{chr(10).join(f"- {path}" for path in schema_data.get("usage_paths", [])[:10])}
"""

        # Create metadata
        metadata = DocumentMetadata(
            path_id=path_id,
            ids_name="identifier_schema",
            path_name=schema_name,
            units="",
            data_type="identifier_schema",
            coordinates=(),
            physics_domain=schema_data.get("physics_domains", [""])[0]
            if schema_data.get("physics_domains")
            else "",
            physics_phenomena=tuple(schema_data.get("physics_domains", [])),
        )

        # Create document with enhanced raw_data for MCP tools
        raw_data = {
            **schema_data,
            "document_type": "identifier_schema",
            "schema_name": schema_name,
            "is_identifier": True,
            "branching_logic": {
                "total_options": schema_data.get("total_options", 0),
                "complexity": schema_data.get("branching_complexity", 0),
                "enumeration_space": schema_data.get("total_options", 0),
            },
        }

        document = Document(
            metadata=metadata,
            documentation=documentation,
            physics_context={
                "domain": metadata.physics_domain,
                "phenomena": list(metadata.physics_phenomena),
            },
            relationships={"identifier_paths": schema_data.get("usage_paths", [])},
            raw_data=raw_data,
        )

        # Set empty units since this is a schema document
        document.set_units(self._unit_accessor)
        return document

    def _create_identifier_path_document(
        self, ids_name: str, path_data: dict[str, Any]
    ) -> Document:
        """Create a Document for an identifier path reference."""
        path = path_data.get("path", "")
        path_id = f"identifier_path/{path}"

        documentation = f"""Identifier Path: {path}

{path_data.get("description", "")}

Schema: {path_data.get("schema_name", "")} ({path_data.get("option_count", 0)} options)
Physics Domain: {path_data.get("physics_domain", "unspecified")}

This path uses identifier enumeration logic that defines branching behavior in the {ids_name} IDS.
See the '{path_data.get("schema_name", "")}' identifier schema for available options.
"""

        # Create metadata
        metadata = DocumentMetadata(
            path_id=path_id,
            ids_name=ids_name,
            path_name=path,
            units="",
            data_type="identifier_path",
            coordinates=(),
            physics_domain=path_data.get("physics_domain", ""),
            physics_phenomena=tuple(
                [path_data.get("physics_domain")]
                if path_data.get("physics_domain")
                else []
            ),
        )

        # Create document with enhanced raw_data
        raw_data = {
            **path_data,
            "document_type": "identifier_path",
            "is_identifier": True,
            "has_branching_logic": True,
            "enumeration_options": path_data.get("option_count", 0),
        }

        document = Document(
            metadata=metadata,
            documentation=documentation,
            physics_context={
                "domain": metadata.physics_domain,
                "phenomena": list(metadata.physics_phenomena),
            },
            relationships={"schema_reference": path_data.get("schema_name", "")},
            raw_data=raw_data,
        )

        document.set_units(self._unit_accessor)
        return document

    def _create_document(
        self, ids_name: str, path_name: str, path_data: dict[str, Any]
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
        document.set_units(self._unit_accessor)

        return document

    # Fast access methods for LLM tools
    def get_document(self, path_id: str) -> Document | None:
        """Get document by path ID, loading on-demand if needed."""
        # Try to get from cache first
        document = self._index.by_path_id.get(path_id)
        if document:
            return document

        # If not found, try to determine which IDS this path belongs to and load it
        if "/" in path_id:
            ids_name = path_id.split("/")[0]
            self._ensure_ids_loaded([ids_name])
            return self._index.by_path_id.get(path_id)

        return None

    def get_documents_by_ids(self, ids_name: str) -> list[Document]:
        """Get all documents for an IDS, loading on-demand if needed."""
        # Load this specific IDS if not already loaded
        self._ensure_ids_loaded([ids_name])
        path_ids = self._index.by_ids_name.get(ids_name, [])
        return [self._index.by_path_id[pid] for pid in path_ids]

    def get_all_documents(self) -> list[Document]:
        """Get all documents for embedding generation, respecting ids_set filter."""
        self._ensure_loaded()
        return list(self._index.by_path_id.values())

    def __len__(self) -> int:
        """Get the number of documents in the store, respecting ids_set filter."""
        self._ensure_loaded()
        return len(self._index.by_path_id)

    def get_document_count(self) -> int:
        """Get the count of documents in the store."""
        return len(self)

    def search_by_keywords(
        self, keywords: list[str], max_results: int = 50
    ) -> list[Document]:
        """Fast keyword search using in-memory indices."""
        # For keyword search, we need to load all documents since we don't know which IDS contain the keywords
        self._ensure_loaded()
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

    def search_by_physics_domain(self, domain: str) -> list[Document]:
        """Search by physics domain using index."""
        # For physics domain search, we need all documents loaded
        self._ensure_loaded()
        path_ids = self._index.by_physics_domain.get(domain, set())
        return [self._index.by_path_id[pid] for pid in path_ids]

    def search_by_units(self, units: str) -> list[Document]:
        """Search by units using index."""
        # For units search, we need all documents loaded
        self._ensure_loaded()
        path_ids = self._index.by_units.get(units, set())
        return [self._index.by_path_id[pid] for pid in path_ids]

    # Identifier-specific access methods
    def get_identifier_schemas(self) -> list[Document]:
        """Get all identifier schema documents."""
        return [
            doc
            for doc in self._index.by_path_id.values()
            if doc.metadata.data_type == "identifier_schema"
        ]

    def get_identifier_paths(self) -> list[Document]:
        """Get all documents that have identifier schemas (branching logic)."""
        return [
            doc
            for doc in self._index.by_path_id.values()
            if doc.raw_data.get("identifier_schema")
            or doc.metadata.data_type == "identifier_path"
        ]

    def get_identifier_schema_by_name(self, schema_name: str) -> Document | None:
        """Get a specific identifier schema by name."""
        schema_path_id = f"identifier_schema/{schema_name.lower().replace(' ', '_')}"
        return self.get_document(schema_path_id)

    def search_identifier_schemas(self, query: str) -> list[Document]:
        """Search specifically in identifier schemas."""
        all_schemas = self.get_identifier_schemas()
        query_lower = query.lower()

        matching_schemas = []
        for schema in all_schemas:
            # Search in schema name, description, and option names/descriptions
            if (
                query_lower in schema.metadata.path_name.lower()
                or query_lower in schema.documentation.lower()
            ):
                matching_schemas.append(schema)
            else:
                # Search in individual options
                options = schema.raw_data.get("options", [])
                for option in options:
                    if (
                        query_lower in option.get("name", "").lower()
                        or query_lower in option.get("description", "").lower()
                    ):
                        matching_schemas.append(schema)
                        break

        return matching_schemas

    def get_paths_with_identifiers_by_ids(self, ids_name: str) -> list[Document]:
        """Get all paths in an IDS that have identifier schemas."""
        all_docs = self.get_documents_by_ids(ids_name)
        return [
            doc
            for doc in all_docs
            if doc.raw_data.get("identifier_schema")
            or doc.raw_data.get("is_identifier")
        ]

    def get_identifier_branching_summary(self) -> dict[str, Any]:
        """Get a summary of identifier branching logic across all IDS."""
        schemas = self.get_identifier_schemas()

        # Group by physics domain
        by_physics_domain = {}
        total_options = 0

        for schema in schemas:
            schema_data = schema.raw_data
            options = schema_data.get("total_options", 0)
            total_options += options

            domains = schema_data.get("physics_domains", ["unspecified"])
            for domain in domains:
                if domain not in by_physics_domain:
                    by_physics_domain[domain] = {
                        "schemas": [],
                        "total_options": 0,
                        "paths": [],
                    }
                by_physics_domain[domain]["schemas"].append(schema.metadata.path_name)
                by_physics_domain[domain]["total_options"] += options

        # Group paths by IDS - include both identifier_path documents and regular docs with identifier_schema
        by_ids = {}
        for doc in self._index.by_path_id.values():
            # Check if this document has identifier schema (regular IDS documents)
            has_identifier = (
                doc.raw_data.get("identifier_schema")
                or doc.metadata.data_type == "identifier_path"
                or doc.raw_data.get("is_identifier", False)
            )

            if has_identifier:
                ids_name = doc.metadata.ids_name
                if ids_name not in by_ids:
                    by_ids[ids_name] = []
                by_ids[ids_name].append(doc.metadata.path_name)

        # Count total identifier paths correctly
        total_identifier_paths = sum(len(paths) for paths in by_ids.values())

        return {
            "total_schemas": len(schemas),
            "total_identifier_paths": total_identifier_paths,
            "total_enumeration_options": total_options,
            "by_physics_domain": by_physics_domain,
            "by_ids": by_ids,
            "complexity_metrics": {
                "avg_options_per_schema": total_options / len(schemas)
                if schemas
                else 0,
                "max_complexity": max(
                    (s.raw_data.get("branching_complexity", 0) for s in schemas),
                    default=0,
                ),
            },
        }

    def search_by_coordinates(self, coordinate: str) -> list[Document]:
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
            # Store ids_set for cache validation
            ids_set_str = "_".join(sorted(self.ids_set)) if self.ids_set else "all"
            conn.execute(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
                ("ids_set", ids_set_str),
            )
            # Persist data dictionary version (dd_version) for health endpoint visibility
            dd_version = self._get_dd_version()
            conn.execute(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
                ("version", dd_version),
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

    def _preprocess_fts_query(self, query: str) -> str:
        """
        Preprocess FTS query to handle common user patterns that cause SQL errors.

        Args:
            query: Raw user query string

        Returns:
            Processed query safe for FTS5
        """
        # First, find all quoted sections and temporarily replace them to protect from processing
        quoted_sections = []
        quote_pattern = r'"[^"]*"'

        def protect_quotes(match):
            quoted_sections.append(match.group(0))
            return f"__QUOTE_{len(quoted_sections) - 1}__"

        # Protect quoted sections
        protected_query = re.sub(quote_pattern, protect_quotes, query)

        # Handle various problematic dash patterns:

        # 1. Handle "-word" patterns by converting to "NOT word"
        # This prevents "no such column" errors when users type things like "plasma -wall"
        def replace_minus_term(match):
            prefix = match.group(1)
            term = match.group(2)
            # Only replace if the term doesn't look like it's part of a field specification
            if ":" not in term:
                return f"{prefix}NOT {term}"
            return match.group(0)

        # Pattern to match "-word" but not "field:-word"
        # Look for space or start of string, followed by -, followed by word characters
        processed_query = re.sub(
            r"(^|\s)-([a-zA-Z_][a-zA-Z0-9_]*)", replace_minus_term, protected_query
        )

        # 2. Handle standalone dashes and dash with spaces that could cause syntax errors
        # Remove isolated dashes (- surrounded by spaces or at boundaries)
        processed_query = re.sub(r"\s+-\s+", " ", processed_query)  # " - " -> " "
        processed_query = re.sub(r"^-\s+", "", processed_query)  # "- " at start -> ""
        processed_query = re.sub(r"\s+-$", "", processed_query)  # " -" at end -> ""
        processed_query = re.sub(r"^-$", "", processed_query)  # just "-" -> ""

        # 3. Clean up multiple spaces
        processed_query = re.sub(r"\s+", " ", processed_query)

        # Restore quoted sections
        for i, quoted_text in enumerate(quoted_sections):
            processed_query = processed_query.replace(f"__QUOTE_{i}__", quoted_text)

        return processed_query.strip()

    def search_full_text(
        self, query: str, fields: list[str] | None = None, max_results: int = 50
    ) -> list[Document]:
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
            search_full_text('plasma temperature -wall')  # converted to 'plasma temperature NOT wall'
        """
        # Ensure documents are loaded and FTS index exists
        self._ensure_loaded()

        # Check if FTS index needs to be built
        if self._should_rebuild_fts_index():
            self._build_sqlite_fts_index()

        # Preprocess query to handle common user patterns
        processed_query = self._preprocess_fts_query(query)

        with self._sqlite_connection() as conn:
            # Build FTS5 query
            if fields:
                # Search specific fields
                field_queries = []
                for field in fields:
                    field_queries.append(f"{field}:{processed_query}")
                fts_query = " OR ".join(field_queries)
            else:
                # Search all fields
                fts_query = processed_query

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

    def get_statistics(self) -> dict[str, Any]:
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

    def get_physics_domains(self) -> list[str]:
        """Get all available physics domains."""
        return list(self._index.by_physics_domain.keys())

    def get_available_units(self) -> list[str]:
        """Get all available units."""
        return list(self._index.by_units.keys())

    def get_coordinate_systems(self) -> list[str]:
        """Get all available coordinate systems."""
        return list(self._index.by_coordinates.keys())

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
                stored_ids_set = metadata.get("ids_set", "all")
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

                # Check if ids_set matches
                current_ids_set = (
                    "_".join(sorted(self.ids_set)) if self.ids_set else "all"
                )
                if stored_ids_set != current_ids_set:
                    logger.debug(
                        f"IDS set mismatch: stored={stored_ids_set}, current={current_ids_set}"
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
        """Compute hash of data directory path and ids_set for cache validation."""
        hash_data = str(self._data_dir.resolve())

        # Include ids_set in hash for proper cache isolation
        if self.ids_set:
            ids_str = "_".join(sorted(self.ids_set))
            hash_data += f"_ids_{ids_str}"

        return hashlib.md5(hash_data.encode()).hexdigest()

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

    def get_cache_info(self) -> dict[str, Any]:
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

    def get_index_metadata(self) -> dict[str, Any]:
        """Return lightweight index metadata for health/status endpoints.

        Provides a stable, minimal subset without forcing callers to parse the
        full cache info structure. Falls back to in-memory counters if cache
        metadata is unavailable.
        """
        info = {}
        try:
            info = self.get_cache_info()
        except Exception:  # pragma: no cover - defensive
            info = {}

        version = info.get("version") if isinstance(info, dict) else None
        # Fallbacks rely on in-memory index if cache missing
        document_count = info.get("document_count") if isinstance(info, dict) else None
        if not document_count:
            document_count = getattr(self._index, "total_documents", None)
        ids_count = info.get("ids_count") if isinstance(info, dict) else None
        if not ids_count:
            ids_count = getattr(self._index, "total_ids", None)

        return {
            "version": version,
            "document_count": document_count,
            "ids_count": ids_count,
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
