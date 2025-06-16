"""
Composable data dictionary accessor for IMAS DD.

This module provides a composable pattern for accessing the IMAS Data Dictionary,
allowing the system to work with or without the imas-data-dictionary package installed.
"""

import abc
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

from packaging.version import Version

logger = logging.getLogger(__name__)


class DataDictionaryAccessor(abc.ABC):
    """Abstract base class for data dictionary access."""

    @abc.abstractmethod
    def get_xml_tree(self) -> ET.ElementTree:
        """Get the XML ElementTree for the data dictionary."""
        pass

    @abc.abstractmethod
    def get_version(self) -> Version:
        """Get the data dictionary version."""
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the data dictionary is available."""
        pass


class ImasDataDictionaryAccessor(DataDictionaryAccessor):
    """Accessor that uses the imas-data-dictionary package."""

    def __init__(self):
        self._imas_dd = None
        self._load_imas_dd()

    def _load_imas_dd(self) -> None:
        """Load the imas-data-dictionary package."""
        try:
            import imas_data_dictionary

            self._imas_dd = imas_data_dictionary
        except ImportError as e:
            raise ImportError(
                "imas-data-dictionary package is required for index building. "
                "Install with: pip install imas-mcp[build]"
            ) from e

    def get_xml_tree(self) -> ET.ElementTree:
        """Get the XML ElementTree for the data dictionary."""
        if not self._imas_dd:
            raise RuntimeError("imas-data-dictionary not available")

        xml_path = self._imas_dd.get_xml_resource("IDSDef.xml")
        with xml_path.open("rb") as f:
            return ET.parse(f)

    def get_version(self) -> Version:
        """Get the data dictionary version."""
        tree = self.get_xml_tree()
        root = tree.getroot()
        version_elem = root.find(".//version")

        if version_elem is None or version_elem.text is None:
            raise ValueError(
                "Version element or its text content not found in XML tree"
            )

        return Version(version_elem.text)

    def is_available(self) -> bool:
        """Check if the data dictionary is available."""
        return self._imas_dd is not None


class MetadataDataDictionaryAccessor(DataDictionaryAccessor):
    """Accessor that uses cached metadata files."""

    def __init__(self, metadata_dir: Path):
        self.metadata_dir = metadata_dir
        self._cached_metadata: Optional[Dict[str, Any]] = None

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from the most recent metadata file."""
        if self._cached_metadata is not None:
            return self._cached_metadata

        metadata_files = list(self.metadata_dir.glob("*.metadata.json"))
        if not metadata_files:
            raise FileNotFoundError("No metadata files found")

        # Use the most recent metadata file
        latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)

        with latest_metadata.open("r") as f:
            self._cached_metadata = json.load(f)

        return self._cached_metadata

    def get_xml_tree(self) -> ET.ElementTree:
        """Not supported by metadata accessor."""
        raise NotImplementedError("XML tree access not supported by metadata accessor")

    def get_version(self) -> Version:
        """Get the data dictionary version from metadata."""
        metadata = self._load_metadata()
        version_str = metadata.get("dd_version")

        if not version_str:
            raise ValueError("No dd_version found in metadata")

        return Version(version_str)

    def is_available(self) -> bool:
        """Check if metadata is available."""
        try:
            self._load_metadata()
            return True
        except (FileNotFoundError, ValueError):
            return False


class IndexNameDataDictionaryAccessor(DataDictionaryAccessor):
    """Accessor that extracts version from index name."""

    def __init__(self, index_name: str, index_prefix: str):
        self.index_name = index_name
        self.index_prefix = index_prefix

    def get_xml_tree(self) -> ET.ElementTree:
        """Not supported by index name accessor."""
        raise NotImplementedError(
            "XML tree access not supported by index name accessor"
        )

    def get_version(self) -> Version:
        """Extract version from index name."""
        # Parse pattern: lexicographic_4.0.1.dev164-9dbb96e3
        pattern = rf"{re.escape(self.index_prefix)}_([^-]+)(?:-[a-f0-9]+)?$"
        match = re.match(pattern, self.index_name)

        if not match:
            raise ValueError(
                f"Cannot extract version from index name: {self.index_name}"
            )

        return Version(match.group(1))

    def is_available(self) -> bool:
        """Check if version can be extracted from index name."""
        try:
            self.get_version()
            return True
        except ValueError:
            return False


class EnvironmentDataDictionaryAccessor(DataDictionaryAccessor):
    """Accessor that uses environment variables."""

    def __init__(self, env_var: str = "IMAS_DD_VERSION"):
        self.env_var = env_var

    def get_xml_tree(self) -> ET.ElementTree:
        """Not supported by environment accessor."""
        raise NotImplementedError(
            "XML tree access not supported by environment accessor"
        )

    def get_version(self) -> Version:
        """Get version from environment variable."""
        version_str = os.getenv(self.env_var)
        if not version_str:
            raise ValueError(f"Environment variable {self.env_var} not set")

        return Version(version_str)

    def is_available(self) -> bool:
        """Check if environment variable is set."""
        return os.getenv(self.env_var) is not None


class CompositeDataDictionaryAccessor(DataDictionaryAccessor):
    """Composite accessor that tries multiple accessors in order."""

    def __init__(self, accessors: List[DataDictionaryAccessor]):
        self.accessors = accessors
        self._primary_accessor: Optional[DataDictionaryAccessor] = None

    def _get_available_accessor(self) -> DataDictionaryAccessor:
        """Get the first available accessor."""
        if self._primary_accessor and self._primary_accessor.is_available():
            return self._primary_accessor

        for accessor in self.accessors:
            if accessor.is_available():
                self._primary_accessor = accessor
                return accessor

        raise RuntimeError("No data dictionary accessor is available")

    def get_xml_tree(self) -> ET.ElementTree:
        """Get XML tree from the first available accessor that supports it."""
        for accessor in self.accessors:
            if accessor.is_available():
                try:
                    return accessor.get_xml_tree()
                except NotImplementedError:
                    continue

        raise RuntimeError("No accessor supports XML tree access")

    def get_version(self) -> Version:
        """Get version from the first available accessor."""
        accessor = self._get_available_accessor()
        return accessor.get_version()

    def is_available(self) -> bool:
        """Check if any accessor is available."""
        return any(accessor.is_available() for accessor in self.accessors)


def create_dd_accessor(
    metadata_dir: Optional[Path] = None,
    index_name: Optional[str] = None,
    index_prefix: Optional[str] = None,
) -> DataDictionaryAccessor:
    """Create a composite data dictionary accessor with fallback chain."""
    accessors: List[DataDictionaryAccessor] = []

    # 1. Environment variable (highest priority)
    accessors.append(EnvironmentDataDictionaryAccessor())

    # 2. Metadata file
    if metadata_dir:
        accessors.append(MetadataDataDictionaryAccessor(metadata_dir))

    # 3. Index name parsing
    if index_name and index_prefix:
        accessors.append(IndexNameDataDictionaryAccessor(index_name, index_prefix))

    # 4. IMAS data dictionary package (lowest priority)
    try:
        accessors.append(ImasDataDictionaryAccessor())
    except ImportError:
        logger.debug("imas-data-dictionary package not available")

    return CompositeDataDictionaryAccessor(accessors)


def save_index_metadata(
    metadata_dir: Path,
    index_name: str,
    dd_version: Version,
    ids_names: List[str],
    total_documents: int,
    index_type: str,
    build_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save index metadata to a JSON file."""
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dd_version": str(dd_version),
        "build_timestamp": datetime.utcnow().isoformat(),
        "ids_names": sorted(ids_names),
        "total_documents": total_documents,
        "index_type": index_type,
        "index_name": index_name,
    }

    if build_metadata:
        metadata.update(build_metadata)

    metadata_file = metadata_dir / f"{index_name}.metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_file}")
