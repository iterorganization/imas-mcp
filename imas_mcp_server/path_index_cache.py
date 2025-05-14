#!/usr/bin/env python
"""
Build and serialize the IMAS PathIndex for faster imports.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Set
import functools
import imas
import shutil

from imas_mcp_server.path_index import PathIndex

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PathIndexCache")


@dataclass
class PathIndexCache:
    """Builds and serializes the IMAS DD PathIndex."""

    # Attributes for locating or rebuilding the path index cache
    cache_dir: Path | None = None  # Directory to save the index
    ids_set: Set[str] | None = None  # Set of IDS names to index
    version: str | None = None  # IMAS version to use
    xml_path: Path | None = None  # Path to the XML file

    # Attributes initialized later or derived
    path_index: PathIndex = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize derived attributes after dataclass init."""
        if self.cache_dir is None:
            self.cache_dir = Path(__file__).parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Create or load the path index
        self.path_index = self.load()

    @property
    def index_dir(self) -> Path:
        """Return the directory for the Whoosh index."""
        assert self.cache_dir is not None, "Cache directory not set"
        assert self.dd_version is not None, "DD version not set"
        return self.cache_dir / f"path_index_{self.dd_version}"

    @functools.cached_property
    def dd_etree(self):
        """Return the IMAS DD XML tree."""
        tree = imas.dd_zip.dd_etree(version=self.version, xml_path=self.xml_path)
        if tree is None:
            raise ValueError("Failed to load IMAS data dictionary XML tree")
        return tree

    @functools.cached_property
    def dd_version(self) -> str:
        """Return the IMAS DD version from the IMAS XML tree."""
        root = self.dd_etree.getroot()
        if root is None:
            raise ValueError("Root element not found in XML tree")
        version_elem = root.find(".//version")
        if version_elem is None:
            raise ValueError("Version element not found in XML tree")
        version = version_elem.text
        assert version is not None, "Version not found in XML tree"
        return version

    @functools.cached_property
    def ids(self) -> Set[str]:
        """Return a set of path index IDS names."""
        if self.ids_set:  # return requested set of IDS names if provided
            return self.ids_set
        # construct full IDS list from etree when ids is None
        return set(
            name
            for elem in self.dd_etree.findall(".//IDS[@name]")
            if (name := elem.get("name")) is not None
        )

    def load(self) -> PathIndex:
        """Return cached path index or build it if not found."""
        # Check if there's a Whoosh index directory
        if self.index_dir.exists():
            logger.info(f"Loading Whoosh index from {self.index_dir}")
            try:
                path_index = PathIndex(
                    version=self.dd_version, index_dir=self.index_dir
                )
                # Add timeout to avoid hanging indefinitely if index is locked or corrupted
                if self.ids_set and path_index.get_ids_set() != self.ids:
                    logger.info(
                        "Whoosh index has different IDS set than requested, rebuilding..."
                    )
                    return self.build()
                return path_index
            except Exception as e:
                logger.error(f"Error loading Whoosh index: {e}")
                logger.info("Rebuilding index due to loading error")
                return self.build()

        logger.info("No valid index found, building a new one.")
        return self.build()

    def _build_hierarchical_documentation(self, documentation_parts):
        """Build hierarchical documentation string with increasing prominence for
        leaf nodes.

        Parameters:
        -----------
        documentation_parts : dict[str, str]
            Dictionary of documentation strings with keys as node names and values as documentation

        Returns:
        --------
        str
            Formatted hierarchical documentation with leaf nodes emphasized
        """
        if not documentation_parts:
            return ""

        # Create a hierarchical documentation string
        doc_levels = []
        # documentation_parts is already in reverse order (leaf to root)
        for path, doc in documentation_parts.items():
            doc_levels.append(f"## {path}\n{doc}")

        # Join with clear section separation
        return "\n\n".join(doc_levels)

    def build(self) -> PathIndex:
        """Build an optimized index of all available IDS paths.
        This method creates an index of all IDS paths by traversing the IMAS data structures.
        It constructs paths from elements with attributes up to their respective IDS roots.
        Returns:
            PathIndex: The constructed PathIndex object with all indexed paths
        """
        logger.info("Building IDS path index...")
        start_time = time.time()

        # Remove old Whoosh index if it exists
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
            logger.info(
                f"Removed existing Whoosh index at: {self.index_dir}"
            )  # Create the Whoosh-based PathIndex with index directory
        path_index = PathIndex(
            version=self.dd_version, index_dir=self.index_dir
        )  # Collect paths in batches for better performance
        paths_batch = {}
        batch_size = 1000  # Process in batches of 1000 paths

        for ids in self.dd_etree.findall(".//IDS"):
            ids_name = ids.get("name")
            if not ids_name:
                continue
            # restrict search to ids set
            if ids_name not in self.ids:
                continue

            # Find all elements with attributes
            for elem in ids.findall(".//*[@name]"):
                # Build path from this element up to the IDS
                path_parts = []
                documentation_parts = []

                # Walk up the tree to build the path
                current = elem
                while current is not None and current != ids:
                    if name := current.get("name"):
                        path_parts.insert(0, name)
                    if name and (doc := current.get("documentation")):
                        documentation_parts.append(doc)

                    # Navigate to parent - handle lxml or ElementTree
                    parent = None
                    if hasattr(current, "getparent"):
                        # Likely lxml
                        parent = current.getparent()  # type: ignore
                    else:
                        # Standard ElementTree - requires searching
                        found = False
                        for potential_parent in ids.iter():
                            if current in list(potential_parent):
                                parent = potential_parent
                                found = True
                                break
                        if not found:
                            parent = None  # Should not happen in valid XML unless current is root

                    current = parent  # Move to the found parent

                if path_parts:
                    path = f"{ids_name}/{'/'.join(path_parts)}"  # Create hierarchical path keys from leaf to root
                    documentation_keys = []
                    for i in range(len(path_parts)):
                        documentation_keys.insert(0, "/".join(path_parts[: i + 1]))
                    documentation_dict = dict(
                        zip(documentation_keys, documentation_parts)
                    )
                    documentation = self._build_hierarchical_documentation(
                        documentation_dict
                    )  # Add to batch instead of adding directly
                    paths_batch[path] = documentation

                    # Process batch if it reaches the threshold
                    if len(paths_batch) >= batch_size:
                        path_index.batch_add_paths(paths_batch)
                        paths_batch = {}

        # Process any remaining paths in the batch
        if paths_batch:
            path_index.batch_add_paths(paths_batch, optimize=True)

        build_time = time.time() - start_time
        logger.info(
            f"Built index with {path_index.count_paths()} paths in {build_time:.2f} seconds"
        )

        logger.info(f"Whoosh index saved to directory: {self.index_dir}")
        return path_index

    def clear(self):
        """Clear Whoosh index directory."""
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
            logger.info(f"Cleared Whoosh index directory: {self.index_dir}")


if __name__ == "__main__":
    path_index_cache = PathIndexCache(version="4.0.0", ids_set={"pf_passive"})

    path_index = path_index_cache.path_index  # Test access to a path's documentation
    print(path_index.get_document("equilibrium/time_slice/global_quantities/beta_pol"))

    # Test search
    results = path_index.search_by_keywords("electron temperature profile")
    for result in results[:3]:  # Show top 3 results
        print(f"Path: {result['path']}")
        print(f"Score: {result['score']}")
        print(f"Doc: {result['doc'][:50]}...")
        print()
