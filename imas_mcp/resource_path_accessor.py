"""
Centralized resource path accessor for version-aware resource management.

This module provides a single source of truth for resource paths, ensuring that
both build-time and runtime code access resources from consistent version-specific
directories under imas_data_dictionary/{version}/.
"""

import os
from functools import cached_property
from importlib import resources
from pathlib import Path

from imas_mcp.dd_accessor import DataDictionaryAccessor


class ResourcePathAccessor:
    """Provides version-aware paths for all resource types."""

    def __init__(self, dd_version: str):
        """
        Initialize with DD version string for fast path resolution.

        Args:
            dd_version: DD version string (e.g., '4.0.1.dev277' or '3.42.2').
                       Used to construct version-specific resource paths.
        """
        self._dd_version = dd_version
        self._base_dir = self._get_base_resources_dir()
        self._dd_accessor: DataDictionaryAccessor | None = None  # Lazy-loaded

    @property
    def dd_accessor(self) -> DataDictionaryAccessor:
        """
        Lazy-load DD accessor only when needed for XML parsing.

        This defers expensive operations (package import + XML parsing ~1s)
        until actually required, keeping package import fast.
        """
        if self._dd_accessor is None:
            self._dd_accessor = self._create_dd_accessor_from_env()
        return self._dd_accessor

    def _create_dd_accessor_from_env(self) -> DataDictionaryAccessor:
        """Create DD accessor based on IMAS_DD_VERSION environment variable."""
        dd_version = os.environ.get("IMAS_DD_VERSION", "")

        if dd_version:
            # Check if this is a dev version (contains 'dev' in version string)
            # Dev versions only exist in the git package (imas-data-dictionary)
            if "dev" in dd_version.lower():
                try:
                    from imas_mcp.dd_accessor import ImasDataDictionaryAccessor

                    return ImasDataDictionaryAccessor()
                except ImportError:
                    # Fall back to PyPI package if git package not available
                    from imas_mcp.dd_accessor import ImasDataDictionariesAccessor

                    return ImasDataDictionariesAccessor(dd_version)
            else:
                # Use imas_data_dictionaries (PyPI) for specific stable version
                from imas_mcp.dd_accessor import ImasDataDictionariesAccessor

                return ImasDataDictionariesAccessor(dd_version)
        else:
            # Try default imas-data-dictionary package (git) first
            try:
                from imas_mcp.dd_accessor import ImasDataDictionaryAccessor

                return ImasDataDictionaryAccessor()
            except ImportError:
                # Fall back to PyPI package with default version
                from imas_mcp.dd_accessor import ImasDataDictionariesAccessor

                return ImasDataDictionariesAccessor("4.0.0")

    def _get_base_resources_dir(self) -> Path:
        """Get the base resources directory (imas_mcp/resources/)."""
        try:
            # Use importlib.resources for installed packages
            base = resources.files("imas_mcp") / "resources"
            return Path(str(base))
        except (ImportError, FileNotFoundError):
            # Fallback for development
            import imas_mcp

            return Path(imas_mcp.__file__).parent / "resources"

    def _get_subdir_path(self, subdir_name: str, *, create: bool = True) -> Path:
        """
        Get path to a subdirectory under the version-specific root.

        Args:
            subdir_name: Name of the subdirectory (e.g., 'schemas', 'embeddings')
            create: If True, create the directory if it doesn't exist

        Returns:
            Path to the subdirectory
        """
        dir_path = self.version_dir / subdir_name
        if create:
            dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def check_path_exists(self, subdir_name: str) -> bool:
        """
        Check if a subdirectory exists without creating it.

        Args:
            subdir_name: Name of the subdirectory to check (e.g., 'embeddings', 'database')

        Returns:
            True if the directory exists, False otherwise
        """
        return (self.version_dir / subdir_name).exists()

    @property
    def version_dir(self) -> Path:
        """Get the version-specific root directory (imas_data_dictionary/{dd_version}/).

        Note: Does not create the directory. Use subdirectory properties which will
        create their paths as needed.
        """
        return self._base_dir / "imas_data_dictionary" / self._dd_version

    @cached_property
    def schemas_dir(self) -> Path:
        """Get the schemas directory (imas_data_dictionary/{version}/schemas/)."""
        return self._get_subdir_path("schemas", create=True)

    @cached_property
    def embeddings_dir(self) -> Path:
        """Get the embeddings directory (imas_data_dictionary/{version}/embeddings/)."""
        return self._get_subdir_path("embeddings", create=True)

    @cached_property
    def database_dir(self) -> Path:
        """Get the database directory (imas_data_dictionary/{version}/database/)."""
        return self._get_subdir_path("database", create=True)

    @cached_property
    def mermaid_dir(self) -> Path:
        """Get the mermaid graphs directory (imas_data_dictionary/{version}/mermaid/)."""
        return self._get_subdir_path("mermaid", create=True)

    @cached_property
    def mappings_dir(self) -> Path:
        """Get the path mappings directory (imas_data_dictionary/{version}/mappings/)."""
        return self._get_subdir_path("mappings", create=True)

    @property
    def version(self) -> str:
        """Get the DD version string."""
        return self._dd_version
