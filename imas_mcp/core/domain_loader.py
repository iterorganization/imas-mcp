"""
YAML-based Physics Domain Loader for IMAS Data Dictionary.

This module loads physics domain definitions from YAML files in the definitions folder,
replacing the hardcoded Python-based categorization system.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from importlib import resources

logger = logging.getLogger(__name__)


class DomainDefinitionLoader:
    """Loads physics domain definitions from YAML files."""

    def __init__(self, definitions_dir: Optional[Path] = None):
        """Initialize the loader with the definitions directory."""
        if definitions_dir is None:
            try:
                definitions_package = resources.files(
                    "imas_mcp.definitions.physics_domains"
                )
                definitions_dir = Path(str(definitions_package))
            except (ImportError, FileNotFoundError):
                # Fallback to relative path for development
                current_file = Path(__file__).parent
                package_root = current_file.parent  # imas_mcp package directory
                definitions_dir = package_root / "definitions" / "physics_domains"

        self.definitions_dir = Path(definitions_dir)
        self._domain_characteristics = None
        self._ids_mapping = None
        self._domain_relationships = None

        if not self.definitions_dir.exists():
            logger.warning(f"Definitions directory not found: {self.definitions_dir}")

    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load a YAML file from the definitions directory using PyYAML."""
        try:
            # Try using importlib.resources first (works for installed packages)
            definitions_package = resources.files(
                "imas_mcp.definitions.physics_domains"
            )
            file_ref = definitions_package / filename

            if file_ref.is_file():
                with file_ref.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                logger.debug(
                    f"Loaded {filename}: {data.get('metadata', {}).get('description', 'No description')}"
                )
                return data
        except (ImportError, FileNotFoundError, AttributeError):
            # Fallback to filesystem path
            pass

        # Fallback method using filesystem path
        file_path = self.definitions_dir / filename
        if not file_path.exists():
            logger.error(f"Definition file not found: {file_path}")
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            logger.debug(
                f"Loaded {filename}: {data.get('metadata', {}).get('description', 'No description')}"
            )
            return data
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return {}

    def load_domain_characteristics(self) -> Dict[str, Dict[str, Any]]:
        """Load domain characteristics from YAML."""
        if self._domain_characteristics is None:
            data = self._load_yaml_file("domain_characteristics.yaml")
            self._domain_characteristics = data.get("domains", {})
        return self._domain_characteristics

    def load_ids_mapping(self) -> Dict[str, str]:
        """Load IDS to domain mapping from YAML."""
        if self._ids_mapping is None:
            data = self._load_yaml_file("ids_mapping.yaml")

            # Flatten the domain -> IDS list structure to IDS -> domain mapping
            mapping = {}
            domains_data = data.copy()
            domains_data.pop("metadata", None)  # Remove metadata section

            for domain, ids_list in domains_data.items():
                for ids_name in ids_list:
                    mapping[ids_name.lower()] = domain

            self._ids_mapping = mapping
        return self._ids_mapping

    def load_domain_relationships(self) -> Dict[str, List[str]]:
        """Load domain relationships from YAML."""
        if self._domain_relationships is None:
            data = self._load_yaml_file("domain_relationships.yaml")
            self._domain_relationships = data.get("relationships", {})
        return self._domain_relationships

    def get_metadata(self, definition_type: str) -> Dict[str, Any]:
        """Get metadata for a specific definition type."""
        filename_map = {
            "characteristics": "domain_characteristics.yaml",
            "mapping": "ids_mapping.yaml",
            "relationships": "domain_relationships.yaml",
        }

        filename = filename_map.get(definition_type)
        if not filename:
            return {}

        data = self._load_yaml_file(filename)
        return data.get("metadata", {})

    def validate_definitions(self) -> Dict[str, Any]:
        """Validate the loaded definitions for consistency."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {},
        }

        try:
            characteristics = self.load_domain_characteristics()
            ids_mapping = self.load_ids_mapping()
            relationships = self.load_domain_relationships()

            # Check if all domains in mapping exist in characteristics
            mapped_domains = set(ids_mapping.values())
            defined_domains = set(characteristics.keys())

            missing_characteristics = mapped_domains - defined_domains
            if missing_characteristics:
                validation_results["errors"].append(
                    f"Domains in mapping missing from characteristics: {missing_characteristics}"
                )
                validation_results["valid"] = False

            # Check if all domains in relationships exist in characteristics
            relationship_domains = set(relationships.keys())
            for domain, related in relationships.items():
                relationship_domains.update(related)

            missing_from_relationships = relationship_domains - defined_domains
            if missing_from_relationships:
                validation_results["warnings"].append(
                    f"Domains in relationships missing from characteristics: {missing_from_relationships}"
                )

            # Collect statistics
            validation_results["statistics"] = {
                "total_domains": len(defined_domains),
                "total_ids": len(ids_mapping),
                "domains_with_relationships": len(relationships),
                "average_relationships_per_domain": sum(
                    len(r) for r in relationships.values()
                )
                / len(relationships)
                if relationships
                else 0,
            }

        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation failed: {e}")

        return validation_results


# Global loader instance
_domain_loader = None


def get_domain_loader() -> DomainDefinitionLoader:
    """Get the global domain definition loader instance."""
    global _domain_loader
    if _domain_loader is None:
        _domain_loader = DomainDefinitionLoader()
    return _domain_loader


def load_physics_domains_from_yaml() -> Dict[str, Any]:
    """Load complete physics domain definitions from YAML files."""
    loader = get_domain_loader()

    return {
        "characteristics": loader.load_domain_characteristics(),
        "ids_mapping": loader.load_ids_mapping(),
        "relationships": loader.load_domain_relationships(),
        "metadata": {
            "characteristics": loader.get_metadata("characteristics"),
            "mapping": loader.get_metadata("mapping"),
            "relationships": loader.get_metadata("relationships"),
        },
        "validation": loader.validate_definitions(),
    }
