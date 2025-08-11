"""
Unit context loader for IMAS Data Dictionary.

This module loads unit context definitions from YAML files in the definitions
directory to provide semantic context for physical units used in semantic search.
Uses pint to automatically include long-form unit names in contexts.
"""

import importlib.resources as resources
import logging

import yaml

from imas_mcp.units import unit_registry

logger = logging.getLogger(__name__)


def load_unit_contexts() -> dict[str, str]:
    """
    Load unit context definitions from YAML file with pint long-form unit names.

    Returns:
        Dictionary mapping unit strings to semantic context descriptions

    Raises:
        FileNotFoundError: If unit_contexts.yaml is not found
        yaml.YAMLError: If YAML parsing fails
    """
    try:
        # Use importlib.resources with the new files() API
        yaml_file = resources.files("imas_mcp.definitions") / "unit_contexts.yaml"
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))

        unit_contexts = data.get("unit_contexts", {})

        logger.info(f"Loaded {len(unit_contexts)} unit context definitions")
        return unit_contexts

    except yaml.YAMLError as e:
        logger.error(f"Error parsing unit_contexts.yaml: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading unit contexts: {e}")
        return {}


def load_unit_categories() -> dict[str, list[str]]:
    """
    Load unit category definitions from YAML file.

    Returns:
        Dictionary mapping category names to lists of units
    """
    try:
        yaml_file = resources.files("imas_mcp.definitions") / "unit_contexts.yaml"
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))

        return data.get("unit_categories", {})

    except Exception as e:
        logger.error(f"Error loading unit categories: {e}")
        return {}


def load_physics_domain_hints() -> dict[str, list[str]]:
    """
    Load physics domain hints based on unit categories.

    Returns:
        Dictionary mapping unit categories to likely physics domains
    """
    try:
        yaml_file = resources.files("imas_mcp.definitions") / "unit_contexts.yaml"
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))

        return data.get("physics_domain_hints", {})

    except Exception as e:
        logger.error(f"Error loading physics domain hints: {e}")
        return {}


def get_unit_category(
    unit: str, categories: dict[str, list[str]] | None = None
) -> str | None:
    """
    Determine which category a unit belongs to.

    Args:
        unit: Unit string to categorize
        categories: Optional pre-loaded categories dict

    Returns:
        Category name if found, None otherwise
    """
    if categories is None:
        categories = load_unit_categories()

    for category, units in categories.items():
        if unit in units:
            return category

    return None


def get_unit_physics_domains(
    unit: str,
    categories: dict[str, list[str]] | None = None,
    domain_hints: dict[str, list[str]] | None = None,
) -> list[str]:
    """
    Get physics domains associated with a unit based on its category.

    Args:
        unit: Unit string
        categories: Optional pre-loaded categories dict
        domain_hints: Optional pre-loaded domain hints dict

    Returns:
        List of physics domains for this unit
    """
    if categories is None:
        categories = load_unit_categories()
    if domain_hints is None:
        domain_hints = load_physics_domain_hints()

    category = get_unit_category(unit, categories)
    if category and category in domain_hints:
        return domain_hints[category]

    return []


def get_unit_name(unit_str: str) -> str:
    """
    Get unit name for a unit using pint with custom 'U' formatter.

    Args:
        unit_str: Unit string (e.g., "T", "eV", "Pa")

    Returns:
        Unit name from pint formatter
    """

    try:
        # Handle special case for dimensionless units
        if unit_str in ("1", "dimensionless", ""):
            return "dimensionless"

        # Parse the unit with pint
        unit = unit_registry(unit_str)

        # Use the custom 'U' formatter to get unit names
        unit_name = f"{unit.units:U}"
        return unit_name

    except Exception as e:
        logger.debug(f"Could not parse unit '{unit_str}' with pint: {e}")
        return ""


def get_unit_dimensionality(unit_str: str) -> str:
    """
    Get dimensionality description for a unit using pint's built-in formatting.

    Args:
        unit_str: Unit string (e.g., "T", "eV", "Pa")

    Returns:
        Pint's dimensionality string or empty string
    """
    try:
        # Handle special case for dimensionless units
        if unit_str in ("1", "dimensionless", ""):
            return "dimensionless"

        unit = unit_registry(unit_str)

        # Check if unit has dimensionality attribute (some parsed units might not)
        if hasattr(unit, "dimensionality"):
            dimensionality = str(unit.dimensionality)
        elif hasattr(unit, "units") and hasattr(unit.units, "dimensionality"):
            dimensionality = str(unit.units.dimensionality)
        else:
            # Fallback for cases where dimensionality is not accessible
            logger.debug(
                f"Unit '{unit_str}' parsed as {type(unit)} but no dimensionality found"
            )
            return "unknown"

        # Return pint's clean dimensionality formatting directly
        return dimensionality

    except Exception as e:
        logger.debug(f"Could not get dimensionality for unit '{unit_str}': {e}")

    return ""


if __name__ == "__main__":
    # Test the loader
    print("Testing unit context loader...")

    contexts = load_unit_contexts()
    print(f"Loaded {len(contexts)} unit contexts")

    # Test some common units
    test_units = ["T", "m", "s", "eV", "Pa"]
    for unit in test_units:
        context = contexts.get(unit, "No context found")
        category = get_unit_category(unit)
        domains = get_unit_physics_domains(unit)
        print(f"{unit:<5}: {context}")
        print(f"{'':5}  Category: {category}")
        print(f"{'':5}  Domains: {domains}")
        print()
