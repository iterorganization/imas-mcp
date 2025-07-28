"""Centralized enums for IMAS MCP models.

This module contains all enum definitions used across search, response, and physics models.
Consolidating enums here prevents conflicts and provides a single source of truth.
"""

from enum import Enum


# ============================================================================
# SEARCH ENUMS
# ============================================================================


class SearchMode(Enum):
    """Enumeration of available search modes."""

    SEMANTIC = "semantic"  # AI-powered semantic search using sentence transformers
    LEXICAL = "lexical"  # Traditional full-text search using SQLite FTS5
    HYBRID = "hybrid"  # Combination of semantic and lexical search
    AUTO = "auto"  # Automatically choose best mode based on query


# ============================================================================
# RESPONSE ENUMS
# ============================================================================


class DetailLevel(str, Enum):
    """Detail levels for explanations."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class RelationshipType(str, Enum):
    """Types of relationships to explore."""

    ALL = "all"
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"


class IdentifierScope(str, Enum):
    """Scopes for identifier exploration."""

    ALL = "all"
    PATHS = "paths"
    SCHEMAS = "schemas"


# ============================================================================
# PHYSICS ENUMS
# ============================================================================


class ConceptType(str, Enum):
    """Types of physics concepts that can be embedded."""

    DOMAIN = "domain"
    PHENOMENON = "phenomenon"
    UNIT = "unit"
    MEASUREMENT_METHOD = "measurement_method"


class ComplexityLevel(str, Enum):
    """Complexity levels for concept explanations."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class UnitCategory(str, Enum):
    """Categories for physics units."""

    MAGNETIC_FIELD = "magnetic_field"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    DENSITY = "density"
    ENERGY = "energy"
    TIME = "time"
    LENGTH = "length"
    VELOCITY = "velocity"
    CURRENT = "current"
    VOLTAGE = "voltage"
    FORCE = "force"
    POWER = "power"
    FREQUENCY = "frequency"
    ANGULAR_FREQUENCY = "angular_frequency"
    DIMENSIONLESS = "dimensionless"
    UNKNOWN = "unknown"
