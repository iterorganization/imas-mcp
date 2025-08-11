"""Constants for IMAS MCP models.

This module contains all constant definitions used across search, response, and physics models.
"""

from enum import Enum

# ============================================================================
# SEARCH CONSTANTS
# ============================================================================


class SearchMode(Enum):
    """Enumeration of available search modes."""

    SEMANTIC = "semantic"  # AI-powered semantic search using sentence transformers
    LEXICAL = "lexical"  # Traditional full-text search using SQLite FTS5
    HYBRID = "hybrid"  # Combination of semantic and lexical search
    AUTO = "auto"  # Automatically choose best mode based on query


# ============================================================================
# RESPONSE CONSTANTS
# ============================================================================


class DetailLevel(str, Enum):
    """Detail levels for explanations."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class RelationshipType(str, Enum):
    """Types of relationships to explore."""

    ALL = "all"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    PHYSICS = "physics"
    MEASUREMENT = "measurement"


class IdentifierScope(str, Enum):
    """Scopes for identifier exploration."""

    ALL = "all"
    ENUMS = "enums"
    IDENTIFIERS = "identifiers"
    COORDINATES = "coordinates"
    CONSTANTS = "constants"


class OutputFormat(str, Enum):
    """Output formats for export tools."""

    STRUCTURED = "structured"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"


class AnalysisDepth(str, Enum):
    """Analysis depth levels for exports."""

    OVERVIEW = "overview"
    FOCUSED = "focused"
    COMPREHENSIVE = "comprehensive"
    DETAILED = "detailed"


# ============================================================================
# PHYSICS CONSTANTS
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
