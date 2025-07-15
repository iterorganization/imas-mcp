"""
Models for search results and indexable documents in the IMAS MCP server.

This module contains Pydantic models that represent documents that can be indexed
in the search engine and the search results returned from the search engine.
"""

from typing import Optional

import pint
import pydantic
from pydantic import ConfigDict

from imas_mcp.units import unit_registry


# Base model for document validation
class IndexableDocument(pydantic.BaseModel):
    """Base model for documents that can be indexed in search engines."""

    model_config = ConfigDict(
        extra="forbid",  # Prevent additional fields not in schema
        validate_assignment=True,
    )


class DataDictionaryEntry(IndexableDocument):
    """IMAS Data Dictionary document model for validating IDS entries."""

    path: str
    documentation: str
    units: str = ""
    ids_name: Optional[str] = None

    # Extended fields from JSON data
    coordinates: Optional[str] = None
    lifecycle: Optional[str] = None
    data_type: Optional[str] = None
    physics_context: Optional[str] = None
    related_paths: Optional[str] = None
    usage_examples: Optional[str] = None
    validation_rules: Optional[str] = None
    relationships: Optional[str] = None
    introduced_after: Optional[str] = None
    coordinate1: Optional[str] = None
    coordinate2: Optional[str] = None
    timebase: Optional[str] = None
    type: Optional[str] = None

    @pydantic.field_validator("units", mode="after")
    @classmethod
    def parse_units(cls, units: str, info: pydantic.ValidationInfo) -> str:
        """Return units formatted as custom UDUNITS."""
        context = info.context or {}
        skip_unit_parsing = context.get("skip_unit_parsing", False)

        if skip_unit_parsing:
            return units

        if units.endswith("^dimension"):
            # Handle units with '^dimension' suffix
            # This is a workaround for the IMAS DD units that have a '^dimension' suffix
            units = units[:-10].strip() + "__pow__dimension"
        if units in ["", "1", "dimensionless"]:  # dimensionless attribute
            return ""
        if units == "none":  # handle no unit case
            return units
        try:
            return f"{unit_registry.Unit(units):~U}"
        except pint.errors.UndefinedUnitError as e:
            raise ValueError(f"Invalid units '{units}': {e}")

    @pydantic.model_validator(mode="after")
    def update_fields(self) -> "DataDictionaryEntry":
        """Update unset fields."""
        if self.ids_name is None:
            self.ids_name = self.path.split("/")[0]
        return self


class SearchResult(pydantic.BaseModel):
    """Model for storing a single search result."""

    path: str
    score: float
    documentation: str
    units: str
    ids_name: str
    highlights: str = ""

    @property
    def relevance(self) -> float:
        """Alias for score to maintain compatibility."""
        return self.score

    # Extended fields from JSON data
    coordinates: str = ""
    lifecycle: str = ""
    data_type: str = ""
    physics_context: str = ""
    related_paths: str = ""
    usage_examples: str = ""
    validation_rules: str = ""
    relationships: str = ""
    introduced_after: str = ""
    coordinate1: str = ""
    coordinate2: str = ""
    timebase: str = ""
    type: str = ""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def __str__(self) -> str:
        """Return a string representation of the SearchResult."""
        doc_preview = (
            self.documentation[:100] + "..."
            if len(self.documentation) > 100
            else self.documentation
        )
        lines = [
            f"Path: {self.path}",
            f"  Score: {self.score:.4f}",
            f"  IDS: {self.ids_name if self.ids_name else 'N/A'}",
            f"  Units: {self.units if self.units else 'N/A'}",
            f"  Documentation: {doc_preview}",
        ]

        # Add extended fields if they contain data
        if self.lifecycle:
            lines.append(f"  Lifecycle: {self.lifecycle}")
        if self.data_type:
            lines.append(f"  Data Type: {self.data_type}")
        if self.coordinates:
            lines.append(f"  Coordinates: {self.coordinates}")
        if self.physics_context:
            lines.append(f"  Physics Context: {self.physics_context}")
        if self.related_paths:
            related_preview = (
                self.related_paths[:50] + "..."
                if len(self.related_paths) > 50
                else self.related_paths
            )
            lines.append(f"  Related Paths: {related_preview}")
        if self.coordinate1:
            lines.append(f"  Coordinate1: {self.coordinate1}")
        if self.coordinate2:
            lines.append(f"  Coordinate2: {self.coordinate2}")
        if self.timebase:
            lines.append(f"  Timebase: {self.timebase}")
        if self.type:
            lines.append(f"  Type: {self.type}")
        if self.introduced_after:
            lines.append(f"  Introduced After: {self.introduced_after}")

        if self.highlights:  # Check if highlights string is not empty
            lines.append(f"  Highlights: {self.highlights}")
        return "\n".join(lines)
