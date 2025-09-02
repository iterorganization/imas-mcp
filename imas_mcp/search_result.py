"""
Models for search results and indexable documents in the IMAS MCP server.

This module contains Pydantic models that represent documents that can be indexed
in the search engine and the search results returned from the search engine.
"""

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
    ids_name: str | None = None

    # Extended fields from JSON data
    coordinates: str | None = None
    lifecycle: str | None = None
    data_type: str | None = None
    physics_context: str | None = None
    related_paths: str | None = None
    usage_examples: str | None = None
    validation_rules: str | None = None
    relationships: str | None = None
    introduced_after: str | None = None
    coordinate1: str | None = None
    coordinate2: str | None = None
    timebase: str | None = None
    type: str | None = None

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
            raise ValueError(f"Invalid units '{units}': {e}") from e

    @pydantic.model_validator(mode="after")
    def update_fields(self) -> "DataDictionaryEntry":
        """Update unset fields."""
        if self.ids_name is None:
            self.ids_name = self.path.split("/")[0]
        return self


# Note: SearchResult for MCP tool responses is now in models/result_models.py
# This file contains only internal search engine models
