"""
Models for search results and indexable documents in the IMAS MCP server.

This module contains Pydantic models that represent documents that can be indexed
in the search engine and the search results returned from the search engine.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING
import importlib.util

import pint
import pydantic
from pydantic import ConfigDict

from imas_mcp.units import unit_registry

# Conditional import for whoosh (legacy dependency)
if TYPE_CHECKING or importlib.util.find_spec("whoosh") is not None:
    import whoosh.searching
else:
    whoosh = None


# Base model for document validation
class IndexableDocument(pydantic.BaseModel):
    """Base model for documents that can be indexed in WhooshIndex."""

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
    path_segments: Optional[str] = None

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
        if self.path_segments is None:
            self.path_segments = " ".join(self.path.split("/"))
        return self


class SearchResult(pydantic.BaseModel):
    """Model for storing a single search result."""

    path: str
    score: float
    documentation: str
    units: str
    ids_name: str
    highlights: str = ""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def from_hit(cls, hit: "whoosh.searching.Hit") -> "SearchResult":
        """Create a SearchResult instance from a Whoosh Hit object."""
        if importlib.util.find_spec("whoosh") is None:
            raise ImportError(
                "Whoosh not available - cannot create SearchResult from Hit"
            )

        return cls(
            path=hit["path"],
            score=hit.score if hit.score is not None else 0.0,
            documentation=hit.get("documentation", ""),
            units=hit.get("units", ""),
            ids_name=hit.get("ids_name", ""),
            highlights=hit.highlights("documentation", ""),
        )

    @classmethod
    def from_document(cls, document: Dict[str, Any]) -> "SearchResult":
        """Create a SearchResult instance from a Whoosh document dictionary."""
        return cls(
            path=document["path"],
            score=1.0,  # Exact match, so score is 1.0
            documentation=document.get("documentation", ""),
            units=document.get("units", ""),
            ids_name=document.get("ids_name", ""),
            highlights="",  # No highlights for direct document retrieval
        )

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
        if self.highlights:  # Check if highlights string is not empty
            lines.append(f"  Highlights: {self.highlights}")
        return "\\n".join(lines)  # Corrected newline character
