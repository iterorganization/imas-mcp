"""Base protocol for SN extraction sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ExtractionBatch:
    """A batch of candidates extracted from a source for LLM composition.

    Each batch groups related items (e.g., same IDS, same cluster, same diagnostic)
    to give the LLM coherent context for generating standard names.
    """

    source: str  # "dd" or "signals"
    group_key: str  # e.g., IDS name or diagnostic name
    items: list[dict]  # Source-specific extraction data
    context: str  # Human-readable context for the LLM prompt
    existing_names: set[str] = field(
        default_factory=set
    )  # Known standard names for dedup
    dd_version: str | None = None  # DD version whose conventions apply
    cocos_version: int | None = None  # COCOS convention from that DD version
    cocos_params: dict | None = None  # Full COCOS node properties


class ExtractionSource(Protocol):
    """Protocol for source extraction plugins."""

    def extract(
        self,
        *,
        ids_filter: str | None = None,
        domain_filter: str | None = None,
        facility: str | None = None,
        limit: int = 500,
    ) -> list[ExtractionBatch]:
        """Extract candidate batches from this source."""
        ...
