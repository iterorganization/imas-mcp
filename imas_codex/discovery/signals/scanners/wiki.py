"""Wiki-based signal scanner plugin.

Extracts signal definitions from wiki content already ingested into the
knowledge graph. Wiki pages often contain structured tables mapping
signal names to MDSplus paths, units, and descriptions — metadata that
would otherwise require expensive LLM enrichment to generate.

Discovery strategy:
  1. Query graph for WikiChunk nodes with mdsplus_paths_mentioned or
     signal-table content for the target facility
  2. Parse structured signal definitions from wiki tables
  3. Create FacilitySignal nodes with pre-populated descriptions and units
  4. Return wiki_context for injection into LLM enrichment of other scanners

This scanner is complementary — it runs alongside primary scanners (TDI, PPF,
EDAS) to provide high-quality metadata from curated documentation. Signals
discovered here may overlap with other scanners; deduplication happens at
graph merge time (MERGE on signal ID).

Cost savings: Wiki-documented signals with descriptions and units can skip
or reduce LLM enrichment, potentially saving >80% of enrichment cost for
well-documented facilities.

Config: Activated for any facility with wiki_sites configured.
Not a data_sources key — added automatically by get_scanners_for_facility().
"""

from __future__ import annotations

import logging
import re
from typing import Any

from imas_codex.discovery.signals.scanners.base import (
    ScanResult,
    register_scanner,
)
from imas_codex.graph.models import (
    FacilitySignal,
    FacilitySignalStatus,
)

logger = logging.getLogger(__name__)

# Patterns for extracting signal info from wiki table content
_MDS_PATH_PATTERN = re.compile(
    r"\\\\?(\w+)::(\w+(?:[:.]\w+)*)",  # \TREE::SUBTREE:NODE or \TREE::SUB.NODE
)

# Common wiki table patterns for signal metadata
_UNIT_PATTERNS = re.compile(
    r"\b(A|V|T|eV|keV|m|cm|mm|m\^-3|m\^-2|W|MW|kW|Pa|"
    r"rad|deg|s|ms|us|Hz|kHz|MHz|K|Ohm|Tesla|Weber|dimensionless)\b"
)


def _extract_signals_from_chunk(
    chunk_text: str,
    facility: str,
    page_title: str,
) -> list[dict[str, str]]:
    """Extract signal definitions from a wiki chunk's text.

    Looks for structured content like:
    - MDSplus path tables: \\TREE::NODE | description | units
    - Signal catalogs: signal_name | mds_path | description
    - TDI function documentation: tcv_eq('QUANTITY') -> description

    Returns list of dicts with keys: path, name, description, units, page.
    """
    signals = []

    # Split into lines for table parsing
    lines = chunk_text.split("\n")

    for line in lines:
        # Look for MDSplus paths in the line
        mds_matches = _MDS_PATH_PATTERN.findall(line)
        if not mds_matches:
            continue

        for tree, node_path in mds_matches:
            full_path = f"\\{tree}::{node_path}"

            # Extract the signal name from the node path
            name = node_path.split(":")[-1].split(".")[-1]

            # Look for description text in the same line
            # Remove the path itself and common separators
            desc_text = _MDS_PATH_PATTERN.sub("", line)
            desc_text = re.sub(r"[|│┃\t]{2,}", "|", desc_text)
            parts = [p.strip() for p in desc_text.split("|") if p.strip()]

            description = ""
            units = ""

            for part in parts:
                # Skip if it's just the path or very short
                if len(part) < 3:
                    continue
                # Check if it looks like a unit
                if _UNIT_PATTERNS.match(part.strip()):
                    units = part.strip()
                elif not description and len(part) > 5:
                    description = part

            signals.append(
                {
                    "path": full_path,
                    "tree": tree,
                    "name": name,
                    "description": description,
                    "units": units,
                    "page": page_title,
                }
            )

    return signals


def _build_wiki_context(
    facility: str,
) -> dict[str, dict[str, str]]:
    """Query graph for wiki content relevant to signal discovery.

    Two-phase extraction:
    1. Pre-extracted paths: Use mdsplus_paths_mentioned, ppf_paths_mentioned,
       and imas_paths_mentioned fields (populated during wiki ingestion) for
       fast, structured path lookup.
    2. Content parsing: For chunks with path mentions, parse the surrounding
       content to extract descriptions and units.

    Returns a dict keyed by signal path (uppercase), with values containing
    description, units, and source page. This context is injected into the
    LLM enrichment prompt to reduce hallucination and improve quality.
    """
    from imas_codex.graph import GraphClient

    context: dict[str, dict[str, str]] = {}

    try:
        with GraphClient() as gc:
            # Phase 1: Use pre-extracted path fields for structured lookup.
            # These fields were populated during wiki ingestion and contain
            # parsed MDSplus/PPF/IMAS paths per chunk.
            results = gc.query(
                """
                MATCH (c:WikiChunk)-[:HAS_CHUNK]-(p:WikiPage)-[:AT_FACILITY]->(f:Facility {id: $facility})
                WHERE size(c.mdsplus_paths_mentioned) > 0
                   OR size(c.ppf_paths_mentioned) > 0
                   OR size(c.imas_paths_mentioned) > 0
                RETURN c.content AS content,
                       c.mdsplus_paths_mentioned AS mdsplus_paths,
                       c.ppf_paths_mentioned AS ppf_paths,
                       c.imas_paths_mentioned AS imas_paths,
                       c.units_mentioned AS units_mentioned,
                       p.title AS page_title
                """,
                facility=facility,
            )

            for row in results:
                content = row.get("content", "")
                page_title = row.get("page_title", "")
                if not content:
                    continue

                # Process pre-extracted MDSplus paths
                mdsplus_paths = row.get("mdsplus_paths") or []
                for path in mdsplus_paths:
                    path_key = path.upper()
                    if path_key in context:
                        continue
                    # Extract name from path: \TREE::NODE:LEAF -> LEAF
                    name = ""
                    if "::" in path:
                        after_tree = path.split("::", 1)[1]
                        segments = re.split(r"[.:]", after_tree)
                        name = segments[-1] if segments else ""
                    tree = (
                        path.split("::")[0].lstrip("\\").upper() if "::" in path else ""
                    )
                    context[path_key] = {
                        "description": "",
                        "units": "",
                        "page": page_title,
                        "tree": tree,
                        "name": name,
                    }

                # Process pre-extracted PPF paths (JET: DDA/DTYPE format)
                ppf_paths = row.get("ppf_paths") or []
                for path in ppf_paths:
                    path_key = path.upper()
                    if path_key in context:
                        continue
                    parts = path.split("/")
                    name = parts[-1] if parts else path
                    context[path_key] = {
                        "description": "",
                        "units": "",
                        "page": page_title,
                        "tree": parts[0] if len(parts) > 1 else "",
                        "name": name,
                    }

                # Process pre-extracted IMAS paths
                imas_paths = row.get("imas_paths") or []
                for path in imas_paths:
                    path_key = path.upper()
                    if path_key in context:
                        continue
                    segments = path.split("/")
                    name = segments[-1] if segments else path
                    context[path_key] = {
                        "description": "",
                        "units": "",
                        "page": page_title,
                        "tree": "",
                        "name": name,
                    }

                # Phase 2: Parse content to extract descriptions/units
                # for paths that appear in this chunk's text
                extracted = _extract_signals_from_chunk(content, facility, page_title)
                for sig in extracted:
                    path_key = sig["path"].upper()
                    existing = context.get(path_key, {})
                    # Prefer entries with more metadata
                    if (
                        not existing
                        or (sig["description"] and not existing.get("description"))
                        or (sig["units"] and not existing.get("units"))
                    ):
                        context[path_key] = {
                            "description": sig.get("description", ""),
                            "units": sig.get("units", ""),
                            "page": sig.get("page", ""),
                            "tree": sig.get("tree", ""),
                            "name": sig.get("name", ""),
                        }

            logger.info(
                "Wiki context: extracted %d signal definitions for %s from %d chunks",
                len(context),
                facility,
                len(results),
            )

    except Exception as e:
        logger.warning("Failed to build wiki context for %s: %s", facility, e)

    return context


class WikiScanner:
    """Extract signal definitions from wiki content in the knowledge graph.

    This scanner queries WikiChunk nodes for structured signal documentation
    (MDSplus path tables, signal catalogs, diagnostic node listings) and
    creates FacilitySignal nodes with pre-populated descriptions and units.

    It also builds a wiki_context dict that other scanners' enrichment
    pipelines can use to inject high-quality metadata into LLM prompts,
    reducing cost and improving accuracy.

    This scanner has no facility-specific logic — it works for any facility
    whose wiki content has been ingested into the graph.
    """

    scanner_type: str = "wiki"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Extract signals from wiki content in the graph.

        Queries WikiChunk nodes containing MDSplus paths, parses structured
        tables, and creates FacilitySignal nodes with wiki-sourced metadata.

        Also builds wiki_context for injection into other scanners' enrichment.
        """
        import asyncio

        # Build wiki context (runs graph queries)
        wiki_context = await asyncio.to_thread(_build_wiki_context, facility)

        if not wiki_context:
            logger.info(
                "Wiki scanner: no signal definitions found in wiki for %s",
                facility,
            )
            return ScanResult(
                wiki_context={},
                stats={
                    "signals_discovered": 0,
                    "wiki_paths_found": 0,
                    "note": "No wiki signal content found",
                },
            )

        # Create FacilitySignal nodes for well-documented wiki signals
        signals = []
        for path_key, info in wiki_context.items():
            name = info.get("name", "")
            if not name:
                continue

            tree = info.get("tree", "").lower()
            description = info.get("description", "")
            units = info.get("units", "")

            # Only create signals for entries with meaningful metadata
            if not description and not units:
                continue

            signal_id = f"{facility}:general/{tree}_{name.lower()}"
            accessor = f"data({path_key})"

            signals.append(
                FacilitySignal(
                    id=signal_id,
                    facility_id=facility,
                    status=FacilitySignalStatus.discovered,
                    physics_domain="general",  # Enriched by LLM
                    name=name,
                    accessor=accessor,
                    data_access=f"{facility}:mdsplus:tree",
                    tree_name=tree,
                    node_path=path_key,
                    units=units,
                    description=description,
                    discovery_source="wiki_extraction",
                )
            )

        logger.info(
            "Wiki scanner: %d signals from wiki (%d total wiki paths) for %s",
            len(signals),
            len(wiki_context),
            facility,
        )

        return ScanResult(
            signals=signals,
            wiki_context=wiki_context,
            stats={
                "signals_discovered": len(signals),
                "wiki_paths_found": len(wiki_context),
                "signals_with_description": sum(
                    1 for v in wiki_context.values() if v.get("description")
                ),
                "signals_with_units": sum(
                    1 for v in wiki_context.values() if v.get("units")
                ),
            },
        )

    async def check(
        self,
        facility: str,
        ssh_host: str,
        signals: list[FacilitySignal],
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> list[dict[str, Any]]:
        """Wiki-discovered signals are validated by other scanner checks.

        The wiki scanner only provides metadata — actual data access
        validation is handled by the MDSplus, TDI, PPF, or EDAS scanners
        since they know the access protocol.
        """
        return [
            {
                "signal_id": s.id,
                "valid": True,
                "note": "Wiki-sourced signal; validate via primary scanner",
            }
            for s in signals
        ]


# Auto-register on import
register_scanner(WikiScanner())
