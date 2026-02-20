"""Facility-aware entity extraction from text.

Extracts data paths, tool mentions, units, and conventions from wiki text,
code comments, and OCR output. Patterns are selected based on the facility's
configured data_systems from facility config (DataAccessPatternsConfig).

MDSplus paths are only extracted for facilities with 'mdsplus' or 'tdi'.
IMAS paths are always extracted (universal target format for all facilities).
PPF paths are extracted for facilities with 'ppf' in data_systems.
Tool mentions use the facility's key_tools and code_import_patterns.

Usage:
    extractor = FacilityEntityExtractor("tcv")
    result = extractor.extract("Signal \\\\ATLAS::IP measured in MA")
    print(result.mdsplus_paths)  # ['\\\\ATLAS::IP']
    print(result.units)          # ['MA']
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from imas_codex.discovery.base.imas_patterns import (
    build_imas_pattern as _build_imas_pattern,
    extract_imas_paths as _extract_imas_paths_shared,
    get_all_ids_names,  # noqa: F401 – re-exported for backwards compat
)

logger = logging.getLogger(__name__)


# =============================================================================
# Extraction Result
# =============================================================================


@dataclass
class ExtractionResult:
    """Result of entity extraction from text.

    Each field holds a deduplicated, sorted list of matches for that
    entity type. Fields are only populated when the facility's data_systems
    include the relevant data system.
    """

    mdsplus_paths: list[str] = field(default_factory=list)
    imas_paths: list[str] = field(default_factory=list)
    ppf_paths: list[str] = field(default_factory=list)
    tool_mentions: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    conventions: list[dict[str, str]] = field(default_factory=list)


# get_all_ids_names is re-exported from shared module for backwards compat
# _build_imas_pattern is imported from shared module above


# =============================================================================
# Pattern Definitions
# =============================================================================

# MDSplus path patterns: \TREE::NODE or \\TREE::NODE:PATH
MDSPLUS_PATH_PATTERN = re.compile(
    r"\\\\?[A-Z_][A-Z_0-9]*::[A-Z_][A-Z_0-9:]*",
    re.IGNORECASE,
)

# PPF path patterns: DDA/Dtype (e.g., EFIT/Q95, KK3/TE, BOLO/KB5H)
# Uses capturing groups: (DDA, Dtype)
PPF_PATH_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9]{1,7})/([A-Z][A-Z0-9_]{0,9})\b",
)

# Physical units (common in fusion data)
UNIT_PATTERN = re.compile(
    r"\b(eV|keV|MeV|GeV|"  # Energy
    r"m\^?-?[123]|cm\^?-?[123]|mm|"  # Length/density
    r"Tesla|T|Wb|Weber|"  # Magnetic
    r"Amp(?:ere)?s?|A|MA|kA|"  # Current
    r"Ohm|ohm|Ω|"  # Resistance
    r"Volt|V|kV|"  # Voltage
    r"Watt|W|MW|kW|"  # Power
    r"m/s|m\.s\^-1|"  # Velocity
    r"rad|radian|degrees?|°|"  # Angle
    r"seconds?|s|ms|μs|ns|"  # Time
    r"Hz|kHz|MHz|GHz|"  # Frequency
    r"Pa|kPa|MPa|bar|mbar|Torr)\b",
    re.IGNORECASE,
)

# COCOS convention patterns
COCOS_PATTERN = re.compile(
    r"COCOS\s*(\d{1,2})|"
    r"cocos\s*=\s*(\d{1,2})|"
    r"coordinate\s+convention[s]?\s+(\d{1,2})",
    re.IGNORECASE,
)

# Sign convention patterns
SIGN_CONVENTION_PATTERN = re.compile(
    r"(positive|negative)\s+(clockwise|counter-?clockwise|outward|inward|"
    r"upward|downward|toroidal|poloidal|radial)|"
    r"(sign\s+convention)|"
    r"(I_?[pP]\s*[><]=?\s*0)|"
    r"(B_?[tTφ]\s*[><]=?\s*0)",
    re.IGNORECASE,
)

# Known JET PPF DDAs for validation (reduces false positives).
# This set can be expanded as new DDAs are encountered.
_JET_KNOWN_DDAS = frozenset(
    {
        "BOLO",
        "CHAIN",
        "CO2I",
        "CXFM",
        "CXRS",
        "ECE",
        "EDGP",
        "EFIT",
        "EHTR",
        "FAST",
        "HRTS",
        "INTS",
        "KG1V",
        "KK3",
        "KK3H",
        "LIDR",
        "MAGN",
        "MSE",
        "NBI1",
        "NBI2",
        "ICRH",
        "LHCD",
        "PELC",
        "PPED",
        "PPFX",
        "SXR",
        "VIS",
    }
)


# =============================================================================
# FacilityEntityExtractor
# =============================================================================


class FacilityEntityExtractor:
    """Extract entities from text using facility-specific patterns.

    Loads the facility's data_systems and data_access_patterns config
    to determine which extraction patterns to apply. Creates compiled
    regex patterns once at construction and reuses across all calls.

    Args:
        facility_id: Facility identifier (e.g., 'tcv', 'jet', 'iter')

    Example:
        extractor = FacilityEntityExtractor("tcv")
        result = extractor.extract("Signal \\\\ATLAS::IP measured in MA")
        print(result.mdsplus_paths)  # ['\\\\ATLAS::IP']
    """

    def __init__(self, facility_id: str) -> None:
        self.facility_id = facility_id
        self._data_systems: list[str] = []
        self._key_tools: list[str] = []
        self._code_import_patterns: list[str] = []

        self._load_config()
        # Build IMAS pattern once (cached globally)
        self._imas_pattern = _build_imas_pattern()

    def _load_config(self) -> None:
        """Load facility config and set up extraction patterns."""
        try:
            from imas_codex.discovery.base.facility import get_facility

            config = get_facility(self.facility_id)

            self._data_systems = config.get("data_systems", [])

            dap = config.get("data_access_patterns")
            if dap:
                self._key_tools = dap.get("key_tools", [])
                self._code_import_patterns = dap.get("code_import_patterns", [])
        except Exception:
            logger.debug(
                "Could not load config for facility %s, using universal patterns only",
                self.facility_id,
            )

    @property
    def has_mdsplus(self) -> bool:
        """Whether this facility uses MDSplus."""
        return "mdsplus" in self._data_systems or "tdi" in self._data_systems

    @property
    def has_ppf(self) -> bool:
        """Whether this facility uses PPF (JET)."""
        return "ppf" in self._data_systems

    def extract(self, text: str) -> ExtractionResult:
        """Extract all entities from text.

        Applies extraction patterns based on facility data_systems:
        - MDSplus paths: only if facility has 'mdsplus' or 'tdi'
        - IMAS paths: always (universal target format)
        - PPF paths: only if facility has 'ppf'
        - Tool mentions: from facility key_tools/code_import_patterns
        - Units and conventions: always (universal)

        Args:
            text: Raw text content

        Returns:
            ExtractionResult with categorized entities
        """
        if not text:
            return ExtractionResult()

        result = ExtractionResult()

        # Facility-conditional extraction
        if self.has_mdsplus:
            result.mdsplus_paths = extract_mdsplus_paths(text)

        if self.has_ppf:
            result.ppf_paths = extract_ppf_paths(text)

        # Universal extraction
        result.imas_paths = extract_imas_paths(text, self._imas_pattern)
        result.units = extract_units(text)
        result.conventions = extract_conventions(text)
        result.tool_mentions = extract_facility_tool_mentions(
            text, self._key_tools or None, self._code_import_patterns or None
        )

        return result

    def to_chunk_properties(self, result: ExtractionResult) -> dict:
        """Convert extraction result to graph chunk properties.

        Returns a dict with keys matching the WikiChunk graph schema
        property names used in Cypher UNWIND queries.
        """
        return {
            "mdsplus_paths": result.mdsplus_paths,
            "imas_paths": result.imas_paths,
            "ppf_paths": result.ppf_paths,
            "units": result.units,
            "conventions": [c.get("name", "") for c in result.conventions],
            "tool_mentions": result.tool_mentions,
        }


# =============================================================================
# Standalone Extraction Functions
# =============================================================================


def extract_mdsplus_paths(text: str) -> list[str]:
    """Extract MDSplus paths from text.

    Only meaningful for facilities with MDSplus/TDI data systems.

    Args:
        text: Raw text content

    Returns:
        Deduplicated list of MDSplus paths found
    """
    matches = MDSPLUS_PATH_PATTERN.findall(text)
    # Normalize: ensure single backslash prefix, uppercase
    normalized = set()
    for m in matches:
        path = m.lstrip("\\")
        path = "\\" + path.upper()
        normalized.add(path)
    return sorted(normalized)


def extract_imas_paths(text: str, pattern: re.Pattern | None = None) -> list[str]:
    """Extract IMAS data dictionary paths from text.

    Delegates to the shared ``imas_codex.discovery.base.imas_patterns``
    module which handles index variable notation (``[:]``, ``[0]``, etc.)
    and normalizes paths consistently across the entire pipeline.

    Args:
        text: Raw text content
        pattern: Pre-compiled IMAS pattern (uses cached default if None)

    Returns:
        Deduplicated list of IMAS paths found (normalized to lowercase with / separators)
    """
    return _extract_imas_paths_shared(text, pattern=pattern)


def extract_ppf_paths(text: str) -> list[str]:
    """Extract JET PPF DDA/Dtype paths from text.

    Matches patterns like EFIT/Q95, KK3/TE, BOLO/KB5H.
    Validates DDA component against known JET DDAs to reduce
    false positives from generic UPPERCASE/UPPERCASE patterns.

    Args:
        text: Raw text content

    Returns:
        Deduplicated list of PPF paths (DDA/DTYPE format)
    """
    matches = PPF_PATH_PATTERN.findall(text)
    paths = set()
    for dda, dtype in matches:
        if dda.upper() in _JET_KNOWN_DDAS:
            paths.add(f"{dda.upper()}/{dtype.upper()}")
    return sorted(paths)


def extract_units(text: str) -> list[str]:
    """Extract physical units from text.

    Args:
        text: Raw text content

    Returns:
        Deduplicated list of unit symbols found
    """
    matches = UNIT_PATTERN.findall(text)
    return sorted(set(matches))


def extract_conventions(text: str) -> list[dict[str, str]]:
    """Extract sign and coordinate conventions from text.

    Args:
        text: Raw text content

    Returns:
        List of convention dicts with type, name, and description
    """
    conventions = []

    # Find COCOS references
    for match in COCOS_PATTERN.finditer(text):
        cocos_num = match.group(1) or match.group(2) or match.group(3)
        if cocos_num:
            conventions.append(
                {
                    "type": "cocos",
                    "name": f"COCOS {cocos_num}",
                    "cocos_index": int(cocos_num),
                    "context": text[max(0, match.start() - 50) : match.end() + 50],
                }
            )

    # Find sign conventions
    for match in SIGN_CONVENTION_PATTERN.finditer(text):
        matched_text = match.group(0)
        conventions.append(
            {
                "type": "sign",
                "name": matched_text.strip(),
                "context": text[max(0, match.start() - 50) : match.end() + 50],
            }
        )

    return conventions


def extract_facility_tool_mentions(
    text: str,
    key_tools: list[str] | None = None,
    code_import_patterns: list[str] | None = None,
) -> list[str]:
    """Extract facility-specific tool/API mentions from text.

    Matches against the facility's configured key_tools and
    code_import_patterns from DataAccessPatternsConfig. This
    complements the structural path extraction with facility-specific
    tool recognition.

    Args:
        text: Raw text content (chunk text, OCR text, etc.)
        key_tools: Facility key_tools list (e.g., ['ppfget', 'tdiExecute'])
        code_import_patterns: Facility code patterns (e.g., ['import ppf'])

    Returns:
        Deduplicated list of matched tool/pattern names found in text
    """
    if not text:
        return []

    mentions: set[str] = set()
    text_lower = text.lower()

    # Match key_tools (case-insensitive word boundary match)
    for tool in key_tools or []:
        # Normalize tool name for matching (strip parens, dots for search)
        tool_clean = tool.rstrip("(").split(".")[-1]
        if len(tool_clean) < 2:
            continue
        # Use word boundary matching to avoid false positives
        # e.g., "sal" should match "import sal" but not "universal"
        tool_pattern = re.compile(r"\b" + re.escape(tool_clean) + r"\b", re.IGNORECASE)
        if tool_pattern.search(text):
            mentions.add(tool)

    # Match code_import_patterns (substring match)
    for pattern in code_import_patterns or []:
        if pattern.lower() in text_lower:
            mentions.add(pattern)

    return sorted(mentions)
