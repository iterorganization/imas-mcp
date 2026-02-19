"""Shared IMAS path detection and extraction.

Canonical module for detecting and normalizing IMAS Data Dictionary paths
across the entire discovery pipeline. All discovery tools (paths, files,
wiki, signals) and the ingestion pipeline import from here.

Supports paths in multiple notations:
  - Slash-separated: equilibrium/time_slice/profiles_1d/psi
  - Dot-separated:   equilibrium.time_slice.profiles_1d.psi
  - With indices:    equilibrium/time_slice[:]/profiles_1d/psi
  - With prefix:     ids.equilibrium.global_quantities.ip
  - Array indices:   core_profiles/profiles_1d[0]/electrons/temperature

Index expressions ([0], [:], [i], etc.) are stripped during normalization
since the DD schema does not include them — they are runtime access notation.
"""

from __future__ import annotations

import functools
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "get_all_ids_names",
    "build_imas_pattern",
    "extract_imas_paths",
    "normalize_imas_path",
    "build_imas_rg_pattern",
]


# =============================================================================
# IDS Name Discovery
# =============================================================================


@functools.cache
def get_all_ids_names() -> tuple[str, ...]:
    """Get all IDS names from the data dictionary.

    Reads the detailed schema directory to discover all available IDS names.
    Cached after first call. Falls back to a comprehensive hardcoded list
    if the DD resources are not available.

    Returns:
        Tuple of all IDS names (sorted, lowercase)
    """
    try:
        from imas_codex.settings import get_dd_version

        dd_version = get_dd_version()
        resources_dir = (
            Path(__file__).parents[2]
            / "resources"
            / "imas_data_dictionary"
            / dd_version
            / "schemas"
            / "detailed"
        )
        if resources_dir.is_dir():
            names = sorted(p.stem for p in resources_dir.glob("*.json"))
            if names:
                return tuple(names)
    except Exception:
        pass

    # Comprehensive fallback list (DD 4.1.0, 82 IDS)
    return _FALLBACK_IDS_NAMES


# fmt: off
_FALLBACK_IDS_NAMES: tuple[str, ...] = (
    "amns_data",
    "b_field_non_axisymmetric",
    "balance_of_plant",
    "barometry",
    "bolometer",
    "breeding_blanket",
    "bremsstrahlung_visible",
    "calorimetry",
    "camera_ir",
    "camera_visible",
    "camera_x_rays",
    "charge_exchange",
    "coils_non_axisymmetric",
    "controllers",
    "core_instant_changes",
    "core_profiles",
    "core_sources",
    "core_transport",
    "cryostat",
    "dataset_fair",
    "disruption",
    "distribution_sources",
    "distributions",
    "divertors",
    "ec_launchers",
    "ece",
    "edge_profiles",
    "edge_sources",
    "edge_transport",
    "em_coupling",
    "equilibrium",
    "ferritic",
    "focs",
    "gas_injection",
    "gas_pumping",
    "gyrokinetics_local",
    "hard_x_rays",
    "ic_antennas",
    "interferometer",
    "iron_core",
    "langmuir_probes",
    "lh_antennas",
    "magnetics",
    "mhd",
    "mhd_linear",
    "mse",
    "nbi",
    "neutron_diagnostic",
    "ntms",
    "operational_instrumentation",
    "pellets",
    "pf_active",
    "pf_passive",
    "pf_plasma",
    "plasma_initiation",
    "plasma_profiles",
    "plasma_sources",
    "plasma_transport",
    "polarimeter",
    "pulse_schedule",
    "radiation",
    "real_time_data",
    "reflectometer_fluctuation",
    "reflectometer_profile",
    "refractometer",
    "runaway_electrons",
    "sawteeth",
    "soft_x_rays",
    "spectrometer_mass",
    "spectrometer_uv",
    "spectrometer_visible",
    "spectrometer_x_ray_crystal",
    "spi",
    "summary",
    "temporary",
    "tf",
    "thomson_scattering",
    "transport_solver_numerics",
    "turbulence",
    "vacuum",
    "wall",
    "waves",
)
# fmt: on

# Regex to strip array index expressions: [0], [:], [i], [1:N], etc.
_INDEX_STRIP_RE = re.compile(r"\[[^\]]*\]")


# =============================================================================
# IMAS Path Pattern
# =============================================================================


@functools.cache
def build_imas_pattern() -> re.Pattern[str]:
    """Build IMAS path regex from all known IDS names.

    Matches IMAS DD paths in multiple notations:
      - equilibrium.time_slice[0].profiles_1d.psi
      - core_profiles/profiles_1d/electrons/temperature
      - ids.equilibrium.global_quantities.ip
      - equilibrium/time_slice[:]/profiles_1d/psi

    The character class allows:
      - a-z, 0-9, _ : path segment characters
      - . / : path separators
      - [ ] : : array index notation (stripped during normalization)

    Uses non-capturing group so findall() returns the full match.
    """
    ids_names = get_all_ids_names()
    ids_alternation = "|".join(re.escape(name) for name in ids_names)
    pattern = rf"\b(?:ids\.)?(?:{ids_alternation})[./][a-z_0-9\[\]:./]+"
    return re.compile(pattern, re.IGNORECASE)


def normalize_imas_path(raw: str) -> str:
    """Normalize an IMAS path to canonical form.

    Canonical form:
      - Lowercase
      - Slash-separated (dots converted to slashes)
      - No ``ids/`` prefix
      - No array index expressions (``[0]``, ``[:]``, ``[i]`` stripped)
      - No trailing slashes

    Examples:
        >>> normalize_imas_path("ids.equilibrium.global_quantities.ip")
        'equilibrium/global_quantities/ip'
        >>> normalize_imas_path("equilibrium/time_slice[:]/profiles_1d/psi")
        'equilibrium/time_slice/profiles_1d/psi'
        >>> normalize_imas_path("Core_Profiles.Profiles_1D[0].Electrons.Temperature")
        'core_profiles/profiles_1d/electrons/temperature'
    """
    path = raw.lower()
    # Strip index expressions before separator conversion
    path = _INDEX_STRIP_RE.sub("", path)
    # Convert dot separators to slashes
    path = path.replace(".", "/")
    # Strip leading ids/ prefix
    if path.startswith("ids/"):
        path = path[4:]
    # Clean up any double slashes or trailing slash
    while "//" in path:
        path = path.replace("//", "/")
    return path.rstrip("/")


def extract_imas_paths(text: str, pattern: re.Pattern[str] | None = None) -> list[str]:
    """Extract IMAS data dictionary paths from text.

    Uses the full DD IDS name list (82+ IDS) to build a regex that
    matches paths starting with any known IDS name followed by
    sub-path segments. Index expressions like ``[:]`` and ``[0]``
    are accepted in the input and stripped during normalization.

    Args:
        text: Raw text content (code, documentation, wiki markup, etc.)
        pattern: Pre-compiled IMAS pattern (uses cached default if None)

    Returns:
        Deduplicated, sorted list of normalized IMAS paths found
    """
    if not text:
        return []

    if pattern is None:
        pattern = build_imas_pattern()

    matches = pattern.findall(text)
    normalized = {normalize_imas_path(m) for m in matches}
    return sorted(normalized)


def extract_ids_names(text: str) -> set[str]:
    """Extract IDS names referenced in text.

    Detects IDS names via common code patterns:
      - ``ids_factory.new("equilibrium")``
      - ``factory.equilibrium()``
      - String literals matching known IDS names

    Args:
        text: Text to scan (code, documentation, etc.)

    Returns:
        Set of lowercase IDS names found
    """
    known = frozenset(get_all_ids_names())
    found: set[str] = set()

    # Patterns for code-level IDS references
    code_patterns = [
        r'\.new\(["\'](\w+)["\']\)',
        r"factory\.(\w+)\(\)",
        r'["\'](\w+)["\']',
    ]

    for pat in code_patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            candidate = match.group(1).lower()
            if candidate in known:
                found.add(candidate)
    return found


# =============================================================================
# Ripgrep Pattern for Remote Enrichment
# =============================================================================


def build_imas_rg_pattern() -> str:
    """Build a ripgrep-compatible regex for IMAS path detection.

    Returns a pattern suitable for ``rg -e <pattern>`` on remote hosts
    during path enrichment. This detects IMAS DD path references in code
    (e.g., ``equilibrium/time_slice`` or ``core_profiles.profiles_1d``).

    The pattern matches IDS-rooted multi-segment paths while avoiding
    false positives from single-word IDS name mentions.
    """
    ids_names = get_all_ids_names()
    # Use the most common/important IDS names to keep the pattern manageable
    # rg handles alternation efficiently but very long patterns can hit ARG_MAX
    ids_alternation = "|".join(re.escape(name) for name in ids_names)
    return rf"\b({ids_alternation})[./\[]\S+"
