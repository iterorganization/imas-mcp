"""Seed the graph with reference standard names from external sources.

Supports two sources:
- **ISN examples**: Shipped with ``imas-standard-names``, 42 entries
  covering core physics domains. Imported as ``review_status='accepted'``
  and ``source_type='reference'``.
- **WEST catalog**: ~305 entries from the ``west-standard-names`` repo.
  Requires two fixes (add ``physics_domain``, strip primary tags) before
  ISN validation. Imported as ``review_status='drafted'`` and
  ``source_type='west'``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# WEST directory-name → physics_domain mapping
# =============================================================================

DIR_TO_DOMAIN: dict[str, str] = {
    "coils-and-control": "magnetic_field_systems",
    "core-physics": "core_plasma_physics",
    "data-products": "data_management",
    "ec-heating": "auxiliary_heating",
    "edge-physics": "edge_plasma_physics",
    "equilibrium": "equilibrium",
    "fast-particles": "fast_particles",
    "fueling": "fueling",
    "fundamental": "general",
    "ic-heating": "auxiliary_heating",
    "imaging": "plasma_measurement_diagnostics",
    "interferometry": "electromagnetic_wave_diagnostics",
    "lh-heating": "auxiliary_heating",
    "magnetics": "magnetic_field_diagnostics",
    "mhd": "magnetohydrodynamics",
    "nbi": "auxiliary_heating",
    "neutronics": "neutronics",
    "radiation-diagnostics": "radiation_measurement_diagnostics",
    "reflectometry": "electromagnetic_wave_diagnostics",
    "spectroscopy": "spectroscopy",
    "thomson-scattering": "plasma_measurement_diagnostics",
    "transport": "transport",
    "turbulence": "turbulence",
}

# Tags that correspond to directory names (primary classification).
# These must be stripped from the ``tags`` list because ISN validation
# only allows secondary tags defined in ``grammar/vocabularies/tags.yml``.
PRIMARY_TAGS: set[str] = {
    "coils-and-control",
    "core-physics",
    "data-products",
    "ec-heating",
    "edge-physics",
    "equilibrium",
    "fast-particles",
    "fueling",
    "fundamental",
    "ic-heating",
    "imaging",
    "interferometry",
    "lh-heating",
    "magnetics",
    "mhd",
    "nbi",
    "neutronics",
    "radiation-diagnostics",
    "reflectometry",
    "spectroscopy",
    "thomson-scattering",
    "transport",
    "turbulence",
    "wall-and-structures",
    "pulse-management",
    "utilities",
    "waves",
    "runaway-electrons",
    "plasma-initiation",
}


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass
class SeedResult:
    """Summary of a seed operation."""

    loaded: int = 0
    validated: int = 0
    written: int = 0
    skipped: int = 0
    grammar_mismatches: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)


# =============================================================================
# Grammar helpers (reuse catalog_import pattern)
# =============================================================================


def _parse_grammar_fields(name: str) -> dict[str, str | None]:
    """Derive grammar fields from a standard name string.

    Returns a dict with keys expected by ``write_standard_names()``:
    ``physical_base``, ``subject``, ``component``, ``coordinate``,
    ``position``, ``process``.  Values are strings or ``None``.
    """
    try:
        from imas_standard_names.grammar import parse_standard_name

        parsed = parse_standard_name(name)
        return {
            "physical_base": (
                str(parsed.physical_base) if parsed.physical_base else None
            ),
            "subject": str(parsed.subject.value) if parsed.subject else None,
            "component": str(parsed.component.value) if parsed.component else None,
            "coordinate": str(parsed.coordinate.value) if parsed.coordinate else None,
            "position": str(parsed.position.value) if parsed.position else None,
            "process": str(parsed.process.value) if parsed.process else None,
        }
    except Exception:
        logger.debug("Grammar parse failed for name: %r", name)
        return {
            "physical_base": None,
            "subject": None,
            "component": None,
            "coordinate": None,
            "position": None,
            "process": None,
        }


def _check_grammar_roundtrip(name: str) -> str | None:
    """Return a mismatch message if name doesn't round-trip, else None."""
    try:
        from imas_standard_names.grammar import (
            compose_standard_name,
            parse_standard_name,
        )

        parsed = parse_standard_name(name)
        composed = compose_standard_name(parsed)
        if composed != name:
            return f"{name} → {composed}"
    except Exception as exc:
        return f"{name}: parse error ({exc})"
    return None


# =============================================================================
# Entry builders
# =============================================================================


def _entry_to_graph_dict(
    data: dict[str, Any],
    *,
    review_status: str,
    source_type: str,
) -> dict[str, Any]:
    """Convert a validated catalog-style dict to a ``write_standard_names`` dict.

    Parameters
    ----------
    data:
        Dict with ``name``, ``description``, ``documentation``, ``kind``,
        ``unit``, ``tags``, ``physics_domain``, plus optional ``links``,
        ``ids_paths``, ``validity_domain``, ``constraints``.
    review_status:
        Graph review_status to assign.
    source_type:
        Graph source_type to assign (``'reference'`` or ``'west'``).
    """
    grammar = _parse_grammar_fields(data["name"])

    tags = data.get("tags") or []
    links = data.get("links") or []
    ids_paths = data.get("ids_paths") or []

    return {
        "id": data["name"],
        "source_type": source_type,
        # No source_id — these are external catalog entries, not DD/signal derived
        "description": data.get("description") or None,
        "documentation": data.get("documentation") or None,
        "kind": data.get("kind") or None,
        "unit": data.get("unit") or None,
        "tags": [str(t) for t in tags] if tags else None,
        "links": [str(lnk) for lnk in links] if links else None,
        "imas_paths": list(ids_paths) if ids_paths else None,
        "validity_domain": data.get("validity_domain") or None,
        "constraints": list(data["constraints"]) if data.get("constraints") else None,
        "physics_domain": data.get("physics_domain") or None,
        "review_status": review_status,
        # Grammar fields
        **grammar,
    }


# =============================================================================
# ISN reference examples
# =============================================================================


def load_isn_examples() -> tuple[list[dict[str, Any]], list[str]]:
    """Load ISN shipped reference examples and validate them.

    Returns
    -------
    tuple
        (valid_entries, validation_errors) where *valid_entries* is a list
        of dicts ready for ``write_standard_names()`` and *validation_errors*
        lists any entries that failed ISN validation.
    """
    import importlib.resources as ir

    import imas_standard_names
    import yaml
    from imas_standard_names.models import create_standard_name_entry
    from pydantic import ValidationError

    examples_pkg = (
        ir.files(imas_standard_names) / "resources" / "standard_name_examples"
    )
    examples_dir = Path(str(examples_pkg))

    if not examples_dir.exists():
        logger.warning("ISN examples directory not found: %s", examples_dir)
        return [], [f"ISN examples directory not found: {examples_dir}"]

    entries: list[dict[str, Any]] = []
    errors: list[str] = []

    for domain_dir in sorted(examples_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        for f in sorted(domain_dir.glob("*.yml")):
            data = yaml.safe_load(f.read_text())
            if not isinstance(data, dict):
                errors.append(f"{f.name}: not a YAML mapping")
                continue

            # Validate through ISN
            try:
                create_standard_name_entry(data)
            except (ValidationError, Exception) as exc:
                errors.append(f"{f.name}: {exc}")
                logger.debug("ISN validation failed for %s: %s", f.name, exc)
                continue

            entries.append(
                _entry_to_graph_dict(
                    data, review_status="accepted", source_type="reference"
                )
            )

    return entries, errors


def seed_isn_examples(dry_run: bool = False) -> SeedResult:
    """Import ISN shipped reference examples as accepted calibration anchors.

    Parameters
    ----------
    dry_run:
        If True, validate and count but don't write to graph.

    Returns
    -------
    SeedResult with import statistics.
    """
    result = SeedResult()
    entries, errors = load_isn_examples()
    result.loaded = len(entries) + len(errors)
    result.validated = len(entries)
    result.validation_errors = errors

    # Check grammar round-trips
    for entry in entries:
        mismatch = _check_grammar_roundtrip(entry["id"])
        if mismatch:
            result.grammar_mismatches.append(mismatch)

    if dry_run:
        result.written = 0
        logger.info(
            "ISN dry run: %d validated, %d errors", result.validated, len(errors)
        )
        return result

    if not entries:
        return result

    from imas_codex.sn.graph_ops import write_standard_names

    result.written = write_standard_names(entries)
    logger.info("ISN seed: wrote %d entries", result.written)
    return result


# =============================================================================
# WEST catalog
# =============================================================================

_WEST_DEFAULT_DIR = Path("~/Code/west-standard-names/standard_names").expanduser()


def _fix_west_entry(data: dict[str, Any], dir_name: str) -> dict[str, Any]:
    """Apply WEST→ISN migration fixes to a raw YAML dict.

    1. Add ``physics_domain`` from directory name mapping.
    2. Strip primary tags from the ``tags`` list, keeping only secondary tags.

    Returns a new dict (does not mutate the input).
    """
    fixed = dict(data)

    # Fix 1: physics_domain from directory name
    if not fixed.get("physics_domain"):
        fixed["physics_domain"] = DIR_TO_DOMAIN.get(dir_name, "general")

    # Fix 2: strip primary tags
    if fixed.get("tags"):
        fixed["tags"] = [t for t in fixed["tags"] if t not in PRIMARY_TAGS]

    return fixed


def load_west_catalog(
    west_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load WEST standard names with migration fixes and validate them.

    Parameters
    ----------
    west_dir:
        Path to the ``standard_names/`` directory inside the
        west-standard-names repo.

    Returns
    -------
    tuple
        (valid_entries, validation_errors) where *valid_entries* is a list
        of dicts ready for ``write_standard_names()``.
    """
    import yaml
    from imas_standard_names.models import create_standard_name_entry
    from pydantic import ValidationError

    base = west_dir or _WEST_DEFAULT_DIR

    if not base.exists():
        logger.warning("WEST catalog directory not found: %s", base)
        return [], [f"WEST catalog directory not found: {base}"]

    entries: list[dict[str, Any]] = []
    errors: list[str] = []

    for domain_dir in sorted(base.iterdir()):
        if not domain_dir.is_dir():
            continue
        dir_name = domain_dir.name
        for f in sorted(domain_dir.glob("*.yml")):
            raw = yaml.safe_load(f.read_text())
            if not isinstance(raw, dict):
                errors.append(f"{f.name}: not a YAML mapping")
                continue

            # Apply migration fixes
            fixed = _fix_west_entry(raw, dir_name)

            # Validate through ISN
            try:
                create_standard_name_entry(fixed)
            except (ValidationError, Exception) as exc:
                errors.append(f"{dir_name}/{f.name}: {exc}")
                logger.debug(
                    "WEST validation failed for %s/%s: %s", dir_name, f.name, exc
                )
                continue

            entries.append(
                _entry_to_graph_dict(fixed, review_status="drafted", source_type="west")
            )

    return entries, errors


def seed_west_catalog(
    west_dir: Path | None = None,
    dry_run: bool = False,
) -> SeedResult:
    """Import WEST standard names with physics_domain fix and tag cleanup.

    Parameters
    ----------
    west_dir:
        Path to the ``standard_names/`` directory inside the
        west-standard-names repo.
    dry_run:
        If True, validate and count but don't write to graph.

    Returns
    -------
    SeedResult with import statistics.
    """
    result = SeedResult()
    entries, errors = load_west_catalog(west_dir)
    result.loaded = len(entries) + len(errors)
    result.validated = len(entries)
    result.validation_errors = errors

    # Check grammar round-trips
    for entry in entries:
        mismatch = _check_grammar_roundtrip(entry["id"])
        if mismatch:
            result.grammar_mismatches.append(mismatch)

    if dry_run:
        result.written = 0
        logger.info(
            "WEST dry run: %d validated, %d errors", result.validated, len(errors)
        )
        return result

    if not entries:
        return result

    from imas_codex.sn.graph_ops import write_standard_names

    result.written = write_standard_names(entries)
    logger.info("WEST seed: wrote %d entries", result.written)
    return result
