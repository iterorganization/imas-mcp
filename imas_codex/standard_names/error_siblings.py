"""Deterministic error-sibling minting for B9.

IMAS ``_error_upper`` / ``_error_lower`` / ``_error_index`` companion
fields inherit the parent's semantic — they never need a separate LLM
compose call.  Once the parent has a standard name ``P``, the three
siblings are:

    upper_uncertainty_of_<P>
    lower_uncertainty_of_<P>
    uncertainty_index_of_<P>

Each sibling:
- Inherits the parent's unit via HAS_UNIT
- Links to its own error IMASNode via FROM_DD_PATH (source_id)
- Sets pipeline_status='named', validation_status='valid'
- Skips LLM compose + review (deterministic construction)
- Uses model='deterministic:dd_error_modifier' for provenance
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from imas_codex.standard_names.source_paths import encode_source_path

logger = logging.getLogger(__name__)

# Mapping from IMAS error-field suffix → ISN uncertainty operator
ERROR_SUFFIX_TO_OPERATOR: dict[str, str] = {
    "_error_upper": "upper_uncertainty",
    "_error_lower": "lower_uncertainty",
    "_error_index": "uncertainty_index",
}

# Units that carry no physical dimension — uncertainty *index* is meaningless
# for dimensionless quantities (the index discretises a continuous distribution,
# which only makes sense for dimensional measurables).
_DIMENSIONLESS_UNITS: frozenset[str] = frozenset({"", "1", "-", "none"})

# Name suffixes that indicate categorical / identifier fields.  These fields
# encode discrete labels or codes, not physical measurements, so an
# uncertainty_index sibling would be semantically invalid.
_CATEGORICAL_SUFFIXES: tuple[str, ...] = (
    "_status",
    "_type",
    "_index",
    "_id",
    "_label",
    "_flag",
    "_identifier",
    "_code",
    "_name",
)


def _parent_supports_uncertainty_index(parent_name: str, unit: str | None) -> bool:
    """Return True only when an ``uncertainty_index_of_<parent>`` sibling is
    semantically meaningful.

    Deny rules (any matching rule returns False):

    1. **Process attribution** – name contains ``_due_to_`` or ``caused_by_``:
       these are breakdown terms (e.g. ``power_due_to_thermalization``), not
       directly measured quantities.
    2. **Data-type descriptor prefix** – name starts with ``constant_`` or
       ``generic_``: these label generic DD nodes, not physical observables.
    3. **Categorical / identifier suffix** – name ends with one of
       ``_status``, ``_type``, ``_index``, ``_id``, ``_label``, ``_flag``,
       ``_identifier``, ``_code``, ``_name``: these encode discrete labels,
       not continuous measurements.
    4. **Dimensionless unit** – *unit* is ``None``, ``""``, ``"1"``, or
       ``"-"``: uncertainty discretisation has no meaning for dimensionless
       scalars (ratios, fractions, counters).

    All other parents are considered suitable and return True.
    """
    # Rule 1: process attribution patterns
    if "_due_to_" in parent_name or "caused_by_" in parent_name:
        return False

    # Rule 2: data-type descriptor prefixes
    if parent_name.startswith("constant_") or parent_name.startswith("generic_"):
        return False

    # Rule 3: categorical / identifier suffixes
    if parent_name.endswith(_CATEGORICAL_SUFFIXES):
        return False

    # Rule 4: dimensionless unit
    if unit is None or unit.strip().lower() in _DIMENSIONLESS_UNITS:
        return False

    return True


def _detect_error_suffix(error_node_id: str) -> str | None:
    """Detect which error suffix an IMASNode id carries.

    Returns the suffix string (e.g. ``"_error_upper"``) or ``None``
    if the id does not end with a known error suffix.
    """
    for suffix in ERROR_SUFFIX_TO_OPERATOR:
        if error_node_id.endswith(suffix):
            return suffix
    return None


def mint_error_siblings(
    parent_name: str,
    *,
    error_node_ids: list[str],
    unit: str | None,
    physics_domain: str | None,
    cocos_type: str | None,
    cocos_version: int | None,
    dd_version: str | None,
) -> list[dict[str, Any]]:
    """Build candidate dicts for error-sibling StandardNames.

    For each error node ID, determines the IMAS suffix, maps to the
    corresponding ISN uncertainty operator, builds the sibling name,
    validates it via ISN grammar round-trip, and returns a candidate
    dict ready for ``persist_composed_batch``.

    Invalid siblings (grammar parse failure) are logged and skipped.

    Parameters
    ----------
    parent_name:
        The standard name of the parent (e.g. ``"plasma_current"``).
    error_node_ids:
        List of error IMASNode IDs from ``candidate.error_node_ids``.
    unit:
        Parent's canonical unit string (inherited by siblings).
    physics_domain:
        Parent's physics domain (inherited by siblings).
    cocos_type:
        Parent's COCOS transformation type (inherited).
    cocos_version:
        COCOS convention integer (inherited).
    dd_version:
        DD version string (inherited).

    Returns
    -------
    list[dict]
        Candidate dicts ready for ``persist_composed_batch``, one per
        valid sibling. May be empty if all siblings fail validation.
    """
    siblings: list[dict[str, Any]] = []
    now = datetime.now(UTC).isoformat()
    error_siblings_skipped: int = 0

    for error_id in error_node_ids:
        suffix = _detect_error_suffix(error_id)
        if suffix is None:
            logger.warning(
                "Error node %r does not end with a known error suffix — skipping",
                error_id,
            )
            continue

        # Semantic gate: skip uncertainty_index siblings when the parent is
        # a process term, categorical field, or dimensionless quantity.
        # upper/lower uncertainty bounds are always valid and are not gated.
        if suffix == "_error_index" and not _parent_supports_uncertainty_index(
            parent_name, unit
        ):
            reason = (
                "dimensionless unit"
                if (unit is None or unit.strip().lower() in _DIMENSIONLESS_UNITS)
                else (
                    "process attribution"
                    if ("_due_to_" in parent_name or "caused_by_" in parent_name)
                    else (
                        "data-type descriptor prefix"
                        if parent_name.startswith(("constant_", "generic_"))
                        else "categorical/identifier suffix"
                    )
                )
            )
            logger.debug(
                "skipping uncertainty_index sibling for %r: %s",
                parent_name,
                reason,
            )
            error_siblings_skipped += 1
            continue

        operator = ERROR_SUFFIX_TO_OPERATOR[suffix]
        sibling_name = f"{operator}_of_{parent_name}"

        # Validate via ISN vNext grammar round-trip (strict)
        try:
            from imas_standard_names.grammar.parser import parse as vnext_parse
            from imas_standard_names.grammar.render import compose as vnext_compose

            result = vnext_parse(sibling_name)
            normalized = vnext_compose(result.ir)
            sibling_name = normalized
        except Exception as exc:
            logger.warning(
                "Error sibling %r failed ISN validation: %s — skipping",
                sibling_name,
                exc,
            )
            continue

        siblings.append(
            {
                "id": sibling_name,
                "source_types": ["dd"],
                "source_id": error_id,
                "kind": "scalar" if suffix == "_error_index" else None,
                "source_paths": [encode_source_path("dd", error_id)],
                "confidence": 1.0,
                "unit": unit,
                "physics_domain": physics_domain,
                "cocos_transformation_type": cocos_type,
                "cocos": cocos_version,
                "dd_version": dd_version,
                # Provenance: deterministic, not LLM
                "model": "deterministic:dd_error_modifier",
                "pipeline_status": "named",
                "validation_status": "valid",
                "generated_at": now,
                # Mark review phases as complete (deterministic names
                # pass trivially — don't waste LLM calls reviewing them)
                "reviewer_score_name": 1.0,
                "reviewed_name_at": now,
                "reviewer_score_docs": 1.0,
                "reviewed_docs_at": now,
            }
        )

    if error_siblings_skipped:
        logger.debug(
            "mint_error_siblings: skipped %d uncertainty_index sibling(s) "
            "for parent %r (semantic gate)",
            error_siblings_skipped,
            parent_name,
        )

    return siblings
