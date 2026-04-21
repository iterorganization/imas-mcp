"""Naming-scope classifier for DD paths.

Classifies IMAS Data Dictionary paths for standard name generation.

After Plan 30, the DD graph's ``node_category`` owns all semantic
classification (fit_artifact, representation, coordinate, structural,
error, metadata, identifier).  The SN extractor's ``SN_SOURCE_CATEGORIES``
filter ({quantity, geometry}) pre-excludes those categories, so this
classifier only handles a few SN-level policies that DD cannot express:

- **S0 string type defensive**: skip ``STR_*`` data types (name,
  description, etc.) in case they reach the pipeline.
- **S1 core_instant_changes**: IDS-level dedup — leaves ARE quantity,
  but peer IDSs expose the same concept at better granularity.
- **S2 error defensive**: catch ``_error_*`` suffixes in case the
  extractor's DD-level filter missed one.
- **S3 placeholder skip**: skip generic constant-value containers
  (e.g. ``constant_float_value``) that describe data types, not physics.

Returns ``"quantity"`` (proceed to SN extraction) or ``"skip"``
(do not extract).
"""

from __future__ import annotations

import re
from typing import Literal

# Type alias for the binary classification.
Scope = Literal["quantity", "skip"]

#: Suffixes in path segments that indicate error companion fields.
ERROR_SUFFIXES: tuple[str, ...] = ("_error_upper", "_error_lower", "_error_index")

#: Pattern matching placeholder / generic-container leaf names that
#: describe data types rather than physics quantities.
_PLACEHOLDER_RE = re.compile(
    r"^constant_(float|integer|boolean|string)_value$"
    r"|^generic_(float|integer)$"
)


def classify_path(node: dict) -> Scope:
    """Classify a DD path for standard name generation.

    Args:
        node: Dict with keys from the enriched DD query:

            - **path** (*str*): Full DD path
            - **data_type** (*str*): DD data type (e.g. ``STR_0D``)
            - (other keys are accepted but not used)

    Returns:
        ``"quantity"`` – proceed to StandardName extraction.
        ``"skip"`` – do not extract.
    """
    path: str = node.get("path", "")
    data_type: str = node.get("data_type", "")

    # ------------------------------------------------------------------
    # S0: String-typed leaves → skip (defensive — names can never be
    #     standard names; normally pre-filtered by DD node_category).
    # ------------------------------------------------------------------
    if data_type and data_type.startswith("STR"):
        return "skip"

    # ------------------------------------------------------------------
    # S1: Entire event-delta IDS → skip (dedup policy, not DD fact).
    #
    # ``core_instant_changes`` records before/after snapshots during
    # transient events (ELM, pellet, sawtooth, MHD).  Its leaves are
    # copies of ``core_profiles`` quantities prefixed with "change in X".
    # Minting ``change_in_*`` StandardNames duplicates the core_profiles
    # vocabulary and forces contrived grammar.  Codes consuming this IDS
    # reuse the underlying core_profiles StandardNames via path linkage.
    # ------------------------------------------------------------------
    if path.startswith("core_instant_changes/") or path == "core_instant_changes":
        return "skip"

    # ------------------------------------------------------------------
    # S2: Error fields → skip (defensive — normally pre-filtered by DD).
    # ------------------------------------------------------------------
    if _is_error_field(path):
        return "skip"

    # ------------------------------------------------------------------
    # S3: Placeholder / generic-container leaves → skip.
    #
    # Paths like ``summary/local/parameter/*/value`` store typed
    # constants (``constant_float_value``, ``constant_integer_value``).
    # These describe data containers, not measurable physics quantities,
    # and should never become standard names.
    # ------------------------------------------------------------------
    if _is_placeholder(path):
        return "skip"

    return "quantity"


def _is_error_field(path: str) -> bool:
    """Return True if *path* contains an error-field suffix."""
    return any(suffix in path for suffix in ERROR_SUFFIXES)


def _is_placeholder(path: str) -> bool:
    """Return True if the leaf segment is a generic container name."""
    leaf = path.rsplit("/", 1)[-1]
    return bool(_PLACEHOLDER_RE.match(leaf))
