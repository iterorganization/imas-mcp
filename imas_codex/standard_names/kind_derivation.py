"""Auto-derive ``kind`` from the standard name string (D5/P0.3).

Deterministic pattern-match overriding the LLM's ``kind`` field.
The LLM defaults to ``scalar`` for everything; this module inspects
the name tokens to assign the structurally correct ``StandardNameKind``.

All returned values are validated against the LinkML ``StandardNameKind``
enum at import time.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Valid Kind enum values (imported lazily to avoid circular imports at
# module level; validated once on first call).
_VALID_KINDS: frozenset[str] | None = None


def _load_valid_kinds() -> frozenset[str]:
    """Load the StandardNameKind enum values from generated models.

    Always unions with the full fallback set so a stale/partial enum load
    (e.g. a long-running process holding an old ``models.py``) cannot
    silently disable pattern rules.
    """
    global _VALID_KINDS
    if _VALID_KINDS is not None:
        return _VALID_KINDS
    fallback = frozenset(
        {
            "scalar",
            "vector",
            "vector_component",
            "tensor",
            "tensor_component",
            "eigenfunction",
            "spectrum",
            "complex_part",
            "metadata",
        }
    )
    try:
        from imas_codex.graph.models import StandardNameKind

        loaded = frozenset(e.value for e in StandardNameKind)
        _VALID_KINDS = loaded | fallback
    except Exception:
        _VALID_KINDS = fallback
    return _VALID_KINDS


def derive_kind(name: str) -> str:
    """Return the most specific ``StandardNameKind`` value for *name*.

    Pattern rules (evaluated in order — first match wins):

    1. ``_component_of_`` → ``vector_component``
    2. ``_tensor`` token (e.g. ``metric_tensor``, ``stress_tensor``) →
       ``tensor_component``
    3. ``_eigenfunction`` → ``eigenfunction``
    4. endswith ``_spectrum`` → ``spectrum``
    5. ``real_part`` or ``imaginary_part`` → ``complex_part``
    6. default → ``scalar``
    """
    valid = _load_valid_kinds()
    name_lower = name.lower()

    # 1. Vector component
    if "_component_of_" in name_lower:
        if "vector_component" in valid:
            return "vector_component"
        if "vector" in valid:
            return "vector"

    # 2. Tensor
    # Match tokens like _tensor_, _tensor (end of name), but NOT
    # names that merely mention tensor in a qualifier
    if "_tensor" in name_lower:
        # Check it's a real tensor reference (not e.g. "tensor_product_of_...")
        # by verifying _tensor is at the end or followed by _
        import re

        if re.search(r"_tensor(?:_|$)", name_lower):
            if "tensor_component" in valid:
                return "tensor_component"
            if "tensor" in valid:
                return "tensor"

    # 3. Eigenfunction
    if "_eigenfunction" in name_lower or name_lower == "eigenfunction":
        if "eigenfunction" in valid:
            return "eigenfunction"

    # 4. Spectrum
    if name_lower.endswith("_spectrum"):
        if "spectrum" in valid:
            return "spectrum"

    # 5. Complex part (real/imaginary)
    if "real_part" in name_lower or "imaginary_part" in name_lower:
        if "complex_part" in valid:
            return "complex_part"

    # 6. Default
    return "scalar"


# Mapping from extended local kinds → ISN's 3-kind discriminator.
# The ISN library (imas_standard_names) currently only validates
# {scalar, vector, metadata}.  We retain the richer taxonomy in our
# graph for filtering/search but collapse to ISN's vocabulary when
# constructing the validation model via ``create_standard_name_entry``.
_ISN_KIND_MAP: dict[str, str] = {
    "scalar": "scalar",
    "vector": "vector",
    "vector_component": "scalar",  # one component is a scalar field
    "tensor": "scalar",
    "tensor_component": "scalar",  # one component is a scalar field
    "eigenfunction": "scalar",
    "spectrum": "scalar",
    "complex_part": "scalar",
    "metadata": "metadata",
}


def to_isn_kind(kind: str | None) -> str:
    """Map a local extended kind value to one ISN's discriminator accepts.

    Defaults to ``scalar`` for unknown values so validation never crashes
    on a kind the ISN library doesn't recognise.
    """
    if not kind:
        return "scalar"
    return _ISN_KIND_MAP.get(kind, "scalar")
