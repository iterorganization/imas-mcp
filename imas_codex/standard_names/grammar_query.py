"""Plan 40 Phase 2 — query tokenisation and tier policy.

Pure, dependency-light helpers used by the three-stream RRF in
``imas_codex.standard_names.search``.

The grammar stream partitions the 12 ISN segments into three tiers and
applies tier-dependent RRF weights. See plan 40 §5.4 for rationale and
worked example.
"""

from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------------
# Tokeniser (Plan 40 §5.2)
# ---------------------------------------------------------------------------

#: ISN connector stopwords. Carry no segment information; only inflate
#: keyword recall when left in the token stream.
STOPWORDS: Final[frozenset[str]] = frozenset(
    {"of", "at", "from", "in", "to", "for", "the", "a", "an"}
)


_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def tokenise_query(query: str) -> list[str]:
    """Lower-case, snake-split, drop ISN connector stopwords.

    >>> tokenise_query("x_component_of_magnetic_field")
    ['x', 'component', 'magnetic', 'field']
    >>> tokenise_query("electron_temperature_at_outboard_midplane")
    ['electron', 'temperature', 'outboard', 'midplane']
    >>> tokenise_query("")
    []

    Empty / whitespace-only inputs return an empty list. Non-ASCII
    characters are coerced via ``str.lower`` then split on the same
    regex; consumers should treat the output as ``list[str]``.
    """
    if not query or not query.strip():
        return []
    parts = _SPLIT_RE.split(query.lower())
    return [p for p in parts if p and p not in STOPWORDS]


# ---------------------------------------------------------------------------
# Tier policy (Plan 40 §5.4)
# ---------------------------------------------------------------------------

#: Tier 1 — physical concepts. Always contributes; can solely surface a result.
TIER1_SEGMENTS: Final[frozenset[str]] = frozenset(
    {"physical_base", "subject", "geometric_base"}
)

#: Tier 2 — operational modifiers. Contributes only with a Tier-1 anchor
#: AND co-occurrence in the vector or keyword stream (v3.2 strict AND-gate).
TIER2_SEGMENTS: Final[frozenset[str]] = frozenset(
    {"transformation", "component", "position", "process"}
)

#: Tier 3 — geometric / device modifiers. Tie-break boost only; never
#: surfaces a candidate alone, requires Tier-1 anchor.
TIER3_SEGMENTS: Final[frozenset[str]] = frozenset(
    {"coordinate", "geometry", "region", "device", "object"}
)

#: All segments stored as bare-name columns by ``_write_grammar_decomposition``.
ALL_TIER_SEGMENTS: Final[frozenset[str]] = (
    TIER1_SEGMENTS | TIER2_SEGMENTS | TIER3_SEGMENTS
)

#: RRF weights per tier. Chosen so a single Tier-1 hit at vector/keyword
#: rank 1 outranks an unbounded flood of Tier-2/3-only hits (§5.4 N1).
TIER_WEIGHT: Final[dict[int, float]] = {1: 1.0, 2: 0.5, 3: 0.25}


def tier_of(segment: str) -> int:
    """Return the tier (1/2/3) for *segment*. Unknown segments return 0."""
    if segment in TIER1_SEGMENTS:
        return 1
    if segment in TIER2_SEGMENTS:
        return 2
    if segment in TIER3_SEGMENTS:
        return 3
    return 0


def filter_by_tier_policy(
    by_id: dict[str, dict[int, list[tuple[str, int, float]]]],
    vector_hits: set[str],
    keyword_hits: set[str],
) -> set[str]:
    """Apply v3.2 strict tier-eligibility AND-gate.

    Returns the set of SN ids whose grammar-stream hits are admissible.

    Eligibility rules (plan 40 §5.4, v3.2 strict AND-gate — resolved in
    favour of the §9.4 test):

    - **Tier 1 hit, vector/keyword co-occurrence:** admitted (physical
      anchor + corroboration).
    - **Tier 1 hit, no vector/keyword co-occurrence:** dropped — pure
      grammar evidence on a Tier-1 segment is not enough on its own.
    - **Tier 2 hit, no Tier 1 anchor:** dropped.
    - **Tier 2 hit + Tier 1 anchor + (vector OR keyword) hit:** admitted.
    - **Tier 3 hit:** requires a Tier 1 anchor; vector/keyword
      co-occurrence is irrelevant for tier 3.

    The §17.1 §9.4 test ``test_tier2_requires_tier1_anchor_and_vk_cooccurrence``
    asserts the strict AND-gate: even a Tier-1-bearing candidate must
    also appear in vector or keyword to qualify when the only evidence
    is grammar. This is a deviation from the plan §5.4 code sketch
    (lines 339-352) which admitted Tier-1 hits unconditionally; the test
    is the source of truth.

    Args:
        by_id: ``{sn_id: {tier: [(segment, rank, weight), …]}}``.
        vector_hits: set of SN ids that appear in the vector stream.
        keyword_hits: set of SN ids that appear in the keyword stream.

    Returns:
        Set of SN ids admitted to the grammar-stream RRF input.
    """
    admitted: set[str] = set()
    for sn_id, tiers in by_id.items():
        has_t1 = 1 in tiers
        has_t2 = 2 in tiers
        has_t3 = 3 in tiers
        in_vk = sn_id in vector_hits or sn_id in keyword_hits
        if has_t1 and in_vk:
            # Anchor + corroboration — strongest case.
            admitted.add(sn_id)
        elif has_t2 and has_t1 and in_vk:
            # Tier-2 needs both anchor AND co-occurrence.
            admitted.add(sn_id)
        elif has_t3 and has_t1:
            # Tier-3 only contributes when anchored; vector/keyword optional.
            admitted.add(sn_id)
        # else: dropped
    return admitted
