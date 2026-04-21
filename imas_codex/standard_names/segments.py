"""Grammar segment classification — open vs closed vocabulary.

The ISN grammar distinguishes *closed* segments (fixed vocabulary — any token
outside the list is a real vocabulary gap) from *open* segments (free-form
compounds — any novel token is legitimate by design).

The LLM composer and our segment-edge writer both occasionally emit "missing
token" reports on open segments, which pollutes the ``VocabGap`` node
population with nonsensical entries.  The ISN release process then has to
manually filter these out.  This module is the single source of truth used by
codex to decide whether a reported gap is real.

Open segments are derived from ``SEGMENT_TOKEN_MAP`` in the installed
imas-standard-names package: any segment with an empty token list is treated
as open.  The LLM composer also reports structural ambiguity via a pseudo
segment ``grammar_ambiguity`` — these are grammar findings, not missing
tokens, and are likewise filtered.

When the ISN package is unavailable at import time we fall back to a
conservative hard-coded set so tests and offline tools still behave sensibly.
"""

from __future__ import annotations

from functools import lru_cache

# Hard fallback (matches imas-standard-names v0.7.0rc18).
_FALLBACK_OPEN_SEGMENTS: frozenset[str] = frozenset({"physical_base"})

# Pseudo segments reported by the composer but that are not real grammar
# segments — these are structural findings, not missing tokens.  Treated as
# "open" for VocabGap filtering purposes.
PSEUDO_SEGMENTS: frozenset[str] = frozenset({"grammar_ambiguity"})


@lru_cache(maxsize=1)
def open_segments() -> frozenset[str]:
    """Return the set of ISN grammar segments with open vocabulary.

    A segment is considered *open* when its closed-vocabulary token list is
    empty — any token is admissible by design (``physical_base`` is the
    canonical example).  Emitting a :class:`VocabGap` for such a segment is
    nonsensical.

    Results are memoised across the process lifetime because
    ``SEGMENT_TOKEN_MAP`` is immutable and cheap-but-not-free to introspect.
    """
    try:
        from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP
    except ImportError:
        return _FALLBACK_OPEN_SEGMENTS

    try:
        return frozenset(seg for seg, tokens in SEGMENT_TOKEN_MAP.items() if not tokens)
    except Exception:  # pragma: no cover — defensive
        return _FALLBACK_OPEN_SEGMENTS


def is_open_segment(segment: str | None) -> bool:
    """Return ``True`` if ``segment`` is open-vocab or a pseudo segment.

    Gaps reported against open or pseudo segments should never materialise as
    :class:`VocabGap` nodes: they do not indicate a missing closed-vocabulary
    token.
    """
    if not segment:
        return False
    if segment in PSEUDO_SEGMENTS:
        return True
    return segment in open_segments()


def filter_closed_segment_gaps(
    gaps: list[dict],
    *,
    segment_key: str = "segment",
) -> tuple[list[dict], list[dict]]:
    """Split gap records into (closed, open) by their grammar segment.

    ``gaps`` is a list of dicts with at least a ``segment`` key.  Returns the
    tuple ``(kept, dropped)`` — ``kept`` is emitted as ``VocabGap`` nodes,
    ``dropped`` is logged and discarded.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    for g in gaps:
        if is_open_segment(g.get(segment_key)):
            dropped.append(g)
        else:
            kept.append(g)
    return kept, dropped


__all__ = [
    "PSEUDO_SEGMENTS",
    "filter_closed_segment_gaps",
    "is_open_segment",
    "open_segments",
]
