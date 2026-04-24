"""Vocab-gap token classifier — reject noise, reclassify locus tokens.

When the LLM composer reports a "missing token" via VocabGap, this module
decides whether the token is a genuine vocabulary gap or noise that should
be filtered before it becomes a graph node or ISN PR candidate.

Rejection rules:
- **R1 hardware**: device-specific identifiers, manufacturer names, vendor codes.
- **R2 signal codenames**: facility signal shorthand that leaked from pipelines.
- **R3 plural dedup**: English plurals of tokens already in the grammar.
- **R4 short/numeric**: tokens shorter than 3 chars or containing digits.
- **R5 locus reclassification**: position-like tokens rerouted to ``position``
  segment instead of being flagged as a gap on their original segment.

Usage::

    from imas_codex.standard_names.vocab_token_filter import classify_vocab_token
    verdict = classify_vocab_token("vsm_1", "subject")
    # verdict.action == "reject", verdict.reason == "R1: hardware/device identifier"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# -- Verdict types -----------------------------------------------------------

Action = Literal["accept", "reject", "reclassify"]


@dataclass(frozen=True, slots=True)
class TokenVerdict:
    """Outcome of classifying a proposed vocab-gap token."""

    action: Action
    reason: str
    reclassify_segment: str | None = None


# -- Known noise patterns ----------------------------------------------------

#: Facility-specific signal codenames that leak from MDSplus/TDI pipelines.
_SIGNAL_CODENAMES: frozenset[str] = frozenset(
    {
        "ip",
        "btf",
        "bpol",
        "nel",
        "nbi",
        "ech",
        "icrh",
        "ecrh",
        "lhcd",
        "rcp",
        "lcp",
        "sxr",
        "hxr",
        "fir",
        "dcn",
        "dto",
        "dml",
        "ece",
    }
)

#: Device/manufacturer names and vendor codes.
_HARDWARE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^vsm", re.IGNORECASE),  # VSM probes
    re.compile(r"^kstar", re.IGNORECASE),  # KSTAR device
    re.compile(r"^mast_?u?$", re.IGNORECASE),  # MAST-U
    re.compile(r"^jet$", re.IGNORECASE),  # JET
    re.compile(r"^iter$", re.IGNORECASE),  # ITER
    re.compile(r"^tcv$", re.IGNORECASE),  # TCV
    re.compile(r"^west$", re.IGNORECASE),  # WEST
    re.compile(r"^jt_?60", re.IGNORECASE),  # JT-60SA
    re.compile(r"^diii_?d$", re.IGNORECASE),  # DIII-D
    re.compile(r"^asdex", re.IGNORECASE),  # ASDEX
    re.compile(r"^w7_?x$", re.IGNORECASE),  # W7-X
)

#: Tokens that look like positions/loci — these should be reclassified
#: to the ``position`` segment rather than flagged as missing vocab.
_LOCUS_TOKENS: frozenset[str] = frozenset(
    {
        "midplane",
        "lfs",
        "hfs",
        "inboard",
        "outboard",
        "divertor",
        "x_point",
        "o_point",
        "strike_point",
        "baffle",
        "limiter",
        "antenna",
        "vessel_wall",
        "first_wall",
        "blanket",
        "inner_midplane",
        "outer_midplane",
        "upstream",
        "downstream",
        "target",
        "inner_target",
        "outer_target",
        "pedestal_top",
    }
)

# Regex: contains digit
_HAS_DIGIT = re.compile(r"\d")


# -- Classification ----------------------------------------------------------


def classify_vocab_token(
    token: str,
    segment: str,
    *,
    existing_tokens: frozenset[str] | None = None,
) -> TokenVerdict:
    """Classify a proposed vocab-gap token.

    Args:
        token: The ``needed_token`` string from the LLM composer.
        segment: The grammar segment the token was proposed for.
        existing_tokens: Optional set of tokens already in the grammar
            vocabulary (for plural-dedup checking).  When ``None``,
            the ISN package is queried at runtime.

    Returns:
        A :class:`TokenVerdict` with ``action`` (accept/reject/reclassify),
        ``reason``, and optional ``reclassify_segment``.
    """
    # Normalise
    tok = token.strip().lower()

    # ---------------------------------------------------------------
    # R4: Short tokens or tokens containing digits
    # ---------------------------------------------------------------
    if len(tok) < 3:
        return TokenVerdict("reject", "R4: token shorter than 3 characters")
    if _HAS_DIGIT.search(tok):
        return TokenVerdict("reject", "R4: token contains digits")

    # ---------------------------------------------------------------
    # R1: Hardware / device identifiers
    # ---------------------------------------------------------------
    if any(pat.search(tok) for pat in _HARDWARE_PATTERNS):
        return TokenVerdict("reject", "R1: hardware/device identifier")

    # ---------------------------------------------------------------
    # R2: Signal codenames
    # ---------------------------------------------------------------
    if tok in _SIGNAL_CODENAMES:
        return TokenVerdict("reject", "R2: facility signal codename")

    # ---------------------------------------------------------------
    # R5: Locus reclassification — position-like tokens
    # ---------------------------------------------------------------
    if tok in _LOCUS_TOKENS and segment != "position":
        return TokenVerdict(
            "reclassify",
            f"R5: locus token reclassified from '{segment}' to 'position'",
            reclassify_segment="position",
        )

    # ---------------------------------------------------------------
    # R3: Plural dedup — English plural of an existing token
    # ---------------------------------------------------------------
    if existing_tokens is None:
        existing_tokens = _load_existing_tokens()

    if tok.endswith("s") and tok[:-1] in existing_tokens:
        return TokenVerdict(
            "reject",
            f"R3: plural of existing token '{tok[:-1]}'",
        )
    # Handle -es plurals (e.g. "processes" → "process")
    if tok.endswith("es") and tok[:-2] in existing_tokens:
        return TokenVerdict(
            "reject",
            f"R3: plural of existing token '{tok[:-2]}'",
        )

    return TokenVerdict("accept", "token accepted")


def filter_vocab_gaps(
    gaps: list[dict],
    *,
    existing_tokens: frozenset[str] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Classify a batch of vocab-gap dicts and split into three buckets.

    Returns:
        ``(accepted, rejected, reclassified)`` — three lists of gap dicts.
        Reclassified entries have their ``segment`` field updated in-place.
    """
    accepted: list[dict] = []
    rejected: list[dict] = []
    reclassified: list[dict] = []

    if existing_tokens is None:
        existing_tokens = _load_existing_tokens()

    for gap in gaps:
        token = gap.get("needed_token", "")
        segment = gap.get("segment", "")
        verdict = classify_vocab_token(token, segment, existing_tokens=existing_tokens)
        if verdict.action == "accept":
            accepted.append(gap)
        elif verdict.action == "reject":
            gap["_rejection_reason"] = verdict.reason
            rejected.append(gap)
        elif verdict.action == "reclassify":
            gap["segment"] = verdict.reclassify_segment or segment
            gap["_reclassify_reason"] = verdict.reason
            reclassified.append(gap)

    return accepted, rejected, reclassified


def _load_existing_tokens() -> frozenset[str]:
    """Load all grammar tokens from the installed ISN package."""
    try:
        from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

        all_tokens: set[str] = set()
        for tokens in SEGMENT_TOKEN_MAP.values():
            all_tokens.update(tokens)
        return frozenset(all_tokens)
    except ImportError:
        return frozenset()


__all__ = [
    "TokenVerdict",
    "classify_vocab_token",
    "filter_vocab_gaps",
]
