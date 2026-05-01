"""Trigger predicate + arm assignment for refine-site fan-out (plan 39 §5.1, §8.4).

The trigger gate is the orthogonality knob between fan-out and the
existing B12 grammar-retry compose path: fan-out fires *only* when
prior reviewer feedback shows the kind of ambiguity / disambiguation
trouble that targeted DD context could plausibly resolve.

Public surface
--------------

- :func:`extract_reviewer_excerpt` — flatten the dim-allow-listed
  values of ``reviewer_comments_per_dim_name`` (parsing the JSON-string
  representation if necessary) into a single truncated excerpt.
- :func:`should_trigger_fanout` — return ``(fire, excerpt)`` where
  ``fire`` is ``True`` only when *all* of the predicate's clauses hold
  (chain length, B12 enrichment, keyword presence in the allow-list
  excerpt).
- :func:`assign_arm` — deterministic 50/50 (or
  ``refine_fanout_arm_percent``) within-cohort A/B label, hashing
  ``(sn_id, cycle_index)`` via blake2b for stability across runs.

This module is intentionally pure-Python with no graph or LLM imports;
the refine worker calls it once per item before any LLM spend.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "assign_arm",
    "extract_reviewer_excerpt",
    "should_trigger_fanout",
]


def _parse_comments(raw: object) -> dict[str, str]:
    """Parse the per-dim reviewer-comments field.

    Accepts either a ``dict[str, str]`` (already decoded) or a JSON
    string (legacy graph storage).  Returns ``{}`` for ``None`` or
    invalid input — the trigger predicate then naturally evaluates to
    ``False``.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items() if v is not None}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items() if v is not None}
    return {}


def extract_reviewer_excerpt(
    reviewer_comments_per_dim: object,
    *,
    dims: tuple[str, ...] | list[str],
    char_cap: int,
) -> str:
    """Concatenate allow-listed reviewer comments and truncate.

    Parameters
    ----------
    reviewer_comments_per_dim:
        The graph-stored field (dict, JSON string, or ``None``).
    dims:
        Allow-list of dim keys to include (plan 39 §5.1 I3).  Other
        dims (e.g. ``convention``, ``grammar``) are ignored — those
        failure modes are not what fan-out is designed to address.
    char_cap:
        Total excerpt length cap (S3).

    Returns
    -------
    The concatenated excerpt (`"<dim>: <comment>\\n..."` for each
    allow-listed dim with a non-empty value), truncated to
    ``char_cap`` characters.  Empty string when no allow-listed dim
    has a value.
    """
    parsed = _parse_comments(reviewer_comments_per_dim)
    if not parsed:
        return ""
    parts: list[str] = []
    for dim in dims:
        value = parsed.get(dim)
        if not value:
            continue
        parts.append(f"{dim}: {value}")
    if not parts:
        return ""
    excerpt = "\n".join(parts)
    if char_cap > 0 and len(excerpt) > char_cap:
        excerpt = excerpt[:char_cap]
    return excerpt


def should_trigger_fanout(
    *,
    reviewer_comments_per_dim: object,
    chain_length: int,
    chain_history: list[Any] | None,
    keywords: tuple[str, ...] | list[str],
    dims: tuple[str, ...] | list[str],
    char_cap: int,
) -> tuple[bool, str]:
    """Return ``(fire, excerpt)`` per plan 39 §5.1.

    Fan-out fires only when *all* of:

    1. ``chain_length > 0`` — the candidate has been refined at least
       once, so reviewer feedback exists to draw signal from.
    2. ``chain_history`` is populated — B12 deterministic enrichment
       has been applied for the current cycle (the claim batch
       attaches it; an empty list means the enrichment did not run).
    3. At least one allow-listed reviewer-comment value contains a
       trigger keyword (case-insensitive).

    The returned ``excerpt`` is the truncated allow-listed comment
    block, suitable for direct injection into the proposer prompt.
    When ``fire`` is ``False`` the excerpt is still returned so
    callers can log it — but they must not pass it to ``run_fanout``.
    """
    if chain_length <= 0:
        return False, ""
    if not chain_history:
        return False, ""

    excerpt = extract_reviewer_excerpt(
        reviewer_comments_per_dim,
        dims=dims,
        char_cap=char_cap,
    )
    if not excerpt:
        return False, ""

    haystack = excerpt.casefold()
    for kw in keywords:
        if kw and kw.casefold() in haystack:
            return True, excerpt
    return False, excerpt


def assign_arm(
    sn_id: str,
    cycle_index: int,
    *,
    arm_percent: int = 50,
) -> str:
    """Assign a stable ``"on"`` / ``"off"`` arm for the within-cohort A/B.

    Hashes ``(sn_id, cycle_index)`` via blake2b (8-byte digest) and
    maps the integer mod 100 against ``arm_percent``.  Stability:

    - Identical ``(sn_id, cycle_index)`` always lands in the same arm.
    - Different ``cycle_index`` values for the same ``sn_id`` are
      independently distributed (no correlation across cycles).
    - Default ``arm_percent=50`` yields a 50/50 split; values 0 and
      100 are honoured (all-off / all-on respectively).

    Plan 39 §8.4 specified ``hash((sn_id, chain_length)) % 2``; we
    use blake2b instead of Python's ``hash()`` because the latter is
    PYTHONHASHSEED-randomised between processes.  The functional
    contract (deterministic 50/50) is preserved.
    """
    if arm_percent <= 0:
        return "off"
    if arm_percent >= 100:
        return "on"
    payload = f"{sn_id}|{int(cycle_index)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="big") % 100
    return "on" if bucket < arm_percent else "off"
