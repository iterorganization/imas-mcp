"""Decomposition audit for standard names.

Detects closed-vocabulary tokens (``subject``, ``component``,
``transformation``, ``position``, ``coordinate``, ``object``, ``geometry``,
``process``, ``device``, ``geometric_base``, ``region``) that were absorbed
into an open-vocabulary ``physical_base`` compound instead of being promoted
to their own grammar segment.

Motivating examples:

- ``toroidal_torque`` — ``toroidal`` is a closed ``component`` token that
  should live in the ``component`` segment, not embedded in
  ``physical_base``.
- ``volume_averaged_electron_temperature`` — ``volume_averaged`` is a closed
  ``transformation`` token; ``electron`` is a closed ``subject`` token.
- ``normalized_poloidal_flux`` — ``normalized`` is a closed
  ``transformation``; ``poloidal`` is a closed ``coordinate``/``component``.

The utility surfaces candidates.  The reviewer decides whether each hit is a
genuine decomposition issue or a legitimate lexicalised compound
(e.g. ``poloidal_flux`` as an atomic named quantity).  This is deliberately
text-level matching — semantic nuance stays with the LLM reviewer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# Tokens shorter than this length are skipped to avoid spurious hits on
# single-letter closed vocab (x, y, z) that frequently collide with longer
# legitimate atomic bases.
_MIN_TOKEN_LEN = 3


def find_absorbed_closed_tokens(
    haystack: str | None,
    closed_vocab: Mapping[str, Iterable[str]],
    *,
    min_token_len: int = _MIN_TOKEN_LEN,
) -> list[tuple[str, str]]:
    """Return ``(token, segment)`` pairs for closed-vocab hits inside ``haystack``.

    Matching uses underscore word boundaries: a token ``t`` is a hit iff the
    substring ``_t_`` appears in ``_haystack_``.  Tokens shorter than
    ``min_token_len`` are skipped to avoid noise from single-character
    coordinate tokens.

    Args:
        haystack: The compound to audit — typically the ``physical_base``
            slot returned by ``parse_standard_name``, or the full standard
            name.
        closed_vocab: Mapping of ``segment_name`` → iterable of closed tokens.
            Callers typically pass the ``grammar_enums`` dict exposed to the
            reviewer prompt.
        min_token_len: Minimum token length to consider (default 3).

    Returns:
        Sorted list of unique ``(token, segment)`` pairs found embedded in
        ``haystack``.  Stable ordering (segment, then token) simplifies
        testing and downstream rendering.
    """
    if not haystack:
        return []

    padded = f"_{haystack}_"
    seen: set[tuple[str, str]] = set()

    for segment, tokens in closed_vocab.items():
        for tok in tokens:
            if not tok or len(tok) < min_token_len:
                continue
            if f"_{tok}_" in padded:
                seen.add((tok, segment))

    return sorted(seen, key=lambda p: (p[1], p[0]))


def format_absorbed_tokens(
    hits: list[tuple[str, str]],
) -> str:
    """Render absorbed-token hits as a comma-separated text string.

    Convenience formatter for prompt injection / log output.
    """
    return ", ".join(f"{tok} ({seg})" for tok, seg in hits)


__all__ = ["find_absorbed_closed_tokens", "format_absorbed_tokens"]
