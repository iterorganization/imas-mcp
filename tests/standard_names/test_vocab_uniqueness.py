"""Diagnostic test for ISN vocabulary token uniqueness across segments.

Reports tokens that appear in more than one closed segment (not a failure —
this is a watch-list for ISN editorial).  Also asserts that well-known
tokens round-trip through ``is_known_token``.
"""

from __future__ import annotations

import warnings

import pytest


class TestVocabUniqueness:
    """Iterate closed segments and report cross-segment token overlap."""

    def test_report_cross_segment_tokens(self):
        """Report (don't fail) tokens appearing in >1 closed segment."""
        try:
            from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP
        except ImportError:
            pytest.skip("imas-standard-names not installed")

        # Build reverse index: token → [segments]
        token_to_segments: dict[str, list[str]] = {}
        for seg, tokens in SEGMENT_TOKEN_MAP.items():
            if not tokens:
                continue  # skip open-vocabulary segments
            for tok in tokens:
                token_to_segments.setdefault(tok, []).append(seg)

        overlaps = {
            tok: segs for tok, segs in token_to_segments.items() if len(segs) > 1
        }

        if overlaps:
            lines = [f"  {tok}: {segs}" for tok, segs in sorted(overlaps.items())]
            warnings.warn(
                f"{len(overlaps)} tokens appear in >1 closed segment:\n"
                + "\n".join(lines),
                stacklevel=1,
            )

    @pytest.mark.parametrize(
        "token",
        ["parallel", "radial", "toroidal", "ion", "particle"],
    )
    def test_well_known_tokens_round_trip(self, token: str):
        """Well-known tokens must be found by is_known_token."""
        try:
            from imas_standard_names.grammar.constants import (
                SEGMENT_TOKEN_MAP,  # noqa: F401
            )
        except ImportError:
            pytest.skip("imas-standard-names not installed")

        from imas_codex.standard_names.segments import is_known_token

        segments = is_known_token(token)
        assert segments, f"{token!r} not found in any closed segment"

    def test_is_known_token_returns_list(self):
        """is_known_token always returns a list, even for unknown tokens."""
        from imas_codex.standard_names.segments import is_known_token

        result = is_known_token("__definitely_not_a_token__")
        assert isinstance(result, list)
        assert result == []
