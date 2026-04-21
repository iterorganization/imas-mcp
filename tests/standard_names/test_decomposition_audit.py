"""Tests for physical_base decomposition audit (Problem 2).

The decomposition audit surfaces closed-vocabulary tokens that were absorbed
into an open ``physical_base`` compound — cases where the LLM composer hid
structure inside the open slot rather than populating the appropriate
closed-vocab segment.
"""

from __future__ import annotations

import pytest


class TestFindAbsorbedClosedTokens:
    """Core string-matching logic for decomposition audit."""

    @pytest.fixture
    def closed_vocab(self) -> dict[str, list[str]]:
        """Minimal synthetic closed vocabulary mirroring ISN segments."""
        return {
            "subject": ["electron", "ion", "deuterium"],
            "component": ["radial", "toroidal", "poloidal", "parallel"],
            "coordinate": ["radial", "toroidal", "poloidal", "x", "y", "z"],
            "transformation": [
                "normalized",
                "volume_averaged",
                "flux_surface_averaged",
                "time_derivative_of",
            ],
            "position": ["magnetic_axis", "plasma_boundary"],
            "process": ["conduction", "convection"],
        }

    def test_empty_haystack_returns_empty(self, closed_vocab):
        from imas_codex.standard_names.decomposition import (
            find_absorbed_closed_tokens,
        )

        assert find_absorbed_closed_tokens("", closed_vocab) == []
        assert find_absorbed_closed_tokens(None, closed_vocab) == []

    def test_toroidal_torque_flags_component(self, closed_vocab):
        """``toroidal_torque`` hides a ``component`` token in ``physical_base``."""
        from imas_codex.standard_names.decomposition import (
            find_absorbed_closed_tokens,
        )

        hits = find_absorbed_closed_tokens("toroidal_torque", closed_vocab)
        hit_pairs = set(hits)
        # toroidal is in both component and coordinate — both should be flagged
        assert ("toroidal", "component") in hit_pairs
        assert ("toroidal", "coordinate") in hit_pairs

    def test_volume_averaged_electron_temperature(self, closed_vocab):
        """Multi-word transformations and single-word subjects both detected."""
        from imas_codex.standard_names.decomposition import (
            find_absorbed_closed_tokens,
        )

        hits = find_absorbed_closed_tokens(
            "volume_averaged_electron_temperature", closed_vocab
        )
        hit_pairs = set(hits)
        assert ("volume_averaged", "transformation") in hit_pairs
        assert ("electron", "subject") in hit_pairs

    def test_normalized_poloidal_flux(self, closed_vocab):
        """``normalized`` (transformation) and ``poloidal`` (coord/comp) flagged."""
        from imas_codex.standard_names.decomposition import (
            find_absorbed_closed_tokens,
        )

        hits = find_absorbed_closed_tokens("normalized_poloidal_flux", closed_vocab)
        hit_pairs = set(hits)
        assert ("normalized", "transformation") in hit_pairs
        assert ("poloidal", "component") in hit_pairs
        assert ("poloidal", "coordinate") in hit_pairs

    def test_clean_atomic_base_no_hits(self, closed_vocab):
        """A truly atomic ``physical_base`` produces no hits."""
        from imas_codex.standard_names.decomposition import (
            find_absorbed_closed_tokens,
        )

        # pressure / temperature / density are atomic and not in closed vocab
        assert find_absorbed_closed_tokens("pressure", closed_vocab) == []
        assert find_absorbed_closed_tokens("cross_sectional_area", closed_vocab) == []

    def test_word_boundary_prevents_substring_false_positive(self, closed_vocab):
        """``ion`` must not match inside ``ionisation`` — word boundary required."""
        from imas_codex.standard_names.decomposition import (
            find_absorbed_closed_tokens,
        )

        hits = find_absorbed_closed_tokens("ionisation_rate", closed_vocab)
        # 'ion' does not appear as a whole underscore-separated token
        assert ("ion", "subject") not in set(hits)

    def test_word_boundary_matches_at_start_and_end(self, closed_vocab):
        """Tokens at the start or end of the haystack are matched."""
        from imas_codex.standard_names.decomposition import (
            find_absorbed_closed_tokens,
        )

        # 'electron' at the start
        start_hits = set(
            find_absorbed_closed_tokens("electron_temperature", closed_vocab)
        )
        assert ("electron", "subject") in start_hits

        # 'conduction' at the end
        end_hits = set(
            find_absorbed_closed_tokens("heat_flux_conduction", closed_vocab)
        )
        assert ("conduction", "process") in end_hits

    def test_short_single_character_tokens_skipped(self, closed_vocab):
        """Single-char coordinate tokens like ``x`` / ``y`` / ``z`` ignored by default."""
        from imas_codex.standard_names.decomposition import (
            find_absorbed_closed_tokens,
        )

        # The haystack contains 'x' and 'y' as separate underscore tokens but
        # they must not fire because they are shorter than the minimum length.
        hits = find_absorbed_closed_tokens("something_x_y_foo", closed_vocab)
        flagged_tokens = {t for t, _ in hits}
        assert "x" not in flagged_tokens
        assert "y" not in flagged_tokens

    def test_min_token_len_override(self, closed_vocab):
        """Callers can relax the min-length threshold explicitly."""
        from imas_codex.standard_names.decomposition import (
            find_absorbed_closed_tokens,
        )

        hits = find_absorbed_closed_tokens(
            "profile_x_component", closed_vocab, min_token_len=1
        )
        flagged = {t for t, _ in hits}
        assert "x" in flagged

    def test_result_is_sorted_and_unique(self, closed_vocab):
        """Result order is stable and duplicates collapse."""
        from imas_codex.standard_names.decomposition import (
            find_absorbed_closed_tokens,
        )

        hits = find_absorbed_closed_tokens("toroidal_radial_pressure", closed_vocab)
        # No duplicates
        assert len(hits) == len(set(hits))
        # Sorted by (segment, token)
        assert hits == sorted(hits, key=lambda p: (p[1], p[0]))


class TestFormatAbsorbedTokens:
    def test_format_pairs(self):
        from imas_codex.standard_names.decomposition import format_absorbed_tokens

        s = format_absorbed_tokens(
            [("toroidal", "component"), ("normalized", "transformation")]
        )
        assert "toroidal (component)" in s
        assert "normalized (transformation)" in s

    def test_format_empty(self):
        from imas_codex.standard_names.decomposition import format_absorbed_tokens

        assert format_absorbed_tokens([]) == ""


class TestAgainstRealISNVocabulary:
    """Spot-check decomposition audit against the installed ISN vocabulary.

    Skipped gracefully if imas_standard_names is unavailable at test time
    (e.g. in minimal CI environments).
    """

    @pytest.fixture
    def isn_closed_vocab(self) -> dict[str, list[str]]:
        from imas_standard_names.grammar import (
            Component,
            Position,
            Process,
            Subject,
            Transformation,
        )

        return {
            "subject": [e.value for e in Subject],
            "component": [e.value for e in Component],
            "position": [e.value for e in Position],
            "process": [e.value for e in Process],
            "transformation": [e.value for e in Transformation],
        }

    def test_toroidal_torque_flags_component(self, isn_closed_vocab):
        """Real ISN vocab: ``toroidal_torque`` flags ``toroidal`` as component."""
        from imas_codex.standard_names.decomposition import (
            find_absorbed_closed_tokens,
        )

        hits = {
            seg: tok
            for tok, seg in find_absorbed_closed_tokens(
                "toroidal_torque", isn_closed_vocab
            )
        }
        assert hits.get("component") == "toroidal"

    def test_electron_absorbed_as_subject(self, isn_closed_vocab):
        """Real ISN vocab flags ``electron`` inside an open compound."""
        from imas_codex.standard_names.decomposition import (
            find_absorbed_closed_tokens,
        )

        pairs = set(
            find_absorbed_closed_tokens(
                "volume_averaged_electron_temperature", isn_closed_vocab
            )
        )
        assert ("electron", "subject") in pairs
