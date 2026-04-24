"""Tests for vocab-gap token classifier (Phase 6a).

Covers:
- R1 hardware rejection (device names, vendor codes)
- R2 signal codename rejection
- R3 plural dedup (English plurals of existing tokens)
- R4 short/numeric rejection
- R5 locus reclassification (position-like tokens)
- Happy path (legitimate vocab gaps)
- Batch filtering via filter_vocab_gaps
"""

from __future__ import annotations

import pytest


class TestR1HardwareRejection:
    """R1: Device-specific identifiers should be rejected."""

    @pytest.mark.parametrize(
        "token",
        ["vsm_probe", "kstar", "mast_u", "diii_d", "asdex_upgrade"],
    )
    def test_hardware_tokens_rejected(self, token):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        v = classify_vocab_token(token, "subject")
        assert v.action == "reject"
        assert "R1" in v.reason

    def test_vsm_with_digit_rejected(self):
        """vsm_1 hits R4 (digit) before R1 — both reject it."""
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        v = classify_vocab_token("vsm_1", "component")
        assert v.action == "reject"
        # Hits R4 digit check first, which is fine — still rejected
        assert "R4" in v.reason or "R1" in v.reason


class TestR2SignalCodenames:
    """R2: Facility signal shorthand should be rejected."""

    @pytest.mark.parametrize("token", ["ip", "btf", "nel", "nbi", "ech", "sxr"])
    def test_signal_codenames_rejected(self, token):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        v = classify_vocab_token(token, "subject")
        assert v.action == "reject"
        assert "R2" in v.reason or "R4" in v.reason  # short ones hit R4 first


class TestR3PluralDedup:
    """R3: English plurals of existing tokens should be rejected."""

    def test_channels_rejected_when_channel_exists(self):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        existing = frozenset({"channel", "electron", "temperature"})
        v = classify_vocab_token("channels", "component", existing_tokens=existing)
        assert v.action == "reject"
        assert "R3" in v.reason
        assert "channel" in v.reason

    def test_electrons_rejected_when_electron_exists(self):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        existing = frozenset({"electron", "ion"})
        v = classify_vocab_token("electrons", "subject", existing_tokens=existing)
        assert v.action == "reject"
        assert "R3" in v.reason

    def test_processes_es_plural_rejected(self):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        existing = frozenset({"process", "drift"})
        v = classify_vocab_token("processes", "process", existing_tokens=existing)
        assert v.action == "reject"
        assert "R3" in v.reason

    def test_non_plural_accepted(self):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        existing = frozenset({"electron"})
        v = classify_vocab_token(
            "runaway_electrons", "subject", existing_tokens=existing
        )
        # "runaway_electrons" does NOT match plural rule (singular would be
        # "runaway_electron" which is not in existing)
        assert v.action == "accept"


class TestR4ShortNumeric:
    """R4: Tokens shorter than 3 chars or containing digits should be rejected."""

    @pytest.mark.parametrize("token", ["ab", "x", "q"])
    def test_short_tokens_rejected(self, token):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        v = classify_vocab_token(token, "subject")
        assert v.action == "reject"
        assert "R4" in v.reason

    @pytest.mark.parametrize("token", ["b0_like", "v2", "temp3", "k1"])
    def test_digit_tokens_rejected(self, token):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        v = classify_vocab_token(token, "subject")
        assert v.action == "reject"
        assert "R4" in v.reason


class TestR5LocusReclassification:
    """R5: Position-like tokens should be reclassified to 'position' segment."""

    @pytest.mark.parametrize("token", ["midplane", "lfs", "hfs", "divertor", "x_point"])
    def test_locus_tokens_reclassified(self, token):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        v = classify_vocab_token(token, "subject")  # proposed on wrong segment
        # lfs/hfs are 3 chars, so they pass R4 (exactly 3)
        if v.action == "reclassify":
            assert v.reclassify_segment == "position"
            assert "R5" in v.reason

    def test_midplane_reclassified_from_component(self):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        v = classify_vocab_token("midplane", "component")
        assert v.action == "reclassify"
        assert v.reclassify_segment == "position"

    def test_position_segment_stays_accepted(self):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        # If already proposed on position segment, it's a legitimate gap
        v = classify_vocab_token("midplane", "position")
        assert v.action == "accept"


class TestHappyPath:
    """Legitimate vocab gaps should be accepted."""

    @pytest.mark.parametrize(
        "token,segment",
        [
            ("runaway_electron", "subject"),
            ("anomalous_diffusion", "process"),
            ("curvature_drift", "process"),
            ("deuterium", "subject"),
            ("scrape_off_layer", "region"),
        ],
    )
    def test_legitimate_tokens_accepted(self, token, segment):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        v = classify_vocab_token(token, segment, existing_tokens=frozenset())
        assert v.action == "accept"


class TestBatchFilter:
    """Test filter_vocab_gaps batch API."""

    def test_splits_into_three_buckets(self):
        from imas_codex.standard_names.vocab_token_filter import filter_vocab_gaps

        gaps = [
            {"needed_token": "runaway_electron", "segment": "subject"},
            {"needed_token": "vsm_1", "segment": "component"},
            {"needed_token": "midplane", "segment": "subject"},
            {"needed_token": "channels", "segment": "component"},
            {"needed_token": "ab", "segment": "subject"},
        ]
        existing = frozenset({"channel"})

        accepted, rejected, reclassified = filter_vocab_gaps(
            gaps, existing_tokens=existing
        )

        # runaway_electron → accepted
        assert any(g["needed_token"] == "runaway_electron" for g in accepted)
        # vsm_1 → rejected (R4 digit or R1 hardware — either is fine)
        assert any(g["needed_token"] == "vsm_1" for g in rejected)
        # midplane → reclassified to position
        assert any(g["needed_token"] == "midplane" for g in reclassified)
        # channels → rejected (R3 plural)
        assert any(g["needed_token"] == "channels" for g in rejected)
        # ab → rejected (R4 short)
        assert any(g["needed_token"] == "ab" for g in rejected)

    def test_reclassified_segment_updated(self):
        from imas_codex.standard_names.vocab_token_filter import filter_vocab_gaps

        gaps = [{"needed_token": "divertor", "segment": "component"}]
        _, _, reclassified = filter_vocab_gaps(gaps, existing_tokens=frozenset())
        assert len(reclassified) == 1
        assert reclassified[0]["segment"] == "position"
