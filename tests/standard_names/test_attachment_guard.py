"""Tests for the tense-consistency guard on LLM-proposed attachments."""

import pytest

from imas_codex.standard_names.workers import _is_attachment_consistent


@pytest.mark.parametrize(
    "source_id,sn_name",
    [
        ("core_profiles/profiles_1d/electrons/density", "electron_density"),
        (
            "core_instant_changes/change/profiles_1d/electrons/density",
            "change_in_electron_density",
        ),
        (
            "core_profiles/profiles_1d/electrons/temperature",
            "electron_temperature",
        ),
        (
            "core_instant_changes/change/profiles_1d/electrons/temperature",
            "tendency_of_electron_temperature",
        ),
        (
            "equilibrium/time_slice/global_quantities/ip",
            "rate_of_change_of_plasma_current",
        ),  # rate path heuristic relies on SN prefix only — base path + rate SN flagged
    ],
)
def test_consistent_pairs(source_id: str, sn_name: str) -> None:
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    if sn_name.startswith(
        ("change_in_", "tendency_of_", "rate_of_", "time_derivative_of_")
    ):
        # Path must contain a change/tendency token to pass.
        if any(
            tok in source_id
            for tok in ("instant_changes", "/change", "_delta", "tendency_")
        ):
            assert ok, reason
        else:
            assert not ok, "rate/change SN with base path must be rejected"
    else:
        assert ok, reason


@pytest.mark.parametrize(
    "source_id,sn_name",
    [
        # Base path → change SN: must be rejected.
        ("core_profiles/profiles_1d/electrons/density", "change_in_electron_density"),
        (
            "core_profiles/profiles_1d/electrons/temperature",
            "tendency_of_electron_temperature",
        ),
        (
            "equilibrium/time_slice/global_quantities/ip",
            "rate_of_change_of_plasma_current",
        ),
        # Change path → base SN: must be rejected.
        (
            "core_instant_changes/change/profiles_1d/electrons/density",
            "electron_density",
        ),
        (
            "core_instant_changes/change/global_quantities/ip",
            "plasma_current",
        ),
    ],
)
def test_inconsistent_pairs_are_rejected(source_id: str, sn_name: str) -> None:
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    assert not ok
    assert "tense mismatch" in reason
