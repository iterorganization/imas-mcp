"""Tests for species context extraction and batch splitting."""

import pytest

from imas_codex.standard_names.enrichment import extract_species_context


@pytest.mark.parametrize(
    "path, expected",
    [
        ("edge_transport/model/ggd/electrons/energy/v_parallel/values", "electron"),
        ("edge_transport/model/ggd/ion/energy/v_parallel/values", "ion"),
        ("edge_transport/model/ggd/neutral/energy/v_parallel/values", "neutral"),
        ("core_transport/model/profiles_1d/electrons/particles/d", "electron"),
        (
            "core_transport/model/profiles_1d/ion/energy/flux/flux_multiplied_by_area",
            "ion",
        ),
        ("edge_transport/model/ggd/fast_ion/energy/flux", "fast_ion"),
        ("equilibrium/time_slice/profiles_1d/psi", None),
        ("magnetics/bpol_probe/field/data", None),
        ("core_profiles/profiles_1d/t_i_average", None),
    ],
)
def test_extract_species_context(path, expected):
    assert extract_species_context(path) == expected
