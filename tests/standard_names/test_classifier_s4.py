"""Tests for S4 instrument-geometry classifier rule."""

import pytest

from imas_codex.standard_names.classifier import classify_path


@pytest.mark.parametrize(
    "path",
    [
        "neutron_diagnostic/detector/geometry/outline/x1",
        "neutron_diagnostic/detector/geometry/outline/x2",
        "neutron_diagnostic/detector/aperture/x1_width",
        "neutron_diagnostic/detector/aperture/x2_width",
        "spectrometer_visible/channel/detector/geometry/outline/x1",
        "spectrometer_visible/channel/detector/aperture/outline/x1",
        "bolometer/channel/detector/geometry/outline/x1",
        "bolometer/channel/line_of_sight/first_point/r",
        "bolometer/channel/line_of_sight/second_point/z",
        "ec_launchers/beam/launching_position/spot/size",
        "spectrometer_visible/channel/detector/aperture/x1_width",
        "radiation_measurement/channel/detector/geometry/outline/x2",
        "charge_exchange/channel/detector/aperture/x2_width",
    ],
)
def test_s4_skips_instrument_geometry(path):
    node = {"path": path, "data_type": "FLT_1D"}
    assert classify_path(node) == "skip"


@pytest.mark.parametrize(
    "path",
    [
        # Physics measurements - keep these
        "neutron_diagnostic/detector/counts",
        "neutron_diagnostic/detector/efficiency",
        "spectrometer_visible/channel/detector/signal",
        "bolometer/channel/power",
        # Regular physics paths
        "equilibrium/time_slice/profiles_1d/psi",
        "core_profiles/profiles_1d/electrons/temperature",
        "edge_transport/model/ggd/electrons/energy/v_parallel/values",
    ],
)
def test_s4_keeps_physics_measurements(path):
    node = {"path": path, "data_type": "FLT_1D"}
    assert classify_path(node) == "quantity"
