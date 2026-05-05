"""Tests that geometry paths are classified as quantity (not skipped).

Instrument geometry paths describe valid geometric quantities (outline
coordinates, aperture widths, line-of-sight endpoints) that deserve
proper standard names like ``radial_coordinate_of_detector_outline``.
"""

import pytest

from imas_codex.standard_names.classifier import classify_path


@pytest.mark.parametrize(
    "path",
    [
        # Detector geometry — valid geometric quantities
        "neutron_diagnostic/detector/geometry/outline/x1",
        "neutron_diagnostic/detector/geometry/outline/x2",
        "neutron_diagnostic/detector/aperture/x1_width",
        "spectrometer_visible/channel/detector/geometry/outline/x1",
        "bolometer/channel/detector/geometry/outline/x1",
        "bolometer/channel/line_of_sight/first_point/r",
        "bolometer/channel/line_of_sight/second_point/z",
        # Physics measurements
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
def test_geometry_and_physics_paths_are_quantity(path):
    """All geometry and physics paths classify as quantity."""
    node = {"path": path, "data_type": "FLT_1D"}
    assert classify_path(node) == "quantity"
