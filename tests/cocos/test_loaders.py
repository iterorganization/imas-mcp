"""Tests for COCOS data loaders module."""

from pathlib import Path

import pytest

from imas_codex.cocos import EquilibriumData, load_from_eqdsk


class TestEquilibriumData:
    """Test EquilibriumData dataclass."""

    def test_required_fields(self):
        """Should require psi_axis, psi_edge, ip, b0."""
        data = EquilibriumData(
            psi_axis=0.5,
            psi_edge=-0.2,
            ip=-1e6,
            b0=-5.0,
        )
        assert data.psi_axis == 0.5
        assert data.psi_edge == -0.2
        assert data.ip == -1e6
        assert data.b0 == -5.0

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        data = EquilibriumData(
            psi_axis=0.5,
            psi_edge=-0.2,
            ip=-1e6,
            b0=-5.0,
        )
        assert data.q is None
        assert data.dp_dpsi is None
        assert data.source == ""

    def test_optional_fields_provided(self):
        """Should accept optional fields."""
        data = EquilibriumData(
            psi_axis=0.5,
            psi_edge=-0.2,
            ip=-1e6,
            b0=-5.0,
            q=3.0,
            dp_dpsi=-1e3,
            source="test",
        )
        assert data.q == 3.0
        assert data.dp_dpsi == -1e3
        assert data.source == "test"


class TestLoadFromEQDSK:
    """Test load_from_eqdsk function."""

    def test_file_not_found(self, tmp_path: Path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_from_eqdsk(tmp_path / "nonexistent.eqdsk")

    def test_invalid_file_raises_value_error(self, tmp_path: Path):
        """Should raise ValueError for invalid file content."""
        bad_file = tmp_path / "bad.eqdsk"
        bad_file.write_text("not valid eqdsk content\n")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_from_eqdsk(bad_file)

    def test_minimal_eqdsk(self, tmp_path: Path):
        """Should parse a minimal valid EQDSK file."""
        # Create minimal EQDSK-like file
        # Format: header, then lines with 5 floats each
        eqdsk_content = """\
 EFIT     0  65  65
  1.6000E+00  2.0000E+00  1.8500E+00  1.0000E+00  0.0000E+00
  1.8500E+00  0.0000E+00  1.2345E-01 -2.3456E-01  2.5000E+00
  1.5000E+06  1.2345E-01  0.0000E+00  1.8500E+00  0.0000E+00
  0.0000E+00  0.0000E+00  0.0000E+00  0.0000E+00  65
"""
        eqdsk_file = tmp_path / "g012345.00100"
        eqdsk_file.write_text(eqdsk_content)

        data = load_from_eqdsk(eqdsk_file)

        assert isinstance(data, EquilibriumData)
        assert data.source == f"eqdsk:{eqdsk_file.name}"
        # Check that physics values were extracted
        assert data.psi_axis == pytest.approx(1.2345e-1)
        assert data.psi_edge == pytest.approx(-2.3456e-1)
        assert data.ip == pytest.approx(1.5e6)
        assert data.b0 == pytest.approx(2.5)
