"""Tests for StandardNameCandidate description field (Track A1)."""

from __future__ import annotations

from imas_codex.standard_names.models import StandardNameCandidate


class TestStandardNameCandidateDescription:
    """Verify the description field on StandardNameCandidate."""

    def test_description_default_empty(self) -> None:
        """Description defaults to empty string when not provided."""
        c = StandardNameCandidate(
            source_id="eq/time_slice/profiles_1d/psi",
            standard_name="poloidal_magnetic_flux",
            reason="test",
        )
        assert c.description == ""

    def test_description_persists(self) -> None:
        """Description is stored when provided."""
        c = StandardNameCandidate(
            source_id="eq/time_slice/profiles_1d/psi",
            standard_name="poloidal_magnetic_flux",
            description="Poloidal magnetic flux on the 1D grid",
            reason="test",
        )
        assert c.description == "Poloidal magnetic flux on the 1D grid"

    def test_description_in_dict(self) -> None:
        """Description appears in model_dump output."""
        c = StandardNameCandidate(
            source_id="eq/time_slice/profiles_1d/psi",
            standard_name="poloidal_magnetic_flux",
            description="Poloidal magnetic flux on the 1D grid",
            reason="test",
        )
        d = c.model_dump()
        assert d["description"] == "Poloidal magnetic flux on the 1D grid"

    def test_description_from_dict(self) -> None:
        """Description is parsed from dict input."""
        data = {
            "source_id": "eq/time_slice/profiles_1d/psi",
            "standard_name": "poloidal_magnetic_flux",
            "description": "Poloidal magnetic flux",
            "reason": "test",
        }
        c = StandardNameCandidate.model_validate(data)
        assert c.description == "Poloidal magnetic flux"

    def test_existing_fields_unchanged(self) -> None:
        """Adding description does not break existing field access."""
        c = StandardNameCandidate(
            source_id="path",
            standard_name="electron_temperature",
            description="Temperature of electrons",
            kind="scalar",
            dd_paths=["core_profiles/profiles_1d/electrons/temperature"],
            grammar_fields={"subject": "electron", "physical_base": "temperature"},
            reason="test",
        )
        assert c.source_id == "path"
        assert c.standard_name == "electron_temperature"
        assert c.kind == "scalar"
        assert c.dd_paths == ["core_profiles/profiles_1d/electrons/temperature"]
        assert c.grammar_fields == {
            "subject": "electron",
            "physical_base": "temperature",
        }
        assert c.reason == "test"
