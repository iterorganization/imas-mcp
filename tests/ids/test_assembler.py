"""Tests for IDS assembly engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from imas_codex.ids.assembler import (
    IDSAssembler,
    _coil_index_from_path,
    _resolve_epoch_id,
    _set_nested,
    list_recipes,
)

# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestResolveEpochId:
    def test_short_epoch(self):
        assert _resolve_epoch_id("jet", "p68613") == "jet:device_xml:p68613"

    def test_already_qualified(self):
        assert (
            _resolve_epoch_id("jet", "jet:device_xml:p68613") == "jet:device_xml:p68613"
        )


class TestCoilIndexFromPath:
    def test_device_xml_path(self):
        assert _coil_index_from_path("jet:device_xml:p68613:pfcoils:5") == 5

    def test_jec2020_path(self):
        assert _coil_index_from_path("jet:jec2020:pf_coil:12") == 12


class TestSetNested:
    def test_simple_attr(self):
        obj = MagicMock()
        _set_nested(obj, "name", "test")
        obj.__setattr__("name", "test")

    def test_dotted_path(self):
        obj = MagicMock()
        _set_nested(obj, "geometry.rectangle.r", 1.5)
        obj.geometry.rectangle.__setattr__("r", 1.5)


# ---------------------------------------------------------------------------
# Recipe tests
# ---------------------------------------------------------------------------


class TestListRecipes:
    def test_list_all(self):
        recipes = list_recipes()
        assert len(recipes) >= 1
        assert any(r["ids_name"] == "pf_active" for r in recipes)

    def test_list_by_facility(self):
        recipes = list_recipes("jet")
        assert len(recipes) >= 1
        assert all(r["facility"] == "jet" for r in recipes)

    def test_list_unknown_facility(self):
        recipes = list_recipes("nonexistent")
        assert recipes == []


class TestRecipeValidation:
    def test_valid_recipe(self, tmp_path):
        recipe = {
            "ids_name": "pf_active",
            "facility_id": "jet",
            "dd_version": "4.1.1",
        }
        recipe_file = tmp_path / "test.yaml"
        recipe_file.write_text(yaml.dump(recipe))
        assembler = IDSAssembler("jet", "pf_active", recipe_path=recipe_file)
        assert assembler.recipe["ids_name"] == "pf_active"

    def test_missing_required_field(self, tmp_path):
        recipe = {"ids_name": "pf_active"}
        recipe_file = tmp_path / "test.yaml"
        recipe_file.write_text(yaml.dump(recipe))
        with pytest.raises(ValueError, match="missing required fields"):
            IDSAssembler("jet", "pf_active", recipe_path=recipe_file)

    def test_missing_recipe_file(self):
        with pytest.raises(FileNotFoundError):
            IDSAssembler("jet", "nonexistent_ids")


class TestJetPfActiveRecipe:
    """Validate the JET pf_active recipe file structure."""

    @pytest.fixture
    def recipe(self):
        recipe_path = (
            Path(__file__).parent.parent.parent
            / "imas_codex"
            / "ids"
            / "recipes"
            / "jet"
            / "pf_active.yaml"
        )
        return yaml.safe_load(recipe_path.read_text())

    def test_recipe_metadata(self, recipe):
        assert recipe["ids_name"] == "pf_active"
        assert recipe["facility_id"] == "jet"
        assert recipe["dd_version"] == "4.1.1"
        assert "provider" in recipe

    def test_static_fields(self, recipe):
        static = recipe.get("static", {})
        assert "ids_properties.homogeneous_time" in static
        assert static["ids_properties.homogeneous_time"] == 0

    def test_coil_source_query(self, recipe):
        arrays = recipe.get("arrays", {})
        assert "coil" in arrays
        coil = arrays["coil"]
        assert "source" in coil
        query = coil["source"]["query"]
        assert "device_xml" in query
        assert "PF" in query

    def test_coil_elements(self, recipe):
        elements = recipe["arrays"]["coil"]["elements"]
        assert elements["geometry_type"] == 2  # rectangle
        fields = elements["fields"]
        assert "geometry.rectangle.r" in fields
        assert "geometry.rectangle.z" in fields
        assert "turns_with_sign" in fields

    def test_circuit_definition(self, recipe):
        arrays = recipe.get("arrays", {})
        assert "circuit" in arrays


# ---------------------------------------------------------------------------
# Integration tests (require Neo4j)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAssemblerIntegration:
    """Integration tests that require a running Neo4j graph with JET data."""

    def test_assemble_pf_active(self):
        import imas

        assembler = IDSAssembler("jet", "pf_active")
        ids = assembler.assemble("p68613")

        assert len(ids.coil) == 22
        assert len(ids.circuit) == 16

        # Verify first coil has elements
        coil0 = ids.coil[0]
        assert len(coil0.element) > 0
        assert coil0.element[0].geometry.geometry_type == 2

        # Verify geometry values are populated
        r = float(coil0.element[0].geometry.rectangle.r)
        z = float(coil0.element[0].geometry.rectangle.z)
        assert r > 0
        assert z != 0 or r != 0  # at least one non-zero

    def test_export_hdf5(self, tmp_path):
        import imas

        assembler = IDSAssembler("jet", "pf_active")
        out = assembler.export(tmp_path / "test_pf", "p68613")

        # Read back
        entry = imas.DBEntry(f"imas:hdf5?path={out}", "r")
        pf = entry.get("pf_active")
        entry.close()

        assert len(pf.coil) == 22
        assert pf.ids_properties.homogeneous_time == 0
        assert "imas-codex" in str(pf.ids_properties.provider)

    def test_summary(self):
        assembler = IDSAssembler("jet", "pf_active")
        summary = assembler.summary("p68613")

        assert summary["ids_name"] == "pf_active"
        assert summary["arrays"]["coil"]["count"] == 22
        assert summary["arrays"]["coil"]["total_elements"] > 1000
        assert summary["arrays"]["circuit"]["count"] == 16

    def test_list_epochs(self):
        assembler = IDSAssembler("jet", "pf_active")
        epochs = assembler.list_epochs()

        assert len(epochs) > 0
        epoch_ids = [e["id"] for e in epochs]
        assert any("p68613" in eid for eid in epoch_ids)
