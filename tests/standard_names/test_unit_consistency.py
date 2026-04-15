"""Tests ensuring StandardName uses unit/HAS_UNIT consistently.

Validates that the legacy canonical_units/CANONICAL_UNITS naming is not
reintroduced. All unit references in standard-name Cypher and schema must
use the same convention as IMASNode and SignalNode: property ``unit`` with
relationship type ``HAS_UNIT`` to ``Unit`` nodes.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Modules whose source code should never contain CANONICAL_UNITS or
# the property name canonical_units in Cypher/Python.
SN_MODULES = [
    "imas_codex.standard_names.graph_ops",
    "imas_codex.standard_names.search",
    "imas_codex.standard_names.catalog_import",
    "imas_codex.standard_names.publish",
    "imas_codex.standard_names.sources.signals",
    "imas_codex.llm.sn_tools",
    "imas_codex.cli.sn",
]


class TestNoLegacyCanonicalUnits:
    """Ensure no source file uses the legacy canonical_units naming."""

    @pytest.mark.parametrize("module_path", SN_MODULES)
    def test_no_canonical_units_in_source(self, module_path: str) -> None:
        """Source code must not contain CANONICAL_UNITS relationship type."""
        mod = importlib.import_module(module_path)
        source = inspect.getsource(mod)
        assert "CANONICAL_UNITS" not in source, (
            f"{module_path} still references CANONICAL_UNITS relationship type"
        )
        for i, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "canonical_units" in stripped.lower():
                pytest.fail(
                    f"{module_path}:{i} uses legacy canonical_units: {stripped}"
                )


class TestSchemaUnitSlot:
    """Validate the standard_name.yaml schema defines unit correctly."""

    def test_schema_has_unit_not_canonical_units(self) -> None:
        """StandardName schema must use 'unit' slot, not 'canonical_units'."""
        import yaml

        schema_path = (
            Path(__file__).resolve().parents[2]
            / "imas_codex"
            / "schemas"
            / "standard_name.yaml"
        )
        with open(schema_path) as f:
            schema = yaml.safe_load(f)

        sn_attrs = schema["classes"]["StandardName"]["attributes"]
        assert "unit" in sn_attrs, "StandardName must have a 'unit' attribute"
        assert "canonical_units" not in sn_attrs, (
            "StandardName must not have 'canonical_units' — use 'unit' instead"
        )

    def test_unit_slot_has_range_and_relationship(self) -> None:
        """The unit slot must have range: Unit and relationship_type: HAS_UNIT."""
        import yaml

        schema_path = (
            Path(__file__).resolve().parents[2]
            / "imas_codex"
            / "schemas"
            / "standard_name.yaml"
        )
        with open(schema_path) as f:
            schema = yaml.safe_load(f)

        unit_slot = schema["classes"]["StandardName"]["attributes"]["unit"]
        assert unit_slot.get("range") == "Unit", (
            f"unit slot must have range: Unit, got {unit_slot.get('range')}"
        )
        rel_type = unit_slot.get("annotations", {}).get("relationship_type")
        assert rel_type == "HAS_UNIT", (
            f"unit slot must have relationship_type: HAS_UNIT, got {rel_type}"
        )


class TestCypherUsesHasUnit:
    """Verify Cypher in graph_ops uses HAS_UNIT for StandardName → Unit."""

    def test_write_creates_has_unit_relationship(self) -> None:
        """write_standard_names must MERGE HAS_UNIT, not CANONICAL_UNITS."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [{"id": "test_quantity", "unit": "eV", "source_types": ["dd"]}]
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(names)

        all_cypher = " ".join(str(call) for call in mock_gc.query.call_args_list)
        assert "HAS_UNIT" in all_cypher, "Should use HAS_UNIT relationship"
        assert "CANONICAL_UNITS" not in all_cypher, (
            "Must not use legacy CANONICAL_UNITS relationship"
        )

    def test_write_sets_unit_property(self) -> None:
        """write_standard_names must SET sn.unit, not sn.canonical_units."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [{"id": "test_quantity", "unit": "eV", "source_types": ["dd"]}]
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(names)

        # Find the MERGE query
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "MERGE (sn:StandardName" in cypher:
                assert "sn.unit" in cypher, "MERGE query must SET sn.unit property"
                assert "sn.canonical_units" not in cypher, (
                    "MERGE query must not SET sn.canonical_units"
                )
                break
        else:
            pytest.fail("No MERGE StandardName query found")

    def test_unit_conflict_detection_uses_unit(self) -> None:
        """Unit conflict detection must check sn.unit, not sn.canonical_units."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [{"id": "test_quantity", "unit": "eV", "source_types": ["dd"]}]
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(names)

        # Find the conflict detection query
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if (
                "sn.unit IS NOT NULL" in cypher
                or "sn.canonical_units IS NOT NULL" in cypher
            ):
                assert "sn.unit IS NOT NULL" in cypher
                assert "sn.canonical_units" not in cypher
                break

    def test_reset_deletes_has_unit_relationships(self) -> None:
        """reset_standard_names must delete HAS_UNIT, not CANONICAL_UNITS."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 1}])

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import reset_standard_names

            reset_standard_names(from_status="drafted")

        all_cypher = " ".join(call[0][0] for call in mock_gc.query.call_args_list)
        assert "HAS_UNIT" in all_cypher
        assert "CANONICAL_UNITS" not in all_cypher
